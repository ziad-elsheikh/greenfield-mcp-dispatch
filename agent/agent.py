from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

from server.rag.verifier import self_rag_verify
from memory.memory import ShortTermMemory
from context_eval.strategies import recursive_summarization

from .schema import (
    ACTION_INPUT_SCHEMAS,
    AgentStep,
    build_agent_step_model,
    TERMINAL_ACTIONS,
    MAX_STEPS,
    build_system_prompt
)

load_dotenv()


def build_structured_model(action_names: List[str]):
    step_model = build_agent_step_model(action_names)
    return init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq",
        max_tokens=1024,
        temperature=0.1,
        max_retries=3,
    ).with_structured_output(step_model)


async def discover_tools(client) -> Dict[str, Any]:
    tools_list = await client.list_tools()
    return {tool.name: tool for tool in tools_list}


def validate_step(step: AgentStep, tools: Dict[str, Any]) -> bool:
    return step.action in TERMINAL_ACTIONS or step.action in tools


def handle_final_action(step: AgentStep) -> bool:
    """Check if the step represents a terminal state."""
    return step.action in TERMINAL_ACTIONS


async def tool_call(client, step: AgentStep) -> Any:
    payload = step.action_input or {}

    # Validate schema using defined Pydantic model
    schema_cls = ACTION_INPUT_SCHEMAS.get(step.action)
    if schema_cls and isinstance(payload, dict):
        validated_input = schema_cls(**payload)
        payload = validated_input.model_dump()

    # Wrap payload for FastMCP tool signature expectations
    mcp_payload = {"input_data": payload} if payload else {}

    # Invoke tool via MCP client call interface
    result = await client.call_tool(step.action, mcp_payload)
    return result


async def agent_step(client, memory: ShortTermMemory, user_input: str) -> Optional[AgentStep]:
    memory.add_user(user_input)

    tools = await discover_tools(client)
    current_tool_names = sorted(tools.keys())

    # Extract semantic facts and recent episodic events
    semantic_context = ""
    if hasattr(memory, "long_term"):
        active_facts = list(memory.long_term.get_active_facts().values())
        v_mem = self_rag_verify(user_input, active_facts, user_input)
        if v_mem.is_relevant:
            semantic_context += "\nVerified Known Facts:\n" + "\n".join([f"- {f}" for f in active_facts]) + "\n"
        if memory.long_term.semantic_facts:
            facts_list = [f"- {k}: {v}" for k, v in memory.long_term.semantic_facts.items()]
            semantic_context += "\nKnown Facts:\n" + "\n".join(facts_list) + "\n"

        if memory.long_term.episodic_events:
            # Take the 5 most recent episodic events
            events_list = [
                f"- {e.get('summary')}: Context={e.get('context')}, Outcome={e.get('outcome')}"
                for e in memory.long_term.episodic_events[-5:]
                if e.get("summary")
            ]
            if events_list:
                semantic_context += "\nPast Logged Events:\n" + "\n".join(events_list) + "\n"

    system_prompt = (
        f"Current plan: {memory.scratchpad.get('plan')}\n"
        f"Sub-goal: {memory.scratchpad.get('current_subgoal')}\n"
        f"{semantic_context}"
        f"{build_system_prompt(current_tool_names)}"
    )
    memory.set_system_prompt(system_prompt)

    model = build_structured_model(current_tool_names)

    for step_num in range(MAX_STEPS):
        print(f"\n--- Step {step_num + 1} ---")

        try:
            raw_context = memory.get_context() 
            pruned_context = recursive_summarization(raw_context, model, keep_recent=6)
            step: AgentStep = await model.ainvoke(pruned_context)
        except Exception as e:
            print(f"[Agent Step Error]: {e}")
            memory.add_observation(f"Failed to generate structured step: {str(e)}")
            return None

        print(f"Thought: {step.thought}")
        print(f"Action: {step.action}")
        memory.add_ai(f"Thought: {step.thought}\nAction: {step.action}\nInput: {step.action_input}")

        # Update planning scratchpad if updated in step
        if getattr(step, "plan_updated", False):
            memory.scratchpad["plan"] = getattr(step, "new_plan", None)
            memory.scratchpad["current_subgoal"] = getattr(step, "next_subgoal", None)

        if handle_final_action(step):
            answer_text = str(step.action_input.get("answer") if isinstance(step.action_input, dict) else step.action_input)
            recent_context = [m.content for m in memory.get_context() if isinstance(m, HumanMessage)][-3:]
            v_result = self_rag_verify(user_input, recent_context, answer_text)
            
            if not v_result.is_supported:
                print(f"[Self-RAG Warning]: Answer lacks sufficient grounding ({v_result.reasoning})")
            return step

        if not validate_step(step, tools):
            memory.add_observation(
                f"Error: '{step.action}' is not a valid tool. Valid tools: {current_tool_names}"
            )
            continue

        try:
            result = await tool_call(client, step)
            print(f"Observation from {step.action}: {result}")
            memory.add_observation(f"Result from {step.action}: {result}")
        except ValidationError as e:
            err_msg = f"Invalid schema arguments for {step.action}: {e.errors()}"
            print(err_msg)
            memory.add_observation(err_msg)
            continue

    print("Reached maximum execution steps without a final answer.")
    return None