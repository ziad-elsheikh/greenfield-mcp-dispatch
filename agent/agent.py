from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model

from server.rag.verifier import self_rag_verify
from memory.memory import ShortTermMemory
from context_eval.strategies import recursive_summarization
from algorithms.decomposition import decompose_goal, execute_plan, final_output
from algorithms.dynamic_decomposition import dynamic_decomposition
from algorithms.models import Plan

from .schema import (
    ACTION_INPUT_SCHEMAS,
    AgentStep,
    build_agent_step_model,
    TERMINAL_ACTIONS,
    MAX_STEPS,
    build_system_prompt
)

load_dotenv()


def get_base_llm() -> BaseChatModel:
    """Returns the default BaseChatModel for decomposition and planning tasks."""
    return init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq",
        max_tokens=1024,
        temperature=0.1,
        max_retries=3,
    )


def initialize_plan(
    goal: str,
    memory: ShortTermMemory,
    llm: Optional[BaseChatModel] = None,
    dynamic: bool = False,
) -> Optional[Plan]:
    """
    Decomposes a user goal using static DAG decomposition (or dynamic decomposition)
    and initializes the planning scratchpad with the current plan and active subgoal.
    """
    base_llm = llm or get_base_llm()
    try:
        if dynamic:
            history = dynamic_decomposition(goal=goal, llm=base_llm, max_steps=4)
            if history:
                tasks_summary = "\n".join(f"- {task}: {result}" for task, result in history)
                memory.scratchpad["plan"] = tasks_summary
                memory.scratchpad["current_subgoal"] = history[-1][0] if history else None
                memory.scratchpad["dynamic_history"] = history
            return None
        else:
            plan = decompose_goal(goal=goal, llm=base_llm)
            ordered_tasks = plan.topological_order()
            tasks_repr = "\n".join(
                f"[{t.id}] {t.instruction} (depends on: {', '.join(t.depends_on) or 'none'})"
                for t in plan.tasks
            )
            memory.scratchpad["plan"] = tasks_repr
            memory.scratchpad["plan_dag"] = plan
            first_task = plan.task(ordered_tasks[0]) if ordered_tasks else None
            memory.scratchpad["current_subgoal"] = (
                f"[{first_task.id}] {first_task.instruction}" if first_task else None
            )
            return plan
    except Exception as e:
        print(f"[Plan Initialization Warning]: Could not decompose goal: {e}")
        return None


def run_static_plan(goal: str, llm: Optional[BaseChatModel] = None, max_workers: int = 4) -> str:
    """
    Decomposes a goal into a DAG plan, executes all task nodes in topological batches,
    and returns the final synthesized output.
    """
    base_llm = llm or get_base_llm()
    plan = decompose_goal(goal=goal, llm=base_llm)
    outputs = execute_plan(plan=plan, llm=base_llm, max_workers=max_workers)
    return final_output(plan=plan, outputs=outputs)


def run_dynamic_plan(goal: str, llm: Optional[BaseChatModel] = None, max_steps: int = 4) -> list[tuple[str, str]]:
    """
    Runs adaptive dynamic decomposition step-by-step until the goal is met.
    """
    base_llm = llm or get_base_llm()
    return dynamic_decomposition(goal=goal, llm=base_llm, max_steps=max_steps)


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


async def agent_step(
    client,
    memory: ShortTermMemory,
    user_input: str,
    llm: Optional[BaseChatModel] = None,
    use_dynamic_plan: bool = False,
) -> Optional[AgentStep]:
    memory.add_user(user_input)

    # Initialize plan decomposition in scratchpad if not already populated
    if not memory.scratchpad.get("plan"):
        initialize_plan(user_input, memory, llm=llm, dynamic=use_dynamic_plan)

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
            if getattr(step, "new_plan", None):
                memory.scratchpad["plan"] = step.new_plan
            if getattr(step, "next_subgoal", None):
                memory.scratchpad["current_subgoal"] = step.next_subgoal

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