from typing import Literal, Optional, List
from pydantic import BaseModel, ConfigDict, Field, create_model

MAX_STEPS = 6

TERMINAL_ACTIONS = {
    "escalate",
    "end_conversation",
    "final_answer",
}

class AgentStep(BaseModel):
    """Runtime step produced by the agent."""

    thought: str
    action: str
    action_input: Optional[dict] = None
    plan_updated: bool
    new_plan: Optional[str] = None
    next_subgoal: Optional[str] = None
    is_final: bool


def build_agent_step_model(action_names: List[str]):
    """
    Build an AgentStep model whose action field is restricted
    to the MCP tools currently exposed by the server plus the
    terminal actions.
    """
    allowed = tuple(sorted(set(action_names) | TERMINAL_ACTIONS))

    return create_model(
        "AgentStep",
        thought=(str, ...),
        action=(Literal[allowed], ...),
        action_input=(dict, Field(default_factory=dict)),
        is_final=(bool, ...),
    )


# ==========================================================
# Base Input
# ==========================================================

class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyInput(StrictInput):
    """Tool takes no parameters."""
    pass


# ==========================================================
# Terminal Actions
# ==========================================================

class FinalAnswerInput(StrictInput):
    answer: str = Field(
        description="Final response shown to the user."
    )


class EscalationInput(StrictInput):
    reason: str = Field(
        description="Why the request should be escalated."
    )


# ==========================================================
# Agricultural Tool Schemas
# ==========================================================

class DispatchEquipmentInput(StrictInput):
    equipment_id: int
    field_id: int
    customer_id: int
    job_type: Literal["till", "harvest", "spray"]
    chemical_id: int | None = None


class BatchDispatchInput(StrictInput):
    equipment_ids: list[int]
    field_id: int


class PaymentInput(StrictInput):
    customer_id: int


class IncidentInput(StrictInput):
    raw_note: str


class ReportInput(StrictInput):
    month: str


class KnowledgeSearchInput(StrictInput):
    query: str = Field(
        description="Search query to lookup manuals, chemical policies, or operating procedures."
    )


def build_system_prompt(tool_names: List[str]) -> str:
    tool_list = "\n".join(sorted(tool_names))
    return f"""You are a constrained support agent for Greenfield Agriculture.

Available Tools:
{tool_list}

Strict Execution Instructions:
1. MANDATORY RAG SEARCH FOR KNOWLEDGE / POLICIES:
   - If the user asks about operating speeds, chemical rules, buffer zones, SOP codes, or equipment manuals, you MUST call 'search_agricultural_knowledge' FIRST to retrieve grounded document context before giving a final answer.
   - Never answer compliance or manual questions from memory without searching.

2. FLEET & EQUIPMENT STATUS:
   - When asked to check equipment or overall fleet status, do NOT talk about the tool in 'final_answer'. Set action to 'equipment_status_snapshot' immediately to fetch live data.
   - Do NOT invoke 'dispatch_equipment' or 'batch_dispatch' unless you are executing an actual job with explicit IDs.

3. MEMORY & USER FACTS:
   - When acknowledging user facts, preferences, or allergies (e.g. "Customer 1 is allergic to SPR-3001"), acknowledge them directly using 'final_answer'. Do NOT call 'log_incident_note' or other tools. Memory eviction and consolidation handle context automatically.
   - Only use 'log_incident_note' for actual physical farm emergencies, chemical spills, or equipment damage. When calling it, pass a single string field named 'raw_note'.

4. FINAL RESPONSES:
   - When you have enough information or need to respond directly to the user, set action to 'final_answer' and put your response message inside 'action_input.answer'.
   - Output clean, valid JSON strings for all tool arguments and responses.

Think step by step and return only the structured response."""


# ==========================================================
# Action → Input Schema Mapping
# ==========================================================

ACTION_INPUT_SCHEMAS = {
    # ===== MCP Tools =====
    "dispatch_equipment": DispatchEquipmentInput,
    "batch_dispatch": BatchDispatchInput,
    "process_payment": PaymentInput,
    "log_incident_note": IncidentInput,
    "generate_fleet_report": ReportInput,
    "equipment_status_snapshot": EmptyInput,
    "pesticide_compliance_policy": EmptyInput,
    "draft_delay_explanation": EmptyInput,

    # ===== RAG Tools =====
    "search_agricultural_knowledge": KnowledgeSearchInput,

    # ===== Terminal =====
    "escalate": EscalationInput,
    "end_conversation": FinalAnswerInput,
    "final_answer": FinalAnswerInput,
}