"""Plan-and-Solve Algorithm Implementation for Sub-tasks."""

from typing import Any, Dict, List, Optional
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from algorithms.models import Task


class PlanAndSolveState(BaseModel):
    """Schema for Plan-and-Solve explicit phases."""
    understanding: str = Field(description="Analysis of the problem and required outputs")
    plan_steps: List[str] = Field(description="Sequential execution steps to solve the sub-task")


class ExecutionResult(BaseModel):
    """Schema for final execution output."""
    answer: str = Field(description="Final result after carrying out the planned steps")


async def run_plan_and_solve(
    task: Task,
    context: str,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes a sub-task using explicit two-phase Plan-and-Solve technique.
    Phase 1: Understand the sub-task and derive a step-by-step plan.
    Phase 2: Carry out the plan sequentially to produce the final outcome.
    """
    if llm is None:
        llm = init_chat_model(
            model="llama-3.3-70b-versatile",
            model_provider="groq",
            max_tokens=1024,
        )

    # Phase 1: Understand & Plan
    planner = llm.with_structured_output(PlanAndSolveState)
    plan_prompt = (
        f"Goal/Sub-task: {task.instruction}\n"
        f"Context from prerequisites: {context}\n"
        "Let's first understand the problem thoroughly and devise a detailed, step-by-step plan."
    )
    plan_state: PlanAndSolveState = await planner.ainvoke(plan_prompt)

    # Phase 2: Solve & Execute
    solver = llm.with_structured_output(ExecutionResult)
    solve_prompt = (
        f"Sub-task: {task.instruction}\n"
        f"Understanding: {plan_state.understanding}\n"
        f"Plan Steps: {plan_state.plan_steps}\n"
        "Now, carry out the plan step-by-step to compute the final answer."
    )
    result: ExecutionResult = await solver.ainvoke(solve_prompt)

    return {
        "task_id": task.id,
        "understanding": plan_state.understanding,
        "plan": plan_state.plan_steps,
        "result": result.answer,
    }
