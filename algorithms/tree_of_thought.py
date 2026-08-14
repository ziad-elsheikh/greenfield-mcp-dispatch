"""Tree of Thoughts (ToT) Search Algorithm Implementation."""

from typing import Any, Dict, List, Optional
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from algorithms.models import Task, Thought


class ThoughtCandidates(BaseModel):
    """Schema for proposing multiple candidate next thoughts."""
    candidates: List[Thought] = Field(
        description="List of proposed candidate thoughts with scores and rationales"
    )


class FinalTreeOutput(BaseModel):
    """Schema for selecting the best path after search."""
    best_solution: str = Field(description="The finalized reasoning outcome from the winning branch")


async def run_tree_of_thoughts(
    task: Task,
    context: str,
    beam_width: int = 3,
    max_depth: int = 2,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes a sub-task using Tree of Thoughts (ToT) with BFS search.
    - Generates multiple candidate thoughts per level.
    - Evaluates and scores each thought.
    - Keeps the top 'beam_width' candidates and explores up to 'max_depth'.
    """
    if llm is None:
        llm = init_chat_model(
            model="llama-3.3-70b-versatile",
            model_provider="groq",
            max_tokens=1024,
        )

    proposer = llm.with_structured_output(ThoughtCandidates)
    
    # Frontier for BFS search: stores current paths (strings)
    frontier: List[str] = [f"Initial Context: {context}"]
    all_evaluated_thoughts: List[Dict[str, Any]] = []

    for depth in range(max_depth):
        candidates_pool: List[tuple[float, str]] = []

        for current_path in frontier:
            prompt = (
                f"Sub-task Goal: {task.instruction}\n"
                f"Current Reasoning Path:\n{current_path}\n\n"
                f"Propose {beam_width} distinct next candidate reasoning thoughts or steps. "
                f"For each thought, evaluate its viability with a score between 0.0 (impossible) and 1.0 (sure)."
            )
            
            try:
                result: ThoughtCandidates = await proposer.ainvoke(prompt)
                for cand in result.candidates:
                    new_path = f"{current_path}\n-> Thought: {cand.state} (Score: {cand.score})"
                    candidates_pool.append((cand.score, new_path))
                    all_evaluated_thoughts.append({
                        "depth": depth + 1,
                        "state": cand.state,
                        "score": cand.score,
                        "rationale": cand.rationale,
                    })
            except Exception:
                continue

        if not candidates_pool:
            break

        # Sort by score descending and prune to beam_width
        candidates_pool.sort(key=lambda x: x[0], reverse=True)
        frontier = [path for _, path in candidates_pool[:beam_width]]

    # Final decision phase from the best surviving branch
    best_path = frontier[0] if frontier else f"Context: {context}"
    evaluator = llm.with_structured_output(FinalTreeOutput)
    final_prompt = (
        f"Sub-task Goal: {task.instruction}\n"
        f"Selected Best Search Path:\n{best_path}\n\n"
        "Synthesize the final, verified answer for this sub-task based on the selected reasoning path."
    )
    final_result: FinalTreeOutput = await evaluator.ainvoke(final_prompt)

    return {
        "task_id": task.id,
        "best_path": best_path,
        "evaluated_nodes_count": len(all_evaluated_thoughts),
        "result": final_result.best_solution,
    }
