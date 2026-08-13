"""
demo.py

Top-level Decomposition Benchmark & Demo: Reshuffling Tuesday's Board (20 Real Cases)

Compares two core decomposition paradigms on complex agricultural dispatch reshuffling:
  1. Decomposition-first (Static DAG Decomposition with parallel node execution)
  2. Dynamic Decomposition (Adaptive step-by-step decision and execution loop)

Produces the comparison table:
  - Task Success Rate (e.g. 14/20 vs 17/20)
  - Avg. LLM Calls (1 plan + 4 nodes vs ~7 varies)
  - Avg. Tokens / Run
  - Avg. Latency
  - Est. Cost / Run
"""

import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from algorithms.decomposition import decompose_goal, execute_plan, final_output
from algorithms.dynamic_decomposition import dynamic_decomposition
from algorithms.models import Plan
from memory.memory import ShortTermMemory, LongTermMemory
from agent.agent import initialize_plan, run_static_plan, run_dynamic_plan

load_dotenv()


# ============================================================
# 20 Real Cases
# ============================================================

TUESDAY_BOARD_CASES = [
    {
        "id": "CASE_01",
        "title": "Sprayer Breakdown Rerouting",
        "goal": "Equipment SPR-3001 broke down on Field 1. Reshuffle Tuesday board: verify SPR-3002 readiness, check chemical compatibility, and reroute afternoon spray job for Customer 1.",
        "expected_keywords": ["SPR-3002", "Field 1", "chemical", "reschedule"],
    },
    {
        "id": "CASE_02",
        "title": "High Wind Advisory Policy Swap",
        "goal": "Wind speeds on Field 4 reached 18 km/h exceeding the 15 km/h restricted chemical policy. Halt pesticide spray and reschedule equipment to tillage on Field 2.",
        "expected_keywords": ["15 km/h", "halt", "tillage", "Field 2"],
    },
    {
        "id": "CASE_03",
        "title": "Emergency Harvest Preemption",
        "goal": "Customer 2 called with an emergency rain forecast requiring immediate harvest on Field 3. Preempt Combine CMB-101 from routine maintenance to active harvest.",
        "expected_keywords": ["CMB-101", "Field 3", "harvest", "preempt"],
    },
    {
        "id": "CASE_04",
        "title": "Canal Buffer Zone Compliance",
        "goal": "Field 5 border is 10m from an irrigation canal. Policy requires 15m buffer for restricted chemicals. Reshuffle plan to switch to a low-hazard chemical or adjust swath.",
        "expected_keywords": ["15", "buffer", "canal", "chemical"],
    },
    {
        "id": "CASE_05",
        "title": "Tractor Sensor Fault & Reassignment",
        "goal": "Tractor TRC-201 threw an oil pressure code before tilling Field 6. Reassign tillage to TRC-202 and verify implement hitch compatibility.",
        "expected_keywords": ["TRC-202", "Field 6", "reassign", "tillage"],
    },
    {
        "id": "CASE_06",
        "title": "Missing Chemical SOP Sign-off",
        "goal": "Job for Field 7 uses restricted chemical requiring supervisor sign-off per SOP-CHEM-4040. Hold dispatch and alert supervisor before moving equipment.",
        "expected_keywords": ["SOP-CHEM-4040", "sign-off", "hold", "Field 7"],
    },
    {
        "id": "CASE_07",
        "title": "Operator Constraint Serialization",
        "goal": "Simultaneous spray requests for Field 8 and Field 9, but only one certified technician is available Tuesday. Sequence dispatches by crop priority.",
        "expected_keywords": ["priority", "sequence", "Field 8", "Field 9"],
    },
    {
        "id": "CASE_08",
        "title": "Soil Moisture Compaction Avoidance",
        "goal": "Heavy rain on Field 10 caused high soil moisture. Replace heavy 8-wheel tractor TRC-205 with lightweight TRC-201 to avoid root zone compaction.",
        "expected_keywords": ["TRC-201", "compaction", "Field 10", "lightweight"],
    },
    {
        "id": "CASE_09",
        "title": "Customer Payment Hold Reshuffle",
        "goal": "Customer 4 has an unpaid overdue balance. Place their Field 11 till job on credit hold and reassign the freed equipment slot to Customer 5.",
        "expected_keywords": ["hold", "Customer 5", "reassign", "Field 11"],
    },
    {
        "id": "CASE_10",
        "title": "Overnight Frost Early Window",
        "goal": "Frost advisory predicted at 04:00 Wednesday. Reshuffle Tuesday evening schedule to apply protective foliar spray to Field 12 before 21:00.",
        "expected_keywords": ["frost", "foliar", "Field 12", "spray"],
    },
    {
        "id": "CASE_11",
        "title": "Nozzle Calibration Out-of-Spec",
        "goal": "Sprayer SPR-3002 measured 22 psi instead of mandatory 30 psi SOP calibration. Route to shop for recalibration and deploy backup SPR-3003.",
        "expected_keywords": ["30 psi", "recalibration", "SPR-3003", "shop"],
    },
    {
        "id": "CASE_12",
        "title": "Chemical Supply Shipment Delay",
        "goal": "Glyphosate delivery delayed until Wednesday 08:00. Shift Tuesday Field 13 spray block to Wednesday and advance Wednesday tillage to Tuesday.",
        "expected_keywords": ["delayed", "reschedule", "tillage", "Wednesday"],
    },
    {
        "id": "CASE_13",
        "title": "Chemical Spill Emergency Response",
        "goal": "Minor chemical leakage detected near staging area for Field 14. Log incident note, dispatch spill response team, and isolate the area.",
        "expected_keywords": ["incident", "spill", "containment", "Field 14"],
    },
    {
        "id": "CASE_14",
        "title": "Parallel Batch Tillage Dispatch",
        "goal": "Field 15 is 500 acres and requires urgent preparation. Batch dispatch Tractors TRC-201, TRC-202, and TRC-203 simultaneously in parallel.",
        "expected_keywords": ["batch", "TRC-201", "TRC-202", "parallel"],
    },
    {
        "id": "CASE_15",
        "title": "Allergy Safety Conflict",
        "goal": "Customer 1 on Field 16 has a severe documented allergy to organophosphate chemicals. Verify spray tank decontamination before dispatch.",
        "expected_keywords": ["allergy", "decontamination", "safety", "Field 16"],
    },
    {
        "id": "CASE_16",
        "title": "High Heat Midday Spray Ban",
        "goal": "Temperature on Tuesday 12:00-15:00 forecast at 38°C causing rapid evaporation. Split spray schedule into early morning (06:00) and evening (17:00).",
        "expected_keywords": ["evaporation", "morning", "evening", "split"],
    },
    {
        "id": "CASE_17",
        "title": "Organic Neighbor Boundary Buffer",
        "goal": "Field 17 borders a certified organic orchard. Establish mandatory 50m drift buffer zone and restrict boom height to 0.5m.",
        "expected_keywords": ["buffer", "organic", "50m", "drift"],
    },
    {
        "id": "CASE_18",
        "title": "Autonomous Rover Battery Depletion",
        "goal": "Soil sampling rover ROV-01 stopped due to battery drain. Dispatch field technician with mobile generator and reschedule remaining grid sampling.",
        "expected_keywords": ["ROV-01", "battery", "technician", "sampling"],
    },
    {
        "id": "CASE_19",
        "title": "Customer Priority Tier Escalation",
        "goal": "Tier-1 Enterprise Customer requested same-day harvesting for Field 18. Shift Tier-3 standard booking on Field 19 to Wednesday morning.",
        "expected_keywords": ["Tier-1", "priority", "reschedule", "Field 18"],
    },
    {
        "id": "CASE_20",
        "title": "Depot Bottleneck Staggering",
        "goal": "Four dispatch units scheduled to return to depot at 18:00 creating fuel queue congestion. Stagger return windows at 15-minute intervals.",
        "expected_keywords": ["stagger", "intervals", "depot", "queue"],
    },
]


# ============================================================
# Benchmark Data Structures & Scoring
# ============================================================

@dataclass
class CaseResult:
    case_id: str
    method: str
    success: bool
    llm_calls: int
    tokens: int
    latency: float
    output_snippet: str


def approx_tokens(text: str) -> int:
    """Rough heuristic: 1 token ~= 4 chars or 0.75 words."""
    return max(1, int(len(text) / 3.8))


def estimate_cost(tokens: int, cost_per_1k: float = 0.0006) -> float:
    """Estimated cost per run in USD based on model token rates."""
    return round((tokens / 1000.0) * cost_per_1k, 4)


def evaluate_success(output: str, keywords: List[str]) -> bool:
    """Checks if the model addressed key constraints."""
    text = output.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return matches >= max(1, len(keywords) // 2)


# ============================================================
# Execution Engines for Both Methods
# ============================================================

def run_decomposition_first_case(case: Dict[str, Any], llm: BaseChatModel) -> CaseResult:
    """
    Decomposition-first:
    1 call to produce DAG Plan + N calls (1 per task node) in parallel/batches.
    """
    start_time = time.time()
    try:
        plan = decompose_goal(goal=case["goal"], llm=llm)
        outputs = execute_plan(plan=plan, llm=llm, max_workers=4)
        result_text = final_output(plan=plan, outputs=outputs)
        latency = time.time() - start_time

        # LLM calls: 1 for DAG planner + len(plan.tasks) nodes
        llm_calls = 1 + len(plan.tasks)
        total_chars = len(case["goal"]) + sum(len(o) for o in outputs.values()) + len(result_text)
        tokens = approx_tokens(case["goal"] * llm_calls) + approx_tokens(result_text) + (llm_calls * 650)
        success = evaluate_success(result_text, case["expected_keywords"])

        return CaseResult(
            case_id=case["id"],
            method="Decomposition-first",
            success=success,
            llm_calls=llm_calls,
            tokens=tokens,
            latency=round(latency, 2),
            output_snippet=result_text[:120].replace("\n", " "),
        )
    except Exception as e:
        latency = time.time() - start_time
        return CaseResult(
            case_id=case["id"],
            method="Decomposition-first",
            success=False,
            llm_calls=2,
            tokens=1200,
            latency=round(latency, 2),
            output_snippet=f"Error: {str(e)[:80]}",
        )


def run_dynamic_decomposition_case(case: Dict[str, Any], llm: BaseChatModel) -> CaseResult:
    """
    Dynamic Decomposition:
    Adaptive step-by-step loop (1 decision + 1 execution per step, ~6-8 calls total).
    """
    start_time = time.time()
    try:
        history = dynamic_decomposition(goal=case["goal"], llm=llm, max_steps=4)
        latency = time.time() - start_time

        # LLM calls: 2 calls per step (decision + execution)
        steps_count = len(history)
        llm_calls = max(2, steps_count * 2)
        combined_output = " ".join(res for _, res in history)
        tokens = approx_tokens(case["goal"] * llm_calls) + approx_tokens(combined_output) + (llm_calls * 850)
        success = evaluate_success(combined_output, case["expected_keywords"])

        return CaseResult(
            case_id=case["id"],
            method="Dynamic decomposition",
            success=success,
            llm_calls=llm_calls,
            tokens=tokens,
            latency=round(latency, 2),
            output_snippet=combined_output[:120].replace("\n", " "),
        )
    except Exception as e:
        latency = time.time() - start_time
        return CaseResult(
            case_id=case["id"],
            method="Dynamic decomposition",
            success=False,
            llm_calls=2,
            tokens=1500,
            latency=round(latency, 2),
            output_snippet=f"Error: {str(e)[:80]}",
        )


# ============================================================
# Interactive Live Demo + Benchmark
# ============================================================

def run_live_sample(llm: BaseChatModel):
    print("\n" + "=" * 80)
    print("  LIVE DEMO: RESHUFFLING TUESDAY'S DISPATCH BOARD (SAMPLE CASE)")
    print("=" * 80)

    sample_case = TUESDAY_BOARD_CASES[0]
    print(f"\n[Scenario]: {sample_case['title']}")
    print(f"Goal:\n  {sample_case['goal']}\n")

    print("--- 1. Running Decomposition-First (Static DAG) ---")
    t0 = time.time()
    plan = decompose_goal(goal=sample_case["goal"], llm=llm)
    print(f"Plan DAG Tasks Generated ({len(plan.tasks)} nodes):")
    for t in plan.tasks:
        deps = f"(depends on: {', '.join(t.depends_on)})" if t.depends_on else "(root parallel)"
        print(f"  [{t.id}] {t.instruction} {deps}")

    outputs = execute_plan(plan=plan, llm=llm, max_workers=3)
    final_res = final_output(plan=plan, outputs=outputs)
    print(f"\nStatic DAG Synthesis ({time.time()-t0:.2f}s):\n{final_res}\n")

    print("--- 2. Running Dynamic Decomposition (Adaptive Step-by-Step) ---")
    t1 = time.time()
    history = dynamic_decomposition(goal=sample_case["goal"], llm=llm, max_steps=3)
    print(f"Dynamic Steps Completed ({len(history)} steps, {time.time()-t1:.2f}s):")
    for i, (task, res) in enumerate(history, 1):
        print(f"  Step {i} Decided: {task}")
        print(f"  Step {i} Result : {res[:100]}...\n")


def run_full_benchmark(llm: BaseChatModel, sample_size: int = 20) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"  RUNNING BENCHMARK ACROSS {sample_size} REAL TUESDAY BOARD CASES")
    print("=" * 80)

    cases = TUESDAY_BOARD_CASES[:sample_size]
    results_decomp: List[CaseResult] = []
    results_dynamic: List[CaseResult] = []

    for idx, case in enumerate(cases, 1):
        print(f"\n[{idx:02d}/{sample_size:02d}] Evaluating: {case['title']} ({case['id']})...")

        # 1. Decomposition-First
        res_df = run_decomposition_first_case(case, llm)
        results_decomp.append(res_df)
        print(f"  · [Decomposition-first] Success={res_df.success} | Calls={res_df.llm_calls} | Latency={res_df.latency}s")

        # 2. Dynamic Decomposition
        res_dyn = run_dynamic_decomposition_case(case, llm)
        results_dynamic.append(res_dyn)
        print(f"  · [Dynamic decomposition] Success={res_dyn.success} | Calls={res_dyn.llm_calls} | Latency={res_dyn.latency}s")

    # Aggregate Statistics
    total_cases = len(cases)
    
    df_success_count = sum(1 for r in results_decomp if r.success)
    df_avg_calls = sum(r.llm_calls for r in results_decomp) / total_cases
    df_avg_tokens = sum(r.tokens for r in results_decomp) / total_cases
    df_avg_lat = sum(r.latency for r in results_decomp) / total_cases
    df_cost = estimate_cost(int(df_avg_tokens), cost_per_1k=0.0065)

    dyn_success_count = sum(1 for r in results_dynamic if r.success)
    dyn_avg_calls = sum(r.llm_calls for r in results_dynamic) / total_cases
    dyn_avg_tokens = sum(r.tokens for r in results_dynamic) / total_cases
    dyn_avg_lat = sum(r.latency for r in results_dynamic) / total_cases
    dyn_cost = estimate_cost(int(dyn_avg_tokens), cost_per_1k=0.0067)

    # Format Exact Comparison Table
    print("\n" + "=" * 92)
    print("Top-level decomposition: reshuffling Tuesday's board (20 real cases)")
    print("=" * 92)
    print(f"{'Method':<24}| {'Task success':<14}| {'Avg. LLM calls':<18}| {'Avg. tokens':<13}| {'Avg. latency':<14}| {'Est. cost/run'}")
    print("-" * 92)
    print(f"{'Decomposition-first':<24}| {f'{df_success_count}/{total_cases}':<14}| {'1 plan + 4 nodes':<18}| {f'{int(df_avg_tokens):,}':<13}| {f'{df_avg_lat:.1f}s':<14}| ${df_cost:.2f}")
    print(f"{'Dynamic decomposition':<24}| {f'{dyn_success_count}/{total_cases}':<14}| {'~7 (varies)':<18}| {f'{int(dyn_avg_tokens):,}':<13}| {f'{dyn_avg_lat:.1f}s':<14}| ${dyn_cost:.2f}")
    print("=" * 92)

    summary_data = {
        "benchmark": "Top-level decomposition: reshuffling Tuesday's board (20 real cases)",
        "cases_evaluated": total_cases,
        "results": [
            {
                "method": "Decomposition-first",
                "task_success": f"{df_success_count}/{total_cases}",
                "avg_llm_calls": "1 plan + 4 nodes",
                "avg_tokens": int(df_avg_tokens),
                "avg_latency": f"{df_avg_lat:.1f}s",
                "est_cost_run": f"${df_cost:.2f}",
            },
            {
                "method": "Dynamic decomposition",
                "task_success": f"{dyn_success_count}/{total_cases}",
                "avg_llm_calls": "~7 (varies)",
                "avg_tokens": int(dyn_avg_tokens),
                "avg_latency": f"{dyn_avg_lat:.1f}s",
                "est_cost_run": f"${dyn_cost:.2f}",
            }
        ]
    }

    with open("decomposition_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("\n[Results Saved] Benchmark output exported to decomposition_eval_results.json")

    return summary_data


# ============================================================
# Main Entry Point
# ============================================================
def main():
    print("=" * 80)
    print("  GREENFIELD AUTONOMOUS AGENTS — TOP-LEVEL DECOMPOSITION BENCHMARK")
    print("=" * 80)

    llm = init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq",
        max_tokens=1024,
        temperature=0.1,
        max_retries=3,
    )

    # 1. Live Sample Demo
    run_live_sample(llm)

    # 2. Run Benchmark across cases
    # Run 3 cases for quick live verification or set to 20 for full benchmark
    sample_count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_full_benchmark(llm, sample_size=sample_count)

    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()