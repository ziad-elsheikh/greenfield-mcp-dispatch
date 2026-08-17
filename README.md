# Greenfield Dispatch & Fleet Logistics Planning Agent
## Decomposition & Planning Lab: Autonomous Multi-Field Agricultural Scheduling

---

## 1. Problem Identification & Operational Rationale

### The Real-World Planning Problem
In high-throughput agricultural operations at **Greenfield Agricultural Agency**, daily field operations (tilling, harvesting, and restricted chemical spraying) are governed by coupled physical, regulatory, and mechanical constraints:
1. **Fleet & Implement Compatibility:** Tractors, high-clearance sprayers (`SPR-3001`), and combines have distinct hitch attachments, operational statuses (`idle`, `dispatched`, `maintenance`), and soil-compaction limits.
2. **Environmental & Safety Regulations:** Restricted chemical spraying is illegal when wind speeds exceed **15 km/h** (spray drift risk) or within **15m of irrigation canals** and **50m of organic boundaries** (`SOP-CHEM-4040`).
3. **Personnel & Customer Constraints:** Only certified technician dispatchers can execute restricted applications, and dispatches are blocked if a customer has an active credit hold.

**The Recurring Failure Point:** Multiple times a week, unforeseen operational disruptions occur—a primary sprayer suffers a hydraulic failure, unexpected 18 km/h wind gusts arise across northern fields, or flash-rain forecasts demand emergency harvest preemption. The logistics dispatcher must **completely reshuffle and re-plan the day's multi-field dispatch board**.

### Why This is a Planning Problem (Not Memory, Not RAG, Not Single-Shot)
* **Beyond Single Tool Calls:** A safe dispatch requires checking customer credit, verifying technician certification, validating wind/canal buffer constraints, and selecting alternative idle machines before dispatching.
* **Beyond RAG:** RAG retrieves static policy texts (e.g., "15m canal buffer required"). It cannot dynamically optimize multi-machine routes or resolve conflicting field priorities.
* **Beyond Memory:** Memory stores past notes and holds. It cannot explore permutations of candidate schedules under real-time resource contention.
* **High Cost of Plan Failure:** A bad plan causes chemical drift lawsuits, equipment compaction damage, or emergency harvest delays costing hundreds of thousands of dollars.

---

## 2. Architecture & Decomposition Stack

```
                          ┌───────────────────────────────────────────────┐
                          │    Top-Level Request / Operational Shock      │
                          │  "SPR-3001 broke down + Wind Advisory Field 4"│
                          └───────────────────────┬───────────────────────┘
                                                  │
                       ┌──────────────────────────┴──────────────────────────┐
                       ▼                                                     ▼
          [Decomposition-First (Static DAG)]                     [Dynamic / Interleaved DAG]
          • One-shot DAG plan generation                         • Step-by-step observation loop
          • Strict topological sort execution                    • Dynamically pivots if step fails
                       │                                                     │
                       └──────────────────────────┬──────────────────────────┘
                                                  │
            ┌─────────────────────────────────────┼─────────────────────────────────────┐
            ▼                                     ▼                                     ▼
     [Plan-and-Solve (PS)]              [Tree of Thoughts (ToT)]                 [Grounded LATS]
     • Linear calculation tasks         • Combinatorial ranking                • High-stakes final dispatch
     • Acreage / dosage math            • Priority sorting under               • Multi-step candidate rollout
     • Low latency & token cost           constrained technician capacity      • Verified by DB & buffer rules
            │                                     │                                     │
            └─────────────────────────────────────┼─────────────────────────────────────┘
                                                  │
                       ┌──────────────────────────┴──────────────────────────┐
                       ▼                                                     ▼
            [Self-Refine (Fast Loop)]                              [Reflexion (Deep Loop)]
            • Single-draft rubric critique                         • Multi-trial search with memory buffer
            • Technician work-order formatting                     • Multi-resource allocation conflicts
```

---

## 3. Implementation of the Five Core Concerns

### 1. Task Decomposition (Static DAG vs. Dynamic Interleaved)
* **Static DAG (`algorithms/decomposition.py`):** Generates an upfront acyclic dependency graph, validates acyclicity via topological sort (detecting cycles at construction time), and executes sub-tasks in dependency order.
* **Dynamic Decomposition (`algorithms/dynamic_decomposition.py`):** Generates subsequent sub-tasks conditionally after observing intermediate execution outputs.
* **The Divergence Case:** When sprayer `SPR-3001` breaks down, Static DAG plans to reroute `SPR-3002` immediately. Dynamic Decomposition queries `farm.db` in sub-task 1, discovers `SPR-3002` is already committed to high-priority Field 7, and **dynamically shifts course** to evaluate tractor `TRC-202` with a spray implement. Static DAG executes blindly and errors out at tool dispatch.

### 2. Planning Algorithms & Sub-Task Routing
* **Plan-and-Solve (`algorithms/plan_and_solve.py`):** Routed to deterministic, linear sub-tasks (e.g., computing chemical tank mix ratios and fuel requirements). Minimal latency (0.9s) and single LLM turn.
* **Tree of Thoughts (`algorithms/tree_of_thought.py`):** Routed to multi-criteria prioritization (e.g., ordering 5 queued fields under a 1-technician availability limit). Explores candidate sequences via BFS/DFS, scoring by crop vulnerability and weather windows.
* **LATS (`algorithms/lats.py`):** Routed to the high-stakes final fleet dispatch proposal. Uses Monte Carlo Tree Search (Select, Expand, Simulate, Reflect, Backpropagate) guided by real external environmental feedback.

### 3. Self-Correction Scopes (Self-Refine vs. Reflexion)
* **Self-Refine (`algorithms/self_refine.py`):** Cheap single-pass drafting and rubric critique for technician work orders and incident summaries (verifying PPE, nozzle pressure limits, and boundary warnings).
* **Reflexion (`algorithms/reflexion.py`):** Multi-trial constraint satisfaction with an episodic buffer. When a candidate dispatch violates soil moisture limits on wet Field 10, Reflexion stores a verbal reflection (*"Attempt 1 failed: TRC-205 causes soil compaction on wet soil; use lightweight TRC-201"*) and passes it into subsequent attempts.

### 4. Grounded vs. Ungrounded Validation (`algorithms/environment.py`)
* **The Danger of Ungrounded Evaluation:** Ungrounded LLM self-evaluation hallucinates that all machines are free and overlooks canal proximity.
* **Grounded `GreenfieldEnvironment`:** Evaluates candidate states against real ground truth:
  1. Real SQLite table checks (`Equipment.status == 'idle'`, `Customers.credit_hold == 0`).
  2. Agricultural domain rules (15 km/h wind limits, mandatory 15m canal buffer, 50m organic boundary).
* **Caught Failure Case:** An ungrounded critic approves spraying `Parathion` on Field 5. The grounded validator identifies Field 5's 10m proximity to an irrigation canal, penalizes the state with a `-0.35` score drop, and forces the search agent to select a low-hazard chemical alternative.

---

## 4. Cost & Quality Comparison Benchmark

### A. Top-Level Decomposition Benchmark: Reshuffling Tuesday's Board (20 Real Cases)
| Method | Task Success | Avg. LLM Calls | Avg. Tokens | Avg. Latency | Est. Cost / Run | Verdict & Deployment Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Decomposition-First (Static DAG)** | 14/20 (70%) | 1 plan + 4 nodes | 6,100 | 3.1s | $0.04 | Fast for routine jobs, but blind to mid-execution DB state changes. |
| **Dynamic Decomposition** | **17/20 (85%)** | ~7 (varies) | 8,900 | 5.4s | $0.06 | **Production Winner:** Essential for top-level reshuffle to react to dynamic blockers. |

### B. Sub-Task Planning Algorithms: Ranking & Proposal (15 Real Cases)
| Method | Sub-Task Success | Avg. LLM Calls | Avg. Tokens | Avg. Latency | Est. Cost / Run | Architectural Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Plan-and-Solve (Ranking)** | 11/15 (73%) | 1 | 1,400 | 0.9s | $0.01 | Fails complex combinatorial trade-offs; best for linear calculations. |
| **Tree of Thoughts (Ranking)** | **14/15 (93%)** | 9 | 5,200 | 3.8s | $0.04 | **Production Winner for Ranking:** Evaluates multiple valid field sequences. |
| **LATS (Ungrounded Default)** | 9/15 (60%) | 11 | 7,600 | 6.2s | $0.06 | Unsafe theater: hallucinated scores approve illegal spray dispatches. |
| **LATS (Grounded Environment)** | **14/15 (93%)** | 13 | 8,300 | 6.9s | $0.07 | **Production Winner for Final Proposal:** Hard DB & safety verification. |

---

## 5. Locatable Codebase Structure

```
greenfield-mcp-dispatch/
├── algorithms/
│   ├── decomposition.py          # Static DAG generation, topological sort & cycle check
│   ├── dynamic_decomposition.py  # Adaptive step-by-step dynamic decomposition
│   ├── plan_and_solve.py         # Plan-and-Solve linear planning algorithm
│   ├── tree_of_thought.py        # Tree of Thoughts (BFS/DFS search over permutations)
│   ├── lats.py                   # Language Agent Tree Search with MCTS & reflections
│   ├── self_refine.py            # Single-pass draft, critique, and refine loop
│   ├── reflexion.py              # Multi-trial search carrying episodic reflection memory
│   └── environment.py            # Grounded GreenfieldEnvironment (DB + domain validator)
├── agent/
│   ├── agent.py                  # Fleet planning agent & sub-task routing coordinator
│   └── schema.py                 # Pydantic schemas for DAG nodes and step actions
├── db/
│   ├── farm.db                   # SQLite database (Equipment, Fields, Customers, Jobs)
│   ├── schema.sql                # Relational schema with safety & status constraints
│   └── seed.sql                  # Production seed data
├── server/
│   ├── server.py                 # FastMCP dispatch server
│   └── tools.py                  # Atomic MCP tools (dispatch_equipment, process_payment)
├── demo.py                       # Full benchmark harness executing all 5 lab concerns
└── README.md                     # Architectural documentation & evaluation report
```

---

## 6. How to Run the Evaluation & Benchmarks

### 1. Environment Setup
```bash
# Clone the repository and install dependencies
git clone https://github.com/your-org/greenfield-mcp-dispatch.git
cd greenfield-mcp-dispatch
uv sync
```

### 2. Configure Environment Variables
Create a `.env` file with your LLM credentials:
```bash
GROQ_API_KEY=your_groq_api_key_here
GREENFIELD_DB_PATH=db/farm.db
```

### 3. Run the Full Decomposition & Planning Benchmark
```bash
python demo.py
```
This executes the 20 top-level Tuesday dispatch board cases and 15 sub-task ranking evaluations, outputting evaluation traces to `artifacts/` and generating comparative metrics.
