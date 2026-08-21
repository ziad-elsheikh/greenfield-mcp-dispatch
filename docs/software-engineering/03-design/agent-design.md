# 05. Target Agent & Algorithm Design

## 1. LangGraph StateGraph Architecture
The target agent replaces the sequential while-loop with an explicit state machine:

```
[Start] ──> [Classify Request] ──> [Select Algorithm / Tool]
                                           │
       ┌───────────────────────────────────┴───────────────────────────────────┐
       ▼                                   ▼                                   ▼
 [Static DAG Plan]                [Dynamic Adaptive]                  [Grounded LATS / ToT]
       │                                   │                                   │
       └───────────────────────────────────┬───────────────────────────────────┘
                                           ▼
                              [Grounded DB Validator]
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
             (Safety Conflict)                            (Checks Passed)
                     │                                           │
         [Human Sign-off Elicit]                                 ▼
                     │                                   [Execute Tool Call]
                     ▼                                           │
           (Approved / Declined)                                 ▼
                     │                                    [Self-RAG Check]
                     └─────────────────────┬─────────────────────┘
                                           ▼
                                 [Synthesize & Output]
```

---

## 2. Grounded Validator Architecture
Replaces text heuristics with direct parameterized database queries:

```python
class GroundedSQLValidator:
    def __init__(self, db_conn):
        self.conn = db_conn

    def validate_dispatch(self, equipment_id: int, field_id: int, chemical_id: Optional[int]) -> ValidationReport:
        # 1. Live Equipment Status Query
        eq = self.conn.execute("SELECT status FROM Equipment WHERE equipment_id = ?", (equipment_id,)).fetchone()
        if not eq or eq["status"] != "idle":
            return ValidationReport(valid=False, reason=f"Equipment {equipment_id} is not idle (status={eq['status'] if eq else 'not found'})")

        # 2. Live Proximity & Buffer Query
        if chemical_id:
            chem = self.conn.execute("SELECT * FROM Chemicals WHERE chemical_id = ?", (chemical_id,)).fetchone()
            field = self.conn.execute("SELECT * FROM Fields WHERE field_id = ?", (field_id,)).fetchone()
            if field["canal_distance_meters"] < chem["min_canal_buffer_m"]:
                return ValidationReport(valid=False, reason=f"Field canal proximity ({field['canal_distance_meters']}m) violates minimum {chem['min_canal_buffer_m']}m buffer")

        return ValidationReport(valid=True)
```
