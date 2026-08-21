# 04. Target Database Design

## 1. Schema Enhancements & Normalization
The target database schema extends the relational model to support complete auditability, precise geographic buffer attributes, and incident resolution:

### Enhanced / New Columns:
1. **`Fields` Table**: Added `canal_distance_meters REAL` and `is_organic_border BOOLEAN` to enable direct SQL buffer queries instead of text parsing.
2. **`Chemicals` Table**: Added `min_canal_buffer_m REAL`, `min_organic_buffer_m REAL`, and `max_wind_speed_kmh REAL` to parameterize all regulatory checks in data.
3. **`Equipment` Table**: Added `operating_hours REAL` and `last_calibrated_at DATETIME`.
4. **`Incident_Notes` Table**: Added `dispatch_id FK` and `resolved_at DATETIME`.
5. **`Audit_Logs` Table (NEW)**:
   - `audit_id INTEGER PRIMARY KEY AUTOINCREMENT`
   - `actor_type TEXT NOT NULL`
   - `actor_id INTEGER NOT NULL`
   - `action_name TEXT NOT NULL`
   - `payload_json TEXT NOT NULL`
   - `status TEXT NOT NULL`
   - `timestamp DATETIME DEFAULT CURRENT_TIMESTAMP`

---

## 2. Migration & Indexing Strategy
- **Migration Framework**: Integrate **Alembic** to manage versioned database migrations (`alembic revision --autogenerate`).
- **Performance Indexes**:
  - `CREATE INDEX idx_equipment_status ON Equipment(status);`
  - `CREATE INDEX idx_fields_customer ON Fields(customer_id);`
  - `CREATE INDEX idx_dispatch_status ON Dispatch_Jobs(status);`
  - `CREATE INDEX idx_incident_equipment ON Incident_Notes(equipment_id);`

---

## 3. Architectural Diagram Reference
- TO-BE ER Diagram: [`diagrams/to-be/database-er.mmd`](../diagrams/to-be/database-er.mmd)
