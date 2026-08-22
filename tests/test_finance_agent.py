"""
tests/test_finance_agent.py

Unit & Integration Tests for Greenfield Finance Agent Graph.
Tests all paths in agent/workflows/finance_graph.mmd including Advice, Financing,
HITL admin reviews, Document validation loops, Provider responses, and Farmer confirmations.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from agent.workflows.finance_agent import (
    build_finance_graph,
    create_finance_agent,
    run_finance_turn,
    FinanceState,
    get_db_connection,
)


@pytest.fixture(autouse=True)
def ensure_db_clean():
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO Customers (customer_id, company_name, credit_hold) VALUES (1, 'Nile Delta Farms', 0)")
        conn.execute("INSERT OR IGNORE INTO Customers (customer_id, company_name, credit_hold) VALUES (2, 'Behera Agro Cooperative', 1)")
        conn.execute("INSERT OR IGNORE INTO Customers (customer_id, company_name, credit_hold) VALUES (3, 'Fayoum Green Estates', 0)")
        conn.execute("UPDATE Customers SET credit_hold = 0 WHERE customer_id = 1")
        conn.execute("UPDATE Customers SET credit_hold = 1 WHERE customer_id = 2")
        conn.commit()


# ==============================================================================
# Helpers
# ==============================================================================

def has_log(result: dict, tag: str) -> bool:
    return any(tag in str(item) for item in result.get("execution_log", []))


# ==============================================================================
# Financial Advice Path Tests
# ==============================================================================

def test_advice_equipment_path():
    """Tests financial advice workflow routing through equipment specialist."""
    graph = create_finance_agent(interactive=False)
    state_input: FinanceState = {
        "farmer_id": 1,
        "farmer_request": "Should I lease or purchase a new high-clearance sprayer SPR-3001 for next season?",
        "request_type": "advice",
        "specialist_type": "equipment",
    }
    result = graph.invoke(state_input)

    assert result["request_type"] == "advice"
    assert has_log(result, "EQUIPMENT")
    assert has_log(result, "OPTIONS")
    assert has_log(result, "TOT")
    assert has_log(result, "RAG")
    assert has_log(result, "RECOMMEND")
    assert result.get("recommendation") is not None
    assert len(result.get("financial_options", [])) > 0


def test_advice_crop_path():
    """Tests financial advice workflow routing through crop specialist."""
    graph = create_finance_agent(interactive=False)
    state_input: FinanceState = {
        "farmer_id": 1,
        "farmer_request": "What is the expected yield ROI and seasonal cashflow for wheat on Field 1?",
        "request_type": "advice",
        "specialist_type": "crop",
    }
    result = graph.invoke(state_input)

    assert result["request_type"] == "advice"
    assert has_log(result, "CROP")
    assert has_log(result, "OPTIONS")
    assert has_log(result, "TOT")
    assert has_log(result, "RAG")
    assert has_log(result, "RECOMMEND")
    assert result.get("recommendation") is not None


def test_advice_general_path():
    """Tests financial advice workflow with no domain specialist needed."""
    graph = create_finance_agent(interactive=False)
    state_input: FinanceState = {
        "farmer_id": 1,
        "farmer_request": "General working capital budgeting advice for farm operations next quarter.",
        "request_type": "advice",
        "specialist_type": "no",
    }
    result = graph.invoke(state_input)

    assert result["request_type"] == "advice"
    assert has_log(result, "OPTIONS")
    assert has_log(result, "TOT")
    assert has_log(result, "RECOMMEND")


# ==============================================================================
# Financing Request Path Tests
# ==============================================================================

def test_financing_ineligible_rejection():
    """Tests customer on credit hold being rejected immediately at eligibility step."""
    graph = create_finance_agent(interactive=False)
    state_input: FinanceState = {
        "farmer_id": 2,  # Customer 2 has credit_hold = 1
        "farmer_request": "I want to apply for a $30,000 seasonal crop loan.",
        "request_type": "financing",
    }
    result = graph.invoke(state_input)

    assert result["request_type"] == "financing"
    assert result.get("eligibility_status") is False
    assert has_log(result, "REJECT")
    assert "credit hold" in (result.get("rejection_reason") or "").lower()


def test_financing_successful_standard_path():
    """Tests standard financing path without HITL (amount < $50k, valid docs)."""
    graph = create_finance_agent(interactive=False)
    state_input: FinanceState = {
        "farmer_id": 1,
        "farmer_request": "Need $35,000 loan for irrigation upgrades.",
        "request_type": "financing",
        "documents_submitted": {
            "government_id": "id_doc.pdf",
            "farm_tax_return": "tax_2025.pdf",
            "bank_statements": "bank_stmts.pdf",
            "land_deed_or_lease": "deed.pdf",
        },
        "financial_analysis": {
            "assessed_amount": 35000.0,
            "dscr": 1.6,
            "risk_level": "low",
            "recommended_term_months": 36,
            "max_borrowing_capacity": 60000.0,
        },
        "provider_response": "approved",
        "farmer_accepts": True,
    }
    result = graph.invoke(state_input)

    assert result.get("eligibility_status") is True
    assert result.get("documents_valid") is True
    assert result.get("hitl_required") is False
    assert has_log(result, "SUBMIT")
    assert has_log(result, "FARMER_CONFIRM")
    assert has_log(result, "PROCESS")
    assert has_log(result, "VERIFY")
    assert result.get("transaction_verification", {}).get("verified") is True


def test_financing_hitl_admin_approval():
    """Tests large loan (>= $50k) triggering HITL admin review and manager approval."""
    graph = create_finance_agent(interactive=False)
    state_input: FinanceState = {
        "farmer_id": 1,
        "farmer_request": "Need $80,000 financing for new tractor equipment.",
        "request_type": "financing",
        "documents_submitted": {
            "government_id": "id_doc.pdf",
            "farm_tax_return": "tax_2025.pdf",
            "bank_statements": "bank_stmts.pdf",
            "land_deed_or_lease": "deed.pdf",
        },
        "financial_analysis": {
            "assessed_amount": 80000.0,
            "dscr": 1.35,
            "risk_level": "medium",
            "recommended_term_months": 48,
            "max_borrowing_capacity": 100000.0,
        },
        "admin_decision": "approve",
        "provider_response": "approved",
        "farmer_accepts": True,
    }
    result = graph.invoke(state_input)

    assert result.get("hitl_required") is True
    assert has_log(result, "ADMIN")
    assert has_log(result, "SUBMIT")
    assert has_log(result, "PROCESS")
    assert has_log(result, "VERIFY")


def test_financing_hitl_admin_rejection():
    """Tests HITL admin review where manager rejects application."""
    graph = create_finance_agent(interactive=False)
    state_input: FinanceState = {
        "farmer_id": 1,
        "farmer_request": "Need $95,000 loan.",
        "request_type": "financing",
        "documents_submitted": {
            "government_id": "id_doc.pdf",
            "farm_tax_return": "tax_2025.pdf",
            "bank_statements": "bank_stmts.pdf",
            "land_deed_or_lease": "deed.pdf",
        },
        "financial_analysis": {
            "assessed_amount": 95000.0,
            "dscr": 1.1,
            "risk_level": "high",
            "recommended_term_months": 60,
            "max_borrowing_capacity": 50000.0,
        },
        "admin_decision": "reject",
    }
    result = graph.invoke(state_input)

    assert result.get("hitl_required") is True
    assert has_log(result, "ADMIN")
    assert has_log(result, "REJECT")
    assert not has_log(result, "PROCESS")


def test_financing_provider_rejection_to_alternatives():
    """Tests provider declining application, routing to alternative option generation and recommendation."""
    graph = create_finance_agent(interactive=False)
    state_input: FinanceState = {
        "farmer_id": 1,
        "farmer_request": "Need $40,000 loan.",
        "request_type": "financing",
        "documents_submitted": {
            "government_id": "id_doc.pdf",
            "farm_tax_return": "tax_2025.pdf",
            "bank_statements": "bank_stmts.pdf",
            "land_deed_or_lease": "deed.pdf",
        },
        "financial_analysis": {
            "assessed_amount": 40000.0,
            "dscr": 1.5,
            "risk_level": "low",
            "recommended_term_months": 36,
            "max_borrowing_capacity": 50000.0,
        },
        "provider_response": "rejected",
    }
    result = graph.invoke(state_input)

    assert has_log(result, "PROVIDER_REJECTED")
    assert has_log(result, "ALTERNATIVE")
    assert has_log(result, "RECOMMEND")
    assert len(result.get("alternative_options", [])) > 0


def test_financing_farmer_declined_to_alternatives():
    """Tests farmer rejecting offered provider terms, routing to alternative option generation."""
    graph = create_finance_agent(interactive=False)
    state_input: FinanceState = {
        "farmer_id": 1,
        "farmer_request": "Need $30,000 loan.",
        "request_type": "financing",
        "documents_submitted": {
            "government_id": "id_doc.pdf",
            "farm_tax_return": "tax_2025.pdf",
            "bank_statements": "bank_stmts.pdf",
            "land_deed_or_lease": "deed.pdf",
        },
        "financial_analysis": {
            "assessed_amount": 30000.0,
            "dscr": 1.6,
            "risk_level": "low",
            "recommended_term_months": 24,
            "max_borrowing_capacity": 50000.0,
        },
        "provider_response": "approved",
        "farmer_accepts": False,  # Farmer declines terms
    }
    result = graph.invoke(state_input)

    assert has_log(result, "FARMER_CONFIRM")
    assert has_log(result, "ALTERNATIVE")
    assert has_log(result, "RECOMMEND")
    assert not has_log(result, "PROCESS")


# ==============================================================================
# Interactive Turn / Interrupt Resumption Tests
# ==============================================================================

def test_interactive_turn_document_upload_resumption():
    """Tests interactive checkpointing across multiple turns: documents, provider, and farmer acceptance."""
    mem = MemorySaver()
    graph = create_finance_agent(checkpointer=mem, interactive=True)
    thread_id = "test_thread_interactive_docs_101"
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1: Initial financing request -> pauses before wait_farmer
    initial_state = {
        "farmer_id": 1,
        "farmer_request": "I want to apply for $20,000 equipment financing.",
        "request_type": "financing",
    }
    run_finance_turn(graph, thread_id, initial_state)
    state_snap1 = graph.get_state(config)
    assert "wait_farmer" in state_snap1.next

    # Turn 2: Farmer uploads documents -> resumes through validation & submit -> pauses before wait_provider
    doc_update = {
        "documents_submitted": {
            "government_id": "id.pdf",
            "farm_tax_return": "tax.pdf",
            "bank_statements": "bank.pdf",
            "land_deed_or_lease": "deed.pdf",
        },
        "financial_analysis": {
            "assessed_amount": 20000.0,
            "dscr": 1.8,
            "risk_level": "low",
            "recommended_term_months": 24,
            "max_borrowing_capacity": 50000.0,
        },
    }
    run_finance_turn(graph, thread_id, doc_update)
    state_snap2 = graph.get_state(config)
    assert "wait_provider" in state_snap2.next

    # Turn 3: Provider returns approval -> pauses before farmer_confirm
    provider_update = {
        "provider_response": "approved",
    }
    run_finance_turn(graph, thread_id, provider_update)
    state_snap3 = graph.get_state(config)
    assert "farmer_confirm" in state_snap3.next

    # Turn 4: Farmer accepts loan terms -> completes process and verification
    farmer_update = {
        "farmer_accepts": True,
    }
    final_snap = run_finance_turn(graph, thread_id, farmer_update)
    assert has_log(final_snap, "VERIFY")
    assert final_snap.get("transaction_verification", {}).get("verified") is True



