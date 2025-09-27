from icd.runtime.orchestrator import evaluate_acceptance


def test_evaluate_acceptance_pass():
    decision = evaluate_acceptance(delta_J=-0.02, epsilon_J=0.01, retry_budget=0)
    assert decision == {"accepted": True, "rolled_back": False, "retry": False}


def test_evaluate_acceptance_retry():
    decision = evaluate_acceptance(delta_J=0.005, epsilon_J=0.01, retry_budget=2)
    assert decision["accepted"] is False
    assert decision["rolled_back"] is True
    assert decision["retry"] is True
