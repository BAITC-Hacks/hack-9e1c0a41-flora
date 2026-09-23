"""
Регрессионные тесты агента (найдены при ревью 23.09.2026).

    python tests/test_agent_regressions.py      # или: python -m pytest tests/

1. Запасной (жадный) планировщик должен соблюдать правило риска v5: если пилоты показали,
   что история ненадёжна, дополнительный push по непроверенным ячейкам не делается.
   Контроль: при надёжной истории тот же сценарий даёт дополнительный push.
2. Объяснение не обещает положительную нижнюю границу кампаниям дополнительного push.
3. Ошибка солвера не роняет агента на мок-среде: план строится запасным отбором.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from agent import Agent  # noqa: E402

OPT_COLUMNS = ["cell", "target_tariff", "channel", "v_mean", "v_low", "unit_cost", "n_use"]


def _fallback_plan(scale):
    """Одна непроверенная ячейка с положительной средней и отрицательной нижней оценкой;
    основной отбор пуст, солвер падает — работает запасной жадный путь."""
    agent = Agent()
    agent.scale_ = scale
    agent._scale = lambda env, channel: 1.0
    empty = pd.DataFrame(columns=OPT_COLUMNS)
    agent._options = lambda *args: empty
    agent._pack_and_report = lambda env, cand, mu, sd, chosen: chosen

    def fail_solver(*args):
        raise RuntimeError("injected solver failure")

    agent._plan_milp = fail_solver
    cand = pd.DataFrame([dict(cell="a|MID", current_tariff="a", target_tariff="b", arpu_segment="MID",
                              n_use=10, n_obs=0, arpu_mean=100.0)])
    env = SimpleNamespace(remaining_contacts=100, remaining_budget=100.0)
    return agent, agent._plan(env, cand, np.array([0.1]), np.array([[0.25]]))


def test_fallback_respects_risk_rule_when_history_unreliable():
    agent, plan = _fallback_plan((2.0, 2.0))
    assert agent._history_unreliable()
    assert [c for c in plan if c.get("channel") == "push"] == [], plan


def test_fallback_keeps_extra_push_when_history_reliable():
    agent, plan = _fallback_plan((1.0, 1.0))
    assert not agent._history_unreliable()
    assert len(plan) == 1 and plan[0]["channel"] == "push" and plan[0].get("extra"), plan


def test_explanation_does_not_overpromise_extra_push():
    agent = Agent()
    agent.trace, agent.n_extra_campaigns_ = [], 2
    text = agent._plan_summary([{}] * 5)
    assert "3 — из связок" in text and "не гарантирована" in text, text


def test_solver_failure_on_mock_env_still_returns_valid_plan():
    from mock_environment import make_mock_env
    env, _ = make_mock_env(seed=42)
    agent = Agent()

    def fail_solver(*args):
        raise RuntimeError("injected solver failure")

    agent._select_milp = fail_solver
    plan = agent.act(env)
    assert 1 <= len(plan) <= 10
    assert any("Ошибка оптимизатора" in t["note"] for t in agent.trace)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"Все тесты пройдены: {len(tests)}")
