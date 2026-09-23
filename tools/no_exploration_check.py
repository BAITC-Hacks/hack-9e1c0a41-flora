"""
Калибровка стресс-стенда по подсказке ТЗ: «стратегия без разведки приносит в ~15 раз меньше,
чем знающая истинные эффекты». Для каждого сценария считаем план «доверяем истории, без пилотов»
и делим на оракул (ЛП-верхняя граница). Сценарии с долей около 1/15 считаем похожими на судейские.

    python tools/no_exploration_check.py
"""
import contextlib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd  # noqa: E402

import agent as A  # noqa: E402
from mock_environment import CHANNELS, _mock_fallback, _mock_impact_model  # noqa: E402
from scoring_core import score_campaigns  # noqa: E402
from stress_eval import FILTER_COLUMNS, _scenarios, oracle  # noqa: E402


class _Env:
    pass


def main():
    profile = pd.read_csv("customer_profile.csv")
    dict_tariff = pd.read_csv("data/dict_tariff.csv")
    scen = _scenarios(_mock_impact_model(pd.read_csv("data/change_tariff.csv")))
    env = _Env()
    env.customer_profile, env.tariffs, env.channels = profile, dict_tariff, CHANNELS
    agent = A.Agent()
    cand = agent._candidates(env, agent._prior(env))
    cand["v"] = cand["m0"] * cand["arpu_mean"] - CHANNELS["sms"]["cost_per_contact"]
    best = cand.sort_values("v", ascending=False).groupby("cell").head(1)
    best = best[best["v"] > 0].sort_values("v", ascending=False)
    best = best[best["n_use"].cumsum() <= 15000]
    groups = (best.groupby(["arpu_segment", "target_tariff"])
              .agg(t=("current_tariff", ";".join), val=("v", "sum")).reset_index()
              .sort_values("val", ascending=False).head(10))
    plan = pd.DataFrame([{"campaign_name": f"h{i}", "filter_arpu_segment": r.arpu_segment,
                          "filter_current_tariff": r.t, "target_tariff": r.target_tariff, "channel": "sms"}
                         for i, r in enumerate(groups.itertuples())])
    for c in FILTER_COLUMNS + ["explicit_ids"]:
        if c not in plan:
            plan[c] = None
    for name, model in scen.items():
        with contextlib.redirect_stdout(io.StringIO()):
            res = score_campaigns(plan, profile, model, dict_tariff, profile["predicted_arpu"].sum(), _mock_fallback)
        opt = oracle(model, profile, dict_tariff)
        net = res["net_arpu_gain"]
        print(f"{name:38s} без разведки {net:>13,.0f}  оракул {opt:>13,.0f}  доля {net / opt:6.1%}")


if __name__ == "__main__":
    main()
