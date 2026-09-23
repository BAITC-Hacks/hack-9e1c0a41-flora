"""
Стресс-проверка агента на «чужих» моделях эффектов.

Зачем. Мок-среда организаторов построена из той же истории (data/change_tariff.csv),
что и априорная оценка агента, поэтому local_eval.py завышает результат любого агента,
который опирается на историю. На судействе эффекты другие. Здесь мы строим несколько
синтетических моделей эффектов, которые расходятся с историей по-разному, и
проверяем, что агент не разваливается.

Стенд использует только публичную механику (environment.make_environment и
scoring_core.score_campaigns) и собственные синтетические эффекты. Агенту он
ничего не передаёт: agent.py этот файл не импортирует.

    python stress_eval.py                  # все сценарии, seed 0-4, агент из agent.py
    python stress_eval.py --seeds 10       # больше seed
    python stress_eval.py --template       # сравнить с шаблоном организаторов
"""

import argparse
import contextlib
import io
import time

import numpy as np
import pandas as pd

from environment import make_environment
from mock_environment import CHANNELS, TOTAL_BUDGET, MAX_TOTAL_CONTACTS, _mock_impact_model, _mock_fallback
from scoring_core import MAX_CAMPAIGNS, score_campaigns, sanitize_campaigns

FILTER_COLUMNS = ["filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff"]


def _scenarios(base: pd.DataFrame, rng_seed=2026):
    """Синтетические модели эффектов, расходящиеся с историей."""
    rng = np.random.default_rng(rng_seed)
    out = {"history (как мок)": base}

    s = base.copy()
    targets = s["tariff_plan_code_to"].unique()
    factor = dict(zip(targets, rng.uniform(0.3, 1.5, len(targets))))
    s["arpu_change_pct"] = s["arpu_change_pct"] * s["tariff_plan_code_to"].map(factor) - 0.05
    out["масштаб по тарифу −0.05"] = s

    s = base.copy()
    s["arpu_change_pct"] = s["arpu_change_pct"] + rng.normal(0, 0.35, len(s))
    out["шум ±0.35 на связку"] = s

    s = base.copy()
    flip = rng.random(len(s)) < 0.3
    s.loc[flip, "arpu_change_pct"] = -s.loc[flip, "arpu_change_pct"]
    out["30% связок с обратным знаком"] = s

    s = base.copy()
    s["arpu_change_pct"] = (s.groupby(["tariff_plan_code_from", "arpu_segment"], observed=True)["arpu_change_pct"]
                            .transform(lambda x: x.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).values))
    out["история перемешана"] = s

    s = base.copy()
    s["arpu_change_pct"] = s["arpu_change_pct"] - 0.35
    out["пессимистичный сдвиг −0.35"] = s

    s = base.copy()
    s["arpu_change_pct"] = rng.normal(0.1, 0.8, len(s)).clip(-1, 3)
    out["эффекты не связаны с историей"] = s

    s = base.copy()
    s["arpu_change_pct"] = (0.3 * s["arpu_change_pct"] + rng.normal(0.05, 0.6, len(s))).clip(-1, 3)
    out["слабая связь с историей"] = s

    s = base.copy()
    s["conversion_rate"] = (s["conversion_rate"] * rng.uniform(0.3, 2.0, len(s))).clip(upper=1.0)
    out["конверсия ×0.3–2.0"] = s
    return out


def run_once(agent_factory, impact_model, seed, profile, dict_tariff):
    env, internals = make_environment(
        customer_profile=profile, impact_model=impact_model, dict_tariff=dict_tariff,
        channels=CHANNELS, total_budget=TOTAL_BUDGET, max_total_contacts=MAX_TOTAL_CONTACTS,
        fallback_predict=_mock_fallback, seed=seed)
    agent = agent_factory()
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            final = agent.act(env) or []
            crashed = False
        except Exception:
            final, crashed = [], True
        final = sanitize_campaigns(final, env.tariffs)[:MAX_CAMPAIGNS]
    elapsed = time.time() - t0
    pilots = internals.executed_pilot_campaigns()
    camps = pd.DataFrame(pilots + final)
    if camps.empty:
        return {"net": 0.0, "n_final": 0, "n_pilots": 0, "sec": elapsed, "crashed": crashed}
    for col in FILTER_COLUMNS + ["explicit_ids"]:
        if col not in camps.columns:
            camps[col] = None
    with contextlib.redirect_stdout(io.StringIO()):
        res = score_campaigns(camps, env.customer_profile, impact_model, env.tariffs,
                              profile["predicted_arpu"].sum(), _mock_fallback)
    return {"net": res["net_arpu_gain"], "n_final": len(final), "n_pilots": len(pilots),
            "sec": elapsed, "crashed": crashed}


def optimum(impact_model, profile, dict_tariff):
    """Грубая верхняя оценка: лучший SMS-эффект по каждой ячейке, если бы эффекты были известны."""
    cells = profile.groupby(["current_tariff", "arpu_segment"]).agg(
        n=("ID_NUMBER", "size"), a=("predicted_arpu", "mean")).reset_index()
    m = impact_model.copy()
    m["r"] = m["arpu_change_pct"] * (m["conversion_rate"] * CHANNELS["sms"]["conversion_multiplier"]).clip(upper=1)
    best = m.groupby(["tariff_plan_code_from", "arpu_segment"], observed=True)["r"].max().reset_index()
    x = cells.merge(best, left_on=["current_tariff", "arpu_segment"],
                    right_on=["tariff_plan_code_from", "arpu_segment"])
    x["v"] = (x["r"] * x["a"] - CHANNELS["sms"]["cost_per_contact"]).clip(lower=0)
    x = x[x["v"] > 0].sort_values("v", ascending=False)
    x["n_use"] = x["n"].clip(upper=5000)
    x = x[x["n_use"].cumsum() <= MAX_TOTAL_CONTACTS]
    return float((x["v"] * x["n_use"]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--template", action="store_true", help="добавить шаблон организаторов для сравнения")
    args = ap.parse_args()

    profile = pd.read_csv("customer_profile.csv")
    dict_tariff = pd.read_csv("data/dict_tariff.csv")
    base = _mock_impact_model(pd.read_csv("data/change_tariff.csv"))

    from agent import Agent
    agents = {"agent": Agent}
    if args.template:
        from agent_template import Agent as TemplateAgent
        agents["template"] = TemplateAgent

    rows = []
    for name, model in _scenarios(base).items():
        opt = optimum(model, profile, dict_tariff)
        for label, factory in agents.items():
            for seed in range(args.seeds):
                r = run_once(factory, model, seed, profile, dict_tariff)
                rows.append({"scenario": name, "agent": label, "seed": seed, "optimum_sms": opt, **r})

    df = pd.DataFrame(rows)
    summary = (df.groupby(["scenario", "agent"], sort=False)
               .agg(median=("net", "median"), min=("net", "min"), max=("net", "max"),
                    positive=("net", lambda s: f"{(s > 0).sum()}/{len(s)}"),
                    optimum=("optimum_sms", "first"), pilots=("n_pilots", "mean"),
                    final=("n_final", "mean"), sec=("sec", "max"), crashed=("crashed", "sum")))
    pd.set_option("display.width", 200)
    print(summary.to_string(float_format=lambda v: f"{v:,.0f}"))
    agent_rows = df[df["agent"] == "agent"]
    print(f"\nИтог агента: медиана по всем сценариям {agent_rows['net'].median():,.0f}; "
          f"в плюсе {(agent_rows['net'] > 0).sum()}/{len(agent_rows)}; "
          f"худший {agent_rows['net'].min():,.0f}; макс. время {agent_rows['sec'].max():.1f} с")


if __name__ == "__main__":
    main()
