"""Семейство реалистичных сценариев: эффекты не связаны / слабо связаны с историей, разные случайные реализации.

Автор стенда — независимый ревьюер (Claude, аналитический чат команды). Запуск из корня репозитория:
    python tools/realistic_bench.py "{'agent': ('agent', {})}" 3 300
"""
import ast, os, sys, importlib, warnings, contextlib, io
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
# Использование: python REVIEW_04_realistic_bench.py "{'имя': ('модуль', {kwargs})}" <число seed> [базовый seed реализаций, по умолчанию 100]
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from stress_eval import run_once, oracle
from mock_environment import _mock_impact_model
profile = pd.read_csv("customer_profile.csv"); dt = pd.read_csv("data/dict_tariff.csv")
base = _mock_impact_model(pd.read_csv("data/change_tariff.csv"))
scen = {}
for k in range(6):
    rng = np.random.default_rng(int(sys.argv[3]) + k if len(sys.argv) > 3 else 100 + k); s = base.copy()
    s["arpu_change_pct"] = rng.normal(0.1, 0.8, len(s)).clip(-1, 3); scen[f"unrel#{k}"] = s
for k in range(3):
    rng = np.random.default_rng((int(sys.argv[3]) + 100 if len(sys.argv) > 3 else 200) + k); s = base.copy()
    s["arpu_change_pct"] = (0.3 * s["arpu_change_pct"] + rng.normal(0.05, 0.6, len(s))).clip(-1, 3); scen[f"weak#{k}"] = s
configs = ast.literal_eval(sys.argv[1]); seeds = range(int(sys.argv[2]))
opt = {n: oracle(m, profile, dt) for n, m in scen.items()}
out = []
for name, (mod, kw) in configs.items():
    A = importlib.import_module(mod)
    for sname, model in scen.items():
        for sd in seeds:
            r = run_once(lambda: A.Agent(**kw), model, sd, profile, dt)
            out.append((name, sname, sd, r["net"], r["net"] / opt[sname]))
df = pd.DataFrame(out, columns=["cfg", "scen", "seed", "net", "eff"])
pd.set_option("display.width", 250)
print((df.groupby(["scen", "cfg"], sort=False)["net"].median().unstack() / 1e6).round(2).to_string())
print(df.groupby("cfg", sort=False).agg(mean_net=("net", "mean"), worst=("net", "min"), mean_eff=("eff", "mean"),
      pos=("net", lambda s: (s > 0).mean())).round(2).to_string())
