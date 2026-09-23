"""
Сравнение версий/настроек агента на сценариях стресс-стенда.

    python tools/compare_versions.py "old=agent_old.py,new=agent.py" 0,1,2 "{'new': {'z_safe': 0.5}}"

Каждая версия — путь к файлу с классом Agent; kwargs задаются словарём по имени версии.
Запускать из корня репозитория.
"""
import ast
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd  # noqa: E402

from mock_environment import _mock_impact_model  # noqa: E402
from stress_eval import _scenarios, run_once  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mods = {n: load(p, n) for n, p in (a.split("=") for a in sys.argv[1].split(","))}
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    kwargs = ast.literal_eval(sys.argv[3]) if len(sys.argv) > 3 else {}
    profile = pd.read_csv("customer_profile.csv")
    dict_tariff = pd.read_csv("data/dict_tariff.csv")
    scen = _scenarios(_mock_impact_model(pd.read_csv("data/change_tariff.csv")))
    rows = []
    for name, mod in mods.items():
        for sname, model in scen.items():
            for s in seeds:
                r = run_once(lambda: mod.Agent(**kwargs.get(name, {})), model, s, profile, dict_tariff)
                rows.append((name, sname, s, r["net"]))
    df = pd.DataFrame(rows, columns=["cfg", "scen", "seed", "net"])
    pd.set_option("display.width", 250)
    print((df.groupby(["scen", "cfg"], sort=False)["net"].median().unstack() / 1e6).round(2).to_string())
    print(df.groupby("cfg", sort=False).agg(median=("net", "median"), mean=("net", "mean"), worst=("net", "min"),
                                            pos=("net", lambda s: (s > 0).mean())).round(0).to_string())


if __name__ == "__main__":
    main()
