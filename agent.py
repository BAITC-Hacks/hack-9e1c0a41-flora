"""
Агент Flora для кейса «Beeline Tariff Marketing Campaigns».

Постановка. Гипотеза = связка (текущий тариф, ARPU-сегмент) -> целевой тариф.
Её эффект θ (относительный прирост ARPU при SMS-контакте, конверсия внутри)
неизвестен. История смен тарифов даёт априорную оценку m0, но судейская
аудитория ведёт себя иначе, поэтому агент:

1. Строит совместную гауссову модель убеждений
       θ_c = β0 + β1 · m0_c + u_c,   u_c ~ N(0, τ_c²)
   Калибровка (β0, β1) общая для всех связок: каждый пилот уточняет оценки
   сразу всех гипотез, а не только проверенной (эмпирический Байес,
   коррелированные убеждения). Если история не работает, β1 уходит к нулю
   и агент перестаёт ей доверять.
2. Выбирает пилоты по Knowledge Gradient (Frazier, Powell, Dayanik):
   следующий пилот — тот, что сильнее всего увеличивает ожидаемую ценность
   итогового плана, плюс ожидаемый эффект самих пилотных контактов.
3. В финальный план берёт только связки, у которых НИЖНЯЯ граница оценки
   окупает контакт, выбирает канал (push / sms / digital_ads) под ценность
   абонента и распределяет лимиты контактов, бюджета и кампаний.

Решения детерминированы: при одинаковой среде и seed план одинаковый.

После act(env) доступны:
    agent.trace       — журнал решений (list[dict]);
    agent.plan_table  — итоговый план с обоснованием (DataFrame);
    agent.explanation — шаблонное текстовое объяснение (str).
"""

import numpy as np
import pandas as pd

# Публичные параметры механики (docs/PARTICIPANT_GUIDE.md, environment.py, scoring_core.py).
PER_CUSTOMER_STD = 0.804
MAX_CUSTOMERS_PER_CAMPAIGN = 5000
MAX_CAMPAIGNS = 10
MIN_PILOT, MAX_PILOT = 10, 200
ARPU_BINS = [-np.inf, 1000, 5000, np.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]
BASE_CHANNEL = "sms"
# call исключён: конверсия ограничена единицей, перенос оценки с SMS на звонок её завышает.
FINAL_CHANNELS = ("push", "sms", "digital_ads")

# Узлы Гаусса-Эрмита для математического ожидания по N(0, 1).
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()


class Agent:
    def __init__(self, history_path="data/change_tariff.csv", pilot_size=200,
                 z_safe=1.0, z_unpiloted=1.5, max_targets_per_cell=6,
                 beta_prior_sd=(0.03, 0.5), residual=(0.02, 0.3), stop_rule="full", min_kg=1.0,
                 scale_grid=(1.0, 1.5, 2.0, 3.0, 4.0, 6.0), verify=True):
        self.history_path = history_path
        self.pilot_size = pilot_size
        self.z_safe = z_safe                    # осторожность для проверенных пилотом связок
        self.z_unpiloted = z_unpiloted          # осторожность для связок, оценённых только через калибровку
        self.max_targets_per_cell = max_targets_per_cell
        self.beta_prior_sd = beta_prior_sd      # априорная неопределённость сдвига и наклона калибровки
        self.residual = residual                # индивидуальное отклонение связки: a + b·|m0|
        self.stop_rule = stop_rule              # "full" — тратить пилоты, пока они информативны; "kg" — строгий KG
        self.min_kg = min_kg
        self.scale_grid = scale_grid            # кандидаты масштаба отклонений для эмпирического Байеса
        self.verify = verify                    # остаток пилотов — на проверку крупнейших ставок плана
        self.scale_ = 1.0
        self.trace, self.plan_table, self.explanation = [], None, None

    # ------------------------------------------------------------------ utils
    def _log(self, phase, note, **fields):
        row = {"step": len(self.trace) + 1, "phase": phase, "current_tariff": None,
               "arpu_segment": None, "target_tariff": None, "channel": None, "n": None,
               "observed": None, "post_mean": None, "post_sd": None, "note": note}
        row.update(fields)
        self.trace.append(row)

    def _scale(self, env, channel):
        """Во сколько раз ratio канала больше ratio SMS (без потолка конверсии)."""
        return (env.channels[channel]["conversion_multiplier"]
                / env.channels[BASE_CHANNEL]["conversion_multiplier"])

    # ------------------------------------------------------------------ prior
    def _prior(self, env):
        """Априорная оценка эффекта (в единицах SMS) по истории смен тарифов."""
        try:
            hist = pd.read_csv(self.history_path)
        except (OSError, ValueError):
            return None
        hist = hist[hist["AVG_ARPU_PREV_3M"] >= 100].copy()
        hist["arpu_segment"] = pd.cut(hist["AVG_ARPU_PREV_3M"], bins=ARPU_BINS, labels=ARPU_LABELS).astype(str)
        hist["pct"] = ((hist["AVG_ARPU_NEXT_3M"] - hist["AVG_ARPU_PREV_3M"])
                       / hist["AVG_ARPU_PREV_3M"]).clip(-1, 3)
        g = (hist.groupby(["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"])["pct"]
             .agg(pct_mean="mean", pct_sd="std", n_hist="size").reset_index())
        total = g.groupby(["tariff_plan_code_from", "arpu_segment"])["n_hist"].transform("sum")
        conv = (g["n_hist"] / total * env.channels[BASE_CHANNEL]["conversion_multiplier"]).clip(upper=1.0)
        g["m0"] = g["pct_mean"] * conv
        pct_sd = g["pct_sd"].fillna(g["pct_sd"].median())
        g["se_hist"] = conv * pct_sd / np.sqrt(g["n_hist"])
        return g.rename(columns={"tariff_plan_code_from": "current_tariff",
                                 "tariff_plan_code_to": "target_tariff"})[
            ["current_tariff", "arpu_segment", "target_tariff", "m0", "se_hist", "n_hist"]]

    # -------------------------------------------------------------- candidates
    def _candidates(self, env, prior):
        profile = env.customer_profile
        cells = (profile.groupby(["current_tariff", "arpu_segment"])
                 .agg(n_cell=("ID_NUMBER", "size"), arpu_mean=("predicted_arpu", "mean"))
                 .reset_index())
        cells["n_use"] = cells["n_cell"].clip(upper=MAX_CUSTOMERS_PER_CAMPAIGN)
        cells["w"] = cells["arpu_mean"] * cells["n_use"]      # ценность единицы ratio в ячейке

        known = set(env.tariffs["tariff_plan_code"])
        cand = cells.merge(prior, on=["current_tariff", "arpu_segment"], how="inner")
        cand = cand[(cand["target_tariff"] != cand["current_tariff"]) & cand["target_tariff"].isin(known)]
        a, b = self.residual
        cand["tau"] = np.sqrt(cand["se_hist"] ** 2 + (a + b * cand["m0"].abs()) ** 2)
        cand["opt"] = (cand["m0"] + cand["tau"]) * cand["w"]
        cand = (cand.sort_values(["current_tariff", "arpu_segment", "opt", "target_tariff"],
                                 ascending=[True, True, False, True])
                .groupby(["current_tariff", "arpu_segment"]).head(self.max_targets_per_cell)
                .reset_index(drop=True))
        cand["cell"] = cand["current_tariff"] + "|" + cand["arpu_segment"]
        cand["n_obs"] = 0
        return cand

    def _init_belief(self, cand, scale=1.0):
        x = np.column_stack([np.ones(len(cand)), cand["m0"].to_numpy()])
        sb = np.diag(np.square(self.beta_prior_sd))
        mu = x @ np.array([0.0, 1.0])
        cov = x @ sb @ x.T + np.diag((scale * cand["tau"].to_numpy()) ** 2)
        return mu, cov

    def _posterior(self, cand, obs):
        """Апостериорная оценка по всем пилотам. Масштаб индивидуальных отклонений связок
        выбирается по правдоподобию пилотов (эмпирический Байес): если история плохо
        предсказывает пилоты, неопределённость непроверенных связок растёт."""
        if not obs:
            mu, cov = self._init_belief(cand, 1.0)
            return mu, cov, 1.0
        idx = np.array([o[0] for o in obs])
        y = np.array([o[1] for o in obs])
        r = np.diag([PER_CUSTOMER_STD ** 2 / o[2] for o in obs])
        best = None
        for s in self.scale_grid:
            mu0, cov0 = self._init_belief(cand, s)
            k_obs = cov0[np.ix_(idx, idx)] + r
            resid = y - mu0[idx]
            sign, logdet = np.linalg.slogdet(k_obs)
            ll = -0.5 * (logdet + resid @ np.linalg.solve(k_obs, resid))
            if best is None or ll > best[0] + 1e-9:
                best = (ll, s, mu0, cov0, k_obs, resid)
        _, s, mu0, cov0, k_obs, resid = best
        gain = np.linalg.solve(k_obs, cov0[idx, :]).T              # (C, m)
        mu = mu0 + gain @ resid
        cov = cov0 - gain @ cov0[idx, :]
        cov = 0.5 * (cov + cov.T)
        return mu, cov, s

    # ------------------------------------------------------------ KG machinery
    def _value(self, means, w, cost_total, starts):
        """Ценность итогового плана при данных средних: по ячейке лучшая связка или ничего."""
        v = means * w - cost_total
        best = np.maximum.reduceat(v, starts, axis=-1)
        return np.clip(best, 0.0, None).sum(axis=-1)

    def _kg_scores(self, mu, cov, n_pilot, w, cost_total, starts, arpu_mean, pilot_cost):
        noise = PER_CUSTOMER_STD ** 2 / n_pilot                       # (C,)
        sigma_tilde = cov / np.sqrt(np.diag(cov) + noise)[None, :]     # столбец i = сдвиг средних от пилота i
        base = self._value(mu, w, cost_total, starts)
        # means[i, k, :] = mu + sigma_tilde[:, i] * z_k
        means = mu[None, None, :] + sigma_tilde.T[:, None, :] * _GH_X[None, :, None]
        exp_value = (self._value(means, w, cost_total, starts) * _GH_W[None, :]).sum(axis=1)
        kg = exp_value - base
        immediate = mu * arpu_mean * n_pilot - pilot_cost              # эффект самих пилотных контактов
        return kg, immediate

    # ----------------------------------------------------------------- pilots
    def _should_stop(self, kg, immediate, pilot_cost):
        """Близорукий KG недооценивает серию пилотов, поэтому останавливаемся, только если
        пилот не несёт информации и сам по себе ожидаемо убыточен."""
        if self.stop_rule == "kg":
            return kg <= pilot_cost
        return kg + immediate <= 0 or kg <= self.min_kg

    def _verify_target(self, cand, mu, w, cost_total):
        """Крупнейшая непроверенная ставка плана: лучшая по средней оценке связка ячейки
        с положительной ожидаемой ценностью, которую ещё не проверяли пилотом."""
        value = mu * w - cost_total
        df = pd.DataFrame({"cell": cand["cell"].to_numpy(), "value": value,
                           "n_obs": cand["n_obs"].to_numpy()})
        best = df.sort_values(["cell", "value"], ascending=[True, False], kind="mergesort").groupby("cell").head(1)
        best = best[(best["value"] > 0) & (best["n_obs"] == 0)]
        if best.empty:
            return None
        return int(best["value"].idxmax())

    def _explore(self, env, cand):
        cost = env.channels[BASE_CHANNEL]["cost_per_contact"]
        w = cand["w"].to_numpy()
        cost_total = cand["n_use"].to_numpy() * cost
        starts = np.flatnonzero(np.r_[True, cand["cell"].to_numpy()[1:] != cand["cell"].to_numpy()[:-1]])
        arpu_mean = cand["arpu_mean"].to_numpy()
        n_pilot = np.clip(np.minimum(self.pilot_size, cand["n_cell"].to_numpy()), MIN_PILOT, MAX_PILOT).astype(float)
        obs = []
        mu, cov, self.scale_ = self._posterior(cand, obs)
        mode = "kg"

        while env.pilots_left > 0:
            affordable = (n_pilot * cost <= env.remaining_budget) & (n_pilot <= env.remaining_contacts)
            i, kg_i = None, 0.0
            if mode == "kg":
                kg, immediate = self._kg_scores(mu, cov, n_pilot, w, cost_total, starts, arpu_mean, n_pilot * cost)
                score = np.where(affordable, kg + immediate, -np.inf)
                j = int(np.argmax(score))
                if np.isfinite(score[j]) and not self._should_stop(kg[j], immediate[j], n_pilot[j] * cost):
                    i, kg_i, why = j, float(kg[j]), f"ожидаемая польза для плана {kg[j]:,.0f}"
                elif self.verify:
                    mode = "verify"
                    self._log("pilot", "Поиск по Knowledge Gradient исчерпан: оставшиеся пилоты — на проверку "
                                       "крупнейших непроверенных ставок плана.")
                else:
                    self._log("pilot", "Разведка остановлена: ни один пилот не повышает ожидаемую ценность плана.")
                    break
            if mode == "verify":
                j = self._verify_target(cand, np.where(affordable, mu, -np.inf), w, cost_total)
                if j is None:
                    self._log("pilot", "Все крупные ставки плана проверены — разведка завершена.")
                    break
                i, why = j, "проверка ставки плана, оценённой только через калибровку"
            row = cand.iloc[i]
            try:
                res = env.run_pilot(target_tariff=row["target_tariff"], channel=BASE_CHANNEL,
                                    n_customers=int(n_pilot[i]), filter_arpu_segment=row["arpu_segment"],
                                    filter_current_tariff=row["current_tariff"])
            except (RuntimeError, ValueError) as e:
                self._log("pilot", f"Пилот не запущен: {e}")
                break
            n_act = int(res["n_customers"])
            y = float(res["observed_lift_ratio"])
            prior_i, prior_sd_i = float(mu[i]), float(np.sqrt(cov[i, i]))
            obs.append((i, y, n_act))
            cand.iloc[i, cand.columns.get_loc("n_obs")] += n_act
            mu, cov, self.scale_ = self._posterior(cand, obs)
            post_sd = float(np.sqrt(max(cov[i, i], 0.0)))
            self._log("pilot",
                      f"Пилот {row['current_tariff']}/{row['arpu_segment']} → {row['target_tariff']} "
                      f"(n={n_act}): наблюдали {y:+.3f}, до пилота {prior_i:+.3f}±{prior_sd_i:.3f}, "
                      f"после {mu[i]:+.3f}±{post_sd:.3f}; {why}. Доверие к истории: ×{self.scale_:g} к разбросу.",
                      current_tariff=row["current_tariff"], arpu_segment=row["arpu_segment"],
                      target_tariff=row["target_tariff"], channel=BASE_CHANNEL, n=n_act, observed=y,
                      post_mean=float(mu[i]), post_sd=post_sd)
        return cand, mu, cov

    # --------------------------------------------------------------- planning
    def _options(self, env, cand, mu, sd):
        """Все допустимые варианты (связка, канал) с ожидаемой и нижней ценностью на абонента."""
        rows = []
        z = np.where(cand["n_obs"].to_numpy() > 0, self.z_safe, self.z_unpiloted)
        low = mu - z * sd
        for ch in FINAL_CHANNELS:
            s = self._scale(env, ch)
            c = env.channels[ch]["cost_per_contact"]
            a = cand["arpu_mean"].to_numpy()
            rows.append(cand.assign(channel=ch, mean=mu, sd=sd, low=low,
                                    v_mean=s * mu * a - c, v_low=s * low * a - c, unit_cost=c))
        opt = pd.concat(rows, ignore_index=True)
        return opt[opt["v_low"] > 0]

    def _plan(self, env, cand, mu, cov):
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        opts = self._options(env, cand, mu, sd)
        contacts_left = int(env.remaining_contacts)
        money_left = float(env.remaining_budget)

        # 1) по ячейке: лучшая связка по нижней ценности через дешёвые каналы (push/sms)
        cheap = opts[opts["channel"].isin(["push", "sms"])]
        pick = (cheap.sort_values(["cell", "v_low", "v_mean", "target_tariff"], ascending=[True, False, False, True])
                .groupby("cell").head(1))
        pick = pick.sort_values(["v_mean", "cell"], ascending=[False, True], kind="mergesort")

        chosen = []
        for _, r in pick.iterrows():
            take = int(min(r["n_use"], contacts_left))
            if r["unit_cost"] > 0:
                take = int(min(take, money_left // r["unit_cost"]))
            if take <= 0:
                continue
            chosen.append(dict(r, take=take))
            contacts_left -= take
            money_left -= take * r["unit_cost"]

        # 2) апгрейд канала на digital_ads там, где дополнительный эффект окупает доплату
        ads = opts[opts["channel"] == "digital_ads"].set_index(["cell", "target_tariff"])
        upgrades = []
        for j, r in enumerate(chosen):
            key = (r["cell"], r["target_tariff"])
            if key not in ads.index:
                continue
            a = ads.loc[key]
            extra_low = (a["v_low"] - r["v_low"]) * r["take"]
            extra_cost = (a["unit_cost"] - r["unit_cost"]) * r["take"]
            if extra_low > 0 and extra_cost > 0:
                upgrades.append((extra_low / extra_cost, j, a, extra_cost))
        for _, j, a, extra_cost in sorted(upgrades, key=lambda t: -t[0]):
            if extra_cost <= money_left:
                chosen[j].update(channel="digital_ads", v_mean=a["v_mean"], v_low=a["v_low"],
                                 unit_cost=a["unit_cost"])
                money_left -= extra_cost

        # 3) журнал отказов по проверенным связкам
        chosen_keys = {(r["cell"], r["target_tariff"]) for r in chosen}
        tested = cand[cand["n_obs"] > 0]
        for i, r in tested.iterrows():
            if (r["cell"], r["target_tariff"]) in chosen_keys:
                continue
            self._log("decision", f"Не берём {r['current_tariff']}/{r['arpu_segment']} → {r['target_tariff']}: "
                                  f"оценка {mu[i]:+.3f}±{sd[i]:.3f} — нижняя граница не окупает контакт "
                                  f"или в ячейке есть связка лучше.",
                      current_tariff=r["current_tariff"], arpu_segment=r["arpu_segment"],
                      target_tariff=r["target_tariff"], post_mean=float(mu[i]), post_sd=float(sd[i]))

        # 4) упаковка в ≤10 кампаний: общий сегмент ARPU, целевой тариф и канал
        groups = {}
        for r in chosen:
            groups.setdefault((r["arpu_segment"], r["target_tariff"], r["channel"]), []).append(r)
        campaigns = []
        for key, items in groups.items():
            chunk, size = [], 0
            for r in items:
                if chunk and size + r["take"] > MAX_CUSTOMERS_PER_CAMPAIGN:
                    campaigns.append((key, chunk))
                    chunk, size = [], 0
                chunk.append(r)
                size += r["take"]
            if chunk:
                campaigns.append((key, chunk))
        campaigns.sort(key=lambda c: -sum(r["v_mean"] * r["take"] for r in c[1]))
        dropped = campaigns[MAX_CAMPAIGNS:]
        campaigns = campaigns[:MAX_CAMPAIGNS]
        if dropped:
            self._log("decision", f"Лимит 10 кампаний: отброшены {len(dropped)} наименее ценных групп.")

        result, rows = [], []
        for i, ((seg, target, channel), items) in enumerate(campaigns, 1):
            tariffs = ";".join(sorted(r["current_tariff"] for r in items))
            n_c = int(sum(r["take"] for r in items))
            exp_net = float(sum(r["v_mean"] * r["take"] for r in items))
            low_net = float(sum(r["v_low"] * r["take"] for r in items))
            name = f"c{i}_{seg}_{target}_{channel}"
            result.append({"campaign_name": name, "filter_arpu_segment": seg,
                           "filter_current_tariff": tariffs, "target_tariff": target, "channel": channel})
            reason = "; ".join(
                f"{r['current_tariff']}: эффект {r['mean']:+.3f}±{r['sd']:.3f} "
                f"({'пилоты ' + str(int(r['n_obs'])) + ' абон.' if r['n_obs'] > 0 else 'по калибровке пилотов'})"
                for r in items)
            rows.append({"campaign_name": name, "filters": f"arpu={seg}; tariffs={tariffs}",
                         "target_tariff": target, "channel": channel, "n_customers": n_c,
                         "expected_net": exp_net, "lower_bound": low_net, "reason": reason})
            self._log("decision", f"Кампания {name}: {n_c} абонентов, ожидаемо {exp_net:,.0f}, "
                                  f"нижняя оценка {low_net:,.0f}.",
                      arpu_segment=seg, target_tariff=target, channel=channel, n=n_c)

        if not result:
            result, rows = self._fallback(env, cand, mu)
        self.plan_table = pd.DataFrame(rows)
        return result

    def _fallback(self, env, cand, mu):
        """Нет доказанно выгодных связок: одна минимальная кампания с наименьшим ожидаемым ущербом."""
        order = np.argsort(-mu, kind="mergesort")
        r = cand.iloc[int(order[0])]
        channel = BASE_CHANNEL if mu[order[0]] > 0 else "push"
        profile = env.customer_profile
        seg = profile[(profile["current_tariff"] == r["current_tariff"]) & (profile["arpu_segment"] == r["arpu_segment"])]
        sizes = seg.groupby(["data_segment", "call_segment"]).size()
        data_seg, call_seg = sizes[sizes > 0].sort_values(kind="mergesort").index[0]
        n_c = int(sizes[(data_seg, call_seg)])
        self._log("decision", "Ни одна связка не прошла порог окупаемости: возвращаем одну минимальную кампанию "
                              "с наименьшим ожидаемым ущербом (требование 1–10 кампаний).",
                  current_tariff=r["current_tariff"], arpu_segment=r["arpu_segment"],
                  target_tariff=r["target_tariff"], channel=channel, n=n_c)
        camp = {"campaign_name": "fallback_min", "filter_arpu_segment": r["arpu_segment"],
                "filter_data_segment": data_seg, "filter_call_segment": call_seg,
                "filter_current_tariff": r["current_tariff"], "target_tariff": r["target_tariff"],
                "channel": channel}
        row = {"campaign_name": "fallback_min",
               "filters": f"arpu={r['arpu_segment']}; data={data_seg}; call={call_seg}; tariffs={r['current_tariff']}",
               "target_tariff": r["target_tariff"], "channel": channel, "n_customers": n_c,
               "expected_net": float("nan"), "lower_bound": float("nan"),
               "reason": "Резервная кампания: выгодных связок не найдено, прибыль не гарантирована."}
        return [camp], [row]

    def _no_history_prior(self, env):
        """История недоступна: одинаковая нейтральная априорная оценка для всех связок ячеек."""
        tariffs = list(env.tariffs["tariff_plan_code"])
        cells = env.customer_profile[["current_tariff", "arpu_segment"]].drop_duplicates()
        rows = [(c.current_tariff, c.arpu_segment, t, 0.0, 0.05, 0) for c in cells.itertuples()
                for t in tariffs if t != c.current_tariff]
        return pd.DataFrame(rows, columns=["current_tariff", "arpu_segment", "target_tariff", "m0", "se_hist", "n_hist"])

    # -------------------------------------------------------------------- act
    def act(self, env):
        self.trace, self.plan_table, self.explanation = [], None, None
        prior = self._prior(env)
        if prior is None or prior.empty:
            prior = self._no_history_prior(env)
            self._log("prior", "История недоступна: используем нейтральную априорную оценку.")
        cand = self._candidates(env, prior)
        self._log("prior", f"Априорная оценка по истории: {len(cand)} гипотез в {cand['cell'].nunique()} ячейках. "
                           f"Калибровка «история → эта аудитория» будет выучена по пилотам.")
        cand, mu, cov = self._explore(env, cand)
        campaigns = self._plan(env, cand, mu, cov)

        n_pilots = sum(1 for t in self.trace if t["phase"] == "pilot" and t["n"])
        n_cells = cand.loc[cand["n_obs"] > 0, "cell"].nunique()
        self.explanation = (
            f"Агент провёл {n_pilots} пилотов в {n_cells} ячейках, выбирая каждый следующий пилот по ожидаемой "
            f"пользе для итогового плана (Knowledge Gradient). По результатам пилотов он откалибровал историю "
            f"под эту аудиторию и отобрал {len(campaigns)} кампаний, у которых нижняя граница оценки окупает "
            f"контакт. Канал выбран под ценность абонента: digital_ads — там, где доплата окупается, "
            f"push — где эффект мал, иначе SMS.")
        return campaigns
