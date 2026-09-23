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

import os

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
                 z_safe=0.5, z_unpiloted=1.0, max_targets_per_cell=6,
                 beta_prior_sd=(0.03, 0.5), residual=(0.02, 0.3), stop_rule="full", min_kg=1.0,
                 scale_grid=None, verify=True, residual_feature="conv", push_leftover=True, planner="milp",
                 adaptive_risk=True, z_unpiloted_hard=1.5, risk_trigger=(1.5, 1.5)):
        self.history_path = history_path
        self.pilot_size = pilot_size
        self.z_safe = z_safe                    # осторожность для проверенных пилотом связок
        self.z_unpiloted = z_unpiloted          # осторожность для связок, оценённых только через калибровку
        self.max_targets_per_cell = max_targets_per_cell
        self.beta_prior_sd = beta_prior_sd      # априорная неопределённость сдвига и наклона калибровки
        self.residual = residual                # индивидуальное отклонение связки: a + b·|m0|
        self.stop_rule = stop_rule              # "full" — тратить пилоты, пока они информативны; "kg" — строгий KG
        self.min_kg = min_kg
        # кандидаты масштаба (постоянная часть, пропорциональная часть) для эмпирического Байеса;
        # первый элемент — априорный выбор до пилотов
        self.scale_grid = scale_grid or [(1.0, 1.0)] + [(sa, sb) for sa in (1.0, 2.0, 3.0, 5.0, 8.0)
                                                        for sb in (0.5, 1.0, 2.0, 4.0, 6.0) if (sa, sb) != (1.0, 1.0)]
        self.residual_feature = residual_feature  # от чего зависит разброс переноса: "conv" или "m0"
        self.adaptive_risk = adaptive_risk      # если история ненадёжна: без рискованного push, строже порог
        self.z_unpiloted_hard = z_unpiloted_hard
        self.risk_trigger = risk_trigger        # порог масштаба (постоянная, пропорциональная часть)
        self.planner = planner                  # "milp" — точный отбор целых ячеек; "greedy" — жадный
        self.push_leftover = push_leftover      # остаток контактов — в бесплатный push по средней оценке
        self.verify = verify                    # остаток пилотов — на проверку крупнейших ставок плана
        self.scale_ = (1.0, 1.0)
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
        hist = None
        here = os.path.dirname(os.path.abspath(__file__))
        for path in (os.path.join(here, self.history_path), self.history_path):
            try:
                hist = pd.read_csv(path)
                self.history_used_ = path
                break
            except (OSError, ValueError):
                continue
        if hist is None:
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
        g["conv"] = conv
        return g.rename(columns={"tariff_plan_code_from": "current_tariff",
                                 "tariff_plan_code_to": "target_tariff"})[
            ["current_tariff", "arpu_segment", "target_tariff", "m0", "se_hist", "n_hist", "conv"]]

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
        prop = cand["conv"] if self.residual_feature == "conv" else cand["m0"].abs()
        cand["tau"] = np.sqrt(cand["se_hist"] ** 2 + a ** 2 + (b * prop) ** 2)
        cand["opt"] = (cand["m0"] + cand["tau"]) * cand["w"]
        cand = (cand.sort_values(["current_tariff", "arpu_segment", "opt", "target_tariff"],
                                 ascending=[True, True, False, True])
                .groupby(["current_tariff", "arpu_segment"]).head(self.max_targets_per_cell)
                .reset_index(drop=True))
        cand["cell"] = cand["current_tariff"] + "|" + cand["arpu_segment"]
        cand["n_obs"] = 0
        return cand

    def _init_belief(self, cand, scale=(1.0, 1.0)):
        """Априорная совместная модель. scale = (множитель постоянной части отклонения,
        множитель части, пропорциональной |m0|)."""
        x = np.column_stack([np.ones(len(cand)), cand["m0"].to_numpy()])
        sb = np.diag(np.square(self.beta_prior_sd))
        mu = x @ np.array([0.0, 1.0])
        a, b = self.residual
        # эффект = изменение ARPU × конверсия, поэтому ошибка переноса пропорциональна конверсии
        prop = cand["conv"].to_numpy() if self.residual_feature == "conv" else cand["m0"].abs().to_numpy()
        tau2 = cand["se_hist"].to_numpy() ** 2 + (scale[0] * a) ** 2 + (scale[1] * b * prop) ** 2
        cov = x @ sb @ x.T + np.diag(tau2)
        return mu, cov

    def _posterior(self, cand, obs):
        """Апостериорная оценка по всем пилотам. Масштаб индивидуальных отклонений связок
        выбирается по правдоподобию пилотов (эмпирический Байес): если история плохо
        предсказывает пилоты, неопределённость непроверенных связок растёт."""
        if not obs:
            mu, cov = self._init_belief(cand, self.scale_grid[0])
            return mu, cov, self.scale_grid[0]
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
        return kg + min(immediate, 0.0) <= 0 or kg <= self.min_kg

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
                # выгода пилотных контактов почти всегда дублирует финальную кампанию, поэтому
                # учитываем только ожидаемый ущерб пилота, а не его «прибыль»
                score = np.where(affordable, kg + np.minimum(immediate, 0.0), -np.inf)
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
                      f"после {mu[i]:+.3f}±{post_sd:.3f}; {why}. Разброс «история → аудитория»: ×{self.scale_[0]:g} / ×{self.scale_[1]:g}.",
                      current_tariff=row["current_tariff"], arpu_segment=row["arpu_segment"],
                      target_tariff=row["target_tariff"], channel=BASE_CHANNEL, n=n_act, observed=y,
                      post_mean=float(mu[i]), post_sd=post_sd)
        return cand, mu, cov

    # --------------------------------------------------------------- planning
    def _history_unreliable(self):
        """Эмпирический Байес выбрал разброс «история → аудитория» шире априорного: калибровка по
        пилотам (выбранным как самые перспективные) завышает оценки непроверенных ячеек."""
        if not self.adaptive_risk:
            return False
        sa, sb = self.scale_
        return sa >= self.risk_trigger[0] or sb >= self.risk_trigger[1]

    def _z_unpiloted_now(self):
        return self.z_unpiloted_hard if self._history_unreliable() else self.z_unpiloted

    def _options(self, env, cand, mu, sd):
        """Все допустимые варианты (связка, канал) с ожидаемой и нижней ценностью на абонента."""
        rows = []
        z = np.where(cand["n_obs"].to_numpy() > 0, self.z_safe, self._z_unpiloted_now())
        low = mu - z * sd
        for ch in FINAL_CHANNELS:
            s = self._scale(env, ch)
            c = env.channels[ch]["cost_per_contact"]
            a = cand["arpu_mean"].to_numpy()
            rows.append(cand.assign(channel=ch, mean=mu, sd=sd, low=low,
                                    v_mean=s * mu * a - c, v_low=s * low * a - c, unit_cost=c))
        opt = pd.concat(rows, ignore_index=True)
        return opt[opt["v_low"] > 0]

    def _select_milp(self, opts, contacts_left, money_left, slots):
        """Точный выбор: по одной опции (связка, канал) на ячейку, ячейка берётся целиком.
        Ограничения: контакты, бюджет, ≤ slots кампаний (кампания = сегмент ARPU × тариф × канал,
        до 5 000 абонентов). Цель — сумма ожидаемой ценности. Возвращает строки opts."""
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix, hstack, vstack

        opts = opts[(opts["n_use"] <= contacts_left)
                    & (opts["unit_cost"] * opts["n_use"] <= money_left)].reset_index(drop=True)
        if opts.empty or slots <= 0:
            return opts.iloc[0:0]
        n_o = opts["n_use"].to_numpy(dtype=float)
        val = opts["objective"].to_numpy(dtype=float) * n_o
        cell_id = opts["cell"].astype("category").cat.codes.to_numpy().astype(np.int64)
        grp = (opts["arpu_segment"] + "|" + opts["target_tariff"] + "|" + opts["channel"]).astype("category")
        grp_id = grp.cat.codes.to_numpy().astype(np.int64)
        n_opt, n_cell, n_grp = len(opts), int(cell_id.max()) + 1, int(grp_id.max()) + 1
        rows = np.arange(n_opt)
        per_cell = hstack([csr_matrix((np.ones(n_opt), (cell_id, rows)), shape=(n_cell, n_opt)),
                           csr_matrix((n_cell, n_grp))])
        budget = hstack([csr_matrix(np.vstack([n_o, opts["unit_cost"].to_numpy() * n_o])),
                         csr_matrix((2, n_grp))])
        capacity = hstack([csr_matrix((n_o, (grp_id, rows)), shape=(n_grp, n_opt)),
                           csr_matrix(-MAX_CUSTOMERS_PER_CAMPAIGN * np.eye(n_grp))])
        campaigns = hstack([csr_matrix((1, n_opt)), csr_matrix(np.ones((1, n_grp)))])
        a = vstack([per_cell, budget, capacity, campaigns]).tocsr()
        ub = np.r_[np.ones(n_cell), contacts_left, money_left, np.zeros(n_grp), slots]
        cons = LinearConstraint(a, -np.inf, ub)
        c = np.r_[-val, np.zeros(n_grp)]
        bounds = Bounds(np.zeros(n_opt + n_grp), np.r_[np.ones(n_opt), np.full(n_grp, slots)])
        res = milp(c, constraints=cons, integrality=np.ones(n_opt + n_grp), bounds=bounds,
                   options={"time_limit": 60, "disp": False})
        if res.x is None:
            return None
        return opts[res.x[:n_opt] > 0.5]

    def _plan(self, env, cand, mu, cov):
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        opts = self._options(env, cand, mu, sd)
        contacts_left = int(env.remaining_contacts)
        money_left = float(env.remaining_budget)

        if self.planner == "milp":
            try:
                chosen = self._plan_milp(env, cand, mu, sd, opts, contacts_left, money_left)
            except Exception as e:                   # страховка: агент не должен падать из-за солвера
                chosen = None
                self._log("decision", f"Ошибка оптимизатора ({type(e).__name__}) — используем жадный отбор.")
            if chosen is not None:
                return self._pack_and_report(env, cand, mu, sd, chosen)
            if not any("Ошибка оптимизатора" in t["note"] for t in self.trace):
                self._log("decision", "Оптимизатор не нашёл решения — используем жадный отбор.")

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

        # 2б) остаток контактов — бесплатный push по ячейкам с положительной средней оценкой
        #     (как и в MILP-пути, отключается, если пилоты показали, что история ненадёжна)
        if self.push_leftover and contacts_left > 0 and not self._history_unreliable():
            s_push = self._scale(env, "push")
            taken_cells = {r["cell"] for r in chosen}
            rest = cand.assign(mean=mu, sd=sd)
            rest = rest[~rest["cell"].isin(taken_cells)]
            rest = rest.assign(v_mean=s_push * rest["mean"] * rest["arpu_mean"])
            rest = (rest[rest["v_mean"] > 0].sort_values(["cell", "v_mean"], ascending=[True, False])
                    .groupby("cell").head(1).sort_values("v_mean", ascending=False, kind="mergesort"))
            for _, r in rest.iterrows():
                take = int(min(r["n_use"], contacts_left))
                if take <= 0:
                    break
                z_r = self.z_safe if r["n_obs"] > 0 else self.z_unpiloted
                d = dict(r, take=take, channel="push", unit_cost=0.0, extra=True,
                         v_low=s_push * (r["mean"] - z_r * r["sd"]) * r["arpu_mean"],
                         low=r["mean"] - z_r * r["sd"])
                chosen.append(d)
                contacts_left -= take

        return self._pack_and_report(env, cand, mu, sd, chosen)

    def _plan_milp(self, env, cand, mu, sd, opts, contacts_left, money_left):
        """Два этапа: (1) связки, у которых нижняя граница окупает контакт; (2) остаток контактов
        и слотов — бесплатный push по положительной средней оценке."""
        opts = opts.assign(objective=opts["v_mean"])
        sel1 = self._select_milp(opts, contacts_left, money_left, MAX_CAMPAIGNS)
        if sel1 is None:
            return None
        chosen = [dict(r, take=int(r["n_use"])) for _, r in sel1.iterrows()]
        used_contacts = sum(r["take"] for r in chosen)
        used_money = sum(r["take"] * r["unit_cost"] for r in chosen)
        slots_used = len({(r["arpu_segment"], r["target_tariff"], r["channel"]) for r in chosen})
        if self.push_leftover and not self._history_unreliable():
            s_push = self._scale(env, "push")
            z = np.where(cand["n_obs"].to_numpy() > 0, self.z_safe, self._z_unpiloted_now())
            rest = cand.assign(mean=mu, sd=sd, low=mu - z * sd, channel="push", unit_cost=0.0)
            rest = rest.assign(v_mean=s_push * rest["mean"] * rest["arpu_mean"],
                               v_low=s_push * rest["low"] * rest["arpu_mean"])
            taken = {r["cell"] for r in chosen}
            rest = rest[(~rest["cell"].isin(taken)) & (rest["v_mean"] > 0)]
            # push-группы, уже открытые на этапе 1, не занимают новый слот: разрешаем их с запасом
            sel2 = self._select_milp(rest.assign(objective=rest["v_mean"]), contacts_left - used_contacts,
                                     money_left - used_money, MAX_CAMPAIGNS - slots_used)
            if sel2 is not None:
                chosen += [dict(r, take=int(r["n_use"]), extra=True) for _, r in sel2.iterrows()]
        return chosen

    def _pack_and_report(self, env, cand, mu, sd, chosen):
        self.n_extra_campaigns_ = 0
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
            bins = []                                   # first-fit decreasing: ≤ 5 000 абонентов в кампании
            for r in sorted(items, key=lambda r: (-r["take"], r["cell"])):
                for b in bins:
                    if b[0] + r["take"] <= MAX_CUSTOMERS_PER_CAMPAIGN:
                        b[0] += r["take"]
                        b[1].append(r)
                        break
                else:
                    bins.append([r["take"], [r]])
            campaigns.extend((key, b[1]) for b in bins)
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
                f"({'пилоты ' + str(int(r['n_obs'])) + ' абон.' if r['n_obs'] > 0 else 'по калибровке пилотов'}"
                f"{'; доп. бесплатный push по положительной средней оценке, нижняя граница не гарантирована' if r.get('extra') else ''})"
                for r in items)
            if any(r.get("extra") for r in items):
                self.n_extra_campaigns_ += 1
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
        rows = [(c.current_tariff, c.arpu_segment, t, 0.0, 0.05, 0, 0.1) for c in cells.itertuples()
                for t in tariffs if t != c.current_tariff]
        return pd.DataFrame(rows, columns=["current_tariff", "arpu_segment", "target_tariff", "m0", "se_hist",
                                           "n_hist", "conv"])

    def _plan_summary(self, campaigns):
        if any(t["phase"] == "decision" and "минимальную кампанию" in t["note"] for t in self.trace):
            return ("Ни одна связка не прошла порог окупаемости, поэтому возвращена одна минимальная "
                    "резервная кампания; прибыль не гарантирована.")
        extra = getattr(self, "n_extra_campaigns_", 0)
        main = len(campaigns) - extra
        text = f"В плане {len(campaigns)} кампаний: {main} — из связок, у которых нижняя граница оценки окупает контакт."
        if extra:
            text += (f" Ещё {extra} включают бесплатный push на остаток контактов по положительной средней оценке — "
                     f"для них нижняя граница не гарантирована (помечены в обосновании).")
        return text

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
            f"под эту аудиторию. {self._plan_summary(campaigns)} Канал выбран под ценность абонента: "
            f"digital_ads — там, где доплата окупается, push — где эффект мал, иначе SMS.")
        return campaigns
