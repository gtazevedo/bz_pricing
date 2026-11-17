import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from bz_pricing.core.options_core import (
    black76_price_greeks as _cy_black76,
    bsm_price_greeks as _cy_bsm,
    crr_american_price as _cy_crr_american,
    implied_vol_black76 as _cy_iv_black76,
)
from bz_pricing.volatility.surface import monotone_cubic_interp


def _longest_monotonic_indices(values: Sequence[float], non_increasing: bool = True) -> List[int]:
    """Retorna os índices da maior subsequência monótona preservando a ordem original."""
    n = len(values)
    if n == 0:
        return []
    dp = [1] * n
    parent = [-1] * n
    best_len = 1
    best_idx = 0
    for i in range(n):
        for j in range(i):
            cond = (values[i] <= values[j] + 1e-12) if non_increasing else (values[i] >= values[j] - 1e-12)
            if cond and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
        if dp[i] >= best_len:
            best_len = dp[i]
            best_idx = i
    seq: List[int] = []
    cur = best_idx
    while cur != -1:
        seq.append(cur)
        cur = parent[cur]
    seq.reverse()
    return seq


def filter_option_quotes_bbg(
    df: pd.DataFrame,
    spot: Optional[float] = None,
    maturity_years: Optional[float] = None,
    max_age_days: int = 3,
    min_price: float = 0.001,
) -> pd.DataFrame:
    """
    Aplica a filtragem de cotações para extração de forwards e dividendos implícitos:
    remove preços defasados, ofertas abaixo do piso mínimo, spreads bid/ask aberrantes
    e pontos fora da maior subsequência monótona de calls e puts por strike.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()

    if "age_days" in work.columns:
        work = work[pd.to_numeric(work["age_days"], errors="coerce").fillna(0) <= int(max_age_days)]

    has_ba = ("bid" in work.columns) and ("ask" in work.columns)
    if has_ba:
        work["bid"] = pd.to_numeric(work["bid"], errors="coerce")
        work["ask"] = pd.to_numeric(work["ask"], errors="coerce")
        work = work[(work["bid"] >= min_price) & (work["ask"] >= work["bid"])]
        if not work.empty:
            positive_bid = work["bid"] > 0.0
            work = work[~positive_bid | (work["ask"] <= 3.0 * work["bid"])]
            if not work.empty:
                spreads = work["ask"] - work["bid"]
                mean_s = float(spreads.mean())
                std_s = float(spreads.std(ddof=0)) if len(spreads) > 1 else 0.0
                if std_s > 0.0:
                    work = work[spreads <= mean_s + 3.0 * std_s]
                if spot is not None and spot > 0.0 and maturity_years is not None:
                    cap_mult = 0.0009 if float(maturity_years) <= 1.0 else 0.021
                    work = work[(work["ask"] - work["bid"]) <= cap_mult * float(spot)]

    p_col = next((c for c in ("mid", "price", "RefPric", "LastPric") if c in work.columns), None)
    if p_col is not None:
        work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
        work = work[work[p_col] >= min_price]

    k_col = next((c for c in ("strike", "ExrcPric", "K") if c in work.columns), None)
    t_col = next((c for c in ("option_type", "type", "OptnTp") if c in work.columns), None)
    if k_col is None or t_col is None or p_col is None or work.empty:
        return work.reset_index(drop=True)

    kept_parts: List[pd.DataFrame] = []
    is_call = work[t_col].astype(str).str.lower().str.startswith("c")
    for call_flag in (True, False):
        sub = work[is_call if call_flag else (~is_call)].sort_values(k_col).reset_index(drop=True)
        if not sub.empty:
            keep_idx = _longest_monotonic_indices(sub[p_col].to_numpy(dtype=np.float64), non_increasing=call_flag)
            kept_parts.append(sub.iloc[keep_idx])

    if not kept_parts:
        return work.iloc[0:0].copy()
    return pd.concat(kept_parts, ignore_index=True).sort_values([k_col, t_col]).reset_index(drop=True)


def _select_straddling_indices(strikes: np.ndarray, fwd_est: float) -> List[int]:
    """Seleciona os dois strikes mais próximos que ladeiam o forward corrente."""
    n = len(strikes)
    if n <= 2:
        return list(range(n))
    left = [i for i in range(n) if strikes[i] <= fwd_est]
    right = [i for i in range(n) if strikes[i] >= fwd_est]
    if left and right:
        i_l = left[-1]
        i_r = right[0]
        if i_l == i_r:
            chosen = [i_l]
            if i_l > 0:
                chosen.insert(0, i_l - 1)
            elif i_l + 1 < n:
                chosen.append(i_l + 1)
            return sorted(set(chosen))
        return [i_l, i_r]
    order = np.argsort(np.abs(strikes - fwd_est))
    return sorted(order[:2].tolist())


def _implied_vol_american_crr(
    market_price: float,
    spot: float,
    strike: float,
    r_cont: float,
    q_cont: float,
    T: float,
    is_call: bool,
    crr_steps: int = 120,
) -> float:
    dt = max(1e-8, float(T)) / max(1, int(crr_steps))
    vol_floor = max(1e-3, abs(float(r_cont) - float(q_cont)) * math.sqrt(dt) * 1.25)
    grid = [vol_floor, max(vol_floor * 2.0, 0.02), 0.05, 0.15, 0.35, 0.75, 1.5, 3.0]
    grid = sorted(set(v for v in grid if v >= vol_floor))
    vals = [(v, float(_cy_crr_american(spot, strike, r_cont, q_cont, T, v, is_call, crr_steps))) for v in grid]
    vals = [(v, p) for v, p in vals if math.isfinite(p)]
    if not vals:
        return float("nan")

    v_lo, p_lo = vals[0]
    if market_price <= p_lo:
        return float(v_lo)
    for i in range(len(vals) - 1):
        v_a, p_a = vals[i]
        v_b, p_b = vals[i + 1]
        if p_a <= market_price <= p_b:
            try:
                return float(
                    brentq(
                        lambda v: float(_cy_crr_american(spot, strike, r_cont, q_cont, T, v, is_call, crr_steps)) - market_price,
                        v_a,
                        v_b,
                        xtol=1e-7,
                    )
                )
            except Exception:
                return float("nan")
    return float("nan")


def deamericanize_and_infer_forward_bbg(
    strikes: Sequence[float],
    call_prices: Sequence[float],
    put_prices: Sequence[float],
    spot: float,
    r_cont: float,
    T: float,
    exercise: str = "EUROPEAN",
    call_bids: Optional[Sequence[float]] = None,
    call_asks: Optional[Sequence[float]] = None,
    put_bids: Optional[Sequence[float]] = None,
    put_asks: Optional[Sequence[float]] = None,
    crr_steps: int = 120,
    tol: float = 1e-3,
    max_iter: int = 25,
) -> Dict[str, object]:
    """
    Infere o preço forward e o dividendo implícito via paridade put-call, aplicando
    desamericanização iterativa no modelo binomial para opções americanas até que o
    forward estimado permaneça entre os dois strikes selecionados com variação < 0,1%.
    """
    k_arr = np.asarray(strikes, dtype=np.float64)
    c_arr = np.asarray(call_prices, dtype=np.float64)
    p_arr = np.asarray(put_prices, dtype=np.float64)
    if len(k_arr) == 0 or len(k_arr) != len(c_arr) or len(k_arr) != len(p_arr):
        raise ValueError("strikes, call_prices e put_prices devem ter o mesmo comprimento não nulo.")

    order = np.argsort(k_arr)
    k_arr = k_arr[order]
    c_arr = c_arr[order]
    p_arr = p_arr[order]

    spot_f = float(spot)
    r_f = float(r_cont)
    t_f = max(1e-8, float(T))
    df_r = math.exp(-r_f * t_f)
    inv_df = 1.0 / df_r
    is_am = exercise.strip().upper().startswith("AM")

    fwd_cur = spot_f * math.exp(r_f * t_f)
    c_eu = c_arr.copy()
    p_eu = p_arr.copy()
    n_iter = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        sel = _select_straddling_indices(k_arr, fwd_cur)
        if is_am:
            q_est = r_f - math.log(max(fwd_cur, 1e-12) / spot_f) / t_f
            c_eu_work = c_eu.copy()
            p_eu_work = p_eu.copy()
            for idx in sel:
                k_i = float(k_arr[idx])
                sig_c = _implied_vol_american_crr(float(c_arr[idx]), spot_f, k_i, r_f, q_est, t_f, True, crr_steps=crr_steps)
                sig_p = _implied_vol_american_crr(float(p_arr[idx]), spot_f, k_i, r_f, q_est, t_f, False, crr_steps=crr_steps)
                if math.isfinite(sig_c):
                    c_eu_work[idx] = float(_cy_black76(fwd_cur, k_i, r_f, t_f, sig_c, True)[0])
                if math.isfinite(sig_p):
                    p_eu_work[idx] = float(_cy_black76(fwd_cur, k_i, r_f, t_f, sig_p, False)[0])
            c_eu = c_eu_work
            p_eu = p_eu_work

        fwds_pair = [float(k_arr[idx] + inv_df * (c_eu[idx] - p_eu[idx])) for idx in sel]
        fwd_new = float(np.mean(fwds_pair))
        rel_diff = abs(fwd_new - fwd_cur) / max(abs(fwd_cur), 1e-12)

        in_bracket = True
        if len(sel) == 2 and k_arr[0] <= fwd_new <= k_arr[-1]:
            in_bracket = bool(k_arr[sel[0]] <= fwd_new <= k_arr[sel[1]])

        fwd_cur = fwd_new
        if not is_am:
            if in_bracket or it >= 2:
                break
        elif in_bracket and rel_diff < float(tol):
            break

    d_yield = math.log((spot_f * inv_df) / max(fwd_cur, 1e-12)) / t_f
    d_cash = spot_f * inv_df - fwd_cur

    fwd_bid = fwd_cur
    fwd_ask = fwd_cur
    if call_bids is not None and call_asks is not None and put_bids is not None and put_asks is not None:
        cb = np.asarray(call_bids, dtype=np.float64)[order]
        ca = np.asarray(call_asks, dtype=np.float64)[order]
        pb = np.asarray(put_bids, dtype=np.float64)[order]
        pa = np.asarray(put_asks, dtype=np.float64)[order]
        sel = _select_straddling_indices(k_arr, fwd_cur)
        c_adj = c_eu - c_arr
        p_adj = p_eu - p_arr
        f_bids = [float(k_arr[i] + inv_df * ((cb[i] + c_adj[i]) - (pa[i] + p_adj[i]))) for i in sel]
        f_asks = [float(k_arr[i] + inv_df * ((ca[i] + c_adj[i]) - (pb[i] + p_adj[i]))) for i in sel]
        lo_b = max(f_bids)
        hi_a = min(f_asks)
        if lo_b <= hi_a:
            fwd_bid, fwd_ask = lo_b, hi_a
        else:
            nearest = sel[0]
            fwd_bid, fwd_ask = f_bids[0] if len(f_bids) > 0 else lo_b, f_asks[0] if len(f_asks) > 0 else hi_a
            if fwd_bid > fwd_ask:
                mid_f = 0.5 * (f_bids[0] + f_asks[0])
                fwd_bid, fwd_ask = mid_f, mid_f
                _ = nearest

    d_cash_bid = spot_f * inv_df - fwd_ask
    d_cash_ask = spot_f * inv_df - fwd_bid
    d_yield_bid = math.log((spot_f * inv_df) / max(fwd_ask, 1e-12)) / t_f
    d_yield_ask = math.log((spot_f * inv_df) / max(fwd_bid, 1e-12)) / t_f

    return {
        "forward": float(fwd_cur),
        "dividend_yield": float(d_yield),
        "cash_dividend": float(d_cash),
        "forward_bid": float(fwd_bid),
        "forward_ask": float(fwd_ask),
        "dividend_yield_bid": float(min(d_yield_bid, d_yield_ask)),
        "dividend_yield_ask": float(max(d_yield_bid, d_yield_ask)),
        "cash_dividend_bid": float(min(d_cash_bid, d_cash_ask)),
        "cash_dividend_ask": float(max(d_cash_bid, d_cash_ask)),
        "european_calls": c_eu,
        "european_puts": p_eu,
        "iterations": int(n_iter),
    }


class KalmanDividendFilter:
    """
    Filtro de Kalman para suavização intradiária de dividendos implícitos com piso
    de ganho de 0,03, aceleração para 0,5 em violações de bandas bid/ask e descarte após 7 dias.
    """

    def __init__(self, alpha_over_sigma: float = 1.0, min_gain: float = 0.03, violation_gain: float = 0.5, stale_days: float = 7.0):
        self.ratio = float(alpha_over_sigma)
        self.sigma = 1.0
        self.alpha = float(alpha_over_sigma)
        self.min_gain = float(min_gain)
        self.violation_gain = float(violation_gain)
        self.stale_days = float(stale_days)
        self.state: Optional[float] = None
        self.variance: Optional[float] = None
        self.last_dt_days: Optional[float] = None

    def reset(self) -> None:
        self.state = None
        self.variance = None
        self.last_dt_days = None

    def apply_ex_dividend(self, ex_dividend_amount: float) -> None:
        if self.state is not None:
            self.state -= float(ex_dividend_amount)

    def update(
        self,
        raw_dividend: float,
        d_bid: Optional[float] = None,
        d_ask: Optional[float] = None,
        dt_steps: float = 1.0,
        elapsed_days: Optional[float] = None,
    ) -> Dict[str, float]:
        if elapsed_days is not None and float(elapsed_days) > self.stale_days:
            self.reset()

        obs = float(raw_dividend)
        if self.state is None or self.variance is None:
            self.state = obs
            self.variance = self.alpha * self.alpha
            return {"dividend": float(self.state), "variance": float(self.variance), "gain": 1.0, "violation": False}

        p_minus = self.variance + (self.sigma * self.sigma) * max(1e-6, float(dt_steps))
        inside_bounds = True
        if d_bid is not None and d_ask is not None:
            inside_bounds = (float(d_bid) < self.state < float(d_ask))

        if inside_bounds:
            gain = max(self.min_gain, p_minus / (p_minus + self.alpha * self.alpha))
            violation = False
        else:
            gain = self.violation_gain
            violation = True

        self.state = self.state + gain * (obs - self.state)
        self.variance = (1.0 - gain) * p_minus
        return {
            "dividend": float(self.state),
            "variance": float(self.variance),
            "gain": float(gain),
            "violation": bool(violation),
        }


class DividendCurveBBG:
    """
    Construtor de curva de dividendos e forwards implícitos segundo o modelo da Bloomberg:
    suporta sazonalidade mensal/semanal, ajuste de spot em horário estendido e
    decomposição híbrida em dividendos em caixa, proporcionais e contínuo.
    """

    def __init__(self, is_index: bool = False):
        self.is_index = bool(is_index)
        self.n_bins = 52 if self.is_index else 12
        self.seasonality_weights = np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)

    def fit_seasonality(self, dates: Sequence, amounts: Sequence[float]) -> np.ndarray:
        """Calcula pesos sazonais médios por mês (ações) ou por semana do ano (índices)."""
        df = pd.DataFrame({"dt": pd.to_datetime(dates), "amt": np.asarray(amounts, dtype=np.float64)})
        df = df[df["amt"] > 0.0].copy()
        if df.empty:
            self.seasonality_weights = np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)
            return self.seasonality_weights

        df["year"] = df["dt"].dt.year
        if self.is_index:
            df["bin"] = np.minimum(52, np.maximum(1, ((df["dt"].dt.dayofyear - 1) // 7) + 1)) - 1
        else:
            df["bin"] = df["dt"].dt.month - 1

        yearly_vecs: List[np.ndarray] = []
        for _, grp in df.groupby("year"):
            tot = float(grp["amt"].sum())
            if tot <= 0.0:
                continue
            vec = np.zeros(self.n_bins, dtype=np.float64)
            for b_idx, b_val in grp.groupby("bin")["amt"].sum().items():
                vec[int(b_idx)] = float(b_val) / tot
            yearly_vecs.append(vec)

        if yearly_vecs:
            mean_vec = np.mean(yearly_vecs, axis=0)
            s_tot = float(np.sum(mean_vec))
            self.seasonality_weights = (mean_vec / s_tot) if s_tot > 0.0 else np.full(self.n_bins, 1.0 / self.n_bins)
        return self.seasonality_weights

    def interpolate_cash_dividend(
        self,
        t1: float,
        d1: float,
        t2: float,
        d2: float,
        t_eval: float,
        start_month_or_week: int = 0,
    ) -> float:
        """Aloca o acréscimo de dividendo em caixa entre t1 e t2 proporcionalmente aos pesos sazonais."""
        if t_eval <= t1:
            return float(d1)
        if t_eval >= t2:
            return float(d2)

        def _cum_season(t_a: float, t_b: float) -> float:
            steps = max(2, int(round((t_b - t_a) * self.n_bins)) * 2)
            grid = np.linspace(t_a, t_b, steps + 1)
            mids = 0.5 * (grid[:-1] + grid[1:])
            dt_step = (t_b - t_a) / steps
            bins = np.floor((start_month_or_week + mids * self.n_bins)).astype(int) % self.n_bins
            return float(np.sum(self.seasonality_weights[bins]) * dt_step)

        w_tot = _cum_season(t1, t2)
        if w_tot <= 1e-12:
            return float(d1 + (d2 - d1) * ((t_eval - t1) / (t2 - t1)))
        w_part = _cum_season(t1, t_eval)
        return float(d1 + (d2 - d1) * (w_part / w_tot))

    @staticmethod
    def extrapolate_long_forward_yield(
        times: Sequence[float],
        implied_forwards: Sequence[float],
        spot: float,
        r_cont: Union[float, Sequence[float]],
        t_eval: float,
    ) -> float:
        """Extrapolação de rendimento a termo a partir do par mais longo com separação >= 1 ano."""
        t_arr = np.asarray(times, dtype=np.float64)
        f_arr = np.asarray(implied_forwards, dtype=np.float64)
        order = np.argsort(t_arr)
        t_arr, f_arr = t_arr[order], f_arr[order]
        r_arr = np.full_like(t_arr, float(r_cont)) if np.isscalar(r_cont) else np.asarray(r_cont, dtype=np.float64)[order]

        i_end = len(t_arr) - 1
        i_start = 0
        for i in range(i_end - 1, -1, -1):
            if t_arr[i_end] - t_arr[i] >= 1.0 - 1e-9:
                i_start = i
                break

        dt_span = max(1e-8, float(t_arr[i_end] - t_arr[i_start]))
        y_i = math.log((float(spot) * math.exp(float(r_arr[i_start]) * t_arr[i_start])) / max(1e-12, float(f_arr[i_start]))) / max(1e-8, float(t_arr[i_start]))
        y_n = math.log((float(spot) * math.exp(float(r_arr[i_end]) * t_arr[i_end])) / max(1e-12, float(f_arr[i_end]))) / max(1e-8, float(t_arr[i_end]))
        fwd_yield = (y_n * t_arr[i_end] - y_i * t_arr[i_start]) / dt_span
        if float(t_eval) <= float(t_arr[i_end]):
            return float(y_n)
        cum = y_n * t_arr[i_end] + fwd_yield * (float(t_eval) - float(t_arr[i_end]))
        return float(cum / float(t_eval))

    @staticmethod
    def extended_trading_hours_spot(
        spot_ref: float,
        fut_ets: float,
        r_ets: float,
        fut_ref: float,
        r_ref: float,
        maturity_years: float,
    ) -> float:
        """Estima o preço à vista em horário estendido preservando a razão para o futuro descontado."""
        t = float(maturity_years)
        num = float(fut_ets) * math.exp(-float(r_ets) * t)
        den = max(1e-12, float(fut_ref) * math.exp(-float(r_ref) * t))
        return float(spot_ref) * (num / den)

    @staticmethod
    def decompose_blended_dividends(
        div_times: Sequence[float],
        spot_f0: float,
        fwd_T: float,
        r_cont: float,
        T: float,
        is_index: bool = False,
    ) -> Dict[str, object]:
        """
        Decompõe o dividendo implícito acumulado até T em dividendos em caixa d_i,
        dividendos proporcionais y_i e taxa contínua residual q conforme os pesos por horizonte.
        """
        t_arr = np.sort(np.asarray([float(x) for x in div_times if 0.0 < float(x) <= float(T)], dtype=np.float64))
        n = len(t_arr)
        f0 = float(spot_f0)
        fT = float(fwd_T)
        r = float(r_cont)
        t_mat = float(T)

        if n == 0 or is_index:
            q_val = (r - math.log(max(1e-12, fT) / f0) / t_mat)
            return {"q": float(q_val), "d_global": 0.0, "d_hat": 0.0, "times": t_arr, "d_cash": np.zeros(n), "y_prop": np.zeros(n)}

        w_d = np.zeros(n, dtype=np.float64)
        w_y = np.zeros(n, dtype=np.float64)
        w_q = np.zeros(n, dtype=np.float64)
        for i, ti in enumerate(t_arr):
            if ti <= 1.0:
                w_d[i] = 1.0
            elif ti <= 2.0:
                w_d[i] = 0.5
                w_y[i] = 0.5
            elif ti <= 10.0:
                w_y[i] = 0.5
                w_q[i] = 0.5
            else:
                w_q[i] = 1.0

        disc_factors = np.exp(r * (t_mat - t_arr))
        denom_1 = float(np.sum(disc_factors))
        d_hat = (f0 * math.exp(r * t_mat) - fT) / max(1e-12, denom_1)

        f0_rq_T = f0 * math.exp(r * t_mat) - d_hat * float(np.sum(w_q * disc_factors))
        f0_rq_T = max(1e-12, f0_rq_T)
        q_val = r - math.log(f0_rq_T / f0) / t_mat

        disc_rq = np.exp((r - q_val) * (t_mat - t_arr))
        denom_3 = float(np.sum(disc_rq))
        d_global = (f0 * math.exp((r - q_val) * t_mat) - fT) / max(1e-12, denom_3)

        d_cash = np.zeros(n, dtype=np.float64)
        y_prop = np.zeros(n, dtype=np.float64)
        f_prev = f0
        t_prev = 0.0
        for i in range(n):
            wy_wd = w_d[i] + w_y[i]
            d_cash[i] = d_global * (w_d[i] / wy_wd) if wy_wd > 0.0 else 0.0
            drift = f_prev * math.exp((r - q_val) * (t_arr[i] - t_prev))
            f_i = drift - d_global
            y_prop[i] = 1.0 - ((f_i + d_cash[i]) / max(1e-12, drift))
            f_prev = f_i
            t_prev = float(t_arr[i])

        return {
            "q": float(q_val),
            "d_global": float(d_global),
            "d_hat": float(d_hat),
            "times": t_arr,
            "d_cash": d_cash,
            "y_prop": y_prop,
            "w_d": w_d,
            "w_y": w_y,
            "w_q": w_q,
        }


def filter_surface_quotes_bbg(
    df: pd.DataFrame,
    forward_by_expiry: Dict[object, float],
    r_cont_by_expiry: Dict[object, float],
    du_by_expiry: Optional[Dict[object, int]] = None,
    t_by_expiry: Optional[Dict[object, float]] = None,
    expiry_col: str = "XprtnDt",
    min_price: float = 0.001,
    max_age_days: int = 3,
    use_bid_ask: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Executa os critérios de seleção bid/ask vs settlement e os sete filtros de limpeza
    de superfície da Bloomberg (prazos >= 1 dia, somente OTM, subsequência monótona,
    exclusão em dois estágios de desvios de IV e corte por liquidez mínima relativa).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()

    k_col = next((c for c in ("ExrcPric", "strike", "Strike", "K") if c in work.columns), None)
    tp_col = next((c for c in ("option_type", "type", "OptnTp") if c in work.columns), None)
    if k_col is None or tp_col is None:
        raise ValueError("DataFrame deve conter colunas de strike e option_type.")

    has_ba = ("bid" in work.columns and "ask" in work.columns) or ("PREOFC" in work.columns and "PREOFV" in work.columns)
    b_col = "bid" if "bid" in work.columns else ("PREOFC" if "PREOFC" in work.columns else None)
    a_col = "ask" if "ask" in work.columns else ("PREOFV" if "PREOFV" in work.columns else None)
    p_col = next((c for c in ("RefPric", "mid", "MidBidAsk", "price", "LastPric") if c in work.columns), None)

    if use_bid_ask is None:
        use_bid_ask = False
        if has_ba and b_col and a_col:
            good_mats = 0
            total_mats = work[expiry_col].nunique()
            for _, grp in work.groupby(expiry_col):
                vb = pd.to_numeric(grp[b_col], errors="coerce")
                va = pd.to_numeric(grp[a_col], errors="coerce")
                n_ok = int(((vb >= min_price) & (va >= vb)).sum())
                if n_ok >= 3 and (n_ok / max(1, len(grp))) >= 0.10:
                    good_mats += 1
            if good_mats >= 3 or (total_mats > 0 and (good_mats / total_mats) >= 0.10):
                use_bid_ask = True
    else:
        use_bid_ask = bool(use_bid_ask and has_ba and b_col and a_col)

    if "age_days" in work.columns:
        work = work[pd.to_numeric(work["age_days"], errors="coerce").fillna(0) <= int(max_age_days)]

    def _norm_exp(val) -> str:
        try:
            return pd.to_datetime(val).strftime("%Y-%m-%d")
        except Exception:
            return str(val)

    fwd_map = {_norm_exp(k): float(v) for k, v in forward_by_expiry.items()}
    r_map = {_norm_exp(k): float(v) for k, v in r_cont_by_expiry.items()}
    du_map = {_norm_exp(k): int(v) for k, v in du_by_expiry.items()} if du_by_expiry else {}
    t_map = {_norm_exp(k): float(v) for k, v in t_by_expiry.items()} if t_by_expiry else {}

    rows: List[pd.DataFrame] = []
    for exp_key, grp in work.groupby(expiry_col):
        ek = _norm_exp(exp_key)
        if ek not in fwd_map:
            continue
        fwd = fwd_map[ek]
        r_c = r_map.get(ek, 0.0)
        t_val = t_map[ek] if ek in t_map else (float(du_map[ek]) / 252.0 if ek in du_map else 0.0)
        if t_val * 252.0 < 1.0:
            continue

        sub = grp.copy()
        if use_bid_ask and b_col and a_col:
            vb = pd.to_numeric(sub[b_col], errors="coerce")
            va = pd.to_numeric(sub[a_col], errors="coerce")
            valid_ba = (vb >= min_price) & (va >= vb) & ((vb == 0.0) | (va <= 3.0 * vb))
            sub = sub[valid_ba].copy()
            sub["_price"] = 0.5 * (vb[valid_ba] + va[valid_ba])
        else:
            if p_col is None:
                continue
            sub["_price"] = pd.to_numeric(sub[p_col], errors="coerce")
            sub = sub[sub["_price"] >= min_price].copy()

        sub["_k"] = pd.to_numeric(sub[k_col], errors="coerce")
        sub["_is_call"] = sub[tp_col].astype(str).str.lower().str.startswith("c")
        # Seleciona somente opções fora do dinheiro (OTM)
        otm = (sub["_is_call"] & (sub["_k"] >= fwd)) | ((~sub["_is_call"]) & (sub["_k"] <= fwd))
        sub = sub[otm].copy()
        if sub.empty:
            continue

        # Maior subsequência monótona separadamente para calls (decrescente) e puts (crescente)
        mono_parts: List[pd.DataFrame] = []
        for ic in (True, False):
            part = sub[sub["_is_call"] == ic].sort_values("_k").reset_index(drop=True)
            if not part.empty:
                keep_idx = _longest_monotonic_indices(part["_price"].to_numpy(dtype=np.float64), non_increasing=ic)
                mono_parts.append(part.iloc[keep_idx])
        if not mono_parts:
            continue
        sub = pd.concat(mono_parts, ignore_index=True)

        ivs = np.array(
            [
                _cy_iv_black76(float(p), fwd, float(k), r_c, t_val, bool(ic))
                for p, k, ic in zip(sub["_price"], sub["_k"], sub["_is_call"])
            ],
            dtype=np.float64,
        )
        sub["_iv"] = ivs
        sub = sub[np.isfinite(sub["_iv"]) & (sub["_iv"] > 1e-4)].copy()
        if not sub.empty:
            sub["_T"] = t_val
            sub["_fwd"] = fwd
            sub["_r_cont"] = r_c
            rows.append(sub)

    if not rows:
        return pd.DataFrame()
    comb = pd.concat(rows, ignore_index=True)
    global_med = float(comb["_iv"].median())
    comb = comb[comb["_iv"].between(global_med / 15.0, global_med * 15.0)].copy()

    # Filtro local por vencimento e remoção de vencimentos com menos de 20% da mediana de pontos
    refined: List[pd.DataFrame] = []
    counts: List[int] = []
    for _, grp in comb.groupby(expiry_col):
        med_m = float(grp["_iv"].median())
        g2 = grp[grp["_iv"].between(med_m * 0.2, med_m * 5.0)].copy()
        if not g2.empty:
            refined.append(g2)
            counts.append(len(g2))
    if not refined:
        return pd.DataFrame()
    med_count = float(np.median(counts))
    kept = [g for g in refined if len(g) >= 0.20 * med_count]
    if not kept:
        return pd.DataFrame()
    return pd.concat(kept, ignore_index=True)


class LognormalMixtureSlice:
    """
    Fatia temporal do modelo de mistura de distribuições lognormais com estado de default
    (Bloomberg Equity Implied Volatility Surface):

        pdf(T, S) = sum_{l=1}^N p_l(T) * lognormalpdf(S; xi_l(T)*F(T), Sigma_l(T)),
        Q(T) + sum_{l=1}^N p_l(T) = 1,   sum_{l=1}^N p_l(T)*xi_l(T) = 1.
    """

    def __init__(
        self,
        weights: Sequence[float],
        xis: Sequence[float],
        sigmas_total: Sequence[float],
        forward: float,
        T: float,
        k_min_damp: Optional[float] = None,
    ):
        self.p = np.asarray(weights, dtype=np.float64)
        self.xi = np.asarray(xis, dtype=np.float64)
        self.Sigma = np.asarray(sigmas_total, dtype=np.float64)
        self.F = float(forward)
        self.T = max(1e-8, float(T))
        self.N = len(self.p)
        if len(self.xi) != self.N or len(self.Sigma) != self.N or self.N < 2:
            raise ValueError("weights, xis e sigmas_total devem possuir o mesmo tamanho (>= 2).")

        self.P_survival = float(np.sum(self.p))
        self.Q_default = max(0.0, 1.0 - self.P_survival)
        self.lambda_hazard = -math.log(max(1e-12, self.P_survival)) / self.T if self.Q_default > 1e-12 else 0.0
        self.etas = np.log(np.maximum(1e-12, self.xi[1:] / self.xi[:-1]))
        self.k_min_damp = float(k_min_damp) if k_min_damp is not None else 0.9 * self.F

    @staticmethod
    def _reconstruct_xis(weights: np.ndarray, etas: np.ndarray) -> np.ndarray:
        p_sum = float(np.sum(weights))
        if p_sum <= 1e-12:
            return np.ones_like(weights)
        rel = np.ones_like(weights)
        rel[1:] = np.exp(np.cumsum(etas))
        denom = float(np.dot(weights, rel))
        xi1 = 1.0 / max(1e-12, denom)
        return xi1 * rel

    def damped_parameters(self, k: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Aplica amortecimento da intensidade de default para strikes na asa esquerda."""
        if self.Q_default <= 1e-12 or k >= self.k_min_damp:
            return self.p, self.xi, self.Q_default
        x = math.log(max(1e-12, float(k)) / self.F)
        x_m = math.log(max(1e-12, self.k_min_damp) / self.F)
        lam_new = self.lambda_hazard * math.exp((x_m * x_m - x * x) / (2.0 * self.T))
        p_surv_new = math.exp(-lam_new * self.T)
        p_new = self.p * (p_surv_new / max(1e-12, self.P_survival))
        xi_new = self._reconstruct_xis(p_new, self.etas)
        return p_new, xi_new, max(0.0, 1.0 - p_surv_new)

    def pdf(self, S: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Densidade neutra ao risco da mistura de lognormais (parcela contínua em S > 0)."""
        s_arr = np.asarray(S, dtype=np.float64)
        out = np.zeros_like(s_arr)
        pos = s_arr > 0.0
        if np.any(pos):
            sv = s_arr[pos]
            for pl, xil, sigl in zip(self.p, self.xi, self.Sigma):
                mu = math.log(max(1e-12, xil * self.F)) - 0.5 * sigl * sigl
                z = (np.log(sv) - mu) / max(1e-12, sigl)
                out[pos] += pl * np.exp(-0.5 * z * z) / (sv * sigl * math.sqrt(2.0 * math.pi))
        return float(out) if np.isscalar(S) else out

    def option_price(
        self,
        strike: float,
        r_cont: float = 0.0,
        is_call: bool = True,
        apply_hazard_damping: bool = True,
        beta_sigma: float = 1.0,
    ) -> float:
        """Preço europeu exato da mistura de lognormais acrescido do átomo de default nas puts."""
        k_f = float(strike)
        r_f = float(r_cont)
        p_eff, xi_eff, q_eff = self.damped_parameters(k_f) if apply_hazard_damping else (self.p, self.xi, self.Q_default)

        sqrt_t = math.sqrt(self.T)
        total = 0.0
        for pl, xil, sigl in zip(p_eff, xi_eff, self.Sigma):
            vol_l = max(1e-8, float(beta_sigma) * float(sigl) / sqrt_t)
            total += float(pl) * float(_cy_black76(float(xil) * self.F, k_f, r_f, self.T, vol_l, is_call)[0])

        if not is_call and q_eff > 0.0:
            total += math.exp(-r_f * self.T) * q_eff * k_f
        return float(total)

    def _raw_implied_vol(self, strike: float, r_cont: float = 0.0, beta_sigma: float = 1.0) -> float:
        k_f = float(strike)
        is_call = k_f >= self.F
        price = self.option_price(k_f, r_cont=r_cont, is_call=is_call, apply_hazard_damping=True, beta_sigma=beta_sigma)
        iv = float(_cy_iv_black76(price, self.F, k_f, float(r_cont), self.T, is_call))
        if not math.isfinite(iv) or iv <= 1e-6:
            return float(np.average(self.Sigma / math.sqrt(self.T), weights=np.maximum(self.p, 1e-8)))
        return iv

    def implied_vol(
        self,
        strikes: Union[float, Sequence[float], np.ndarray],
        r_cont: float = 0.0,
        beta_sigma: float = 1.0,
        use_wing_extrap: bool = True,
    ) -> Union[float, np.ndarray]:
        """Volatilidade implícita da mistura com extrapolação de Roger Lee fora de [k_min, k_max]."""
        scalar = np.isscalar(strikes)
        k_arr = np.atleast_1d(np.asarray(strikes, dtype=np.float64))

        if not use_wing_extrap:
            res = np.array([self._raw_implied_vol(float(k), r_cont=r_cont, beta_sigma=beta_sigma) for k in k_arr], dtype=np.float64)
            return float(res[0]) if scalar else res

        sig_atm = self._raw_implied_vol(self.F, r_cont=r_cont, beta_sigma=beta_sigma)
        v_est = max(0.05, 3.0 * sig_atm)
        sqrt_t = math.sqrt(self.T)
        k_min = self.F * math.exp(-4.0 * v_est * sqrt_t - 0.5 * v_est * v_est * self.T)
        k_max = self.F * math.exp(5.0 * v_est * sqrt_t + 0.5 * v_est * v_est * self.T)

        x_min = math.log(k_min / self.F)
        x_max = math.log(k_max / self.F)
        v_min_iv = self._raw_implied_vol(k_min, r_cont=r_cont, beta_sigma=beta_sigma)
        v_max_iv = self._raw_implied_vol(k_max, r_cont=r_cont, beta_sigma=beta_sigma)
        w_min = v_min_iv * v_min_iv * self.T
        w_max = v_max_iv * v_max_iv * self.T

        eps = 1e-3
        v_min_in = self._raw_implied_vol(k_min * math.exp(eps), r_cont=r_cont, beta_sigma=beta_sigma)
        v_max_in = self._raw_implied_vol(k_max * math.exp(-eps), r_cont=r_cont, beta_sigma=beta_sigma)
        slope_left = min(1.99, max(0.0, (v_min_iv * v_min_iv * self.T - v_min_in * v_min_in * self.T) / eps))
        slope_right = min(1.99, max(0.0, (v_max_iv * v_max_iv * self.T - v_max_in * v_max_in * self.T) / eps))

        out = np.empty(len(k_arr), dtype=np.float64)
        for i, k in enumerate(k_arr):
            kf = max(1e-12, float(k))
            if kf < k_min:
                dx = x_min - math.log(kf / self.F)
                w_ext = w_min + slope_left * dx
                out[i] = math.sqrt(max(1e-12, w_ext) / self.T)
            elif kf > k_max:
                dx = math.log(kf / self.F) - x_max
                w_ext = w_max + slope_right * dx
                out[i] = math.sqrt(max(1e-12, w_ext) / self.T)
            else:
                out[i] = self._raw_implied_vol(kf, r_cont=r_cont, beta_sigma=beta_sigma)

        return float(out[0]) if scalar else out

    @classmethod
    def fit(
        cls,
        strikes: Sequence[float],
        market_prices: Sequence[float],
        option_types: Sequence[str],
        forward: float,
        r_cont: float,
        T: float,
        n_components: int = 4,
        allow_default: bool = True,
        min_Sigma: Optional[Sequence[float]] = None,
        max_P_survival: float = 1.0,
    ) -> "LognormalMixtureSlice":
        """
        Calibra a mistura de lognormais com estado de default contra preços OTM de mercado
        por mínimos quadrados não lineares restritos com hierarquia anti-degenerescência.
        """
        k_np = np.asarray(strikes, dtype=np.float64)
        p_np = np.asarray(market_prices, dtype=np.float64)
        is_c = np.array([str(t).lower().startswith("c") for t in option_types], dtype=bool)
        fwd = float(forward)
        r_c = float(r_cont)
        t_val = max(1e-8, float(T))
        n = int(n_components)
        if n < 2:
            raise ValueError("n_components deve ser >= 2.")

        sqrt_t = math.sqrt(t_val)
        ivs = np.array([_cy_iv_black76(float(p), fwd, float(k), r_c, t_val, bool(ic)) for p, k, ic in zip(p_np, k_np, is_c)], dtype=np.float64)
        valid_ivs = ivs[np.isfinite(ivs) & (ivs > 1e-4)]
        sig_scale = float(np.median(valid_ivs)) * sqrt_t if len(valid_ivs) > 0 else 0.25 * sqrt_t

        # Ordena os parâmetros para eliminar degenerescência por permutação: xi_1 < ... < xi_N
        k_span = max(0.05, float(np.max(np.abs(np.log(k_np / fwd)))))
        eta0 = np.full(n - 1, max(0.03, (0.8 * k_span) / (n - 1)), dtype=np.float64)
        u0 = np.zeros(n - 1, dtype=np.float64)
        if n == 4:
            u0 = np.array([1.2, 1.2, 0.2], dtype=np.float64)  # pesos maiores no centro (2 e 3)
        sig0 = np.linspace(sig_scale * 1.4, sig_scale * 0.6, n)
        if min_Sigma is not None:
            sig0 = np.maximum(sig0, np.asarray(min_Sigma, dtype=np.float64) * 1.02)
        q0 = min(0.02, max(0.0, 1.0 - float(max_P_survival)))

        x0 = np.concatenate([u0, eta0, sig0, [q0]] if allow_default else [u0, eta0, sig0])
        lb_u = np.full(n - 1, -6.0)
        ub_u = np.full(n - 1, 6.0)
        lb_e = np.full(n - 1, 1e-3)
        ub_e = np.full(n - 1, 1.5)
        lb_s = np.asarray(min_Sigma, dtype=np.float64) if min_Sigma is not None else np.full(n, 1e-4)
        ub_s = np.full(n, max(3.0 * sqrt_t, 2.0))
        if allow_default:
            lb = np.concatenate([lb_u, lb_e, lb_s, [max(0.0, 1.0 - float(max_P_survival))]])
            ub = np.concatenate([ub_u, ub_e, ub_s, [0.35]])
        else:
            lb = np.concatenate([lb_u, lb_e, lb_s])
            ub = np.concatenate([ub_u, ub_e, ub_s])
        x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

        # Pesos balanceados pela escala dos prêmios para estabilidade numérica em strikes variados
        df_r = math.exp(-r_c * t_val)
        p_scale = max(1e-6, float(np.median(p_np)))

        def _unpack(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            u = vec[: n - 1]
            etas = vec[n - 1 : 2 * n - 2]
            sigs = vec[2 * n - 2 : 3 * n - 2]
            q_def = float(vec[3 * n - 2]) if allow_default else (1.0 - float(max_P_survival))
            logits = np.concatenate([[0.0], u])
            logits -= np.max(logits)
            exp_l = np.exp(logits)
            p_weights = (1.0 - q_def) * (exp_l / np.sum(exp_l))
            xis = cls._reconstruct_xis(p_weights, etas)
            return p_weights, xis, sigs

        def _res(vec: np.ndarray) -> np.ndarray:
            p_w, xis, sigs = _unpack(vec)
            pred = np.empty(len(k_np), dtype=np.float64)
            q_def = max(0.0, 1.0 - float(np.sum(p_w)))
            for idx_k, (kf, ic) in enumerate(zip(k_np, is_c)):
                tot = 0.0
                for pl, xil, sigl in zip(p_w, xis, sigs):
                    vol_l = sigl / sqrt_t
                    tot += pl * _cy_black76(xil * fwd, float(kf), r_c, t_val, vol_l, bool(ic))[0]
                if not ic and q_def > 0.0:
                    tot += df_r * q_def * float(kf)
                pred[idx_k] = tot
            return (pred - p_np) / p_scale

        sol = least_squares(_res, x0, bounds=(lb, ub), max_nfev=800)
        p_opt, xi_opt, sig_opt = _unpack(sol.x)
        k_damp = min(float(np.min(k_np)), 0.9 * fwd)
        return cls(p_opt, xi_opt, sig_opt, forward=fwd, T=t_val, k_min_damp=k_damp)


class BloombergMixtureSurface:
    """
    Superfície bidimensional baseada no modelo de mistura de lognormais com estado de
    default (Bloomberg): suporta interpolação Hermite monótona em variância total,
    extrapolação de curto e longo prazos (via superfície de referência ou nível ATM de Heston)
    e interpolação de spreads bid/ask.
    """

    def __init__(
        self,
        slices_by_du: Dict[int, LognormalMixtureSlice],
        heston_params: Optional[Tuple[float, float, float]] = None,
        reference_atm_fn=None,
    ):
        self.dus = sorted([int(d) for d in slices_by_du.keys()])
        self.slices: Dict[int, LognormalMixtureSlice] = {int(d): slices_by_du[d] for d in self.dus}
        self.times = np.array([d / 252.0 for d in self.dus], dtype=np.float64)
        self.heston_params = heston_params
        self.reference_atm_fn = reference_atm_fn
        if self.heston_params is None and len(self.dus) >= 2:
            self.heston_params = self._fit_heston_atm()

    def _fit_heston_atm(self) -> Tuple[float, float, float]:
        t_arr = self.times
        atm_vols = np.array([self.slices[d].implied_vol(self.slices[d].F) for d in self.dus], dtype=np.float64)
        v_target = atm_vols * atm_vols

        def _h_var(p: np.ndarray) -> np.ndarray:
            v0, th, kap = p
            kt = kap * t_arr
            factor = np.where(kt > 1e-6, (1.0 - np.exp(-kt)) / kt, 1.0 - 0.5 * kt)
            return np.sqrt(np.maximum(1e-8, th + (v0 - th) * factor)) - atm_vols

        x0 = np.array([float(v_target[0]), float(v_target[-1]), 1.5], dtype=np.float64)
        res = least_squares(_h_var, x0, bounds=([1e-5, 1e-5, 1e-3], [2.0, 2.0, 20.0]), max_nfev=300)
        return (float(res.x[0]), float(res.x[1]), float(res.x[2]))

    def heston_atm_vol(self, t_years: float) -> float:
        """Aproximação de Gatheral para o nível ATM a termo sob volatilidade estocástica de Heston."""
        if self.heston_params is None:
            last_sl = self.slices[self.dus[-1]]
            return float(last_sl.implied_vol(last_sl.F))
        v0, th, kap = self.heston_params
        t_eff = max(1e-6, float(t_years))
        kt = kap * t_eff
        factor = (1.0 - math.exp(-kt)) / kt if kt > 1e-6 else (1.0 - 0.5 * kt)
        return math.sqrt(max(1e-8, th + (v0 - th) * factor))

    def _short_term_extrap_slice(self, t_years: float, fwd_t: float, strike: float) -> float:
        du1 = self.dus[0]
        sl1 = self.slices[du1]
        T1 = sl1.T
        t = max(1e-6, float(t_years))
        x = math.log(max(1e-12, float(strike)) / float(fwd_t))

        if sl1.Q_default > 1e-12:
            if strike < 0.9 * fwd_t:
                k_eq = fwd_t * math.exp(x)
                _, _, q_new_t1 = sl1.damped_parameters(sl1.F * math.exp(x))
                lam_new = -math.log(max(1e-12, 1.0 - q_new_t1)) / T1
                lam_hat = lam_new * math.exp(0.5 * x * x * (1.0 / T1 - 1.0 / t))
                _ = k_eq
            else:
                lam_hat = sl1.lambda_hazard * (T1 / t)
            p_surv_t = math.exp(-lam_hat * t)
            p_t = sl1.p * (p_surv_t / max(1e-12, sl1.P_survival))
        else:
            p_t = sl1.p.copy()

        sig_t = sl1.Sigma * math.sqrt(t / T1)
        eta_t = sl1.etas * math.sqrt(t / T1)
        xi_t = LognormalMixtureSlice._reconstruct_xis(p_t, eta_t)
        sl_t = LognormalMixtureSlice(p_t, xi_t, sig_t, forward=float(fwd_t), T=t)
        return float(sl_t.implied_vol(strike, r_cont=0.0, use_wing_extrap=False))

    def _long_term_extrap_slice(self, t_years: float, fwd_t: float, strike: float) -> float:
        du_n = self.dus[-1]
        sl_n = self.slices[du_n]
        Tn = sl_n.T
        t = max(Tn + 1e-6, float(t_years))

        if sl_n.Q_default > 1e-12:
            p_surv_t = 0.5 * sl_n.P_survival + 0.5 * math.exp(-sl_n.lambda_hazard * t)
            p_t = sl_n.p * (p_surv_t / max(1e-12, sl_n.P_survival))
        else:
            p_t = sl_n.p.copy()

        sig_t = sl_n.Sigma * math.sqrt(t / Tn)
        eta_t = sl_n.etas * math.sqrt(t / Tn)
        xi_t = LognormalMixtureSlice._reconstruct_xis(p_t, eta_t)
        sl_t = LognormalMixtureSlice(p_t, xi_t, sig_t, forward=float(fwd_t), T=t)

        if self.reference_atm_fn is not None:
            mkt_atm_n = float(sl_n.implied_vol(sl_n.F))
            ref_atm_n = float(self.reference_atm_fn(Tn))
            target_atm = float(self.reference_atm_fn(t)) + (mkt_atm_n - ref_atm_n) * math.sqrt(Tn / t)
        else:
            base_atm = float(sl_t.implied_vol(fwd_t))
            h_atm = self.heston_atm_vol(t)
            target_atm = max(base_atm, h_atm)

        def _atm_err(beta: float) -> float:
            return float(sl_t._raw_implied_vol(fwd_t, r_cont=0.0, beta_sigma=beta)) - target_atm

        beta_opt = 1.0
        sl_final = sl_t
        if abs(_atm_err(1.0)) > 1e-5:
            try:
                beta_opt = float(brentq(_atm_err, 0.25, 4.0, xtol=1e-6))
            except Exception:
                # Fallback hierárquico: ajustar hazard rate ou reduzir distâncias eta_l entre médias
                matched = False
                if sl_t.P_survival < 1.0:
                    def _lam_err(lam_try: float) -> float:
                        p_surv_new = math.exp(-max(0.0, lam_try) * t)
                        p_try = sl_t.p * (p_surv_new / max(1e-12, sl_t.P_survival))
                        xi_try = LognormalMixtureSlice._reconstruct_xis(p_try, eta_t)
                        cand = LognormalMixtureSlice(p_try, xi_try, sig_t, forward=float(fwd_t), T=t)
                        return float(cand._raw_implied_vol(fwd_t, r_cont=0.0)) - target_atm
                    try:
                        lam_opt = float(brentq(_lam_err, 0.0, 2.0, xtol=1e-6))
                        p_surv_new = math.exp(-lam_opt * t)
                        p_try = sl_t.p * (p_surv_new / max(1e-12, sl_t.P_survival))
                        sl_final = LognormalMixtureSlice(
                            p_try, LognormalMixtureSlice._reconstruct_xis(p_try, eta_t), sig_t, forward=float(fwd_t), T=t
                        )
                        beta_opt = 1.0
                        matched = True
                    except Exception:
                        pass
                if not matched:
                    def _eta_err(scale_eta: float) -> float:
                        xi_try = LognormalMixtureSlice._reconstruct_xis(p_t, eta_t * scale_eta)
                        cand = LognormalMixtureSlice(p_t, xi_try, sig_t, forward=float(fwd_t), T=t)
                        return float(cand._raw_implied_vol(fwd_t, r_cont=0.0)) - target_atm
                    try:
                        s_eta = float(brentq(_eta_err, 0.0, 2.0, xtol=1e-6))
                        sl_final = LognormalMixtureSlice(
                            p_t, LognormalMixtureSlice._reconstruct_xis(p_t, eta_t * s_eta), sig_t, forward=float(fwd_t), T=t
                        )
                        beta_opt = 1.0
                    except Exception:
                        beta_opt = 1.0

        return float(sl_final.implied_vol(strike, r_cont=0.0, beta_sigma=beta_opt, use_wing_extrap=True))

    @staticmethod
    def apply_reference_atm_shift(
        market_vol_prev: float,
        ref_vol_prev: float,
        ref_vol_curr: float,
    ) -> float:
        """Desloca o nível de referência pelo diferencial observado no vencimento vizinho."""
        return float(ref_vol_curr + (market_vol_prev - ref_vol_prev))

    @staticmethod
    def extrapolate_long_skew_reference(
        t_last: float,
        t_eval: float,
        ref_vol_eval: float,
        delta_sigma_last: float,
    ) -> float:
        """Decai o ajuste da superfície de referência proporcionalmente a sqrt(T_n / t) em prazos longos."""
        scale = math.sqrt(max(1e-12, float(t_last)) / max(1e-12, float(t_eval)))
        return float(ref_vol_eval + delta_sigma_last * scale)

    @staticmethod
    def interpolate_vol_spread(
        dus: Sequence[int],
        strikes_by_du: Dict[int, Sequence[float]],
        spreads_by_du: Dict[int, Sequence[float]],
        du_eval: int,
        strike_eval: float,
    ) -> float:
        """Interpola spreads bid/ask de volatilidade (linear/flat em strike e Hermite no tempo)."""
        sorted_dus = sorted(int(d) for d in dus)
        vals = []
        for d in sorted_dus:
            ks = np.asarray(strikes_by_du[d], dtype=np.float64)
            sp = np.asarray(spreads_by_du[d], dtype=np.float64)
            ord_i = np.argsort(ks)
            vals.append(float(np.interp(float(strike_eval), ks[ord_i], sp[ord_i])))
        if du_eval <= sorted_dus[0]:
            return vals[0]
        if du_eval >= sorted_dus[-1]:
            return vals[-1]
        return float(monotone_cubic_interp(sorted_dus, vals, float(du_eval)))

    def get_vol(
        self,
        du: int,
        strike: float,
        forward: Optional[float] = None,
    ) -> float:
        """Avalia a volatilidade implícita com interpolação Hermite monótona e extrapolação temporal."""
        du_i = int(du)
        if du_i in self.slices:
            sl = self.slices[du_i]
            fwd_use = sl.F if forward is None else float(forward)
            k_scaled = float(strike) * (sl.F / max(1e-12, fwd_use))
            return float(sl.implied_vol(k_scaled))

        t_eval = max(1.0 / 252.0, du_i / 252.0)
        if du_i < self.dus[0]:
            fwd_use = self.slices[self.dus[0]].F if forward is None else float(forward)
            return self._short_term_extrap_slice(t_eval, fwd_use, float(strike))

        if du_i > self.dus[-1]:
            fwd_use = self.slices[self.dus[-1]].F if forward is None else float(forward)
            return self._long_term_extrap_slice(t_eval, fwd_use, float(strike))

        idx = int(np.searchsorted(self.dus, du_i, side="right") - 1)
        idx_lo = max(0, idx - 1)
        idx_hi = min(len(self.dus) - 1, idx + 2)
        sub_dus = self.dus[idx_lo : idx_hi + 1]
        fwd_eval = self.slices[self.dus[idx]].F if forward is None else float(forward)
        m_ratio = float(strike) / max(1e-12, fwd_eval)

        t_sub = np.array([d / 252.0 for d in sub_dus], dtype=np.float64)
        w_sub = np.empty(len(sub_dus), dtype=np.float64)
        for j, d_j in enumerate(sub_dus):
            sl_j = self.slices[d_j]
            v_j = float(sl_j.implied_vol(m_ratio * sl_j.F))
            w_sub[j] = v_j * v_j * sl_j.T

        if len(t_sub) >= 2:
            w_eval = monotone_cubic_interp(t_sub, w_sub, t_eval)
        else:
            w_eval = float(w_sub[0])
        return math.sqrt(max(1e-12, w_eval) / t_eval)

    @classmethod
    def fit(
        cls,
        slices_data: Dict[int, Dict[str, object]],
        n_components: int = 4,
        allow_default: bool = True,
        enforce_term_monotonicity: bool = True,
    ) -> "BloombergMixtureSurface":
        """Calibra a superfície multi-vencimento em cascata preservando monotonicidade temporal."""
        sorted_dus = sorted(slices_data.keys())
        fitted: Dict[int, LognormalMixtureSlice] = {}
        prev_sig = None
        prev_p_surv = 1.0
        for du_k in sorted_dus:
            d = slices_data[du_k]
            sl = LognormalMixtureSlice.fit(
                strikes=d["strikes"],
                market_prices=d["market_prices"],
                option_types=d["option_types"],
                forward=float(d["forward"]),
                r_cont=float(d.get("r_cont", 0.0)),
                T=du_k / 252.0,
                n_components=n_components,
                allow_default=allow_default,
                min_Sigma=prev_sig if enforce_term_monotonicity else None,
                max_P_survival=prev_p_surv if enforce_term_monotonicity else 1.0,
            )
            fitted[int(du_k)] = sl
            prev_sig = sl.Sigma
            prev_p_surv = sl.P_survival
        return cls(fitted)
