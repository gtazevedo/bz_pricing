import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from bz_pricing.core.curves import cum_seasonality_exact as _cy_cum_season
from bz_pricing.core.options_core import (
    black76_price_greeks as _cy_black76,
    bloomberg_mixture_prices_and_jac as _cy_bbg_mix_prices_jac,
    bloomberg_mixture_prices_vec as _cy_bbg_mix_prices_vec,
    bsm_price_greeks as _cy_bsm,
    crr_american_discrete_div_price_greeks as _cy_crr_discrete,
    crr_american_price as _cy_crr_american,
    implied_vol_american_crr as _cy_iv_american_crr,
    implied_vol_black76 as _cy_iv_black76,
    longest_monotonic_subsequence as _cy_lms,
)
from bz_pricing.volatility.surface import monotone_cubic_interp


def _longest_monotonic_indices(values: Sequence[float], non_increasing: bool = True) -> List[int]:
    """Retorna os índices da maior subsequência monótona via núcleo Cython."""
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if len(arr) == 0:
        return []
    return _cy_lms(arr, bool(non_increasing)).tolist()


def _filter_quotes_single_expiry(
    df_sub: pd.DataFrame,
    spot: Optional[float],
    maturity_years: Optional[float],
    max_age_days: int,
    min_price: float,
    max_spread_short: float,
    max_spread_long: float,
) -> pd.DataFrame:
    work = df_sub.copy()
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
                    cap_mult = float(max_spread_short) if float(maturity_years) <= 1.0 else float(max_spread_long)
                    work = work[(work["ask"] - work["bid"]) <= cap_mult * float(spot)]

    p_col = next((c for c in ("mid", "price", "RefPric", "LastPric") if c in work.columns), None)
    if p_col is not None:
        work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
        work = work[work[p_col] >= min_price]

    k_col = next((c for c in ("strike", "ExrcPric", "K") if c in work.columns), None)
    t_col = next((c for c in ("option_type", "type", "OptnTp") if c in work.columns), None)
    if k_col is None or t_col is None or p_col is None or work.empty:
        return work

    kept_parts: List[pd.DataFrame] = []
    is_call = work[t_col].astype(str).str.lower().str.startswith("c")
    for call_flag in (True, False):
        sub = work[is_call if call_flag else (~is_call)].sort_values(k_col).reset_index(drop=True)
        if not sub.empty:
            keep_idx = _longest_monotonic_indices(sub[p_col].to_numpy(dtype=np.float64), non_increasing=call_flag)
            kept_parts.append(sub.iloc[keep_idx])

    if not kept_parts:
        return work.iloc[0:0].copy()
    return pd.concat(kept_parts, ignore_index=True)


def filter_option_quotes_bbg(
    df: pd.DataFrame,
    spot: Optional[float] = None,
    maturity_years: Optional[Union[float, Dict[object, float]]] = None,
    max_age_days: int = 3,
    min_price: float = 0.001,
    expiry_col: Optional[str] = None,
    max_spread_short: float = 0.0009,
    max_spread_long: float = 0.021,
) -> pd.DataFrame:
    """
    Aplica a filtragem de cotações por vencimento para extração de forwards e dividendos
    implícitos: remove preços defasados, ofertas abaixo do piso mínimo, spreads bid/ask
    aberrantes (razão > 3x, > média + 3 desvios ou acima do teto proporcional ao spot)
    e pontos fora da maior subsequência monótona de calls e puts por strike.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    exp_c = expiry_col if (expiry_col and expiry_col in df.columns) else next(
        (c for c in ("XprtnDt", "expiry", "maturity") if c in df.columns), None
    )
    k_col = next((c for c in ("strike", "ExrcPric", "K") if c in df.columns), None)
    t_col = next((c for c in ("option_type", "type", "OptnTp") if c in df.columns), None)

    if exp_c is None or df[exp_c].nunique() <= 1:
        t_use = float(next(iter(maturity_years.values()))) if isinstance(maturity_years, dict) and maturity_years else (
            None if isinstance(maturity_years, dict) else maturity_years
        )
        out = _filter_quotes_single_expiry(df, spot, t_use, max_age_days, min_price, max_spread_short, max_spread_long)
        if k_col and t_col and not out.empty:
            out = out.sort_values([k_col, t_col])
        return out.reset_index(drop=True)

    parts: List[pd.DataFrame] = []
    for exp_k, grp in df.groupby(exp_c, sort=False):
        if isinstance(maturity_years, dict):
            t_use = maturity_years.get(exp_k, maturity_years.get(str(exp_k), None))
        else:
            t_use = maturity_years
        part = _filter_quotes_single_expiry(grp, spot, t_use, max_age_days, min_price, max_spread_short, max_spread_long)
        if not part.empty:
            parts.append(part)
    if not parts:
        return df.iloc[0:0].copy()
    out = pd.concat(parts, ignore_index=True)
    if k_col and t_col:
        out = out.sort_values([exp_c, k_col, t_col])
    return out.reset_index(drop=True)


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


def _unpack_dividend_schedules(
    T: float,
    discrete_dividends: Optional[Union[Sequence[Tuple[float, float]], Dict[float, float]]],
    prop_dividends: Optional[Sequence[Tuple[float, float]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    div_pairs: List[Tuple[float, float]] = []
    if discrete_dividends is not None:
        items = discrete_dividends.items() if isinstance(discrete_dividends, dict) else discrete_dividends
        for t_d, a_d in items:
            tf, af = float(t_d), float(a_d)
            if 0.0 < tf <= T + 1e-12 and af > 0.0:
                div_pairs.append((tf, af))
    div_pairs.sort(key=lambda x: x[0])

    prop_pairs: List[Tuple[float, float]] = []
    if prop_dividends is not None:
        for t_p, y_p in prop_dividends:
            tf, yf = float(t_p), float(y_p)
            if 0.0 < tf <= T + 1e-12 and yf > 0.0:
                prop_pairs.append((tf, min(0.999, yf)))
    prop_pairs.sort(key=lambda x: x[0])

    dt_arr = np.array([x[0] for x in div_pairs], dtype=np.float64)
    da_arr = np.array([x[1] for x in div_pairs], dtype=np.float64)
    pt_arr = np.array([x[0] for x in prop_pairs], dtype=np.float64)
    py_arr = np.array([x[1] for x in prop_pairs], dtype=np.float64)
    return dt_arr, da_arr, pt_arr, py_arr


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
    discrete_dividends: Optional[Union[Sequence[Tuple[float, float]], Dict[float, float]]] = None,
    prop_dividends: Optional[Sequence[Tuple[float, float]]] = None,
    initial_forward: Optional[float] = None,
    deamericanize_all: bool = True,
    crr_steps: int = 120,
    tol: float = 1e-3,
    max_iter: int = 25,
) -> Dict[str, object]:
    """
    Infere o preço forward e o dividendo implícito via paridade put-call, aplicando
    desamericanização iterativa na árvore binomial Cython (com suporte a dividendos
    discretos em dinheiro e proporcionais) e convertendo toda a cadeia para equivalentes europeus.
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

    dt_arr, da_arr, pt_arr, py_arr = _unpack_dividend_schedules(t_f, discrete_dividends, prop_dividends)
    pv_div0 = float(np.sum(da_arr * np.exp(-r_f * dt_arr))) if len(dt_arr) > 0 else 0.0
    prop_mult = float(np.prod(1.0 - py_arr)) if len(py_arr) > 0 else 1.0
    s_adj = max(1e-12, (spot_f - pv_div0) * prop_mult)

    fwd_cur = float(initial_forward) if initial_forward is not None else (s_adj * math.exp(r_f * t_f))
    c_eu = c_arr.copy()
    p_eu = p_arr.copy()
    iv_c = np.full(len(k_arr), float("nan"), dtype=np.float64)
    iv_p = np.full(len(k_arr), float("nan"), dtype=np.float64)
    n_iter = 0
    dt_arg = dt_arr if len(dt_arr) > 0 else None
    da_arg = da_arr if len(da_arr) > 0 else None
    pt_arg = pt_arr if len(pt_arr) > 0 else None
    py_arg = py_arr if len(py_arr) > 0 else None

    def _deam_indices(idxs: Sequence[int], fwd_ref: float) -> None:
        q_est = r_f - math.log(max(fwd_ref, 1e-12) / s_adj) / t_f
        for idx in idxs:
            k_i = float(k_arr[idx])
            sig_c = float(
                _cy_iv_american_crr(
                    float(c_arr[idx]), spot_f, k_i, r_f, q_est, t_f, True, crr_steps,
                    dt_arg, da_arg, pt_arg, py_arg,
                )
            )
            sig_p = float(
                _cy_iv_american_crr(
                    float(p_arr[idx]), spot_f, k_i, r_f, q_est, t_f, False, crr_steps,
                    dt_arg, da_arg, pt_arg, py_arg,
                )
            )
            if math.isfinite(sig_c):
                iv_c[idx] = sig_c
                c_eu[idx] = float(_cy_black76(fwd_ref, k_i, r_f, t_f, sig_c, True)[0])
            if math.isfinite(sig_p):
                iv_p[idx] = sig_p
                p_eu[idx] = float(_cy_black76(fwd_ref, k_i, r_f, t_f, sig_p, False)[0])

    for it in range(1, max_iter + 1):
        n_iter = it
        sel = _select_straddling_indices(k_arr, fwd_cur)
        if is_am:
            _deam_indices(sel, fwd_cur)

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

    if is_am and deamericanize_all:
        _deam_indices(range(len(k_arr)), fwd_cur)
    else:
        for idx in range(len(k_arr)):
            iv_c[idx] = float(_cy_iv_black76(float(c_eu[idx]), fwd_cur, float(k_arr[idx]), r_f, t_f, True))
            iv_p[idx] = float(_cy_iv_black76(float(p_eu[idx]), fwd_cur, float(k_arr[idx]), r_f, t_f, False))

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
            nearest_pos = int(np.argmin(np.abs(k_arr[sel] - fwd_cur)))
            fwd_bid, fwd_ask = f_bids[nearest_pos], f_asks[nearest_pos]
            if fwd_bid > fwd_ask:
                mid_f = 0.5 * (fwd_bid + fwd_ask)
                fwd_bid, fwd_ask = mid_f, mid_f

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
        "implied_vols_call": iv_c,
        "implied_vols_put": iv_p,
        "iterations": int(n_iter),
    }


def reconcile_multi_maturity_dividends_bbg(
    maturities: Sequence[float],
    d_bids: Sequence[float],
    d_asks: Sequence[float],
    d_mids: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """
    Reconcilia intervalos bid/ask de dividendo oriundos de múltiplos vencimentos precedidos
    por uma mesma data ex-dividendo: adota o ponto médio da interseção comum ou, se vazia,
    o ponto médio do vencimento mais próximo atribuindo as diferenças residuais ao aluguel.
    """
    t_arr = np.asarray(maturities, dtype=np.float64)
    b_arr = np.asarray(d_bids, dtype=np.float64)
    a_arr = np.asarray(d_asks, dtype=np.float64)
    if len(t_arr) == 0 or len(t_arr) != len(b_arr) or len(t_arr) != len(a_arr):
        raise ValueError("maturities, d_bids e d_asks devem ter o mesmo comprimento não nulo.")

    order = np.argsort(t_arr)
    t_arr, b_arr, a_arr = t_arr[order], b_arr[order], a_arr[order]
    m_arr = np.asarray(d_mids, dtype=np.float64)[order] if d_mids is not None else 0.5 * (b_arr + a_arr)

    lo = float(np.max(b_arr))
    hi = float(np.min(a_arr))
    if lo <= hi:
        div_est = 0.5 * (lo + hi)
        consistent = True
        ref_mat = float(t_arr[0])
        b_out, a_out = lo, hi
    else:
        b_out, a_out = float(b_arr[0]), float(a_arr[0])
        div_est = 0.5 * (b_out + a_out)
        consistent = False
        ref_mat = float(t_arr[0])

    discrepancies = {float(t_v): float(m_v - div_est) for t_v, m_v in zip(t_arr, m_arr)}
    return {
        "dividend": float(div_est),
        "bid": float(b_out),
        "ask": float(a_out),
        "consistent": bool(consistent),
        "reference_maturity": float(ref_mat),
        "borrow_discrepancies": discrepancies,
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

    def reset(self) -> None:
        self.state = None
        self.variance = None

    def apply_ex_dividend(self, ex_dividend_amount: float) -> None:
        """Subtraí o dividendo ex-data em caixa do nível suavizado para ações."""
        if self.state is not None:
            self.state -= float(ex_dividend_amount)

    def apply_ex_dividend_index(self, spot: float, ex_dividend_points: float, maturity_years: float) -> None:
        """Converte o ajuste ex-dividendo agregado de constituintes em rendimento equivalente."""
        if self.state is not None and spot > 0.0 and maturity_years > 0.0:
            frac = min(0.999999, max(0.0, float(ex_dividend_points) / float(spot)))
            dy = -math.log(max(1e-12, 1.0 - frac)) / float(maturity_years)
            self.state -= dy

    def update(
        self,
        raw_dividend: Optional[float],
        d_bid: Optional[float] = None,
        d_ask: Optional[float] = None,
        dt_steps: float = 1.0,
        elapsed_days: Optional[float] = None,
    ) -> Dict[str, object]:
        if elapsed_days is not None and float(elapsed_days) > self.stale_days:
            self.reset()

        if raw_dividend is None or (isinstance(raw_dividend, float) and not math.isfinite(raw_dividend)):
            if self.state is None or self.variance is None:
                return {"dividend": float("nan"), "variance": float("nan"), "gain": 0.0, "violation": False, "missing": True}
            self.variance = self.variance + (self.sigma * self.sigma) * max(0.0, float(dt_steps))
            return {
                "dividend": float(self.state),
                "variance": float(self.variance),
                "gain": 0.0,
                "violation": False,
                "missing": True,
            }

        obs = float(raw_dividend)
        if self.state is None or self.variance is None:
            self.state = obs
            self.variance = self.alpha * self.alpha
            return {"dividend": float(self.state), "variance": float(self.variance), "gain": 1.0, "violation": False, "missing": False}

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
            "missing": False,
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

    def fit_seasonality(
        self,
        dates: Optional[Sequence] = None,
        amounts: Optional[Sequence[float]] = None,
        forecast_dates: Optional[Sequence] = None,
        forecast_amounts: Optional[Sequence[float]] = None,
        historical_dates: Optional[Sequence] = None,
        historical_amounts: Optional[Sequence[float]] = None,
        full_years_only: bool = False,
        sample_start: Optional[object] = None,
        sample_end: Optional[object] = None,
    ) -> np.ndarray:
        """
        Calcula pesos sazonais médios por mês (ações) ou por semana do ano (índices),
        suportando combinação de previsões com série histórica para cobrir ao menos 1 ano.
        """
        dt_parts: List[pd.Series] = []
        amt_parts: List[np.ndarray] = []

        if forecast_dates is not None and forecast_amounts is not None and len(forecast_dates) > 0:
            fd = pd.to_datetime(forecast_dates)
            fa = np.asarray(forecast_amounts, dtype=np.float64)
            dt_parts.append(pd.Series(fd))
            amt_parts.append(fa)
            span_days = float((fd.max() - fd.min()).days) if len(fd) > 0 else 0.0
            if span_days < 365.0 and historical_dates is not None and historical_amounts is not None and len(historical_dates) > 0:
                dt_parts.append(pd.Series(pd.to_datetime(historical_dates)))
                amt_parts.append(np.asarray(historical_amounts, dtype=np.float64))
        elif historical_dates is not None and historical_amounts is not None and len(historical_dates) > 0:
            dt_parts.append(pd.Series(pd.to_datetime(historical_dates)))
            amt_parts.append(np.asarray(historical_amounts, dtype=np.float64))
        elif dates is not None and amounts is not None and len(dates) > 0:
            dt_parts.append(pd.Series(pd.to_datetime(dates)))
            amt_parts.append(np.asarray(amounts, dtype=np.float64))

        if not dt_parts:
            self.seasonality_weights = np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)
            return self.seasonality_weights

        df = pd.DataFrame({
            "dt": pd.concat(dt_parts, ignore_index=True),
            "amt": np.concatenate(amt_parts),
        })
        df = df[df["amt"] > 0.0].copy()
        if df.empty:
            self.seasonality_weights = np.full(self.n_bins, 1.0 / self.n_bins, dtype=np.float64)
            return self.seasonality_weights

        df["year"] = df["dt"].dt.year
        if full_years_only:
            yrs = sorted(df["year"].unique())
            if len(yrs) >= 3:
                start_dt = pd.to_datetime(sample_start) if sample_start is not None else df["dt"].min()
                end_dt = pd.to_datetime(sample_end) if sample_end is not None else df["dt"].max()
                drop_yrs = set()
                if start_dt.month > 1 or start_dt.day > 15:
                    drop_yrs.add(yrs[0])
                if end_dt.month < 12 or end_dt.day < 15:
                    drop_yrs.add(yrs[-1])
                kept = df[~df["year"].isin(drop_yrs)]
                if not kept.empty:
                    df = kept

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
        start_month_or_week: float = 0.0,
    ) -> float:
        """Aloca o acréscimo de dividendo entre t1 e t2 por integração exata da sazonalidade em Cython."""
        if t_eval <= t1:
            return float(d1)
        if t_eval >= t2:
            return float(d2)

        w_arr = np.ascontiguousarray(self.seasonality_weights, dtype=np.float64)
        s0 = float(start_month_or_week)
        w_tot = float(_cy_cum_season(w_arr, float(t1), float(t2), s0))
        if w_tot <= 1e-12:
            return float(d1 + (d2 - d1) * ((t_eval - t1) / (t2 - t1)))
        w_part = float(_cy_cum_season(w_arr, float(t1), float(t_eval), s0))
        return float(d1 + (d2 - d1) * (w_part / w_tot))

    def allocate_to_ex_dates(
        self,
        t1: float,
        d_pv1: float,
        t2: float,
        d_pv2: float,
        ex_times: Sequence[float],
        r_cont: float = 0.0,
        start_month_or_week: float = 0.0,
        ex_weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Aloca o incremento de valor presente de dividendos entre t1 e t2 exatamente às
        datas ex-dividendo discretas em (t1, t2], respeitando sazonalidade e fator de desconto.
        """
        ex_t = np.asarray(ex_times, dtype=np.float64)
        if len(ex_t) == 0:
            return np.empty(0, dtype=np.float64)
        delta_pv = float(d_pv2) - float(d_pv1)
        if ex_weights is not None:
            w_ex = np.maximum(0.0, np.asarray(ex_weights, dtype=np.float64))
        else:
            bins = np.floor(float(start_month_or_week) + ex_t * self.n_bins).astype(int) % self.n_bins
            w_ex = self.seasonality_weights[bins]

        in_win = (ex_t > float(t1) + 1e-12) & (ex_t <= float(t2) + 1e-12)
        pv_w = np.where(in_win, w_ex * np.exp(-float(r_cont) * ex_t), 0.0)
        denom = float(np.sum(pv_w))
        if denom <= 1e-14:
            out = np.zeros(len(ex_t), dtype=np.float64)
            if np.any(in_win):
                disc_sum = float(np.sum(np.exp(-float(r_cont) * ex_t[in_win])))
                out[in_win] = delta_pv / max(1e-12, disc_sum)
            return out
        alpha = delta_pv / denom
        return np.where(in_win, alpha * w_ex, 0.0)

    @staticmethod
    def extrapolate_long_forward_yield(
        times: Sequence[float],
        implied_forwards: Sequence[float],
        spot: float,
        r_cont: Union[float, Sequence[float]],
        t_eval: float,
        augment_div_pv: Optional[float] = None,
        augment_yield: Optional[float] = None,
        augment_horizon: float = 1.0,
    ) -> float:
        """
        Extrapolação de rendimento a termo a partir do par mais longo com separação >= 1 ano,
        com aumento por previsões quando os vencimentos listados cobrem menos de 1 ano.
        """
        t_arr = np.asarray(times, dtype=np.float64)
        f_arr = np.asarray(implied_forwards, dtype=np.float64)
        order = np.argsort(t_arr)
        t_arr, f_arr = t_arr[order], f_arr[order]
        r_arr = np.full_like(t_arr, float(r_cont)) if np.isscalar(r_cont) else np.asarray(r_cont, dtype=np.float64)[order]
        y_arr = np.log((float(spot) * np.exp(r_arr * t_arr)) / np.maximum(1e-12, f_arr)) / np.maximum(1e-8, t_arr)

        t_q = float(t_eval)
        if t_q <= float(t_arr[-1]) + 1e-12:
            if t_q <= float(t_arr[0]):
                return float(y_arr[0])
            cum_grid = y_arr * t_arr
            return float(np.interp(t_q, t_arr, cum_grid) / max(1e-8, t_q))

        i_end = len(t_arr) - 1
        t_end = float(t_arr[i_end])
        y_n = float(y_arr[i_end])
        i_start = -1
        for i in range(i_end - 1, -1, -1):
            if t_end - float(t_arr[i]) >= 1.0 - 1e-9:
                i_start = i
                break

        if i_start >= 0:
            dt_span = max(1e-8, t_end - float(t_arr[i_start]))
            fwd_yield = (y_n * t_end - float(y_arr[i_start]) * float(t_arr[i_start])) / dt_span
        elif t_end < 1.0 - 1e-9 and (augment_div_pv is not None or augment_yield is not None):
            t_h = max(1.0, float(augment_horizon), t_end + 1e-6)
            if augment_div_pv is not None:
                d_pv_end = float(spot) - float(f_arr[i_end]) * math.exp(-float(r_arr[i_end]) * t_end)
                d_pv_tot = d_pv_end + float(augment_div_pv)
                r_h = float(r_arr[i_end])
                f_h = max(1e-12, float(spot) - d_pv_tot) * math.exp(r_h * t_h)
                cum_h = r_h * t_h - math.log(f_h / float(spot))
            else:
                cum_h = y_n * t_end + float(augment_yield) * (t_h - t_end)
            fwd_yield = cum_h / t_h
        elif len(t_arr) >= 2:
            dt_span = max(1e-8, t_end - float(t_arr[0]))
            fwd_yield = (y_n * t_end - float(y_arr[0]) * float(t_arr[0])) / dt_span
        else:
            fwd_yield = y_n

        cum = y_n * t_end + fwd_yield * (t_q - t_end)
        return float(cum / t_q)

    @staticmethod
    def extrapolate_equity_cash_dividends(
        times: Sequence[float],
        implied_forwards: Sequence[float],
        spot: float,
        r_cont: Union[float, Sequence[float]],
        target_div_times: Sequence[float],
        cash_cutoff_years: float = 10.0,
        augment_div_pv: Optional[float] = None,
        augment_yield: Optional[float] = None,
        augment_horizon: float = 1.0,
    ) -> Dict[str, object]:
        """Converte o rendimento extrapolado em fluxos discretos em caixa até 10 anos e yield contínuo depois."""
        t_list = sorted([float(x) for x in target_div_times if 0.0 < float(x) <= float(cash_cutoff_years) + 1e-12])
        r_tail = float(r_cont) if np.isscalar(r_cont) else float(r_cont[-1])
        pv_prev = 0.0
        cash_items: List[Dict[str, float]] = []
        for t_j in t_list:
            y_j = DividendCurveBBG.extrapolate_long_forward_yield(
                times, implied_forwards, spot, r_cont, t_j,
                augment_div_pv=augment_div_pv, augment_yield=augment_yield, augment_horizon=augment_horizon,
            )
            pv_j = float(spot) * (1.0 - math.exp(-y_j * t_j))
            d_pv_j = pv_j - pv_prev
            d_cash_j = d_pv_j * math.exp(r_tail * t_j)
            cash_items.append({"t_years": t_j, "cash_dividend": float(d_cash_j), "pv": float(d_pv_j)})
            pv_prev = pv_j

        y_15 = DividendCurveBBG.extrapolate_long_forward_yield(
            times, implied_forwards, spot, r_cont, max(float(times[-1]) + 1.0, 15.0),
            augment_div_pv=augment_div_pv, augment_yield=augment_yield, augment_horizon=augment_horizon,
        )
        y_14 = DividendCurveBBG.extrapolate_long_forward_yield(
            times, implied_forwards, spot, r_cont, max(float(times[-1]) + 1.0, 15.0) - 1.0,
            augment_div_pv=augment_div_pv, augment_yield=augment_yield, augment_horizon=augment_horizon,
        )
        fwd_tail_yield = y_15 * max(float(times[-1]) + 1.0, 15.0) - y_14 * (max(float(times[-1]) + 1.0, 15.0) - 1.0)
        return {
            "discrete_cash_dividends": cash_items,
            "residual_long_yield": float(fwd_tail_yield),
            "cutoff_years": float(cash_cutoff_years),
        }

    @staticmethod
    def option_implied_spot(
        fwd_first: float,
        r_cont: float,
        T_first: float,
        forecast_div_pv: float = 0.0,
    ) -> float:
        """Estima o spot implícito pela primeira série de opções ajustada por dividendos previstos."""
        return float(fwd_first) * math.exp(-float(r_cont) * float(T_first)) + float(forecast_div_pv)

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
        div_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """
        Decompõe o dividendo implícito acumulado até T em dividendos em caixa d_i,
        dividendos proporcionais y_i e taxa contínua residual q conforme os pesos por horizonte.
        Pondera explicitamente a parcela discreta por (omega_i^d + omega_i^y) para que datas com
        peso 100% contínuo (omega_i^q = 1) não recebam saltos discretos ou proporcionais residuais.
        """
        raw_pairs: List[Tuple[float, float]] = []
        if div_weights is None:
            for x in div_times:
                xf = float(x)
                if 0.0 < xf <= float(T):
                    raw_pairs.append((xf, 1.0))
        else:
            if len(div_weights) != len(div_times):
                raise ValueError("div_weights deve ter o mesmo comprimento de div_times.")
            for x, w in zip(div_times, div_weights):
                xf, wf = float(x), max(0.0, float(w))
                if 0.0 < xf <= float(T):
                    raw_pairs.append((xf, wf))
        raw_pairs.sort(key=lambda p: p[0])

        t_arr = np.array([p[0] for p in raw_pairs], dtype=np.float64)
        div_w = np.array([p[1] for p in raw_pairs], dtype=np.float64)
        n = len(t_arr)
        if n > 0:
            mean_w = float(np.mean(div_w))
            if mean_w > 0.0:
                div_w = div_w / mean_w
            else:
                div_w = np.ones(n, dtype=np.float64)

        f0 = float(spot_f0)
        fT = float(fwd_T)
        r = float(r_cont)
        t_mat = float(T)

        if n == 0 or is_index:
            q_val = r - math.log(max(1e-12, fT) / f0) / t_mat
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

        w_disc = w_d + w_y
        disc_factors = np.exp(r * (t_mat - t_arr))
        denom_1 = float(np.sum(div_w * disc_factors))
        d_hat = (f0 * math.exp(r * t_mat) - fT) / max(1e-12, denom_1)

        f0_rq_T = f0 * math.exp(r * t_mat) - d_hat * float(np.sum(div_w * w_q * disc_factors))
        f0_rq_T = max(1e-12, f0_rq_T)
        q_val = r - math.log(f0_rq_T / f0) / t_mat

        disc_rq = np.exp((r - q_val) * (t_mat - t_arr))
        denom_3 = float(np.sum(div_w * w_disc * disc_rq))
        if denom_3 <= 1e-12:
            return {
                "q": float(q_val),
                "d_global": 0.0,
                "d_hat": float(d_hat),
                "times": t_arr,
                "d_cash": np.zeros(n, dtype=np.float64),
                "y_prop": np.zeros(n, dtype=np.float64),
                "w_d": w_d,
                "w_y": w_y,
                "w_q": w_q,
            }

        d_global = (f0 * math.exp((r - q_val) * t_mat) - fT) / denom_3
        d_cash = np.zeros(n, dtype=np.float64)
        y_prop = np.zeros(n, dtype=np.float64)
        f_prev = f0
        t_prev = 0.0
        for i in range(n):
            drift = f_prev * math.exp((r - q_val) * (t_arr[i] - t_prev))
            d_eff_i = d_global * div_w[i] * w_disc[i]
            d_cash[i] = d_global * div_w[i] * w_d[i]
            f_i = drift - d_eff_i
            if w_y[i] > 1e-15:
                y_prop[i] = 1.0 - ((f_i + d_cash[i]) / max(1e-12, drift))
            else:
                y_prop[i] = 0.0
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
    min_options_per_maturity: int = 3,
    exercise: str = "EUROPEAN",
    spot: Optional[float] = None,
    discrete_dividends: Optional[Union[Sequence[Tuple[float, float]], Dict[float, float]]] = None,
    prop_dividends: Optional[Sequence[Tuple[float, float]]] = None,
    crr_steps: int = 120,
) -> pd.DataFrame:
    """
    Executa os critérios de seleção bid/ask vs settlement e os sete filtros de limpeza
    de superfície da Bloomberg (prazos >= 1 dia, somente OTM, desamericanização opcional,
    subsequência monótona, filtro de spreads > média + 3 desvios, exclusão em dois estágios
    de desvios de IV e corte mínimo de 3 cotações e 20% da mediana por vencimento).
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
                valid_mask = (vb >= min_price) & (va >= vb) & ((vb == 0.0) | (va <= 3.0 * vb))
                n_ok = int(valid_mask.sum())
                if n_ok >= 3 and (n_ok / max(1, len(grp))) >= 0.10:
                    good_mats += 1
            if good_mats >= 3 or (total_mats > 0 and (good_mats / total_mats) >= 0.10):
                use_bid_ask = True
    else:
        use_bid_ask = bool(use_bid_ask and has_ba and b_col and a_col)

    if not use_bid_ask and "age_days" in work.columns:
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
    is_am = exercise.strip().upper().startswith("AM") and spot is not None and float(spot) > 0.0

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
            if sub.empty:
                continue
            sp_ba = pd.to_numeric(sub[a_col], errors="coerce") - pd.to_numeric(sub[b_col], errors="coerce")
            mean_sp = float(sp_ba.mean())
            std_sp = float(sp_ba.std(ddof=0)) if len(sp_ba) > 1 else 0.0
            if std_sp > 0.0:
                sub = sub[sp_ba <= mean_sp + 3.0 * std_sp].copy()
            sub["_price"] = 0.5 * (
                pd.to_numeric(sub[b_col], errors="coerce") + pd.to_numeric(sub[a_col], errors="coerce")
            )
            sub["_spread"] = pd.to_numeric(sub[a_col], errors="coerce") - pd.to_numeric(sub[b_col], errors="coerce")
        else:
            if p_col is None:
                continue
            sub["_price"] = pd.to_numeric(sub[p_col], errors="coerce")
            sub = sub[sub["_price"] >= min_price].copy()

        sub["_k"] = pd.to_numeric(sub[k_col], errors="coerce")
        sub["_is_call"] = sub[tp_col].astype(str).str.lower().str.startswith("c")
        otm = (sub["_is_call"] & (sub["_k"] >= fwd)) | ((~sub["_is_call"]) & (sub["_k"] <= fwd))
        sub = sub[otm].copy()
        if sub.empty:
            continue

        mono_parts: List[pd.DataFrame] = []
        for ic in (True, False):
            part = sub[sub["_is_call"] == ic].sort_values("_k").reset_index(drop=True)
            if not part.empty:
                keep_idx = _longest_monotonic_indices(part["_price"].to_numpy(dtype=np.float64), non_increasing=ic)
                mono_parts.append(part.iloc[keep_idx])
        if not mono_parts:
            continue
        sub = pd.concat(mono_parts, ignore_index=True)

        if is_am:
            dt_arr, da_arr, pt_arr, py_arr = _unpack_dividend_schedules(t_val, discrete_dividends, prop_dividends)
            pv_div0 = float(np.sum(da_arr * np.exp(-r_c * dt_arr))) if len(dt_arr) > 0 else 0.0
            prop_mult = float(np.prod(1.0 - py_arr)) if len(py_arr) > 0 else 1.0
            s_adj = max(1e-12, (float(spot) - pv_div0) * prop_mult)
            q_est = r_c - math.log(max(fwd, 1e-12) / s_adj) / max(1e-8, t_val)
            dt_arg = dt_arr if len(dt_arr) > 0 else None
            da_arg = da_arr if len(da_arr) > 0 else None
            pt_arg = pt_arr if len(pt_arr) > 0 else None
            py_arg = py_arr if len(py_arr) > 0 else None
            ivs = np.array(
                [
                    float(
                        _cy_iv_american_crr(
                            float(p), float(spot), float(k), r_c, q_est, t_val, bool(ic), crr_steps,
                            dt_arg, da_arg, pt_arg, py_arg,
                        )
                    )
                    for p, k, ic in zip(sub["_price"], sub["_k"], sub["_is_call"])
                ],
                dtype=np.float64,
            )
            sub["_price_raw"] = sub["_price"]
            sub["_price"] = np.array(
                [
                    float(_cy_black76(fwd, float(k), r_c, t_val, float(v), bool(ic))[0])
                    if math.isfinite(v) else float("nan")
                    for k, v, ic in zip(sub["_k"], ivs, sub["_is_call"])
                ],
                dtype=np.float64,
            )
        else:
            ivs = np.array(
                [
                    _cy_iv_black76(float(p), fwd, float(k), r_c, t_val, bool(ic))
                    for p, k, ic in zip(sub["_price"], sub["_k"], sub["_is_call"])
                ],
                dtype=np.float64,
            )
        sub["_iv"] = ivs
        sub = sub[np.isfinite(sub["_iv"]) & (sub["_iv"] > 1e-4) & np.isfinite(sub["_price"])].copy()
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
    min_req = max(int(min_options_per_maturity), int(math.ceil(0.20 * med_count)))
    kept = [g for g in refined if len(g) >= min_req]
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
        smooth_damping: bool = True,
        use_hazard_damping: bool = False,
        fit_diagnostics: Optional[Dict[str, object]] = None,
    ):
        self.p = np.ascontiguousarray(weights, dtype=np.float64)
        self.xi = np.ascontiguousarray(xis, dtype=np.float64)
        self.Sigma = np.ascontiguousarray(sigmas_total, dtype=np.float64)
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
        self.smooth_damping = bool(smooth_damping)
        self.use_hazard_damping = bool(use_hazard_damping)
        self.fit_diagnostics: Dict[str, object] = dict(fit_diagnostics) if fit_diagnostics is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "LognormalMixtureSlice",
            "weights": self.p.tolist(),
            "xis": self.xi.tolist(),
            "sigmas_total": self.Sigma.tolist(),
            "forward": self.F,
            "T": self.T,
            "k_min_damp": self.k_min_damp,
            "smooth_damping": self.smooth_damping,
            "use_hazard_damping": self.use_hazard_damping,
            "fit_diagnostics": dict(self.fit_diagnostics) if self.fit_diagnostics else {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LognormalMixtureSlice":
        return cls(
            weights=data["weights"],
            xis=data["xis"],
            sigmas_total=data["sigmas_total"],
            forward=float(data["forward"]),
            T=float(data["T"]),
            k_min_damp=float(data["k_min_damp"]) if data.get("k_min_damp") is not None else None,
            smooth_damping=bool(data.get("smooth_damping", True)),
            use_hazard_damping=bool(data.get("use_hazard_damping", False)),
            fit_diagnostics=data.get("fit_diagnostics"),
        )

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
        if self.smooth_damping:
            exponent = -((x_m - x) * (x_m - x)) / (2.0 * self.T)
        else:
            exponent = (x_m * x_m - x * x) / (2.0 * self.T)
        lam_new = self.lambda_hazard * math.exp(exponent)
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
        apply_hazard_damping: Optional[bool] = None,
        beta_sigma: float = 1.0,
    ) -> float:
        """Preço europeu exato da mistura de lognormais via núcleo Cython acrescido do átomo de default nas puts."""
        k_f = float(strike)
        r_f = float(r_cont)
        damp_flag = self.use_hazard_damping if apply_hazard_damping is None else bool(apply_hazard_damping)
        p_eff, xi_eff, q_eff = self.damped_parameters(k_f) if damp_flag else (self.p, self.xi, self.Q_default)
        res = _cy_bbg_mix_prices_vec(
            np.array([k_f], dtype=np.float64),
            np.array([1 if is_call else 0], dtype=np.uint8),
            self.F,
            r_f,
            self.T,
            np.ascontiguousarray(p_eff, dtype=np.float64),
            np.ascontiguousarray(xi_eff, dtype=np.float64),
            self.Sigma,
            float(q_eff),
            float(beta_sigma),
        )
        return float(res[0])

    def option_prices_vec(
        self,
        strikes: Sequence[float],
        is_calls: Sequence[bool],
        r_cont: float = 0.0,
        apply_hazard_damping: Optional[bool] = None,
        beta_sigma: float = 1.0,
    ) -> np.ndarray:
        """Avaliação em lote Cython de prêmios europeus da mistura com amortecimento opcional."""
        k_arr = np.ascontiguousarray(strikes, dtype=np.float64)
        ic_arr = np.ascontiguousarray(is_calls, dtype=np.uint8)
        damp_flag = self.use_hazard_damping if apply_hazard_damping is None else bool(apply_hazard_damping)
        if not damp_flag or self.Q_default <= 1e-12 or bool(np.all(k_arr >= self.k_min_damp)):
            return _cy_bbg_mix_prices_vec(
                k_arr, ic_arr, self.F, float(r_cont), self.T,
                self.p, self.xi, self.Sigma, self.Q_default, float(beta_sigma),
            )
        out = np.empty(len(k_arr), dtype=np.float64)
        above = k_arr >= self.k_min_damp
        if np.any(above):
            out[above] = _cy_bbg_mix_prices_vec(
                k_arr[above], ic_arr[above], self.F, float(r_cont), self.T,
                self.p, self.xi, self.Sigma, self.Q_default, float(beta_sigma),
            )
        for i in np.flatnonzero(~above):
            out[i] = self.option_price(
                float(k_arr[i]), r_cont=r_cont, is_call=bool(ic_arr[i]),
                apply_hazard_damping=damp_flag, beta_sigma=beta_sigma,
            )
        return out

    def _raw_implied_vol(
        self,
        strike: float,
        r_cont: float = 0.0,
        beta_sigma: float = 1.0,
        apply_hazard_damping: Optional[bool] = None,
    ) -> float:
        k_f = float(strike)
        is_call = k_f >= self.F
        damp_flag = self.use_hazard_damping if apply_hazard_damping is None else bool(apply_hazard_damping)
        price = self.option_price(
            k_f,
            r_cont=r_cont,
            is_call=is_call,
            apply_hazard_damping=damp_flag,
            beta_sigma=beta_sigma,
        )
        iv = float(_cy_iv_black76(price, self.F, k_f, float(r_cont), self.T, is_call))
        if not math.isfinite(iv) or iv <= 1e-6:
            return float(np.average((float(beta_sigma) * self.Sigma) / math.sqrt(self.T), weights=np.maximum(self.p, 1e-8)))
        return iv

    def _safe_wing_anchor(
        self,
        k_anchor: float,
        is_left: bool,
        r_cont: float,
        beta_sigma: float,
        apply_hazard_damping: Optional[bool],
    ) -> Tuple[float, float]:
        x_a = math.log(max(1e-12, float(k_anchor)) / self.F)
        damp_flag = self.use_hazard_damping if apply_hazard_damping is None else bool(apply_hazard_damping)
        for _ in range(14):
            k_try = self.F * math.exp(x_a)
            is_c = not is_left
            p_try = self.option_price(k_try, r_cont=r_cont, is_call=is_c, apply_hazard_damping=damp_flag, beta_sigma=beta_sigma)
            iv_try = float(_cy_iv_black76(p_try, self.F, k_try, float(r_cont), self.T, is_c))
            if math.isfinite(iv_try) and iv_try > 1e-6 and p_try > 1e-24 * max(1.0, self.F):
                return k_try, iv_try
            x_a *= 0.85
        k_fb = self.F * math.exp(x_a)
        return k_fb, self._raw_implied_vol(k_fb, r_cont=r_cont, beta_sigma=beta_sigma, apply_hazard_damping=damp_flag)

    def implied_vol(
        self,
        strikes: Union[float, Sequence[float], np.ndarray],
        r_cont: float = 0.0,
        beta_sigma: float = 1.0,
        use_wing_extrap: bool = True,
        apply_hazard_damping: Optional[bool] = None,
    ) -> Union[float, np.ndarray]:
        """Volatilidade implícita da mistura com extrapolação de Roger Lee segura contra underflow nas asas."""
        scalar = np.isscalar(strikes)
        k_arr = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
        damp_flag = self.use_hazard_damping if apply_hazard_damping is None else bool(apply_hazard_damping)

        if not use_wing_extrap:
            res = np.array(
                [
                    self._raw_implied_vol(
                        float(k), r_cont=r_cont, beta_sigma=beta_sigma, apply_hazard_damping=damp_flag
                    )
                    for k in k_arr
                ],
                dtype=np.float64,
            )
            return float(res[0]) if scalar else res

        sig_atm = self._raw_implied_vol(self.F, r_cont=r_cont, beta_sigma=beta_sigma, apply_hazard_damping=damp_flag)
        v_est = max(0.05, 3.0 * sig_atm)
        sqrt_t = math.sqrt(self.T)
        k_min_target = self.F * math.exp(-4.0 * v_est * sqrt_t - 0.5 * v_est * v_est * self.T)
        k_max_target = self.F * math.exp(5.0 * v_est * sqrt_t + 0.5 * v_est * v_est * self.T)

        k_min, v_min_iv = self._safe_wing_anchor(k_min_target, True, r_cont, beta_sigma, damp_flag)
        k_max, v_max_iv = self._safe_wing_anchor(k_max_target, False, r_cont, beta_sigma, damp_flag)

        x_min = math.log(k_min / self.F)
        x_max = math.log(k_max / self.F)
        w_min = v_min_iv * v_min_iv * self.T
        w_max = v_max_iv * v_max_iv * self.T

        eps = 1e-3
        v_min_in = self._raw_implied_vol(k_min * math.exp(eps), r_cont=r_cont, beta_sigma=beta_sigma, apply_hazard_damping=damp_flag)
        v_max_in = self._raw_implied_vol(k_max * math.exp(-eps), r_cont=r_cont, beta_sigma=beta_sigma, apply_hazard_damping=damp_flag)
        slope_left = min(2.0, max(0.0, (w_min - v_min_in * v_min_in * self.T) / eps))
        slope_right = min(2.0, max(0.0, (w_max - v_max_in * v_max_in * self.T) / eps))

        out = np.empty(len(k_arr), dtype=np.float64)
        for i, k in enumerate(k_arr):
            kf = max(1e-12, float(k))
            if kf < k_min:
                dx = x_min - math.log(kf / self.F)
                w_ext = w_min + slope_left * dx
                out[i] = math.sqrt(max(1e-16, w_ext) / self.T)
            elif kf > k_max:
                dx = math.log(kf / self.F) - x_max
                w_ext = w_max + slope_right * dx
                out[i] = math.sqrt(max(1e-16, w_ext) / self.T)
            else:
                out[i] = self._raw_implied_vol(
                    kf, r_cont=r_cont, beta_sigma=beta_sigma, apply_hazard_damping=damp_flag
                )

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
        weighting: str = "price",
        bid_ask_spreads: Optional[Sequence[float]] = None,
        multistage: bool = True,
        init_slice: Optional["LognormalMixtureSlice"] = None,
        min_weight: float = 1e-3,
        reg_lambda: float = 1e-4,
        preset: Optional[str] = None,
        max_Q_default: float = 0.35,
    ) -> "LognormalMixtureSlice":
        """
        Calibra a mistura de lognormais com estado de default em múltiplos estágios utilizando
        o Jacobiano analítico exato em Cython, suporte a ponderação (`price`, `vega`, `bid_ask`),
        warm-start entre vencimentos e modo de índice líquido com N = 5 e Q = 0 (`preset='index_deep_otm'`).
        """
        if preset is not None and str(preset).strip().lower() in ("index_deep_otm", "index_n5"):
            n_components = 5
            allow_default = False

        k_np = np.ascontiguousarray(strikes, dtype=np.float64)
        p_np = np.ascontiguousarray(market_prices, dtype=np.float64)
        is_c = np.ascontiguousarray([str(t).lower().startswith("c") for t in option_types], dtype=np.uint8)
        fwd = float(forward)
        r_c = float(r_cont)
        t_val = max(1e-8, float(T))
        n = int(n_components)
        if n < 2:
            raise ValueError("n_components deve ser >= 2.")

        sqrt_t = math.sqrt(t_val)
        ivs = np.array(
            [_cy_iv_black76(float(p), fwd, float(k), r_c, t_val, bool(ic)) for p, k, ic in zip(p_np, k_np, is_c)],
            dtype=np.float64,
        )
        valid_iv_mask = np.isfinite(ivs) & (ivs > 1e-4)
        valid_ivs = ivs[valid_iv_mask]
        ann_med_iv = float(np.median(valid_ivs)) if len(valid_ivs) > 0 else 0.25
        sig_scale = ann_med_iv * sqrt_t

        w_mode = str(weighting).strip().lower()
        if w_mode == "price":
            p_scale = max(1e-6, float(np.median(p_np)))
            obs_w = np.full(len(p_np), 1.0 / p_scale, dtype=np.float64)
        elif w_mode == "vega":
            vegas = np.empty(len(p_np), dtype=np.float64)
            for idx_k, (kf, ic, iv_k) in enumerate(zip(k_np, is_c, ivs)):
                v_use = float(iv_k) if (math.isfinite(iv_k) and iv_k > 1e-4) else ann_med_iv
                vegas[idx_k] = float(_cy_black76(fwd, float(kf), r_c, t_val, v_use, bool(ic))[3])
            pos_v = vegas[vegas > 1e-12]
            med_v = float(np.median(pos_v)) if len(pos_v) > 0 else max(1e-6, float(np.median(p_np)))
            vega_safe = np.clip(vegas, 0.05 * med_v, 10.0 * med_v)
            obs_w = 1.0 / np.maximum(1e-10, vega_safe)
        elif w_mode == "bid_ask":
            if bid_ask_spreads is None:
                raise ValueError("bid_ask_spreads é obrigatório quando weighting='bid_ask'.")
            sp_np = np.asarray(bid_ask_spreads, dtype=np.float64)
            sp_safe = np.maximum(sp_np, max(1e-5, 1e-4 * fwd))
            obs_w = 1.0 / sp_safe
            obs_w = obs_w / max(1e-8, float(np.median(obs_w)))
        else:
            raise ValueError(f"weighting inválido: {weighting}. Use 'price', 'vega' ou 'bid_ask'.")

        k_span = max(0.05, float(np.max(np.abs(np.log(k_np / fwd)))))
        eta0 = np.full(n - 1, max(0.03, (0.8 * k_span) / (n - 1)), dtype=np.float64)
        if n == 4:
            u0 = np.array([1.2, 1.2, 0.2], dtype=np.float64)
        elif n == 5:
            u0 = np.array([1.0, 1.4, 1.2, -0.3], dtype=np.float64)
        else:
            center = 0.5 * (n - 1)
            raw_u = np.array([-0.5 * ((idx_l - center) ** 2) for idx_l in range(n)], dtype=np.float64)
            u0 = raw_u[1:] - raw_u[0]

        # Hierarquia inicial decrescente em volatilidade para quebrar degenerescência por permutação
        sig0 = np.linspace(sig_scale * 1.45, sig_scale * 0.65, n)

        lb_u = np.full(n - 1, -6.0)
        ub_u = np.full(n - 1, 6.0)
        lb_e = np.full(n - 1, 1e-3)
        ub_e = np.full(n - 1, 1.5)
        lb_s = np.asarray(min_Sigma, dtype=np.float64) if min_Sigma is not None else np.full(n, max(1e-4, 0.08 * ann_med_iv * sqrt_t))
        max_ann_cap = max(1.0, 3.0 * ann_med_iv)
        ub_s = np.maximum(lb_s * 1.05 + 1e-3, max_ann_cap * sqrt_t)

        fixed_q = max(0.0, 1.0 - float(max_P_survival))
        put_mask = (is_c == 0) & (p_np > 0.0) & (k_np > 0.0)
        q_put_cap = float(1.10 * np.min(np.exp(r_c * t_val) * p_np[put_mask] / k_np[put_mask])) if np.any(put_mask) else float(max_Q_default)
        ub_q = max(fixed_q + 1e-6, min(float(max_Q_default), q_put_cap))
        q0 = max(fixed_q, min(0.02, 0.25 * ub_q))
        if allow_default:
            lb = np.concatenate([lb_u, lb_e, lb_s, [fixed_q]])
            ub = np.concatenate([ub_u, ub_e, ub_s, [ub_q]])
            x0 = np.concatenate([u0, eta0, sig0, [q0]])
        else:
            lb = np.concatenate([lb_u, lb_e, lb_s])
            ub = np.concatenate([ub_u, ub_e, ub_s])
            x0 = np.concatenate([u0, eta0, sig0])

        if min_Sigma is not None:
            sig0 = np.maximum(sig0, lb_s * 1.02)
            x0[2 * n - 2 : 3 * n - 2] = sig0
        x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

        n_params = len(x0)
        m_obs = len(k_np)
        sig_soft_cap = max(0.70, 2.2 * ann_med_iv) * sqrt_t
        use_reg = float(reg_lambda) > 0.0
        reg_scale = math.sqrt(max(0.0, float(reg_lambda)))

        def _unpack(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
            u = vec[: n - 1]
            etas = vec[n - 1 : 2 * n - 2]
            sigs = vec[2 * n - 2 : 3 * n - 2]
            q_def = float(vec[3 * n - 2]) if allow_default else fixed_q
            logits = np.concatenate([[0.0], u])
            logits -= np.max(logits)
            exp_l = np.exp(logits)
            s_w = exp_l / np.sum(exp_l)
            p_weights = (1.0 - q_def) * s_w
            xis = cls._reconstruct_xis(p_weights, etas)
            return p_weights, xis, sigs, q_def, s_w

        cache: Dict[str, Any] = {"vec": None, "res": None, "jac": None}

        def _eval_full(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if cache["vec"] is not None and np.array_equal(vec, cache["vec"]):
                return cache["res"], cache["jac"]
            vec_c = np.ascontiguousarray(vec, dtype=np.float64)
            prices, jac_p = _cy_bbg_mix_prices_jac(
                k_np, is_c, fwd, r_c, t_val, vec_c, n, bool(allow_default), fixed_q
            )
            res_data = (prices - p_np) * obs_w
            jac_data = obs_w[:, None] * jac_p
            if not use_reg:
                cache["vec"] = vec_c.copy()
                cache["res"] = res_data
                cache["jac"] = jac_data
                return res_data, jac_data

            p_w, _, sigs, q_def, s_w = _unpack(vec_c)
            n_reg = 2 * n + (1 if allow_default else 0)
            res_reg = np.zeros(n_reg, dtype=np.float64)
            jac_reg = np.zeros((n_reg, n_params), dtype=np.float64)

            def _softplus_and_grad(z: float, temp: float) -> Tuple[float, float]:
                r = z / temp
                if r > 30.0:
                    return z, 1.0
                if r < -30.0:
                    return 0.0, 0.0
                er = math.exp(r)
                return temp * math.log1p(er), er / (1.0 + er)

            if min_weight > 0.0:
                w_coef = reg_scale / max(1e-8, float(min_weight))
                temp_w = max(1e-6, 0.25 * float(min_weight))
                for l in range(n):
                    sp_w, gz_w = _softplus_and_grad(float(min_weight) - p_w[l], temp_w)
                    res_reg[l] = w_coef * sp_w
                    if gz_w > 0.0:
                        for k_u in range(1, n):
                            dp_du = (p_w[k_u] if l == k_u else 0.0) - s_w[k_u] * p_w[l]
                            jac_reg[l, k_u - 1] = -w_coef * gz_w * dp_du
                        if allow_default:
                            jac_reg[l, 3 * n - 2] = w_coef * gz_w * s_w[l]

            s_coef = reg_scale / max(1e-8, sig_soft_cap)
            temp_s = max(1e-6, 0.05 * sig_soft_cap)
            for l in range(n):
                sp_s, gz_s = _softplus_and_grad(sigs[l] - sig_soft_cap, temp_s)
                res_reg[n + l] = s_coef * sp_s
                if gz_s > 0.0:
                    jac_reg[n + l, (2 * n - 2) + l] = s_coef * gz_s

            if allow_default:
                q_coef = 0.1 * reg_scale / max(0.05, t_val)
                res_reg[2 * n] = q_coef * q_def
                jac_reg[2 * n, 3 * n - 2] = q_coef

            res_all = np.concatenate([res_data, res_reg])
            jac_all = np.vstack([jac_data, jac_reg])
            cache["vec"] = vec_c.copy()
            cache["res"] = res_all
            cache["jac"] = jac_all
            return res_all, jac_all

        # Estágio 1: quebra de simetria fixando pesos e default (comparando também contra warm-start)
        if multistage:
            lb_s1 = np.concatenate([lb_e, lb_s])
            ub_s1 = np.concatenate([ub_e, ub_s])
            x_s1_0 = np.clip(x0[n - 1 : 3 * n - 2], lb_s1 + 1e-8, ub_s1 - 1e-8)

            def _expand_s1(z1: np.ndarray) -> np.ndarray:
                full = x0.copy()
                full[n - 1 : 3 * n - 2] = z1
                return full

            sol_s1 = least_squares(
                lambda z1: _eval_full(_expand_s1(z1))[0],
                x_s1_0,
                jac=lambda z1: _eval_full(_expand_s1(z1))[1][:, n - 1 : 3 * n - 2],
                bounds=(lb_s1, ub_s1),
                x_scale="jac",
                max_nfev=150,
            )
            x_cand = np.clip(_expand_s1(sol_s1.x), lb + 1e-8, ub - 1e-8)
            r_cand = _eval_full(x_cand)[0][:m_obs]
            best_cost = float(np.dot(r_cand, r_cand))
            x0 = x_cand
        else:
            r_0 = _eval_full(x0)[0][:m_obs]
            best_cost = float(np.dot(r_0, r_0))

        if init_slice is not None and init_slice.N == n:
            scale_t = math.sqrt(t_val / init_slice.T)
            etas_ws = init_slice.etas * scale_t
            sigs_ws = init_slice.Sigma * scale_t
            p_safe = np.maximum(1e-6, init_slice.p)
            u_ws = np.log(p_safe[1:] / p_safe[0])
            if allow_default:
                q_ws = 1.0 - (max(1e-6, init_slice.P_survival) ** (t_val / init_slice.T))
                x_ws = np.concatenate([u_ws, etas_ws, sigs_ws, [q_ws]])
            else:
                x_ws = np.concatenate([u_ws, etas_ws, sigs_ws])
            x_ws = np.clip(x_ws, lb + 1e-8, ub - 1e-8)
            r_ws = _eval_full(x_ws)[0][:m_obs]
            if float(np.dot(r_ws, r_ws)) <= best_cost:
                x0 = x_ws

        # Estágio 2: otimização conjunta com Jacobiano analítico completo
        sol = least_squares(
            lambda v: _eval_full(v)[0],
            x0,
            jac=lambda v: _eval_full(v)[1],
            bounds=(lb, ub),
            x_scale="jac",
            max_nfev=800,
        )
        p_opt, xi_opt, sig_opt, q_opt, _ = _unpack(sol.x)
        k_damp = min(float(np.min(k_np)), 0.9 * fwd)

        sl_out = cls(p_opt, xi_opt, sig_opt, forward=fwd, T=t_val, k_min_damp=k_damp)
        pred_prices = _cy_bbg_mix_prices_vec(k_np, is_c, fwd, r_c, t_val, p_opt, xi_opt, sig_opt, q_opt, 1.0)
        pred_ivs = np.asarray(sl_out.implied_vol(k_np, r_cont=r_c), dtype=np.float64)
        mask_diag = valid_iv_mask & np.isfinite(pred_ivs)
        if np.any(mask_diag):
            err_bps = (pred_ivs[mask_diag] - ivs[mask_diag]) * 1e4
            rmse_bps = float(np.sqrt(np.mean(err_bps * err_bps)))
            max_bps = float(np.max(np.abs(err_bps)))
        else:
            rmse_bps = float("nan")
            max_bps = float("nan")

        hit_rate = None
        if bid_ask_spreads is not None:
            half_sp = 0.5 * np.asarray(bid_ask_spreads, dtype=np.float64)
            hit_rate = float(np.mean(np.abs(pred_prices - p_np) <= half_sp + 1e-12))

        sl_out.fit_diagnostics = {
            "weighting": w_mode,
            "multistage": bool(multistage),
            "n_components": n,
            "allow_default": bool(allow_default),
            "nfev": int(sol.nfev),
            "rmse_vol_bps": rmse_bps,
            "max_vol_err_bps": max_bps,
            "rmse_price": float(np.sqrt(np.mean((pred_prices - p_np) ** 2))),
            "bid_ask_hit_rate": hit_rate,
        }
        return sl_out


class BloombergMixtureSurface:
    """
    Superfície bidimensional baseada no modelo de mistura de lognormais com estado de
    default (Bloomberg): suporta interpolação Hermite monótona em variância total,
    extrapolação contínua de curto prazo e de longo prazo (superfície de referência ou
    nível ATM de Heston), base de contagem configurável e interpolação de spreads bid/ask.
    """

    def __init__(
        self,
        slices_by_du: Dict[int, LognormalMixtureSlice],
        heston_params: Optional[Tuple[float, float, float]] = None,
        reference_atm_fn=None,
        day_count_basis: float = 252.0,
        spread_grid: Optional[Dict[int, Tuple[Sequence[float], Sequence[float]]]] = None,
        short_atm_anchor: Optional[Dict[int, float]] = None,
        reference_atm_points: Optional[Dict[float, float]] = None,
    ):
        self.day_count_basis = float(day_count_basis)
        if self.day_count_basis <= 0.0:
            raise ValueError("day_count_basis deve ser estritamente positivo.")
        self.dus = sorted([int(d) for d in slices_by_du.keys()])
        self.slices: Dict[int, LognormalMixtureSlice] = {int(d): slices_by_du[d] for d in self.dus}
        self.times = np.array([d / self.day_count_basis for d in self.dus], dtype=np.float64)
        self.heston_params = heston_params
        self.reference_atm_fn = reference_atm_fn
        self.reference_atm_points: Dict[float, float] = (
            {float(k): float(v) for k, v in reference_atm_points.items()} if reference_atm_points else {}
        )
        self.short_atm_anchor: Dict[int, float] = (
            {int(k): float(v) for k, v in short_atm_anchor.items()} if short_atm_anchor else {}
        )
        self.spread_grid: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if spread_grid is not None:
            for d_k, (ks_s, sp_s) in spread_grid.items():
                ks_arr = np.asarray(ks_s, dtype=np.float64)
                sp_arr = np.asarray(sp_s, dtype=np.float64)
                ord_s = np.argsort(ks_arr)
                self.spread_grid[int(d_k)] = (ks_arr[ord_s], sp_arr[ord_s])
        if self.heston_params is None and len(self.dus) >= 2:
            self.heston_params = self._fit_heston_atm()

    def _fit_heston_atm(self) -> Tuple[float, float, float]:
        t_arr = self.times
        atm_vols = np.array([self.slices[d].implied_vol(self.slices[d].F) for d in self.dus], dtype=np.float64)
        v_target = atm_vols * atm_vols
        th_lb = max(1e-5, 0.20 * float(np.min(v_target)))

        if len(t_arr) == 2:
            kap = 1.5
            kt = kap * t_arr
            fac = np.where(kt > 1e-6, (1.0 - np.exp(-kt)) / kt, 1.0 - 0.5 * kt)
            denom = float(fac[0] - fac[1])
            if abs(denom) > 1e-9:
                delta = float(v_target[0] - v_target[1]) / denom
                th = float(np.clip(v_target[1] - delta * fac[1], th_lb, 2.0))
                v0 = float(np.clip(th + delta, 1e-5, 2.0))
                return (v0, th, kap)

        def _h_var(p: np.ndarray) -> np.ndarray:
            v0, th, kap = p
            kt = kap * t_arr
            factor = np.where(kt > 1e-6, (1.0 - np.exp(-kt)) / kt, 1.0 - 0.5 * kt)
            res_v = np.sqrt(np.maximum(1e-8, th + (v0 - th) * factor)) - atm_vols
            if float(np.max(t_arr)) < 0.5:
                return np.concatenate([res_v, [0.01 * (kap - 1.5)]])
            return res_v

        x0 = np.clip(np.array([float(v_target[0]), max(th_lb * 1.1, float(v_target[-1])), 1.5], dtype=np.float64), [2e-5, th_lb + 1e-6, 0.05], [1.9, 1.9, 19.0])
        res = least_squares(_h_var, x0, bounds=([1e-5, th_lb, 1e-3], [2.0, 2.0, 20.0]), max_nfev=300)
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

    def _eval_reference_atm(self, t_years: float) -> Optional[float]:
        if self.reference_atm_fn is not None:
            return float(self.reference_atm_fn(t_years))
        if self.reference_atm_points:
            ts = sorted(self.reference_atm_points.keys())
            vs = [self.reference_atm_points[k] for k in ts]
            return float(np.interp(float(t_years), ts, vs))
        return None

    def _short_term_extrap_slice(self, du: int, t_years: float, fwd_t: float, strike: float) -> float:
        du1 = self.dus[0]
        sl1 = self.slices[du1]
        T1 = sl1.T
        t = max(1e-6, float(t_years))
        x = math.log(max(1e-12, float(strike)) / float(fwd_t))
        x_m = math.log(0.9)

        if sl1.Q_default > 1e-12:
            if strike < 0.9 * fwd_t:
                _, _, q_new_t1 = sl1.damped_parameters(sl1.F * math.exp(x))
                lam_new = -math.log(max(1e-12, 1.0 - q_new_t1)) / T1
                lam_hat = lam_new * (T1 / t) * math.exp(0.5 * (x_m * x_m - x * x) * (1.0 / t - 1.0 / T1))
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
        vol = float(sl_t.implied_vol(strike, r_cont=0.0, use_wing_extrap=True, apply_hazard_damping=False))
        if du in self.short_atm_anchor:
            vol_atm_base = float(sl_t.implied_vol(fwd_t, r_cont=0.0, use_wing_extrap=False, apply_hazard_damping=False))
            vol = max(1e-6, vol + (self.short_atm_anchor[du] - vol_atm_base))
        return vol

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
        base_atm = float(sl_t.implied_vol(fwd_t))

        ref_t = self._eval_reference_atm(t)
        ref_tn = self._eval_reference_atm(Tn)
        if ref_t is not None and ref_tn is not None:
            mkt_atm_n = float(sl_n.implied_vol(sl_n.F))
            target_atm = ref_t + (mkt_atm_n - ref_tn) * math.sqrt(Tn / t)
        else:
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
                # Estágio 2: reescala cada Sigma_l por target_atm / base_atm antes de resolver lambda
                matched = False
                scale_sig = target_atm / max(1e-6, base_atm)
                sig_scaled = sig_t * scale_sig
                if sl_t.P_survival < 1.0:
                    def _lam_err(lam_try: float) -> float:
                        p_surv_new = math.exp(-max(0.0, lam_try) * t)
                        p_try = sl_t.p * (p_surv_new / max(1e-12, sl_t.P_survival))
                        xi_try = LognormalMixtureSlice._reconstruct_xis(p_try, eta_t)
                        cand = LognormalMixtureSlice(p_try, xi_try, sig_scaled, forward=float(fwd_t), T=t)
                        return float(cand._raw_implied_vol(fwd_t, r_cont=0.0)) - target_atm
                    try:
                        lam_opt = float(brentq(_lam_err, 0.0, 2.0, xtol=1e-6))
                        p_surv_new = math.exp(-lam_opt * t)
                        p_try = sl_t.p * (p_surv_new / max(1e-12, sl_t.P_survival))
                        sl_final = LognormalMixtureSlice(
                            p_try, LognormalMixtureSlice._reconstruct_xis(p_try, eta_t), sig_scaled, forward=float(fwd_t), T=t
                        )
                        beta_opt = 1.0
                        matched = True
                    except Exception:
                        pass
                if not matched:
                    # Estágio 3: reduz distâncias eta_l entre médias até o limite lognormal se necessário
                    def _eta_err(scale_eta: float) -> float:
                        xi_try = LognormalMixtureSlice._reconstruct_xis(p_t, eta_t * scale_eta)
                        cand = LognormalMixtureSlice(p_t, xi_try, sig_scaled, forward=float(fwd_t), T=t)
                        return float(cand._raw_implied_vol(fwd_t, r_cont=0.0)) - target_atm
                    try:
                        s_eta = float(brentq(_eta_err, 0.0, 2.0, xtol=1e-6))
                        sl_final = LognormalMixtureSlice(
                            p_t, LognormalMixtureSlice._reconstruct_xis(p_t, eta_t * s_eta), sig_scaled, forward=float(fwd_t), T=t
                        )
                        beta_opt = 1.0
                    except Exception:
                        p_pure = sl_t.p / max(1e-12, sl_t.P_survival)
                        sig_pure = np.full(sl_t.N, target_atm * math.sqrt(t), dtype=np.float64)
                        sl_final = LognormalMixtureSlice(p_pure, np.ones(sl_t.N, dtype=np.float64), sig_pure, forward=float(fwd_t), T=t)
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

    def get_bid_ask_vol(
        self,
        du: int,
        strike: float,
        forward: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Retorna (vol_bid, vol_mid, vol_ask) combinando o smile calibrado ao spread interpolado."""
        v_mid = self.get_vol(du, strike, forward=forward)
        if not self.spread_grid:
            return (v_mid, v_mid, v_mid)
        s_dus = sorted(self.spread_grid.keys())
        ks_by_du = {d: self.spread_grid[d][0] for d in s_dus}
        sp_by_du = {d: self.spread_grid[d][1] for d in s_dus}
        sp = max(0.0, self.interpolate_vol_spread(s_dus, ks_by_du, sp_by_du, int(du), float(strike)))
        return (max(1e-6, v_mid - 0.5 * sp), v_mid, v_mid + 0.5 * sp)

    def _interp_forward(self, du_i: int) -> float:
        if du_i in self.slices:
            return float(self.slices[du_i].F)
        if du_i <= self.dus[0]:
            return float(self.slices[self.dus[0]].F)
        if du_i >= self.dus[-1]:
            return float(self.slices[self.dus[-1]].F)
        idx = int(np.searchsorted(self.dus, du_i, side="right") - 1)
        du_a, du_p = self.dus[idx], self.dus[idx + 1]
        fa, fp = self.slices[du_a].F, self.slices[du_p].F
        w = (du_i - du_a) / max(1e-12, float(du_p - du_a))
        return math.exp((1.0 - w) * math.log(max(1e-12, fa)) + w * math.log(max(1e-12, fp)))

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

        t_eval = max(1.0 / self.day_count_basis, du_i / self.day_count_basis)
        if du_i < self.dus[0]:
            fwd_use = self.slices[self.dus[0]].F if forward is None else float(forward)
            return self._short_term_extrap_slice(du_i, t_eval, fwd_use, float(strike))

        if du_i > self.dus[-1]:
            fwd_use = self.slices[self.dus[-1]].F if forward is None else float(forward)
            return self._long_term_extrap_slice(t_eval, fwd_use, float(strike))

        idx = int(np.searchsorted(self.dus, du_i, side="right") - 1)
        idx_lo = max(0, idx - 1)
        idx_hi = min(len(self.dus) - 1, idx + 2)
        sub_dus = self.dus[idx_lo : idx_hi + 1]
        fwd_eval = self._interp_forward(du_i) if forward is None else float(forward)
        m_ratio = float(strike) / max(1e-12, fwd_eval)

        t_sub = np.array([d / self.day_count_basis for d in sub_dus], dtype=np.float64)
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
        day_count_basis: float = 252.0,
        weighting: str = "price",
        multistage: bool = True,
        preset: Optional[str] = None,
        reference_surface=None,
        reference_long_tenors: Optional[Sequence[int]] = None,
        spread_grid: Optional[Dict[int, Tuple[Sequence[float], Sequence[float]]]] = None,
    ) -> "BloombergMixtureSurface":
        """
        Calibra a superfície multi-vencimento em cascata com warm-start, suporte opcional à
        superfície de referência (ATM, asas e skews até 10 anos) e preservação de monotonicidade.
        """
        if preset is not None and str(preset).strip().lower() in ("index_deep_otm", "index_n5"):
            n_components = 5
            allow_default = False

        basis = float(day_count_basis)
        sorted_dus = sorted(int(k) for k in slices_data.keys())

        # Localiza vencimento com liquidez ATM para ajuste de vencimentos sem cotações em [0.90, 1.10]*F
        atm_anchor_du = None
        atm_shift_anchor = 0.0
        if reference_surface is not None:
            for du_k in sorted_dus:
                d = slices_data[du_k]
                ks_d = np.asarray(d["strikes"], dtype=np.float64)
                fwd_d = float(d["forward"])
                in_atm = (ks_d >= 0.90 * fwd_d) & (ks_d <= 1.10 * fwd_d)
                if np.any(in_atm):
                    i_atm = int(np.argmin(np.abs(ks_d - fwd_d)))
                    k_atm = float(ks_d[i_atm])
                    p_atm = float(np.asarray(d["market_prices"], dtype=np.float64)[i_atm])
                    ic_atm = str(list(d["option_types"])[i_atm]).lower().startswith("c")
                    iv_mkt_atm = float(_cy_iv_black76(p_atm, fwd_d, k_atm, float(d.get("r_cont", 0.0)), du_k / basis, ic_atm))
                    iv_ref_atm = float(reference_surface.get_vol(du_k, strike=k_atm, forward=fwd_d))
                    if math.isfinite(iv_mkt_atm) and math.isfinite(iv_ref_atm):
                        atm_anchor_du = du_k
                        atm_shift_anchor = iv_mkt_atm - iv_ref_atm
                        break

        fitted: Dict[int, LognormalMixtureSlice] = {}
        prev_sig = None
        prev_slice: Optional[LognormalMixtureSlice] = None
        prev_p_surv = 1.0
        auto_spreads: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        q_ceil_by_du: Dict[int, float] = {}
        if allow_default and enforce_term_monotonicity and sorted_dus:
            q_obs: Dict[int, float] = {}
            for du_k in sorted_dus:
                d = slices_data[du_k]
                ks_d = np.asarray(d["strikes"], dtype=np.float64)
                ps_d = np.asarray(d["market_prices"], dtype=np.float64)
                is_c_d = np.array([str(t).lower().startswith("c") for t in d["option_types"]], dtype=bool)
                r_k_d = float(d.get("r_cont", 0.0))
                t_k_d = du_k / basis
                pm = (~is_c_d) & (ps_d > 0.0) & (ks_d > 0.0)
                if np.any(pm):
                    q_obs[du_k] = min(0.35, float(1.10 * np.min(np.exp(r_k_d * t_k_d) * ps_d[pm] / ks_d[pm])))
                else:
                    q_obs[du_k] = 0.35
            running_min = 0.35
            for du_k in reversed(sorted_dus):
                running_min = min(running_min, q_obs[du_k])
                q_ceil_by_du[du_k] = max(1e-6, running_min)

        for du_k in sorted_dus:
            d = slices_data[du_k]
            ks_use = np.asarray(d["strikes"], dtype=np.float64)
            ps_use = np.asarray(d["market_prices"], dtype=np.float64)
            ts_use = [str(t) for t in d["option_types"]]
            fwd_k = float(d["forward"])
            r_k = float(d.get("r_cont", 0.0))
            t_k = du_k / basis
            ba_sp = d.get("bid_ask_spreads")
            ba_sp_arr = np.asarray(ba_sp, dtype=np.float64) if ba_sp is not None else None

            if "vol_spreads" in d and d["vol_spreads"] is not None:
                auto_spreads[du_k] = (ks_use.copy(), np.asarray(d["vol_spreads"], dtype=np.float64))

            if reference_surface is not None:
                extra_k, extra_p, extra_t = [], [], []
                has_atm = bool(np.any((ks_use >= 0.90 * fwd_k) & (ks_use <= 1.10 * fwd_k)))
                if not has_atm and atm_anchor_du is not None:
                    for m_atm in (0.95, 1.00, 1.05):
                        k_syn = m_atm * fwd_k
                        v_ref = float(reference_surface.get_vol(du_k, strike=k_syn, forward=fwd_k))
                        v_syn = max(1e-3, v_ref + atm_shift_anchor)
                        ic_syn = k_syn >= fwd_k
                        extra_k.append(k_syn)
                        extra_p.append(float(_cy_black76(fwd_k, k_syn, r_k, t_k, v_syn, ic_syn)[0]))
                        extra_t.append("CALL" if ic_syn else "PUT")

                # Tratamento de asas quando a faixa cotada for estreita
                m_lo, m_hi = float(np.min(ks_use) / fwd_k), float(np.max(ks_use) / fwd_k)
                if m_lo > 0.80 or m_hi < 1.20:
                    k_edge_l, k_edge_h = float(np.min(ks_use)), float(np.max(ks_use))
                    iv_l = float(_cy_iv_black76(float(ps_use[np.argmin(ks_use)]), fwd_k, k_edge_l, r_k, t_k, ts_use[int(np.argmin(ks_use))].lower().startswith("c")))
                    iv_h = float(_cy_iv_black76(float(ps_use[np.argmax(ks_use)]), fwd_k, k_edge_h, r_k, t_k, ts_use[int(np.argmax(ks_use))].lower().startswith("c")))
                    shift_l = (iv_l - float(reference_surface.get_vol(du_k, strike=k_edge_l, forward=fwd_k))) if math.isfinite(iv_l) else atm_shift_anchor
                    shift_h = (iv_h - float(reference_surface.get_vol(du_k, strike=k_edge_h, forward=fwd_k))) if math.isfinite(iv_h) else atm_shift_anchor
                    for m_w in (0.65, 0.75, 1.25, 1.35):
                        if m_w < m_lo - 0.02 or m_w > m_hi + 0.02:
                            k_w = m_w * fwd_k
                            sh_w = shift_l if m_w < 1.0 else shift_h
                            v_w = max(1e-3, float(reference_surface.get_vol(du_k, strike=k_w, forward=fwd_k)) + sh_w)
                            ic_w = k_w >= fwd_k
                            extra_k.append(k_w)
                            extra_p.append(float(_cy_black76(fwd_k, k_w, r_k, t_k, v_w, ic_w)[0]))
                            extra_t.append("CALL" if ic_w else "PUT")

                if extra_k:
                    ks_use = np.concatenate([ks_use, np.array(extra_k, dtype=np.float64)])
                    ps_use = np.concatenate([ps_use, np.array(extra_p, dtype=np.float64)])
                    ts_use = ts_use + extra_t
                    if ba_sp_arr is not None:
                        med_sp = float(np.median(ba_sp_arr))
                        ba_sp_arr = np.concatenate([ba_sp_arr, np.full(len(extra_k), 2.0 * med_sp)])

            max_q_k = q_ceil_by_du.get(du_k, 0.35) if (allow_default and enforce_term_monotonicity) else 0.35
            sl = LognormalMixtureSlice.fit(
                strikes=ks_use,
                market_prices=ps_use,
                option_types=ts_use,
                forward=fwd_k,
                r_cont=r_k,
                T=t_k,
                n_components=n_components,
                allow_default=allow_default,
                min_Sigma=prev_sig if enforce_term_monotonicity else None,
                max_P_survival=prev_p_surv if enforce_term_monotonicity else 1.0,
                weighting=weighting,
                bid_ask_spreads=ba_sp_arr,
                multistage=multistage,
                init_slice=prev_slice,
                preset=preset,
                max_Q_default=max_q_k,
            )
            fitted[du_k] = sl
            prev_sig = sl.Sigma
            prev_slice = sl
            prev_p_surv = sl.P_survival

        # Skews de longo prazo via superfície de referência (decaimento sqrt(T_n / t))
        ref_atm_pts: Dict[float, float] = {}
        if reference_surface is not None:
            du_last = sorted_dus[-1]
            sl_last = fitted[du_last]
            d_last = slices_data[du_last]
            fwd_last = sl_last.F
            r_last = float(d_last.get("r_cont", 0.0))
            ks_last = np.asarray(d_last["strikes"], dtype=np.float64)
            ps_last = np.asarray(d_last["market_prices"], dtype=np.float64)
            ts_last = [str(t) for t in d_last["option_types"]]
            ivs_last = np.array(
                [
                    _cy_iv_black76(float(p), fwd_last, float(k), r_last, sl_last.T, str(tp).lower().startswith("c"))
                    for p, k, tp in zip(ps_last, ks_last, ts_last)
                ],
                dtype=np.float64,
            )
            m_last = ks_last / fwd_last
            valid_l = np.isfinite(ivs_last)
            if np.any(valid_l):
                m_v = m_last[valid_l]
                d_sig_v = np.array(
                    [float(iv) - float(reference_surface.get_vol(du_last, strike=float(k), forward=fwd_last)) for iv, k in zip(ivs_last[valid_l], ks_last[valid_l])],
                    dtype=np.float64,
                )
                ord_l = np.argsort(m_v)
                m_v, d_sig_v = m_v[ord_l], d_sig_v[ord_l]
                ref_atm_pts[sl_last.T] = float(reference_surface.get_vol(du_last, strike=fwd_last, forward=fwd_last))

                long_dus = [int(d) for d in reference_long_tenors if int(d) > du_last] if reference_long_tenors else []
                m_syn = np.linspace(0.60, 1.40, 11)
                for du_l in long_dus:
                    t_l = du_l / basis
                    fwd_l = fwd_last * math.exp(r_last * (t_l - sl_last.T))
                    ref_atm_pts[t_l] = float(reference_surface.get_vol(du_l, strike=fwd_l, forward=fwd_l))
                    ks_syn = m_syn * fwd_l
                    ps_syn = []
                    tp_syn = []
                    for m_i, k_s in zip(m_syn, ks_syn):
                        d_sig_i = float(np.interp(m_i, m_v, d_sig_v))
                        v_ref_l = float(reference_surface.get_vol(du_l, strike=float(k_s), forward=fwd_l))
                        v_target_l = max(1e-3, cls.extrapolate_long_skew_reference(sl_last.T, t_l, v_ref_l, d_sig_i))
                        ic_s = k_s >= fwd_l
                        ps_syn.append(float(_cy_black76(fwd_l, float(k_s), r_last, t_l, v_target_l, ic_s)[0]))
                        tp_syn.append("CALL" if ic_s else "PUT")
                    sl_l = LognormalMixtureSlice.fit(
                        strikes=ks_syn,
                        market_prices=ps_syn,
                        option_types=tp_syn,
                        forward=fwd_l,
                        r_cont=r_last,
                        T=t_l,
                        n_components=n_components,
                        allow_default=allow_default,
                        min_Sigma=prev_sig if enforce_term_monotonicity else None,
                        max_P_survival=prev_p_surv if enforce_term_monotonicity else 1.0,
                        weighting="vega",
                        multistage=False,
                        init_slice=prev_slice,
                        preset=preset,
                    )
                    fitted[du_l] = sl_l
                    prev_sig = sl_l.Sigma
                    prev_slice = sl_l
                    prev_p_surv = sl_l.P_survival

        ref_atm_fn = None
        if reference_surface is not None:
            def _ref_atm_fn(t_q: float) -> float:
                du_q = max(1, int(round(float(t_q) * basis)))
                sl_nearest = fitted[min(fitted.keys(), key=lambda dk: abs(dk - du_q))]
                return float(reference_surface.get_vol(du_q, strike=sl_nearest.F, forward=sl_nearest.F))
            ref_atm_fn = _ref_atm_fn
            for t_ext in (1.0, 2.0, 5.0, 10.0):
                ref_atm_pts[t_ext] = float(ref_atm_fn(t_ext))

        sp_final = spread_grid if spread_grid is not None else (auto_spreads if auto_spreads else None)
        return cls(
            fitted,
            reference_atm_fn=ref_atm_fn,
            day_count_basis=basis,
            spread_grid=sp_final,
            reference_atm_points=ref_atm_pts if ref_atm_pts else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "BloombergMixtureSurface",
            "day_count_basis": self.day_count_basis,
            "heston_params": list(self.heston_params) if self.heston_params is not None else None,
            "reference_atm_points": {str(k): v for k, v in self.reference_atm_points.items()} if self.reference_atm_points else None,
            "short_atm_anchor": {str(k): v for k, v in self.short_atm_anchor.items()} if self.short_atm_anchor else None,
            "spread_grid": {
                str(du_k): {"strikes": ks.tolist(), "spreads": sp.tolist()}
                for du_k, (ks, sp) in self.spread_grid.items()
            } if self.spread_grid else None,
            "slices": {str(du_k): sl.to_dict() for du_k, sl in self.slices.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BloombergMixtureSurface":
        slices = {int(k): LognormalMixtureSlice.from_dict(v) for k, v in data["slices"].items()}
        hp_raw = data.get("heston_params")
        hp = (float(hp_raw[0]), float(hp_raw[1]), float(hp_raw[2])) if hp_raw is not None else None
        ref_pts_raw = data.get("reference_atm_points")
        ref_pts = {float(k): float(v) for k, v in ref_pts_raw.items()} if ref_pts_raw else None
        short_raw = data.get("short_atm_anchor")
        short_anc = {int(k): float(v) for k, v in short_raw.items()} if short_raw else None
        sp_raw = data.get("spread_grid")
        sp_grid = {int(k): (v["strikes"], v["spreads"]) for k, v in sp_raw.items()} if sp_raw else None
        return cls(
            slices_by_du=slices,
            heston_params=hp,
            day_count_basis=float(data.get("day_count_basis", 252.0)),
            spread_grid=sp_grid,
            short_atm_anchor=short_anc,
            reference_atm_points=ref_pts,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "BloombergMixtureSurface":
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BloombergMixtureSurface":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
