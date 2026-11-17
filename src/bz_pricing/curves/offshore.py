import datetime
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from bz_pricing.core.curves import (
    Curve,
    NelsonSiegelCurve,
    SimpleCurve,
    SvenssonCurve,
)
from bz_pricing.core.interpolators import (
    Interpolator,
    Linear360Interpolator,
    LogCubicDiscountInterpolator,
    LogLinearDiscountInterpolator,
    MonotoneConvexInterpolator,
)


def resolve_day_count_basis(convention: Union[str, float, int]) -> float:
    """Resolve a base numérica anual associada a uma convenção de contagem de dias."""
    if isinstance(convention, (int, float)):
        val = float(convention)
        if val <= 0.0:
            raise ValueError("day_count_basis deve ser positivo.")
        return val
    conv = str(convention).strip().upper()
    if conv in ("ACT/360", "DC/360", "MONEY_MARKET_360"):
        return 360.0
    if conv in ("ACT/365", "ACT/365F", "ACT/365_FIXED", "DC/365"):
        return 365.0
    if conv in ("ACT/ACT", "ACT/ACT_ISDA", "ACT/ACT_ICMA"):
        return 365.25
    if conv in ("BUS/252", "DU/252"):
        return 252.0
    if conv in ("30/360", "THIRTY_360"):
        return 360.0
    raise ValueError(f"Convenção de contagem de dias não suportada: {convention}")


def make_offshore_interpolator(
    method: str = "monotone_convex",
    day_count_basis: float = 365.0,
    compounding: str = "continuous",
    freq: int = 1,
    extrap_left: int = 0,
    extrap_right: int = 1,
) -> Interpolator:
    """Instancia o interpolador Cython correspondente ao método offshore solicitado."""
    m = str(method).strip().lower()
    if m in ("monotone_convex", "hagan_west", "mc"):
        return MonotoneConvexInterpolator(
            extrap_left=extrap_left,
            extrap_right=extrap_right,
            day_count_basis=float(day_count_basis),
            compounding=compounding,
            freq=int(freq),
        )
    if m in ("log_linear", "loglinear", "log_linear_discount", "piecewise_loglinear"):
        return LogLinearDiscountInterpolator(
            extrap_left=extrap_left,
            extrap_right=extrap_right,
            day_count_basis=float(day_count_basis),
            compounding=compounding,
            freq=int(freq),
        )
    if m in ("log_cubic", "logcubic", "monotonic_log_cubic", "log_cubic_discount"):
        return LogCubicDiscountInterpolator(
            extrap_left=extrap_left,
            extrap_right=extrap_right,
            day_count_basis=float(day_count_basis),
            compounding=compounding,
            freq=int(freq),
        )
    if m in ("linear_360", "lin_360", "linear360"):
        return Linear360Interpolator(extrap_left=extrap_left, extrap_right=extrap_right)
    raise ValueError(
        f"Método de interpolação offshore '{method}' inválido; use 'monotone_convex', "
        "'log_linear', 'log_cubic', 'linear_360', 'nelson_siegel' ou 'svensson'."
    )


def _df_from_zero_rate(rate: float, dc: float, basis: float, compounding: str, freq: int = 1) -> float:
    tau = float(dc) / float(basis)
    if tau <= 0.0:
        return 1.0
    comp = str(compounding).strip().lower()
    if comp in ("continuous", "cont"):
        return math.exp(-float(rate) * tau)
    if comp in ("simple", "simple_linear", "linear"):
        return 1.0 / (1.0 + float(rate) * tau)
    if comp in ("exponential", "exponential_b3", "exp_252", "b3"):
        return (1.0 + float(rate)) ** (-tau)
    k = float(max(1, int(freq)))
    return (1.0 + float(rate) / k) ** (-k * tau)


def _zero_rate_from_df(df: float, dc: float, basis: float, compounding: str, freq: int = 1) -> float:
    if df <= 0.0:
        raise ValueError("Fator de desconto deve ser estritamente positivo.")
    tau = float(dc) / float(basis)
    if tau <= 1e-12:
        return 0.0
    comp = str(compounding).strip().lower()
    if comp in ("continuous", "cont"):
        return -math.log(df) / tau
    if comp in ("simple", "simple_linear", "linear"):
        return (1.0 / df - 1.0) / tau
    if comp in ("exponential", "exponential_b3", "exp_252", "b3"):
        return df ** (-1.0 / tau) - 1.0
    k = float(max(1, int(freq)))
    return k * (df ** (-1.0 / (k * tau)) - 1.0)


def _normalize_dc_and_rates(
    pillars_dc: Sequence[Union[float, int, datetime.date, str]],
    rates: Sequence[float],
    base_date: Optional[Union[datetime.date, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pillars_dc) != len(rates) or len(pillars_dc) == 0:
        raise ValueError("pillars_dc e rates devem ter o mesmo comprimento não nulo.")

    base_dt = pd.to_datetime(base_date).date() if base_date is not None else None
    dcs: List[float] = []
    for p in pillars_dc:
        if isinstance(p, (datetime.date, datetime.datetime, pd.Timestamp, str)):
            if base_dt is None:
                try:
                    dcs.append(float(p))
                    continue
                except (TypeError, ValueError) as exc:
                    raise ValueError("Informe base_date ao fornecer pilares como datas.") from exc
            p_dt = pd.to_datetime(p).date()
            dcs.append(float((p_dt - base_dt).days))
        else:
            dcs.append(float(p))

    r_arr = np.asarray(rates, dtype=np.float64)
    dc_arr = np.asarray(dcs, dtype=np.float64)
    valid = dc_arr > 0.0
    dc_arr = dc_arr[valid]
    r_arr = r_arr[valid]
    if len(dc_arr) == 0:
        raise ValueError("São necessários vértices com prazo positivo em dias corridos.")
    order = np.argsort(dc_arr)
    dc_sorted, uniq_idx = np.unique(dc_arr[order], return_index=True)
    return dc_sorted.astype(np.float64), r_arr[order][uniq_idx].astype(np.float64)


def _coupon_schedule_dc(maturity_dc: float, step_days: float) -> List[float]:
    """Gera agenda de datas de pagamento até maturity_dc caminhando de trás para frente."""
    if maturity_dc <= step_days + 1e-6:
        return [float(maturity_dc)]
    pts: List[float] = []
    cur = float(maturity_dc)
    while cur > 1e-6:
        pts.append(cur)
        cur -= float(step_days)
    pts.reverse()
    return pts


def _solve_pillar_zero_rate(
    settled_dc: List[float],
    settled_y: List[float],
    target_dc: float,
    cashflow_schedule: List[Tuple[float, float]],
    basis: float,
    compounding: str,
    freq: int,
    interp: Interpolator,
) -> float:
    """Resolve via interpolação e busca de raiz o zero-rate no vértice que garante PV par igual a 1,0."""

    def _pv_error(y_cand: float) -> float:
        t_arr = np.asarray(settled_dc + [target_dc], dtype=np.float64)
        y_arr = np.asarray(settled_y + [y_cand], dtype=np.float64)
        crv = SimpleCurve(t_arr, y_arr, interp)
        pv = 0.0
        for cf_dc, cf_amt in cashflow_schedule:
            r_cf = crv.get_rate(float(cf_dc))
            pv += float(cf_amt) * _df_from_zero_rate(r_cf, cf_dc, basis, compounding, freq)
        return pv - 1.0

    # Aproximação inicial usando a última taxa ajustada ou a taxa simples do vértice
    init_y = settled_y[-1] if settled_y else 0.04
    lo = max(-0.08, init_y - 0.15)
    hi = init_y + 0.25
    f_lo = _pv_error(lo)
    f_hi = _pv_error(hi)
    tries = 0
    while f_lo * f_hi > 0.0 and tries < 10:
        lo -= 0.10
        hi += 0.20
        f_lo = _pv_error(lo)
        f_hi = _pv_error(hi)
        tries += 1
    if f_lo * f_hi > 0.0:
        raise ValueError(f"Não foi possível enquadrar raiz para o vértice DC={target_dc}.")
    return float(brentq(_pv_error, lo, hi, xtol=1e-13))


def _polish_sparse_schedules(
    settled_dc: List[float],
    settled_y: List[float],
    schedules_by_idx: Dict[int, List[Tuple[float, float]]],
    basis: float,
    compounding: str,
    interp: Interpolator,
    n_passes: int = 3,
) -> List[float]:
    """Refina nós acoplados por interpolação intermediária para garantir PV par exato (< 1e-12)."""
    if not schedules_by_idx or len(settled_dc) < 2:
        return settled_y
    y_work = list(settled_y)
    t_arr = np.asarray(settled_dc, dtype=np.float64)
    for _ in range(n_passes):
        max_upd = 0.0
        for idx, sched in sorted(schedules_by_idx.items()):
            def _err(y_cand: float) -> float:
                tmp = list(y_work)
                tmp[idx] = y_cand
                crv = SimpleCurve(t_arr, np.asarray(tmp, dtype=np.float64), interp)
                pv = sum(
                    amt * _df_from_zero_rate(crv.get_rate(float(d)), d, basis, compounding, 1)
                    for d, amt in sched
                )
                return pv - 1.0

            cur_y = y_work[idx]
            new_y = float(brentq(_err, cur_y - 0.05, cur_y + 0.05, xtol=1e-13))
            max_upd = max(max_upd, abs(new_y - cur_y))
            y_work[idx] = new_y
        if max_upd < 1e-13:
            break
    return y_work


def bootstrap_ois_curve(
    pillars_dc: Sequence[Union[float, int, datetime.date, str]],
    par_rates: Sequence[float],
    convention: Union[str, float, int] = "ACT/360",
    payment_freq_months: int = 12,
    method: str = "monotone_convex",
    compounding: str = "continuous",
    futures_pillars: Optional[Sequence[Tuple[float, float, float]]] = None,
    base_date: Optional[Union[datetime.date, str]] = None,
    round_decimals: int = -1,
    round_mode: str = "round",
) -> Curve:
    """
    Constrói curva zero cupom OIS (SOFR, €STR, SONIA, F-TIIE) por bootstrapping
    exato de taxas de curto prazo, contratos futuros de 3M (opcional) e swaps OIS.

    Parâmetros:
    -----------
    pillars_dc : sequência de prazos em dias corridos (DC) ou datas se `base_date` informado.
    par_rates : taxas par dos instrumentos OIS em formato decimal (ex.: 0.045 para 4,50%).
    convention : 'ACT/360' (SOFR/€STR/F-TIIE), 'ACT/365' (SONIA) ou base numérica anual.
    payment_freq_months : periodicidade de pagamentos dos swaps OIS em meses (padrão: 12).
    method : 'monotone_convex' (padrão Hagan-West), 'log_linear' (padrão CME/LCH),
             'log_cubic', 'linear_360', 'nelson_siegel' ou 'svensson'.
    compounding : regime de representação da taxa zero retornada ('continuous', 'simple', 'compounded').
    futures_pillars : sequência opcional de tuplas (dc_inicio, dc_fim, taxa_forward_ou_preco).
                      Se valor > 20, interpreta como preço futuro (100 - taxa_pct).
    """
    basis = resolve_day_count_basis(convention)
    dc_arr, r_arr = _normalize_dc_and_rates(pillars_dc, par_rates, base_date=base_date)

    method_low = str(method).strip().lower()
    raw_method = "monotone_convex" if method_low in ("nelson_siegel", "svensson") else method_low
    extrap_r = 0 if raw_method == "linear_360" else 1
    interp = make_offshore_interpolator(
        raw_method, day_count_basis=basis, compounding=compounding, extrap_left=0, extrap_right=extrap_r
    )

    step_days = basis * (float(payment_freq_months) / 12.0) if payment_freq_months > 0 else 1e9

    settled_dc: List[float] = []
    settled_y: List[float] = []
    schedules_by_idx: Dict[int, List[Tuple[float, float]]] = {}

    # Processamento sequencial dos futuros OIS caso fornecidos para a ponta curta/média
    fut_map: Dict[float, Tuple[float, float]] = {}
    if futures_pillars:
        for dc_s, dc_e, val in sorted(futures_pillars, key=lambda x: float(x[1])):
            fwd_rate = (100.0 - float(val)) / 100.0 if float(val) > 20.0 else float(val)
            fut_map[float(dc_e)] = (float(dc_s), fwd_rate)

    # Combina vértices de futuros e swaps ordenados por maturidade final
    all_nodes: List[Tuple[float, str, Any]] = []
    fut_ends = set(fut_map.keys())
    for dc_v, rate_v in zip(dc_arr, r_arr):
        if float(dc_v) not in fut_ends:
            all_nodes.append((float(dc_v), "ois", float(rate_v)))
    for dc_e, (dc_s, fwd_r) in sorted(fut_map.items()):
        all_nodes.append((dc_e, "fut", (dc_s, fwd_r)))
    all_nodes.sort(key=lambda x: x[0])

    for dc_v, kind, payload in all_nodes:
        if kind == "fut":
            dc_s, fwd_r = payload
            if dc_s <= 0.0 or not settled_dc:
                df_start = 1.0 / (1.0 + fwd_r * (max(0.0, dc_s) / basis))
            else:
                temp_crv = SimpleCurve(np.asarray(settled_dc, dtype=np.float64), np.asarray(settled_y, dtype=np.float64), interp)
                df_start = _df_from_zero_rate(temp_crv.get_rate(dc_s), dc_s, basis, compounding)
            df_end = df_start / (1.0 + fwd_r * ((dc_v - dc_s) / basis))
            y_sol = _zero_rate_from_df(df_end, dc_v, basis, compounding)
        else:
            par_r = float(payload)
            if dc_v <= step_days + 1e-6:
                df_v = 1.0 / (1.0 + par_r * (dc_v / basis))
                y_sol = _zero_rate_from_df(df_v, dc_v, basis, compounding)
            else:
                pay_dates = _coupon_schedule_dc(dc_v, step_days)
                schedule: List[Tuple[float, float]] = []
                prev_d = 0.0
                for j, d_pay in enumerate(pay_dates):
                    alpha = (d_pay - prev_d) / basis
                    amt = par_r * alpha + (1.0 if j == len(pay_dates) - 1 else 0.0)
                    schedule.append((d_pay, amt))
                    prev_d = d_pay

                last_settled = settled_dc[-1] if settled_dc else 0.0
                if all(d_p <= last_settled + 1e-9 for d_p, _ in schedule[:-1]) and settled_dc:
                    temp_crv = SimpleCurve(np.asarray(settled_dc, dtype=np.float64), np.asarray(settled_y, dtype=np.float64), interp)
                    pv_cpns = 0.0
                    for d_p, amt_p in schedule[:-1]:
                        pv_cpns += amt_p * _df_from_zero_rate(temp_crv.get_rate(d_p), d_p, basis, compounding)
                    df_v = (1.0 - pv_cpns) / schedule[-1][1]
                    y_sol = _zero_rate_from_df(df_v, dc_v, basis, compounding)
                else:
                    y_sol = _solve_pillar_zero_rate(
                        settled_dc, settled_y, dc_v, schedule, basis, compounding, 1, interp
                    )
                schedules_by_idx[len(settled_dc)] = schedule

        settled_dc.append(dc_v)
        settled_y.append(y_sol)

    settled_y = _polish_sparse_schedules(settled_dc, settled_y, schedules_by_idx, basis, compounding, interp)

    t_nodes = np.asarray(settled_dc, dtype=np.float64)
    y_nodes = np.asarray(settled_y, dtype=np.float64)

    if method_low == "svensson":
        return SvenssonCurve.fit(t_nodes, y_nodes, day_count_basis=basis, compounding=compounding)
    if method_low == "nelson_siegel":
        return NelsonSiegelCurve.fit(t_nodes, y_nodes, day_count_basis=basis, compounding=compounding)

    return SimpleCurve(t_nodes, y_nodes, interp, round_decimals=round_decimals, round_mode=round_mode)


def bootstrap_treasury_curve(
    pillars_dc: Sequence[Union[float, int, datetime.date, str]],
    par_yields: Sequence[float],
    coupon_freq: int = 2,
    day_count_basis: Union[str, float, int] = 365.0,
    method: str = "monotone_convex",
    compounding: str = "continuous",
    base_date: Optional[Union[datetime.date, str]] = None,
    round_decimals: int = -1,
    round_mode: str = "round",
) -> Curve:
    """
    Constrói a curva soberana de Treasuries a partir de taxas par (Bills de curto prazo e
    Notes/Bonds com cupom periódico), utilizando Monotone Convex (padrão do Tesouro dos EUA),
    Svensson (padrão do Federal Reserve Board / BCE), Nelson-Siegel, Log-Linear ou Log-Cubic.
    """
    basis = resolve_day_count_basis(day_count_basis)
    dc_arr, y_arr = _normalize_dc_and_rates(pillars_dc, par_yields, base_date=base_date)

    method_low = str(method).strip().lower()
    raw_method = "monotone_convex" if method_low in ("nelson_siegel", "svensson") else method_low
    extrap_r = 0 if raw_method == "linear_360" else 1
    interp = make_offshore_interpolator(
        raw_method, day_count_basis=basis, compounding=compounding, extrap_left=0, extrap_right=extrap_r
    )

    step_days = basis / float(max(1, int(coupon_freq)))
    settled_dc: List[float] = []
    settled_y: List[float] = []
    schedules_by_idx: Dict[int, List[Tuple[float, float]]] = {}

    for dc_v, y_par in zip(dc_arr, y_arr):
        if dc_v <= step_days + 1e-6:
            # Título de período único / zero-cupom (Treasury Bill ou par de 6M)
            df_v = 1.0 / (1.0 + float(y_par) * (dc_v / basis))
            y_sol = _zero_rate_from_df(df_v, dc_v, basis, compounding)
        else:
            pay_dates = _coupon_schedule_dc(dc_v, step_days)
            schedule: List[Tuple[float, float]] = []
            prev_d = 0.0
            for j, d_pay in enumerate(pay_dates):
                alpha = (d_pay - prev_d) / basis
                amt = float(y_par) * alpha + (1.0 if j == len(pay_dates) - 1 else 0.0)
                schedule.append((d_pay, amt))
                prev_d = d_pay

            last_settled = settled_dc[-1] if settled_dc else 0.0
            if all(d_p <= last_settled + 1e-9 for d_p, _ in schedule[:-1]) and settled_dc:
                temp_crv = SimpleCurve(np.asarray(settled_dc, dtype=np.float64), np.asarray(settled_y, dtype=np.float64), interp)
                pv_cpns = 0.0
                for d_p, amt_p in schedule[:-1]:
                    pv_cpns += amt_p * _df_from_zero_rate(temp_crv.get_rate(d_p), d_p, basis, compounding)
                df_v = (1.0 - pv_cpns) / schedule[-1][1]
                y_sol = _zero_rate_from_df(df_v, dc_v, basis, compounding)
            else:
                y_sol = _solve_pillar_zero_rate(
                    settled_dc, settled_y, dc_v, schedule, basis, compounding, 1, interp
                )
            schedules_by_idx[len(settled_dc)] = schedule

        settled_dc.append(float(dc_v))
        settled_y.append(float(y_sol))

    settled_y = _polish_sparse_schedules(settled_dc, settled_y, schedules_by_idx, basis, compounding, interp)

    t_nodes = np.asarray(settled_dc, dtype=np.float64)
    y_nodes = np.asarray(settled_y, dtype=np.float64)

    if method_low == "svensson":
        return SvenssonCurve.fit(t_nodes, y_nodes, day_count_basis=basis, compounding=compounding)
    if method_low == "nelson_siegel":
        return NelsonSiegelCurve.fit(t_nodes, y_nodes, day_count_basis=basis, compounding=compounding)

    return SimpleCurve(t_nodes, y_nodes, interp, round_decimals=round_decimals, round_mode=round_mode)
