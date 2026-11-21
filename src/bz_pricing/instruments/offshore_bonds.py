import calendar
import datetime
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from bz_pricing.core.calendars import get_calendar
from bz_pricing.core.credit_core import risky_bond_pv as _cy_risky_bond_pv
from bz_pricing.core.curves import Curve


def _to_date(d: Union[datetime.date, str]) -> datetime.date:
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    if isinstance(d, pd.Timestamp):
        return d.date()
    return pd.to_datetime(str(d).strip(), dayfirst="/" in str(d)).date()


def _is_end_of_month(d: datetime.date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def _add_months(d: datetime.date, months: int, end_of_month: bool = False) -> datetime.date:
    total_m = (d.year * 12 + (d.month - 1)) + months
    year = total_m // 12
    month = (total_m % 12) + 1
    last_day = calendar.monthrange(year, month)[1]
    day = last_day if (end_of_month or d.day > last_day) else d.day
    return datetime.date(year, month, day)


def _days_30_360_us(d1: datetime.date, d2: datetime.date) -> int:
    dd1 = 30 if d1.day == 31 else d1.day
    dd2 = 30 if (d2.day == 31 and dd1 == 30) else d2.day
    return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (dd2 - dd1)


def _normalize_day_count(convention: str) -> str:
    c = str(convention).strip().upper().replace(" ", "_")
    aliases = {
        "ACT/ACT": "ACT/ACT_ICMA",
        "ACT/ACT_ISMA": "ACT/ACT_ICMA",
        "ACT/ACT_ICMA": "ACT/ACT_ICMA",
        "ISMA": "ACT/ACT_ICMA",
        "ICMA": "ACT/ACT_ICMA",
        "30/360": "30/360",
        "THIRTY_360": "30/360",
        "30/360_US": "30/360",
        "ACT/365": "ACT/365",
        "ACT/365F": "ACT/365",
        "ACT/365_FIXED": "ACT/365",
        "ACT/360": "ACT/360",
    }
    if c not in aliases:
        raise ValueError(
            f"Convenção de contagem de dias '{convention}' não suportada. "
            "Use 'ACT/ACT_ICMA', '30/360', 'ACT/365' ou 'ACT/360'."
        )
    return aliases[c]


def _year_fraction_bond(
    d1: datetime.date,
    d2: datetime.date,
    day_count: str,
    freq: int,
    ref_start: Optional[datetime.date] = None,
    ref_end: Optional[datetime.date] = None,
) -> float:
    if d2 <= d1:
        return 0.0
    dc = _normalize_day_count(day_count)
    if dc == "ACT/360":
        return float((d2 - d1).days) / 360.0
    if dc == "ACT/365":
        return float((d2 - d1).days) / 365.0
    if dc == "30/360":
        return float(_days_30_360_us(d1, d2)) / 360.0
    # ACT/ACT_ICMA: fração do período contratual dividida pela frequência anual
    if ref_start is not None and ref_end is not None and ref_end > ref_start:
        denom = float((ref_end - ref_start).days)
        return float((d2 - d1).days) / (denom * float(freq))
    return float((d2 - d1).days) / 365.25


def _modified_following(cal: Any, dt: datetime.date) -> datetime.date:
    adj = cal.following(dt)
    if adj.month != dt.month:
        # Volta para o último dia útil anterior dentro do mesmo mês
        prev_fn = getattr(cal, "preceding", None)
        if prev_fn is not None:
            return prev_fn(dt)
        cur = dt - datetime.timedelta(days=1)
        while not cal.isbizday(cur):
            cur -= datetime.timedelta(days=1)
        return cur
    return adj


def _generate_schedule_dates(
    start_date: datetime.date,
    maturity_date: datetime.date,
    payment_freq_months: int,
    cal: Any = None,
    date_adjustment: str = "modified_following",
) -> List[datetime.date]:
    """Gera sequência de datas [start_date, ..., maturity_date] regressivamente pelo período em meses."""
    if payment_freq_months <= 0:
        dates = [start_date, maturity_date]
    else:
        eom = _is_end_of_month(maturity_date)
        raw: List[datetime.date] = [maturity_date]
        k = 1
        while True:
            prev_dt = _add_months(maturity_date, -k * int(payment_freq_months), end_of_month=eom)
            if prev_dt <= start_date:
                break
            raw.append(prev_dt)
            k += 1
        raw.append(start_date)
        raw.reverse()
        dates = raw

    adj_norm = str(date_adjustment).strip().lower()
    if cal is None or adj_norm in ("unadjusted", "none"):
        return sorted(set(dates))

    out: List[datetime.date] = [dates[0]]
    for d in dates[1:]:
        if adj_norm == "following":
            adj = cal.following(d)
        else:
            adj = _modified_following(cal, d)
        if adj > out[-1]:
            out.append(adj)
    return out


class _ShiftedCurve(Curve):
    """Envoltório leve de deslocamento paralelo (+shift) preservando metadados de convenção."""

    def __init__(self, base_curve: Curve, shift: float, default_basis: float = 365.0):
        super().__init__()
        self._base = base_curve
        self._shift = float(shift)
        self._default_basis = float(default_basis)

    @property
    def interpolator(self):
        return getattr(self._base, "interpolator", None)

    @property
    def day_count_basis(self):
        b = getattr(self._base, "day_count_basis", None)
        if b is not None:
            return float(b)
        interp = self.interpolator
        if interp is not None and hasattr(interp, "day_count_basis"):
            return float(interp.day_count_basis)
        return None

    @property
    def compounding(self):
        c = getattr(self._base, "compounding", None)
        if c is not None:
            return c
        interp = self.interpolator
        if interp is not None and hasattr(interp, "compounding"):
            return interp.compounding
        return None

    @property
    def freq(self):
        f = getattr(self._base, "freq", None)
        if f is not None:
            return f
        interp = self.interpolator
        if interp is not None and hasattr(interp, "freq"):
            return interp.freq
        return 1

    def get_rate(self, t: float, t_sec: float = 0.0) -> float:
        return float(self._base.get_rate(t, t_sec)) + self._shift

    def discount_factor(self, t: float) -> float:
        if t <= 0.0:
            return 1.0
        basis = self.day_count_basis or self._default_basis
        tau = float(t) / float(basis)
        _, df_base = _curve_zero_rate_and_df(self._base, float(t), tau)
        return float(df_base) * math.exp(-self._shift * tau)


def _resolve_discount_factor(
    rate_or_curve: Union[float, Curve],
    base_dt: datetime.date,
    target_dt: datetime.date,
    cal: Any,
    day_count: str = "ACT/360",
    rate_compounding: str = "auto",
) -> Tuple[float, float]:
    """Calcula (taxa, fator_desconto) entre base_dt e target_dt em convenções onshore ou offshore."""
    if target_dt <= base_dt:
        return 0.0, 1.0
    dc_upper = str(day_count).strip().upper()
    is_bus252 = dc_upper in ("BUS/252", "DU/252")
    dc_days = float((target_dt - base_dt).days)
    if is_bus252:
        du_days = max(0, int(cal.bizdays(base_dt, target_dt)))
        tau = float(du_days) / 252.0
    else:
        du_days = 0
        tau = _year_fraction_bond(base_dt, target_dt, day_count, freq=1)

    comp = str(rate_compounding).strip().lower()
    if isinstance(rate_or_curve, Curve):
        if hasattr(rate_or_curve, "discount_factor"):
            return _curve_zero_rate_and_df(rate_or_curve, dc_days, tau)
        interp = getattr(rate_or_curve, "interpolator", None)
        basis_meta = getattr(rate_or_curve, "day_count_basis", None)
        if basis_meta is None and interp is not None:
            basis_meta = getattr(interp, "day_count_basis", None)
        if basis_meta is not None and abs(float(basis_meta) - 252.0) > 1e-6:
            return _curve_zero_rate_and_df(rate_or_curve, dc_days, tau)

        t_query = float(max(1, du_days)) if is_bus252 else float(max(1.0, dc_days))
        r_val = float(rate_or_curve.get_rate(t_query, dc_days))
    else:
        r_val = float(rate_or_curve)

    if comp == "auto":
        comp_eff = "exponential_252" if is_bus252 else "continuous"
    else:
        comp_eff = comp

    if comp_eff in ("continuous", "cont"):
        df = math.exp(-r_val * tau)
    elif comp_eff in ("simple", "simple_linear", "linear"):
        df = 1.0 / max(1e-12, 1.0 + r_val * tau)
    elif comp_eff in ("exponential_252", "exponential", "exponential_b3", "exp_252", "b3"):
        df = math.pow(1.0 + max(-0.99, r_val), -tau)
    else:
        raise ValueError(
            f"rate_compounding inválido: '{rate_compounding}'. "
            "Use 'auto', 'exponential_252', 'continuous' ou 'simple'."
        )
    return r_val, float(df)


def _curve_zero_rate_and_df(curve: Curve, dc: float, tau_fallback: float) -> Tuple[float, float]:
    """Avalia a curva zero offshore (SimpleCurve, ParametricCurve ou FlatCurve) no prazo dc."""
    if dc <= 0.0:
        return 0.0, 1.0
    if hasattr(curve, "discount_factor"):
        df = float(curve.discount_factor(float(dc)))
        r = float(curve.get_rate(float(dc), float(dc)))
        return r, df

    interp = getattr(curve, "interpolator", None)
    if interp is not None and hasattr(interp, "day_count_basis"):
        basis = float(interp.day_count_basis)
        comp = str(interp.compounding).strip().lower()
        freq = int(getattr(interp, "freq", 1))
        r = float(curve.get_rate(float(dc), float(dc)))
        tau = float(dc) / basis
        if comp in ("continuous", "cont"):
            df = math.exp(-r * tau)
        elif comp in ("simple", "simple_linear", "linear"):
            df = 1.0 / max(1e-12, 1.0 + r * tau)
        elif comp in ("exponential", "exponential_b3", "exp_252", "b3"):
            df = (1.0 + max(-0.99, r)) ** (-tau)
        else:
            k = float(max(1, freq))
            df = (1.0 + max(-0.99, r) / k) ** (-k * tau)
        return r, df

    r = float(curve.get_rate(float(dc), float(dc)))
    return r, math.exp(-r * max(0.0, tau_fallback))


def _coerce_cpi_series(cpi_history: Any) -> pd.Series:
    if isinstance(cpi_history, pd.Series):
        s = cpi_history.copy()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    elif isinstance(cpi_history, dict):
        s = pd.Series(list(cpi_history.values()), index=pd.to_datetime(list(cpi_history.keys())), dtype=float)
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    elif isinstance(cpi_history, pd.DataFrame):
        d_col = next((c for c in cpi_history.columns if str(c).lower() in ("date", "data", "month", "mes")), cpi_history.columns[0])
        v_col = next((c for c in cpi_history.columns if str(c).lower() in ("value", "valor", "cpi", "index", "indice")), cpi_history.columns[-1])
        s = pd.Series(cpi_history[v_col].astype(float).values, index=pd.to_datetime(cpi_history[d_col]).dt.to_period("M").dt.to_timestamp())
    else:
        raise TypeError(f"Formato não suportado para cpi_history: {type(cpi_history)}")
    return s[~s.index.duplicated(keep="last")].astype(float).sort_index()


def _interpolate_ust_par_curve(curve_or_points: Any, target_years: float, target_dc: float) -> float:
    if isinstance(curve_or_points, (int, float)):
        return float(curve_or_points)
    if hasattr(curve_or_points, "get_rate"):
        # Curva zerada ou par em dias corridos
        if hasattr(curve_or_points, "discount_factor") or (
            getattr(curve_or_points, "interpolator", None) is not None
            and hasattr(curve_or_points.interpolator, "day_count_basis")
        ):
            return float(curve_or_points.get_rate(float(target_dc), float(target_dc)))
        return float(curve_or_points.get_rate(float(target_dc)))
    if isinstance(curve_or_points, dict):
        pts = sorted((float(k), float(v)) for k, v in curve_or_points.items())
    elif isinstance(curve_or_points, (pd.Series, pd.DataFrame)):
        if isinstance(curve_or_points, pd.Series):
            pts = sorted((float(k), float(v)) for k, v in curve_or_points.items())
        else:
            x_c = next((c for c in curve_or_points.columns if str(c).lower() in ("years", "tenor", "prazo", "dc", "maturity")), curve_or_points.columns[0])
            y_c = next((c for c in curve_or_points.columns if str(c).lower() in ("yield", "rate", "taxa", "par")), curve_or_points.columns[-1])
            pts = sorted((float(r[x_c]), float(r[y_c])) for _, r in curve_or_points.iterrows())
    else:
        pts = sorted((float(x), float(y)) for x, y in curve_or_points)
    if not pts:
        raise ValueError("Série de rendimentos do benchmark Treasury não pode estar vazia.")
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    # Se eixo estiver em dias (> 50), usa target_dc; caso contrário, anos
    query_x = float(target_dc) if xs.max() > 50.0 else float(target_years)
    return float(np.interp(query_x, xs, ys))


class USTreasuryBill:
    """
    Treasury Bill dos EUA precificada integralmente offline em base ACT/360.
    Suporta conversão entre taxa de desconto bancário, preço limpo/sujo por 100 de face,
    rendimento Money Market, Bond Equivalent Yield (31 CFR Part 356) e desconto por curva zero.
    """

    def __init__(
        self,
        maturity_date: Union[datetime.date, str],
        face_value: float = 100.0,
        calendar_name: str = "US",
        holidays: Any = None,
    ):
        self.maturity_date = _to_date(maturity_date)
        self.face_value = float(face_value)
        self.calendar = get_calendar(calendar_name, holidays=holidays)

    def settlement_date(self, trade_date: Union[datetime.date, str], settlement_days: int = 1) -> datetime.date:
        return self.calendar.offset(_to_date(trade_date), int(settlement_days))

    def dc(self, base_date: Union[datetime.date, str]) -> int:
        return max(0, int((self.maturity_date - _to_date(base_date)).days))

    def price_per_100(self, base_date: Union[datetime.date, str], discount_rate: float) -> float:
        dc_val = self.dc(base_date)
        if dc_val <= 0:
            return 100.0
        return 100.0 * (1.0 - float(discount_rate) * (dc_val / 360.0))

    def discount_rate_from_price(self, base_date: Union[datetime.date, str], price_per_100: float) -> float:
        dc_val = self.dc(base_date)
        if dc_val <= 0:
            return 0.0
        return (1.0 - float(price_per_100) / 100.0) * (360.0 / float(dc_val))

    def money_market_yield(self, base_date: Union[datetime.date, str], discount_rate: float) -> float:
        dc_val = self.dc(base_date)
        if dc_val <= 0:
            return 0.0
        denom = 1.0 - float(discount_rate) * (dc_val / 360.0)
        if denom <= 0.0:
            raise ValueError("Taxa de desconto resulta em preço não positivo para o prazo informado.")
        return float(discount_rate) / denom

    def bond_equivalent_yield(self, base_date: Union[datetime.date, str], discount_rate: float) -> float:
        dc_val = self.dc(base_date)
        if dc_val <= 0:
            return 0.0
        p100 = self.price_per_100(base_date, discount_rate)
        if p100 <= 0.0:
            raise ValueError("Preço implícito não positivo para cálculo de BEY.")
        if dc_val <= 182:
            return ((100.0 - p100) / p100) * (365.0 / float(dc_val))
        t = float(dc_val) / 365.0
        disc_term = t * t - (2.0 * t - 1.0) * (1.0 - 100.0 / p100)
        if disc_term < 0.0:
            raise ValueError("Não há raiz real para o Bond Equivalent Yield neste preço.")
        return (-2.0 * t + 2.0 * math.sqrt(disc_term)) / (2.0 * t - 1.0)

    def cashflow_table(
        self,
        base_date: Union[datetime.date, str],
        discount_rate: Optional[float] = None,
        discount_curve: Optional[Curve] = None,
    ) -> pd.DataFrame:
        base_dt = _to_date(base_date)
        dc_val = self.dc(base_dt)
        if dc_val <= 0:
            return pd.DataFrame(columns=["data_fluxo", "dc", "tipo", "fluxo_nominal", "taxa", "fator_desconto", "valor_presente"])

        if discount_curve is not None:
            r_val, df_val = _curve_zero_rate_and_df(discount_curve, float(dc_val), dc_val / 360.0)
        elif discount_rate is not None:
            r_val = float(discount_rate)
            df_val = max(0.0, 1.0 - r_val * (dc_val / 360.0))
        else:
            raise ValueError("Informe 'discount_rate' ou 'discount_curve'.")

        pv = self.face_value * df_val
        return pd.DataFrame([{
            "data_fluxo": self.maturity_date,
            "dc": dc_val,
            "tipo": "Principal",
            "fluxo_nominal": self.face_value,
            "taxa": r_val,
            "fator_desconto": df_val,
            "valor_presente": pv,
        }])

    def present_value(
        self,
        base_date: Union[datetime.date, str],
        discount_rate: Optional[float] = None,
        discount_curve: Optional[Curve] = None,
    ) -> float:
        dc_val = self.dc(base_date)
        if dc_val <= 0:
            return self.face_value
        if discount_curve is not None:
            _, df_val = _curve_zero_rate_and_df(discount_curve, float(dc_val), dc_val / 360.0)
            return self.face_value * df_val
        if discount_rate is not None:
            return self.face_value * (self.price_per_100(base_date, discount_rate) / 100.0)
        raise ValueError("Informe 'discount_rate' ou 'discount_curve'.")

    def dv01(self, base_date: Union[datetime.date, str], discount_rate: float) -> float:
        """Sensibilidade do PV por 1 bp na taxa de desconto bancário ACT/360."""
        dc_val = self.dc(base_date)
        if dc_val <= 0:
            return 0.0
        return self.face_value * (dc_val / 360.0) * 1e-4

    def risk_metrics(self, base_date: Union[datetime.date, str], discount_rate: float) -> Dict[str, float]:
        dc_val = self.dc(base_date)
        p100 = self.price_per_100(base_date, discount_rate)
        bey = self.bond_equivalent_yield(base_date, discount_rate)
        mm = self.money_market_yield(base_date, discount_rate)
        pv = self.face_value * (p100 / 100.0)
        return {
            "dc": float(dc_val),
            "price_per_100": float(p100),
            "present_value": float(pv),
            "discount_rate": float(discount_rate),
            "money_market_yield": float(mm),
            "bond_equivalent_yield": float(bey),
            "dv01_discount": float(self.dv01(base_date, discount_rate)),
        }


class USTreasuryNoteBond:
    """
    US Treasury Note ou Bond prefixada com cupons semestrais em convenção ACT/ACT ICMA.
    Suporta preço limpo/sujo, juros acruados exatos, Street YTM, duration, convexidade e curva zero.
    """

    def __init__(
        self,
        maturity_date: Union[datetime.date, str],
        coupon_rate: float,
        issue_date: Optional[Union[datetime.date, str]] = None,
        face_value: float = 100.0,
        freq: int = 2,
        day_count: str = "ACT/ACT_ICMA",
        calendar_name: str = "US",
        end_of_month: Optional[bool] = None,
        holidays: Any = None,
    ):
        if int(freq) not in (1, 2, 4, 12):
            raise ValueError("freq deve ser 1, 2, 4 ou 12.")
        self.maturity_date = _to_date(maturity_date)
        self.issue_date = _to_date(issue_date) if issue_date is not None else None
        self.coupon_rate = float(coupon_rate)
        self.face_value = float(face_value)
        self.freq = int(freq)
        self.day_count = _normalize_day_count(day_count)
        self.calendar = get_calendar(calendar_name, holidays=holidays)
        self.end_of_month = bool(end_of_month) if end_of_month is not None else _is_end_of_month(self.maturity_date)

    def settlement_date(self, trade_date: Union[datetime.date, str], settlement_days: int = 1) -> datetime.date:
        return self.calendar.offset(_to_date(trade_date), int(settlement_days))

    def _full_nominal_schedule(self, base_date: datetime.date) -> List[datetime.date]:
        """Gera a sequência regressiva de datas contratuais cobrindo [prev_cpn, ..., maturity]."""
        step_months = 12 // self.freq
        dates: List[datetime.date] = []
        k = 0
        lower_bound = min(base_date, self.issue_date) if self.issue_date is not None else base_date
        while True:
            dt = _add_months(self.maturity_date, -k * step_months, end_of_month=self.end_of_month)
            dates.append(dt)
            if dt <= lower_bound and dt <= base_date:
                break
            k += 1
            if k > 1200:
                raise RuntimeError("Loop excedido ao gerar agenda de cupons.")
        dates.reverse()
        return dates

    def _period_bounds(self, base_date: datetime.date) -> Tuple[datetime.date, List[datetime.date]]:
        sched = self._full_nominal_schedule(base_date)
        future = [d for d in sched if d > base_date]
        past_or_eq = [d for d in sched if d <= base_date]
        prev_cpn = past_or_eq[-1] if past_or_eq else sched[0]
        return prev_cpn, future

    def get_coupon_dates(self, base_date: Union[datetime.date, str]) -> List[datetime.date]:
        _, future = self._period_bounds(_to_date(base_date))
        return future

    def payment_date(self, nominal_date: datetime.date) -> datetime.date:
        return self.calendar.following(nominal_date)

    def fractional_period(self, base_date: Union[datetime.date, str]) -> float:
        base_dt = _to_date(base_date)
        if base_dt >= self.maturity_date:
            return 0.0
        prev_cpn, future = self._period_bounds(base_dt)
        next_cpn = future[0]
        if self.issue_date is not None and base_dt < self.issue_date:
            return 1.0
        if self.day_count == "ACT/ACT_ICMA":
            span = (next_cpn - prev_cpn).days
            return float((next_cpn - base_dt).days) / float(span) if span > 0 else 0.0
        acc_yf = _year_fraction_bond(prev_cpn, base_dt, self.day_count, self.freq, prev_cpn, next_cpn)
        return max(0.0, 1.0 - float(self.freq) * acc_yf)

    def accrued_interest(self, base_date: Union[datetime.date, str]) -> float:
        base_dt = _to_date(base_date)
        if base_dt >= self.maturity_date or (self.issue_date is not None and base_dt <= self.issue_date):
            return 0.0
        prev_cpn, future = self._period_bounds(base_dt)
        next_cpn = future[0]
        yf_acc = _year_fraction_bond(prev_cpn, base_dt, self.day_count, self.freq, prev_cpn, next_cpn)
        return self.face_value * self.coupon_rate * yf_acc

    def _cashflow_times_and_amounts(self, base_dt: datetime.date) -> Tuple[List[datetime.date], List[float], List[float]]:
        prev_cpn, future = self._period_bounds(base_dt)
        if not future:
            return [], [], []
        cpn_amt = self.face_value * (self.coupon_rate / float(self.freq))
        w = self.fractional_period(base_dt)
        taus: List[float] = []
        amounts: List[float] = []
        for idx, d in enumerate(future):
            prev_d = prev_cpn if idx == 0 else future[idx - 1]
            if self.day_count == "ACT/ACT_ICMA":
                tau_k = (float(idx) + w) / float(self.freq)
                amt_cpn = cpn_amt
            elif self.day_count in ("ACT/365", "ACT/360"):
                tau_k = _year_fraction_bond(base_dt, d, self.day_count, self.freq, prev_d, d)
                amt_cpn = self.face_value * self.coupon_rate * _year_fraction_bond(prev_d, d, self.day_count, self.freq, prev_d, d)
            else:
                tau_k = _year_fraction_bond(base_dt, d, self.day_count, self.freq, prev_d, d)
                amt_cpn = cpn_amt
            amt = amt_cpn + (self.face_value if idx == len(future) - 1 else 0.0)
            taus.append(tau_k)
            amounts.append(amt)
        return future, taus, amounts

    def dirty_price(self, base_date: Union[datetime.date, str], ytm: float) -> float:
        base_dt = _to_date(base_date)
        if base_dt >= self.maturity_date:
            return self.face_value
        _, taus, amounts = self._cashflow_times_and_amounts(base_dt)
        y = float(ytm)
        f = float(self.freq)
        if 1.0 + y / f <= 0.0:
            raise ValueError("Rendimento inválido: 1 + ytm/freq deve ser positivo.")
        total = 0.0
        for tau_k, amt in zip(taus, amounts):
            total += amt / math.pow(1.0 + y / f, f * tau_k)
        return float(total)

    def clean_price(self, base_date: Union[datetime.date, str], ytm: float) -> float:
        return self.dirty_price(base_date, ytm) - self.accrued_interest(base_date)

    def present_value(
        self,
        base_date: Union[datetime.date, str],
        yield_or_curve: Union[float, Curve],
        return_cashflows: bool = False,
    ) -> Union[float, Tuple[float, pd.DataFrame]]:
        df_cf = self.cashflow_table(base_date, yield_or_curve)
        pv = self.face_value if df_cf.empty else float(df_cf["valor_presente"].sum())
        if return_cashflows:
            return pv, df_cf
        return pv

    def cashflow_table(
        self,
        base_date: Union[datetime.date, str],
        yield_or_curve: Union[float, Curve],
    ) -> pd.DataFrame:
        base_dt = _to_date(base_date)
        future, taus, amounts = self._cashflow_times_and_amounts(base_dt)
        if not future:
            return pd.DataFrame(
                columns=["data_nominal", "data_pagamento", "dc", "tau", "tipo", "fluxo_nominal", "taxa", "fator_desconto", "valor_presente"]
            )
        rows = []
        is_crv = isinstance(yield_or_curve, Curve)
        f = float(self.freq)
        for idx, (d_nom, tau_k, amt) in enumerate(zip(future, taus, amounts)):
            dc_k = (d_nom - base_dt).days
            if is_crv:
                r_k, df_k = _curve_zero_rate_and_df(yield_or_curve, float(dc_k), tau_k)
            else:
                r_k = float(yield_or_curve)
                df_k = math.pow(1.0 + r_k / f, -f * tau_k)
            pv_k = amt * df_k
            rows.append({
                "data_nominal": d_nom,
                "data_pagamento": self.payment_date(d_nom),
                "dc": dc_k,
                "tau": tau_k,
                "tipo": "Cupom + Principal" if idx == len(future) - 1 else "Cupom",
                "fluxo_nominal": amt,
                "taxa": r_k,
                "fator_desconto": df_k,
                "valor_presente": pv_k,
            })
        return pd.DataFrame(rows)

    def ytm_from_dirty_price(self, base_date: Union[datetime.date, str], target_dirty_price: float, xtol: float = 1e-12) -> float:
        base_dt = _to_date(base_date)
        if base_dt >= self.maturity_date:
            return 0.0
        target = float(target_dirty_price)
        if target <= 0.0:
            raise ValueError("target_dirty_price deve ser estritamente positivo.")

        def _obj(y: float) -> float:
            return self.dirty_price(base_dt, y) - target

        return float(brentq(_obj, -0.50, 2.0, xtol=xtol))

    def ytm_from_clean_price(self, base_date: Union[datetime.date, str], target_clean_price: float, xtol: float = 1e-12) -> float:
        dirty = float(target_clean_price) + self.accrued_interest(base_date)
        return self.ytm_from_dirty_price(base_date, dirty, xtol=xtol)

    def macaulay_duration(self, base_date: Union[datetime.date, str], ytm: float) -> float:
        base_dt = _to_date(base_date)
        if base_dt >= self.maturity_date:
            return 0.0
        _, taus, amounts = self._cashflow_times_and_amounts(base_dt)
        y = float(ytm)
        f = float(self.freq)
        p_dirty = 0.0
        weighted = 0.0
        for tau_k, amt in zip(taus, amounts):
            pv_k = amt / math.pow(1.0 + y / f, f * tau_k)
            p_dirty += pv_k
            weighted += tau_k * pv_k
        return weighted / p_dirty if p_dirty > 0.0 else 0.0

    def modified_duration(self, base_date: Union[datetime.date, str], ytm: float) -> float:
        d_mac = self.macaulay_duration(base_date, ytm)
        return d_mac / (1.0 + float(ytm) / float(self.freq))

    def dv01(self, base_date: Union[datetime.date, str], ytm: float) -> float:
        base_dt = _to_date(base_date)
        if base_dt >= self.maturity_date:
            return 0.0
        return self.dirty_price(base_dt, ytm) * self.modified_duration(base_dt, ytm) * 1e-4

    def convexity(self, base_date: Union[datetime.date, str], ytm: float) -> float:
        base_dt = _to_date(base_date)
        if base_dt >= self.maturity_date:
            return 0.0
        _, taus, amounts = self._cashflow_times_and_amounts(base_dt)
        y = float(ytm)
        f = float(self.freq)
        base_fac = 1.0 + y / f
        p_dirty = 0.0
        acc = 0.0
        for tau_k, amt in zip(taus, amounts):
            pv_k = amt / math.pow(base_fac, f * tau_k)
            p_dirty += pv_k
            acc += tau_k * (tau_k + 1.0 / f) * pv_k
        return acc / (p_dirty * base_fac * base_fac) if p_dirty > 0.0 else 0.0

    def risk_metrics(self, base_date: Union[datetime.date, str], ytm: float) -> Dict[str, float]:
        ai = self.accrued_interest(base_date)
        dirty = self.dirty_price(base_date, ytm)
        clean = dirty - ai
        d_mac = self.macaulay_duration(base_date, ytm)
        d_mod = self.modified_duration(base_date, ytm)
        return {
            "clean_price": float(clean),
            "accrued_interest": float(ai),
            "dirty_price": float(dirty),
            "ytm": float(ytm),
            "macaulay_duration": float(d_mac),
            "modified_duration": float(d_mod),
            "dv01": float(dirty * d_mod * 1e-4),
            "convexity": float(self.convexity(base_date, ytm)),
        }


class TIPS(USTreasuryNoteBond):
    """
    Treasury Inflation-Protected Securities (TIPS) dos EUA conforme 31 CFR Part 356:
    interpolação linear da Referência de CPI-U com defasagem 3M/2M e piso contra deflação no vencimento.
    """

    def __init__(
        self,
        maturity_date: Union[datetime.date, str],
        coupon_rate: float,
        issue_cpi: float,
        issue_date: Optional[Union[datetime.date, str]] = None,
        face_value: float = 100.0,
        calendar_name: str = "US",
        end_of_month: Optional[bool] = None,
        round_ref_cpi: bool = True,
        round_index_ratio: bool = True,
        holidays: Any = None,
    ):
        super().__init__(
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            issue_date=issue_date,
            face_value=face_value,
            freq=2,
            day_count="ACT/ACT_ICMA",
            calendar_name=calendar_name,
            end_of_month=end_of_month,
            holidays=holidays,
        )
        if float(issue_cpi) <= 0.0:
            raise ValueError("issue_cpi deve ser estritamente positivo.")
        self.issue_cpi = float(issue_cpi)
        self.round_ref_cpi = bool(round_ref_cpi)
        self.round_index_ratio = bool(round_index_ratio)

    @staticmethod
    def _month_offset(d: datetime.date, delta_months: int) -> pd.Timestamp:
        total_m = (d.year * 12 + (d.month - 1)) + delta_months
        return pd.Timestamp(year=total_m // 12, month=(total_m % 12) + 1, day=1)

    def reference_cpi(self, target_date: Union[datetime.date, str], cpi_history: Any) -> float:
        """Calcula o Reference CPI na data por interpolação linear entre os meses t-3 e t-2."""
        dt = _to_date(target_date)
        s_cpi = _coerce_cpi_series(cpi_history)
        ts_m3 = self._month_offset(dt, -3)
        ts_m2 = self._month_offset(dt, -2)
        if ts_m3 not in s_cpi.index or ts_m2 not in s_cpi.index:
            raise KeyError(
                f"Séries de CPI ausentes para {ts_m3.strftime('%Y-%m')} e {ts_m2.strftime('%Y-%m')}."
            )
        cpi_m3 = float(s_cpi.loc[ts_m3])
        cpi_m2 = float(s_cpi.loc[ts_m2])
        days_in_month = float(calendar.monthrange(dt.year, dt.month)[1])
        ref = cpi_m3 + ((dt.day - 1) / days_in_month) * (cpi_m2 - cpi_m3)
        return round(ref, 5) if self.round_ref_cpi else float(ref)

    def index_ratio(
        self,
        base_date: Union[datetime.date, str],
        ref_cpi: Optional[float] = None,
        cpi_history: Any = None,
    ) -> float:
        if ref_cpi is None:
            if cpi_history is None:
                raise ValueError("Informe 'ref_cpi', 'cpi_history' ou 'index_ratio' diretamente.")
            ref_val = self.reference_cpi(base_date, cpi_history)
        else:
            ref_val = float(ref_cpi)
        ir = ref_val / self.issue_cpi
        return round(ir, 5) if self.round_index_ratio else float(ir)

    def _resolve_ir(
        self,
        base_date: Union[datetime.date, str],
        index_ratio: Optional[float] = None,
        ref_cpi: Optional[float] = None,
        cpi_history: Any = None,
    ) -> float:
        if index_ratio is not None:
            return round(float(index_ratio), 5) if self.round_index_ratio else float(index_ratio)
        if ref_cpi is not None or cpi_history is not None:
            return self.index_ratio(base_date, ref_cpi=ref_cpi, cpi_history=cpi_history)
        return 1.0

    def adjusted_accrued_interest(
        self,
        base_date: Union[datetime.date, str],
        index_ratio: Optional[float] = None,
        ref_cpi: Optional[float] = None,
        cpi_history: Any = None,
    ) -> float:
        ir = self._resolve_ir(base_date, index_ratio=index_ratio, ref_cpi=ref_cpi, cpi_history=cpi_history)
        return self.accrued_interest(base_date) * ir

    def adjusted_clean_price(
        self,
        base_date: Union[datetime.date, str],
        real_ytm: float,
        index_ratio: Optional[float] = None,
        ref_cpi: Optional[float] = None,
        cpi_history: Any = None,
    ) -> float:
        ir = self._resolve_ir(base_date, index_ratio=index_ratio, ref_cpi=ref_cpi, cpi_history=cpi_history)
        return self.clean_price(base_date, real_ytm) * ir

    def adjusted_dirty_price(
        self,
        base_date: Union[datetime.date, str],
        real_ytm: float,
        index_ratio: Optional[float] = None,
        ref_cpi: Optional[float] = None,
        cpi_history: Any = None,
    ) -> float:
        ir = self._resolve_ir(base_date, index_ratio=index_ratio, ref_cpi=ref_cpi, cpi_history=cpi_history)
        return self.dirty_price(base_date, real_ytm) * ir

    def deflation_floor_principal(
        self,
        terminal_index_ratio: Optional[float] = None,
        terminal_ref_cpi: Optional[float] = None,
        cpi_history: Any = None,
    ) -> float:
        """Retorna o principal corrigido no vencimento com piso par (max(Index Ratio, 1))."""
        if terminal_index_ratio is not None:
            ir_t = float(terminal_index_ratio)
        elif terminal_ref_cpi is not None:
            ir_t = float(terminal_ref_cpi) / self.issue_cpi
        elif cpi_history is not None:
            ir_t = self.index_ratio(self.maturity_date, cpi_history=cpi_history)
        else:
            ir_t = 1.0
        return self.face_value * max(1.0, ir_t)

    def real_ytm_from_clean_price(
        self,
        base_date: Union[datetime.date, str],
        real_clean_price: float,
        xtol: float = 1e-12,
    ) -> float:
        return self.ytm_from_clean_price(base_date, real_clean_price, xtol=xtol)

    def risk_metrics(
        self,
        base_date: Union[datetime.date, str],
        real_ytm: float,
        index_ratio: Optional[float] = None,
        ref_cpi: Optional[float] = None,
        cpi_history: Any = None,
    ) -> Dict[str, float]:
        ir = self._resolve_ir(base_date, index_ratio=index_ratio, ref_cpi=ref_cpi, cpi_history=cpi_history)
        real_m = super().risk_metrics(base_date, real_ytm)
        return {
            "index_ratio": float(ir),
            "real_clean_price": real_m["clean_price"],
            "real_accrued_interest": real_m["accrued_interest"],
            "real_dirty_price": real_m["dirty_price"],
            "adjusted_clean_price": real_m["clean_price"] * ir,
            "adjusted_accrued_interest": real_m["accrued_interest"] * ir,
            "adjusted_dirty_price": real_m["dirty_price"] * ir,
            "real_ytm": float(real_ytm),
            "real_macaulay_duration": real_m["macaulay_duration"],
            "real_modified_duration": real_m["modified_duration"],
            "real_dv01": real_m["dv01"],
            "adjusted_dv01": real_m["dv01"] * ir,
            "real_convexity": real_m["convexity"],
        }


class OffshoreBond(USTreasuryNoteBond):
    """
    Título Corporativo ou Soberano Global/Eurobond offline.
    Suporta convenções 30/360, ACT/ACT_ICMA, ACT/365 e ACT/360, G-Spread sobre curva
    Treasury, Z-Spread sobre curvas OIS offshore e valor presente ajustado ao risco de crédito.
    """

    def __init__(
        self,
        maturity_date: Union[datetime.date, str],
        coupon_rate: float,
        issue_date: Optional[Union[datetime.date, str]] = None,
        face_value: float = 100.0,
        freq: int = 2,
        day_count: str = "30/360",
        calendar_name: str = "US",
        end_of_month: Optional[bool] = None,
        holidays: Any = None,
    ):
        super().__init__(
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            issue_date=issue_date,
            face_value=face_value,
            freq=freq,
            day_count=day_count,
            calendar_name=calendar_name,
            end_of_month=end_of_month,
            holidays=holidays,
        )

    def g_spread(
        self,
        base_date: Union[datetime.date, str],
        ytm_or_clean_price: float,
        treasury_benchmark: Any,
        price_input: bool = False,
    ) -> float:
        """Calcula o G-Spread (YTM do título menos a taxa Treasury interpolada no prazo)."""
        base_dt = _to_date(base_date)
        bond_ytm = (
            self.ytm_from_clean_price(base_dt, ytm_or_clean_price)
            if price_input
            else float(ytm_or_clean_price)
        )
        dc_mat = float(max(1, (self.maturity_date - base_dt).days))
        years_mat = dc_mat / 365.25
        bench_y = _interpolate_ust_par_curve(treasury_benchmark, years_mat, dc_mat)
        return float(bond_ytm - bench_y)

    def z_spread(
        self,
        base_date: Union[datetime.date, str],
        target_price: float,
        zero_curve: Curve,
        clean_input: bool = True,
        spread_compounding: str = "continuous",
        xtol: float = 1e-11,
    ) -> float:
        """
        Calcula o Z-Spread aditivo constante sobre a curva zero offshore (ex.: SOFR_OIS ou ESTR_OIS)
        de forma que a soma dos fluxos descontados iguale o preço sujo alvo.
        """
        base_dt = _to_date(base_date)
        target_dirty = float(target_price) + (self.accrued_interest(base_dt) if clean_input else 0.0)
        if target_dirty <= 0.0:
            raise ValueError("Preço alvo sujo deve ser estritamente positivo.")

        df_cf = self.cashflow_table(base_dt, zero_curve)
        if df_cf.empty:
            return 0.0

        curve_basis = getattr(zero_curve, "day_count_basis", None)
        if curve_basis is None:
            interp = getattr(zero_curve, "interpolator", None)
            if interp is not None and hasattr(interp, "day_count_basis"):
                curve_basis = float(interp.day_count_basis)
        if curve_basis is not None and float(curve_basis) > 0.0:
            taus = df_cf["dc"].to_numpy(dtype=np.float64) / float(curve_basis)
        else:
            taus = df_cf["tau"].to_numpy(dtype=np.float64)
        cfs = df_cf["fluxo_nominal"].to_numpy(dtype=np.float64)
        base_dfs = df_cf["fator_desconto"].to_numpy(dtype=np.float64)
        comp = str(spread_compounding).strip().lower()
        f = float(self.freq)

        def _pv_err(z: float) -> float:
            if comp in ("continuous", "cont"):
                adj = np.exp(-float(z) * taus)
            else:
                adj = np.power(1.0 + float(z) / f, -f * taus)
            return float(np.sum(cfs * base_dfs * adj) - target_dirty)

        return float(brentq(_pv_err, -0.50, 1.50, xtol=xtol))

    def risky_present_value(
        self,
        base_date: Union[datetime.date, str],
        risk_free_rate_or_curve: Union[float, Curve],
        hazard_curve: Any = None,
        hazard_rate: Optional[float] = None,
        recovery_rate: float = 0.40,
    ) -> Dict[str, float]:
        """Calcula o Valor Presente ajustado ao risco de crédito em convenção de anos offshore."""
        base_dt = _to_date(base_date)
        df_cf = self.cashflow_table(base_dt, risk_free_rate_or_curve)
        if df_cf.empty:
            return {"pv_risky": 0.0, "pv_risk_free": 0.0, "cva": 0.0, "expected_loss_pct": 0.0, "survival_at_maturity": 1.0}

        taus = df_cf["tau"].to_numpy(dtype=np.float64)
        cfs = df_cf["fluxo_nominal"].to_numpy(dtype=np.float64)
        dfs = df_cf["fator_desconto"].to_numpy(dtype=np.float64)

        if hazard_curve is not None:
            recovery_rate = float(hazard_curve.recovery_rate)
            # Avalia probabilidade de sobrevivência pelo tempo efetivo tau em base 252 equivalente do objeto
            q_arr = np.array([hazard_curve.survival_prob(int(round(t * 252.0))) for t in taus], dtype=np.float64)
        else:
            lam = 0.0 if hazard_rate is None else max(0.0, float(hazard_rate))
            q_arr = np.exp(-lam * taus)

        outstanding_arr = np.full(len(cfs), self.face_value, dtype=np.float64)
        pv_r, pv_rf, cva = _cy_risky_bond_pv(
            cfs, dfs, q_arr, float(recovery_rate), float(self.face_value), outstanding_arr
        )

        return {
            "pv_risky": float(pv_r),
            "pv_risk_free": float(pv_rf),
            "cva": float(cva),
            "expected_loss_pct": float((cva / pv_rf) * 100.0) if pv_rf > 0.0 else 0.0,
            "survival_at_maturity": float(q_arr[-1]),
        }

    def implied_hazard_rate(
        self,
        base_date: Union[datetime.date, str],
        market_dirty_price: float,
        risk_free_rate_or_curve: Union[float, Curve],
        recovery_rate: float = 0.40,
        xtol: float = 1e-10,
    ) -> float:
        def _obj(lam: float) -> float:
            return self.risky_present_value(
                base_date, risk_free_rate_or_curve, hazard_rate=lam, recovery_rate=recovery_rate
            )["pv_risky"] - float(market_dirty_price)

        f0 = _obj(0.0)
        if abs(f0) <= 1e-10:
            return 0.0
        if f0 < -1e-10:
            return float("nan")
        high = 2.0
        while _obj(high) > 0.0 and high < 20.0:
            high *= 2.0
        return float(brentq(_obj, 0.0, high, xtol=xtol))
