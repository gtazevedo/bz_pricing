import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from bz_pricing.core.options_core import (
    black76_price_greeks as _cy_black76,
    norm_cdf as _cy_norm_cdf,
    norm_ppf as _cy_norm_ppf,
    sabr_implied_vol as _cy_sabr_vol,
    svi_eval_vec as _cy_svi_eval_vec,
    svi_total_variance as _cy_svi_total_var,
)


def delta_to_strike(delta_fwd: float, forward_price: float, vol: float, T: float) -> float:
    """
    Conversão de Delta forward (Delta_F = N(d1)) para Strike (padrão B3):
        K = A * exp(0.5 * sigma^2 * T - N^-1(Delta_F) * sigma * sqrt(T))
    """
    d_clamped = min(0.9999, max(0.0001, float(delta_fwd)))
    inv_n = _cy_norm_ppf(d_clamped)
    return float(forward_price) * math.exp(0.5 * vol * vol * T - inv_n * vol * math.sqrt(max(1e-12, T)))


def strike_to_delta(strike: float, forward_price: float, vol: float, T: float) -> float:
    """Conversão inversa de Strike para Delta forward N(d1)."""
    if T <= 0.0 or vol <= 0.0:
        return 1.0 if forward_price > strike else 0.0
    d1 = (math.log(float(forward_price) / float(strike)) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    return float(_cy_norm_cdf(d1))


def monotone_cubic_interp(x_nodes: Sequence[float], y_nodes: Sequence[float], x_eval: float) -> float:
    """
    Interpolação por spline cúbico monótono de Fritsch-Carlson (padrão B3 para smile de
    volatilidade e estrutura temporal de IDI).
    """
    x = np.asarray(x_nodes, dtype=np.float64)
    y = np.asarray(y_nodes, dtype=np.float64)
    n = len(x)
    if n == 0:
        raise ValueError("Nós de interpolação vazios.")
    if n == 1 or x_eval <= x[0]:
        return float(y[0])
    if x_eval >= x[-1]:
        return float(y[-1])

    d = np.diff(y) / np.diff(x)
    m = np.zeros(n, dtype=np.float64)
    m[0] = d[0]
    m[-1] = d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] > 0.0:
            m[i] = 0.5 * (d[i - 1] + d[i])
        else:
            m[i] = 0.0

    for i in range(n - 1):
        if abs(d[i]) < 1e-15:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            alpha = m[i] / d[i]
            beta = m[i + 1] / d[i]
            cond_ok = (
                (alpha + beta - 2.0 <= 0.0)
                or (2.0 * alpha + beta - 3.0 <= 0.0)
                or (alpha + 2.0 * beta - 3.0 <= 0.0)
                or (alpha - ((2.0 * alpha + beta - 3.0) ** 2) / (3.0 * (alpha + beta - 2.0)) >= 0.0)
            )
            if not cond_ok:
                tau = 3.0 / math.sqrt(alpha * alpha + beta * beta)
                m[i] = tau * alpha * d[i]
                m[i + 1] = tau * beta * d[i]

    idx = int(np.searchsorted(x, x_eval, side="right") - 1)
    idx = max(0, min(n - 2, idx))
    h = x[idx + 1] - x[idx]
    t = (x_eval - x[idx]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return float(y[idx] * h00 + h * m[idx] * h10 + y[idx + 1] * h01 + h * m[idx + 1] * h11)


def exp_smile_interp(x_nodes: Sequence[float], y_nodes: Sequence[float], x_eval: float) -> float:
    """Interpolação exponencial por partes entre vértices do smile."""
    x = np.asarray(x_nodes, dtype=np.float64)
    y = np.asarray(y_nodes, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("Nós de interpolação vazios.")
    if len(x) == 1 or x_eval <= x[0]:
        return float(y[0])
    if x_eval >= x[-1]:
        return float(y[-1])
    idx = int(np.searchsorted(x, x_eval, side="right") - 1)
    idx = max(0, min(len(x) - 2, idx))
    w = (x_eval - x[idx]) / (x[idx + 1] - x[idx])
    return float(y[idx] * ((y[idx + 1] / y[idx]) ** w))


class SVISlice:
    """
    Fatia temporal parametrizada por SVI (Stochastic Volatility Inspired - Gatheral / B3 / Bloomberg):
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    onde k = ln(K / F_n) e w(k) = sigma_BS^2 * T.
    """

    def __init__(self, a: float, b: float, rho: float, m: float, sigma: float, T: float):
        self.a = float(a)
        self.b = float(b)
        self.rho = float(rho)
        self.m = float(m)
        self.sigma = float(sigma)
        self.T = float(T)

    def total_variance(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(k, np.ndarray):
            return _cy_svi_eval_vec(k.astype(np.float64), self.a, self.b, self.rho, self.m, self.sigma)
        return float(_cy_svi_total_var(float(k), self.a, self.b, self.rho, self.m, self.sigma))

    def implied_vol(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        w = self.total_variance(k)
        t_eff = max(1e-8, self.T)
        if isinstance(w, np.ndarray):
            return np.sqrt(np.maximum(w, 1e-12) / t_eff)
        return math.sqrt(max(w, 1e-12) / t_eff)

    @classmethod
    def fit(
        cls,
        k_arr: Sequence[float],
        vols_or_w: Sequence[float],
        T: float,
        weights: Optional[Sequence[float]] = None,
        input_is_total_var: bool = False,
    ) -> "SVISlice":
        """Calibra os 5 parâmetros (a, b, rho, m, sigma) por mínimos quadrados não lineares."""
        k_np = np.asarray(k_arr, dtype=np.float64)
        vals = np.asarray(vols_or_w, dtype=np.float64)
        w_target = vals if input_is_total_var else (vals * vals * float(T))
        wt = np.ones_like(k_np) if weights is None else np.sqrt(np.asarray(weights, dtype=np.float64))

        w_min = float(np.min(w_target))
        x0 = np.array([max(1e-4, w_min * 0.8), 0.1, -0.3, 0.0, 0.1], dtype=np.float64)
        lb = np.array([-0.5, 1e-5, -0.999, -2.0, 1e-4], dtype=np.float64)
        ub = np.array([2.0, 5.0, 0.999, 2.0, 2.0], dtype=np.float64)

        def _res(params):
            a, b, rho, m, sig = params
            w_pred = _cy_svi_eval_vec(k_np, float(a), float(b), float(rho), float(m), float(sig))
            # Penaliza eventual variância mínima negativa
            min_w = a + b * sig * math.sqrt(max(1e-8, 1.0 - rho * rho))
            pen = max(0.0, 1e-6 - min_w) * 100.0
            return np.append(wt * (w_pred - w_target), pen)

        res = least_squares(_res, x0, bounds=(lb, ub), max_nfev=1000)
        a, b, rho, m, sig = [float(v) for v in res.x]
        return cls(a, b, rho, m, sig, float(T))


class SABRSlice:
    """
    Fatia temporal parametrizada pelo modelo SABR (Hagan et al., 2002), adotado pela B3
    para construção de smiles de opções sobre juros (DI1 / IDI).
    """

    def __init__(self, alpha: float, beta: float, rho: float, nu: float, F: float, T: float):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.nu = float(nu)
        self.F = float(F)
        self.T = float(T)

    def implied_vol(self, K: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(K, np.ndarray):
            return np.array([float(_cy_sabr_vol(self.F, float(k), self.T, self.alpha, self.beta, self.rho, self.nu)) for k in K])
        return float(_cy_sabr_vol(self.F, float(K), self.T, self.alpha, self.beta, self.rho, self.nu))

    @classmethod
    def fit(
        cls,
        F: float,
        K_arr: Sequence[float],
        vols: Sequence[float],
        T: float,
        beta: float = 0.5,
    ) -> "SABRSlice":
        """Calibra (alpha, rho, nu) com beta fixo sobre um smile de volatilidades implícitas."""
        k_np = np.asarray(K_arr, dtype=np.float64)
        v_np = np.asarray(vols, dtype=np.float64)
        atm_v = float(np.median(v_np))
        alpha0 = max(1e-4, atm_v * (float(F) ** (1.0 - float(beta))))
        x0 = np.array([alpha0, -0.2, 0.4], dtype=np.float64)
        lb = np.array([1e-5, -0.999, 1e-4], dtype=np.float64)
        ub = np.array([5.0, 0.999, 5.0], dtype=np.float64)

        def _res(params):
            al, rh, nu = params
            pred = np.array([_cy_sabr_vol(float(F), float(k), float(T), float(al), float(beta), float(rh), float(nu)) for k in k_np])
            return pred - v_np

        res = least_squares(_res, x0, bounds=(lb, ub), max_nfev=1000)
        al, rh, nu = [float(v) for v in res.x]
        return cls(al, float(beta), rh, nu, float(F), float(T))


class VolatilitySurface:
    """
    Superfície de volatilidade implícita (padrão B3 / Bloomberg):
    - Interpolação no smile por `monotone_cubic`, `exponential`, `svi` ou `sabr`.
    - Interpolação temporal em variância total `V_T = sigma^2 * T` ou `monotone_cubic`.
    - Conversão Delta <-> Strike (com clamping em [1%, 99%]) e auditoria de não arbitragem.
    """

    def __init__(
        self,
        dus: Sequence[int],
        smile_mode: str = "grid_strike",
        smile_interp: str = "monotone_cubic",
        time_interp: str = "total_variance",
    ):
        self.dus = sorted([int(d) for d in dus])
        self.smile_mode = smile_mode
        self.smile_interp = smile_interp
        self.time_interp = time_interp
        self._slices: Dict[int, object] = {}

    @classmethod
    def from_svi_slices(cls, slices_by_du: Dict[int, SVISlice], time_interp: str = "total_variance") -> "VolatilitySurface":
        surf = cls(list(slices_by_du.keys()), smile_mode="svi", smile_interp="svi", time_interp=time_interp)
        surf._slices = {int(k): v for k, v in slices_by_du.items()}
        return surf

    @classmethod
    def from_sabr_slices(cls, slices_by_du: Dict[int, SABRSlice], time_interp: str = "total_variance") -> "VolatilitySurface":
        surf = cls(list(slices_by_du.keys()), smile_mode="sabr", smile_interp="sabr", time_interp=time_interp)
        surf._slices = {int(k): v for k, v in slices_by_du.items()}
        return surf

    @classmethod
    def from_delta_grid(
        cls,
        dus: Sequence[int],
        deltas: Sequence[float],
        vols_matrix: Sequence[Sequence[float]],
        smile_interp: str = "monotone_cubic",
        time_interp: str = "total_variance",
    ) -> "VolatilitySurface":
        surf = cls(dus, smile_mode="grid_delta", smile_interp=smile_interp, time_interp=time_interp)
        d_arr = np.asarray(deltas, dtype=np.float64)
        order = np.argsort(d_arr)
        d_sorted = d_arr[order]
        for idx, du in enumerate(dus):
            v_arr = np.asarray(vols_matrix[idx], dtype=np.float64)[order]
            surf._slices[int(du)] = (d_sorted, v_arr)
        return surf

    @classmethod
    def from_strike_grid(
        cls,
        dus: Sequence[int],
        strikes: Sequence[float],
        vols_matrix: Sequence[Sequence[float]],
        smile_interp: str = "monotone_cubic",
        time_interp: str = "total_variance",
    ) -> "VolatilitySurface":
        surf = cls(dus, smile_mode="grid_strike", smile_interp=smile_interp, time_interp=time_interp)
        k_arr = np.asarray(strikes, dtype=np.float64)
        order = np.argsort(k_arr)
        k_sorted = k_arr[order]
        for idx, du in enumerate(dus):
            v_arr = np.asarray(vols_matrix[idx], dtype=np.float64)[order]
            surf._slices[int(du)] = (k_sorted, v_arr)
        return surf

    def _eval_slice(
        self,
        du: int,
        strike: Optional[float] = None,
        delta: Optional[float] = None,
        forward: Optional[float] = None,
    ) -> float:
        sl = self._slices[du]
        if self.smile_mode == "svi":
            assert isinstance(sl, SVISlice)
            if strike is not None and forward is not None:
                k = math.log(float(strike) / float(forward))
            elif delta is not None:
                # Aproxima k a partir do delta usando vol ATM inicial
                sig_atm = sl.implied_vol(0.0)
                k_est = delta_to_strike(delta, 1.0, sig_atm, sl.T)
                k = math.log(k_est)
            else:
                raise ValueError("Informe 'strike' e 'forward' (ou 'delta') para superfície SVI.")
            return float(sl.implied_vol(k))

        if self.smile_mode == "sabr":
            assert isinstance(sl, SABRSlice)
            if strike is None:
                raise ValueError("Informe 'strike' para superfície SABR.")
            return float(sl.implied_vol(float(strike)))

        x_nodes, y_nodes = sl
        if self.smile_mode == "grid_delta":
            if delta is not None:
                x_eval = min(0.99, max(0.01, float(delta)))
            elif strike is not None and forward is not None:
                t_val = du / 252.0
                vol_atm = monotone_cubic_interp(x_nodes, y_nodes, 0.5)
                x_eval = min(0.99, max(0.01, strike_to_delta(float(strike), float(forward), vol_atm, t_val)))
            else:
                raise ValueError("Informe 'delta' ou ('strike', 'forward') para superfície por delta.")
        else:
            if strike is None:
                raise ValueError("Informe 'strike' para superfície por strike.")
            x_eval = float(strike)

        if self.smile_interp == "exponential":
            return exp_smile_interp(x_nodes, y_nodes, x_eval)
        return monotone_cubic_interp(x_nodes, y_nodes, x_eval)

    def get_vol(
        self,
        du: int,
        strike: Optional[float] = None,
        delta: Optional[float] = None,
        forward: Optional[float] = None,
    ) -> float:
        du_i = int(du)
        if du_i in self._slices:
            return self._eval_slice(du_i, strike=strike, delta=delta, forward=forward)

        if du_i <= self.dus[0]:
            return self._eval_slice(self.dus[0], strike=strike, delta=delta, forward=forward)
        if du_i >= self.dus[-1]:
            return self._eval_slice(self.dus[-1], strike=strike, delta=delta, forward=forward)

        idx = int(np.searchsorted(self.dus, du_i, side="right") - 1)
        du_a = self.dus[idx]
        du_p = self.dus[idx + 1]
        vol_a = self._eval_slice(du_a, strike=strike, delta=delta, forward=forward)
        vol_p = self._eval_slice(du_p, strike=strike, delta=delta, forward=forward)

        if self.time_interp == "monotone_cubic":
            vols_all = [self._eval_slice(d, strike=strike, delta=delta, forward=forward) for d in self.dus]
            return monotone_cubic_interp(self.dus, vols_all, float(du_i))

        # Interpolação linear em variância total V_T = sigma^2 * T (padrão B3 / Bloomberg)
        t_a = du_a / 252.0
        t_p = du_p / 252.0
        t_eval = du_i / 252.0
        v_a = (vol_a * vol_a) * t_a
        v_p = (vol_p * vol_p) * t_p
        v_eval = v_a + ((v_p - v_a) / (t_p - t_a)) * (t_eval - t_a)
        return math.sqrt(max(1e-12, v_eval) / t_eval)

    def check_no_arbitrage(
        self,
        forward_by_du: Union[float, Dict[int, float], Callable[[int], float]],
        r_cont_by_du: Union[float, Dict[int, float], Callable[[int], float]],
        strikes: Optional[Sequence[float]] = None,
        tol: float = 1e-8,
    ) -> Dict[str, Union[bool, int]]:
        """
        Audita os três critérios de não arbitragem do Manual de Opções da B3:
        1. Monotonicidade em strike: dC/dK < 0
        2. Convexidade borboleta: d^2C/dK^2 >= 0
        3. Monotonicidade temporal da variância total: d(sigma^2 * T)/dT >= 0
        """
        def _resolve(obj, d):
            if callable(obj):
                return float(obj(d))
            if isinstance(obj, dict):
                return float(obj[d])
            return float(obj)

        mono_viol = 0
        bfly_viol = 0
        cal_viol = 0

        for du in self.dus:
            fwd = _resolve(forward_by_du, du)
            r_c = _resolve(r_cont_by_du, du)
            t_val = du / 252.0
            k_grid = np.linspace(0.7 * fwd, 1.3 * fwd, 25) if strikes is None else np.asarray(strikes, dtype=np.float64)
            calls = []
            for k in k_grid:
                vol = self.get_vol(du, strike=float(k), forward=fwd)
                calls.append(_cy_black76(fwd, float(k), r_c, t_val, vol, True)[0])
            calls_arr = np.asarray(calls, dtype=np.float64)
            mono_viol += int(np.sum(np.diff(calls_arr) > tol))
            if len(calls_arr) >= 3:
                d2 = np.diff(calls_arr, n=2)
                bfly_viol += int(np.sum(d2 < -tol))

        if len(self.dus) >= 2:
            fwd0 = _resolve(forward_by_du, self.dus[0])
            k_grid = np.linspace(0.8 * fwd0, 1.2 * fwd0, 13) if strikes is None else np.asarray(strikes, dtype=np.float64)
            for k in k_grid:
                tot_vars = []
                for du in self.dus:
                    fwd = _resolve(forward_by_du, du)
                    vol = self.get_vol(du, strike=float(k), forward=fwd)
                    tot_vars.append(vol * vol * (du / 252.0))
                cal_viol += int(np.sum(np.diff(tot_vars) < -tol))

        return {
            "passed": (mono_viol == 0 and bfly_viol == 0 and cal_viol == 0),
            "strike_monotonicity_violations": mono_viol,
            "butterfly_violations": bfly_viol,
            "calendar_violations": cal_viol,
        }

