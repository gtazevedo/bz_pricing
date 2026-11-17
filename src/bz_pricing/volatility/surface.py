import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from bz_pricing.core.options_core import (
    black76_price_greeks as _cy_black76,
    implied_vol_black76 as _cy_iv_black76,
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
        if weights is None:
            wt = np.ones_like(k_np)
        else:
            w_raw = np.maximum(0.0, np.asarray(weights, dtype=np.float64))
            w_mean = float(np.mean(w_raw)) if len(w_raw) > 0 else 1.0
            wt = np.sqrt(w_raw / w_mean) if w_mean > 0.0 else np.ones_like(k_np)

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
        weights: Optional[Sequence[float]] = None,
    ) -> "SABRSlice":
        """Calibra (alpha, rho, nu) com beta fixo sobre um smile de volatilidades implícitas."""
        k_np = np.asarray(K_arr, dtype=np.float64)
        v_np = np.asarray(vols, dtype=np.float64)
        if weights is None:
            wt = np.ones_like(k_np)
        else:
            w_raw = np.maximum(0.0, np.asarray(weights, dtype=np.float64))
            w_mean = float(np.mean(w_raw)) if len(w_raw) > 0 else 1.0
            wt = np.sqrt(w_raw / w_mean) if w_mean > 0.0 else np.ones_like(k_np)
        atm_v = float(np.median(v_np))
        alpha0 = max(1e-4, atm_v * (float(F) ** (1.0 - float(beta))))
        ub_alpha = max(10.0, alpha0 * 20.0)
        x0 = np.array([min(ub_alpha * 0.95, alpha0), -0.2, 0.4], dtype=np.float64)
        lb = np.array([1e-5, -0.999, 1e-4], dtype=np.float64)
        ub = np.array([ub_alpha, 0.999, 5.0], dtype=np.float64)

        def _res(params):
            al, rh, nu = params
            pred = np.array([_cy_sabr_vol(float(F), float(k), float(T), float(al), float(beta), float(rh), float(nu)) for k in k_np])
            return wt * (pred - v_np)

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
        if len(vols_matrix) != len(dus):
            raise ValueError("vols_matrix deve ter o mesmo número de linhas que dus.")
        d_arr = np.asarray(deltas, dtype=np.float64)
        order = np.argsort(d_arr)
        d_sorted = d_arr[order]
        for idx, du in enumerate(dus):
            if len(vols_matrix[idx]) != len(d_arr):
                raise ValueError(f"Linha {idx} de vols_matrix deve ter o mesmo comprimento de deltas.")
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
        if len(vols_matrix) != len(dus):
            raise ValueError("vols_matrix deve ter o mesmo número de linhas que dus.")
        k_arr = np.asarray(strikes, dtype=np.float64)
        order = np.argsort(k_arr)
        k_sorted = k_arr[order]
        for idx, du in enumerate(dus):
            if len(vols_matrix[idx]) != len(k_arr):
                raise ValueError(f"Linha {idx} de vols_matrix deve ter o mesmo comprimento de strikes.")
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
                # Ponto fixo para k(delta) consistente com sigma(k)
                sig_est = sl.implied_vol(0.0)
                k = 0.0
                for _ in range(5):
                    k_est = delta_to_strike(delta, 1.0, sig_est, sl.T)
                    k = math.log(k_est)
                    sig_est = float(sl.implied_vol(k))
                return sig_est
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
                vol_est = monotone_cubic_interp(x_nodes, y_nodes, 0.5)
                x_eval = 0.5
                for _ in range(5):
                    x_eval = min(0.99, max(0.01, strike_to_delta(float(strike), float(forward), vol_est, t_val)))
                    vol_est = (
                        exp_smile_interp(x_nodes, y_nodes, x_eval)
                        if self.smile_interp == "exponential"
                        else monotone_cubic_interp(x_nodes, y_nodes, x_eval)
                    )
                return vol_est
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
            if strikes is None:
                moneyness_grid = np.linspace(0.8, 1.2, 13)
                for m_val in moneyness_grid:
                    tot_vars = []
                    for du in self.dus:
                        fwd = _resolve(forward_by_du, du)
                        vol = self.get_vol(du, strike=float(m_val * fwd), forward=fwd)
                        tot_vars.append(vol * vol * (du / 252.0))
                    cal_viol += int(np.sum(np.diff(tot_vars) < -tol))
            else:
                k_grid = np.asarray(strikes, dtype=np.float64)
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

    def wing_slope_check(self) -> Dict[str, object]:
        """
        Verifica o limite de momentos de Lee (2004) nas asas das fatias SVI:
            b * (1 + |rho|) <= 2, com b >= 0 e |rho| < 1.
        """
        details = []
        all_ok = True
        for du in self.dus:
            sl = self._slices.get(du) if self._slices else None
            if isinstance(sl, SVISlice):
                max_slope = float(sl.b * (1.0 + abs(sl.rho)))
                ok = (sl.b >= 0.0) and (abs(sl.rho) < 1.0) and (max_slope <= 2.0 + 1e-9)
                all_ok = all_ok and ok
                details.append({
                    "du": du,
                    "b": float(sl.b),
                    "rho": float(sl.rho),
                    "max_slope": max_slope,
                    "passed": ok,
                })
        return {"passed": all_ok, "slices": details}

    def breeden_litzenberger_density(
        self,
        du: int,
        forward: float,
        r_cont: float,
        strikes: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """
        Extrai a densidade neutra ao risco via Breeden-Litzenberger:
            f(K) = exp(r * T) * d^2 C / dK^2
        """
        k_grid = (
            np.linspace(0.6 * forward, 1.4 * forward, 81)
            if strikes is None
            else np.asarray(strikes, dtype=np.float64)
        )
        if len(k_grid) < 3:
            raise ValueError("São necessários ao menos 3 strikes para Breeden-Litzenberger.")

        t_val = du / 252.0
        calls = np.array(
            [
                _cy_black76(
                    float(forward),
                    float(k),
                    float(r_cont),
                    t_val,
                    self.get_vol(du, strike=float(k), forward=float(forward)),
                    True,
                )[0]
                for k in k_grid
            ],
            dtype=np.float64,
        )
        dk = np.diff(k_grid)
        dk_mid = 0.5 * (dk[:-1] + dk[1:])
        d2c = (
            (calls[2:] - calls[1:-1]) / dk[1:]
            - (calls[1:-1] - calls[:-2]) / dk[:-1]
        ) / dk_mid
        density = math.exp(r_cont * t_val) * d2c
        k_mid = k_grid[1:-1]
        integral = float(np.trapezoid(density, k_mid)) if hasattr(np, "trapezoid") else float(np.trapz(density, k_mid))
        min_d = float(np.min(density))
        return {
            "strikes": k_mid,
            "density": density,
            "min_density": min_d,
            "integral": integral,
            "passed": min_d >= -1e-6,
        }

    def dupire_local_vol_grid(
        self,
        forward_by_du: Union[float, Dict[int, float], Callable[[int], float]],
        r_cont_by_du: Union[float, Dict[int, float], Callable[[int], float]],
        dus: Optional[Sequence[int]] = None,
        strikes: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """
        Calcula a volatilidade local de Dupire em termos da variância total w(y, T) = sigma^2 * T:
            sigma_loc^2 = (dw/dT) / [1 - (y/w)(dw/dy) + 0.25*(-0.25 - 1/w + y^2/w^2)*(dw/dy)^2 + 0.5*(d^2w/dy^2)]
        """
        def _resolve(obj, d):
            if callable(obj):
                return float(obj(d))
            if isinstance(obj, dict):
                return float(obj[d])
            return float(obj)

        eval_dus = list(self.dus) if dus is None else [int(d) for d in dus]
        fwd_ref = _resolve(forward_by_du, eval_dus[0])
        k_grid = (
            np.linspace(0.75 * fwd_ref, 1.25 * fwd_ref, 21)
            if strikes is None
            else np.asarray(strikes, dtype=np.float64)
        )

        dy = 1e-3
        local_vol = np.full((len(eval_dus), len(k_grid)), np.nan, dtype=np.float64)
        min_var = float("inf")
        neg_nodes = 0

        for i, du in enumerate(eval_dus):
            t_val = du / 252.0
            fwd = _resolve(forward_by_du, du)
            _ = _resolve(r_cont_by_du, du)

            if du <= self.dus[0]:
                du_minus = du
                du_plus = du + 1
            elif du >= self.dus[-1]:
                du_minus = max(1, du - 1)
                du_plus = du
            else:
                du_minus = max(1, du - 1)
                du_plus = du + 1
            dt = max(1.0 / 252.0, (du_plus - du_minus) / 252.0)
            fwd_minus = _resolve(forward_by_du, du_minus) if not isinstance(forward_by_du, dict) or du_minus in forward_by_du else fwd
            fwd_plus = _resolve(forward_by_du, du_plus) if not isinstance(forward_by_du, dict) or du_plus in forward_by_du else fwd

            for j, k in enumerate(k_grid):
                k_f = float(k)
                y = math.log(k_f / fwd)

                sig_c = self.get_vol(du, strike=k_f, forward=fwd)
                w_c = sig_c * sig_c * t_val

                k_up = fwd * math.exp(y + dy)
                k_dn = fwd * math.exp(y - dy)
                sig_up = self.get_vol(du, strike=k_up, forward=fwd)
                sig_dn = self.get_vol(du, strike=k_dn, forward=fwd)
                w_up = sig_up * sig_up * t_val
                w_dn = sig_dn * sig_dn * t_val

                dw_dy = (w_up - w_dn) / (2.0 * dy)
                d2w_dy2 = (w_up - 2.0 * w_c + w_dn) / (dy * dy)

                sig_tp = self.get_vol(du_plus, strike=k_f, forward=fwd_plus)
                sig_tm = self.get_vol(du_minus, strike=k_f, forward=fwd_minus)
                w_tp = sig_tp * sig_tp * (du_plus / 252.0)
                w_tm = sig_tm * sig_tm * (du_minus / 252.0)
                dw_dt = (w_tp - w_tm) / dt

                denom = (
                    1.0
                    - (y / max(1e-12, w_c)) * dw_dy
                    + 0.25 * (-0.25 - 1.0 / max(1e-12, w_c) + (y * y) / max(1e-12, w_c * w_c)) * (dw_dy * dw_dy)
                    + 0.5 * d2w_dy2
                )
                if denom <= 1e-12 or dw_dt <= 0.0:
                    var_loc = dw_dt / denom if abs(denom) > 1e-12 else -1.0
                    neg_nodes += 1
                    min_var = min(min_var, var_loc)
                else:
                    var_loc = dw_dt / denom
                    min_var = min(min_var, var_loc)
                    local_vol[i, j] = math.sqrt(var_loc)

        return {
            "dus": np.asarray(eval_dus, dtype=int),
            "strikes": k_grid,
            "local_vol": local_vol,
            "min_local_var": 0.0 if min_var == float("inf") else float(min_var),
            "negative_nodes": int(neg_nodes),
            "passed": neg_nodes == 0,
        }


def infer_forward_from_put_call_parity(
    df_slice,
    discount_factor: float,
    min_volume: float = 0.0,
) -> float:
    """
    Infere o preço futuro/forward implícito pela paridade put-call em uma fatia de mesmo vencimento:
        F_i = K_i + (C_i - P_i) / DF
    Utiliza a mediana ponderada pela liquidez (ou simples, se indisponível) entre os pares call/put
    de mesmo strike para eliminar dependência de proxy externo de spot.
    """
    import pandas as pd

    if not isinstance(df_slice, pd.DataFrame) or df_slice.empty:
        raise ValueError("df_slice deve ser um DataFrame não vazio.")
    df = df_slice.copy()

    k_col = next((c for c in ("ExrcPric", "strike", "Strike", "K") if c in df.columns), None)
    t_col = next((c for c in ("option_type", "OptnTp", "type", "Type") if c in df.columns), None)
    p_col = next((c for c in ("RefPric", "AdjstdQt", "price", "Price", "LastPric") if c in df.columns), None)
    if k_col is None or t_col is None or p_col is None:
        raise ValueError("df_slice deve conter colunas de strike, tipo de opção e prêmio.")

    v_col = next((c for c in ("RglrTxsQty", "TOTNEG", "OpnIntrst", "volume", "Volume") if c in df.columns), None)
    df["_k"] = pd.to_numeric(df[k_col], errors="coerce")
    df["_p"] = pd.to_numeric(df[p_col], errors="coerce")
    df["_is_call"] = df[t_col].astype(str).str.lower().str.startswith("c")
    df["_vol"] = pd.to_numeric(df[v_col], errors="coerce").fillna(1.0) if v_col else 1.0

    df = df[df["_k"].notna() & df["_p"].notna() & (df["_p"] > 0.0)]
    if min_volume > 0.0 and v_col is not None:
        df = df[df["_vol"] >= float(min_volume)]

    calls = df[df["_is_call"]].groupby("_k", as_index=True).agg(c_p=("_p", "mean"), c_v=("_vol", "sum"))
    puts = df[~df["_is_call"]].groupby("_k", as_index=True).agg(p_p=("_p", "mean"), p_v=("_vol", "sum"))
    paired = calls.join(puts, how="inner")
    if paired.empty:
        raise ValueError("Não há pares call/put no mesmo strike para inferir o forward via paridade.")

    df_eff = max(1e-12, float(discount_factor))
    fwds = paired.index.to_numpy(dtype=np.float64) + (paired["c_p"].to_numpy(dtype=np.float64) - paired["p_p"].to_numpy(dtype=np.float64)) / df_eff
    wts = np.minimum(paired["c_v"].to_numpy(dtype=np.float64), paired["p_v"].to_numpy(dtype=np.float64))
    wts = np.where(np.isfinite(wts) & (wts > 0.0), wts, 1.0)

    order = np.argsort(fwds)
    fwds_s = fwds[order]
    wts_s = wts[order]
    cum_w = np.cumsum(wts_s)
    idx_med = int(np.searchsorted(cum_w, 0.5 * cum_w[-1]))
    idx_med = min(len(fwds_s) - 1, max(0, idx_med))
    return float(fwds_s[idx_med])


def validate_slice_out_of_sample(
    strikes: Sequence[float],
    market_prices: Sequence[float],
    option_types: Sequence[str],
    forward: float,
    r_cont: float,
    du: int,
    train_mask: Optional[Sequence[bool]] = None,
    model: str = "svi",
    sabr_beta: float = 0.5,
    otm_only: bool = True,
    vega_weighted: bool = False,
    split_mode: str = "alternating",
) -> Dict[str, object]:
    """
    Calibra uma fatia (SVI ou SABR) no conjunto de treino e avalia o re-apreçamento
    fora da amostra (out-of-sample) em bps de volatilidade implícita e em prêmio.
    Suporta filtro OTM (`otm_only`), ponderação por vega (`vega_weighted`) e partições
    `split_mode` em {'alternating', 'interpolation', 'wings'}.
    """
    k_arr = np.asarray(strikes, dtype=np.float64)
    p_arr = np.asarray(market_prices, dtype=np.float64)
    is_call_arr = np.array([str(t).lower().startswith("c") for t in option_types], dtype=bool)
    t_val = du / 252.0

    mkt_vols = np.array(
        [
            _cy_iv_black76(float(p), float(forward), float(k), float(r_cont), t_val, bool(ic))
            for p, k, ic in zip(p_arr, k_arr, is_call_arr)
        ],
        dtype=np.float64,
    )

    valid = np.isfinite(mkt_vols) & (mkt_vols > 1e-4)
    if otm_only:
        otm_mask = (is_call_arr & (k_arr >= float(forward) * (1.0 - 1e-12))) | (
            (~is_call_arr) & (k_arr <= float(forward) * (1.0 + 1e-12))
        )
        keep = valid & otm_mask
    else:
        keep = valid

    k_arr = k_arr[keep]
    p_arr = p_arr[keep]
    is_call_arr = is_call_arr[keep]
    mkt_vols = mkt_vols[keep]

    if train_mask is None:
        order = np.argsort(k_arr)
        k_arr = k_arr[order]
        p_arr = p_arr[order]
        is_call_arr = is_call_arr[order]
        mkt_vols = mkt_vols[order]
        n_pts = len(k_arr)
        mode_norm = split_mode.strip().lower()
        if mode_norm == "alternating":
            mask = np.zeros(n_pts, dtype=bool)
            mask[::2] = True
            if np.sum(~mask) == 0 and n_pts > 0:
                mask[-1] = False
        elif mode_norm == "interpolation":
            mask = np.ones(n_pts, dtype=bool)
            if n_pts >= 5:
                mask[2:-2:2] = False
            elif n_pts >= 4:
                mask[1] = False
        elif mode_norm == "wings":
            mask = np.ones(n_pts, dtype=bool)
            if n_pts >= 6:
                n_wing = max(1, (n_pts - max(3, int(round(0.65 * n_pts)))) // 2)
                mask[:n_wing] = False
                mask[-n_wing:] = False
            elif n_pts >= 4:
                mask[-1] = False
        else:
            raise ValueError(f"split_mode inválido: {split_mode}. Use 'alternating', 'interpolation' ou 'wings'.")
    else:
        mask = np.asarray(train_mask, dtype=bool)[keep]

    if np.sum(mask) < 3 or np.sum(~mask) < 1:
        raise ValueError("São necessários ao menos 3 pontos de treino e 1 ponto de teste válidos.")

    fit_weights = None
    if vega_weighted:
        vegas = np.array(
            [
                _cy_black76(float(forward), float(k), float(r_cont), t_val, float(v), bool(ic))[3]
                for k, v, ic in zip(k_arr[mask], mkt_vols[mask], is_call_arr[mask])
            ],
            dtype=np.float64,
        )
        fit_weights = np.maximum(vegas, 1e-12)

    if model.lower() == "svi":
        k_log_train = np.log(k_arr[mask] / float(forward))
        fitted_slice = SVISlice.fit(k_log_train, mkt_vols[mask], t_val, weights=fit_weights)
        pred_vols_test = fitted_slice.implied_vol(np.log(k_arr[~mask] / float(forward)))
    elif model.lower() == "sabr":
        fitted_slice = SABRSlice.fit(
            float(forward), k_arr[mask], mkt_vols[mask], t_val, beta=sabr_beta, weights=fit_weights
        )
        pred_vols_test = fitted_slice.implied_vol(k_arr[~mask])
    else:
        raise ValueError(f"Modelo '{model}' não suportado; use 'svi' ou 'sabr'.")

    pred_prices_test = np.array(
        [
            _cy_black76(float(forward), float(k), float(r_cont), t_val, float(v), bool(ic))[0]
            for k, v, ic in zip(k_arr[~mask], pred_vols_test, is_call_arr[~mask])
        ],
        dtype=np.float64,
    )

    vol_err_bps = (pred_vols_test - mkt_vols[~mask]) * 1e4
    price_err = pred_prices_test - p_arr[~mask]

    return {
        "slice": fitted_slice,
        "train_strikes": k_arr[mask],
        "test_strikes": k_arr[~mask],
        "test_market_vols": mkt_vols[~mask],
        "test_pred_vols": pred_vols_test,
        "test_market_prices": p_arr[~mask],
        "test_pred_prices": pred_prices_test,
        "rmse_vol_bps": float(np.sqrt(np.mean(vol_err_bps ** 2))),
        "mae_vol_bps": float(np.mean(np.abs(vol_err_bps))),
        "rmse_price": float(np.sqrt(np.mean(price_err ** 2))),
        "mae_price": float(np.mean(np.abs(price_err))),
    }

