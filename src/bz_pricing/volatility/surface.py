import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from bz_pricing.core.options_core import (
    bachelier_price_greeks as _cy_bachelier,
    black76_price_greeks as _cy_black76,
    exp_smile_eval as _cy_exp_smile_eval,
    implied_vol_bachelier as _cy_iv_bachelier,
    implied_vol_black76 as _cy_iv_black76,
    monotone_cubic_eval as _cy_monotone_cubic_eval,
    norm_cdf as _cy_norm_cdf,
    norm_ppf as _cy_norm_ppf,
    sabr_alpha_from_atm_vol as _cy_sabr_alpha_from_atm,
    sabr_eval_vec as _cy_sabr_eval_vec,
    sabr_implied_vol as _cy_sabr_vol,
    sabr_normal_implied_vol as _cy_sabr_normal_vol,
    svi_eval_vec as _cy_svi_eval_vec,
    svi_total_variance as _cy_svi_total_var,
)


def _normalize_vol_type(vol_type: str) -> str:
    vt = str(vol_type).strip().lower()
    if vt in ("lognormal", "black", "black76", "bsm"):
        return "lognormal"
    if vt in ("normal", "bachelier"):
        return "normal"
    raise ValueError(f"vol_type inválido: {vol_type}. Use 'lognormal' ou 'normal'.")


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
    Interpolação por spline cúbico monótono de Fritsch-Carlson via núcleo Cython.
    """
    x = np.ascontiguousarray(x_nodes, dtype=np.float64)
    y = np.ascontiguousarray(y_nodes, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("Nós de interpolação vazios.")
    return float(_cy_monotone_cubic_eval(x, y, float(x_eval)))


def exp_smile_interp(x_nodes: Sequence[float], y_nodes: Sequence[float], x_eval: float) -> float:
    """Interpolação exponencial por partes entre vértices do smile via núcleo Cython."""
    x = np.ascontiguousarray(x_nodes, dtype=np.float64)
    y = np.ascontiguousarray(y_nodes, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("Nós de interpolação vazios.")
    return float(_cy_exp_smile_eval(x, y, float(x_eval)))


class SVISlice:
    """
    Fatia temporal parametrizada por SVI (Stochastic Volatility Inspired - Gatheral / B3 / Bloomberg):
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    onde k = ln(K / F_n) e w(k) = sigma^2 * T (lognormal ou normal).
    """

    def __init__(
        self,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
        T: float,
        vol_type: str = "lognormal",
        atm_vol: Optional[float] = None,
    ):
        self.a = float(a)
        self.b = float(b)
        self.rho = float(rho)
        self.m = float(m)
        self.sigma = float(sigma)
        self.T = float(T)
        self.vol_type = _normalize_vol_type(vol_type)
        self.atm_vol = float(atm_vol) if atm_vol is not None else None

    def total_variance(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(k, np.ndarray):
            return _cy_svi_eval_vec(k.astype(np.float64), self.a, self.b, self.rho, self.m, self.sigma)
        return float(_cy_svi_total_var(float(k), self.a, self.b, self.rho, self.m, self.sigma))

    def implied_vol(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        w = self.total_variance(k)
        t_eff = max(1e-8, self.T)
        if isinstance(w, np.ndarray):
            return np.sqrt(np.maximum(w, 1e-16) / t_eff)
        return math.sqrt(max(w, 1e-16) / t_eff)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "SVISlice",
            "a": self.a,
            "b": self.b,
            "rho": self.rho,
            "m": self.m,
            "sigma": self.sigma,
            "T": self.T,
            "vol_type": self.vol_type,
            "atm_vol": self.atm_vol,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SVISlice":
        return cls(
            a=float(data["a"]),
            b=float(data["b"]),
            rho=float(data["rho"]),
            m=float(data["m"]),
            sigma=float(data["sigma"]),
            T=float(data["T"]),
            vol_type=str(data.get("vol_type", "lognormal")),
            atm_vol=float(data["atm_vol"]) if data.get("atm_vol") is not None else None,
        )

    @classmethod
    def fit(
        cls,
        k_arr: Sequence[float],
        vols_or_w: Sequence[float],
        T: float,
        weights: Optional[Sequence[float]] = None,
        input_is_total_var: bool = False,
        vol_type: str = "lognormal",
        anchor_atm: bool = False,
        atm_vol: Optional[float] = None,
    ) -> "SVISlice":
        """
        Calibra a fatia SVI por mínimos quadrados não lineares adaptados à escala da variância total,
        com suporte opcional a ancoragem ATM exata (`anchor_atm=True` / `atm_vol`) e volatilidade normal.
        """
        vt = _normalize_vol_type(vol_type)
        k_np = np.asarray(k_arr, dtype=np.float64)
        vals = np.asarray(vols_or_w, dtype=np.float64)
        t_eff = max(1e-8, float(T))
        w_target = vals if input_is_total_var else (vals * vals * t_eff)
        if weights is None:
            wt = np.ones_like(k_np)
        else:
            w_raw = np.maximum(0.0, np.asarray(weights, dtype=np.float64))
            wt_mean = float(np.mean(w_raw)) if len(w_raw) > 0 else 1.0
            wt = np.sqrt(w_raw / wt_mean) if wt_mean > 0.0 else np.ones_like(k_np)

        w_mean = max(1e-16, float(np.mean(w_target)))
        w_max = max(1e-16, float(np.max(w_target)))
        w_min = float(np.min(w_target))
        w_span = max(1e-16, w_max - w_min)
        k_min_v = float(np.min(k_np))
        k_max_v = float(np.max(k_np))
        k_span = max(1e-4, k_max_v - k_min_v)
        idx_min = int(np.argmin(w_target))

        m0 = float(k_np[idx_min])
        sig0 = max(1e-5, min(0.5, 0.1 * k_span))
        b0 = max(1e-6, w_span / max(1e-4, 0.5 * k_span))

        k_chk = np.unique(
            np.concatenate([np.linspace(min(-0.1, k_min_v * 1.5), max(0.1, k_max_v * 1.5), 41), k_np])
        )

        use_anchor = bool(anchor_atm or (atm_vol is not None))
        atm_target: Optional[float] = None
        if use_anchor:
            if atm_vol is not None:
                atm_target = float(atm_vol)
            else:
                ord_k = np.argsort(k_np)
                ks_u, u_idx = np.unique(k_np[ord_k], return_index=True)
                vs_u = np.sqrt(np.maximum(w_target[ord_k][u_idx], 1e-16) / t_eff)
                if ks_u[0] <= 0.0 <= ks_u[-1] and len(ks_u) >= 2:
                    atm_target = monotone_cubic_interp(ks_u, vs_u, 0.0)
                else:
                    atm_target = float(vs_u[int(np.argmin(np.abs(ks_u)))])
            w_atm = (atm_target * atm_target) * t_eff

            b_ub = max(5.0, 20.0 * b0) if vt == "normal" else max(5.0, min(20.0, 20.0 * b0))
            m_lb = k_min_v - k_span
            m_ub = k_max_v + k_span
            sig_ub = max(2.0, 2.0 * k_span)
            x0 = np.clip(
                np.array([b0, 0.0, m0, sig0], dtype=np.float64),
                np.array([1e-8, -0.998, m_lb + 1e-6, 1e-6]),
                np.array([b_ub * 0.99, 0.998, m_ub - 1e-6, sig_ub * 0.99]),
            )
            lb = np.array([1e-8, -0.999, m_lb, 1e-7], dtype=np.float64)
            ub = np.array([b_ub, 0.999, m_ub, sig_ub], dtype=np.float64)

            def _res_anchored(params):
                b, rho, m, sig = params
                a = w_atm - b * (-rho * m + math.sqrt(m * m + sig * sig))
                w_pred = _cy_svi_eval_vec(k_np, float(a), float(b), float(rho), float(m), float(sig))
                min_w = a + b * sig * math.sqrt(max(1e-8, 1.0 - rho * rho))
                pen_w = max(0.0, 1e-16 - min_w) / w_mean * 100.0
                if vt == "lognormal":
                    pen_lee = max(0.0, b * (1.0 + abs(rho)) - (2.0 - 1e-6)) * 50.0
                    dk = k_chk - m
                    sq = np.sqrt(dk * dk + sig * sig)
                    w_g = np.maximum(1e-16, a + b * (rho * dk + sq))
                    wp = b * (rho + dk / sq)
                    wpp = (b * sig * sig) / (sq * sq * sq)
                    term1 = (1.0 - (k_chk * wp) / (2.0 * w_g)) ** 2
                    term2 = 0.25 * (wp * wp) * (1.0 / w_g + 0.25)
                    g_k = term1 - term2 + 0.5 * wpp
                    pen_bfly = float(np.sqrt(np.sum(np.maximum(0.0, -g_k) ** 2))) * 25.0
                else:
                    pen_lee = 0.0
                    pen_bfly = 0.0
                return np.concatenate([wt * (w_pred - w_target) / w_mean, [pen_w, pen_lee, pen_bfly]])

            res = least_squares(_res_anchored, x0, bounds=(lb, ub), x_scale="jac", max_nfev=2500)
            b, rho, m, sig = [float(v) for v in res.x]
            a = w_atm - b * (-rho * m + math.sqrt(m * m + sig * sig))
            return cls(a, b, rho, m, sig, float(T), vol_type=vt, atm_vol=atm_target)

        a0 = max(-0.5 * w_max, w_min * 0.6)
        m_lb = k_min_v - k_span
        m_ub = k_max_v + k_span
        b_ub = max(5.0, 20.0 * b0)
        sig_ub = max(2.0, 2.0 * k_span)
        lb = np.array([-2.0 * w_max, 1e-8, -0.999, m_lb, 1e-7], dtype=np.float64)
        ub = np.array([2.0 * w_max, b_ub, 0.999, m_ub, sig_ub], dtype=np.float64)
        x0 = np.clip(
            np.array([a0, b0, 0.0, m0, sig0], dtype=np.float64),
            lb + 1e-9,
            ub - 1e-9,
        )

        def _res(params):
            a, b, rho, m, sig = params
            w_pred = _cy_svi_eval_vec(k_np, float(a), float(b), float(rho), float(m), float(sig))
            min_w = a + b * sig * math.sqrt(max(1e-8, 1.0 - rho * rho))
            pen_w = max(0.0, 1e-16 - min_w) / w_mean * 100.0
            if vt == "lognormal":
                pen_lee = max(0.0, b * (1.0 + abs(rho)) - (2.0 - 1e-6)) * 50.0
                dk = k_chk - m
                sq = np.sqrt(dk * dk + sig * sig)
                w_g = np.maximum(1e-16, a + b * (rho * dk + sq))
                wp = b * (rho + dk / sq)
                wpp = (b * sig * sig) / (sq * sq * sq)
                term1 = (1.0 - (k_chk * wp) / (2.0 * w_g)) ** 2
                term2 = 0.25 * (wp * wp) * (1.0 / w_g + 0.25)
                g_k = term1 - term2 + 0.5 * wpp
                pen_bfly = float(np.sqrt(np.sum(np.maximum(0.0, -g_k) ** 2))) * 25.0
            else:
                pen_lee = 0.0
                pen_bfly = 0.0

            return np.concatenate([wt * (w_pred - w_target) / w_mean, [pen_w, pen_lee, pen_bfly]])

        res = least_squares(_res, x0, bounds=(lb, ub), x_scale="jac", max_nfev=2500)
        a, b, rho, m, sig = [float(v) for v in res.x]
        return cls(a, b, rho, m, sig, float(T), vol_type=vt)


class SABRSlice:
    """
    Fatia temporal parametrizada pelo modelo SABR (Hagan et al., 2002), adotado pela B3
    para smiles de opções sobre juros (DI1 / IDI), em volatilidade lognormal ou normal.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        F: float,
        T: float,
        vol_type: str = "lognormal",
        atm_vol: Optional[float] = None,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.nu = float(nu)
        self.F = float(F)
        self.T = float(T)
        self.vol_type = _normalize_vol_type(vol_type)
        self.atm_vol = float(atm_vol) if atm_vol is not None else None

    def implied_vol(self, K: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        is_norm = self.vol_type == "normal"
        if isinstance(K, np.ndarray):
            return _cy_sabr_eval_vec(
                self.F, K.astype(np.float64), self.T, self.alpha, self.beta, self.rho, self.nu, is_norm
            )
        if is_norm:
            return float(_cy_sabr_normal_vol(self.F, float(K), self.T, self.alpha, self.beta, self.rho, self.nu))
        return float(_cy_sabr_vol(self.F, float(K), self.T, self.alpha, self.beta, self.rho, self.nu))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "SABRSlice",
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "nu": self.nu,
            "F": self.F,
            "T": self.T,
            "vol_type": self.vol_type,
            "atm_vol": self.atm_vol,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SABRSlice":
        return cls(
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            rho=float(data["rho"]),
            nu=float(data["nu"]),
            F=float(data["F"]),
            T=float(data["T"]),
            vol_type=str(data.get("vol_type", "lognormal")),
            atm_vol=float(data["atm_vol"]) if data.get("atm_vol") is not None else None,
        )

    @classmethod
    def fit(
        cls,
        F: float,
        K_arr: Sequence[float],
        vols: Sequence[float],
        T: float,
        beta: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
        vol_type: str = "lognormal",
        anchor_atm: bool = False,
        atm_vol: Optional[float] = None,
    ) -> "SABRSlice":
        """
        Calibra (alpha, rho, nu) com beta fixo sobre um smile lognormal ou normal,
        com suporte opcional a ancoragem exata da volatilidade ATM via equação cúbica de Hagan.
        """
        vt = _normalize_vol_type(vol_type)
        is_norm = vt == "normal"
        beta_val = (0.0 if is_norm else 0.5) if beta is None else float(beta)
        k_np = np.asarray(K_arr, dtype=np.float64)
        v_np = np.asarray(vols, dtype=np.float64)
        if weights is None:
            wt = np.ones_like(k_np)
        else:
            w_raw = np.maximum(0.0, np.asarray(weights, dtype=np.float64))
            w_mean = float(np.mean(w_raw)) if len(w_raw) > 0 else 1.0
            wt = np.sqrt(w_raw / w_mean) if w_mean > 0.0 else np.ones_like(k_np)

        v_scale = max(1e-8, float(np.median(v_np)))
        fwd = float(F)
        t_val = float(T)

        use_anchor = bool(anchor_atm or (atm_vol is not None))
        if use_anchor:
            if atm_vol is not None:
                atm_target = float(atm_vol)
            else:
                ord_k = np.argsort(k_np)
                ks_u, u_idx = np.unique(k_np[ord_k], return_index=True)
                vs_u = v_np[ord_k][u_idx]
                if ks_u[0] <= fwd <= ks_u[-1] and len(ks_u) >= 2:
                    atm_target = monotone_cubic_interp(ks_u, vs_u, fwd)
                else:
                    atm_target = float(vs_u[int(np.argmin(np.abs(ks_u - fwd)))])

            x0_a = np.array([-0.1, 0.5], dtype=np.float64)
            lb_a = np.array([-0.999, 1e-4], dtype=np.float64)
            ub_a = np.array([0.999, 10.0], dtype=np.float64)

            def _res_anchored(params):
                rh, nu = float(params[0]), float(params[1])
                al = _cy_sabr_alpha_from_atm(fwd, t_val, atm_target, beta_val, rh, nu, is_norm)
                pred = _cy_sabr_eval_vec(fwd, k_np, t_val, al, beta_val, rh, nu, is_norm)
                return wt * (pred - v_np) / v_scale

            res = least_squares(_res_anchored, x0_a, bounds=(lb_a, ub_a), x_scale="jac", max_nfev=1500)
            rh, nu = float(res.x[0]), float(res.x[1])
            al = float(_cy_sabr_alpha_from_atm(fwd, t_val, atm_target, beta_val, rh, nu, is_norm))
            return cls(al, beta_val, rh, nu, fwd, t_val, vol_type=vt, atm_vol=atm_target)

        alpha0 = (
            max(1e-6, v_scale / max(1e-12, fwd ** beta_val))
            if is_norm
            else max(1e-6, v_scale * (fwd ** (1.0 - beta_val)))
        )
        lb_alpha = max(1e-10, alpha0 * 1e-4)
        ub_alpha = max(10.0, alpha0 * 50.0)
        x0 = np.array([min(ub_alpha * 0.95, max(lb_alpha * 1.05, alpha0)), -0.2, 0.5], dtype=np.float64)
        lb = np.array([lb_alpha, -0.999, 1e-4], dtype=np.float64)
        ub = np.array([ub_alpha, 0.999, 10.0], dtype=np.float64)

        def _res(params):
            al, rh, nu = float(params[0]), float(params[1]), float(params[2])
            pred = _cy_sabr_eval_vec(fwd, k_np, t_val, al, beta_val, rh, nu, is_norm)
            return wt * (pred - v_np) / v_scale

        res = least_squares(_res, x0, bounds=(lb, ub), x_scale="jac", max_nfev=1500)
        al, rh, nu = [float(v) for v in res.x]
        return cls(al, beta_val, rh, nu, fwd, t_val, vol_type=vt)


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
        slice_forwards: Optional[Dict[int, float]] = None,
        vol_type: str = "lognormal",
    ):
        self.dus = sorted([int(d) for d in dus])
        self.smile_mode = smile_mode
        self.smile_interp = smile_interp
        self.time_interp = time_interp
        self.vol_type = _normalize_vol_type(vol_type)
        self._slices: Dict[int, object] = {}
        self.slice_forwards: Dict[int, float] = {int(k): float(v) for k, v in slice_forwards.items()} if slice_forwards else {}
        self._bbg_surface = None

    def add_slice(self, du: int, slice_obj: object, forward: Optional[float] = None) -> None:
        du_i = int(du)
        self._slices[du_i] = slice_obj
        if du_i not in self.dus:
            self.dus = sorted(self.dus + [du_i])
        if forward is not None:
            self.slice_forwards[du_i] = float(forward)

    @classmethod
    def from_svi_slices(
        cls,
        slices_by_du: Dict[int, SVISlice],
        time_interp: str = "total_variance",
        slice_forwards: Optional[Dict[int, float]] = None,
        vol_type: Optional[str] = None,
    ) -> "VolatilitySurface":
        inferred_vt = vol_type or (next(iter(slices_by_du.values())).vol_type if slices_by_du else "lognormal")
        surf = cls(list(slices_by_du.keys()), smile_mode="svi", smile_interp="svi", time_interp=time_interp, slice_forwards=slice_forwards, vol_type=inferred_vt)
        surf._slices = {int(k): v for k, v in slices_by_du.items()}
        return surf

    @classmethod
    def from_sabr_slices(
        cls,
        slices_by_du: Dict[int, SABRSlice],
        time_interp: str = "total_variance",
        slice_forwards: Optional[Dict[int, float]] = None,
        vol_type: Optional[str] = None,
    ) -> "VolatilitySurface":
        fwds = slice_forwards if slice_forwards is not None else {int(k): float(v.F) for k, v in slices_by_du.items()}
        inferred_vt = vol_type or (next(iter(slices_by_du.values())).vol_type if slices_by_du else "lognormal")
        surf = cls(list(slices_by_du.keys()), smile_mode="sabr", smile_interp="sabr", time_interp=time_interp, slice_forwards=fwds, vol_type=inferred_vt)
        surf._slices = {int(k): v for k, v in slices_by_du.items()}
        return surf

    @classmethod
    def from_bloomberg_mixture(
        cls,
        slices_or_surface,
        time_interp: str = "monotone_cubic",
        slice_forwards: Optional[Dict[int, float]] = None,
    ) -> "VolatilitySurface":
        from bz_pricing.volatility.bloomberg_surface import BloombergMixtureSurface

        if isinstance(slices_or_surface, BloombergMixtureSurface):
            bbg_surf = slices_or_surface
        else:
            bbg_surf = BloombergMixtureSurface(slices_or_surface)
        fwds = slice_forwards if slice_forwards is not None else {int(k): float(v.F) for k, v in bbg_surf.slices.items()}
        surf = cls(bbg_surf.dus, smile_mode="bloomberg_mixture", smile_interp="bloomberg_mixture", time_interp=time_interp, slice_forwards=fwds)
        surf._bbg_surface = bbg_surf
        surf._slices = dict(bbg_surf.slices)
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

        if self.smile_mode == "bloomberg_mixture":
            if strike is None:
                raise ValueError("Informe 'strike' para superfície Bloomberg Mixture.")
            fwd_use = sl.F if forward is None else float(forward)
            return float(sl.implied_vol(float(strike) * (sl.F / max(1e-12, fwd_use))))

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

    def _resolve_forward_for_du(
        self,
        du: int,
        forward: Optional[Union[float, Dict[int, float], Callable[[int], float]]],
    ) -> Optional[float]:
        if callable(forward):
            return float(forward(du))
        if isinstance(forward, dict):
            if du in forward:
                return float(forward[du])
            keys = sorted(forward.keys())
            if not keys:
                return None
            if du <= keys[0]:
                return float(forward[keys[0]])
            if du >= keys[-1]:
                return float(forward[keys[-1]])
            vals = [float(forward[k]) for k in keys]
            return float(np.interp(du, keys, vals))
        if forward is None:
            if du in self.slice_forwards:
                return float(self.slice_forwards[du])
            if self.slice_forwards:
                keys = sorted(self.slice_forwards.keys())
                if du <= keys[0]:
                    return float(self.slice_forwards[keys[0]])
                if du >= keys[-1]:
                    return float(self.slice_forwards[keys[-1]])
                vals = [math.log(max(1e-12, float(self.slice_forwards[k]))) for k in keys]
                return math.exp(float(np.interp(du, keys, vals)))
            return None
        return float(forward)

    def get_vol(
        self,
        du: int,
        strike: Optional[float] = None,
        delta: Optional[float] = None,
        forward: Optional[Union[float, Dict[int, float], Callable[[int], float]]] = None,
    ) -> float:
        du_i = int(du)
        if self.smile_mode == "bloomberg_mixture" and self._bbg_surface is not None:
            if strike is None:
                raise ValueError("Informe 'strike' para superfície Bloomberg Mixture.")
            fwd_d = self._resolve_forward_for_du(du_i, forward)
            return float(self._bbg_surface.get_vol(du_i, strike=float(strike), forward=fwd_d))
        if du_i in self._slices:
            fwd_d = self._resolve_forward_for_du(du_i, forward)
            return self._eval_slice(du_i, strike=strike, delta=delta, forward=fwd_d)

        if du_i <= self.dus[0]:
            du_0 = self.dus[0]
            return self._eval_slice(du_0, strike=strike, delta=delta, forward=self._resolve_forward_for_du(du_0, forward))
        if du_i >= self.dus[-1]:
            du_n = self.dus[-1]
            return self._eval_slice(du_n, strike=strike, delta=delta, forward=self._resolve_forward_for_du(du_n, forward))

        idx = int(np.searchsorted(self.dus, du_i, side="right") - 1)
        du_a = self.dus[idx]
        du_p = self.dus[idx + 1]
        fwd_a = self._resolve_forward_for_du(du_a, forward)
        fwd_p = self._resolve_forward_for_du(du_p, forward)
        vol_a = self._eval_slice(du_a, strike=strike, delta=delta, forward=fwd_a)
        vol_p = self._eval_slice(du_p, strike=strike, delta=delta, forward=fwd_p)

        if self.time_interp == "monotone_cubic":
            vols_all = [
                self._eval_slice(d, strike=strike, delta=delta, forward=self._resolve_forward_for_du(d, forward))
                for d in self.dus
            ]
            return monotone_cubic_interp(self.dus, vols_all, float(du_i))

        # Interpolação linear em variância total V_T = sigma^2 * T (padrão B3 / Bloomberg)
        t_a = du_a / 252.0
        t_p = du_p / 252.0
        t_eval = du_i / 252.0
        v_a = (vol_a * vol_a) * t_a
        v_p = (vol_p * vol_p) * t_p
        v_eval = v_a + ((v_p - v_a) / (t_p - t_a)) * (t_eval - t_a)
        return math.sqrt(max(1e-16, v_eval) / t_eval)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a superfície completa (grade ou paramétrica) para dicionário JSON-compatível."""
        slices_ser: Dict[str, Any] = {}
        for du_k in self.dus:
            sl = self._slices.get(du_k)
            if sl is None:
                continue
            if hasattr(sl, "to_dict"):
                slices_ser[str(du_k)] = sl.to_dict()
            else:
                x_n, y_n = sl
                slices_ser[str(du_k)] = {
                    "kind": "GridSlice",
                    "x": np.asarray(x_n, dtype=float).tolist(),
                    "y": np.asarray(y_n, dtype=float).tolist(),
                }
        out: Dict[str, Any] = {
            "kind": "VolatilitySurface",
            "dus": [int(d) for d in self.dus],
            "smile_mode": self.smile_mode,
            "smile_interp": self.smile_interp,
            "time_interp": self.time_interp,
            "vol_type": self.vol_type,
            "slice_forwards": {str(k): float(v) for k, v in self.slice_forwards.items()},
            "slices": slices_ser,
        }
        if self._bbg_surface is not None and hasattr(self._bbg_surface, "to_dict"):
            out["bbg_surface"] = self._bbg_surface.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VolatilitySurface":
        """Reconstrói uma `VolatilitySurface` a partir de dicionário serializado."""
        smile_mode = str(data["smile_mode"])
        if smile_mode == "bloomberg_mixture" and "bbg_surface" in data:
            from bz_pricing.volatility.bloomberg_surface import BloombergMixtureSurface

            bbg_obj = BloombergMixtureSurface.from_dict(data["bbg_surface"])
            fwds = {int(k): float(v) for k, v in data.get("slice_forwards", {}).items()}
            return cls.from_bloomberg_mixture(
                bbg_obj,
                time_interp=str(data.get("time_interp", "monotone_cubic")),
                slice_forwards=fwds if fwds else None,
            )

        fwds = {int(k): float(v) for k, v in data.get("slice_forwards", {}).items()}
        surf = cls(
            dus=[int(d) for d in data["dus"]],
            smile_mode=smile_mode,
            smile_interp=str(data.get("smile_interp", "monotone_cubic")),
            time_interp=str(data.get("time_interp", "total_variance")),
            slice_forwards=fwds if fwds else None,
            vol_type=str(data.get("vol_type", "lognormal")),
        )
        for du_str, s_data in data.get("slices", {}).items():
            du_i = int(du_str)
            kind = str(s_data.get("kind", ""))
            if kind == "SVISlice" or smile_mode == "svi":
                surf._slices[du_i] = SVISlice.from_dict(s_data)
            elif kind == "SABRSlice" or smile_mode == "sabr":
                surf._slices[du_i] = SABRSlice.from_dict(s_data)
            elif kind == "LognormalMixtureSlice":
                from bz_pricing.volatility.bloomberg_surface import LognormalMixtureSlice

                surf._slices[du_i] = LognormalMixtureSlice.from_dict(s_data)
            else:
                surf._slices[du_i] = (
                    np.asarray(s_data["x"], dtype=np.float64),
                    np.asarray(s_data["y"], dtype=np.float64),
                )
        return surf

    def to_json(self) -> str:
        """Serializa a superfície para string JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "VolatilitySurface":
        """Reconstrói a superfície a partir de string JSON."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Union[str, Path]) -> None:
        """Salva a superfície serializada em arquivo JSON no caminho especificado."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VolatilitySurface":
        """Carrega uma superfície serializada de arquivo JSON."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def _call_price_for_vol(self, fwd: float, strike: float, r_c: float, t_val: float, vol: float) -> float:
        if self.vol_type == "normal":
            df_r = math.exp(-r_c * t_val)
            return float(_cy_bachelier(fwd, strike, df_r, t_val, vol, True)[0])
        return float(_cy_black76(fwd, strike, r_c, t_val, vol, True)[0])

    def check_no_arbitrage(
        self,
        forward_by_du: Union[float, Dict[int, float], Callable[[int], float]],
        r_cont_by_du: Union[float, Dict[int, float], Callable[[int], float]],
        strikes: Optional[Union[Sequence[float], Dict[int, Sequence[float]]]] = None,
        tol: float = 1e-8,
        moneyness_grid: Optional[Sequence[float]] = None,
    ) -> Dict[str, Union[bool, int]]:
        """
        Audita os critérios estáticos e de calendário de não arbitragem:
        1. Monotonicidade em strike: dC/dK < 0
        2. Convexidade borboleta: d^2C/dK^2 >= 0
        3. Monotonicidade temporal da variância total: d(sigma^2 * T)/dT >= 0
        Com tolerâncias relativas à escala do forward nos prêmios e à variância no calendário.
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
            price_tol = tol * max(fwd, 1.0)
            if isinstance(strikes, dict):
                k_grid = np.asarray(strikes[du], dtype=np.float64)
            elif strikes is not None:
                k_grid = np.asarray(strikes, dtype=np.float64)
            else:
                k_grid = np.linspace(0.7 * fwd, 1.3 * fwd, 25)
            calls = [self._call_price_for_vol(fwd, float(k), r_c, t_val, self.get_vol(du, strike=float(k), forward=fwd)) for k in k_grid]
            calls_arr = np.asarray(calls, dtype=np.float64)
            mono_viol += int(np.sum(np.diff(calls_arr) > price_tol))
            if len(calls_arr) >= 3:
                dk = np.maximum(1e-12, np.diff(k_grid))
                slopes = np.diff(calls_arr) / dk
                dk_mid = 0.5 * (dk[:-1] + dk[1:])
                bfly_vals = np.diff(slopes) * dk_mid
                bfly_viol += int(np.sum(bfly_vals < -price_tol))

        if len(self.dus) >= 2:
            if moneyness_grid is not None or strikes is None or isinstance(strikes, dict):
                m_grid = np.linspace(0.8, 1.2, 13) if moneyness_grid is None else np.asarray(moneyness_grid, dtype=np.float64)
                for m_val in m_grid:
                    tot_vars = []
                    for du in self.dus:
                        fwd = _resolve(forward_by_du, du)
                        vol = self.get_vol(du, strike=float(m_val * fwd), forward=fwd)
                        tot_vars.append(vol * vol * (du / 252.0))
                    var_tol = tol * max(1e-12, float(np.mean(tot_vars)))
                    cal_viol += int(np.sum(np.diff(tot_vars) < -var_tol))
            else:
                k_grid = np.asarray(strikes, dtype=np.float64)
                for k in k_grid:
                    tot_vars = []
                    for du in self.dus:
                        fwd = _resolve(forward_by_du, du)
                        vol = self.get_vol(du, strike=float(k), forward=fwd)
                        tot_vars.append(vol * vol * (du / 252.0))
                    var_tol = tol * max(1e-12, float(np.mean(tot_vars)))
                    cal_viol += int(np.sum(np.diff(tot_vars) < -var_tol))

        return {
            "passed": (mono_viol == 0 and bfly_viol == 0 and cal_viol == 0),
            "strike_monotonicity_violations": mono_viol,
            "butterfly_violations": bfly_viol,
            "calendar_violations": cal_viol,
        }

    def wing_slope_check(self) -> Dict[str, object]:
        """
        Verifica o limite de momentos de Lee (2004) nas asas das fatias SVI lognormais:
            b * (1 + |rho|) <= 2, com b >= 0 e |rho| < 1.
        """
        details = []
        all_ok = True
        for du in self.dus:
            sl = self._slices.get(du) if self._slices else None
            if isinstance(sl, SVISlice):
                max_slope = float(sl.b * (1.0 + abs(sl.rho)))
                ok = (sl.b >= 0.0) and (abs(sl.rho) < 1.0) and (sl.vol_type == "normal" or max_slope <= 2.0 + 1e-9)
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
                self._call_price_for_vol(
                    float(forward),
                    float(k),
                    float(r_cont),
                    t_val,
                    self.get_vol(du, strike=float(k), forward=float(forward)),
                )
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

            def _resolve_fwd_step(d_step):
                if callable(forward_by_du):
                    return float(forward_by_du(d_step))
                if isinstance(forward_by_du, dict):
                    if d_step in forward_by_du:
                        return float(forward_by_du[d_step])
                    keys = sorted(forward_by_du.keys())
                    if d_step <= keys[0]:
                        r_edge = _resolve(r_cont_by_du, keys[0])
                        return float(forward_by_du[keys[0]]) * math.exp(r_edge * (d_step - keys[0]) / 252.0)
                    if d_step >= keys[-1]:
                        r_edge = _resolve(r_cont_by_du, keys[-1])
                        return float(forward_by_du[keys[-1]]) * math.exp(r_edge * (d_step - keys[-1]) / 252.0)
                    vals = [math.log(max(1e-12, float(forward_by_du[k_i]))) for k_i in keys]
                    return math.exp(float(np.interp(d_step, keys, vals)))
                return float(forward_by_du)

            fwd_minus = _resolve_fwd_step(du_minus)
            fwd_plus = _resolve_fwd_step(du_plus)

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

                # Derivada temporal dw/dT a log-moneyness y = ln(K/F(T)) constante (equação de Gatheral)
                k_tp = fwd_plus * math.exp(y)
                k_tm = fwd_minus * math.exp(y)
                sig_tp = self.get_vol(du_plus, strike=k_tp, forward=fwd_plus)
                sig_tm = self.get_vol(du_minus, strike=k_tm, forward=fwd_minus)
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
    sabr_beta: Optional[float] = None,
    otm_only: bool = True,
    vega_weighted: bool = False,
    split_mode: str = "alternating",
    vol_type: str = "lognormal",
    anchor_atm: bool = False,
    atm_vol: Optional[float] = None,
) -> Dict[str, object]:
    """
    Calibra uma fatia (SVI, SABR ou Bloomberg Mixture) em espaço lognormal ou normal (Bachelier)
    no conjunto de treino e avalia o re-apreçamento fora da amostra (out-of-sample).
    Suporta ancoragem ATM (`anchor_atm`, `atm_vol`), filtro OTM inteligente (`otm_only`),
    ponderação por vega (`vega_weighted`) e partições `split_mode` em {'alternating', 'interpolation', 'wings'}.
    """
    vt = _normalize_vol_type(vol_type)
    is_norm = vt == "normal"
    fwd = float(forward)
    r_c = float(r_cont)
    t_val = du / 252.0
    df_r = math.exp(-r_c * t_val)

    k_arr = np.asarray(strikes, dtype=np.float64)
    p_arr = np.asarray(market_prices, dtype=np.float64)
    is_call_arr = np.array([str(t).lower().startswith("c") for t in option_types], dtype=bool)

    if is_norm:
        mkt_vols = np.array(
            [
                _cy_iv_bachelier(float(p), fwd, float(k), df_r, t_val, bool(ic))
                for p, k, ic in zip(p_arr, k_arr, is_call_arr)
            ],
            dtype=np.float64,
        )
        valid = np.isfinite(mkt_vols) & (mkt_vols > 1e-8)
    else:
        mkt_vols = np.array(
            [
                _cy_iv_black76(float(p), fwd, float(k), r_c, t_val, bool(ic))
                for p, k, ic in zip(p_arr, k_arr, is_call_arr)
            ],
            dtype=np.float64,
        )
        valid = np.isfinite(mkt_vols) & (mkt_vols > 1e-4)

    if otm_only:
        has_call_k: Dict[float, bool] = {}
        has_put_k: Dict[float, bool] = {}
        for k_v, ic_v, ok_v in zip(k_arr, is_call_arr, valid):
            if not ok_v:
                continue
            kr = round(float(k_v), 8)
            if ic_v:
                has_call_k[kr] = True
            else:
                has_put_k[kr] = True
        strict_otm = (is_call_arr & (k_arr >= fwd * (1.0 - 1e-12))) | (
            (~is_call_arr) & (k_arr <= fwd * (1.0 + 1e-12))
        )
        unpaired = np.array(
            [not (has_call_k.get(round(float(k_v), 8), False) and has_put_k.get(round(float(k_v), 8), False)) for k_v in k_arr],
            dtype=bool,
        )
        keep = valid & strict_otm
        # Caso uma asa inteira fique vazia após corte estrito, preserva pontas válidas sem par no strike
        if not np.any(keep & is_call_arr & (k_arr >= fwd * (1.0 - 1e-12))):
            keep = keep | (valid & unpaired & is_call_arr)
        if not np.any(keep & (~is_call_arr) & (k_arr <= fwd * (1.0 + 1e-12))):
            keep = keep | (valid & unpaired & (~is_call_arr))
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
        if is_norm:
            vegas = np.array(
                [
                    _cy_bachelier(fwd, float(k), df_r, t_val, float(v), bool(ic))[3]
                    for k, v, ic in zip(k_arr[mask], mkt_vols[mask], is_call_arr[mask])
                ],
                dtype=np.float64,
            )
        else:
            vegas = np.array(
                [
                    _cy_black76(fwd, float(k), r_c, t_val, float(v), bool(ic))[3]
                    for k, v, ic in zip(k_arr[mask], mkt_vols[mask], is_call_arr[mask])
                ],
                dtype=np.float64,
            )
        fit_weights = np.maximum(vegas, 1e-12)

    mdl_low = model.lower()
    if mdl_low == "svi":
        k_log_train = np.log(k_arr[mask] / fwd)
        fitted_slice = SVISlice.fit(
            k_log_train,
            mkt_vols[mask],
            t_val,
            weights=fit_weights,
            vol_type=vt,
            anchor_atm=anchor_atm,
            atm_vol=atm_vol,
        )
        pred_vols_test = fitted_slice.implied_vol(np.log(k_arr[~mask] / fwd))
    elif mdl_low == "sabr":
        beta_eff = (0.0 if is_norm else 0.5) if sabr_beta is None else float(sabr_beta)
        fitted_slice = SABRSlice.fit(
            fwd,
            k_arr[mask],
            mkt_vols[mask],
            t_val,
            beta=beta_eff,
            weights=fit_weights,
            vol_type=vt,
            anchor_atm=anchor_atm,
            atm_vol=atm_vol,
        )
        pred_vols_test = fitted_slice.implied_vol(k_arr[~mask])
    elif mdl_low in ("bloomberg_mixture", "bbg", "lognormal_mixture"):
        if is_norm:
            raise ValueError("Modelo 'bloomberg_mixture' suporta apenas vol_type='lognormal'.")
        from bz_pricing.volatility.bloomberg_surface import LognormalMixtureSlice

        types_train = ["CALL" if bool(ic) else "PUT" for ic in is_call_arr[mask]]
        fitted_slice = LognormalMixtureSlice.fit(
            strikes=k_arr[mask],
            market_prices=p_arr[mask],
            option_types=types_train,
            forward=fwd,
            r_cont=r_c,
            T=t_val,
        )
        pred_vols_test = np.asarray(fitted_slice.implied_vol(k_arr[~mask], r_cont=r_c), dtype=np.float64)
    else:
        raise ValueError(f"Modelo '{model}' não suportado; use 'svi', 'sabr' ou 'bloomberg_mixture'.")

    if is_norm:
        pred_prices_test = np.array(
            [
                _cy_bachelier(fwd, float(k), df_r, t_val, float(v), bool(ic))[0]
                for k, v, ic in zip(k_arr[~mask], pred_vols_test, is_call_arr[~mask])
            ],
            dtype=np.float64,
        )
        vol_diff = pred_vols_test - mkt_vols[~mask]
        vol_err_bps = (vol_diff / max(1.0, fwd)) * 1e4
        rmse_vol_norm = float(np.sqrt(np.mean(vol_diff ** 2)))
    else:
        pred_prices_test = np.array(
            [
                _cy_black76(fwd, float(k), r_c, t_val, float(v), bool(ic))[0]
                for k, v, ic in zip(k_arr[~mask], pred_vols_test, is_call_arr[~mask])
            ],
            dtype=np.float64,
        )
        vol_err_bps = (pred_vols_test - mkt_vols[~mask]) * 1e4
        rmse_vol_norm = float("nan")

    price_err = pred_prices_test - p_arr[~mask]

    return {
        "slice": fitted_slice,
        "vol_type": vt,
        "train_strikes": k_arr[mask],
        "test_strikes": k_arr[~mask],
        "test_market_vols": mkt_vols[~mask],
        "test_pred_vols": pred_vols_test,
        "test_market_prices": p_arr[~mask],
        "test_pred_prices": pred_prices_test,
        "rmse_vol_bps": float(np.sqrt(np.mean(vol_err_bps ** 2))),
        "mae_vol_bps": float(np.mean(np.abs(vol_err_bps))),
        "rmse_vol_norm": rmse_vol_norm,
        "rmse_price": float(np.sqrt(np.mean(price_err ** 2))),
        "mae_price": float(np.mean(np.abs(price_err))),
    }

