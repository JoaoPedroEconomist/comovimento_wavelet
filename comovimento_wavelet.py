#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comovimento_wavelet.py
======================
Análise de co-movimento entre ativos (B3 / yfinance) por wavelets, usando
apenas preços de fechamento -> log-retornos.

O que o script faz
------------------
1. Baixa fechamentos ajustados (proventos/splits) via yfinance, com cache,
   retry, calendário comum, tratamento de buracos, outliers e padronização.
2. Correlação por escala (MODWT) com IC de Fisher, excluindo coeficientes de
   borda.  Heatmaps (um por escala) para vários ativos.
3. Coerência wavelet cruzada (CWT Morlet) do par escolhido, com:
   significância Monte Carlo (nulo AR(1)), cone de influência (COI) e
   setas de fase.  Estimativa de lead-lag por faixa de período.
4. Coerência PARCIAL (controlando Ibovespa e/ou Dólar) e MÚLTIPLA.
5. Coerência média por escala (gráfico de linha) e por faixa
   (curto 2-8d, médio 8-64d, longo 64d+) para TODOS os pares, com redes de
   co-movimento por faixa, comunidades, clusters hierárquicos e score de
   diversificação.

Dependências
------------
    pip install numpy pandas scipy matplotlib PyWavelets yfinance networkx
    pip install statsmodels      # opcional (teste ADF)

Uso
---
    python comovimento_wavelet.py                       # config padrão
    python comovimento_wavelet.py --tickers PETR4 VALE3 ITUB4 BBDC4 \
           --pair PETR4 VALE3 --start 2016-01-01 --n-surrogates 300
    python comovimento_wavelet.py --synthetic --n-surrogates 40   # teste offline

Convenção de fase (mapa de coerência)
-------------------------------------
    seta ->  : em fase           seta <-  : em anti-fase
    seta ^   : ativo A LIDERA B  seta v   : ativo B LIDERA A
    (A = 1º ativo do --pair, B = 2º; eixo vertical = período, crescente p/ cima)

Cuidados / limitações (leia!)
-----------------------------
* Análise EXPLORATÓRIA, em amostra: a CWT/MODWT sobre a série inteira usa
  informação futura. Não use estes resultados diretamente como sinal de
  trading sem recalcular em janelas deslizantes (walk-forward).
* Resultados na região fora do COI (CWT) e nos primeiros L_j coeficientes
  (MODWT) são descartados por efeito de borda.
* Não-sincronia: o fechamento do dólar (BRL=X, ~24h) não coincide com o da
  B3 (17h). Isso cria defasagem artificial de ~1 pregão no lead-lag com o
  dólar. Interprete a fase com ele com cautela.
* Muitos pares => problema de comparações múltiplas. O limiar de 95% é
  pontual; use as redes como ferramenta descritiva, não inferencial.
* Períodos em PREGÕES (21 ~ mês, 63 ~ trimestre, 252 ~ ano).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter
from scipy.spatial.distance import squareform
from scipy.stats import norm

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
FACTOR_TICKERS = {"IBOV": "^BVSP", "USDBRL": "BRL=X"}
BANDS_DEFAULT = {
    "curto (2-8d)": (2.0, 8.0),
    "médio (8-64d)": (8.0, 64.0),
    "longo (64d+)": (64.0, np.inf),
}
LEVEL_NAMES = {
    1: "2-4d", 2: "4-8d (~semana)", 3: "8-16d (~2 sem.)", 4: "16-32d (~mês)",
    5: "32-64d (~trimestre)", 6: "64-128d (~semestre)", 7: "128-256d (~ano)",
    8: "256-512d (~2 anos)",
}


@dataclass
class Config:
    tickers: list = field(default_factory=lambda: [
        "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "WEGE3", "BBAS3", "SUZB3"])
    pair: tuple = ("PETR4", "VALE3")
    use_factors: bool = True
    start: str = "2016-01-01"
    end: str | None = None
    outdir: str = "output_wavelet"
    cache: bool = True
    wavelet: str = "sym4"          # LA(8): boa escolha p/ MODWT em finanças
    max_levels: int = 7
    dj: float = 1 / 12             # resolução em escala da CWT
    min_period: float = 2.0
    max_period: float | None = None  # padrão: N/4
    n_surrogates: int = 200
    sig_level: float = 0.95
    max_missing_frac: float = 0.05
    ffill_limit: int = 3
    max_zero_ret_frac: float = 0.25
    winsor_k: float | None = 8.0   # em desvios robustos (MAD); None = desliga
    edge_quantile: float = 0.70    # rede: mantém pares acima deste quantil
    n_clusters: int = 3
    synthetic: bool = False
    show: bool = False
    seed: int = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# 1. Dados
# --------------------------------------------------------------------------
def normalize_ticker(t: str) -> str:
    t = t.strip().upper()
    if re.fullmatch(r"[A-Z]{4}\d{1,2}", t):  # ação B3 sem sufixo
        return t + ".SA"
    return t


def short_name(ticker: str) -> str:
    for k, v in FACTOR_TICKERS.items():
        if ticker == v:
            return k
    return ticker.replace(".SA", "")


def download_close(tickers: list, start: str, end: str | None,
                   cache_dir: Path, use_cache: bool = True,
                   retries: int = 3) -> pd.DataFrame:
    """Fechamento AJUSTADO (auto_adjust=True) com cache diário e retry."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(("|".join(sorted(tickers)) + str(start) + str(end)
                       + pd.Timestamp.today().strftime("%Y-%m-%d")
                       ).encode()).hexdigest()[:12]
    path = cache_dir / f"close_{key}.csv"
    if use_cache and path.exists():
        log(f"Cache encontrado: {path.name}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf  # import tardio: modo --synthetic não precisa

    raw, last_err = None, None
    for k in range(retries):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              progress=False, threads=True)
            if raw is not None and len(raw) > 0:
                break
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(2 * (k + 1))
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"Falha ao baixar dados do yfinance: {last_err}")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers[:1]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    empty = [c for c in close.columns if close[c].dropna().empty]
    if empty:
        log(f"AVISO: sem dados para {empty} (ticker inválido?). Removidos.")
        close = close.drop(columns=empty)
    close.to_csv(path)
    return close


def make_synthetic(cfg: Config) -> pd.DataFrame:
    """Dados sintéticos p/ teste offline: A e B compartilham ciclo de 40d
    (A lidera B em 4 pregões) na 2ª metade; C e D compartilham ciclo
    lento com o dólar; todos carregam o fator de mercado."""
    rng = np.random.default_rng(cfg.seed)
    n = 1800
    idx = pd.bdate_range("2018-01-01", periods=n)
    t = np.arange(n)
    mkt = 0.010 * rng.standard_normal(n)
    slow = np.sin(2 * np.pi * t / 120)
    cyc_a = 0.006 * np.sin(2 * np.pi * t / 40) * (t > n / 2)
    cyc_b = 0.006 * np.sin(2 * np.pi * (t - 4) / 40) * (t > n / 2)
    data = {}
    names = [normalize_ticker(x) for x in cfg.tickers]
    for k, tk in enumerate(names):
        beta = rng.uniform(0.7, 1.3)
        r = beta * mkt + 0.010 * rng.standard_normal(n)
        if k == 0:
            r = r + cyc_a
        elif k == 1:
            r = r + cyc_b
        elif k in (2, 3):
            r = r + 0.004 * slow
        data[tk] = 100 * np.exp(np.cumsum(r))
    data[FACTOR_TICKERS["IBOV"]] = 1e5 * np.exp(np.cumsum(mkt))
    usd = -0.3 * mkt + 0.003 * slow + 0.006 * rng.standard_normal(n)
    data[FACTOR_TICKERS["USDBRL"]] = 5 * np.exp(np.cumsum(usd))
    return pd.DataFrame(data, index=idx)


def prepare_returns(close: pd.DataFrame, cfg: Config):
    """Limpa preços e devolve (Z padronizado, log-retornos brutos, relatório)."""
    close = close.sort_index().copy()
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close = close[~close.index.duplicated(keep="last")]
    close = close.where(close > 0)  # preços <= 0 viram NaN

    # Calendário comum: pregões em que >= metade das séries tem preço
    # (evita que dias de mercado fechado na B3 e aberto no FX poluam).
    calendar = close.index[close.notna().mean(axis=1) >= 0.5]
    close = close.loc[calendar]

    miss = close.isna().mean()
    drop = miss[miss > cfg.max_missing_frac].index.tolist()
    if drop:
        log(f"AVISO: removidos por excesso de dados faltantes "
            f"(>{cfg.max_missing_frac:.0%}): {drop}")
        close = close.drop(columns=drop)
    close = close.ffill(limit=cfg.ffill_limit).dropna(how="any")  # inner join

    ret = np.log(close).diff().dropna()
    zero_frac = (ret.abs() < 1e-12).mean()
    illiq = zero_frac[zero_frac > cfg.max_zero_ret_frac].index.tolist()
    if illiq:
        log(f"AVISO: removidos por iliquidez (muitos retornos nulos): {illiq}")
        ret = ret.drop(columns=illiq)

    report = {"n_obs": len(ret), "start": ret.index[0], "end": ret.index[-1],
              "dropped": drop + illiq, "clipped": {}}
    if cfg.winsor_k is not None:  # contém erros de dados sem apagar crashes
        med = ret.median()
        mad = (ret - med).abs().median() * 1.4826
        lo, hi = med - cfg.winsor_k * mad, med + cfg.winsor_k * mad
        report["clipped"] = ((ret < lo) | (ret > hi)).sum().to_dict()
        ret = ret.clip(lower=lo, upper=hi, axis=1)

    Z = (ret - ret.mean()) / ret.std(ddof=0)  # coerência/correlação são
    return Z, ret, report                      # invariantes à escala


def check_stationarity(ret: pd.DataFrame) -> None:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        log("statsmodels ausente: pulando teste ADF.")
        return
    bad = [c for c in ret.columns if adfuller(ret[c].values, autolag="AIC")[1] > 0.05]
    log("ADF dos log-retornos: " + ("todos estacionários (p<0,05)."
                                    if not bad else f"NÃO estacionários: {bad}"))


def lag1(x: np.ndarray) -> float:
    return float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], -0.9, 0.9))


# --------------------------------------------------------------------------
# 2. MODWT e correlação por escala
# --------------------------------------------------------------------------
def modwt(x: np.ndarray, wavelet: str, J: int):
    """MODWT (algoritmo piramidal, filtros circulares). Retorna W (J,N), V_J."""
    w = pywt.Wavelet(wavelet)
    h = np.asarray(w.rec_hi) / np.sqrt(2)
    g = np.asarray(w.rec_lo) / np.sqrt(2)
    V = np.asarray(x, float)
    W = np.zeros((J, len(V)))
    for j in range(1, J + 1):
        step = 2 ** (j - 1)
        Wj, Vj = np.zeros_like(V), np.zeros_like(V)
        for l in range(len(h)):
            sh = np.roll(V, step * l)
            Wj += h[l] * sh
            Vj += g[l] * sh
        W[j - 1], V = Wj, Vj
    return W, V


class ModwtBank:
    """MODWT de vários ativos + correlação por escala (sem coef. de borda)."""

    def __init__(self, Z: pd.DataFrame, wavelet: str, max_levels: int,
                 min_neff: float = 10.0):
        self.names = list(Z.columns)
        self.L = len(pywt.Wavelet(wavelet).rec_lo)
        N = len(Z)
        self.J = 0
        for j in range(1, max_levels + 1):
            if (N - self.Lj(j) + 1) / 2 ** j >= min_neff:
                self.J = j
            else:
                break
        if self.J < 1:
            raise ValueError("Série curta demais para MODWT.")
        self.W = np.stack([modwt(Z[c].values, wavelet, self.J)[0] for c in self.names])
        self.N = N

    def Lj(self, j: int) -> int:
        return (2 ** j - 1) * (self.L - 1) + 1

    def valid(self, j: int) -> np.ndarray:
        return self.W[:, j - 1, self.Lj(j) - 1:]

    def corr_matrix(self, j: int) -> pd.DataFrame:
        return pd.DataFrame(np.corrcoef(self.valid(j)), index=self.names,
                            columns=self.names)

    def pair_table(self, a: str, b: str, ci: float = 0.95) -> pd.DataFrame:
        ia, ib = self.names.index(a), self.names.index(b)
        zc = norm.ppf((1 + ci) / 2)
        rows = []
        for j in range(1, self.J + 1):
            v = self.valid(j)
            r = np.corrcoef(v[ia], v[ib])[0, 1]
            neff = v.shape[1] / 2 ** j  # nº efetivo de obs. independentes
            lo = hi = np.nan
            if neff > 3:
                z, se = np.arctanh(np.clip(r, -0.9999, 0.9999)), 1 / np.sqrt(neff - 3)
                lo, hi = np.tanh(z - zc * se), np.tanh(z + zc * se)
            rows.append({"nivel": j, "periodo": LEVEL_NAMES.get(j, f"{2**j}-{2**(j+1)}d"),
                         "corr": r, "ic_inf": lo, "ic_sup": hi,
                         "n_coef": v.shape[1], "n_efetivo": neff})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Coerência wavelet (Morlet) – implementação própria (Torrence & Webster)
# --------------------------------------------------------------------------
class MorletCoherence:
    def __init__(self, n: int, dj: float = 1 / 12, min_period: float = 2.0,
                 max_period: float | None = None, omega0: float = 6.0):
        self.n, self.omega0, self.dj = n, omega0, dj
        self.ff = 4 * np.pi / (omega0 + np.sqrt(2 + omega0 ** 2))  # período/escala
        max_period = max_period or n / 4
        J = int(np.floor(np.log2(max_period / min_period) / dj))
        self.scales = (min_period / self.ff) * 2 ** (dj * np.arange(J + 1))
        self.periods = self.scales * self.ff
        self.npad = 2 ** int(np.ceil(np.log2(2 * n)))  # zero-padding (evita wrap)
        om = 2 * np.pi * np.fft.fftfreq(self.npad)
        s = self.scales[:, None]
        self.psi_hat = (np.pi ** -0.25 * np.exp(-0.5 * (s * om - omega0) ** 2)
                        * (om > 0) * np.sqrt(2 * np.pi * s))
        self.gauss = np.exp(-0.5 * (s * om) ** 2)       # suavização no tempo
        self.box = max(1, int(round(0.6 / dj)))          # suavização em escala
        d = np.minimum(np.arange(n), n - 1 - np.arange(n)) + 1
        self.coi = self.ff / np.sqrt(2) * d              # COI em unidades de período
        self.valid = self.periods[:, None] < self.coi[None, :]  # fora do COI

    def cwt(self, x: np.ndarray) -> np.ndarray:
        xp = np.zeros(self.npad)
        xp[: self.n] = x - np.mean(x)
        return np.fft.ifft(np.fft.fft(xp)[None, :] * self.psi_hat, axis=1)[:, : self.n]

    def smooth(self, A: np.ndarray) -> np.ndarray:
        Ap = np.zeros((A.shape[0], self.npad), complex)
        Ap[:, : self.n] = A
        As = np.fft.ifft(np.fft.fft(Ap, axis=1) * self.gauss, axis=1)[:, : self.n]
        return (uniform_filter1d(As.real, self.box, axis=0, mode="nearest")
                + 1j * uniform_filter1d(As.imag, self.box, axis=0, mode="nearest"))

    def auto_spectrum(self, W: np.ndarray) -> np.ndarray:
        return self.smooth(np.abs(W) ** 2 / self.scales[:, None]).real

    def coherence(self, W1, S1, W2, S2):
        """Retorna (coerência², fase = fase(1) - fase(2))."""
        S12 = self.smooth(W1 * np.conj(W2) / self.scales[:, None])
        wct = np.abs(S12) ** 2 / np.maximum(S1 * S2, 1e-30)
        return np.clip(wct, 0, 1), np.angle(S12)

    def spectral_matrix(self, Ws: list) -> np.ndarray:
        """Matriz espectral suavizada S[f,t,i,j] (hermitiana)."""
        m, F = len(Ws), len(self.scales)
        S = np.empty((F, self.n, m, m), complex)
        for i in range(m):
            for j in range(i, m):
                sij = self.smooth(Ws[i] * np.conj(Ws[j]) / self.scales[:, None])
                S[:, :, i, j] = sij
                S[:, :, j, i] = np.conj(sij)
        for i in range(m):
            S[:, :, i, i] = S[:, :, i, i].real
        return S

    def significance(self, phi1: float, phi2: float, n_surr: int = 200,
                     level: float = 0.95, seed: int = 0, step: int = 10):
        """Limiar de coerência por escala (Monte Carlo, nulo: 2 AR(1)
        independentes, apenas pontos fora do COI) e coerência média nula."""
        rng = np.random.default_rng(seed)
        F, burn = len(self.scales), 200
        tsel = np.arange(0, self.n, step)
        vals = [[] for _ in range(F)]
        for k in range(n_surr):
            xs = [lfilter([1.0], [1.0, -p], rng.standard_normal(self.n + burn))[burn:]
                  for p in (phi1, phi2)]
            W1, W2 = self.cwt(xs[0]), self.cwt(xs[1])
            wct, _ = self.coherence(W1, self.auto_spectrum(W1), W2, self.auto_spectrum(W2))
            sub, vs = wct[:, tsel], self.valid[:, tsel]
            for f in range(F):
                vals[f].append(sub[f, vs[f]].astype(np.float32))
            if (k + 1) % 50 == 0:
                log(f"  Monte Carlo: {k + 1}/{n_surr}")
        thr, nm = np.full(F, np.nan), np.full(F, np.nan)
        for f in range(F):
            v = np.concatenate(vals[f]) if vals[f] else np.array([])
            if v.size > 30:
                thr[f], nm[f] = np.quantile(v, level), v.mean()
        thr = pd.Series(thr).ffill().bfill().values
        nm = pd.Series(nm).ffill().bfill().values
        return thr, nm


def _cond(S, a, b, c):
    """Densidade espectral cruzada condicional S_ab·c."""
    Sab = S[..., a, b]
    if not c:
        return Sab
    Sac = S[..., a, :][..., c]
    Scb = S[..., c, b]
    Scc = S[..., c, :][..., :, c]
    k = len(c)
    ridge = 1e-9 * np.real(np.trace(Scc, axis1=-2, axis2=-1))[..., None, None] / k
    sol = np.linalg.solve(Scc + ridge * np.eye(k), Scb[..., None])[..., 0]
    return Sab - np.sum(Sac * sol, axis=-1)


def partial_coherence(S: np.ndarray, a: int, b: int, ctrl: list) -> np.ndarray:
    """Coerência parcial² entre a e b, removendo o efeito linear de `ctrl`."""
    Sab = _cond(S, a, b, ctrl)
    Saa, Sbb = _cond(S, a, a, ctrl).real, _cond(S, b, b, ctrl).real
    return np.clip(np.abs(Sab) ** 2 / np.maximum(Saa * Sbb, 1e-30), 0, 1)


def multiple_coherence(S: np.ndarray, y: int, X: list) -> np.ndarray:
    """Coerência múltipla²: fração da variância de y (por tempo/escala)
    explicada linearmente pelo conjunto X."""
    SXX = S[..., X, :][..., :, X]
    SyX = S[..., y, :][..., X]
    SXy = S[..., X, y]
    k = len(X)
    ridge = 1e-9 * np.real(np.trace(SXX, axis1=-2, axis2=-1))[..., None, None] / k
    sol = np.linalg.solve(SXX + ridge * np.eye(k), SXy[..., None])[..., 0]
    r2 = np.real(np.sum(SyX * sol, axis=-1)) / np.maximum(np.real(S[..., y, y]), 1e-30)
    return np.clip(r2, 0, 1)


# --------------------------------------------------------------------------
# Resumos
# --------------------------------------------------------------------------
def mean_by_period(M: np.ndarray, valid: np.ndarray, min_valid: float = 0.25):
    """Média no tempo por período, só fora do COI (NaN se pouca cobertura)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(np.where(valid, M, np.nan), axis=1)
    m[valid.mean(axis=1) < min_valid] = np.nan
    return m


def band_mean(M, valid, periods, lo, hi) -> float:
    rows = (periods >= lo) & (periods < hi)
    if not rows.any():
        return np.nan
    v = np.where(valid[rows], M[rows], np.nan)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def leadlag_table(wct, phase, thr, mc: MorletCoherence, bands: dict) -> pd.DataFrame:
    """Fase média circular nas células significativas fora do COI, por faixa.
    lag > 0  => A lidera B (em pregões)."""
    sig = (wct > thr[:, None]) if thr is not None else np.ones_like(wct, bool)
    rows = []
    for name, (lo, hi) in bands.items():
        sel = ((mc.periods >= lo) & (mc.periods < hi))[:, None] & sig & mc.valid
        if sel.sum() < 20:
            rows.append({"faixa": name, "celulas_sig": int(sel.sum()),
                         "fase_rad": np.nan, "consistencia_R": np.nan, "lag_pregoes": np.nan})
            continue
        z = np.exp(1j * phase[sel]).mean()
        per = np.broadcast_to(mc.periods[:, None], wct.shape)[sel].mean()
        rows.append({"faixa": name, "celulas_sig": int(sel.sum()),
                     "fase_rad": float(np.angle(z)), "consistencia_R": float(abs(z)),
                     "lag_pregoes": float(np.angle(z) / (2 * np.pi) * per)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------
def _period_axis(ax, periods):
    ticks = [p for p in 2.0 ** np.arange(1, 12) if periods[0] * 0.99 <= p <= periods[-1] * 1.01]
    ax.set_yticks(np.log2(ticks))
    ax.set_yticklabels([f"{int(p)}" for p in ticks])
    ax.set_ylim(np.log2(periods[0]), np.log2(periods[-1]))
    ax.set_ylabel("Período (pregões)")


def _time_axis(ax, dates):
    pos = np.linspace(0, len(dates) - 1, 8).astype(int)
    ax.set_xticks(pos)
    ax.set_xticklabels([dates[p].strftime("%Y-%m") for p in pos], rotation=0)
    ax.set_xlim(0, len(dates) - 1)


def _draw_coi(ax, mc: MorletCoherence):
    t, y = np.arange(mc.n), np.log2(mc.periods)
    c = np.log2(np.clip(mc.coi, mc.periods[0], mc.periods[-1]))
    ax.fill_between(t, c, y[-1], facecolor="none", edgecolor="white", hatch="//",
                    linewidth=0, alpha=0.8)
    ax.plot(t, c, color="white", lw=1, ls="--")


def plot_coherence_map(M, phase, thr, mc, dates, title, path, show=False):
    t, y = np.arange(mc.n), np.log2(mc.periods)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    cf = ax.contourf(t, y, M, levels=np.linspace(0, 1, 11), cmap="viridis")
    fig.colorbar(cf, ax=ax, pad=0.01, label="Coerência²")
    if thr is not None:
        ax.contour(t, y, (M > thr[:, None]).astype(float), levels=[0.5],
                   colors="white", linewidths=1.6)
    _draw_coi(ax, mc)
    if phase is not None:
        F = len(mc.periods)
        ti = np.arange(max(1, mc.n // 90), mc.n, max(1, mc.n // 45))
        si = np.arange(0, F, max(1, F // 14))
        T, S = np.meshgrid(ti, si)
        keep = mc.valid[S, T]
        if thr is not None:
            keep &= M[S, T] > thr[S]
        ax.quiver(T[keep], y[S[keep]], np.cos(phase[S, T][keep]), np.sin(phase[S, T][keep]),
                  angles="uv", pivot="mid", scale_units="width", scale=60, width=0.0022,
                  color="k")
    _period_axis(ax, mc.periods)
    _time_axis(ax, dates)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_partial_maps(maps: dict, mc, dates, path, show=False):
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5), sharey=True, squeeze=False)
    t, y = np.arange(mc.n), np.log2(mc.periods)
    for ax, (name, M) in zip(axes[0], maps.items()):
        cf = ax.contourf(t, y, M, levels=np.linspace(0, 1, 11), cmap="viridis")
        _draw_coi(ax, mc)
        _period_axis(ax, mc.periods)
        _time_axis(ax, dates)
        ax.set_title(name)
    fig.colorbar(cf, ax=axes[0].tolist(), pad=0.01, label="Coerência²")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_mean_by_scale(df: pd.DataFrame, thr, mc, title, path, show=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in df.columns:
        ax.plot(df.index, df[c], lw=2.2 if c == "WCT (bruta)" else 1.6, label=c)
    if thr is not None:
        ax.plot(mc.periods, thr, "k:", lw=1.2, label="limiar 95% (nulo AR1, pontual)")
    for p in (5, 21, 63, 252):
        if mc.periods[0] < p < mc.periods[-1]:
            ax.axvline(p, color="grey", lw=0.6, ls="--")
            ax.text(p, 0.015, {5: "semana", 21: "mês", 63: "trim.", 252: "ano"}[p],
                    ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([p for p in 2.0 ** np.arange(1, 12) if mc.periods[0] <= p <= mc.periods[-1]])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Período (pregões)")
    ax.set_ylabel("Coerência² média no tempo (fora do COI)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_modwt_pair(tab: pd.DataFrame, a: str, b: str, path, show=False):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(tab))
    ax.errorbar(x, tab["corr"], yerr=[tab["corr"] - tab["ic_inf"], tab["ic_sup"] - tab["corr"]],
                fmt="o-", capsize=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tab["periodo"], rotation=25, ha="right")
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Correlação (MODWT)")
    ax.set_title(f"Correlação por escala: {a} × {b} (IC 95% de Fisher)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_scale_heatmaps(bank: ModwtBank, path, show=False):
    J, n = bank.J, len(bank.names)
    ncols = min(4, J)
    nrows = int(np.ceil(J / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.2 * nrows), squeeze=False)
    im = None
    for j in range(1, J + 1):
        ax = axes[(j - 1) // ncols][(j - 1) % ncols]
        C = bank.corr_matrix(j).values
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(bank.names, rotation=90, fontsize=7)
        ax.set_yticklabels(bank.names, fontsize=7)
        ax.set_title(f"Escala {j}: {LEVEL_NAMES.get(j, '')}", fontsize=10)
        if n <= 12:
            for a in range(n):
                for b in range(n):
                    ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=6)
    for k in range(J, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Correlação")
    fig.suptitle("Correlação por escala (MODWT)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_band_heatmaps(mats: dict, path, show=False):
    k = len(mats)
    fig, axes = plt.subplots(1, k, figsize=(5.6 * k, 5), squeeze=False)
    for ax, (band, C) in zip(axes[0], mats.items()):
        n = len(C)
        im = ax.imshow(C.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(C.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(C.index, fontsize=7)
        ax.set_title(f"Coerência média – {band}")
        if n <= 12:
            for a in range(n):
                for b in range(n):
                    ax.text(b, a, f"{C.values[a, b]:.2f}", ha="center", va="center",
                            fontsize=6, color="w")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.7)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_networks(mats: dict, factor_names: list, cfg: Config, path, show=False):
    """Redes de co-movimento por faixa + comunidades. Retorna DataFrame de rótulos."""
    try:
        import networkx as nx
    except ImportError:
        log("networkx ausente: pulando redes (pip install networkx).")
        return pd.DataFrame()
    labels = pd.DataFrame(index=next(iter(mats.values())).index)
    k = len(mats)
    fig, axes = plt.subplots(1, k, figsize=(6.4 * k, 6), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, (band, C) in zip(axes[0], mats.items()):
        names = list(C.index)
        pairs = list(combinations(range(len(names)), 2))
        w = np.array([C.values[i, j] for i, j in pairs])
        thr_w = np.quantile(w, cfg.edge_quantile)
        G = nx.Graph()
        G.add_nodes_from(names)
        for (i, j), wt in zip(pairs, w):
            if wt >= thr_w:
                G.add_edge(names[i], names[j], weight=float(wt))
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        comm_of = {n: ci for ci, cset in enumerate(comms) for n in cset}
        labels[f"comunidade | {band}"] = [comm_of[n] for n in names]
        pos = nx.spring_layout(G, weight="weight", seed=cfg.seed, k=1.2 / np.sqrt(len(names)))
        ew = np.array([G[u][v]["weight"] for u, v in G.edges()])
        if len(ew):
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5,
                                   width=1 + 7 * (ew - ew.min()) / (np.ptp(ew) + 1e-9))
        for shape, group in (("o", [n for n in names if n not in factor_names]),
                             ("s", [n for n in names if n in factor_names])):
            if group:
                nx.draw_networkx_nodes(G, pos, nodelist=group, ax=ax, node_shape=shape,
                                       node_size=650, node_color=[cmap(comm_of[n] % 10) for n in group],
                                       edgecolors="k")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        ax.set_title(f"{band}\n(pares ≥ q{int(cfg.edge_quantile * 100)}; "
                     f"{len(comms)} comunidades; □ = fator)")
        ax.axis("off")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return labels


# --------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------
def analyse_pair(cfg, Z, mc, dates, out: Path, bands):
    a, b = [short_name(normalize_ticker(p)) for p in cfg.pair]
    factors = [f for f in FACTOR_TICKERS if cfg.use_factors and f in Z.columns]
    vars_ = [a, b] + factors
    log(f"Par: {a} × {b} | fatores de controle: {factors or 'nenhum'}")

    Ws = {v: mc.cwt(Z[v].values) for v in vars_}
    Sa = {v: mc.auto_spectrum(Ws[v]) for v in vars_}
    wct, phase = mc.coherence(Ws[a], Sa[a], Ws[b], Sa[b])

    thr = null_mean = None
    if cfg.n_surrogates > 0:
        log(f"Monte Carlo de significância ({cfg.n_surrogates} surrogates AR(1))...")
        thr, null_mean = mc.significance(lag1(Z[a].values), lag1(Z[b].values),
                                         cfg.n_surrogates, cfg.sig_level, cfg.seed)

    plot_coherence_map(
        wct, phase, thr, mc, dates,
        f"Coerência wavelet {a} × {b}  (contorno branco = 95% sig.; hachura = COI)\n"
        f"setas: → em fase, ← anti-fase, ↑ {a} lidera {b}, ↓ {b} lidera {a}", out / "fig1_coerencia_mapa.png", cfg.show)

    # Parcial / múltipla
    curves = {"WCT (bruta)": mean_by_period(wct, mc.valid)}
    maps = {f"WCT bruta {a}×{b}": wct}
    if factors:
        log("Coerência parcial/múltipla...")
        S = mc.spectral_matrix([Ws[v] for v in vars_])
        for k, f in enumerate(factors):
            M = partial_coherence(S, 0, 1, [2 + k])
            curves[f"parcial | {f}"] = mean_by_period(M, mc.valid)
            maps[f"parcial | {f}"] = M
        if len(factors) > 1:
            idx = list(range(2, 2 + len(factors)))
            M = partial_coherence(S, 0, 1, idx)
            curves["parcial | " + "+".join(factors)] = mean_by_period(M, mc.valid)
            maps["parcial | " + "+".join(factors)] = M
            curves[f"múltipla {a} ~ fatores"] = mean_by_period(multiple_coherence(S, 0, idx), mc.valid)
        else:
            curves[f"múltipla {a} ~ {factors[0]}"] = mean_by_period(multiple_coherence(S, 0, [2]), mc.valid)
        curves[f"múltipla {a} ~ {b}+fatores"] = mean_by_period(
            multiple_coherence(S, 0, [1] + list(range(2, 2 + len(factors)))), mc.valid)
        plot_partial_maps(maps, mc, dates, out / "fig2_coerencia_parcial_mapas.png", cfg.show)
        del S

    df = pd.DataFrame(curves, index=pd.Index(mc.periods, name="periodo"))
    if thr is not None:
        df["limiar95"] = thr
        df["media_nula"] = null_mean
    df.to_csv(out / "coerencia_media_por_periodo.csv")
    plot_mean_by_scale(df.drop(columns=[c for c in ("limiar95", "media_nula") if c in df]),
                       thr, mc, f"Coerência média por escala: {a} × {b}",
                       out / "fig3_coerencia_media_por_escala.png", cfg.show)

    ll = leadlag_table(wct, phase, thr, mc, bands)
    ll.to_csv(out / "lead_lag_por_faixa.csv", index=False)

    # Resumo de faixa: bruta vs. parcial
    resumo = {n: {k: band_mean(M, mc.valid, mc.periods, lo, hi) for k, M in maps.items()}
              for n, (lo, hi) in bands.items()}
    print("\n=== Coerência² média por faixa (bruta vs. parcial) ===")
    print(pd.DataFrame(resumo).T.round(3).to_string())
    print("\n=== Lead-lag por faixa (lag>0: %s lidera %s; R = consistência da fase) ===" % (a, b))
    print(ll.round(3).to_string(index=False))
    return thr, null_mean


def analyse_multi(cfg, Z, mc, thr, null_mean, out: Path, bands):
    cols = list(Z.columns)
    factor_names = [c for c in cols if c in FACTOR_TICKERS]
    log(f"Coerência média por faixa para {len(cols) * (len(cols) - 1) // 2} pares...")
    Ws = {c: mc.cwt(Z[c].values) for c in cols}
    Sa = {c: mc.auto_spectrum(Ws[c]) for c in cols}
    mats = {n: pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols) for n in bands}
    for i, j in combinations(range(len(cols)), 2):
        wct, _ = mc.coherence(Ws[cols[i]], Sa[cols[i]], Ws[cols[j]], Sa[cols[j]])
        for n, (lo, hi) in bands.items():
            v = band_mean(wct, mc.valid, mc.periods, lo, hi)
            mats[n].iloc[i, j] = mats[n].iloc[j, i] = v
    for n, C in mats.items():
        C.to_csv(out / f"matriz_coerencia_{re.sub(r'[^A-Za-z0-9]+', '_', n)}.csv")
    if null_mean is not None:
        base = {n: float(np.nanmean(null_mean[(mc.periods >= lo) & (mc.periods < hi)]))
                for n, (lo, hi) in bands.items()}
        print("\nCoerência média sob o nulo (séries independentes), por faixa:",
              {k: round(v, 3) for k, v in base.items()})

    plot_band_heatmaps(mats, out / "fig4_coerencia_por_faixa_heatmap.png", cfg.show)
    labels = build_networks(mats, factor_names, cfg, out / "fig5_redes_comovimento.png", cfg.show)

    # Clusters hierárquicos + diversificação (só ativos, sem fatores)
    assets = [c for c in cols if c not in factor_names]
    tab = labels.loc[assets].copy() if len(labels) else pd.DataFrame(index=assets)
    for n, C in mats.items():
        Ca = C.loc[assets, assets]
        tab[f"cluster | {n}"] = 1
        if len(assets) > 2:
            D = np.sqrt(np.clip(1 - Ca.values, 0, None))
            np.fill_diagonal(D, 0)
            Zl = linkage(squareform((D + D.T) / 2, checks=False), "average")
            tab[f"cluster | {n}"] = fcluster(Zl, t=min(cfg.n_clusters, len(assets)), criterion="maxclust")
        tab[f"coerência média c/ demais | {n}"] = (Ca.sum(axis=1) - 1) / (len(assets) - 1)
    tab.to_csv(out / "clusters_e_diversificacao_por_faixa.csv")
    print("\n=== Menor coerência média com a carteira = melhor diversificador no horizonte ===")
    for n in bands:
        s = tab[f"coerência média c/ demais | {n}"].sort_values()
        print(f"  {n}: mais diversificadores -> {list(s.index[:3])}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Co-movimento por wavelets (yfinance).")
    d = Config()
    p.add_argument("--tickers", nargs="+", default=d.tickers)
    p.add_argument("--pair", nargs=2, default=list(d.pair))
    p.add_argument("--no-factors", action="store_true", help="sem Ibovespa/Dólar")
    p.add_argument("--start", default=d.start)
    p.add_argument("--end", default=None)
    p.add_argument("--outdir", default=d.outdir)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--wavelet", default=d.wavelet)
    p.add_argument("--levels", type=int, default=d.max_levels)
    p.add_argument("--max-period", type=float, default=None)
    p.add_argument("--n-surrogates", type=int, default=d.n_surrogates)
    p.add_argument("--edge-quantile", type=float, default=d.edge_quantile)
    p.add_argument("--clusters", type=int, default=d.n_clusters)
    p.add_argument("--synthetic", action="store_true", help="dados simulados (teste offline)")
    p.add_argument("--show", action="store_true")
    p.add_argument("--seed", type=int, default=d.seed)
    a = p.parse_args()
    return Config(tickers=a.tickers, pair=tuple(a.pair), use_factors=not a.no_factors,
                  start=a.start, end=a.end, outdir=a.outdir, cache=not a.no_cache,
                  wavelet=a.wavelet, max_levels=a.levels, max_period=a.max_period,
                  n_surrogates=a.n_surrogates, edge_quantile=a.edge_quantile,
                  n_clusters=a.clusters, synthetic=a.synthetic, show=a.show, seed=a.seed)


def main():
    cfg = parse_args()
    np.random.seed(cfg.seed)
    out = Path(cfg.outdir)
    out.mkdir(parents=True, exist_ok=True)

    tick = [normalize_ticker(t) for t in cfg.tickers]
    pair = [normalize_ticker(t) for t in cfg.pair]
    for p_ in pair:
        if p_ not in tick:
            tick.append(p_)
    all_tick = tick + (list(FACTOR_TICKERS.values()) if cfg.use_factors else [])

    if cfg.synthetic:
        log("Modo SINTÉTICO (dados simulados).")
        close = make_synthetic(cfg)
    else:
        log(f"Baixando {len(all_tick)} séries do yfinance...")
        close = download_close(all_tick, cfg.start, cfg.end, out / "cache", cfg.cache)
    close = close.rename(columns=short_name)

    Z, ret, rep = prepare_returns(close, cfg)
    log(f"Amostra: {rep['n_obs']} retornos ({rep['start']:%Y-%m-%d} a {rep['end']:%Y-%m-%d}); "
        f"{Z.shape[1]} séries.")
    if rep["clipped"] and sum(rep["clipped"].values()):
        log(f"Outliers contidos (winsor {cfg.winsor_k} MAD): "
            f"{ {k: v for k, v in rep['clipped'].items() if v} }")
    if rep["n_obs"] < 500:
        log("AVISO: < 500 observações; escalas longas ficarão no COI/pouco confiáveis.")
    for p_ in cfg.pair:
        if short_name(normalize_ticker(p_)) not in Z.columns:
            raise SystemExit(f"Ativo do par ausente após limpeza: {p_}")
    check_stationarity(ret)
    ret.to_csv(out / "log_retornos.csv")

    dates = Z.index
    bands = BANDS_DEFAULT

    # --- MODWT
    bank = ModwtBank(Z, cfg.wavelet, cfg.max_levels)
    log(f"MODWT ({cfg.wavelet}) com {bank.J} níveis.")
    a, b = [short_name(normalize_ticker(p_)) for p_ in cfg.pair]
    tab = bank.pair_table(a, b)
    tab.to_csv(out / "modwt_correlacao_par.csv", index=False)
    print("\n=== Correlação por escala (MODWT):", a, "×", b, "===")
    print(tab.round(3).to_string(index=False))
    plot_modwt_pair(tab, a, b, out / "fig0_modwt_correlacao_par.png", cfg.show)
    plot_scale_heatmaps(bank, out / "fig6_modwt_heatmaps_por_escala.png", cfg.show)

    # --- CWT
    mc = MorletCoherence(len(Z), cfg.dj, cfg.min_period, cfg.max_period)
    log(f"CWT Morlet: {len(mc.periods)} escalas, períodos {mc.periods[0]:.1f}-{mc.periods[-1]:.0f} pregões.")
    thr, null_mean = analyse_pair(cfg, Z, mc, dates, out, bands)
    analyse_multi(cfg, Z, mc, thr, null_mean, out, bands)
    log(f"Concluído. Resultados em: {out.resolve()}")


if __name__ == "__main__":
    main()
