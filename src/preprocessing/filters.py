"""
filters.py
==========
Stage 3 (Req D): the single broadband cleanup filter.

One 0.5-45 Hz zero-phase Butterworth band-pass (order 4, applied forward+backward
via SciPy's sosfiltfilt) that removes DC drift and high-frequency / EMG noise.
This is a CLEANUP step, NOT a feature step -- the retired 5-band filter bank is
gone. A 60 Hz mains notch is provided for reproducibility but is INERT by
default: 60 Hz already sits deep in the stop-band of the 0.5-45 Hz band-pass
(order-4 filtfilt leaves only ~9% of a 60 Hz component), so applying it changes
almost nothing. It is exposed only for wider-band experiments.

SciPy is imported LAZILY inside the filtering functions, so this module (and the
whole package) can be imported without SciPy present. SciPy is a real runtime
dependency (see requirements.txt) needed to actually filter signals.

Conventions
-----------
* Signals are float arrays shaped (n_channels, n_samples); filtering runs along
  the LAST axis by default, matching EdfRecording.data (18, n_samples).
* All filtering is zero-phase (no group delay), so sample-accurate seizure
  timing is preserved -- essential because RSMMTN is order-sensitive.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

_SCIPY_AVAILABLE: bool = importlib.util.find_spec("scipy") is not None


def _require_scipy():
    """Import scipy.signal lazily, with an actionable error if it is missing."""
    try:
        from scipy import signal  # noqa: WPS433 (intentional lazy import)
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "filters.py needs SciPy to filter signals. Install it with "
            "`pip install scipy` (it is pinned in requirements.txt)."
        ) from exc
    return signal


def _validate_band(fs: float, low: float, high: float) -> None:
    """Pure-Python validation of a band-pass spec (no SciPy needed)."""
    nyq = fs / 2.0
    if not (0.0 < low < high < nyq):
        raise ValueError(
            f"invalid band-pass: need 0 < low < high < Nyquist ({nyq} Hz), "
            f"got low={low}, high={high}"
        )


# ---------------------------------------------------------------------------
# Filter design
# ---------------------------------------------------------------------------
def design_bandpass_sos(
    fs: float = cfg.FS,
    low: float = cfg.BANDPASS_LOW_HZ,
    high: float = cfg.BANDPASS_HIGH_HZ,
    order: int = cfg.FILTER_ORDER,
) -> np.ndarray:
    """Return second-order-sections (SOS) for the cleanup band-pass.

    SOS form is used (instead of b/a) for numerical stability of the
    higher-order band-pass.
    """
    _validate_band(fs, low, high)
    signal = _require_scipy()
    sos = signal.butter(order, [low, high], btype="band", output="sos", fs=fs)
    return np.asarray(sos, dtype=np.float64)


def design_notch(
    fs: float = cfg.FS,
    freq: float = cfg.NOTCH_FREQ_HZ,
    quality: float = cfg.NOTCH_QUALITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (b, a) for the (inert-by-default) mains notch."""
    if not (0.0 < freq < fs / 2.0):
        raise ValueError(f"notch freq {freq} must be in (0, Nyquist={fs/2})")
    signal = _require_scipy()
    b, a = signal.iirnotch(w0=freq, Q=quality, fs=fs)
    return np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)


# ---------------------------------------------------------------------------
# Filter application (zero-phase)
# ---------------------------------------------------------------------------
def bandpass(
    data: np.ndarray,
    fs: float = cfg.FS,
    low: float = cfg.BANDPASS_LOW_HZ,
    high: float = cfg.BANDPASS_HIGH_HZ,
    order: int = cfg.FILTER_ORDER,
    axis: int = -1,
) -> np.ndarray:
    """Zero-phase band-pass filter along `axis`."""
    signal = _require_scipy()
    sos = design_bandpass_sos(fs=fs, low=low, high=high, order=order)
    x = np.asarray(data, dtype=np.float64)
    return signal.sosfiltfilt(sos, x, axis=axis)


def notch(
    data: np.ndarray,
    fs: float = cfg.FS,
    freq: float = cfg.NOTCH_FREQ_HZ,
    quality: float = cfg.NOTCH_QUALITY,
    axis: int = -1,
) -> np.ndarray:
    """Zero-phase mains notch along `axis` (inert within the 0.5-45 Hz band)."""
    signal = _require_scipy()
    b, a = design_notch(fs=fs, freq=freq, quality=quality)
    x = np.asarray(data, dtype=np.float64)
    return signal.filtfilt(b, a, x, axis=axis)


def clean(
    data: np.ndarray,
    fs: float = cfg.FS,
    *,
    low: float = cfg.BANDPASS_LOW_HZ,
    high: float = cfg.BANDPASS_HIGH_HZ,
    order: int = cfg.FILTER_ORDER,
    apply_notch: bool = False,
    notch_freq: float = cfg.NOTCH_FREQ_HZ,
    notch_quality: float = cfg.NOTCH_QUALITY,
    axis: int = -1,
) -> np.ndarray:
    """Full Stage-3 cleanup: optional (inert) notch, then the band-pass.

    `apply_notch` defaults to False because within a 0.5-45 Hz passband the
    60 Hz notch is redundant; it is retained only for reproducibility / wider
    passbands. Returns a float64 array of the same shape as `data`.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError(f"expected 1-D or 2-D array, got shape {x.shape}")
    if apply_notch:
        x = notch(x, fs=fs, freq=notch_freq, quality=notch_quality, axis=axis)
    x = bandpass(x, fs=fs, low=low, high=high, order=order, axis=axis)
    log.debug("clean: filtered array shape=%s (notch=%s)", x.shape, apply_notch)
    return x


# ---------------------------------------------------------------------------
# Self-test  (dual-mode: full numeric with SciPy, structural without)
# ---------------------------------------------------------------------------
def _amp_at(sig: np.ndarray, fs: float, f: float, half: int = 2) -> float:
    """Approximate single-sided amplitude of `sig` at frequency `f` (numpy FFT)."""
    n = sig.shape[-1]
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    idx = int(np.argmin(np.abs(freqs - f)))
    lo, hi = max(0, idx - half), min(len(freqs), idx + half + 1)
    return float(np.max(np.abs(spec[lo:hi])) * 2.0 / n)


def _make_signal(fs: float, seconds: float, comps: Sequence[tuple[float, float]]) -> np.ndarray:
    t = np.arange(int(round(seconds * fs))) / fs
    if not comps:
        return np.zeros_like(t)
    signals = [a * np.sin(2 * np.pi * f * t) for f, a in comps]
    return np.sum(signals, axis=0)


if __name__ == "__main__":
    print("Running filters.py self-test ...\n")

    # --- structural checks (no SciPy required) ---
    for bad in [(-1.0, 45.0), (45.0, 45.0), (45.0, 0.5), (0.5, 200.0)]:
        try:
            _validate_band(cfg.FS, *bad)
            raise AssertionError(f"expected ValueError for band {bad}")
        except ValueError:
            pass
    _validate_band(cfg.FS, cfg.BANDPASS_LOW_HZ, cfg.BANDPASS_HIGH_HZ)  # ok
    assert cfg.BANDPASS_LOW_HZ == 0.5 and cfg.BANDPASS_HIGH_HZ == 45.0
    assert cfg.BANDPASS_HIGH_HZ <= cfg.NOTCH_FREQ_HZ  # notch outside passband -> inert
    print("Structural checks passed (band validation + config wiring).")

    if not _SCIPY_AVAILABLE:
        # Prove the lazy-import contract: filtering raises a clear error.
        try:
            bandpass(np.zeros(1024))
            raise AssertionError("expected ImportError without SciPy")
        except ImportError:
            pass
        print("\nSciPy not installed in this environment -- numeric frequency")
        print("response checks were SKIPPED. They run automatically wherever")
        print("SciPy is available (it is pinned in requirements.txt).")
        print("\nAll filters.py structural self-tests passed.")
        raise SystemExit(0)

    # --- numeric checks (SciPy present) ---
    fs = cfg.FS
    # drift 0.25 Hz (below band), 10 Hz (in band), 60 Hz (mains), 90 Hz (above band)
    comps = [(0.25, 1.0), (10.0, 1.0), (60.0, 1.0), (90.0, 1.0)]
    x = _make_signal(fs, 16.0, comps)

    y = clean(x, fs=fs)  # band-pass only (notch inert/off)
    assert y.shape == x.shape, (y.shape, x.shape)

    ratio = {f: _amp_at(y, fs, f) / _amp_at(x, fs, f) for f, _ in comps}
    print("band-pass retention ratios:",
          {f: round(r, 4) for f, r in ratio.items()})
    assert ratio[10.0] > 0.85, ratio[10.0]     # in-band preserved
    assert ratio[0.25] < 0.05, ratio[0.25]     # DC drift killed
    assert ratio[90.0] < 0.05, ratio[90.0]     # HF killed
    assert ratio[60.0] < 0.20, ratio[60.0]     # 60 Hz already ~90% gone -> notch inert

    # zero-phase => no group delay. The old reverse-identity check
    # (clean(x[::-1])[::-1] == clean(x)) fails at filtfilt's padded edges even
    # though the filter is genuinely zero-phase, so test the delay DIRECTLY:
    # filter a pure in-band tone and confirm the input/output cross-correlation
    # peaks at lag 0. Interior only (first/last second trimmed for transients);
    # search is limited to under one tone period to avoid periodic aliasing.
    tone = _make_signal(fs, 16.0, [(10.0, 1.0)])
    tone_f = clean(tone, fs=fs)
    edge = int(fs)  # drop 1 s of transient at each end
    a = tone[edge:-edge] - np.mean(tone[edge:-edge])
    b = tone_f[edge:-edge] - np.mean(tone_f[edge:-edge])
    xcorr = np.correlate(b, a, mode="full")
    center = len(a) - 1
    search = fs // 10  # < one period of the 10 Hz probe tone
    seg = xcorr[center - search:center + search + 1]
    lag = int(np.argmax(seg)) - search
    assert abs(lag) <= 1, f"nonzero group delay: lag={lag} samples"

    # explicit notch further reduces 60 Hz
    y_notched = clean(x, fs=fs, apply_notch=True)
    r60_notch = _amp_at(y_notched, fs, 60.0) / _amp_at(x, fs, 60.0)
    assert r60_notch <= ratio[60.0] + 1e-9, (r60_notch, ratio[60.0])

    # 2-D (channels x samples) path, matching EdfRecording.data
    X = np.stack([x, 0.5 * x])
    Y = clean(X, fs=fs)
    assert Y.shape == X.shape
    assert np.allclose(Y[1], 0.5 * Y[0], atol=1e-6), "per-channel linearity broken"

    print("\nAll filters.py self-tests passed (numeric + structural).")
