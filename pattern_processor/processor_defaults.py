"""The pattern processor's defaults, in one place.

``DEFAULTS`` is the single owner of every start value of the processor's control panel
(``gpu_pattern_viewer.MainWindow._build_ui`` reads it when it builds the widgets), of the
slider scale factors that turn an integer slider position into the physical value, and of the
rules that size work to the machine's RAM. The helper scripts in this folder (``run_nlpar.py``,
``selftest.py``) read it too, and ``check_defaults.py`` in the repository root builds the GUI
headless and asserts that every widget starts at the value held here.

This module imports nothing heavy on purpose (no Qt, no numpy), so a script can read a default
without loading the viewer.

Change a default here, nowhere else. The mode names (``aperture_modes``, ``bg_modes``) are matched
by name in the viewer's code, so they are a vocabulary rather than a default: renaming one means
editing the viewer too.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessorDefaults:
    # ---- Aperture mask (applied before all processing) ----
    aperture_modes: tuple = ("None", "Circular (EBSD)", "Auto-detect")
    aperture_mode: str = "None"
    # Auto-detect: a pixel is active when its std across the data set exceeds this fraction of the
    # ``variance_percentile``-th percentile of the per-pixel std (a percentile rather than the max,
    # so hot pixels do not set the scale). The global stretch uses the same rule for its active set.
    aperture_threshold: float = 0.05
    variance_percentile: float = 95.0

    # ---- IQ range (patterns outside it are left unprocessed) ----
    # The two sliders run 0..iq_slider_max over the data set's IQ range; the default is the whole range.
    iq_slider_max: int = 1000
    iq_range: tuple = (0, 1000)

    # ---- NPAR ----
    npar_radius: int = 1

    # ---- NLPAR (pyebsdindex) ----
    search_radius: int = 3
    # Canonical NLPAR lambda is dimensionless (after the sigma^2 normalisation); optimal values per
    # Brewick et al. 2019 are typically 0.6-1.2, and Optimize Lambda replaces this value anyway.
    lam: float = 1.0
    saturation_protect: bool = True     # pyebsdindex default
    dthresh: float = 0.0                # pyebsdindex default, 0 = off

    # ---- Static background ----
    static_bg: bool = False
    bg_modes: tuple = ("Subtract", "Divide")
    bg_mode: str = "Subtract"

    # ---- Dynamic background ----
    dyn_bg: bool = False
    dyn_sigma: float = 5.0              # Gaussian sigma in pixels
    sigma_slider_scale: int = 10        # slider position = sigma * scale

    # ---- Normalisation ----
    per_pattern_stretch: bool = True    # per-pattern min-max stretch to [0, 1]
    global_stretch: bool = False        # one (lo, hi) for the whole data set, from these percentiles
    global_percentiles: tuple = (0.5, 99.5)

    # ---- Brightness / contrast (slider units, -100..100) ----
    brightness: int = 0
    contrast: int = 0

    # ---- CLAHE ----
    clahe: bool = False
    clahe_clip: float = 0.010           # skimage equalize_adapthist clip_limit
    clip_slider_scale: int = 1000       # slider position = clip_limit * scale

    # ---- Memory rules (fractions of the RAM available to this process, see available_ram) ----
    ram_fit_fraction: float = 0.6       # a data set (or a second NLPAR copy of it) is held in RAM when it needs at most this much
    ram_chunk_fraction: float = 0.25    # one chunk of a whole-data-set pass may use at most this much
    chunk_cap_bytes: float = 16e9       # and never more than this (pyebsdindex NLPAR tiles)
    ram_fallback_bytes: float = 8e9     # assumed available when psutil is missing

    def global_percentile_text(self) -> str:
        """``"0.5-99.5"``: the percentile pair as the control panel and the log print it."""
        lo, hi = self.global_percentiles
        return f"{lo:g}–{hi:g}"


DEFAULTS = ProcessorDefaults()
