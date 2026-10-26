"""Spherical indexing: the functions behind the Kikuchi workbench's indexing and montage tabs.

The indexing itself is the kikuchipy fork's ``EBSD.spherical_indexing`` (the EMSphInx port with its CuPy backend).
This module wraps each step of a run (load, preprocess, harmonics, detector, index, save, compare) as one function
driven by a :class:`Settings` object, plus the pieces a GUI needs around it: a quasi image quality map, pattern
previews as the indexer sees them, IPF colouring and the handling of orix crystal maps with masked points. Nothing
in here imports Qt, so the module can be used from a script too::

    import sph_core
    st = sph_core.Settings(PATTERN_FILE="scan.up1", SCAN_FILE="scan.ang", MASTER_FILE="Ni.sht", CAMERA_TILT=8)
    s, info = sph_core.load_patterns(st)
    sph_core.preprocess(s, st)
    harmonics = sph_core.load_harmonics(st)
    det = sph_core.make_detector(st, s, harmonics)
    xmap, xmap_indexed, dt = sph_core.index(s, harmonics, det, st)
    sph_core.save_xmap(xmap, "results", "scan_sph_bw68")
    # one pattern, and the master pattern at the orientation found (the workbench's 'Index this point')
    res = sph_core.index_pattern(sph_core.make_point_indexer(harmonics, det, st), s.data[10, 20])
    simulated = sph_core.simulate_pattern(sph_core.lambert_master(harmonics[0]), res["rotation"], det)

:func:`index_blocks` gives the same map band by band (bands of whole map rows) with progress events, for a GUI;
``kikuchi_workbench.py`` and ``montage_tab.py`` drive their live views from it. A preset file (``NAME = value``
lines, values as Python literals) holds settings; :func:`read_preset` reads one and :func:`write_preset` writes one.

Conventions used throughout: map shapes are (rows, cols); an ROI is (col_start, col_stop, row_start, row_stop) as
in the GUI; angles are degrees unless a function says otherwise; Euler angles of the fork's packed rows are ZYZ in
radians; a phase id of -1 is a pattern that was not indexed (masked out or failed).
"""
from __future__ import annotations

import ast
import dataclasses
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from dask.callbacks import Callback

#: A logging callback: every function that reports progress takes ``log: Log`` and defaults to printing.
Log = Callable[[str], None]


def _print(msg: str) -> None:
    """The default ``log``: print flushed, so a worker thread's output appears in order in a console."""
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------


@dataclass
class Settings:
    """Every setting of the spherical indexing. The names are those of the GUI's form and of the preset files.

    Files
        ``PATTERN_FILE``: the pattern file, any format kikuchipy reads (``.up1``/``.up2``, h5ebsd files including
        EDAX ``.edaxh5``/``.oh5``); ``None`` loads kikuchipy's nickel demo data set. ``SCAN_FILE``: the ``.ang``
        or ``.edaxh5`` the scan parameters are read from (``ebsd_params.scan_info``); it supplies the map size and
        step of an old ``.up1`` whose header has neither. ``EDAXH5_MAP``: part of the name (case insensitive) of
        the map to use in a project with several maps.
    Patterns
        ``NAV_SHAPE``: (rows, cols) for a ``.up1`` without map size when there is no scan file; ``STEP_SIZE``:
        (step_x, step_y) in um, likewise. ``LAZY``: ``True``, ``False`` or ``"auto"`` (memory mapped when the
        file is larger than a quarter of the RAM). ``ROI``: (col_start, col_stop, row_start, row_stop) to index
        part of the map. ``FLIP_PATTERNS_UD``: flip every pattern upside down after loading.
    Processing
        ``REMOVE_STATIC_BACKGROUND``, ``REMOVE_DYNAMIC_BACKGROUND``: kikuchipy's background removals, done by
        :func:`preprocess` (the GUI unticks both by default, the processor having done them already).
    Master pattern
        ``MASTER_FILE``: an EMSphInx ``.sht`` or an EMsoft master pattern ``.h5``, or a list of them for several
        phases (phase id = position in the list). ``MASTER_BANDWIDTH``: bandwidth of the harmonics computed from an
        EMsoft ``.h5`` (an ``.sht`` holds its own). ``MASTER_ENERGY``: beam energy in keV to take from an EMsoft
        ``.h5``, ``None`` for the reader's default. ``OVERRIDE_MASTER_SAMPLE_TILT``: allow ``SAMPLE_TILT`` to
        differ from the tilt the master was simulated for.
    Geometry
        ``PC``: projection center (x, y, z) in the ``PC_CONVENTION`` convention (``"edax"``, ``"bruker"``,
        ``"oxford"``, ``"emsoft"``, ... as kikuchipy's ``EBSDDetector``). ``SAMPLE_TILT``: degrees, ``None`` takes
        the master pattern's. ``CAMERA_TILT``: detector tilt in degrees (EMSphInx ``thetac``; about 8 for an EBSD
        camera, -90 for RKD). ``PX_SIZE``: detector pixel size in um, before ``BINNING``. ``TWIST``: rotation of
        the detector about its normal in degrees; the indexed orientations are rotated by it about the sample
        normal afterwards (:func:`apply_twist`).
    Indexing
        ``BANDWIDTH`` of the spherical harmonic transform; ``BACKEND`` ``"gpu"`` or ``"cpu"``; ``N_WORKERS``
        dask threads; ``REFINE`` Newton refinement of the best candidate; ``N_REGIONS`` tiles of the adaptive
        histogram equalization (0 = off); ``CIRCULAR_MASK``, ``GAUSSIAN_BACKGROUND``, ``NORMALIZE`` as the fork's
        keywords; ``CHUNKSIZE`` patterns per dask chunk and GPU batch, ``None`` = automatic; ``TEST_SIZE`` side of
        the square test region in map points.
    Output
        ``OUTPUT_DIR`` and ``OUTPUT_NAME`` (see :func:`output_location`), ``SAVE_ANG``, ``SAVE_H5``,
        ``REFERENCE_FILE`` for :func:`compare_with_reference`.
    """

    # files
    PATTERN_FILE: str | None = None
    SCAN_FILE: str | None = None
    EDAXH5_MAP: str | None = None
    # patterns
    NAV_SHAPE: tuple | None = None
    STEP_SIZE: tuple | None = None
    LAZY: bool | str = "auto"
    ROI: tuple | None = None
    FLIP_PATTERNS_UD: bool = False
    # processing
    REMOVE_STATIC_BACKGROUND: bool = True
    REMOVE_DYNAMIC_BACKGROUND: bool = True
    # master
    MASTER_FILE: str | list | None = None
    MASTER_BANDWIDTH: int = 384
    MASTER_ENERGY: float | None = None
    OVERRIDE_MASTER_SAMPLE_TILT: bool = False
    # geometry
    PC: tuple = (0.5, 0.5, 0.5)
    PC_CONVENTION: str = "edax"
    SAMPLE_TILT: float | None = None
    CAMERA_TILT: float = 0.0
    PX_SIZE: float = 1.0
    BINNING: int = 1
    TWIST: float = 0.0
    # indexing
    BANDWIDTH: int = 68
    BACKEND: str = "gpu"
    N_WORKERS: int = 8
    REFINE: bool = True
    N_REGIONS: int = 10
    CIRCULAR_MASK: bool = False
    GAUSSIAN_BACKGROUND: bool = False
    NORMALIZE: bool = True
    CHUNKSIZE: int | None = None
    TEST_SIZE: int = 50
    # output
    OUTPUT_DIR: str | None = None
    OUTPUT_NAME: str | None = None
    SAVE_ANG: bool = True
    SAVE_H5: bool = True
    REFERENCE_FILE: str | None = None

    def names(self) -> list[str]:
        """The setting names, in declaration order (the order of a written preset)."""
        return [f.name for f in dataclasses.fields(self)]


_PRESET_LINE = re.compile(r"^([A-Z][A-Z0-9_]*) = (.*)$")
_PRESET_COMMENT = re.compile(r"\s+#.*$")


def _preset_value(text: str):
    """The Python literal of a preset value. A trailing ``  # comment`` is only stripped when the whole text is
    not a literal, so a path containing ' #' survives. Raises ValueError or SyntaxError for anything else."""
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return ast.literal_eval(_PRESET_COMMENT.sub("", text).strip())


def read_preset(path: str | Path, settings: Settings | None = None) -> tuple[Settings, list[str]]:
    """Settings from a preset file of ``NAME = value`` lines (values are Python literals).

    Names that are set start from a copy of ``settings`` (a default :class:`Settings` when not given); lines that
    do not look like ``NAME = value`` (comments, blank lines) are skipped. Returns the settings and the names in
    the file that were ignored: names that are not settings, and settings whose value could not be read (listed
    as ``"NAME (value not understood: ...)"``).
    """
    settings = dataclasses.replace(settings) if settings is not None else Settings()
    ignored = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _PRESET_LINE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        if name not in settings.names():
            ignored.append(name)
            continue
        try:
            setattr(settings, name, _preset_value(value))
        except (ValueError, SyntaxError):
            ignored.append(f"{name} (value not understood: {value})")
    return settings, ignored


def write_preset(path: str | Path, settings: Settings, header: str = "", fields: Sequence[str] | None = None) -> None:
    """Write the settings as a preset file (``NAME = value`` lines, values as Python source), overwriting ``path``.

    ``header`` is written first as ``#`` comment lines; ``fields`` restricts the file to those setting names
    (all of them when not given).
    """
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    lines.append("# Preset written by kikuchi_workbench.py: load it with 'Load preset' or --preset")
    for name in settings.names() if fields is None else fields:
        # repr escapes backslashes, and ast.literal_eval reads them back as they were (a raw-string prefix on a
        # repr would double every backslash of a Windows path at each round trip)
        lines.append(f"{name} = {getattr(settings, name)!r}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------------------------------

_HDF5_SUFFIXES = (".h5", ".hdf5", ".h5ebsd", ".edaxh5", ".oh5")


def _lazy_decision(pattern_file: Path, lazy) -> bool:
    """Whether to load ``pattern_file`` lazily (memory mapped): ``lazy`` itself, or for ``"auto"`` when the file is
    larger than a quarter of the machine's RAM (larger than 4 GB when psutil is not installed)."""
    if lazy == "auto":
        try:
            import psutil

            return pattern_file.stat().st_size > 0.25 * psutil.virtual_memory().total
        except ImportError:
            return pattern_file.stat().st_size > 4e9
    return bool(lazy)


def load_patterns(st: Settings, scan: dict | None = None, log: Log = _print):
    """The patterns as a kikuchipy EBSD signal, with the scan parameters, ROI and flip of ``st`` applied.

    ``scan`` is the dict of ``ebsd_params.scan_info()`` (read from ``SCAN_FILE`` here when not given, which also
    works without a scan file). For a ``.up1``/``.up2`` whose header has no map size (``up_has_map_size`` false)
    it supplies the (rows, cols) and the step, unless ``NAV_SHAPE``/``STEP_SIZE`` are set; hexagonal grids are
    left to the reader. An HDF5 pattern file is read by content (see :func:`_load_h5ebsd`), the scan group chosen
    by ``EDAXH5_MAP``. ``PATTERN_FILE = None`` downloads and loads kikuchipy's nickel demo data set.

    Returns ``(signal, scan)``: the signal in RAM or lazy (dask) as :func:`_lazy_decision` says, its navigation
    axes scaled in um when a step is known, and the scan dict (``{}`` for the demo). Logs what was loaded.
    """
    import kikuchipy as kp

    if st.PATTERN_FILE is None:
        log("Loading kikuchipy's nickel demo data set")
        s = kp.data.nickel_ebsd_large(allow_download=True)
        scan = scan or {}
    else:
        pattern_file = Path(st.PATTERN_FILE)
        suffix = pattern_file.suffix.lower()
        if scan is None:
            import ebsd_params

            scan = ebsd_params.scan_info(st.SCAN_FILE, pattern_file=str(pattern_file), map_filter=st.EDAXH5_MAP)
        lazy = _lazy_decision(pattern_file, st.LAZY)
        kwargs = {"lazy": lazy}
        nav_shape_in, step_size = st.NAV_SHAPE, st.STEP_SIZE
        h5_group = None
        if suffix in _HDF5_SUFFIXES:
            h5_group = _h5_scan_group(pattern_file, st.EDAXH5_MAP)
            if h5_group is not None:
                kwargs["scan_group_names"] = h5_group
        if suffix in (".up1", ".up2"):
            if not scan.get("up_has_map_size"):
                if nav_shape_in is None and scan.get("n_rows") and scan.get("grid") != "HexGrid":
                    nav_shape_in = (scan["n_rows"], scan["n_cols"])
                if step_size is None and scan.get("step_x"):
                    step_size = (scan["step_x"], scan["step_y"])
            if nav_shape_in is not None:
                kwargs["nav_shape"] = tuple(int(v) for v in nav_shape_in)
        log(f"Loading {pattern_file.name} (lazy: {lazy})" + (f", scan group {h5_group!r}" if h5_group else ""))
        s = _load_h5ebsd(pattern_file, **kwargs) if suffix in _HDF5_SUFFIXES else kp.load(pattern_file, **kwargs)
        if step_size is not None:
            for axis, step in zip(s.axes_manager.navigation_axes, step_size):
                axis.scale = float(step)
                axis.units = "um"
    s = apply_roi_and_flip(s, st)
    nav_shape = s.axes_manager.navigation_shape[::-1]
    sig_shape = s.axes_manager.signal_shape[::-1]
    log(f"{int(np.prod(nav_shape))} patterns, map {tuple(nav_shape)} (rows, cols), pattern {tuple(sig_shape)} px, "
        f"lazy: {s._lazy}")
    return s, scan


def _h5_scan_group(path: Path, map_filter: str | None) -> str | None:
    """The top-level scan group of an h5ebsd pattern file selected by ``map_filter`` (part of its name, case
    insensitive), or None for the reader's default (the first scan) when there is no filter, no match, or the
    file cannot be opened."""
    if not map_filter:
        return None
    import h5py

    try:
        with h5py.File(path, "r") as f:
            groups = [k for k in f if isinstance(f[k], h5py.Group) and "EBSD" in f[k]]
    except OSError:
        return None
    wanted = str(map_filter).lower()
    matches = [g for g in groups if wanted in g.lower()]
    return matches[0] if matches else None


def _load_h5ebsd(path: Path, lazy: bool = False, **kwargs):
    """A kikuchipy signal from an HDF5 pattern file whatever its extension.

    ``kp.load`` only knows the extensions h5/hdf5/h5ebsd, so a ``.edaxh5``/``.oh5`` file (the extensions EDAX
    uses for the same layout) is handed to the reader matched by the file's content, through kikuchipy's private
    plugin registry. ``kwargs`` go to the reader (``scan_group_names``). Returns one signal, or a list when the
    reader gives several. Raises ValueError when the file is not HDF5 or no reader fits: EDAX APEX projects keep
    their patterns in ``.up1``/``.up2`` files, which is what to give as the pattern file, and the error says so.
    """
    import importlib

    import h5py
    import kikuchipy as kp
    from kikuchipy.io._io import PLUGINS, _dict2signal, _plugin_from_footprints

    if path.suffix.lower() in (".h5", ".hdf5", ".h5ebsd"):
        return kp.load(path, lazy=lazy, **kwargs)
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path.name} is not an HDF5 file")
    plugins = [p for p in PLUGINS if "h5" in p["file_extensions"] or "h5ebsd" in p["file_extensions"]]
    reader = _plugin_from_footprints(str(path), plugins=plugins)
    if reader is None:
        raise ValueError(
            f"{path.name}: no kikuchipy reader matches the layout of this HDF5 file. An EDAX APEX project "
            "(.edaxh5) keeps its patterns in a .up1/.up2 file next to it: give that file as the pattern file "
            "and the project as the scan file"
        )
    signal_dicts = importlib.import_module(reader["api"]).file_reader(str(path), lazy=lazy, **kwargs)
    signals = [_dict2signal(d, lazy=lazy) for d in signal_dicts]
    return signals[0] if len(signals) == 1 else signals


def _sample_patterns(data, nav_shape: tuple, max_patterns: int) -> np.ndarray:
    """Up to ``max_patterns`` patterns spread evenly over the map (every ``n // max_patterns``-th pattern in
    row-major order) as a (k, h, w) array of the data's dtype. They are gathered by index, so a non-contiguous
    map (a column ROI) is not copied whole; works for numpy and dask data, 1D and 2D maps."""
    n = int(np.prod(nav_shape))
    step = max(1, n // max_patterns)
    idx = np.arange(0, n, step)
    if len(nav_shape) == 2:
        rows, cols = np.unravel_index(idx, tuple(int(v) for v in nav_shape))
        picked = data.vindex[rows, cols] if hasattr(data, "vindex") else data[rows, cols]
    else:
        picked = data[idx]
    return np.asarray(picked)


def apply_roi_and_flip(s, st: Settings):
    """The signal cropped to ``ROI`` (col_start, col_stop, row_start, row_stop; a view, when set) and with every
    pattern flipped upside down when ``FLIP_PATTERNS_UD`` (a view of the data, in place on the signal)."""
    if st.ROI is not None:
        c0, c1, r0, r1 = st.ROI
        s = s.inav[c0:c1, r0:r1]
    if st.FLIP_PATTERNS_UD:
        s.data = s.data[..., ::-1, :]
    return s


def preprocess(s, st: Settings, log: Log = _print) -> None:
    """Static and dynamic background removal (``REMOVE_STATIC_BACKGROUND``, ``REMOVE_DYNAMIC_BACKGROUND``) in
    place on the signal, with kikuchipy's defaults. A signal without a stored static background gets one: the mean
    of up to 5000 patterns spread over the map, in the data type of the patterns (:func:`sample_mean_pattern`
    gives the same array). NLPAR is the Kikuchi Pattern Processor's job (pyebsdindex) and is not done here."""
    nav_shape = s.axes_manager.navigation_shape[::-1]
    if st.REMOVE_STATIC_BACKGROUND:
        if s.static_background is None:
            flat = _sample_patterns(s.data, nav_shape, 5000)
            bg = flat.astype(np.float64).mean(axis=0)
            s.static_background = bg.astype(s.data.dtype)
            log(f"Static background: mean of {flat.shape[0]} patterns")
        s.remove_static_background()
        log("Static background removed")
    if st.REMOVE_DYNAMIC_BACKGROUND:
        s.remove_dynamic_background()
        log("Dynamic background removed")


_SLAB_BYTES = 64 * 1024**2


class _Throttle:
    """:meth:`ready` is True at most once per ``interval`` seconds (and the first time it is called)."""

    def __init__(self, interval: float = 0.2):
        self.interval = float(interval)
        self._last = -np.inf

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False


class _FractionCallback(Callback):
    """Dask callback calling ``progress(fraction of tasks done)`` (0..1) after a task finishes, at most once per
    ``interval`` seconds. Used by :func:`quasi_iq` and ``montage.tile_stats`` for lazy signals."""

    def __init__(self, progress: Callable[[float], None], interval: float = 0.2):
        super().__init__()
        self.progress = progress
        self._throttle = _Throttle(interval)

    def _posttask(self, key, result, dsk, state, worker_id):
        n_done = len(state["finished"])
        n_tasks = n_done + len(state["ready"]) + len(state["waiting"]) + len(state["running"])
        if n_tasks and self._throttle.ready():
            self.progress(n_done / n_tasks)


def quasi_iq(s, progress: Callable[[float], None] | None = None) -> np.ndarray:
    """A quick image quality map: the standard deviation of every raw pattern (the pattern processor's
    definition), float32 of the navigation shape ((rows, cols), or (n,) for a line scan).

    An in-RAM signal is done in slabs of about 64 MB of float32, a lazy one with dask in chunks of about that
    size (one read of a memory-mapped file). ``progress(fraction_done)`` (0..1) is called at most about 5 times a
    second when given, and once with 1.0 at the end.
    """
    nav_shape = tuple(s.axes_manager.navigation_shape[::-1])
    sig_shape = tuple(s.axes_manager.signal_shape[::-1])
    n = int(np.prod(nav_shape)) if nav_shape else 1
    pattern_bytes_f32 = 4 * int(np.prod(sig_shape))
    if getattr(s, "_lazy", False):
        data = s.data
        n_nav = data.ndim - 2
        # the chunk limit is for the stored data type, so that a chunk is ~64 MB once cast to float32
        limit = max(1, int(_SLAB_BYTES * data.dtype.itemsize / 4))
        chunks = {i: "auto" for i in range(n_nav)}
        chunks.update({n_nav: -1, n_nav + 1: -1})
        data = data.rechunk(chunks, block_size_limit=limit)
        result = data.astype(np.float32).std(axis=(-2, -1))
        if progress is None:
            out = result.compute()
        else:
            with _FractionCallback(progress):
                out = result.compute()
            progress(1.0)
        return np.asarray(out, dtype=np.float32).reshape(nav_shape)
    data = np.asarray(s.data)
    slab = max(1, _SLAB_BYTES // max(1, pattern_bytes_f32))
    try:
        data = data.reshape((n,) + sig_shape, copy=False)
    except (ValueError, TypeError):
        # a map that is not contiguous (e.g. a column ROI) cannot be flattened without copying it whole, and
        # numpy before 2.1 has no ``copy=`` keyword: the slabs are then taken within rows
        pass
    if data.ndim > 3:  # a 2D map that could not be flattened: slabs within rows
        n_cols = nav_shape[-1]
        slabs = [(r, slice(c0, min(n_cols, c0 + slab))) for r in range(nav_shape[0]) for c0 in range(0, n_cols, slab)]
        out = np.empty(nav_shape, dtype=np.float32)
    else:  # one flat list of patterns (a 2D map flattened, or a line scan)
        slabs = [(slice(i0, min(n, i0 + slab)),) for i0 in range(0, n, slab)]
        out = np.empty((n,), dtype=np.float32)
    throttle = _Throttle()
    for k, idx in enumerate(slabs):
        out[idx] = data[idx].astype(np.float32).std(axis=(-2, -1))
        if progress is not None and (k == len(slabs) - 1 or throttle.ready()):
            progress((k + 1) / len(slabs))
    return out.reshape(nav_shape)


def stretch(image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """``image`` scaled to [0, 1] (float32, same shape) between the ``lo_pct`` and ``hi_pct`` percentiles of its
    finite values, clipped; non-finite values become 0, as does all of a constant image."""
    image = np.asarray(image, dtype=np.float64)
    out = np.zeros(image.shape, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return out
    lo, hi = np.percentile(image[finite], [float(lo_pct), float(hi_pct)])
    if not hi > lo:
        return out
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    out[finite] = scaled[finite]
    return out


def scan_iq_map(st: Settings, nav_shape: tuple, log: Log = _print) -> np.ndarray:
    """The image quality map of ``SCAN_FILE`` (the IQ column of the .ang, or the IQ field of the .edaxh5 map
    chosen by ``EDAXH5_MAP``; ``ebsd_params.scan_iq_map``), cropped to ``ROI`` like the patterns, stretched to
    0..1 (:func:`stretch`) for the live view. ``nav_shape`` is the loaded signal's (rows, cols): a map of another
    size is a ValueError (the scan file is not this scan's, or the ROI is wrong). Raises ValueError without a
    scan file. Logs the source and the time."""
    import ebsd_params

    if not st.SCAN_FILE:
        raise ValueError("SCAN_FILE is not set: no IQ map to read")
    t0 = time.perf_counter()
    iq, info = ebsd_params.scan_iq_map(st.SCAN_FILE, map_filter=st.EDAXH5_MAP)
    if st.ROI is not None:
        c0, c1, r0, r1 = st.ROI
        iq = iq[r0:r1, c0:c1]
    if tuple(iq.shape) != tuple(int(v) for v in nav_shape):
        raise ValueError(f"the IQ map of {Path(st.SCAN_FILE).name} is {iq.shape} (rows, cols) after the ROI, the loaded "
                         f"patterns are {tuple(nav_shape)}: not the same scan?")
    finite = iq[np.isfinite(iq)]
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        raise ValueError(f"the IQ column of {Path(st.SCAN_FILE).name} is constant ({float(finite[0]) if finite.size else 'NaN'}): "
                         "it holds no map (an RKD scan collected without Hough indexing, for instance). Use an indexing "
                         "result's .ang/.h5 as the scan file, or 'Build quasi-IQ map'")
    log(f"IQ map read from {Path(st.SCAN_FILE).name}" + (f" ({info['map']})" if info.get("map") else "")
        + f" in {time.perf_counter() - t0:.1f} s")
    return stretch(iq)


def sample_mean_pattern(s, max_patterns: int = 5000) -> np.ndarray:
    """The static background the indexing would use, float32 of the signal shape: the one stored with the file,
    else the mean of up to ``max_patterns`` patterns spread over the map. Same rule as :func:`preprocess`."""
    if getattr(s, "static_background", None) is not None:
        return np.asarray(s.static_background, dtype=np.float32)
    flat = _sample_patterns(s.data, tuple(s.axes_manager.navigation_shape[::-1]), max_patterns)
    return np.asarray(flat.astype(np.float32).mean(axis=0), dtype=np.float32)


def preview_pattern(pattern: np.ndarray, st: Settings, static_bg: np.ndarray | None = None,
                    apply_background: bool = True) -> np.ndarray:
    """One pattern as the indexer will see it, float32 (h, w), for the live view.

    With ``apply_background`` and the setting ticked: kikuchipy's static background removal (subtract
    ``static_bg``, then the per-pattern rescale to the data type's range, as ``remove_static_background`` does),
    then kikuchipy's dynamic background removal; then always the indexer's own preprocessing when ``N_REGIONS``
    or ``GAUSSIAN_BACKGROUND`` is set (:func:`preprocess_pattern_for_preview`). ``apply_background=False`` skips
    the two background steps, for a pattern that was already corrected (a prepared signal during a run). A float
    pattern is treated as uint8 for the rescale.
    """
    import kikuchipy as kp

    p = np.asarray(pattern)
    if apply_background and (st.REMOVE_STATIC_BACKGROUND or st.REMOVE_DYNAMIC_BACKGROUND):
        dtype = p.dtype if np.issubdtype(p.dtype, np.integer) else np.dtype(np.uint8)
        omax = int(np.iinfo(dtype).max)
        if st.REMOVE_STATIC_BACKGROUND and static_bg is not None:
            # preprocess() stores the background in the data type of the patterns, so the same truncation here
            bg = np.asarray(static_bg).astype(dtype).astype(np.float32)
            q = p.astype(np.float32) - bg
            imin, imax = float(q.min()), float(q.max())
            if imax > imin:
                q = (q - imin) / (imax - imin) * omax
            p = q.astype(dtype)
        if st.REMOVE_DYNAMIC_BACKGROUND:
            # float input, integer output: the signal-level call does the same and the result matches it exactly
            p = kp.pattern.remove_dynamic_background(p.astype(np.float32), operation="subtract",
                                                     filter_domain="frequency", dtype_out=dtype)
    if st.N_REGIONS or st.GAUSSIAN_BACKGROUND:
        p = preprocess_pattern_for_preview(p, st)
    return np.asarray(p, dtype=np.float32)


def preprocess_pattern_for_preview(pattern: np.ndarray, st: Settings, good_pixels: np.ndarray | None = None) -> np.ndarray:
    """One pattern through the indexer's own preprocessing (Gaussian background, adaptive histogram equalization
    with ``N_REGIONS`` tiles), by the fork's private function so the preview matches what is indexed.
    ``good_pixels`` (a bool mask of the pattern shape, e.g. ``SphericalIndexer.good_pixels``, the circular mask)
    restricts the equalization to those pixels as the indexer does; None uses the whole pattern."""
    from kikuchipy.indexing._spherical._preprocessing import _preprocess_pattern

    return _preprocess_pattern(np.asarray(pattern), good_pixels=good_pixels, gaussian_background=st.GAUSSIAN_BACKGROUND,
                               n_regions=st.N_REGIONS)


# ----------------------------------------------------------------------------------------------
# Master pattern(s) and detector
# ----------------------------------------------------------------------------------------------


def load_harmonics(st: Settings, log: Log = _print, demo: bool = False) -> list:
    """The spherical harmonics (``MasterPatternHarmonics``) of every master in ``MASTER_FILE``, one per phase in
    the file's order. An ``.sht`` is read as it is; an EMsoft ``.h5`` is loaded (``MASTER_ENERGY``) and
    transformed at ``MASTER_BANDWIDTH``. ``demo`` uses kikuchipy's small nickel master instead. Logs each
    phase's point group, bandwidth, energy and simulated tilt, and warns when a master's bandwidth is below
    ``BANDWIDTH``. Raises ValueError when ``MASTER_FILE`` is not set (and not ``demo``)."""
    import kikuchipy as kp

    if not demo and st.MASTER_FILE is None:
        raise ValueError("MASTER_FILE is not set")
    if demo:
        mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
        harmonics = [mp.get_spherical_harmonics(bandwidth=st.MASTER_BANDWIDTH)]
    else:
        files = [st.MASTER_FILE] if isinstance(st.MASTER_FILE, (str, Path)) else list(st.MASTER_FILE)
        harmonics = []
        for f in files:
            f = Path(f)
            log(f"Loading master pattern {f.name}")
            if f.suffix.lower() == ".sht":
                harmonics.append(kp.indexing.MasterPatternHarmonics.from_file(f))
            else:
                kwargs = {"projection": "lambert", "hemisphere": "both"}
                if st.MASTER_ENERGY is not None:
                    kwargs["energy"] = st.MASTER_ENERGY
                mp = kp.load(f, **kwargs)
                harmonics.append(mp.get_spherical_harmonics(bandwidth=st.MASTER_BANDWIDTH))
    for h in harmonics:
        log(
            f"{h.phase.name}: point group {h.phase.point_group.name}, bandwidth {h.bandwidth}, "
            f"beam energy {h.beam_energy} keV, simulated sample tilt {h.sample_tilt} deg"
        )
        if h.bandwidth < st.BANDWIDTH:
            log(f"NB! The master of {h.phase.name} has bandwidth {h.bandwidth} < BANDWIDTH {st.BANDWIDTH}")
    return harmonics


def resolve_sample_tilt(st: Settings, harmonics: Sequence, log: Log = _print, demo: bool = False) -> float:
    """The sample tilt in degrees used for indexing: 70 for ``demo``, else ``SAMPLE_TILT``, else the first
    master's simulated tilt. A master simulated for another tilt is an error (ValueError), unless
    ``OVERRIDE_MASTER_SAMPLE_TILT``, which overwrites ``h.sample_tilt`` on the harmonics (in place). Also raises
    ValueError when neither the settings nor the master state a tilt."""
    if demo:
        sample_tilt = 70.0
    elif st.SAMPLE_TILT is not None:
        sample_tilt = float(st.SAMPLE_TILT)
    elif harmonics[0].sample_tilt is not None:
        sample_tilt = float(harmonics[0].sample_tilt)
    else:
        raise ValueError("The master pattern does not state a sample tilt: set SAMPLE_TILT")
    for h in harmonics:
        if h.sample_tilt is not None and not np.isclose(h.sample_tilt, sample_tilt):
            if st.OVERRIDE_MASTER_SAMPLE_TILT:
                log(f"Overriding the master's sample tilt {h.sample_tilt} with {sample_tilt} deg")
                h.sample_tilt = sample_tilt
            else:
                raise ValueError(
                    f"The master of {h.phase.name} was simulated for a sample tilt of {h.sample_tilt} deg but "
                    f"SAMPLE_TILT is {sample_tilt} deg. Use a matching master pattern or set "
                    "OVERRIDE_MASTER_SAMPLE_TILT = True."
                )
    log(f"Sample tilt used: {sample_tilt} deg")
    return sample_tilt


def make_detector(st: Settings, s, harmonics: Sequence, log: Log = _print, demo: bool = False):
    """The kikuchipy ``EBSDDetector`` of the run: the signal's pattern shape (rows, cols), ``PC`` in
    ``PC_CONVENTION``, the sample tilt of :func:`resolve_sample_tilt`, ``CAMERA_TILT``, ``PX_SIZE`` and
    ``BINNING``. For ``demo`` the detector stored with the demo signal, with its average PC."""
    import kikuchipy as kp

    sample_tilt = resolve_sample_tilt(st, harmonics, log, demo)
    if demo:
        det = s.detector.deepcopy()
        det.pc = det.pc_average
        return det
    sig_shape = tuple(s.axes_manager.signal_shape[::-1])
    det = kp.detectors.EBSDDetector(
        shape=sig_shape,
        pc=tuple(float(v) for v in st.PC),
        convention=st.PC_CONVENTION,
        sample_tilt=sample_tilt,
        tilt=float(st.CAMERA_TILT),
        px_size=float(st.PX_SIZE),
        binning=int(st.BINNING),
    )
    log(f"Projection center {tuple(st.PC)} ({st.PC_CONVENTION}), camera tilt {st.CAMERA_TILT} deg")
    return det


def back_project(pattern: np.ndarray, det, bandwidth: int):
    """One pattern on the sphere, for *Check geometry*: ``(north, window, iq)``, the northern hemisphere grid of
    the fork's ``SphericalBackProjector`` at ``bandwidth``, its window mask (True where the detector covers the
    sphere) and the pattern's image quality (float, EMSphInx's measure)."""
    import kikuchipy as kp

    projector = kp.indexing.SphericalBackProjector(det, bandwidth=int(bandwidth))
    north, south, iq = projector.unproject(np.asarray(pattern), return_image_quality=True)
    return north, projector.window_mask()[0], float(iq)


# ----------------------------------------------------------------------------------------------
# Single-pattern indexing (the workbench's 'Index this point')
# ----------------------------------------------------------------------------------------------


def point_indexer_key(st: Settings, det, hkey: tuple) -> tuple:
    """What identifies a cached :func:`make_point_indexer` indexer: the workbench's harmonics key ``hkey``
    (``(MASTER_FILE, MASTER_BANDWIDTH, MASTER_ENERGY, demo)``), the detector's geometry (shape, PC, sample tilt,
    camera tilt, pixel size, binning) and the indexing settings the indexer is built with (``BANDWIDTH``,
    ``NORMALIZE``, ``REFINE``, ``N_REGIONS``, ``GAUSSIAN_BACKGROUND``, ``CIRCULAR_MASK``). The backend is always
    the CPU and is not in the key."""
    return (tuple(hkey), tuple(int(v) for v in det.shape), tuple(np.round(np.asarray(det.pc, dtype=float).ravel(), 8)),
            float(det.sample_tilt), float(det.tilt), float(det.px_size), int(det.binning),
            int(st.BANDWIDTH), bool(st.NORMALIZE), bool(st.REFINE), int(st.N_REGIONS),
            bool(st.GAUSSIAN_BACKGROUND), bool(st.CIRCULAR_MASK))


def make_point_indexer(harmonics: Sequence, det, st: Settings, log: Log = _print):
    """The fork's ``SphericalIndexer`` of every master in ``harmonics`` on ``det``, for the CPU backend, with the
    indexing settings of ``st`` (bandwidth, normalize, refine, n_regions, gaussian_background, circular_mask: the
    fields :func:`index_kwargs` uses; ``BACKEND`` and ``CHUNKSIZE`` do not apply to one pattern). Takes about a
    second at bandwidth 68 and longer at higher bandwidths, so the caller caches it under
    :func:`point_indexer_key`; the indexer is immutable and reusable across calls. Like the runs, it refuses a
    twisted detector, several PCs or a sample tilt other than the master's. Logs the build time."""
    from kikuchipy.indexing import SphericalIndexer

    t0 = time.perf_counter()
    indexer = SphericalIndexer(
        list(harmonics), det, bandwidth=int(st.BANDWIDTH), normalize=bool(st.NORMALIZE), refine=bool(st.REFINE),
        n_regions=int(st.N_REGIONS), gaussian_background=bool(st.GAUSSIAN_BACKGROUND),
        circular_mask=bool(st.CIRCULAR_MASK), backend="cpu",
    )
    log(f"Single-pattern indexer built in {time.perf_counter() - t0:.1f} s (cpu, bandwidth {st.BANDWIDTH})")
    return indexer


def lambert_master(h):
    """The band limited square Lambert master pattern of one ``MasterPatternHarmonics`` (the fork's
    ``to_master_pattern()``: both hemispheres, side ``2 * bandwidth + 1`` = 769 px at bandwidth 384, float64,
    about 9.5 MB), the master :func:`simulate_pattern` projects from: exactly what the indexer matched against.
    Raises ValueError when the harmonics carry no phase with a point group, which ``get_patterns`` needs."""
    phase = getattr(h, "phase", None)
    if phase is None or phase.point_group is None:
        raise ValueError("The master pattern has no phase with a point group: patterns cannot be simulated from it")
    return h.to_master_pattern()


def index_pattern(indexer, pattern: np.ndarray, twist: float = 0.0) -> dict:
    """One pattern through ``indexer.index_patterns`` (``n_best=1``, no progress bar). Returns a dict:

    - ``rotation``: orix ``Rotation`` of shape (1,) as indexed (the fork's ``rotation_from_zyz`` of the ZYZ
      angles, what ``spherical_indexing`` puts in its crystal map and what ``get_patterns`` takes);
    - ``rotation_twisted``: rotated by ``twist`` degrees about the sample normal, as :func:`apply_twist` does to a
      map; ``euler_deg``: its Bunge angles in degrees; ``euler_indexed_deg``: those of ``rotation``;
    - ``zyz`` (radians), ``score``, ``iq``, ``phase_id`` (index into ``indexer.phases``, the order of the
      harmonics), ``phase`` (its name) and ``seconds``.

    The fork's warning about the memory of many workers is silenced (one pattern is one chunk); the one about a
    pattern that could not be indexed is kept, and that case raises ValueError (phase id -1)."""
    from kikuchipy.indexing._spherical._euler import rotation_from_zyz
    from orix.quaternion import Rotation

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Indexing on .* worker")
        res = indexer.index_patterns(np.asarray(pattern)[None], n_best=1, progressbar=False)
    dt = time.perf_counter() - t0
    phase_id = int(res["phase_id"][0, 0])
    if phase_id < 0:
        raise ValueError("The pattern could not be indexed (phase id -1): see the warning in the log")
    rotation = rotation_from_zyz(np.asarray(res["zyz"][0, 0]))
    twisted = rotation * Rotation.from_axes_angles([0, 0, 1], float(twist), degrees=True) if twist else rotation
    return dict(
        rotation=rotation, rotation_twisted=twisted,
        euler_deg=tuple(float(v) for v in np.asarray(twisted.to_euler(degrees=True)).reshape(-1)),
        euler_indexed_deg=tuple(float(v) for v in np.asarray(rotation.to_euler(degrees=True)).reshape(-1)),
        zyz=np.asarray(res["zyz"][0, 0]), score=float(res["scores"][0, 0]), iq=float(res["iq"][0]),
        phase_id=phase_id, phase=str(indexer.phases[phase_id].phase.name), seconds=dt,
    )


def simulate_pattern(master, rotation, det) -> np.ndarray:
    """The pattern of ``rotation`` (orix ``Rotation`` of shape (1,), e.g. ``index_pattern()["rotation"]``) on
    ``det``, projected from ``master`` (:func:`lambert_master`) by the fork's ``EBSDMasterPattern.get_patterns``
    (``compute=True``, no progress bar), as a uint8 (h, w) array: uint8 so that the indexer's preprocessing treats
    it exactly like a uint8 pattern from a file. ``det`` must hold one PC and no twist: use the orientation as
    indexed on the detector it was indexed with (``TWIST`` is applied to the orientations afterwards, never to the
    detector). The first call in a fresh numba cache compiles for several seconds."""
    s = master.get_patterns(rotation, det, compute=True, show_progressbar=False, dtype_out=np.uint8)
    return np.asarray(s.data).reshape(tuple(int(v) for v in det.shape))


# ----------------------------------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------------------------------


def gpu_available() -> tuple[bool, str]:
    """``(True, "<GPU name>, x of y GB VRAM free")`` when CuPy finds a CUDA device, else ``(False, reason)``.
    Never raises; a missing cupy counts as no device."""
    try:
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        free, total = cupy.cuda.runtime.memGetInfo()
        return True, f"{props['name'].decode()}, {free / 1e9:.1f} of {total / 1e9:.1f} GB VRAM free"
    except Exception as e:  # noqa: BLE001
        return False, f"no CUDA device ({type(e).__name__}: {e})"


def gpu_batch_size(bandwidth: int) -> int:
    """The GPU batch size the fork would pick now, from the VRAM that is free now (CuPy's pools are freed first,
    so a previous run does not shrink it). Requires cupy."""
    import cupy
    from kikuchipy.indexing._spherical import _gpu

    free_gpu_memory()
    return int(_gpu._default_batch_size(int(bandwidth), int(cupy.cuda.runtime.memGetInfo()[0])))


def free_gpu_memory() -> None:
    """Return CuPy's pooled device and pinned memory to the driver, so the VRAM shows as free again between runs.
    Does nothing (silently) without cupy or a device."""
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001
        pass


def index_kwargs(st: Settings) -> dict:
    """The keyword arguments of ``EBSD.spherical_indexing`` from the settings: bandwidth, normalize, refine,
    n_regions, gaussian_background, circular_mask, backend and chunksize. On the GPU with ``CHUNKSIZE = None``
    the chunksize is fixed here to :func:`gpu_batch_size`, so that every call of a band-wise run uses the same
    batch size and the size can be logged."""
    kwargs = dict(
        bandwidth=int(st.BANDWIDTH),
        normalize=bool(st.NORMALIZE),
        refine=bool(st.REFINE),
        n_regions=int(st.N_REGIONS),
        gaussian_background=bool(st.GAUSSIAN_BACKGROUND),
        circular_mask=bool(st.CIRCULAR_MASK),
        backend=st.BACKEND,
        chunksize=st.CHUNKSIZE,
    )
    if st.BACKEND == "gpu" and st.CHUNKSIZE is None:
        kwargs["chunksize"] = gpu_batch_size(st.BANDWIDTH)
    return kwargs


def _prepare_indexing(harmonics: Sequence, st: Settings, log: Log) -> tuple:
    """What :func:`index` and :func:`index_blocks` set up alike: dask's global worker count from ``N_WORKERS``
    (a process-wide side effect), the keyword arguments of :func:`index_kwargs` (the GPU batch size logged), and
    the harmonics argument of ``spherical_indexing`` (one ``MasterPatternHarmonics`` for one phase, a list for
    several). Returns ``(harmonics_arg, kwargs)``."""
    import dask

    dask.config.set(num_workers=int(st.N_WORKERS))
    kwargs = index_kwargs(st)
    if st.BACKEND == "gpu":
        log(f"GPU batch size: {kwargs['chunksize']}")
    harmonics_arg = harmonics[0] if len(harmonics) == 1 else list(harmonics)
    return harmonics_arg, kwargs


def apply_twist(xmap, twist: float):
    """A crystal map with every orientation rotated by ``twist`` degrees about the sample normal (the ``TWIST``
    setting); ``xmap`` itself when ``twist`` is 0. The grid and the mask of masked points are kept."""
    from orix.quaternion import Rotation

    if not twist:
        return xmap
    rot = Rotation.from_axes_angles([0, 0, 1], float(twist), degrees=True)
    # Every grid point's stored rotation is rotated: the public ``xmap.rotations`` (and x, y, phase_id, prop)
    # hold the points in the data only, so a map rebuilt from them would lose the masked points and its grid
    return copy_map(xmap, rotations=xmap._rotations * rot)


def copy_map(xmap, rotations=None, phase_id=None, is_in_data=None, phase_list=None):
    """A new crystal map with the whole grid of ``xmap`` (masked points included, the mask kept) and its
    properties, with any of the rotations, phase ids, mask or phase list replaced by full-grid values (arrays of
    one value per grid point, in row-major order). Needed because orix 0.15's ``CrystalMap.deepcopy`` fails on a
    map with masked points, and because the public attributes only hold the points in the data: the private
    ``_rotations``, ``_phase_id``, ``_x``, ``_y`` and the ``_prop`` dict (read with ``dict.__getitem__``, as the
    properties' own ``__getitem__`` masks) hold the full grid."""
    from orix.crystal_map import CrystalMap

    props = {name: np.array(dict.__getitem__(xmap._prop, name)) for name in xmap._prop.keys()}
    kw = dict(
        rotations=xmap._rotations if rotations is None else rotations,
        phase_id=np.array(xmap._phase_id) if phase_id is None else np.asarray(phase_id),
        x=np.array(xmap._x),
        phase_list=xmap.phases if phase_list is None else phase_list,
        prop=props,
        scan_unit=xmap.scan_unit,
        is_in_data=np.array(xmap.is_in_data) if is_in_data is None else np.asarray(is_in_data, dtype=bool),
    )
    if xmap._y is not None:
        kw["y"] = np.array(xmap._y)
    return CrystalMap(**kw)


def test_region(s, test_size: int):
    """The square test region of ``test_size`` map points in the middle of the map, for *Index test region*.

    Returns ``(region, warm_up, (c0, r0))``: the region as a signal view, a view of up to 4 of its patterns to
    index first (numba compilation and GPU set-up outside the timing), and the column and row of the region's
    top-left corner in the map. A line scan gets its first ``test_size**2`` patterns and offset (0, 0).
    """
    nav_shape = s.axes_manager.navigation_shape[::-1]
    if len(nav_shape) == 2:
        n_rows, n_cols = nav_shape
        r0 = max(0, n_rows // 2 - test_size // 2)
        c0 = max(0, n_cols // 2 - test_size // 2)
        s_test = s.inav[c0 : c0 + test_size, r0 : r0 + test_size]
        return s_test, s_test.inav[:4, :1], (c0, r0)
    s_test = s.inav[: test_size * test_size]
    return s_test, s_test.inav[:4], (0, 0)


def index(s, harmonics: Sequence, det, st: Settings, log: Log = _print, navigation_mask=None, warm_up=None,
          verbose: int = 1):
    """Spherical indexing of ``s`` in one ``spherical_indexing`` call.

    ``navigation_mask`` is a boolean array of the map shape, True = do not index (those points are not in the
    returned maps' data). ``warm_up``, when given, is a small signal indexed first (silently) so that numba
    compilation and GPU set-up are not in the timing. ``verbose`` is the fork's. Sets dask's worker count and
    frees CuPy's memory pools before the run. Returns ``(xmap, xmap_indexed, seconds)``: the crystal map with
    ``TWIST`` applied, the map as indexed, and the wall time of the call. Logs the throughput.
    """
    harmonics_arg, kwargs = _prepare_indexing(harmonics, st, log)
    if warm_up is not None:
        warm_up.spherical_indexing(harmonics_arg, det, verbose=0, **kwargs)
    free_gpu_memory()
    t0 = time.perf_counter()
    xmap_indexed = s.spherical_indexing(harmonics_arg, det, navigation_mask=navigation_mask, verbose=verbose, **kwargs)
    dt = time.perf_counter() - t0
    xmap = apply_twist(xmap_indexed, st.TWIST)
    log(f"{xmap.size} patterns in {dt:.1f} s = {xmap.size / dt:.0f} patterns/s "
        f"({st.BACKEND}, bandwidth {st.BANDWIDTH})")
    if st.TWIST:
        log(f"Orientations rotated by {st.TWIST} deg about the sample normal (TWIST)")
    return xmap, xmap_indexed, dt


# Dask task keys of kikuchipy.indexing._spherical._indexer, e.g. ("_index_chunk-<hash>", 0, 0, 0) from
# map_blocks(_index_chunk) and ("refine_chunk-<hash>", 0, 0) from blockwise(_refine_chunk): blockwise strips the
# leading underscore of the function name, map_blocks keeps it. Matched without leading underscores.
_INDEX_TASK = "index_chunk"
_REFINE_TASK = "refine_chunk"


def _task_kind(key) -> str | None:
    """``"index"`` for a coarse-search task key, ``"refine"`` for a refinement task key, None for any other."""
    name = key[0] if isinstance(key, tuple) and key else key
    if isinstance(name, str):
        name = name.lstrip("_")
        if name.startswith(_INDEX_TASK):
            return "index"
        if name.startswith(_REFINE_TASK):
            return "refine"
    return None


class IndexingProgress(Callback):
    """Progress of one ``spherical_indexing`` call, from its dask tasks::

        with IndexingProgress(lambda fi, fr: print(f"search {fi:.0%}, refinement {fr:.0%}")):
            xmap = s.spherical_indexing(...)

    ``on_progress(frac_index, frac_refine)`` gets the fractions (0..1) of the coarse search tasks
    (``_index_chunk``, the expensive stage) and of the refinement tasks (``_refine_chunk``) that are done. It
    is called at most once per ``interval`` seconds, but always when the last task of a stage finishes. A stage
    without tasks counts as done (1.0). NB: ``spherical_indexing`` refines inside ``_index_chunk`` (one
    compute), so for it ``frac_index`` is the whole run and ``frac_refine`` stays 1.0; ``_refine_chunk``
    tasks come from ``refine_orientation_spherical``. When a stage is computed again (e.g. the GPU retries
    with a smaller batch after running out of memory) its count starts over.

    ``on_chunk(offset, packed)``, when given, gets every finished ``_index_chunk`` result in chunk order (see
    :meth:`_hand_on_chunk`). Both callbacks run on the thread that called ``compute`` (dask's local scheduler
    loop), between two tasks, so they must be quick. ``n_tasks`` and ``n_done`` (dicts by stage) are public for
    tests.
    """

    def __init__(self, on_progress: Callable[[float, float], None], interval: float = 0.2,
                 on_chunk: Callable[[int, np.ndarray], None] | None = None):
        super().__init__()
        self.on_progress = on_progress
        self.on_chunk = on_chunk
        self.n_tasks = {"index": 0, "refine": 0}
        self.n_done = {"index": 0, "refine": 0}
        self._throttle = _Throttle(interval)
        self._chunks: dict[int, np.ndarray] = {}  # finished coarse-search chunks not yet handed on
        self._next_chunk = 0
        self._offset = 0

    def _start_state(self, dsk, state):
        counts = {"index": 0, "refine": 0}
        for key in dsk:
            kind = _task_kind(key)
            if kind is not None:
                counts[kind] += 1
        for kind, n in counts.items():
            if n:
                self.n_tasks[kind] = n
                self.n_done[kind] = 0
        if counts["index"]:
            self._chunks, self._next_chunk, self._offset = {}, 0, 0

    def fractions(self) -> tuple[float, float]:
        """The (frac_index, frac_refine) that ``on_progress`` gets now."""
        done = {kind: min(1.0, self.n_done[kind] / n) if n else 1.0 for kind, n in self.n_tasks.items()}
        return done["index"], done["refine"]

    def _posttask(self, key, result, dsk, state, worker_id):
        kind = _task_kind(key)
        if kind is None:
            return
        self.n_done[kind] += 1
        last = self.n_done[kind] >= self.n_tasks[kind]
        if self._throttle.ready() or last:
            self.on_progress(*self.fractions())
        if kind == "index" and self.on_chunk is not None:
            self._hand_on_chunk(key, result)

    def _hand_on_chunk(self, key, result) -> None:
        """``on_chunk(offset, packed)`` for every finished ``_index_chunk`` task, in chunk order.

        The task of chunk ``i`` (key ``(name, i, 0, 0)``) returns the packed rows ``(n_i, n_best, 7)`` of the
        patterns ``offset_i : offset_i + n_i`` of the call, in the order they were given (ZYZ Euler angles in
        radians in columns 0:3, score 3, phase id 4 (-1 = failed), image quality 5, pseudo-symmetry index 6).
        The chunks are all the same size but the last, yet that size is only known from a result, so a chunk is
        handed on once every chunk before it has finished: with several workers a few chunks may wait for an
        earlier, slower one, never for long. Results that are not such arrays are ignored.
        """
        if not (isinstance(key, tuple) and len(key) >= 2 and isinstance(key[1], (int, np.integer))):
            return
        if not (isinstance(result, np.ndarray) and result.ndim == 3 and result.shape[2] >= 5):
            return
        self._chunks[int(key[1])] = result
        while self._next_chunk in self._chunks:
            packed = self._chunks.pop(self._next_chunk)
            self.on_chunk(self._offset, packed)
            self._offset += int(packed.shape[0])
            self._next_chunk += 1


def _navigation_steps(s) -> tuple[float, float]:
    """(step_x, step_y) of a map from the signal's navigation axes' scales; 1.0 where missing or zero."""
    steps = []
    for axis in list(s.axes_manager.navigation_axes)[:2]:
        scale = getattr(axis, "scale", None)
        steps.append(float(scale) if scale else 1.0)
    steps += [1.0] * (2 - len(steps))
    return steps[0], steps[1]


def _scan_unit(s) -> str:
    """The unit of the signal's navigation axes for a crystal map, ``"um"`` when undefined."""
    axes = s.axes_manager.navigation_axes
    unit = str(axes[0].units) if axes else ""
    return unit if unit and unit != "<undefined>" else "um"


def _ipf_direction(direction: str):
    """The orix ``Vector3d`` of the sample direction ``"x"``, ``"y"`` or ``"z"`` (KeyError for anything else)."""
    from orix.vector import Vector3d

    return {"x": Vector3d.xvector(), "y": Vector3d.yvector(), "z": Vector3d.zvector()}[direction]


def _ipf_key(point_group, vec):
    """An orix ``IPFColorKeyTSL`` of ``point_group`` for the sample direction ``vec``, None without a point group."""
    from orix import plot

    return None if point_group is None else plot.IPFColorKeyTSL(point_group, direction=vec)


def ipf_color_keys(harmonics, direction: str = "z") -> list:
    """One orix ``IPFColorKeyTSL`` per phase of ``harmonics`` (a ``MasterPatternHarmonics`` or a sequence of
    them, in phase id order) for the sample direction ``"x"``, ``"y"`` or ``"z"``; ``None`` for a phase without
    a point group."""
    vec = _ipf_direction(direction)
    keys = []
    for h in ([harmonics] if not isinstance(harmonics, (list, tuple)) else list(harmonics)):
        phase = getattr(h, "phase", None)
        keys.append(_ipf_key(getattr(phase, "point_group", None), vec))
    return keys


def packed_rgb(packed: np.ndarray, keys: Sequence) -> np.ndarray:
    """IPF colours (n, 3) float in 0..1 of the best candidate of packed indexing rows of the fork's
    ``_index_chunk``: (n, n_best, 7) as returned, or (n, >= 5) rows of the best candidates (ZYZ Euler angles in
    radians in columns 0:3, phase id in column 4), with ``keys`` from :func:`ipf_color_keys` (index = phase id).
    A pattern that failed (phase -1) or of a phase without a key is white, as in :func:`ipf_rgb`. Refinement,
    when on, happens inside the chunk, so these are the refined orientations. Called by the live view on the GUI
    thread, so it does only the colouring."""
    from kikuchipy.indexing._spherical._euler import rotation_from_zyz

    best = np.asarray(packed, dtype=np.float64)
    if best.ndim == 3:
        best = best[:, 0, :]
    if best.shape[0] == 0:
        return np.ones((0, 3))
    rot = rotation_from_zyz(np.ascontiguousarray(best[:, :3]))
    return _color_points(rot, best[:, 4], {i: k for i, k in enumerate(keys)})


def _block_quaternions(xmap, in_data: np.ndarray) -> np.ndarray:
    """The quaternions (k, ..., 4) of the points in a crystal map's data, bit for bit as stored.

    ``xmap.rotations`` makes a new Rotation, and orix normalizes every new Rotation: reading through it and
    storing the result again changes the last bit of a few quaternions (1e-16), so the assembled map would not
    be identical to one ``spherical_indexing`` call. The stored rotations of all points are read instead
    (``CrystalMap._rotations``, all points of the grid in row-major order, selected by the flat ``in_data``
    mask), with the public property as the fallback should orix rename it.
    """
    stored = getattr(xmap, "_rotations", None)
    if stored is not None and stored.shape[0] == in_data.size:
        return np.asarray(stored.data)[in_data]
    return np.asarray(xmap.rotations.data)


def assemble_blocks(blocks: Sequence, nav_shape: tuple[int, int], step_size: tuple[float, float] = (1.0, 1.0),
                    scan_unit: str = "um"):
    """One crystal map on the full (rows, cols) grid from the crystal maps of bands of whole rows.

    ``blocks`` is a list of ``(r0, r1, xmap)``: map rows r0:r1 indexed as one crystal map of shape
    (r1 - r0, cols), or ``None`` for a band that was skipped (all of it masked out). ``step_size`` is
    (step_x, step_y) in ``scan_unit``. Points that are not in a block's data, and all points of skipped bands,
    are not in the returned map's data (identity rotation, phase -1, NaN float properties and -1 integer
    properties on the full grid, as the fork fills them). The phase list is the union of the blocks' phases
    (the first block's when all are unindexed). Rotations and properties are copied bit for bit, so the result
    equals one ``spherical_indexing`` call of the whole map. Raises ValueError when no block was indexed or a
    block's size does not match its rows.
    """
    from orix.crystal_map import CrystalMap, PhaseList, create_coordinate_arrays
    from orix.quaternion import Rotation

    n_rows, n_cols = (int(v) for v in nav_shape)
    n_all = n_rows * n_cols
    present = [(int(r0), int(r1), x) for r0, r1, x in blocks if x is not None]
    if not present:
        raise ValueError("No block was indexed")
    first = present[0][2]
    rot_tail = first.rotations.shape[1:]  # (n_best,) or ()
    # Identity everywhere, then the blocks' quaternions set as they are: Rotation(array) would normalize them
    # again and change the last bit of some, as spherical_indexing's own map does not
    rot = Rotation.identity((n_all,) + rot_tail)
    quat = rot.data  # a view: written in place
    phase_id = np.full(n_all, -1, dtype=int)
    in_data = np.zeros(n_all, dtype=bool)
    prop = {}
    for name in first.prop:
        values = np.asarray(first.prop[name])
        if np.issubdtype(values.dtype, np.integer):  # e.g. nbest_phase_id, pseudo_symmetry_index
            prop[name] = np.full((n_all,) + values.shape[1:], -1, dtype=values.dtype)
        else:
            dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
            prop[name] = np.full((n_all,) + values.shape[1:], np.nan, dtype=dtype)
    phases = {}
    for r0, r1, xmap in present:
        block_in_data = np.asarray(xmap.is_in_data, dtype=bool).ravel()
        if block_in_data.size != (r1 - r0) * n_cols:
            raise ValueError(f"Block of rows {r0}:{r1} has {block_in_data.size} points, not {(r1 - r0) * n_cols}")
        idx = np.arange(r0 * n_cols, r1 * n_cols)[block_in_data]
        in_data[idx] = True
        quat[idx] = _block_quaternions(xmap, block_in_data).reshape((idx.size,) + rot_tail + (4,))
        phase_id[idx] = xmap.phase_id
        for name, full in prop.items():
            full[idx] = np.asarray(xmap.prop[name])
        for i, phase in xmap.phases:
            if i != -1 and i not in phases:
                phases[i] = phase
    phase_list = PhaseList({i: phases[i] for i in sorted(phases)}) if phases else first.phases
    coords, _ = create_coordinate_arrays((n_rows, n_cols), (float(step_size[1]), float(step_size[0])))
    return CrystalMap(
        rotations=rot,
        phase_id=phase_id,
        x=coords["x"],
        y=coords["y"],
        phase_list=phase_list,
        prop=prop,
        scan_unit=scan_unit,
        is_in_data=in_data,
    )


def index_blocks(s, harmonics: Sequence, det, st: Settings, log: Log = _print, navigation_mask=None,
                 on_event: Callable[[str, dict], None] | None = None, target_seconds: float = 4.0,
                 first_patterns: int | None = None, verbose: int = 0):
    """Spherical indexing of a 2D map band by band (whole map rows, from the top), so that a GUI can show the
    map growing; the result is that of one ``spherical_indexing`` call (every pattern is indexed on its own,
    and :func:`assemble_blocks` copies the results bit for bit).

    The first band has about ``first_patterns`` patterns (default max(columns, 300)), later bands about
    ``target_seconds`` of work at the best speed of a band so far. Bands are sized by the patterns they index, so
    rows that are mostly masked out neither inflate the speed nor shrink the band. ``navigation_mask`` is a
    boolean (rows, cols) array, True = do not index (ValueError when its shape is not the map's). ``verbose`` is
    the fork's (0: silent). Sets dask's worker count and frees CuPy's memory pools before the run; the fork's
    UserWarnings (memory estimate, failed patterns) are logged once each, with a count of the bands that repeated
    them at the end.

    ``on_event(kind, info)`` is called on the thread that called this function (dask's scheduler loop runs
    there too), with these ``kind`` and ``info`` keys:

    * ``"block_start"``: ``r0``, ``r1`` (map rows of the band), ``c0``, ``c1`` (0 and the number of columns),
      ``n_patterns`` (to index in the band), ``index`` (band number from 0), ``n_blocks_estimate``.
    * ``"progress"``: ``frac_index``, ``frac_refine`` (0..1, see :class:`IndexingProgress`), ``r0``, ``r1``,
      ``n_patterns`` (of the band). Emitted between two dask tasks, at most about 5 times a second.
    * ``"chunk_done"``: ``rows``, ``cols`` (int arrays, map coordinates of the patterns of one dask chunk, in
      file order, emitted the moment its task finished), ``packed`` (float64 (n, 5) rows of their best
      candidates: ZYZ Euler angles in radians, score, phase id; the refined orientation when refinement is on,
      the coarse one when it is off, as EMSphInx paints its preview), ``keys`` (the IPF-Z colour keys of
      :func:`ipf_color_keys`, for :func:`packed_rgb`), ``n_patterns`` (of the chunk), ``r0``, ``r1`` (the
      band). A chunk is one GPU batch or one CPU chunk. Colouring is left to the receiver, because this is
      emitted between two dask tasks. Not emitted when no phase has a point group.
    * ``"block_done"``: ``r0``, ``r1``, ``c0``, ``c1``, ``xmap`` (the band's crystal map as indexed, ``TWIST``
      not applied, None for a band that is masked out completely), ``seconds`` (of the band),
      ``patterns_per_second`` (of the band), ``done_patterns``, ``total_patterns`` (to index in the whole run),
      ``elapsed`` (seconds since the run started), ``eta_seconds`` (from the steady speed, 0.0 when unknown).

    (``run_start`` and ``run_end`` are the workbench's own events around this call, not emitted here.) A line
    scan (one navigation axis) is indexed as one band with ``r0, r1 = 0, 1`` and ``c1`` its length.

    Returns ``(xmap, seconds, patterns_per_second)``: the crystal map as indexed on the full grid (``TWIST`` not
    applied; masked-out points not in its data), the summed time of the bands, and the steady speed
    (:func:`_steady_speed`: without the first band, which includes the numba compilation on the CPU, when there
    are at least 3 bands).
    """
    harmonics_arg, kwargs = _prepare_indexing(harmonics, st, log)

    def emit(kind: str, info: dict) -> None:
        if on_event is not None:
            on_event(kind, info)

    nav_shape = tuple(int(v) for v in s.axes_manager.navigation_shape[::-1])
    if navigation_mask is not None:
        navigation_mask = np.asarray(navigation_mask, dtype=bool)
        if navigation_mask.shape != nav_shape:
            raise ValueError(f"navigation_mask has shape {navigation_mask.shape}, the map {nav_shape}")
        if not navigation_mask.any():
            navigation_mask = None
    total_patterns = int(np.prod(nav_shape)) - (int(navigation_mask.sum()) if navigation_mask is not None else 0)
    if kwargs["chunksize"] is None:
        # The fork sizes CPU chunks from the number of patterns of the call, and for a small band that gives a
        # few big chunks which run far slower than the chunks it would use for the whole map (measured: 66
        # against 91 patterns/s). Size the chunks once for the whole map and use that for every band.
        try:
            from kikuchipy.indexing._spherical._indexer import _batch_estimate, _n_workers

            kwargs["chunksize"] = int(_batch_estimate(int(st.BANDWIDTH), _n_workers(), max(1, total_patterns)))
        except Exception:  # noqa: BLE001  (a fork without these helpers: let every call choose)
            pass

    # The fork warns once per spherical_indexing call (e.g. "estimated to need x GiB of memory, which is more
    # than 2 GiB"), which with one call per band would repeat the same line every few seconds. Each distinct
    # warning is logged the first time only; the repeats are counted and reported at the end.
    seen_warnings: dict[str, list] = {}  # digit-free key -> [first text, count]

    # IPF colour keys for the per-chunk preview; without them (e.g. harmonics without a phase) no chunk events
    color_keys = None
    if on_event is not None:
        try:
            color_keys = ipf_color_keys(harmonics, "z")
            if all(k is None for k in color_keys):
                color_keys = None
                log("No point group for the IPF colours: the live view paints the map band by band only")
        except Exception as e:  # noqa: BLE001
            color_keys = None
            log(f"IPF colour keys for the live preview could not be made ({type(e).__name__}: {e}); "
                "painting band by band only")
    n_cols_map = int(nav_shape[-1])

    def run(signal, mask, r0, r1, n):
        """Index one band (``signal``, rows r0:r1 of the map, ``n`` patterns after ``mask``) and return its map."""
        # map positions (row-major within the band) of the patterns handed to the fork, in its order
        keep = None if mask is None else np.flatnonzero(~np.asarray(mask, dtype=bool).ravel())

        def on_chunk(offset, packed):
            # Runs between two dask tasks: only a slice and a signal, no colouring
            k = int(packed.shape[0])
            pos = np.arange(offset, offset + k) if keep is None else keep[offset:offset + k]
            emit("chunk_done", {"rows": r0 + pos // n_cols_map, "cols": pos % n_cols_map,
                                "packed": np.array(packed[:, 0, :5], dtype=np.float64), "keys": color_keys,
                                "n_patterns": k, "r0": r0, "r1": r1})

        # Only the fork's own UserWarnings are recorded (the memory estimate, patterns that failed): the filter
        # is process-wide while the band runs, and "always" for every category would also collect and report
        # third-party DeprecationWarnings that are silent everywhere else
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            with IndexingProgress(
                lambda fi, fr: emit("progress", {"frac_index": fi, "frac_refine": fr, "r0": r0, "r1": r1,
                                                 "n_patterns": n}),
                on_chunk=on_chunk if color_keys is not None else None,
            ):
                xmap_band = signal.spherical_indexing(harmonics_arg, det, navigation_mask=mask, verbose=verbose,
                                                      **kwargs)
        for w in caught:
            if not issubclass(w.category, UserWarning):
                continue
            text = str(w.message)
            key = re.sub(r"[\d.]+", "#", text)
            if key not in seen_warnings:
                log(f"Warning: {text}")
                seen_warnings[key] = [text, 0]
            seen_warnings[key][1] += 1
        return xmap_band

    def block_done(r0, r1, c1, xmap, dt, n, done_patterns, eta):
        emit("block_done", {"r0": r0, "r1": r1, "c0": 0, "c1": c1, "xmap": xmap, "seconds": dt,
                            "patterns_per_second": n / dt if dt > 0 else 0.0, "done_patterns": done_patterns,
                            "total_patterns": total_patterns, "elapsed": time.perf_counter() - t_start,
                            "eta_seconds": eta})

    free_gpu_memory()
    t_start = time.perf_counter()
    if len(nav_shape) != 2:
        n = total_patterns
        emit("block_start", {"r0": 0, "r1": 1, "c0": 0, "c1": nav_shape[0], "n_patterns": n, "index": 0,
                             "n_blocks_estimate": 1})
        t0 = time.perf_counter()
        xmap = run(s, navigation_mask, 0, 1, n)
        dt = time.perf_counter() - t0
        pps = n / dt if dt > 0 else 0.0
        block_done(0, 1, nav_shape[0], xmap, dt, n, n, 0.0)
        log(f"{n} patterns in {dt:.1f} s = {pps:.0f} patterns/s ({st.BACKEND}, bandwidth {st.BANDWIDTH})")
        return xmap, dt, pps

    n_rows, n_cols = nav_shape
    if first_patterns is None:
        first_patterns = max(n_cols, 300)
    blocks = []
    timings = []  # (patterns indexed, seconds) of every band that was indexed
    done_patterns = 0
    rate = None  # best indexed patterns per second of a band so far
    row_patterns = (~navigation_mask).sum(axis=1) if navigation_mask is not None else np.full(n_rows, n_cols)
    cum_patterns = np.concatenate([[0], np.cumsum(row_patterns)])
    r0 = 0
    i = 0
    while r0 < n_rows:
        wanted = first_patterns if rate is None else target_seconds * rate
        # the fewest rows from r0 holding at least ``wanted`` patterns (at least one row)
        r1 = int(np.searchsorted(cum_patterns, cum_patterns[r0] + wanted, side="left"))
        r1 = max(r0 + 1, min(n_rows, r1))
        rows = r1 - r0
        estimate = i + 1 + int(np.ceil((n_rows - r1) / rows))
        mask = navigation_mask[r0:r1] if navigation_mask is not None else None
        n = rows * n_cols - (int(mask.sum()) if mask is not None else 0)
        emit("block_start", {"r0": r0, "r1": r1, "c0": 0, "c1": n_cols, "n_patterns": n, "index": i,
                             "n_blocks_estimate": estimate})
        if n == 0:
            xmap, dt = None, 0.0
        else:
            if mask is not None and not mask.any():
                mask = None
            t0 = time.perf_counter()
            xmap = run(s.inav[:, r0:r1], mask, r0, r1, n)
            dt = time.perf_counter() - t0
            timings.append((n, dt))
            if dt > 0:
                # the best rate so far: a band slowed down by something else must not shrink the next one
                rate = max(rate or 0.0, n / dt)
        blocks.append((r0, r1, xmap))
        done_patterns += n
        steady = _steady_speed(timings)
        remaining = total_patterns - done_patterns
        block_done(r0, r1, n_cols, xmap, dt, n, done_patterns, remaining / steady if steady > 0 else 0.0)
        r0 = r1
        i += 1
    total_seconds = sum(dt for _, dt in timings)
    steady = _steady_speed(timings)
    for text, count in seen_warnings.values():
        if count > 1:
            log(f"(the warning '{text[:60]}...' was raised by {count} bands; shown once)")
    xmap = assemble_blocks(blocks, nav_shape, _navigation_steps(s), _scan_unit(s))
    log(
        f"{total_patterns} patterns in {len(blocks)} bands, {total_seconds:.1f} s; {steady:.0f} patterns/s "
        f"({st.BACKEND}, bandwidth {st.BANDWIDTH})"
    )
    return xmap, total_seconds, steady


def _steady_speed(timings: Sequence[tuple[int, float]]) -> float:
    """Patterns per second over the bands in ``timings`` ((patterns, seconds) each), leaving out the first band
    (numba compilation) when there are 3 or more; 0.0 without any time."""
    use = timings[1:] if len(timings) >= 3 else timings
    n = sum(n for n, _ in use)
    dt = sum(dt for _, dt in use)
    return n / dt if dt > 0 else 0.0


# ----------------------------------------------------------------------------------------------
# Crystal maps: colours, full-grid arrays, saving, comparing
# ----------------------------------------------------------------------------------------------


def _in_data(xmap) -> np.ndarray:
    """The flat boolean mask (one value per grid point, row-major) of the points in a crystal map's data."""
    return np.asarray(xmap.is_in_data, dtype=bool).ravel()


def _best_rotations(xmap):
    """The rotations of the points in a crystal map's data, the best candidate only when it holds ``n_best``."""
    rot = xmap.rotations
    return rot[:, 0] if rot.ndim > 1 else rot


def _color_points(rotations, phase_id: np.ndarray, keys: dict) -> np.ndarray:
    """IPF colours (n, 3) float in 0..1 of ``rotations`` (an orix Rotation of n points) with ``keys`` a dict
    phase id -> ``IPFColorKeyTSL`` (or None). Points of phase -1 (not indexed) or of a phase without a key are
    white."""
    from orix.quaternion import Orientation

    phase_id = np.asarray(phase_id).astype(int).ravel()
    rgb = np.ones((phase_id.size, 3))
    for pid in np.unique(phase_id):
        key = keys.get(int(pid))
        if pid < 0 or key is None:
            continue
        m = phase_id == pid
        rgb[m] = key.orientation2color(Orientation(rotations[m], key.symmetry))
    return rgb


def ipf_rgb(xmap, direction="z") -> np.ndarray:
    """IPF colours (n, 3) of a crystal map's points in the data for the sample direction ``"x"``, ``"y"`` or
    ``"z"``, with orix's TSL colour key of each phase; not indexed points are white."""
    vec = _ipf_direction(direction)
    keys = {int(pid): _ipf_key(phase.point_group, vec) for pid, phase in xmap.phases_in_data if pid != -1}
    return _color_points(_best_rotations(xmap), xmap.phase_id, keys)


def full_shape(xmap) -> tuple:
    """The (rows, cols) of a crystal map's whole grid, masked points included ((n,) for a line scan).

    ``xmap.shape`` is the extent of the points in the data only, so it shrinks when a whole edge row or column is
    masked out; the grid is then recovered from the private full-grid coordinates ``_x``, ``_y``. A map whose grid
    cannot be recovered is reported as one row of all its points.
    """
    n = int(np.asarray(xmap.is_in_data).size)
    if int(np.prod(xmap.shape)) == n:
        return tuple(int(v) for v in xmap.shape)
    ys, xs = getattr(xmap, "_y", None), getattr(xmap, "_x", None)
    if ys is not None and xs is not None and np.asarray(ys).size == n:
        n_rows, n_cols = len(np.unique(np.asarray(ys))), len(np.unique(np.asarray(xs)))
        if n_rows * n_cols == n:
            return (n_rows, n_cols) if n_rows > 1 or xmap.ndim == 2 else (n_cols,)
    return (n,)


def full_map_data(xmap, name: str, fill=np.nan) -> np.ndarray:
    """A property of a crystal map (the best candidate's value when it holds ``n_best``) on its whole grid as a
    float array of :func:`full_shape`, ``fill`` where not in the data (and everywhere when the map has no such
    property)."""
    in_data = _in_data(xmap)
    full = np.full(in_data.size, float(fill), dtype=float)
    if name in xmap.prop:
        values = np.asarray(xmap.prop[name], dtype=float)
        if values.ndim > 1:
            values = values[:, 0]
        full[in_data] = values
    return full.reshape(full_shape(xmap))


def ipf_rgb_map(xmap, direction="z", background=0.6) -> np.ndarray:
    """IPF colours (:func:`ipf_rgb`) as a (rows, cols, 3) float image of the whole grid; points not in the data
    (masked out) take the grey level ``background``."""
    in_data = _in_data(xmap)
    full = np.full((in_data.size, 3), float(background))
    full[in_data] = ipf_rgb(xmap, direction)
    return full.reshape(full_shape(xmap) + (3,))


def block_rgb(xmap, block: dict, direction="z") -> np.ndarray:
    """IPF colours of one band of :func:`index_blocks` as a (rows, cols, 3) image. A band of one map row is a 1D
    crystal map (orix drops the axis), whose image would be (cols, 3); the band's ``r0``, ``r1``, ``c0``, ``c1``
    (the ``block_done`` payload) restore it."""
    rgb = ipf_rgb_map(xmap, direction)
    return rgb.reshape(int(block["r1"]) - int(block["r0"]), int(block["c1"]) - int(block["c0"]), 3)


def on_full_grid(xmap):
    """The crystal map with every grid point in the data: points that were masked out become not indexed
    (phase -1, added to the phase list; identity rotation as stored). The map itself is returned when nothing is
    masked."""
    in_data = _in_data(xmap)
    if in_data.all():
        return xmap
    phases = xmap.phases.deepcopy()
    if -1 not in phases.ids:
        phases.add_not_indexed()
    return copy_map(xmap, phase_id=np.where(in_data, np.asarray(xmap._phase_id), -1), is_in_data=np.ones_like(in_data),
                    phase_list=phases)


def save_xmap(xmap, out_dir: str | Path, stem: str, save_ang: bool = True, save_h5: bool = True,
              iq_prop: str = "iq", ci_prop: str = "scores", log: Log = _print) -> list[Path]:
    """Write a crystal map as ``<out_dir>/<stem>.h5`` (orix's HDF5 format) and ``<out_dir>/<stem>.ang``,
    creating the folder and overwriting existing files. The ``.ang`` takes its IQ and CI columns from the
    properties ``iq_prop`` and ``ci_prop``. Returns the paths written, each logged."""
    from orix import io

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if save_h5:
        path = out_dir / f"{stem}.h5"
        io.save(path, xmap, overwrite=True)
        written.append(path)
    if save_ang:
        path = out_dir / f"{stem}.ang"
        # orix sizes the .ang grid by the extent of the points in the data, so a map whose first or last rows
        # (or columns) are masked out would be written cropped and shifted: write every grid point, the masked
        # ones as not indexed (phase -1: CI -1 in the file, as OIM marks them)
        io.save(path, on_full_grid(xmap), overwrite=True, image_quality_prop=iq_prop, confidence_index_prop=ci_prop)
        written.append(path)
    for path in written:
        log(f"Saved {path}")
    return written


def output_location(st: Settings, source_file: str | None) -> tuple[Path, str]:
    """Where :func:`save_xmap` writes: ``(folder, stem)``. The folder is ``OUTPUT_DIR``, else the pattern file's
    folder, else the current folder; the stem is ``OUTPUT_NAME``, else ``<pattern file stem>_sph_bw<BANDWIDTH>``
    (``nickel_demo`` for the demo)."""
    if st.OUTPUT_DIR:
        out_dir = Path(st.OUTPUT_DIR)
    elif source_file:
        out_dir = Path(source_file).parent
    else:
        out_dir = Path.cwd()
    source_stem = Path(source_file).stem if source_file else "nickel_demo"
    stem = st.OUTPUT_NAME or f"{source_stem}_sph_bw{st.BANDWIDTH}"
    return out_dir, stem


def compare_with_reference(xmap_indexed, reference_file: str | Path, roi=None, full_nav_shape=None) -> np.ndarray:
    """Misorientation in degrees to a reference map of the same grid, as a float array of the map's full grid
    (:func:`full_shape`), NaN at points that were not indexed (masked out).

    ``reference_file`` is an EMSphInx/OIM ``.ang``, ``.ctf`` or ``.h5``: an HDF5 file with an ``EBSD/Data``
    group of ``Phi1``, ``Phi``, ``Phi2`` (EMSphInx's output, Bunge Euler angles in radians) is read directly,
    anything else by orix. ``roi`` (c0, c1, r0, r1) with ``full_nav_shape`` (rows, cols) of the whole scan crops
    a reference of the whole scan to the indexed region. The misorientation uses the point group of the map's
    first indexed phase for both maps (the reference's phases are not read). Raises ValueError when the point
    counts differ.
    """
    import h5py
    from orix import io
    from orix.quaternion import Orientation, Rotation

    filename = Path(reference_file)
    rot_ref = None
    if filename.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(filename, "r") as f:
            scans = [k for k in f if isinstance(f[k], h5py.Group) and "EBSD/Data" in f[k]]
            if scans:
                data = f[scans[0]]["EBSD/Data"]
                if all(k in data for k in ("Phi1", "Phi", "Phi2")):
                    euler = np.stack([data[k][()] for k in ("Phi1", "Phi", "Phi2")], axis=1)
                    rot_ref = Rotation.from_euler(euler)
    if rot_ref is None:
        rot_ref = io.load(filename).rotations
    if roi is not None and full_nav_shape is not None and rot_ref.size == int(np.prod(full_nav_shape)):
        c0, c1, r0, r1 = roi
        quat = rot_ref.data.reshape(tuple(full_nav_shape) + (4,))[r0:r1, c0:c1]
        rot_ref = type(rot_ref)(quat.reshape(-1, 4))
    in_data = _in_data(xmap_indexed)
    if rot_ref.size != in_data.size:
        raise ValueError(f"Reference has {rot_ref.size} points, this map has {in_data.size} grid points")
    ids = [i for i in xmap_indexed.phases_in_data.ids if i != -1]
    pg = xmap_indexed.phases[ids[0]].point_group
    ref = rot_ref.flatten()[in_data]
    out = np.full(in_data.size, np.nan)
    out[in_data] = Orientation(_best_rotations(xmap_indexed), pg).angle_with(Orientation(ref, pg), degrees=True)
    return out.reshape(full_shape(xmap_indexed))
