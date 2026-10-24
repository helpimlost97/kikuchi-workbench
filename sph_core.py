"""Spherical indexing: the functions behind the Kikuchi workbench's indexing and montage tabs.

Each step of the indexing (load, preprocess, harmonics, detector, index, save, compare) is one function taking a
:class:`Settings`. Nothing in here imports Qt, so the module can be used from a script too::

    import sph_core
    st = sph_core.Settings(PATTERN_FILE="scan.up1", SCAN_FILE="scan.ang", MASTER_FILE="Ni.sht", CAMERA_TILT=8)
    s, info = sph_core.load_patterns(st)
    sph_core.preprocess(s, st)
    harmonics = sph_core.load_harmonics(st)
    det = sph_core.make_detector(st, s, harmonics)
    xmap, xmap_indexed, dt = sph_core.index(s, harmonics, det, st)
    sph_core.save_xmap(xmap, "results", "scan_sph_bw68")

:func:`index_blocks` gives the same map band by band with progress events, for a GUI.

A preset file (NAME = value lines, values as Python literals) holds settings; :func:`read_preset` reads one and
:func:`write_preset` writes one.
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

Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------


@dataclass
class Settings:
    """Every setting of the spherical indexing (files, geometry, preprocessing, indexing, output)."""

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
        return [f.name for f in dataclasses.fields(self)]


_PRESET_LINE = re.compile(r"^([A-Z][A-Z0-9_]*) = (.*)$")
_PRESET_COMMENT = re.compile(r"\s+#.*$")


def _preset_value(text: str):
    """The Python literal of a preset value; a trailing ``  # comment`` is only stripped when the whole text is
    not a literal, so a path containing ' #' survives."""
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return ast.literal_eval(_PRESET_COMMENT.sub("", text).strip())


def read_preset(path: str | Path, settings: Settings | None = None) -> tuple[Settings, list[str]]:
    """Settings from a preset file of ``NAME = value`` lines (values are Python literals).

    Returns the settings and the names in the file that are not settings (ignored).
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
    """Write the settings as a preset file (NAME = value lines, values as Python source).
    ``fields`` restricts it to those setting names (all of them when not given)."""
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


def _lazy_decision(pattern_file: Path, lazy) -> bool:
    if lazy == "auto":
        try:
            import psutil

            return pattern_file.stat().st_size > 0.25 * psutil.virtual_memory().total
        except ImportError:
            return pattern_file.stat().st_size > 4e9
    return bool(lazy)


def load_patterns(st: Settings, scan: dict | None = None, log: Log = _print):
    """The patterns as a kikuchipy EBSD signal, with the scan parameters applied.

    ``scan`` is the dict of ebsd_params.scan_info() (read here when not given); it supplies the map size and
    step of old .up1 files. Returns the signal and the scan dict. PATTERN_FILE=None loads the nickel demo.
    """
    import kikuchipy as kp

    if st.PATTERN_FILE is None:
        log("Loading kikuchipy's nickel demo data set")
        s = kp.data.nickel_ebsd_large(allow_download=True)
        scan = scan or {}
    else:
        pattern_file = Path(st.PATTERN_FILE)
        if scan is None:
            import ebsd_params

            scan = ebsd_params.scan_info(st.SCAN_FILE, pattern_file=str(pattern_file), map_filter=st.EDAXH5_MAP)
        lazy = _lazy_decision(pattern_file, st.LAZY)
        kwargs = {"lazy": lazy}
        nav_shape_in, step_size = st.NAV_SHAPE, st.STEP_SIZE
        h5_group = None
        if pattern_file.suffix.lower() in _HDF5_SUFFIXES:
            h5_group = _h5_scan_group(pattern_file, st.EDAXH5_MAP)
            if h5_group is not None:
                kwargs["scan_group_names"] = h5_group
        if pattern_file.suffix.lower() in (".up1", ".up2"):
            if not scan.get("up_has_map_size"):
                if nav_shape_in is None and scan.get("n_rows") and scan.get("grid") != "HexGrid":
                    nav_shape_in = (scan["n_rows"], scan["n_cols"])
                if step_size is None and scan.get("step_x"):
                    step_size = (scan["step_x"], scan["step_y"])
            if nav_shape_in is not None:
                kwargs["nav_shape"] = tuple(int(v) for v in nav_shape_in)
        log(f"Loading {pattern_file.name} (lazy: {lazy})" + (f", scan group {h5_group!r}" if h5_group else ""))
        s = _load_h5ebsd(pattern_file, **kwargs) if pattern_file.suffix.lower() in _HDF5_SUFFIXES else kp.load(pattern_file, **kwargs)
        if step_size is not None:
            for axis, step in zip(s.axes_manager.navigation_axes, step_size):
                axis.scale = float(step)
                axis.units = "um"
    s = apply_roi_and_flip(s, st)
    nav_shape = s.axes_manager.navigation_shape[::-1]
    sig_shape = s.axes_manager.signal_shape[::-1]
    log(f"{int(np.prod(nav_shape))} patterns, map {tuple(nav_shape)} (rows, cols), pattern {tuple(sig_shape)} px, lazy: {s._lazy}")
    return s, scan


_HDF5_SUFFIXES = (".h5", ".hdf5", ".h5ebsd", ".edaxh5", ".oh5")


def _h5_scan_group(path: Path, map_filter: str | None) -> str | None:
    """The top-level scan group of an h5ebsd pattern file selected by ``map_filter`` (part of its name, case
    insensitive), or None for the reader's default (the first scan) when there is no filter or no match."""
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
    """A kikuchipy signal from an HDF5 pattern file whatever its extension: ``kp.load`` only knows the extensions
    h5/hdf5/h5ebsd, so a ``.edaxh5``/``.oh5`` file (the extensions EDAX uses for the same layout) is handed to
    the reader matched by the file's content. EDAX APEX projects keep their patterns in ``.up1``/``.up2``
    files, which is what to give as the pattern file; the error says so when no reader fits."""
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
    """Up to ``max_patterns`` patterns spread evenly over the map as an (k, h, w) array, gathered by index so a
    non-contiguous map (a column ROI) is not copied whole; works for numpy and dask data, 1D and 2D maps."""
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
    if st.ROI is not None:
        c0, c1, r0, r1 = st.ROI
        s = s.inav[c0:c1, r0:r1]
    if st.FLIP_PATTERNS_UD:
        s.data = s.data[..., ::-1, :]
    return s


def preprocess(s, st: Settings, log: Log = _print) -> None:
    """Static and dynamic background removal, in place. NLPAR is the
    Kikuchi Pattern Processor's job (pyebsdindex) and is not done here."""
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
    """True from :meth:`ready` at most once per ``interval`` seconds."""

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
    """Dask callback calling ``progress(fraction of tasks done)``, at most about 5 times a second."""

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
    """A quick image quality map: the standard deviation of every raw pattern, float32 of the navigation
    shape ((rows, cols), or (n,) for a line scan).

    An in-RAM signal is done in slabs of about 64 MB of float32, a lazy one with dask in chunks of about that
    size. ``progress(fraction_done)`` (0..1) is called now and then (at most about 5 times a second) when given.
    """
    nav_shape = tuple(s.axes_manager.navigation_shape[::-1])
    sig_shape = tuple(s.axes_manager.signal_shape[::-1])
    n = int(np.prod(nav_shape)) if nav_shape else 1
    pattern_bytes_f32 = 4 * int(np.prod(sig_shape))
    if getattr(s, "_lazy", False):
        data = s.data
        n_nav = data.ndim - 2
        # Chunks of ~64 MB once cast to float32: the limit is for the stored data type
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
    try:  # one flat list of patterns, if that needs no copy (it does for e.g. a column ROI)
        data = data.reshape((n,) + sig_shape, copy=False)
        slabs = [(slice(i0, min(n, i0 + slab)),) for i0 in range(0, n, slab)]
    except (ValueError, TypeError):  # map that is not contiguous: slabs within rows (or along a line scan)
        n_cols = nav_shape[-1]
        if len(nav_shape) == 2:
            slabs = [(r, slice(c0, min(n_cols, c0 + slab))) for r in range(nav_shape[0]) for c0 in range(0, n_cols, slab)]
        else:
            slabs = [(slice(i0, min(n, i0 + slab)),) for i0 in range(0, n, slab)]
    out = np.empty(nav_shape if data.ndim > 3 else (n,), dtype=np.float32)
    throttle = _Throttle()
    for k, idx in enumerate(slabs):
        out[idx] = data[idx].astype(np.float32).std(axis=(-2, -1))
        if progress is not None and (k == len(slabs) - 1 or throttle.ready()):
            progress((k + 1) / len(slabs))
    return out.reshape(nav_shape)


def stretch(image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """``image`` scaled to [0, 1] (float32) between the lo_pct and hi_pct percentiles of its finite values;
    non-finite values become 0, as does all of a constant image."""
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


def sample_mean_pattern(s, max_patterns: int = 5000) -> np.ndarray:
    """The static background the indexing would use: the one stored with the file, else the mean of up to
    ``max_patterns`` patterns spread over the map (float32, signal shape). Same rule as :func:`preprocess`."""
    if getattr(s, "static_background", None) is not None:
        return np.asarray(s.static_background, dtype=np.float32)
    flat = _sample_patterns(s.data, tuple(s.axes_manager.navigation_shape[::-1]), max_patterns)
    return np.asarray(flat.astype(np.float32).mean(axis=0), dtype=np.float32)


def preview_pattern(pattern: np.ndarray, st: Settings, static_bg: np.ndarray | None = None,
                    apply_background: bool = True) -> np.ndarray:
    """One pattern as the indexer will see it, for the live view: kikuchipy's static background removal
    (subtract, then the per-pattern rescale to the data type range, as ``remove_static_background`` does),
    kikuchipy's dynamic background removal, then the indexer's own preprocessing (Gaussian background,
    adaptive histogram equalization with N_REGIONS tiles). ``apply_background=False`` skips the two
    background steps, for a pattern that was already corrected (a prepared signal during a run)."""
    import kikuchipy as kp

    p = np.asarray(pattern)
    if apply_background and (st.REMOVE_STATIC_BACKGROUND or st.REMOVE_DYNAMIC_BACKGROUND):
        dtype = p.dtype if np.issubdtype(p.dtype, np.integer) else np.dtype(np.uint8)
        omin, omax = 0, int(np.iinfo(dtype).max)
        if st.REMOVE_STATIC_BACKGROUND and static_bg is not None:
            # the signal-level path stores the background in the data type of the patterns (preprocess does
            # ``s.static_background = bg.astype(s.data.dtype)``), so the same truncation is applied here
            bg = np.asarray(static_bg).astype(dtype).astype(np.float32)
            q = p.astype(np.float32) - bg
            imin, imax = float(q.min()), float(q.max())
            if imax > imin:
                q = (q - imin) / (imax - imin) * (omax - omin) + omin
            p = q.astype(dtype)
        if st.REMOVE_DYNAMIC_BACKGROUND:
            # float input, integer output: the signal-level call does the same and the result matches it exactly
            p = kp.pattern.remove_dynamic_background(p.astype(np.float32), operation="subtract", filter_domain="frequency", dtype_out=dtype)
    if st.N_REGIONS or st.GAUSSIAN_BACKGROUND:
        p = preprocess_pattern_for_preview(p, st)
    return np.asarray(p, dtype=np.float32)


def preprocess_pattern_for_preview(pattern: np.ndarray, st: Settings) -> np.ndarray:
    """One pattern as the indexer sees it (adaptive histogram equalization, Gaussian background)."""
    from kikuchipy.indexing._spherical._preprocessing import _preprocess_pattern

    return _preprocess_pattern(np.asarray(pattern), gaussian_background=st.GAUSSIAN_BACKGROUND, n_regions=st.N_REGIONS)


# ----------------------------------------------------------------------------------------------
# Master pattern(s) and detector
# ----------------------------------------------------------------------------------------------


def load_harmonics(st: Settings, log: Log = _print, demo: bool = False) -> list:
    """Harmonics of every master in MASTER_FILE (.sht or EMsoft .h5)."""
    import kikuchipy as kp

    if demo or st.MASTER_FILE is None:
        if not demo:
            raise ValueError("MASTER_FILE is not set")
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
    """The sample tilt used for indexing, from SAMPLE_TILT or the master pattern, and the master tilt check."""
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
    """The EBSDDetector: pattern shape, projection center, tilts and pixel size from the settings."""
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
    """One pattern on the sphere (north hemisphere), its image quality and the window mask."""
    import kikuchipy as kp

    projector = kp.indexing.SphericalBackProjector(det, bandwidth=int(bandwidth))
    north, south, iq = projector.unproject(np.asarray(pattern), return_image_quality=True)
    return north, projector.window_mask()[0], float(iq)


# ----------------------------------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------------------------------


def gpu_available() -> tuple[bool, str]:
    try:
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        free, total = cupy.cuda.runtime.memGetInfo()
        return True, f"{props['name'].decode()}, {free / 1e9:.1f} of {total / 1e9:.1f} GB VRAM free"
    except Exception as e:  # noqa: BLE001
        return False, f"no CUDA device ({type(e).__name__}: {e})"


def gpu_batch_size(bandwidth: int) -> int:
    """The GPU batch size the fork would pick now, from the VRAM that is free now."""
    import cupy
    from kikuchipy.indexing._spherical import _gpu

    free_gpu_memory()
    return int(_gpu._default_batch_size(int(bandwidth), int(cupy.cuda.runtime.memGetInfo()[0])))


def free_gpu_memory() -> None:
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001
        pass


def index_kwargs(st: Settings) -> dict:
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


def apply_twist(xmap, twist: float):
    """Crystal map with the orientations rotated by ``twist`` degrees about the sample normal."""
    from orix.quaternion import Rotation

    if not twist:
        return xmap
    rot = Rotation.from_axes_angles([0, 0, 1], float(twist), degrees=True)
    # Every grid point's stored rotation is rotated: the public ``xmap.rotations`` (and x, y, phase_id, prop)
    # hold the points in the data only, so a map rebuilt from them would lose the masked points and its grid
    return copy_map(xmap, rotations=xmap._rotations * rot)


def copy_map(xmap, rotations=None, phase_id=None, is_in_data=None, phase_list=None):
    """A new crystal map with the whole grid of ``xmap`` (masked points included, the mask kept) and its
    properties, with any of the rotations, phase ids, mask or phase list replaced by full-grid values. (orix
    0.15's ``CrystalMap.deepcopy`` fails on a map with masked points.)"""
    from orix.crystal_map import CrystalMap

    props = {name: np.array(dict.__getitem__(xmap._prop, name)) for name in xmap._prop.keys()}  # full grid
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
    """The TEST_SIZE x TEST_SIZE region in the middle of the map and a few patterns for warm-up."""
    nav_shape = s.axes_manager.navigation_shape[::-1]
    if len(nav_shape) == 2:
        n_rows, n_cols = nav_shape
        r0 = max(0, n_rows // 2 - test_size // 2)
        c0 = max(0, n_cols // 2 - test_size // 2)
        s_test = s.inav[c0 : c0 + test_size, r0 : r0 + test_size]
        return s_test, s_test.inav[:4, :1], (c0, r0)
    s_test = s.inav[: test_size * test_size]
    return s_test, s_test.inav[:4], (0, 0)


def index(s, harmonics: Sequence, det, st: Settings, log: Log = _print, navigation_mask=None, warm_up=None, verbose: int = 1):
    """Spherical indexing of ``s``; returns (xmap with TWIST applied, xmap as indexed, seconds)."""
    import dask

    dask.config.set(num_workers=int(st.N_WORKERS))
    kwargs = index_kwargs(st)
    if st.BACKEND == "gpu":
        log(f"GPU batch size: {kwargs['chunksize']}")
    harmonics_arg = harmonics[0] if len(harmonics) == 1 else list(harmonics)
    if warm_up is not None:
        warm_up.spherical_indexing(harmonics_arg, det, verbose=0, **kwargs)
    free_gpu_memory()
    t0 = time.perf_counter()
    xmap_indexed = s.spherical_indexing(harmonics_arg, det, navigation_mask=navigation_mask, verbose=verbose, **kwargs)
    dt = time.perf_counter() - t0
    xmap = apply_twist(xmap_indexed, st.TWIST)
    log(f"{xmap.size} patterns in {dt:.1f} s = {xmap.size / dt:.0f} patterns/s ({st.BACKEND}, bandwidth {st.BANDWIDTH})")
    if st.TWIST:
        log(f"Orientations rotated by {st.TWIST} deg about the sample normal (TWIST)")
    return xmap, xmap_indexed, dt


# Dask task keys of kikuchipy.indexing._spherical._indexer, e.g. ("_index_chunk-<hash>", 0, 0, 0) from
# map_blocks(_index_chunk) and ("refine_chunk-<hash>", 0, 0) from blockwise(_refine_chunk): blockwise strips the
# leading underscore of the function name, map_blocks keeps it. Matched without leading underscores.
_INDEX_TASK = "index_chunk"
_REFINE_TASK = "refine_chunk"


def _task_kind(key) -> str | None:
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
    is called at most about 5 times a second, but always when the last task of a stage finishes. A stage
    without tasks counts as done (1.0). NB: ``spherical_indexing`` refines inside ``_index_chunk`` (one
    compute), so for it ``frac_index`` is the whole run and ``frac_refine`` stays 1.0; ``_refine_chunk``
    tasks come from ``refine_orientation_spherical``. When a stage is computed again (e.g. the GPU retries
    with a smaller batch after running out of memory) its count starts over.
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
        out = []
        for kind in ("index", "refine"):
            n = self.n_tasks[kind]
            out.append(min(1.0, self.n_done[kind] / n) if n else 1.0)
        return out[0], out[1]

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
        patterns ``offset_i : offset_i + n_i`` of the call, in the order they were given (zyz in columns 0:3,
        score 3, phase id 4 (-1 = failed), image quality 5). The chunks are all the same size but the last, yet
        that size is only known from a result, so a chunk is handed on once every chunk before it has finished:
        with several workers a few chunks may wait for an earlier, slower one, never for long.
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
    """(step_x, step_y) of a 2D map from the signal's navigation axes; 1.0 where missing."""
    steps = []
    for axis in list(s.axes_manager.navigation_axes)[:2]:
        scale = getattr(axis, "scale", None)
        steps.append(float(scale) if scale else 1.0)
    steps += [1.0] * (2 - len(steps))
    return steps[0], steps[1]


def _scan_unit(s) -> str:
    axes = s.axes_manager.navigation_axes
    unit = str(axes[0].units) if axes else ""
    return unit if unit and unit != "<undefined>" else "um"


def ipf_color_keys(harmonics, direction: str = "z") -> list:
    """One orix ``IPFColorKeyTSL`` per phase of ``harmonics`` (a ``MasterPatternHarmonics`` or a sequence of
    them, in phase id order), ``None`` for a phase without a point group."""
    from orix import plot

    vec = _ipf_direction(direction)
    keys = []
    for h in ([harmonics] if not isinstance(harmonics, (list, tuple)) else list(harmonics)):
        phase = getattr(h, "phase", None)
        point_group = getattr(phase, "point_group", None)
        keys.append(None if point_group is None else plot.IPFColorKeyTSL(point_group, direction=vec))
    return keys


def packed_rgb(packed: np.ndarray, keys: Sequence) -> np.ndarray:
    """IPF colours (n, 3) of the best candidate of packed indexing rows of the fork's ``_index_chunk``: (n,
    n_best, 7) as returned, or (n, >= 5) rows of the best candidates (ZYZ Euler angles in columns 0:3, phase
    id in column 4), with ``keys`` from :func:`ipf_color_keys`. A pattern that failed (phase -1) or of a phase
    without a key is white, as in :func:`ipf_rgb`. Refinement, when on, happens inside the chunk, so these are
    the refined orientations."""
    from kikuchipy.indexing._spherical._euler import rotation_from_zyz

    best = np.asarray(packed, dtype=np.float64)
    if best.ndim == 3:
        best = best[:, 0, :]
    if best.shape[0] == 0:
        return np.ones((0, 3))
    rot = rotation_from_zyz(np.ascontiguousarray(best[:, :3]))
    return _color_points(rot, best[:, 4], {i: k for i, k in enumerate(keys)})


def _block_quaternions(xmap, in_data: np.ndarray) -> np.ndarray:
    """The quaternions of the points in a crystal map's data, bit for bit as stored.

    ``xmap.rotations`` makes a new Rotation, and orix normalizes every new Rotation: reading through it and
    storing the result again changes the last bit of a few quaternions (1e-16), so the assembled map would not
    be identical to one ``spherical_indexing`` call. The stored rotations of all points are read instead
    (``CrystalMap._rotations``, all points of the grid in row-major order), with the public property as the
    fallback should orix rename it.
    """
    stored = getattr(xmap, "_rotations", None)
    if stored is not None and stored.shape[0] == in_data.size:
        return np.asarray(stored.data)[in_data]
    return np.asarray(xmap.rotations.data)


def assemble_blocks(blocks: Sequence, nav_shape: tuple[int, int], step_size: tuple[float, float] = (1.0, 1.0),
                    scan_unit: str = "um"):
    """One crystal map on the full (rows, cols) grid from crystal maps of bands of whole rows.

    ``blocks`` is a list of ``(r0, r1, xmap)``: map rows r0:r1 indexed as one crystal map of shape
    (r1 - r0, cols), or ``None`` for a band that was skipped (all of it masked out). ``step_size`` is
    (step_x, step_y). Points that are not in a block's data, and all points of skipped bands, are not in the
    returned map's data (identity rotation, phase -1, NaN properties on the full grid).
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
        if np.issubdtype(values.dtype, np.integer):  # e.g. nbest_phase_id, pseudo_symmetry_index: fill -1 as the fork
            prop[name] = np.full((n_all,) + values.shape[1:], -1, dtype=values.dtype)
        else:
            prop[name] = np.full((n_all,) + values.shape[1:], np.nan, dtype=values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64)
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
            values = np.asarray(xmap.prop[name])
            if full.ndim == 1:
                full[idx] = values
            else:
                for j in range(full.shape[1]):
                    full[idx, j] = values[:, j]
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
    map growing; the result is that of one ``spherical_indexing`` call (every pattern is indexed on its own).

    The first band has about ``first_patterns`` patterns (default max(columns, 300)), later bands about
    ``target_seconds`` of work at the speed of the band before. ``navigation_mask`` is a boolean (rows, cols)
    array, True = do not index. ``on_event(kind, info)`` is called with

    * ``"block_start"``: r0, r1, c0, c1, n_patterns (to index), index, n_blocks_estimate
    * ``"progress"``: frac_index, frac_refine (0..1, see :class:`IndexingProgress`), r0, r1, n_patterns
    * ``"chunk_done"``: rows, cols (int arrays, map coordinates of the patterns of one dask chunk, emitted the
      moment its task finished, in file order), packed (float (n, 5) rows of their best candidates: ZYZ Euler
      angles, score, phase id; the refined orientation when refinement is on, the coarse one when it is off, as
      EMSphInx paints its preview), keys (the IPF-Z colour keys, see :func:`packed_rgb`), n_patterns, r0, r1
      (the band). A chunk is one GPU batch or one CPU chunk; colouring is left to the receiver (cheap on the
      dask scheduler thread, which emits this between two tasks).
    * ``"block_done"``: r0, r1, c0, c1, xmap (the band's crystal map as indexed, None for a band that is
      masked out completely), seconds, patterns_per_second, done_patterns, total_patterns, elapsed,
      eta_seconds

    A line scan (one navigation axis) is indexed in one call (one block). Returns (crystal map as indexed on
    the full grid, TWIST not applied; seconds; steady patterns/s, without the first band, which includes the
    numba compilation on the CPU, when there are at least 3 bands).
    """
    import dask

    dask.config.set(num_workers=int(st.N_WORKERS))
    kwargs = index_kwargs(st)
    if st.BACKEND == "gpu":
        log(f"GPU batch size: {kwargs['chunksize']}")
    harmonics_arg = harmonics[0] if len(harmonics) == 1 else list(harmonics)

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
    if kwargs.get("chunksize") is None:
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
            log(f"IPF colour keys for the live preview could not be made ({type(e).__name__}: {e}); painting band by band only")
    n_cols_map = int(nav_shape[-1])

    def run(signal, mask, r0, r1, n):
        # map positions (row-major within the band) of the patterns handed to the fork, in its order
        keep = None if mask is None else np.flatnonzero(~np.asarray(mask, dtype=bool).ravel())

        def on_chunk(offset, packed):
            # Runs on the dask scheduler thread between two tasks: only a slice and a signal, no colouring
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
                lambda fi, fr: emit("progress", {"frac_index": fi, "frac_refine": fr, "r0": r0, "r1": r1, "n_patterns": n}),
                on_chunk=on_chunk if color_keys is not None else None,
            ):
                xmap_band = signal.spherical_indexing(harmonics_arg, det, navigation_mask=mask, verbose=verbose, **kwargs)
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

    def report_repeated_warnings():
        for text, count in seen_warnings.values():
            if count > 1:
                log(f"(the warning '{text[:60]}...' was raised by {count} bands; shown once)")

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
        emit("block_done", {"r0": 0, "r1": 1, "c0": 0, "c1": nav_shape[0], "xmap": xmap, "seconds": dt,
                            "patterns_per_second": pps, "done_patterns": n, "total_patterns": total_patterns,
                            "elapsed": time.perf_counter() - t_start, "eta_seconds": 0.0})
        log(f"{n} patterns in {dt:.1f} s = {pps:.0f} patterns/s ({st.BACKEND}, bandwidth {st.BANDWIDTH})")
        return xmap, dt, pps

    n_rows, n_cols = nav_shape
    if first_patterns is None:
        first_patterns = max(n_cols, 300)
    blocks = []
    timings = []  # (patterns indexed, seconds) of every band that was indexed
    done_patterns = 0
    rate = None  # best indexed patterns per second of a band so far
    # patterns to index per map row: a band is sized by the patterns it holds, not by its grid points, so that
    # rows which are mostly masked out neither inflate the speed nor shrink the band
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
        emit("block_done", {"r0": r0, "r1": r1, "c0": 0, "c1": n_cols, "xmap": xmap, "seconds": dt,
                            "patterns_per_second": n / dt if dt > 0 else 0.0, "done_patterns": done_patterns,
                            "total_patterns": total_patterns, "elapsed": time.perf_counter() - t_start,
                            "eta_seconds": remaining / steady if steady > 0 else 0.0})
        r0 = r1
        i += 1
    total_seconds = sum(dt for _, dt in timings)
    steady = _steady_speed(timings)
    report_repeated_warnings()
    xmap = assemble_blocks(blocks, nav_shape, _navigation_steps(s), _scan_unit(s))
    log(
        f"{total_patterns} patterns in {len(blocks)} bands, {total_seconds:.1f} s; {steady:.0f} patterns/s "
        f"({st.BACKEND}, bandwidth {st.BANDWIDTH})"
    )
    return xmap, total_seconds, steady


def _steady_speed(timings: Sequence[tuple[int, float]]) -> float:
    """Patterns per second over all bands but the first (numba compilation) when there are 3 or more."""
    use = timings[1:] if len(timings) >= 3 else timings
    n = sum(n for n, _ in use)
    dt = sum(dt for _, dt in use)
    return n / dt if dt > 0 else 0.0


def _ipf_direction(direction: str):
    from orix.vector import Vector3d

    return {"x": Vector3d.xvector(), "y": Vector3d.yvector(), "z": Vector3d.zvector()}[direction]


def _color_points(rotations, phase_id: np.ndarray, keys: dict) -> np.ndarray:
    """IPF colours (n, 3) of ``rotations`` (an orix Rotation of n points) with ``keys`` a dict phase id ->
    ``IPFColorKeyTSL`` (or None). Points of phase -1 (not indexed) or of a phase without a key are white."""
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
    """IPF colours (n, 3) of a crystal map's points in the data for the sample direction "x", "y" or "z"."""
    from orix import plot

    vec = _ipf_direction(direction)
    keys = {int(pid): (plot.IPFColorKeyTSL(phase.point_group, direction=vec) if phase.point_group is not None else None)
            for pid, phase in xmap.phases_in_data if pid != -1}
    rot = xmap.rotations
    if rot.ndim > 1:
        rot = rot[:, 0]
    return _color_points(rot, xmap.phase_id, keys)


def full_shape(xmap) -> tuple:
    """The (rows, cols) of a crystal map's whole grid, masked points included. ``xmap.shape`` is the extent of the
    points in the data only, so it shrinks when a whole edge row or column is masked out."""
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
    """A property of a crystal map on its whole grid (``fill`` where not in the data), float."""
    full = np.full(int(np.asarray(xmap.is_in_data).size), float(fill), dtype=float)
    if name in xmap.prop:
        values = np.asarray(xmap.prop[name], dtype=float)
        if values.ndim > 1:
            values = values[:, 0]
        full[np.asarray(xmap.is_in_data, dtype=bool).ravel()] = values
    return full.reshape(full_shape(xmap))


def ipf_rgb_map(xmap, direction="z", background=0.6) -> np.ndarray:
    """IPF colours as a (rows, cols, 3) image of the whole grid; points not in the data (e.g. masked out) are
    grey."""
    rgb = ipf_rgb(xmap, direction)
    full = np.full((xmap.is_in_data.size, 3), float(background))
    full[np.asarray(xmap.is_in_data, dtype=bool).ravel()] = rgb
    return full.reshape(full_shape(xmap) + (3,))


def block_rgb(xmap, block: dict, direction="z") -> np.ndarray:
    """IPF colours of one band of :func:`index_blocks` as a (rows, cols, 3) image. A band of one map row is a 1D
    crystal map (orix drops the axis), whose image would be (cols, 3); the band's r0, r1, c0, c1 restore it."""
    rgb = ipf_rgb_map(xmap, direction)
    return rgb.reshape(int(block["r1"]) - int(block["r0"]), int(block["c1"]) - int(block["c0"]), 3)


def on_full_grid(xmap):
    """The crystal map with every grid point in the data: points that were masked out become not indexed
    (phase -1, identity rotation as stored). The map itself is returned when nothing is masked."""
    in_data = np.asarray(xmap.is_in_data, dtype=bool).ravel()
    if in_data.all():
        return xmap
    phases = xmap.phases.deepcopy()
    if -1 not in phases.ids:
        phases.add_not_indexed()
    return copy_map(xmap, phase_id=np.where(in_data, np.asarray(xmap._phase_id), -1), is_in_data=np.ones_like(in_data),
                    phase_list=phases)


def save_xmap(xmap, out_dir: str | Path, stem: str, save_ang: bool = True, save_h5: bool = True,
              iq_prop: str = "iq", ci_prop: str = "scores", log: Log = _print) -> list[Path]:
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
    """Misorientation in degrees to a reference map of the same grid (EMSphInx/OIM .ang/.h5/.ctf), as an array
    of the map's full grid (``full_shape``), NaN at points that were not indexed (masked out). ``roi``
    (c0, c1, r0, r1) with ``full_nav_shape`` (rows, cols) of the whole scan crops a reference of the whole
    scan to the indexed region."""
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
    in_data = np.asarray(xmap_indexed.is_in_data, dtype=bool).ravel()
    if rot_ref.size != in_data.size:
        raise ValueError(f"Reference has {rot_ref.size} points, this map has {in_data.size} grid points")
    ids = [i for i in xmap_indexed.phases_in_data.ids if i != -1]
    pg = xmap_indexed.phases[ids[0]].point_group
    rot = xmap_indexed.rotations
    if rot.ndim > 1:
        rot = rot[:, 0]
    ref = rot_ref.flatten()[in_data]
    out = np.full(in_data.size, np.nan)
    out[in_data] = Orientation(rot, pg).angle_with(Orientation(ref, pg), degrees=True)
    return out.reshape(full_shape(xmap_indexed))
