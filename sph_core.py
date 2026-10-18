"""Spherical indexing and HROSM without a notebook: the functions behind kikuchi_workbench.py.

Everything here mirrors the cells of spherical_indexing_template.ipynb (built by build_notebook.py) so that
the GUI and the notebook do the same thing with the same settings names. Nothing in here imports Qt, so the
module can be used from a script too::

    import sph_core
    st = sph_core.Settings(PATTERN_FILE="scan.up1", SCAN_FILE="scan.ang", MASTER_FILE="Ni.sht", CAMERA_TILT=8)
    s, info = sph_core.load_patterns(st)
    sph_core.preprocess(s, st)
    harmonics = sph_core.load_harmonics(st)
    det = sph_core.make_detector(st, s, harmonics)
    xmap, xmap_indexed, dt = sph_core.index(s, harmonics, det, st)
    sph_core.save_xmap(xmap, "results", "scan_sph_bw68")

Settings names are those of the notebook's settings cells, so a notebook preset file (NAME = value lines) can
be read with :func:`read_preset` and written with :func:`write_preset`.
"""
from __future__ import annotations

import ast
import dataclasses
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------


@dataclass
class Settings:
    """The settings of the indexing notebook. Defaults are the notebook's defaults."""

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
    # HROSM
    HROSM_MASTER_FILE: str | None = None
    HROSM_ENERGY: float | None = None
    HROSM_THRESHOLD: float = 5.0
    HROSM_MAX_ANGLE: float = 5.0
    HROSM_N_STEPS: int = 10
    HROSM_KEEP_N: int = 20
    HROSM_N_OSM: int = 10
    HROSM_AVERAGE: str = "mean"
    HROSM_MIN_PIXELS: int = 10
    HROSM_DILATE: bool = False
    HROSM_ROI: tuple | None = None
    # output
    OUTPUT_DIR: str | None = None
    OUTPUT_NAME: str | None = None
    SAVE_ANG: bool = True
    SAVE_H5: bool = True
    REFERENCE_FILE: str | None = None

    def names(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self)]


_PRESET_LINE = re.compile(r"^([A-Z][A-Z0-9_]*) = (.*?)(\s+#.*)?$")


def read_preset(path: str | Path, settings: Settings | None = None) -> tuple[Settings, list[str]]:
    """Settings from a preset file of ``NAME = value`` lines (values are Python literals).

    Returns the settings and the names that are not settings (ignored, as build_notebook.py does).
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
            setattr(settings, name, ast.literal_eval(value))
        except (ValueError, SyntaxError):
            ignored.append(f"{name} (value not understood: {value})")
    return settings, ignored


def write_preset(path: str | Path, settings: Settings, header: str = "") -> None:
    """Write the settings as a preset file that build_notebook.py accepts (values as Python source)."""
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    lines.append("# Preset written by kikuchi_workbench.py; usage: python build_notebook.py my_scan.ipynb this_file.txt")
    for name in settings.names():
        value = getattr(settings, name)
        if isinstance(value, str):
            lines.append(f"{name} = r{value!r}" if "\\" in value or "/" in value else f"{name} = {value!r}")
        elif isinstance(value, list):
            lines.append(f"{name} = [" + ", ".join(f"r{v!r}" for v in value) + "]")
        else:
            lines.append(f"{name} = {value!r}")
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
    """The patterns as a kikuchipy EBSD signal, like sections 0 and 3 of the notebook.

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
        if pattern_file.suffix.lower() in (".up1", ".up2"):
            if not scan.get("up_has_map_size"):
                if nav_shape_in is None and scan.get("n_rows") and scan.get("grid") != "HexGrid":
                    nav_shape_in = (scan["n_rows"], scan["n_cols"])
                if step_size is None and scan.get("step_x"):
                    step_size = (scan["step_x"], scan["step_y"])
            if nav_shape_in is not None:
                kwargs["nav_shape"] = tuple(int(v) for v in nav_shape_in)
        log(f"Loading {pattern_file.name} (lazy: {lazy})")
        s = kp.load(pattern_file, **kwargs)
        if step_size is not None:
            for axis, step in zip(s.axes_manager.navigation_axes, step_size):
                axis.scale = float(step)
                axis.units = "um"
    s = apply_roi_and_flip(s, st)
    nav_shape = s.axes_manager.navigation_shape[::-1]
    sig_shape = s.axes_manager.signal_shape[::-1]
    log(f"{int(np.prod(nav_shape))} patterns, map {tuple(nav_shape)} (rows, cols), pattern {tuple(sig_shape)} px, lazy: {s._lazy}")
    return s, scan


def signal_from_array(patterns: np.ndarray, nav_shape: tuple[int, int], step_size=None, static_background=None):
    """A kikuchipy EBSD signal from an in-RAM (n_patterns, rows, cols) array, e.g. the Kikuchi Pattern
    Processor's processed patterns. ``patterns`` is reshaped (no copy) to the map; missing patterns of a map
    that was stopped early are padded with zeros."""
    import kikuchipy as kp

    n_rows, n_cols = (int(v) for v in nav_shape)
    ph, pw = patterns.shape[-2:]
    n = n_rows * n_cols
    if patterns.shape[0] < n:
        padded = np.zeros((n, ph, pw), dtype=patterns.dtype)
        padded[: patterns.shape[0]] = patterns
        patterns = padded
    s = kp.signals.EBSD(patterns[:n].reshape(n_rows, n_cols, ph, pw))
    for axis, name in zip(s.axes_manager.navigation_axes, ("x", "y")):
        axis.name = name
        axis.units = "um"
    if step_size is not None:
        for axis, step in zip(s.axes_manager.navigation_axes, step_size):
            axis.scale = float(step) if step else 1.0
    if static_background is not None:
        s.static_background = np.asarray(static_background).astype(patterns.dtype)
    return s


def apply_roi_and_flip(s, st: Settings):
    if st.ROI is not None:
        c0, c1, r0, r1 = st.ROI
        s = s.inav[c0:c1, r0:r1]
    if st.FLIP_PATTERNS_UD:
        s.data = s.data[..., ::-1, :]
    return s


def preprocess(s, st: Settings, log: Log = _print) -> None:
    """Static and dynamic background removal, in place, like section 3 of the notebook. NLPAR is the
    Kikuchi Pattern Processor's job (pyebsdindex) and is not done here."""
    nav_shape = s.axes_manager.navigation_shape[::-1]
    sig_shape = tuple(s.axes_manager.signal_shape[::-1])
    n_patterns = int(np.prod(nav_shape))
    if st.REMOVE_STATIC_BACKGROUND:
        if s.static_background is None:
            step = max(1, n_patterns // 5000)
            flat = s.data.reshape((-1,) + sig_shape)[::step]
            bg = np.asarray(flat.mean(axis=0))
            s.static_background = bg.astype(s.data.dtype)
            log(f"Static background: mean of {flat.shape[0]} patterns")
        s.remove_static_background()
        log("Static background removed")
    if st.REMOVE_DYNAMIC_BACKGROUND:
        s.remove_dynamic_background()
        log("Dynamic background removed")


def preprocess_pattern_for_preview(pattern: np.ndarray, st: Settings) -> np.ndarray:
    """One pattern as the indexer sees it (adaptive histogram equalization, Gaussian background)."""
    from kikuchipy.indexing._spherical._preprocessing import _preprocess_pattern

    return _preprocess_pattern(np.asarray(pattern), gaussian_background=st.GAUSSIAN_BACKGROUND, n_regions=st.N_REGIONS)


# ----------------------------------------------------------------------------------------------
# Master pattern(s) and detector
# ----------------------------------------------------------------------------------------------


def load_harmonics(st: Settings, log: Log = _print, demo: bool = False) -> list:
    """Harmonics of every master in MASTER_FILE (.sht or EMsoft .h5), like section 4 of the notebook."""
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
    """The EBSDDetector of section 5 of the notebook."""
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
    from orix.crystal_map import CrystalMap
    from orix.quaternion import Rotation

    if not twist:
        return xmap
    rot = Rotation.from_axes_angles([0, 0, 1], float(twist), degrees=True)
    coordinates = {"x": xmap.x}
    if xmap.y is not None:
        coordinates["y"] = xmap.y
    return CrystalMap(
        rotations=xmap.rotations * rot,
        phase_id=xmap.phase_id,
        phase_list=xmap.phases,
        prop={name: xmap.prop[name] for name in xmap.prop},
        scan_unit=xmap.scan_unit,
        **coordinates,
    )


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
    free_gpu_memory()
    t0 = time.perf_counter()
    xmap_indexed = s.spherical_indexing(harmonics_arg, det, navigation_mask=navigation_mask, verbose=verbose, **kwargs)
    dt = time.perf_counter() - t0
    xmap = apply_twist(xmap_indexed, st.TWIST)
    log(f"{xmap.size} patterns in {dt:.1f} s = {xmap.size / dt:.0f} patterns/s ({st.BACKEND}, bandwidth {st.BANDWIDTH})")
    if st.TWIST:
        log(f"Orientations rotated by {st.TWIST} deg about the sample normal (TWIST)")
    return xmap, xmap_indexed, dt


def ipf_rgb(xmap, direction="z") -> np.ndarray:
    """IPF colours (n, 3) of a crystal map for the sample direction "x", "y" or "z"."""
    from orix import plot
    from orix.quaternion import Orientation
    from orix.vector import Vector3d

    vec = {"x": Vector3d.xvector(), "y": Vector3d.yvector(), "z": Vector3d.zvector()}[direction]
    rgb = np.ones((xmap.size, 3))
    for phase_id, phase in xmap.phases_in_data:
        if phase_id == -1:
            continue
        mask = xmap.phase_id == phase_id
        ckey = plot.IPFColorKeyTSL(phase.point_group, direction=vec)
        rgb[mask] = ckey.orientation2color(Orientation(xmap.rotations[mask], phase.point_group))
    return rgb


def ipf_rgb_map(xmap, direction="z", background=0.6) -> np.ndarray:
    """IPF colours as a (rows, cols, 3) image; points not in the data (e.g. masked out) are grey."""
    rgb = ipf_rgb(xmap, direction)
    full = np.full((xmap.is_in_data.size, 3), float(background))
    full[xmap.is_in_data] = rgb
    return full.reshape(tuple(xmap.shape) + (3,))


def map_data(xmap, name: str) -> np.ndarray:
    return np.asarray(xmap.get_map_data(name), dtype=float)


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
        io.save(path, xmap, overwrite=True, image_quality_prop=iq_prop, confidence_index_prop=ci_prop)
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


def compare_with_reference(xmap_indexed, reference_file: str | Path, roi=None, full_nav_shape=None):
    """Misorientation per point to a reference map of the same points (EMSphInx/OIM .ang/.h5/.ctf)."""
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
    if rot_ref.size != xmap_indexed.size:
        raise ValueError(f"Reference has {rot_ref.size} points, this map has {xmap_indexed.size}")
    ids = [i for i in xmap_indexed.phases_in_data.ids if i != -1]
    pg = xmap_indexed.phases[ids[0]].point_group
    return Orientation(xmap_indexed.rotations, pg).angle_with(Orientation(rot_ref.flatten(), pg), degrees=True)


# ----------------------------------------------------------------------------------------------
# HROSM
# ----------------------------------------------------------------------------------------------


def load_master_pattern(st: Settings, log: Log = _print, demo: bool = False):
    """The EMsoft master pattern (Lambert, both hemispheres) that HROSM simulates patterns from."""
    import kikuchipy as kp

    if demo or st.HROSM_MASTER_FILE is None:
        if not demo:
            raise ValueError("HROSM_MASTER_FILE is not set (an EMsoft master pattern .h5; .sht files cannot be used)")
        return kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    kwargs = {"projection": "lambert", "hemisphere": "both"}
    if st.HROSM_ENERGY is not None:
        kwargs["energy"] = st.HROSM_ENERGY
    log(f"Loading master pattern {Path(st.HROSM_MASTER_FILE).name}")
    return kp.load(st.HROSM_MASTER_FILE, **kwargs)


def run_hrosm(s, xmap_indexed, mp, det, st: Settings, log: Log = _print):
    """EBSD.hrosm() on the map as indexed (without TWIST); returns (xmap with TWIST, xmap as re-indexed, seconds, warnings)."""
    import warnings

    import dask
    import kikuchipy as kp

    dask.config.set(num_workers=int(st.N_WORKERS))
    ids = [i for i in xmap_indexed.phases_in_data.ids if i != -1]
    if len(ids) > 1 and mp.phase.name not in [xmap_indexed.phases[i].name for i in ids]:
        raise ValueError(
            f"The map has the phases {[xmap_indexed.phases[i].name for i in ids]}; the master pattern's phase "
            f"{mp.phase.name!r} must be one of them"
        )
    if mp.phase.point_group != xmap_indexed.phases[ids[0]].point_group:
        log(
            f"NB! Point group of the master pattern ({mp.phase.point_group.name}) and of the map "
            f"({xmap_indexed.phases[ids[0]].point_group.name}) differ"
        )
    nav_shape = s.axes_manager.navigation_shape[::-1]
    nav_mask = None
    if st.HROSM_ROI is not None:
        c0, c1, r0, r1 = st.HROSM_ROI
        nav_mask = np.ones(nav_shape, dtype=bool)
        nav_mask[r0:r1, c0:c1] = False
    signal_mask = ~kp.filters.Window("circular", det.shape).astype(bool) if st.CIRCULAR_MASK else None
    log(
        f"HROSM: ball of {(2 * st.HROSM_N_STEPS + 1) ** 3} orientations within {st.HROSM_MAX_ANGLE} deg "
        f"(mean spacing {kp.indexing.misorientation_ball_spacing(st.HROSM_MAX_ANGLE, st.HROSM_N_STEPS):.3f} deg)"
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        xmap_hrosm_indexed = s.hrosm(
            xmap_indexed,
            mp,
            det,
            energy=st.HROSM_ENERGY,
            threshold=float(st.HROSM_THRESHOLD),
            max_angle=float(st.HROSM_MAX_ANGLE),
            n_steps=int(st.HROSM_N_STEPS),
            keep_n=int(st.HROSM_KEEP_N),
            n_osm=int(st.HROSM_N_OSM),
            average=st.HROSM_AVERAGE,
            min_pixels=int(st.HROSM_MIN_PIXELS),
            dilate=bool(st.HROSM_DILATE),
            signal_mask=signal_mask,
            navigation_mask=nav_mask,
        )
    dt = time.perf_counter() - t0
    messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
    for m in messages:
        log("NB! " + m)
    xmap_hrosm = apply_twist(xmap_hrosm_indexed, st.TWIST)
    reindexed = np.asarray(xmap_hrosm.prop["reindexed"]).astype(bool)
    n_grains = np.unique(np.asarray(xmap_hrosm.prop["grain_id"])[reindexed]).size
    log(f"HROSM: {int(reindexed.sum())} of {reindexed.size} points re-indexed in {n_grains} grains, {dt / 60:.1f} min")
    if reindexed.any():
        osm = np.asarray(xmap_hrosm.prop["osm"])[reindexed]
        log(f"Median OSM of the re-indexed points: {np.nanmedian(osm):.1f} of {st.HROSM_N_OSM}")
    return xmap_hrosm, xmap_hrosm_indexed, dt, messages


def hrosm_output_map(xmap_hrosm, xmap_indexed):
    """The HROSM map with the spherical indexing score and IQ carried over, ready to save."""
    from orix.crystal_map import CrystalMap

    prop = {k: np.asarray(v) for k, v in xmap_hrosm.prop.items() if k != "simulation_indices"}
    prop["scores_sph"] = np.asarray(xmap_indexed.prop["scores"])
    prop["iq"] = np.asarray(xmap_indexed.prop["iq"])
    reindexed = prop["reindexed"].astype(bool)
    scores = prop["scores"]
    best = scores[:, 0] if scores.ndim == 2 else scores
    prop["best_score"] = np.where(reindexed, np.nan_to_num(best), prop["scores_sph"])
    return CrystalMap(
        rotations=xmap_hrosm.rotations,
        phase_id=xmap_hrosm.phase_id,
        phase_list=xmap_hrosm.phases,
        prop=prop,
        scan_unit=xmap_hrosm.scan_unit,
        x=xmap_hrosm.x,
        y=xmap_hrosm.y,
    )


def grod_of(xmap, grain_id: np.ndarray) -> np.ndarray:
    """GROD map of a crystal map about the mean orientations of the given grains."""
    import kikuchipy as kp

    grains = kp.indexing.average_grain_orientations(xmap, grain_id)
    return np.asarray(kp.indexing.grain_reference_orientation_deviation_map(xmap, grain_id, grains), dtype=float)
