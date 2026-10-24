"""Orientation refinement with a projection center (PC) per map point, starting from an EMSphInx result.

CPU only (kikuchipy pattern-matching refinement, numba + dask threads, nlopt). This is the former section 11
of the project's spherical indexing notebook turned into a script, with the starting orientations read from
an .ang file. It is the only form of the per-point refinement in the repository (a cheaper projection center
drift correction without patterns was made as a separate notebook that is not part of the repository).

Flow (https://kikuchipy.org/en/stable/tutorials/hybrid_indexing.html, without the Hough / dictionary parts):

1. Calibration: the best pattern of each block of a PC_GRID over the map is refined in orientation and
   PC together (refine_orientation_projection_center), and a plane is fitted to the refined PCs
   (EBSDDetector.fit_pc). That gives a PC for every map point.
2. "test": the orientations of a TEST_SIZE x TEST_SIZE region in the middle of the map are refined with
   those PCs, which gives the speed and hence the time for the full map.
3. "full": the orientations of the full map (or of REFINE_ROI) are refined and saved.

Usage:
    python refine_pc_per_point.py test     # calibration + test region (minutes)
    python refine_pc_per_point.py full     # calibration + full map (hours)

Everything is set in the SETTINGS block. Nothing below it needs editing.
"""

import os
import sys
import time
from pathlib import Path

import dask
import numpy as np

import kikuchipy as kp
from orix import io
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Orientation, Rotation

# ============================================ SETTINGS ============================================
DATA_DIR = Path(r"/path/to/data")  # <-- the folder on the cluster that holds the files below
PATTERN_FILE = DATA_DIR / "MB6-103 Real Chi_New Sample_Area 1_EBSD+EDS Map 1_NLPAR1-Norm.up1"
ANG_FILE = DATA_DIR / "SphInx_TopRefined.ang"      # EMSphInx result: the starting orientations
MASTER_PATTERN_FILE = DATA_DIR / "Ni-master.h5"    # EMsoft master pattern of the phase (refinement cannot use .sht)
ENERGY = 20            # beam energy in keV of the master pattern to use (check it exists in the .h5)
OUTPUT_DIR = DATA_DIR
OUTPUT_NAME = "SphInx_TopRefined_kp_refined"       # output files are <OUTPUT_NAME>.h5 / .ang / _pc.npy

# Patterns
LAZY = True            # True needs the loader patch for .up1 (see the handoff); False loads the whole file into RAM
NAV_SHAPE = None       # (n_rows, n_cols). None = from the pattern file, else from the .ang header
REMOVE_STATIC_BACKGROUND = False   # the NLPAR "Norm" file is already processed; set True for the raw file
REMOVE_DYNAMIC_BACKGROUND = False

# Detector geometry (EDAX Quattro camera, conventional EBSD). PC None = x-star, y-star, z-star of the .ang header
PC = None
PC_CONVENTION = "edax"
SAMPLE_TILT = 70.0     # degrees
CAMERA_TILT = 8.0      # camera elevation in degrees
PX_SIZE = 260.87       # um per detector pixel (30 mm wide camera at 115 px); only used for a consistency print
BINNING = 1
TWIST = 0.0            # detector rotation about its normal; 0 for conventional EBSD (15 for the RKD setup)

# Refinement
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))   # dask threads
REFINE_BINNING = 1     # bin the patterns by this factor before refinement (patterns are 115 px, so 1)
PC_GRID = (5, 5)       # (rows, cols) of map blocks; the best pattern of each block calibrates the PC
PC_TRUST_REGION = 0.05 # calibration: how far each PC value may move from PC
REFINE_TRUST_REGION = 2.0   # how far each Euler angle may move from the starting orientation, in degrees
REFINE_RTOL = 1e-7     # stop refining a pattern when the score changes by less than this (relative)
REFINE_MAXEVAL = 300   # ... or after this many simulated patterns
REFINE_ROI = None      # (col_start, col_stop, row_start, row_stop) to refine only part of the map. None = all
TEST_SIZE = 50         # side length in map points of the test region
SAVE_ANG = True
SAVE_H5 = True
# ===================================================================================================

stage = sys.argv[1] if len(sys.argv) > 1 else "test"
if stage not in ("test", "full"):
    raise SystemExit("usage: python refine_pc_per_point.py test|full")
dask.config.set(num_workers=N_WORKERS)
print(f"kikuchipy {kp.__version__}, {N_WORKERS} dask workers, stage: {stage}")


# ---------------------------------------- helpers (from the notebook) ----------------------------------------
def rotate_about_z(xmap, degrees):
    """Crystal map with the orientations rotated by `degrees` about the sample normal (the notebook's TWIST)."""
    if not degrees:
        return xmap
    twist = Rotation.from_axes_angles([0, 0, 1], degrees, degrees=True)
    coordinates = {"x": xmap.x}
    if xmap.y is not None:
        coordinates["y"] = xmap.y
    return CrystalMap(
        rotations=xmap.rotations * twist,
        phase_id=xmap.phase_id,
        phase_list=xmap.phases,
        prop={name: xmap.prop[name] for name in xmap.prop},
        scan_unit=xmap.scan_unit,
        **coordinates,
    )


def neighbour_misorientation(xmap):
    """Misorientation of every map point to its right and its lower neighbour in degrees, as two 2D arrays.

    Inside a grain this is mostly the noise of the orientations. Points without that neighbour are NaN.
    """
    point_group = xmap.phases[[i for i in xmap.phases_in_data.ids if i != -1][0]].point_group
    quat = xmap.rotations.data.reshape(xmap.shape + (4,))
    right = np.full(xmap.shape, np.nan)
    below = np.full(xmap.shape, np.nan)
    o1 = Orientation(quat[:, :-1].reshape(-1, 4), point_group)
    o2 = Orientation(quat[:, 1:].reshape(-1, 4), point_group)
    right[:, :-1] = o1.angle_with(o2, degrees=True).reshape(quat[:, :-1].shape[:2])
    o1 = Orientation(quat[:-1].reshape(-1, 4), point_group)
    o2 = Orientation(quat[1:].reshape(-1, 4), point_group)
    below[:-1] = o1.angle_with(o2, degrees=True).reshape(quat[:-1].shape[:2])
    return right, below


def report_refinement(xmap_ref, xmap_before, title=""):
    """Neighbour misorientation before and after refinement, and how far refinement rotated the orientations."""
    before = np.nanmean(neighbour_misorientation(xmap_before), axis=0)
    after = np.nanmean(neighbour_misorientation(xmap_ref), axis=0)
    rotated = xmap_before.orientations.angle_with(xmap_ref.orientations, degrees=True)
    print(f"--- {title} ---")
    for name, values in (("before", before), ("refined", after)):
        inside = values[values < 1]
        print(f"Neighbour misorientation inside grains, {name}: median {np.median(inside):.3f} deg, mean {inside.mean():.3f} deg")
    print(f"Rotated by refinement: median {np.median(rotated):.3f} deg, p99 {np.percentile(rotated, 99):.3f} deg")
    print(f"Score: mean {xmap_ref.scores.mean():.3f}; simulated patterns per point: mean {xmap_ref.num_evals.mean():.0f}, max {xmap_ref.num_evals.max()}")


# ---------------------------------------- 1. starting map from the .ang ----------------------------------------
xmap_ang = io.load(ANG_FILE)
ids = [i for i in xmap_ang.phases_in_data.ids if i != -1]
if len(ids) != 1:
    raise ValueError("This script is written for a map with one phase")
phase = xmap_ang.phases[ids[0]]
if not phase.name:
    phase.name = "ni"
# The notebook's spherical indexing map carries the correlation score as "scores" and the image quality as
# "iq". EMSphInx writes the score into the CI column, which orix reads as "ci".
if "scores" not in xmap_ang.prop:
    xmap_ang.prop["scores"] = xmap_ang.prop["ci"]
# Orientations as indexed, i.e. with the detector twist removed (no-op for TWIST = 0)
xmap_indexed = rotate_about_z(xmap_ang, -TWIST)
nav_shape = xmap_indexed.shape
n_rows, n_cols = nav_shape
n_patterns = xmap_indexed.size
print(f"Starting map: {n_patterns} points, {nav_shape} (rows, cols), phase {phase.name} ({phase.point_group.name})")

# ---------------------------------------- 2. patterns ----------------------------------------
kwargs = {"lazy": LAZY}
if NAV_SHAPE is not None:
    kwargs["nav_shape"] = tuple(NAV_SHAPE)
s = kp.load(PATTERN_FILE, **kwargs)
if s.axes_manager.navigation_shape[::-1] != nav_shape:
    # Old .up1 files store no map size: take it from the .ang
    s = kp.load(PATTERN_FILE, nav_shape=nav_shape, **{k: v for k, v in kwargs.items() if k != "nav_shape"})
for axis, step in zip(s.axes_manager.navigation_axes, (xmap_indexed.dx, xmap_indexed.dy)):
    axis.scale = step
    axis.units = "um"
sig_shape = s.axes_manager.signal_shape[::-1]
if s.axes_manager.navigation_shape[::-1] != nav_shape:
    raise ValueError(f"Pattern file map {s.axes_manager.navigation_shape[::-1]} != .ang map {nav_shape}")
print(f"Patterns: {sig_shape} px, lazy: {s._lazy}")
if REMOVE_STATIC_BACKGROUND:
    s.remove_static_background()
if REMOVE_DYNAMIC_BACKGROUND:
    s.remove_dynamic_background()

# ---------------------------------------- 3. detector ----------------------------------------
if PC is None:
    header = [line for line in open(ANG_FILE, encoding="utf-8", errors="ignore") if line.startswith("#")]
    pc = {}
    for line in header:
        parts = line[1:].replace(":", " ").split()  # '# x-star 0.5' (EMSphInx, orix) and '# x-star: 0.5' (OIM)
        if len(parts) == 2 and parts[0] in ("x-star", "y-star", "z-star"):
            pc[parts[0]] = float(parts[1])
        if len(pc) == 3:
            break
    if len(pc) < 3:
        raise ValueError(f"{ANG_FILE}: no x-star/y-star/z-star in the header; set PC")
    pc = (pc["x-star"], pc["y-star"], pc["z-star"])
else:
    pc = tuple(PC)
print(f"Projection center: {pc} ({PC_CONVENTION} convention)")
det = kp.detectors.EBSDDetector(
    shape=tuple(sig_shape),
    pc=pc,
    convention=PC_CONVENTION,
    sample_tilt=SAMPLE_TILT,
    tilt=CAMERA_TILT,
    px_size=PX_SIZE,
    binning=BINNING,
)

# ---------------------------------------- 11.1 master pattern and patterns for refinement ----------------------------------------
mp = kp.load(MASTER_PATTERN_FILE, projection="lambert", hemisphere="both", energy=ENERGY)
mp.phase = phase.deepcopy()  # refinement requires the phase of the map and of the master pattern to be equal
print(mp)

if REFINE_BINNING > 1:
    rows = sig_shape[0] - sig_shape[0] % REFINE_BINNING
    cols = sig_shape[1] - sig_shape[1] % REFINE_BINNING
    det_ref = det.crop((0, rows, 0, cols))  # updates the PC for the cropped pattern
    det_ref.pc = det_ref.pc_average
    s_ref = s.isig[:cols, :rows].downsample(REFINE_BINNING, inplace=False)
    det_ref.shape = s_ref.axes_manager.signal_shape[::-1]
    det_ref.binning = REFINE_BINNING
else:
    s_ref = s
    det_ref = det.deepcopy()
print(f"Patterns for refinement: {det_ref.shape} px (binning {REFINE_BINNING})")

# ---------------------------------------- 11.2 calibration: a PC per map point ----------------------------------------
# The best pattern of each block of the map: the highest score among the points inside a grain,
# i.e. with all four neighbours within 1 degree
scores_indexed = xmap_indexed.get_map_data("scores")
right, below = neighbour_misorientation(xmap_indexed)
in_grain = (right < 1) & (below < 1)
in_grain[:, 1:] &= right[:, :-1] < 1
in_grain[1:] &= below[:-1] < 1

row_edges = np.linspace(0, n_rows, PC_GRID[0] + 1).astype(int)
col_edges = np.linspace(0, n_cols, PC_GRID[1] + 1).astype(int)
cal_rc = []
for r0, r1 in zip(row_edges[:-1], row_edges[1:]):
    for c0, c1 in zip(col_edges[:-1], col_edges[1:]):
        block = np.where(in_grain[r0:r1, c0:c1], scores_indexed[r0:r1, c0:c1], -np.inf)
        if np.isfinite(block.max()):
            r, c = np.unravel_index(np.argmax(block), block.shape)
            cal_rc.append((r0 + r, c0 + c))
cal_rc = np.array(cal_rc)
print(f"{len(cal_rc)} calibration patterns")

cal_patterns = dask.compute(*[s_ref.data[r, c] for r, c in cal_rc])
s_cal = kp.signals.EBSD(np.stack([np.asarray(p) for p in cal_patterns]))
s_cal.axes_manager[0].name = "x"
cal_flat = np.ravel_multi_index(cal_rc.T, nav_shape)
xmap_cal = CrystalMap(
    rotations=xmap_indexed.rotations[cal_flat],
    phase_id=np.zeros(len(cal_rc), dtype=int),
    phase_list=PhaseList(phase),
    x=np.arange(len(cal_rc), dtype=float),
)
det_cal = det_ref.deepcopy()
det_cal.pc = np.tile(det_ref.pc_average, (len(cal_rc), 1))

t0 = time.perf_counter()
xmap_cal_ref, det_cal_ref = s_cal.refine_orientation_projection_center(
    xmap=xmap_cal,
    detector=det_cal,
    master_pattern=mp,
    energy=ENERGY,
    method="LN_NELDERMEAD",
    trust_region=3 * [REFINE_TRUST_REGION] + 3 * [PC_TRUST_REGION],
    rtol=REFINE_RTOL,
    maxeval=2 * REFINE_MAXEVAL,  # six values are searched for here, three in the refinement of the map
    chunk_kwargs=dict(chunk_shape=1),  # one pattern per chunk, so that all workers are used
)
print(f"Calibration: {time.perf_counter() - t0:.0f} s")

pc_cal = det_cal_ref.pc_flattened
pc_shift = pc_cal - det_ref.pc_average
angles_cal = xmap_cal.orientations.angle_with(xmap_cal_ref.orientations, degrees=True)
# A PC is left out of the fit if its search ran into the trust region or did not finish, or if the PC
# lies far from the others
at_bound = (abs(pc_shift) > 0.95 * PC_TRUST_REGION).any(axis=1) | (xmap_cal_ref.num_evals >= 2 * REFINE_MAXEVAL)
deviation = abs(pc_cal - np.median(pc_cal, axis=0))
spread = 1.4826 * np.median(deviation, axis=0) + 1e-6
is_outlier = at_bound | (deviation > 5 * spread + 0.002).any(axis=1)

print(" #   row  col   score  evals  rotated   PCx      PCy      PCz")
for i, (r, c) in enumerate(cal_rc):
    print(
        f"{i:2d}  {r:4d} {c:4d}  {xmap_cal_ref.scores[i]:.3f}  {xmap_cal_ref.num_evals[i]:5d}  {angles_cal[i]:5.2f} deg"
        f"  {pc_cal[i, 0]:.4f}  {pc_cal[i, 1]:.4f}  {pc_cal[i, 2]:.4f}" + ("   left out" if is_outlier[i] else "")
    )
print(f"PC before (Bruker convention): {np.round(det_ref.pc_average, 4)}")
print(f"PC after, mean of those used:  {np.round(pc_cal[~is_outlier].mean(axis=0), 4)}")
print(f"PC after, their spread (std):  {np.round(pc_cal[~is_outlier].std(axis=0), 4)}")

# Plane of PCs for the whole map. The sample tilt which fit_pc() estimates from the plane is not used.
det_plane = det_cal_ref.fit_pc(
    cal_rc.T,
    map_indices=np.indices(nav_shape),
    transformation="affine",
    is_outlier=is_outlier,
)
det_plane.sample_tilt = det_ref.sample_tilt

pc_plane = det_plane.pc
step_x, step_y = [a.scale for a in s.axes_manager.navigation_axes]
detector_width = PX_SIZE * BINNING * sig_shape[1]  # um
per_col = (pc_plane[0, -1] - pc_plane[0, 0]) / (n_cols - 1)
per_row = (pc_plane[-1, 0] - pc_plane[0, 0]) / (n_rows - 1)
print(f"PC average of the plane (Bruker convention): {np.round(det_plane.pc_average, 4)}")
print("Change of (PCx, PCy, PCz) over the map")
print(f"  along scan x: {np.round(per_col * (n_cols - 1), 4)}")
print(f"  along scan y: {np.round(per_row * (n_rows - 1), 4)}")
print(
    f"Change in the detector plane per scan step: measured {np.hypot(*per_col[:2]):.2e} (x), "
    f"{np.hypot(*per_row[:2]):.2e} (y); expected from step size and detector size {step_x / detector_width:.2e}"
)
print(
    "Direction of the PC shift along scan x, measured from the detector x axis: "
    f"{np.rad2deg(np.arctan2(per_col[1], per_col[0])):.1f} deg (compare with TWIST = {TWIST})"
)
# Check before going on: the refined PCs of the calibration patterns should lie close to a plane, and the
# measured change per scan step should be close to the expected one. If the PCs scatter by more than they
# change over the map, use more calibration patterns (PC_GRID). To go on with one PC for the whole map
# instead: det_plane = det_ref.deepcopy()

# ---------------------------------------- 11.3 / 11.4 refinement ----------------------------------------
ref_kwargs = dict(
    master_pattern=mp,
    energy=ENERGY,
    method="LN_NELDERMEAD",
    trust_region=3 * [REFINE_TRUST_REGION],
    rtol=REFINE_RTOL,
    maxeval=REFINE_MAXEVAL,
)


def refine_region(region):
    """Refine the orientations of the map region (col_start, col_stop, row_start, row_stop).

    Returns the refined map and the starting map of that region, both without TWIST.
    """
    c0, c1, r0, r1 = region
    if (c1 - c0) * (r1 - r0) == n_patterns:
        nav_mask = None
    else:
        nav_mask = np.ones(nav_shape, dtype=bool)  # True = do not refine
        nav_mask[r0:r1, c0:c1] = False
    xmap_region_ref = s_ref.refine_orientation(
        xmap=xmap_indexed, detector=det_plane, navigation_mask=nav_mask, **ref_kwargs
    )
    return xmap_region_ref, xmap_indexed[r0:r1, c0:c1]


refine_roi = (0, n_cols, 0, n_rows) if REFINE_ROI is None else tuple(REFINE_ROI)
n_refine = (refine_roi[1] - refine_roi[0]) * (refine_roi[3] - refine_roi[2])

# Test region in the middle of the map
r0 = max(0, n_rows // 2 - TEST_SIZE // 2)
c0 = max(0, n_cols // 2 - TEST_SIZE // 2)
test_region = (c0, min(n_cols, c0 + TEST_SIZE), r0, min(n_rows, r0 + TEST_SIZE))

_ = refine_region((c0, c0 + 4, r0, r0 + 1))  # warm-up so that numba compilation is not part of the timing
t0 = time.perf_counter()
xmap_test_ref, xmap_test_before = refine_region(test_region)
dt = time.perf_counter() - t0
speed = xmap_test_ref.size / dt
print(f"Test region: {xmap_test_ref.size} patterns in {dt:.1f} s = {speed:.1f} patterns/s ({det_ref.shape} px, {N_WORKERS} workers)")
print(f"Estimated time for the {n_refine} patterns to refine: {n_refine / speed / 3600:.1f} h")
report_refinement(xmap_test_ref, xmap_test_before, "Test region")

if stage == "test":
    sys.exit(0)

t0 = time.perf_counter()
xmap_refined_indexed, xmap_before = refine_region(refine_roi)
dt = time.perf_counter() - t0
print(f"{xmap_refined_indexed.size} patterns in {dt / 3600:.2f} h = {xmap_refined_indexed.size / dt:.1f} patterns/s")

# The refined points as a map of their own, with the image quality and the score of the starting map.
# The coordinates start at zero, as in an .ang file of that region.
xmap_refined_indexed = CrystalMap(
    rotations=xmap_refined_indexed.rotations,
    phase_id=xmap_refined_indexed.phase_id,
    phase_list=xmap_refined_indexed.phases,
    prop={
        "scores": xmap_refined_indexed.scores,
        "num_evals": xmap_refined_indexed.num_evals,
        "iq": xmap_before.prop["iq"],
        "scores_indexed": xmap_before.prop["scores"],
    },
    scan_unit=xmap_refined_indexed.scan_unit,
    x=xmap_refined_indexed.x - xmap_refined_indexed.x.min(),
    y=xmap_refined_indexed.y - xmap_refined_indexed.y.min(),
)
report_refinement(xmap_refined_indexed, xmap_before, "Refined map")
xmap_refined = rotate_about_z(xmap_refined_indexed, TWIST)

# ---------------------------------------- 11.5 save ----------------------------------------
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
name = f"{OUTPUT_NAME}_bin{REFINE_BINNING}"
if SAVE_H5:
    io.save(out_dir / f"{name}.h5", xmap_refined, overwrite=True)
    print("Saved", out_dir / f"{name}.h5")
if SAVE_ANG:
    io.save(
        out_dir / f"{name}.ang",
        xmap_refined,
        overwrite=True,
        image_quality_prop="iq",
        confidence_index_prop="scores",
    )
    print("Saved", out_dir / f"{name}.ang")
np.save(out_dir / f"{name}_pc.npy", det_plane.pc)
print("Saved", out_dir / f"{name}_pc.npy", "(PC per map point for the binned patterns, Bruker convention)")
