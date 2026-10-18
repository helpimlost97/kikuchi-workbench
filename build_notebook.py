"""Build spherical_indexing_template.ipynb"""
import re
import sys
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s, tags=None):
    C.append(nbf.v4.new_code_cell(s.strip("\n"), metadata={"tags": tags or []}))


md(r"""
# Spherical indexing template (kikuchipy, jwestraadt fork)

General-purpose notebook for indexing an EBSD / TKD / RKD pattern file by spherical harmonic
cross-correlation (the Python port of EMSphInx), on CPU or GPU.
It follows the flow of Johan's tutorial (`kikuchipy/doc/tutorials/spherical_indexing.ipynb`).

**How to use it**

1. Type the paths of the pattern file and of its `.ang` file in the first cell (section 0) and run it.
   It prints the pattern width and height, the projection center, the map size and the step size.
2. Edit the **Settings** cell (section 1), e.g. to replace the projection center with a better one.
3. Run all cells down to section 6. That indexes a small test region so you can check the geometry.
4. If the test maps look right, run section 7 for the full map, then sections 8 and 9 to plot and save.

With `PATTERN_FILE = None` the notebook runs on kikuchipy's nickel demo dataset, so you can check
that everything works before pointing it at your own data.
""")

md("## 0. Files and scan parameters")

code(r'''
# ======================= FILES =======================
# Pattern file: EDAX .up1/.up2, or anything else kp.load() reads (EMsoft/kikuchipy/Bruker/EDAX/Oxford h5, ...).
# None = kikuchipy's nickel demo dataset (the detector settings are then taken from the demo data).
PATTERN_FILE = None

# The .ang file written during data collection (not an indexing result such as "..._SphInx.ang").
# An .edaxh5 file works as well. None = skip.
SCAN_FILE = None
EDAXH5_MAP = None     # only for .edaxh5 files with several maps: part of the map's name, e.g. "Area 8"

# Folder that contains ebsd_params.py (this project folder). None = the folder JupyterLab was started from,
# which is right when you start it from this folder (the .bat files do that).
PROJECT_DIR = None    # e.g. r"C:\path\to\GPU-Accelerated EBSD"

# ======================= no need to edit below this line =======================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(PROJECT_DIR) if PROJECT_DIR else Path.cwd()))
import ebsd_params

scan = ebsd_params.scan_info(SCAN_FILE, pattern_file=PATTERN_FILE, map_filter=EDAXH5_MAP)
ebsd_params.print_scan_info(scan)
''', tags=["files"])

md("## 1. Settings")

code(r'''
# ======================= PATTERNS =======================
# The pattern file and the scan file are set in the first cell (section 0).
NAV_SHAPE = None      # (n_rows, n_cols) of the map. None = from the pattern file, or else from the scan file
STEP_SIZE = None      # (step_x, step_y) in um. None = from the pattern file, or else from the scan file
LAZY = "auto"         # True / False / "auto" (lazy when the file is bigger than 25 % of RAM)
ROI = None            # (col_start, col_stop, row_start, row_stop) to index only part of the map. None = everything
FLIP_PATTERNS_UD = False   # flip every pattern upside down (try this if the test maps in section 6 look like noise)

# ======================= PATTERN PROCESSING =======================
# Set both to False for patterns that are already background corrected (e.g. "Processed" / NLPAR output)
REMOVE_STATIC_BACKGROUND = True    # uses the background stored with the file, or else the mean pattern of the scan
REMOVE_DYNAMIC_BACKGROUND = True

# ======================= MASTER PATTERN =======================
# EMSphInx .sht file, or an EMsoft master pattern .h5. Ignored for the demo dataset.
# For several phases give a list of files.
MASTER_FILE = None
# MASTER_FILE = r"C:\path\to\master.sht"
# MASTER_BANDWIDTH: only for .h5 masters, the bandwidth of the harmonics computed from the master pattern. Keep it
# >= 384 (the mp2sht default) even when BANDWIDTH below is much lower: the transform aliases at a low bandwidth and
# the aliasing is not cubic, so the symmetry copies of the correlation peak become unequal and the KAM map fills
# with variant-hop lines and rings (NOTES_master_pattern_symmetry.md).
MASTER_BANDWIDTH = 384
MASTER_ENERGY = None       # only for .h5 masters with several energies: beam energy in keV to load. None = all (weighted)
# The master pattern carries the sample tilt it was simulated for, and indexing refuses to run if it differs
# from SAMPLE_TILT. True = accept the mismatch and use SAMPLE_TILT anyway.
OVERRIDE_MASTER_SAMPLE_TILT = False

# ======================= DETECTOR GEOMETRY =======================
# Ignored for the demo dataset.
# Projection center (x*, y*, z*). As written it is the one of the scan file from the first cell. To use your
# own, replace the whole right hand side, e.g.  PC = (0.5090, 0.5180, 0.4052)
PC = scan["pc"] or (0.5, 0.5, 0.5)
PC_CONVENTION = "edax"     # "edax" (= "tsl"), "bruker", "oxford", "emsoft". Values from an .ang file are "edax"
SAMPLE_TILT = None         # degrees. None = the sample tilt the master pattern was simulated for (as EMSphInx does)
CAMERA_TILT = 0.0          # detector tilt / camera elevation in degrees (EMSphInx "thetac"), e.g. 8 for EBSD, -90 for RKD
PX_SIZE = 1.0              # detector pixel size in um (only needed for the "emsoft" PC convention)
BINNING = 1
# Rotation of the detector about its normal in degrees, e.g. 15 for RKD. Indexing itself cannot use a twisted
# detector, so the indexed orientations are rotated by this angle about the sample normal (Z) afterwards.
# IPF-Z is not changed by it, IPF-X and IPF-Y are. Flip the sign if X and Y come out rotated the wrong way.
TWIST = 0.0

# ======================= INDEXING =======================
BANDWIDTH = 68             # see kp.indexing.fast_bandwidths(); 68 is the EMSphInx default, e.g. 88 or 123 for more accuracy
BACKEND = "gpu"            # "gpu" or "cpu"
N_WORKERS = 8              # Dask threads; about the number of performance cores
REFINE = True              # Newton refinement of the orientation
N_REGIONS = 10             # tiles of the adaptive histogram equalization, 0 = off (EMSphInx "nregions")
CIRCULAR_MASK = False      # use only the inscribed circle of the detector (EMSphInx "circmask")
GAUSSIAN_BACKGROUND = False  # fit and subtract a 2D Gaussian background (EMSphInx "gausbckg")
NORMALIZE = True           # normalized cross-correlation (EMSphInx "normed")
CHUNKSIZE = None           # patterns per chunk = GPU batch size. None = automatic
TEST_SIZE = 50             # side length in map points of the test region indexed in section 6

# Refinement of the orientations with a projection center per map point has its own notebook, built by
# build_refinement_notebook.py from the .h5 that section 9 writes.

# ======================= HROSM (section 11, optional) =======================
# High angular resolution orientation similarity map: every grain of the indexed map is re-indexed by dictionary
# indexing against patterns simulated on a fine misorientation ball around the grain's reference orientation
# (EBSD.hrosm, the fork's port of EMsoftOO's EMHROSM). Gives orientations on a grid far finer than the global search,
# the OSM, KAM, grain labels and GROD. Patterns are simulated, so it needs the EMsoft master pattern .h5 of the phase;
# an .sht file cannot be used here. None = skip section 11 (the demo data set uses its own master pattern).
HROSM_MASTER_FILE = None   # e.g. r"C:\path\to\Ni-master.h5"
HROSM_ENERGY = None        # beam energy in keV of the master pattern to use. None = the highest in the file
HROSM_THRESHOLD = 5.0      # KAM difference in degrees below which neighbours join the same grain (EMHROSM gangle)
HROSM_MAX_ANGLE = 5.0      # radius in degrees of the misorientation ball (EMHROSM misorang); >= the largest GROD
HROSM_N_STEPS = 10         # steps per half axis of the ball: (2 n + 1)^3 orientations per grain. 20 = EMHROSM default
HROSM_KEEP_N = 20          # best matches kept per point
HROSM_N_OSM = 10           # best matches compared between neighbours in the OSM (<= HROSM_KEEP_N)
HROSM_AVERAGE = "mean"     # grain reference orientation: "mean", "center" (EMHROSM default), "vmf", "watson"
HROSM_MIN_PIXELS = 10      # smallest grain that is re-indexed
HROSM_DILATE = False       # grow the grains into unassigned neighbours before averaging
HROSM_ROI = None           # (col_start, col_stop, row_start, row_stop) of the indexed map to re-index. None = all

# ======================= OUTPUT =======================
OUTPUT_DIR = None          # None = next to the pattern file (or the current folder for the demo data)
OUTPUT_NAME = None         # None = "<pattern file name>_sph_bw<BANDWIDTH>"
SAVE_ANG = True
SAVE_H5 = True
REFERENCE_FILE = None      # optional .ang / .h5 / .ctf of the same map (e.g. from EMSphInx or OIM) to compare with
''', tags=["parameters"])

md("## 2. Imports")

code(r'''
%matplotlib inline

import time
import warnings
from pathlib import Path

import dask
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import kikuchipy as kp
from kikuchipy.indexing._spherical._preprocessing import _preprocess_pattern
from orix import io, plot
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Orientation, Rotation
from orix.vector import Vector3d

plt.rcParams.update({"figure.facecolor": "w", "font.size": 12})
_ = dask.config.set(num_workers=N_WORKERS)

print("kikuchipy", kp.__version__)
if BACKEND == "gpu":
    import cupy

    props = cupy.cuda.runtime.getDeviceProperties(0)
    free, total = cupy.cuda.runtime.memGetInfo()
    print(f"GPU: {props['name'].decode()}, {free / 1e9:.1f} of {total / 1e9:.1f} GB VRAM free")
''')

md("## 3. Load the patterns")

code(r'''
demo = PATTERN_FILE is None

if demo:
    s = kp.data.nickel_ebsd_large(allow_download=True)
else:
    pattern_file = Path(PATTERN_FILE)
    lazy = LAZY
    if lazy == "auto":
        import psutil

        lazy = pattern_file.stat().st_size > 0.25 * psutil.virtual_memory().total
    kwargs = {"lazy": bool(lazy)}
    nav_shape_in, step_size = NAV_SHAPE, STEP_SIZE
    if pattern_file.suffix.lower() in (".up1", ".up2"):
        # Old .up1 files store neither the map size nor the step size: take them from the scan file
        if not scan["up_has_map_size"]:
            if nav_shape_in is None and scan["n_rows"] and scan["grid"] != "HexGrid":
                nav_shape_in = (scan["n_rows"], scan["n_cols"])
            if step_size is None and scan["step_x"]:
                step_size = (scan["step_x"], scan["step_y"])
        if nav_shape_in is not None:
            kwargs["nav_shape"] = tuple(nav_shape_in)
    s = kp.load(pattern_file, **kwargs)
    if step_size is not None:
        for axis, step in zip(s.axes_manager.navigation_axes, step_size):
            axis.scale = step
            axis.units = "um"

full_nav_shape = s.axes_manager.navigation_shape[::-1]  # (rows, cols) before any ROI
if ROI is not None:
    c0, c1, r0, r1 = ROI
    s = s.inav[c0:c1, r0:r1]

if FLIP_PATTERNS_UD:
    s.data = s.data[..., ::-1, :]

nav_shape = s.axes_manager.navigation_shape[::-1]  # (rows, cols)
sig_shape = s.axes_manager.signal_shape[::-1]  # (rows, cols)
n_patterns = int(np.prod(nav_shape))
print(f"{n_patterns} patterns, map {nav_shape} (rows, cols), pattern {sig_shape} px (rows, cols), lazy: {s._lazy}")
if len(nav_shape) != 2:
    print("NB! The map has one navigation dimension (hexagonal grid or unknown shape), so maps cannot be plotted.")
s
''')

code(r'''
# A few raw patterns spread over the map
idx = np.linspace(0, n_patterns - 1, 5).astype(int)
fig, axes = plt.subplots(ncols=5, figsize=(15, 3.2))
for ax, i in zip(axes, idx):
    nav_idx = np.unravel_index(i, nav_shape)
    ax.imshow(np.asarray(s.data[nav_idx]), cmap="gray")
    ax.set_title(f"(row, col) = {tuple(int(j) for j in nav_idx)}", fontsize=10)
    ax.axis("off")
fig.suptitle("Raw patterns")
fig.tight_layout()
''')

md("### Pattern processing")

code(r'''
if REMOVE_STATIC_BACKGROUND:
    if s.static_background is None:
        # No background stored with the file: use the mean pattern of the scan
        step = max(1, n_patterns // 5000)  # at most about 5000 patterns
        flat = s.data.reshape((-1,) + tuple(sig_shape))[::step]
        bg = np.asarray(flat.mean(axis=0))
        s.static_background = bg.astype(s.data.dtype)
        print(f"Static background: mean of {flat.shape[0]} patterns")
    s.remove_static_background()
if REMOVE_DYNAMIC_BACKGROUND:
    s.remove_dynamic_background()

# Indexing equalizes every pattern itself (N_REGIONS), so the same routine is applied to this preview
fig, axes = plt.subplots(ncols=5, figsize=(15, 3.2))
for ax, i in zip(axes, idx):
    pattern = np.asarray(s.data[np.unravel_index(i, nav_shape)])
    pattern = _preprocess_pattern(pattern, gaussian_background=GAUSSIAN_BACKGROUND, n_regions=N_REGIONS)
    ax.imshow(pattern, cmap="gray")
    ax.axis("off")
if N_REGIONS > 0:
    equalization = f"adaptive histogram equalization with {N_REGIONS} x {N_REGIONS} tiles"
else:
    equalization = "no adaptive histogram equalization"
fig.suptitle(f"Patterns as indexing sees them ({equalization})")
fig.tight_layout()
''')

md("## 4. Master pattern(s) as spherical harmonics")

code(r'''
def load_harmonics(filename):
    """Harmonics from an EMSphInx .sht file or an EMsoft master pattern .h5 file."""
    filename = Path(filename)
    if filename.suffix.lower() == ".sht":
        return kp.indexing.MasterPatternHarmonics.from_file(filename)
    kwargs = {"projection": "lambert", "hemisphere": "both"}
    if MASTER_ENERGY is not None:
        kwargs["energy"] = MASTER_ENERGY
    mp = kp.load(filename, **kwargs)
    return mp.get_spherical_harmonics(bandwidth=MASTER_BANDWIDTH)


if demo:
    mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    harmonics = [mp.get_spherical_harmonics(bandwidth=MASTER_BANDWIDTH)]
else:
    files = [MASTER_FILE] if isinstance(MASTER_FILE, (str, Path)) else list(MASTER_FILE)
    harmonics = [load_harmonics(f) for f in files]

if demo:
    sample_tilt = 70.0
elif SAMPLE_TILT is not None:
    sample_tilt = float(SAMPLE_TILT)
elif harmonics[0].sample_tilt is not None:
    sample_tilt = float(harmonics[0].sample_tilt)
else:
    raise ValueError("The master pattern does not state a sample tilt: set SAMPLE_TILT in the settings")
print(f"Sample tilt used: {sample_tilt} deg" + ("" if SAMPLE_TILT is not None or demo else " (from the master pattern)"))
for h in harmonics:
    print(
        f"{h.phase.name}: point group {h.phase.point_group.name}, bandwidth {h.bandwidth}, "
        f"beam energy {h.beam_energy} keV, simulated sample tilt {h.sample_tilt} deg"
    )
    if h.bandwidth < BANDWIDTH:
        warnings.warn(f"The master of {h.phase.name} has bandwidth {h.bandwidth} < BANDWIDTH {BANDWIDTH}")
    if h.sample_tilt is not None and not np.isclose(h.sample_tilt, sample_tilt):
        if OVERRIDE_MASTER_SAMPLE_TILT:
            print(f"  -> overriding the master's sample tilt {h.sample_tilt} with {sample_tilt} deg")
            h.sample_tilt = sample_tilt
        else:
            warnings.warn(
                f"The master of {h.phase.name} was simulated for a sample tilt of {h.sample_tilt} deg but "
                f"SAMPLE_TILT is {sample_tilt} deg. Indexing will refuse to run. Use a matching master "
                "pattern or set OVERRIDE_MASTER_SAMPLE_TILT = True."
            )
''')

code(r'''
# The master pattern(s) as seen by indexing, synthesized back from the harmonics
fig, axes = plt.subplots(ncols=len(harmonics), figsize=(5 * len(harmonics), 5), squeeze=False)
for ax, h in zip(axes[0], harmonics):
    ax.imshow(h.to_master_pattern(dim=401).data[0], cmap="gray")
    ax.set_title(f"{h.phase.name} (northern hemisphere)")
    ax.axis("off")
fig.tight_layout()
''')

md("## 5. Detector geometry")

code(r'''
if demo:
    det = s.detector.deepcopy()
    det.pc = det.pc_average
else:
    print(f"Projection center used: {tuple(PC)} ({PC_CONVENTION} convention)")
    det = kp.detectors.EBSDDetector(
        shape=tuple(sig_shape),
        pc=PC,
        convention=PC_CONVENTION,
        sample_tilt=sample_tilt,
        tilt=CAMERA_TILT,
        px_size=PX_SIZE,
        binning=BINNING,
    )
det
''')

code(r'''
# Back-projection of one pattern onto the sphere through this geometry
projector = kp.indexing.SphericalBackProjector(det, bandwidth=BANDWIDTH)
pattern = np.asarray(s.data[np.unravel_index(idx[2], nav_shape)])
north, south, iq = projector.unproject(pattern, return_image_quality=True)

fig, axes = plt.subplots(ncols=3, figsize=(12, 4.2))
axes[0].imshow(pattern, cmap="gray")
axes[0].set(title="Pattern")
axes[1].imshow(north, cmap="gray")
axes[1].set(title="On the sphere (north)")
axes[2].imshow(projector.window_mask()[0], cmap="gray")
axes[2].set(title="Window")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
print(f"Image quality: {iq:.3f}")
''')

md(r"""
## 6. Test run on a small region

Indexes a `TEST_SIZE` x `TEST_SIZE` region from the middle of the map. Check that the IPF map shows
grains rather than noise before running the full map. If it looks like noise, the usual suspects are
the projection center or its convention, the camera tilt, `FLIP_PATTERNS_UD`, or the wrong master pattern.
""")

code(r'''
index_kwargs = dict(
    bandwidth=BANDWIDTH,
    normalize=NORMALIZE,
    refine=REFINE,
    n_regions=N_REGIONS,
    gaussian_background=GAUSSIAN_BACKGROUND,
    circular_mask=CIRCULAR_MASK,
    backend=BACKEND,
    chunksize=CHUNKSIZE,
)
harmonics_arg = harmonics[0] if len(harmonics) == 1 else harmonics

if BACKEND == "gpu" and CHUNKSIZE is None:
    # Fix the GPU batch size once, from the VRAM that is free now. Otherwise every call estimates it again,
    # and GPU memory still held from an earlier call makes later calls pick a tiny batch.
    from kikuchipy.indexing._spherical import _gpu

    index_kwargs["chunksize"] = _gpu._default_batch_size(BANDWIDTH, int(cupy.cuda.runtime.memGetInfo()[0]))
    print("GPU batch size:", index_kwargs["chunksize"])


def plot_maps(xmap, title=""):
    """Score, image quality and IPF-Z maps of a 2D crystal map."""
    if xmap.ndim != 2:
        print("Map is not 2D, skipping the plots")
        return
    fig, axes = plt.subplots(ncols=3, figsize=(16, 5))
    for ax, prop, name in zip(axes, ["scores", "iq"], ["Correlation score", "Image quality"]):
        im = ax.imshow(xmap.get_map_data(prop))
        ax.set(title=name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    rgb = np.ones((xmap.size, 3))
    for phase_id, phase in xmap.phases_in_data:
        if phase_id == -1:
            continue
        mask = xmap.phase_id == phase_id
        ckey = plot.IPFColorKeyTSL(phase.point_group, direction=Vector3d.zvector())
        rgb[mask] = ckey.orientation2color(Orientation(xmap.rotations[mask], phase.point_group))
    axes[2].imshow(rgb.reshape(xmap.shape + (3,)))
    axes[2].set(title="IPF-Z")
    axes[2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()


if len(nav_shape) == 2:
    n_rows, n_cols = nav_shape
    r0 = max(0, n_rows // 2 - TEST_SIZE // 2)
    c0 = max(0, n_cols // 2 - TEST_SIZE // 2)
    s_test = s.inav[c0 : c0 + TEST_SIZE, r0 : r0 + TEST_SIZE]
    s_warm = s_test.inav[:4, :1]
else:
    s_test = s.inav[: TEST_SIZE * TEST_SIZE]
    s_warm = s_test.inav[:4]

def apply_twist(xmap):
    """Crystal map with the orientations rotated by TWIST degrees about the sample normal."""
    if not TWIST:
        return xmap
    twist = Rotation.from_axes_angles([0, 0, 1], TWIST, degrees=True)
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


def free_gpu_memory():
    """Release the GPU memory held by CuPy's pool, so that the automatic batch size sees the real free VRAM."""
    if BACKEND == "gpu":
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()


# Warm-up so that compilation is not part of the timing
_ = s_warm.spherical_indexing(harmonics_arg, det, verbose=0, **index_kwargs)
free_gpu_memory()

t0 = time.perf_counter()
xmap_test = s_test.spherical_indexing(harmonics_arg, det, verbose=0, **index_kwargs)
dt = time.perf_counter() - t0
xmap_test = apply_twist(xmap_test)
speed = xmap_test.size / dt
print(f"Test region: {xmap_test.size} patterns in {dt:.1f} s = {speed:.0f} patterns/s ({BACKEND}, bandwidth {BANDWIDTH})")
print(f"Estimated time for the full map of {n_patterns} patterns: {n_patterns / speed / 60:.1f} min")
plot_maps(xmap_test, "Test region")
''')

md("## 7. Index the full map")

code(r'''
free_gpu_memory()
t0 = time.perf_counter()
xmap_indexed = s.spherical_indexing(harmonics_arg, det, **index_kwargs)  # as indexed, without the twist
dt = time.perf_counter() - t0
xmap = apply_twist(xmap_indexed)
print(f"{xmap.size} patterns in {dt / 60:.2f} min = {xmap.size / dt:.0f} patterns/s")
if TWIST:
    print(f"Orientations rotated by {TWIST} deg about the sample normal (TWIST)")
xmap
''')

md("## 8. Maps")

code(r'''
plot_maps(xmap, "Full map")
''')

code(r'''
# IPF color key(s)
for phase_id, phase in xmap.phases_in_data:
    if phase_id == -1:
        continue
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(projection="ipf", symmetry=phase.point_group)
    ax.plot_ipf_color_key(show_title=False)
    ax.set_title(phase.name, fontsize=11)
''')

md(r"""
## 9. Save

The `.h5` written here is the starting point of the projection center drift correction notebook
(`spherical_refinement_template.ipynb`, built by `build_refinement_notebook.py`), which corrects these orientations
for the projection center moving with the beam. Section 11 below (HROSM) also starts from the indexed map.
""")

code(r'''
if demo:
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path.cwd()
    stem = OUTPUT_NAME or f"nickel_demo_sph_bw{BANDWIDTH}"
else:
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(PATTERN_FILE).parent
    stem = OUTPUT_NAME or f"{Path(PATTERN_FILE).stem}_sph_bw{BANDWIDTH}"
out_dir.mkdir(parents=True, exist_ok=True)

if SAVE_H5:
    io.save(out_dir / f"{stem}.h5", xmap, overwrite=True)
    print("Saved", out_dir / f"{stem}.h5")
if SAVE_ANG:
    io.save(
        out_dir / f"{stem}.ang",
        xmap,
        overwrite=True,
        image_quality_prop="iq",
        confidence_index_prop="scores",
    )
    print("Saved", out_dir / f"{stem}.ang")
''')

md(r"""
## 10. Optional: compare with a reference map

Set `REFERENCE_FILE` in the settings to an `.ang`, `.ctf` or `.h5` crystal map of the same points
(for example the EMSphInx or OIM result). The two maps must have the same number of points.
""")

code(r'''
def load_reference_rotations(filename):
    """Rotations of a reference map: EMSphInx / EDAX style h5 (Scan 1/EBSD/Data) or anything orix reads."""
    import h5py

    filename = Path(filename)
    if filename.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(filename, "r") as f:
            scans = [k for k in f if isinstance(f[k], h5py.Group) and "EBSD/Data" in f[k]]
            if scans:
                data = f[scans[0]]["EBSD/Data"]
                if all(k in data for k in ("Phi1", "Phi", "Phi2")):
                    euler = np.stack([data[k][()] for k in ("Phi1", "Phi", "Phi2")], axis=1)
                    return Rotation.from_euler(euler)
    return io.load(filename).rotations


if REFERENCE_FILE is not None:
    rot_ref = load_reference_rotations(REFERENCE_FILE)
    if rot_ref.size == int(np.prod(full_nav_shape)) and ROI is not None and len(full_nav_shape) == 2:
        # Reference covers the full map: cut out the ROI
        c0, c1, r0, r1 = ROI
        quat = rot_ref.data.reshape(tuple(full_nav_shape) + (4,))[r0:r1, c0:c1]
        rot_ref = type(rot_ref)(quat.reshape(-1, 4))
    if rot_ref.size != xmap.size:
        print(f"Reference has {rot_ref.size} points, this map has {xmap.size}. Cannot compare point by point.")
    else:
        ids = [i for i in xmap.phases_in_data.ids if i != -1]
        pg = xmap.phases[ids[0]].point_group
        # The orientations as indexed are compared, i.e. without TWIST, which the reference does not have either
        ang = Orientation(xmap_indexed.rotations, pg).angle_with(Orientation(rot_ref.flatten(), pg), degrees=True)
        print(
            f"Misorientation to the reference: median {np.median(ang):.2f} deg, "
            f"p90 {np.percentile(ang, 90):.2f} deg, fraction below 2 deg {np.mean(ang < 2):.1%}"
        )
        if xmap.ndim == 2:
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(ang.reshape(xmap.shape), vmax=5)
            ax.axis("off")
            fig.colorbar(im, ax=ax, label="Misorientation [deg]")
            fig.tight_layout()
else:
    print("No REFERENCE_FILE given")
''')

md(r"""
## 11. Optional: HROSM, re-index every grain on a fine misorientation ball

`EBSD.hrosm()` (the fork's port of EMsoftOO's EMHROSM, Marquardt et al. 2017) takes the indexed map of section 7,
computes its kernel average misorientation (KAM), segments it into grains, and re-indexes every grain by dictionary
indexing against patterns simulated on a cubochoric grid within `HROSM_MAX_ANGLE` of the grain's reference
orientation. The result is a high angular resolution orientation similarity map (HROSM), orientations on a grid
much finer than the global search, and the KAM, grain labels and grain reference orientation deviation (GROD).
It needs the EMsoft master pattern `.h5` of the phase (`HROSM_MASTER_FILE`), since patterns are simulated.

Cost: every grain's ball of `(2 HROSM_N_STEPS + 1)^3` orientations is simulated and matched, so the time grows
with the cube of `HROSM_N_STEPS` and with the number of grains, not much with the grain size. Start with
`HROSM_N_STEPS = 10` and an `HROSM_ROI`; the fork's tutorial `doc/tutorials/hrosm.ipynb` discusses the parameters.
The orientations are re-indexed as indexed, i.e. without `TWIST`, which is applied to the result afterwards.
""")

code(r'''
hrosm_master = HROSM_MASTER_FILE is not None or demo
if hrosm_master:
    if demo:
        mp_hrosm = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
    else:
        kwargs = {"projection": "lambert", "hemisphere": "both"}
        if HROSM_ENERGY is not None:
            kwargs["energy"] = HROSM_ENERGY
        mp_hrosm = kp.load(HROSM_MASTER_FILE, **kwargs)
    print(mp_hrosm)
    ids = [i for i in xmap_indexed.phases_in_data.ids if i != -1]
    if len(ids) > 1 and mp_hrosm.phase.name not in [xmap_indexed.phases[i].name for i in ids]:
        raise ValueError(
            f"The map has the phases {[xmap_indexed.phases[i].name for i in ids]}; the master pattern's phase "
            f"{mp_hrosm.phase.name!r} must be one of them (rename it with mp_hrosm.phase.name = ...)"
        )
    if mp_hrosm.phase.point_group != xmap_indexed.phases[ids[0]].point_group:
        warnings.warn(
            f"Point group of the master pattern ({mp_hrosm.phase.point_group.name}) and of the indexed map "
            f"({xmap_indexed.phases[ids[0]].point_group.name}) differ; check HROSM_MASTER_FILE"
        )

    # Only the points of HROSM_ROI are assigned to grains and re-indexed
    nav_mask = None
    if HROSM_ROI is not None:
        c0, c1, r0, r1 = HROSM_ROI
        nav_mask = np.ones(nav_shape, dtype=bool)  # True = leave out
        nav_mask[r0:r1, c0:c1] = False
    signal_mask = ~kp.filters.Window("circular", det.shape).astype(bool) if CIRCULAR_MASK else None

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        xmap_hrosm_indexed = s.hrosm(
            xmap_indexed,
            mp_hrosm,
            det,
            energy=HROSM_ENERGY,
            threshold=HROSM_THRESHOLD,
            max_angle=HROSM_MAX_ANGLE,
            n_steps=HROSM_N_STEPS,
            keep_n=HROSM_KEEP_N,
            n_osm=HROSM_N_OSM,
            average=HROSM_AVERAGE,
            min_pixels=HROSM_MIN_PIXELS,
            dilate=HROSM_DILATE,
            signal_mask=signal_mask,
            navigation_mask=nav_mask,
        )
    dt = time.perf_counter() - t0
    for w in record:
        if issubclass(w.category, UserWarning):
            print("NB!", w.message)
    xmap_hrosm = apply_twist(xmap_hrosm_indexed)

    reindexed = xmap_hrosm.prop["reindexed"].astype(bool)
    n_grains = np.unique(xmap_hrosm.prop["grain_id"][reindexed]).size
    pg = xmap_indexed.phases[ids[0]].point_group
    moved = Orientation(xmap_indexed.rotations[reindexed], pg).angle_with(
        Orientation(xmap_hrosm_indexed.rotations[reindexed], pg), degrees=True
    )
    print(
        f"HROSM: {reindexed.sum()} of {reindexed.size} points re-indexed in {n_grains} grains, "
        f"{(2 * HROSM_N_STEPS + 1) ** 3} orientations per grain "
        f"(mean spacing {kp.indexing.misorientation_ball_spacing(HROSM_MAX_ANGLE, HROSM_N_STEPS):.3f} deg), {dt / 60:.1f} min"
    )
    print(
        f"Rotation from the spherical indexing result: median {np.median(moved):.3f} deg, p99 {np.percentile(moved, 99):.3f} deg; "
        f"median OSM of the re-indexed points {np.nanmedian(xmap_hrosm.prop['osm'][reindexed]):.1f} of {HROSM_N_OSM}"
    )
else:
    print("No HROSM_MASTER_FILE given, HROSM skipped")
''')

code(r'''
if hrosm_master and xmap_hrosm.ndim == 2:
    cmap = plt.get_cmap("viridis").with_extremes(bad="0.6")
    rng = np.random.default_rng(42)
    grain_id = xmap_hrosm.get_map_data("grain_id")
    grain_id = np.nan_to_num(grain_id, nan=0).astype(int)
    grain_colors = rng.uniform(0.2, 1, (grain_id.max() + 1, 3))
    grain_colors[0] = 0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("KAM [deg]", xmap_hrosm.get_map_data("kam"), dict(vmin=0, vmax=2, cmap=cmap)),
        ("Grains (black: no grain)", grain_id, dict(cmap=ListedColormap(grain_colors), interpolation="nearest")),
        ("GROD, spherical indexing [deg]", xmap_hrosm.get_map_data("grod"), dict(vmin=0, vmax=HROSM_MAX_ANGLE, cmap=cmap)),
        (f"HROSM (grey: not re-indexed), of {HROSM_N_OSM}", xmap_hrosm.get_map_data("osm"), dict(vmin=0, vmax=HROSM_N_OSM, cmap=cmap)),
    ]
    for ax, (title, values, kw) in zip(axes.ravel(), panels):
        im = ax.imshow(values, **kw)
        ax.set(title=title)
        ax.axis("off")
        if "Grains" not in title:
            fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    fig.tight_layout()

    # GROD of the re-indexed orientations about the same grain references: the gradients HROSM resolves
    grains = kp.indexing.average_grain_orientations(xmap_hrosm_indexed, grain_id)
    grod_hrosm = kp.indexing.grain_reference_orientation_deviation_map(xmap_hrosm_indexed, grain_id, grains)
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
    vmax = np.nanpercentile(grod_hrosm, 99) if np.isfinite(np.nanmax(grod_hrosm)) else 1
    for ax, values, title in zip(
        axes, [xmap_hrosm.get_map_data("grod"), grod_hrosm], ["GROD, spherical indexing", "GROD, re-indexed (HROSM)"]
    ):
        im = ax.imshow(values, vmin=0, vmax=vmax, cmap=cmap)
        ax.set(title=f"{title} [deg]")
        ax.axis("off")
        fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    fig.tight_layout()
''')

code(r'''
if hrosm_master:
    # out_dir and stem are those of section 9. The best match in the ball is the orientation of every re-indexed
    # point; "scores" holds the HROSM_KEEP_N best NCC scores per point, "scores_sph" the spherical indexing score.
    name = f"{stem}_hrosm_n{HROSM_N_STEPS}"
    prop = {k: v for k, v in xmap_hrosm.prop.items() if k != "simulation_indices"}
    prop["scores_sph"] = xmap_indexed.prop["scores"]
    prop["iq"] = xmap_indexed.prop["iq"]
    xmap_hrosm_out = CrystalMap(
        rotations=xmap_hrosm.rotations,
        phase_id=xmap_hrosm.phase_id,
        phase_list=xmap_hrosm.phases,
        prop=prop,
        scan_unit=xmap_hrosm.scan_unit,
        x=xmap_hrosm.x,
        y=xmap_hrosm.y,
    )
    if SAVE_H5:
        io.save(out_dir / f"{name}.h5", xmap_hrosm_out, overwrite=True)
        print("Saved", out_dir / f"{name}.h5")
    if SAVE_ANG:
        # The .ang takes one score per point: the best NCC of the ball, or the spherical score where not re-indexed
        best = np.where(reindexed, np.nan_to_num(xmap_hrosm.prop["scores"][:, 0]), xmap_indexed.prop["scores"])
        xmap_hrosm_out.prop["best_score"] = best
        io.save(
            out_dir / f"{name}.ang",
            xmap_hrosm_out,
            overwrite=True,
            image_quality_prop="iq",
            confidence_index_prop="best_score",
        )
        print("Saved", out_dir / f"{name}.ang")
''')

def read_settings(source):
    """NAME -> value (Python source) from a preset file of "NAME = value" lines, or from the settings cells
    (tagged "files" / "parameters") of a notebook built by this script or build_refinement_notebook.py."""
    source = Path(source)
    if source.suffix.lower() == ".ipynb":
        cells = nbf.read(source, as_version=4).cells
        text = "\n".join(
            c.source for c in cells if c.cell_type == "code" and {"parameters", "files"} & set(c.metadata.get("tags", []))
        )
    else:
        text = source.read_text(encoding="utf-8")
    settings = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*) = (.*?)(\s+#.*)?$", line)
        if match:
            settings[match.group(1)] = match.group(2)
    return settings


def apply_settings(settings):
    r"""Overwrite the settings of the notebook under construction with those given; names it does not have are listed.

    A value of None does not replace a path placeholder (r"C:\path\to\..."), since the placeholder says what
    goes there and None would not.
    """
    cells = [c for c in C if {"parameters", "files"} & set(c.metadata.get("tags", []))]
    ignored = []
    for name, value in settings.items():
        pattern = re.compile(rf"^{name} = (.*?)(\s+#.*)?$", flags=re.MULTILINE)
        cell = next((c for c in cells if pattern.search(c.source)), None)
        if cell is None:
            ignored.append(name)
            continue
        if value == "None" and pattern.search(cell.source).group(1).startswith(r'r"C:\path\to'):
            continue
        cell.source = pattern.sub(lambda m: f"{name} = {value}" + (m.group(2) or ""), cell.source, count=1)
    if ignored:
        print("ignored (not settings of this notebook):", ", ".join(ignored))


# Optional: settings from preset files ("NAME = value" lines) and/or notebooks, applied in the order given
for source in sys.argv[2:]:
    apply_settings(read_settings(source))

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1])
