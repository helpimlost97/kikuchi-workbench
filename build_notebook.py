"""Build spherical_indexing_template.ipynb"""
import sys

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
MASTER_BANDWIDTH = 188     # only for .h5 masters: bandwidth of the harmonics computed from the master pattern
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

# ======================= REFINEMENT (section 11) =======================
# Refinement matches every pattern to patterns simulated from the master pattern, with a projection center
# per map point. It starts from the spherical indexing result of section 7 and runs on the CPU.
MASTER_PATTERN_FILE = None   # EMsoft master pattern .h5 of the phase (refinement cannot use an .sht file)
ENERGY = 20                  # beam energy in keV of the master pattern to use
REFINE_BINNING = 2           # patterns are binned by this factor for refinement. 1 = full size (slowest, most detail)
PC_GRID = (5, 5)             # (rows, cols) of map blocks; the best pattern of each block is used to calibrate the PC
PC_TRUST_REGION = 0.05       # calibration: how far each PC value may move from PC
REFINE_TRUST_REGION = 2.0    # how far each Euler angle may move from the spherical indexing result, in degrees
REFINE_RTOL = 1e-5           # stop refining a pattern when the score changes by less than this (relative)
REFINE_MAXEVAL = 300         # ... or after this many simulated patterns
REFINE_ROI = None            # (col_start, col_stop, row_start, row_stop) within the indexed map. None = everything

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

md("## 9. Save")

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
## 11. Refinement with a projection center per map point

Approach taken from https://kikuchipy.org/en/stable/tutorials/hybrid_indexing.html.

The flow is that of the tutorial, with spherical indexing in the place of Hough indexing and without the
dictionary indexing:

1. Calibration: orientation and projection center (PC) of a few good patterns spread over the map are refined
   together, and a plane is fitted to their PCs. That gives a PC for every map point.
2. The orientations of a test region are refined with those PCs, which also gives the time for the full map.
3. The orientations of the full map (or of `REFINE_ROI`) are refined.

Everything here starts from `xmap_indexed` of section 7, i.e. the orientations without `TWIST`, since they
belong to the detector without twist. `TWIST` is applied to the refined map at the end.
""")

md(r"""
### 11.1 Master pattern and patterns for refinement
""")

code(r'''
ids = [i for i in xmap_indexed.phases_in_data.ids if i != -1]
if len(ids) != 1:
    raise ValueError("Section 11 is written for a map with one phase")
phase = xmap_indexed.phases[ids[0]]

mp = kp.load(MASTER_PATTERN_FILE, projection="lambert", hemisphere="both", energy=ENERGY)
mp.phase = phase.deepcopy()  # refinement requires the phase of the map and of the master pattern to be equal
print(mp)

# Patterns and detector for refinement: cropped to a multiple of the binning factor, then binned
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
print(f"Patterns for refinement: {det_ref.shape} px (binning {REFINE_BINNING}), lazy: {s_ref._lazy}")
det_ref
''')

md(r"""
### 11.2 Calibration: a projection center per map point
""")

code(r'''
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


# The best pattern of each block of the map: the highest score among the points inside a grain,
# i.e. with all four neighbours within 1 degree
scores_indexed = xmap_indexed.get_map_data("scores")
right, below = neighbour_misorientation(xmap_indexed)
in_grain = (right < 1) & (below < 1)
in_grain[:, 1:] &= right[:, :-1] < 1
in_grain[1:] &= below[:-1] < 1

n_rows, n_cols = nav_shape
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

# The calibration patterns as a signal of their own, with their orientations from spherical indexing
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

fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(scores_indexed)
ax.scatter(cal_rc[:, 1], cal_rc[:, 0], c="r", marker="+", s=120)
for i, (r, c) in enumerate(cal_rc):
    ax.text(c + 3, r - 3, i, color="w")
ax.set(title="Calibration patterns on the score map")
ax.axis("off")
fig.tight_layout()
''')

code(r'''
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
''')

code(r'''
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
det_plane.plot_pc()
''')

md(r"""
Check before going on: the refined PCs of the calibration patterns should lie close to a plane, and the
measured change per scan step should be close to the expected one. If the PCs scatter by more than they
change over the map, use more calibration patterns (`PC_GRID`) or less binning (`REFINE_BINNING`). To go on
with one PC for the whole map instead, run `det_plane = det_ref.deepcopy()` in a new cell.

### 11.3 Test region
""")

code(r'''
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

    Returns the refined map and the map from spherical indexing of that region, both without TWIST.
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


def plot_refinement(xmap_ref, xmap_before, title=""):
    """Neighbour misorientation before and after refinement, and how far refinement rotated the orientations."""
    before = np.nanmean(neighbour_misorientation(xmap_before), axis=0)
    after = np.nanmean(neighbour_misorientation(xmap_ref), axis=0)
    rotated = xmap_before.orientations.angle_with(xmap_ref.orientations, degrees=True).reshape(xmap_ref.shape)
    for name, values in (("spherical indexing", before), ("refined", after)):
        inside = values[values < 1]
        print(f"Neighbour misorientation inside grains, {name}: median {np.median(inside):.3f} deg, mean {inside.mean():.3f} deg")
    print(f"Rotated by refinement: median {np.median(rotated):.3f} deg, p99 {np.percentile(rotated, 99):.3f} deg")
    print(f"Score: mean {xmap_ref.scores.mean():.3f}; simulated patterns per point: mean {xmap_ref.num_evals.mean():.0f}, max {xmap_ref.num_evals.max()}")

    fig, axes = plt.subplots(ncols=4, figsize=(20, 4.5))
    for ax, values, name in zip(axes, (before, after, rotated), ("Neighbours, spherical indexing", "Neighbours, refined", "Rotated by refinement")):
        im = ax.imshow(values, vmin=0, vmax=0.5)
        ax.set(title=name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8, label="deg")
    bins = np.linspace(0, 1, 101)
    axes[3].hist(before.ravel(), bins=bins, histtype="step", label="spherical indexing")
    axes[3].hist(after.ravel(), bins=bins, histtype="step", label="refined")
    axes[3].set(xlabel="Neighbour misorientation [deg]", ylabel="Map points")
    axes[3].legend()
    fig.suptitle(title)
    fig.tight_layout()


if REFINE_ROI is None:
    refine_roi = (0, n_cols, 0, n_rows)
else:
    refine_roi = tuple(REFINE_ROI)
n_refine = (refine_roi[1] - refine_roi[0]) * (refine_roi[3] - refine_roi[2])

# The same region as in section 6
r0 = max(0, n_rows // 2 - TEST_SIZE // 2)
c0 = max(0, n_cols // 2 - TEST_SIZE // 2)
test_region = (c0, min(n_cols, c0 + TEST_SIZE), r0, min(n_rows, r0 + TEST_SIZE))

_ = refine_region((c0, c0 + 4, r0, r0 + 1))  # warm-up so that compilation is not part of the timing
t0 = time.perf_counter()
xmap_test_ref, xmap_test_indexed = refine_region(test_region)
dt = time.perf_counter() - t0
speed = xmap_test_ref.size / dt
print(f"Test region: {xmap_test_ref.size} patterns in {dt:.1f} s = {speed:.1f} patterns/s ({det_ref.shape} px)")
print(f"Estimated time for the {n_refine} patterns to refine: {n_refine / speed / 60:.0f} min")
plot_refinement(xmap_test_ref, xmap_test_indexed, "Test region")
''')

md(r"""
### 11.4 Refine the map
""")

code(r'''
t0 = time.perf_counter()
xmap_refined_indexed, xmap_before = refine_region(refine_roi)  # without the twist
dt = time.perf_counter() - t0
print(f"{xmap_refined_indexed.size} patterns in {dt / 60:.1f} min = {xmap_refined_indexed.size / dt:.1f} patterns/s")

# The refined points as a map of their own, with the image quality and the score of spherical indexing.
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
xmap_refined = apply_twist(xmap_refined_indexed)
if TWIST:
    print(f"Orientations rotated by {TWIST} deg about the sample normal (TWIST)")
xmap_refined
''')

code(r'''
plot_refinement(xmap_refined_indexed, xmap_before, "Refined map")
plot_maps(xmap_refined, "Refined map")
''')

md(r"""
### 11.5 Save
""")

code(r'''
# out_dir and stem are those of section 9
name = f"{stem}_refined_bin{REFINE_BINNING}"
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
''')

# Optional: preset the settings cell from a file of "NAME = value" lines (sys.argv[2])
if len(sys.argv) > 2:
    import re

    cells = [c for c in C if {"parameters", "files"} & set(c.metadata.get("tags", []))]
    for line in open(sys.argv[2], encoding="utf-8").read().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        pattern = re.compile(rf"^{name} = .*?(\s+#.*)?$", flags=re.MULTILINE)
        cell = next((c for c in cells if pattern.search(c.source)), None)
        if cell is None:
            raise ValueError(f"no setting named {name}")
        cell.source = pattern.sub(lambda m: f"{name} = {value}" + (m.group(1) or ""), cell.source, count=1)

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1])
