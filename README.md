# GPU-accelerated spherical indexing of EBSD / RKD patterns

Set-up guide for running Johan Westraadt's Python port of **EMSphInx** (spherical harmonic
EBSD indexing, Lenthe et al. 2019) with its **CuPy GPU backend**, on any Windows or Linux
machine with an NVIDIA GPU. The port lives in his fork of kikuchipy; nothing here is in
upstream kikuchipy or on PyPI, so everything is installed from a git clone.

Tested on: Windows 11, Python 3.14.0, RTX 5070 Laptop GPU (8 GB VRAM), driver with CUDA 13
support, Core Ultra 9 275HX (8 performance cores), 32 GB RAM. Linux works the same way;
only the path syntax in the examples changes.

---

## 1. What is in this folder

| File / folder | Purpose |
|---|---|
| `kikuchipy/` | (Not in the repository; `update_fork.py` creates it in step 3.) Clone of https://github.com/jwestraadt/kikuchipy, branch `develop`, pinned at commit `4c7a0035`, plus the local patches below. Installed in editable mode. |
| `update_fork.py` | Clones or updates that clone, checks out the pinned commit, applies the patch and installs it, in one command. Also undoes an earlier patch first, so re-running it after Johan pushes new commits is safe. See step 3. |
| `kikuchipy_local_patches.patch` | The uncommitted local changes to the fork (GPU memory release, fast lazy loading of `.up1` files). Applied by `update_fork.py`, or by hand, see step 3. |
| `kikuchi_workbench.py`, `live_view.py`, `sph_core.py`, `Open workbench.bat` | The GUI: the Kikuchi Pattern Processor and spherical indexing in one window, see section 6.8. `sph_core.py` holds the indexing code the GUI calls (the notebook's cells as functions, usable from scripts), `live_view.py` the animated progress view. |
| `montage.py`, `montage_tab.py`, `STITCH_DATASET_NOTES.md`, `make_montage_sample.py`, `example_data/montage_sample/` | Batch processing and indexing of an EDAX APEX montage (a scan collected as tiles, to be stitched in OIM Analysis or MTEX afterwards), see section 6.9. `montage.py` reads the `.edaxh5` project, links each tile to its `.up1` by the GUID in the file header and writes results into a copy of the project; `montage_tab.py` is the GUI's third tab. The notes describe the file formats; the sample is a real project cropped to 3 x 40 points per tile by `make_montage_sample.py`. |
| `build_notebook.py` | Generates the indexing notebook (sections 0 to 10 plus an optional HROSM section 11). Edit this, then regenerate; do not hand-edit the generated `.ipynb` unless you know you will not regenerate it. |
| `build_refinement_notebook.py` | Generates the projection center drift correction notebook: corrects an indexed map for the PC moving with the beam, from the scan geometry alone, no patterns needed (minutes), see section 6.6. |
| `spherical_indexing_template.ipynb` | Generated indexing notebook with default settings. Runs on kikuchipy's nickel demo data when no pattern file is given. |
| `spherical_refinement_template.ipynb` | Generated PC drift correction notebook for conventional EBSD. Starts from the `.h5` the indexing notebook writes; also runs on the demo data. |
| `spherical_indexing_RKD.ipynb`, `spherical_refinement_RKD.ipynb`, `preset_RKD.txt` | Example of the two notebooks generated with a preset file (an RKD data set, detector parallel to the sample). The paths in all three are placeholders, see section 6.1. |
| `ebsd_params.py` | Reads acquisition parameters (pattern size, projection center, map size, step) from an EDAX `.ang` or `.edaxh5` file and a `.up1`/`.up2` file. Used by the notebook's first cell and usable from the command line. |
| `sph_benchmark.py` | CPU vs GPU speed and parity benchmark on the nickel demo data set. Use it to verify the install. |
| `test_gpu_memory_fix.py` | Checks that VRAM is released between two GPU runs (tests one of the local patches). |
| `osc_refinement_handoff/` | Full pattern-matching refinement with a PC per map point (hours to days, CPU) as a stand-alone script for a cluster, with its own hand-off notes. |
| `Open ... notebook.bat` | Windows shortcuts that start JupyterLab on a notebook from this folder. They use the `python` on PATH; set the `PYTHON` variable at the top to a full interpreter path if your environment is elsewhere. |

---

## 2. Requirements

- **GPU:** an NVIDIA card with a current driver. The driver decides which CuPy wheel you need
  (step 4). About 6 GB of free VRAM is comfortable; the code auto-sizes its batches to the
  free memory and halves them on out-of-memory, so smaller cards work, just slower.
- **CPU and RAM:** preprocessing, back-projection and the Newton refinement run on the CPU
  through Dask threads, so the core count still matters in GPU mode. Pattern files larger
  than about a quarter of RAM are loaded lazily (memory mapped), so RAM does not need to hold
  the whole file.
- **Python:** 3.10 to 3.14. Use a fresh virtual environment or conda environment; the fork
  pulls in hyperspy, orix, dask, numba and friends, and mixing it with another kikuchipy
  install in the same environment will not work.
- **git**, and on Windows a terminal (PowerShell or cmd).
- No CUDA toolkit install is needed; the CuPy wheel brings its runtime libraries through pip.

---

## 3. Install the kikuchipy fork

Open a terminal in this folder (quote paths that contain spaces) and activate your environment
first. Then

```bash
python update_fork.py                     # clones ./kikuchipy, checks out 4c7a0035, applies the patch, pip installs
python update_fork.py --dir D:\path\to\kikuchipy   # if the clone lives elsewhere
```

That is the whole of this step; the manual equivalent is

```bash
git clone https://github.com/jwestraadt/kikuchipy.git
cd kikuchipy
git checkout 4c7a0035         # the commit everything here was tested with
git apply ../kikuchipy_local_patches.patch
git status --short            # should list 4 modified files
pip install -e .
```

`4c7a0035` is the head of `develop` on 2026-10-18. Compared with the earlier pin `71a1b2f` it adds
NLPAR (`EBSD.average_non_local_neighbour_patterns`; not used here, the Kikuchi Pattern Processor
does NLPAR through pyebsdindex), HROSM (`EBSD.hrosm`, section 6.7) and a HyperSpy 2.5 fix. The
spherical indexing code and its `spherical_indexing()` signature are unchanged, the patch applies
cleanly, and the fork's spherical indexing tests (326, with the patch) and the whole template
notebook on the demo data were re-run against it. To try a newer commit later:

```bash
python update_fork.py --show-latest       # lists the commits on origin/develop after the pin
python update_fork.py --commit develop    # checks it out and re-applies the patch (stops if the patch no longer fits)
```

When moving the pin, change `PINNED_COMMIT` in `update_fork.py` and this README together. If
you already have a clone with the patch applied by hand, `update_fork.py` recognises it (it
reverts exactly that patch before checking out) and refuses to touch any other local change.

What the patch does:

- `io/plugins/edax_binary/_api.py`: lazy loading of large EDAX `.up1`/`.up2` files. Without
  it, opening a 30 GB file lazily takes minutes because Dask hashes and copies the whole
  memory map. With it, the same file opens in a few seconds. Measured.
- `indexing/_spherical/_gpu.py` and `_indexer.py`: clear CuPy's per-thread cuFFT plan caches
  when a GPU run ends, so VRAM is released and the next run gets the same automatic batch
  size. Written to fix a case where about 7 GB stayed allocated after a run. Not yet verified
  on a GPU; `test_gpu_memory_fix.py` is the check.
- `tests/test_indexing/test_spherical_gpu.py`: a no-op to keep the author's test suite in
  step with the change above.

`update_fork.py` has already run `pip install -e kikuchipy`. Add what the notebook and scripts
need beyond the core package:

```bash
pip install pooch nlopt jupyterlab nbformat psutil ipywidgets
```

In that line, `pooch`
downloads the nickel demo data, `nlopt` is used by the refinement stage, `jupyterlab` runs the
notebook, `nbformat` builds it, `psutil` decides whether to load a file lazily. Alternatively
`pip install -e ".[all]"` installs the author's full extra set (adds pyebsdindex, pyvista and
ebsdsim); use that if you want them and nothing else in the environment pins pyebsdindex.

---

## 4. Install CuPy

Find out which CUDA major version your driver supports:

```bash
nvidia-smi
```

The top right of the table says `CUDA Version: 13.x` or `12.x`. (If `nvidia-smi` refuses to
run for permission reasons on Windows, the NVIDIA Control Panel's System Information shows the
driver version; drivers from 580 onwards support CUDA 13.) Then install the matching wheel:

```bash
pip install cupy-cuda13x      # driver supports CUDA 13
# or
pip install cupy-cuda12x      # driver supports CUDA 12 only
```

CuPy 14 is required on Python 3.14; older Python versions can take CuPy 13 as well. If the
first GPU run fails with a missing cuBLAS or cuFFT library, install the runtime libraries
through pip as well (`pip install nvidia-cublas nvidia-cufft` for CUDA 13, or
`pip install nvidia-cublas-cu12 nvidia-cufft-cu12` for CUDA 12).

---

## 5. Verify the install

```bash
python -c "import kikuchipy as kp, cupy; print(kp.__version__, kp.__file__); print(cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())"
```

Expected: version `0.14.dev0`, a file path inside your clone, and your GPU's name. If CuPy
raises `cudaErrorNoDevice` or `cudaErrorInsufficientDriver`, the driver is too old for the
wheel you picked; `backend="gpu"` will refuse to run rather than silently use the CPU.

Then run the benchmark on the nickel demo data set (4125 patterns of 60 x 60 px, downloaded
once, a few MB):

```bash
cd ..                          # back to this folder
python sph_benchmark.py --backend both --workers 8 --compare
```

`--workers` should be about your number of performance cores. The first call of each backend
includes about 10 s of Numba compilation, which the script excludes by a warm-up run.
Reference numbers from the tested laptop, bandwidth 68, refinement on:

| Backend | Throughput |
|---|---|
| CPU, 8 Dask workers | about 230 patterns/s |
| GPU, RTX 5070 Laptop | about 1200 patterns/s |

`--compare` must report identical winners and a median CPU vs GPU misorientation of 0 deg (the
author's design is bitwise-identical image quality and exactly matching refined orientations).

Optional, to test the VRAM-release patch:

```bash
python test_gpu_memory_fix.py
```

It prints `RESULT: PASS` when the free VRAM after two runs is back to the starting value, the
automatic batch size is the same in both runs, and both runs give the same orientations.

---

## 6. Index your own data with the notebook

### 6.1 Placeholder paths

Nothing in this folder is tied to a particular machine. Every setting that needs a file of
yours is written as a placeholder such as `r"C:\path\to\scan.up1"` and, where it is not
obvious, carries a comment that says what goes there:

```python
PATTERN_FILE = r"C:\path\to\scan.up1"  # <-- your pattern file
```

Replace the placeholder with the real path and keep the `r"..."` raw-string form (on Linux
`r"/path/to/scan.up1"` works the same way). Lines marked `# <--` are the ones to set; the
other settings have working defaults.

The notebook's first cell also has `PROJECT_DIR`, the folder that contains `ebsd_params.py`.
Leave it `None` when you start JupyterLab from this folder (the `.bat` files do), and set it
to this folder's path only when you start Jupyter from somewhere else.

### 6.2 Build a notebook

```bash
python build_notebook.py my_scan.ipynb                # default settings
python build_notebook.py my_scan.ipynb my_preset.txt  # with a preset
```

A preset is a text file of `NAME = value` lines that overwrite settings in the notebook's
first two cells. `value` is Python source, so paths are written as raw strings. Names that
are not settings of the notebook being built are listed and ignored, so one preset can serve
both the indexing and the refinement notebook. A notebook built earlier can be given in the
place of a preset; its settings are then copied. `preset_RKD.txt` is a complete example; a
minimal one for conventional EBSD looks like

```
PATTERN_FILE = r"C:\data\scan.up1"
SCAN_FILE = r"C:\data\scan.ang"
MASTER_FILE = r"C:\data\Ni 20kV.sht"
CAMERA_TILT = 8.0
BANDWIDTH = 123
OUTPUT_DIR = r"C:\data\results"
```

### 6.3 Start JupyterLab

```bash
python -m jupyter lab my_scan.ipynb
```

(On Windows the two `.bat` files do exactly this for the two example notebooks. They use the
`python` found on PATH; if your environment's interpreter is elsewhere, put its full path in
the `set PYTHON=` line at the top of the file.)

### 6.4 Walk through the notebook

The notebook follows the author's tutorial, `kikuchipy/doc/tutorials/spherical_indexing.ipynb`.

1. **Section 0, files.** Set `PATTERN_FILE` and `SCAN_FILE` (the `.ang` or `.edaxh5` written at
   acquisition, not an indexing result) and run the cell. It prints pattern size, projection
   center, map size and step size. With `PATTERN_FILE = None` the notebook uses the nickel
   demo data, which is the easiest way to confirm that everything runs.
2. **Section 1, settings.** The ones that matter most:
   - `PC` and `PC_CONVENTION`: the projection center. Values from an `.ang` file are in the
     `"edax"` convention, which is the default.
   - `CAMERA_TILT`: detector tilt / camera elevation in degrees (EMSphInx `thetac`). Typically
     8 for a conventional EBSD camera, -90 for RKD (detector parallel to the sample). The fork
     has no -60 to 60 deg limit on this, unlike the EMSphInx binary.
   - `SAMPLE_TILT = None` takes the tilt the master pattern was simulated for, as EMSphInx
     does. Set `OVERRIDE_MASTER_SAMPLE_TILT = True` only if you deliberately want another value.
   - `MASTER_FILE`: an EMSphInx `.sht` file or an EMsoft master pattern `.h5`. A list of files
     indexes several phases.
   - `REMOVE_STATIC_BACKGROUND` / `REMOVE_DYNAMIC_BACKGROUND`: set both to `False` for pattern
     files that are already background corrected or NLPAR processed.
   - `BANDWIDTH`: 68 is the EMSphInx default, 88 or 123 for more accuracy at lower speed.
     `kp.indexing.fast_bandwidths()` lists the efficient values.
   - `BACKEND = "gpu"`, `N_WORKERS` = number of performance cores, `CHUNKSIZE = None`
     (automatic GPU batch size; set a number if a run reports out of memory).
   - `TWIST`: rotation of the detector about its normal. The indexer cannot use a twisted
     detector, so the notebook rotates the indexed orientations about the sample normal
     afterwards. Flip the sign if the IPF-X and IPF-Y maps come out rotated the wrong way.
3. **Sections 2 to 6.** Run them in order. Section 6 indexes a small test region of
   `TEST_SIZE` x `TEST_SIZE` points so you can check the geometry before committing to the
   full map. If the test maps look like noise, try `FLIP_PATTERNS_UD = True` or check the
   projection center and camera tilt.
4. **Section 7** indexes the full map (or `ROI`), **8** plots it, **9** saves `.ang` and `.h5`
   files to `OUTPUT_DIR`, **10** compares against `REFERENCE_FILE` if you have a result for the
   same map from EMSphInx or OIM.
5. The `.h5` written in section 9 is the input of the PC drift correction notebook, section 6.6.
   Run that on every map wider than a few hundred micrometres.
6. **Section 11, HROSM** (optional, needs an EMsoft master pattern `.h5`): re-indexes every grain on
   a fine misorientation ball, section 6.7.

### 6.5 Reading scan parameters without the notebook

```bash
python ebsd_params.py "scan.ang" --up1 "scan.up1"            # EDAX and EMsoft pattern centers
python ebsd_params.py "scan.edaxh5" --map "Area 1" --px 115
```

### 6.6 Correct the projection center drift

Spherical indexing (like EMSphInx) uses one projection center (PC) for the whole map, but the
true PC moves with the beam by one step length per scan step. Indexed with a fixed PC, that
shift is read as a rotation of about *beam travel / detector distance*: for a 3 mm wide map at
a detector distance of 18 mm it is 10 degrees edge to edge, and about 0.1 degrees across a 30 um
grain. Point-to-point noise is unaffected, but grain reference orientation deviations and
orientation gradients within grains are.

`spherical_refinement_template.ipynb` (built by `build_refinement_notebook.py`) removes that
error from an indexed map without re-indexing and without the pattern file, in minutes:

1. The PC of every map point is extrapolated from one anchor PC and the scan geometry (the
   model of `EBSDDetector.extrapolate_pc()`, generalised to a twisted detector).
2. The rotation a PC offset imposes on an indexed orientation is linearised once for the
   detector geometry, and every orientation is rotated back by its local correction.

Checked against patterns simulated with the true PC and refined with the anchor PC: a fixed-PC
error of 1.3 / 3.4 / 4.7 degrees becomes 0.03 / 0.1 / 0.2 degrees. Gradients within grains are
right to second order everywhere. Build it with the settings of the indexing notebook of the
same scan, so that the geometry matches:

```bash
python build_refinement_notebook.py my_scan_pccorr.ipynb my_scan.ipynb
python build_refinement_notebook.py my_scan_pccorr.ipynb my_scan.ipynb my_preset.txt   # plus a preset
```

1. **Section 0, files.** `INDEXED_FILE`, the `.h5` the indexing notebook wrote (its `.ang`
   works too, but carries no image quality and only the proper point group; an EMSphInx or OIM
   result of the same points works as well), and `SCAN_FILE` for the PC and pattern size.
2. **Section 1, settings.** The detector geometry the map was indexed with: `PC` (the anchor),
   `PC_POSITION` (the map point that PC belongs to, default the centre), `PATTERN_SHAPE` (needed
   when the scan file does not give it), `SAMPLE_TILT`, `CAMERA_TILT`, the detector size via
   `PX_SIZE` or `DETECTOR_WIDTH_UM` (the drift scales with 1 / detector size, so check the
   printed width and distance), `TWIST` and `CIRCULAR_MASK`.
3. **Section 4** shows where in the map the single PC fitted best (indexing score over image
   quality per band), which is where `PC_POSITION` should point. **Section 5** prints the PC
   change over the map and checks the plane against `extrapolate_pc`. **Section 6** prints the
   apparent rotation per millimetre of beam travel.
4. **Section 7** measures on the map itself how much of the geometric drift is really there:
   the mean rotation between points k steps apart within grains, which real gradients average
   out of and the artefact does not. Expect about 1 for a beam-scanned map indexed with one PC
   and about 0 for a map acquired by stage movement (the source then never moves relative to
   the detector) or already indexed with a PC per point. The correction is scaled by the
   measured fraction (`DRIFT_FACTOR = "measured"`, or give a number). **Section 8** corrects
   the orientations and shows the correction angle over the map; the neighbour misorientation
   inside grains must not change. **Section 9** (optional, `SELF_CHECK = True`, a minute)
   verifies the correction on simulated patterns for your geometry. **Section 10** saves
   `<name>_pccorr.h5` / `.ang` with the correction angle as an extra property.

For a twisted detector (RKD, `TWIST != 0`) the drift direction follows kikuchipy's detector
twist, whose sign equivalence with the indexing notebook's `TWIST` is not verified. The full
pattern-matching refinement with a PC per point, which also removes the band limit, remains
available as a cluster script in `osc_refinement_handoff/`; it costs hours to days.

The docstring at the top of `ebsd_params.py` lists all options.

### 6.7 HROSM: re-index every grain on a fine misorientation ball

`EBSD.hrosm()` (new in the fork since `71a1b2f`; Johan's port of EMsoftOO's EMHROSM, Marquardt
et al. 2017) takes an indexed map, computes its kernel average misorientation (KAM), segments it
into grains, and re-indexes every grain by dictionary indexing against patterns simulated on a
cubochoric grid within `HROSM_MAX_ANGLE` of the grain's reference orientation. The result is a high
angular resolution orientation similarity map (HROSM), orientations on a grid much finer than the
global search, and the KAM, grain labels and grain reference orientation deviation (GROD). Section
11 of the indexing notebook runs it on the map of section 7 and saves `<name>_hrosm_n<steps>.h5`
and `.ang`. (The GUI does not do HROSM; it is pattern processing and spherical indexing only.)

- `HROSM_MASTER_FILE`: the EMsoft master pattern `.h5` of the phase. Patterns are simulated, so an
  `.sht` file cannot be used. `None` skips the section (the demo data uses its own master).
- `HROSM_MAX_ANGLE` (5 deg) must be at least the largest GROD of the grains; the run warns and
  names the radius that covers every grain otherwise. `HROSM_N_STEPS` (10) gives `(2 n + 1)^3`
  orientations per grain, 9261 for 10 and 68921 for 20 (EMHROSM's default); the cost grows with the
  cube of it and with the number of grains, not much with the grain size. Start with 10 and an
  `HROSM_ROI`. On the nickel demo (4125 patterns, 31 grains) `HROSM_N_STEPS = 4` took 13 s on 3
  CPU workers; `n_steps=10` on the same map is about 20 s on a 20-core workstation according to
  the fork's tutorial (`kikuchipy/doc/tutorials/hrosm.ipynb`, which also discusses `threshold`,
  `average` and `dilate`).
- HROSM runs on the orientations as indexed (without `TWIST`); the twist is applied to the result.
  The whole computation is on the CPU (Dask threads, `N_WORKERS`); the GPU backend only concerns
  the spherical search.

### 6.8 The GUI: pattern processor and spherical indexing in one window

`kikuchi_workbench.py` puts the [Kikuchi Pattern Processor](https://github.com/helpimlost97/kikuchi-pattern-processor)
and the indexing of this folder into one PyQt5 window with three tabs:

1. **Pattern processor**: the unchanged `gpu_pattern_viewer.py`, imported from its own folder
   (NLPAR through pyebsdindex, aperture, IQ range, background, stretch, CLAHE, `Apply && Save`).
2. **Spherical indexing**: the settings of the notebook as a form on the left (drag the divider to
   resize it), the live view, the result maps and a log on the right.
3. **Montage batch**: the two steps above for every tile of an EDAX APEX montage, see 6.9.

Workflow in the indexing tab:

- **One data set at a time**: loading a file in either tab unloads the other tab's data set
  (the processor's scan and NLPAR result, or the indexer's patterns, map and last result), so two
  scans never sit in memory together. After `Apply && Save`, loading the processed file in the
  indexing tab therefore empties the processor tab; the saved file and its IQ exclusion are kept.
- **Pattern source**: always a file. **Load processed patterns** is greyed out until the
  processor's `Apply && Save` has written a file; it then puts that file into the pattern file box
  and loads it, and the patterns the processor excluded by its IQ range are not indexed (grey in
  the maps). The browse button takes any other file kikuchipy reads, e.g. a scan that was not
  processed. Big files are memory mapped and read chunk by chunk, so RAM is not a concern.
- **Load patterns (quasi-IQ map)**: runs by itself after browsing for a pattern file (and when the
  source is switched to the processor, whose IQ map is reused). It loads the patterns, computes a
  quasi image quality map (the standard deviation of every raw pattern, the processor's definition),
  stretches it between its 1st and 99th percentile and shows it in the **Live** view. For a memory
  mapped file this is one read of the file.
- **Read scan parameters** fills the projection center from the `.ang`; the scan file dialog opens in
  the pattern file's folder. The **pixel size** is filled in automatically as 30 mm / pattern width
  for the EDAX/TSL convention (the EDAX detector is 30 mm wide) and stays editable; binning is
  always 1 here (it only scales that pixel size). Static and
  dynamic background removal are off by default (the processor already did them) and the circular
  mask is on.
- **Click anywhere on the map** (before indexing) to see that pattern on the right exactly as the
  indexer will see it: raw when nothing is ticked, otherwise with the static and dynamic background
  removal, the Gaussian background and the adaptive histogram equalization (`N_REGIONS`) that are
  ticked. Changing one of those boxes refreshes the shown pattern. During a run the same panel
  cycles through the band being indexed, with the same processing applied.
- **Check geometry** shows one pattern on the sphere; **Index test region** and **Index full map**
  index with the live view on by default: the map is indexed band by band (bands of whole rows
  sized to about 8 s each; identical results to one call, at a few per cent of extra time for the
  per-band setup), and the Live view paints each finished
  band with its IPF-Z colours over the grey quasi-IQ map, tints the band being indexed with a
  left-to-right sweep, shows a pattern of that band on the right (cycling through the band), and
  reports elapsed time, ETA, the speed over the last few seconds and the speed of the current band.
  The **Results** tab then shows the score, IQ, IPF-Z and IPF-X maps. Untick *Live view* for a
  single call without animation.
- **Save .ang / .h5**, **Compare with reference**, and **Load/Save preset**. A GUI preset holds the
  indexing parameters only: master bandwidth, PC convention, sample and camera tilt, pixel size,
  twist, the two background removals, bandwidth, workers, refinement, AHE tiles, circular mask,
  Gaussian background, normalization and chunk size. Files, folders and the projection center
  values are never written to or taken from a preset. The file format is the one
  `build_notebook.py` accepts, so a GUI preset also builds a notebook, and a notebook preset such
  as `preset_RKD.txt` loads into the GUI with only those parameters applied.

```bash
python kikuchi_workbench.py                                   # processor found next to this folder
python kikuchi_workbench.py --processor-dir C:\path\to\kikuchi-pattern-processor
python kikuchi_workbench.py --preset preset_RKD.txt
```

On Windows `Open workbench.bat` does the same (set `PYTHON` and, if needed, `KPP_DIR` at its top).
The processor's clone is looked for in `--processor-dir`, `KPP_DIR`, then as `kikuchi-pattern-processor`
next to this folder, inside it, in the current folder or in the home folder. Without it the indexing
tab still works from files. The environment is the one of this README (fork, CuPy) plus the
processor's packages: `pip install pyqt5 pyebsdindex scipy scikit-image` (and `pyopencl` for NLPAR
on the GPU). The processor's own switches (`KPP_GPU=0`, `KPP_THREADS`, `KPP_IN_MEMORY`, ...) apply
inside the workbench too.

Memory: with the background removals off (the default), the loaded patterns are indexed as they
are, memory mapped when the file is bigger than a quarter of RAM; with a removal on, the file is
read again and processed. Long steps run in a worker thread with their console output in the log
pane.

Verified headlessly on the nickel demo data written as a `.up1` (no GPU in that environment, CPU
backend): processor load and `Apply && Save`, `Load processed patterns` with the IQ exclusion
carried over, quasi-IQ map, band-wise test region and full map with the live events, save. The
band-wise result is checked to be identical to a single call.

### 6.9 Montage batch: process and index every tile of an APEX montage

An APEX montage ("stitch") scan is one `.edaxh5` project holding every tile as a map (`OIM Map
1_r000c000`, `..._r000c001`, ...) with its grid, step, pattern center, phases and the Hough results,
plus one `.up1`/`.up2` per tile. The version-4 `.up1` header carries the tile's map GUID, which is
how the files are linked (the project never names the pattern files); `STITCH_DATASET_NOTES.md`
describes both formats. Nothing is stitched here: every tile is processed and indexed on its own
grid, and the stitching is done afterwards in OIM Analysis or MTEX from the output project.

The **3. Montage batch** tab:

- **Project**: browse for the `.edaxh5` (or type the path and *Load project*). The table lists every
  map of the project with its grid, the raw pattern file found for it in the *raw pattern folder*
  (default: the project's folder) and the processed file found in the *output folder* (default:
  `processed` next to the project). Tiles without a pattern file are listed and skipped, never an
  error. A file is matched by GUID, or by name for version-3 headers, and its grid and size are
  checked against the map; a mismatch shows in the status column. The log prints the first tile's
  PC, tilts and phases, and warns when tiles differ in any of them.
- **1. Batch processing** uses the pattern processor tab's settings. *Load selected tile in the
  processor* loads a raw tile in tab 1 to set up the processing (background, NLPAR lambda, IQ
  range, ...). *Process all tiles* then computes one static background from the patterns of all
  tiles together (streamed, when the processor's static background is on), and processes the tiles
  one after the other with those settings into `<tile>_proc.up1` in the output folder (the suffix
  is editable). The header is copied as it is, so the processed file keeps the GUID. Each tile is
  unloaded after it is written, and the processor is empty when the batch ends. The IQ range
  applies per tile as fractions of that tile's IQ range, as in the processor; excluded patterns are
  written as zeros. NLPAR, when enabled, runs per tile with the lambda optimized on the preview
  tile.
- **2. Batch indexing** uses the indexing tab's settings: one projection center, sample tilt,
  camera tilt and pixel size for the whole montage, as typed there (*Copy the project's PC and
  tilts to the indexing tab* fills them from the first tile; check them). *Load selected tile in
  the indexer* puts one processed tile into tab 2 for a test region. *Index all tiles* loads each
  processed tile memory mapped, skips its zero patterns, indexes it band by band with the live view
  of this tab, and writes its orientation matrices, cross-correlation (`CI`), image quality
  (`IQ`, EMSphInx's discrete cosine measure, as EMSphInx's own .ang output; tick *Keep the
  project's (Hough) IQ* to leave the acquisition's IQ in place), phase and `Valid` flag into a
  copy of the project, `<project>_sph_bw<bandwidth>.edaxh5` in the output folder, right after the
  tile is done (the copy is made when the batch starts, so tiles not indexed keep their Hough
  results). The orientation matrix convention of the file was verified against the Hough results
  of a real tile. Untick *Index the processed files* to index the raw files instead.
- **Cancel batch** stops the batch: in processing the current tile is abandoned (its partial file is
  deleted) and the tiles already written stay; in indexing after the current band, with the tiles
  already indexed in the output project.

The two steps are separate on purpose: all processed tiles are on disk and RAM is empty before the
indexing starts. From a script, `montage.py` does the reading and writing:

```python
import montage
project = montage.read_project("GMAW 304L 309.edaxh5")
montage.pattern_folder_tiles(project, "processed", prefer_suffix="_proc")
print(montage.tile_table(project))
out = montage.copy_project(project.path, "processed/GMAW 304L 309_sph_bw68.edaxh5")
montage.write_tile_result(out, project.tiles[0], xmap)   # xmap: the tile's crystal map, TWIST applied
```

`python montage.py project.edaxh5 --patterns FOLDER` prints the tile table from the command line.
`make_montage_sample.py` cuts a few-MB sample (every map cropped to a few rows and columns, the
pattern files cropped to match, GUIDs kept) out of a real montage for tests;
`example_data/montage_sample/` is one such sample.

---

## 7. Using the API directly

The notebook is a thin layer over one method of the fork:

```python
import dask, kikuchipy as kp

s = kp.load("scan.up1", lazy=True)            # or any format kp.load() reads
harmonics = kp.indexing.MasterPatternHarmonics.from_file("Ni 20kV.sht")
det = kp.detectors.EBSDDetector(shape=s.axes_manager.signal_shape[::-1],
                                pc=(0.52, 0.18, 0.70), convention="edax",
                                sample_tilt=70, tilt=8)
dask.config.set(num_workers=8)
xmap = s.spherical_indexing(harmonics, det, bandwidth=68, backend="gpu")
```

`sph_core.py` wraps these steps as functions with the notebook's settings names (see its docstring).
Signature on the pinned commit (unchanged from `71a1b2f`):

```
EBSD.spherical_indexing(harmonics, detector, bandwidth=68, n_best=1, navigation_mask=None,
    signal_mask=None, normalize=True, refine=True, n_regions=10, gaussian_background=False,
    circular_mask=False, emsphinx_compatible=True, pseudo_symmetry_ops=None, backend="cpu",
    chunksize=None, verbose=1) -> CrystalMap
```

There is no `n_jobs`; workers come from `dask.config.set(num_workers=N)`. `chunksize` is
patterns per Dask chunk and, on the GPU, the device batch size.

---

## 8. Performance notes and known issues

- **What runs on the GPU:** only the coarse Euler-grid cross-correlation (about 80 % of the
  CPU time). Preprocessing, back-projection and the float64 Newton refinement stay on the CPU,
  which bounds the speed-up. Consumer GPUs run float64 at a fraction of float32 speed, which is
  why refinement is not on the device.
- **Throughput scales with pattern size and bandwidth.** Measured on the tested laptop:
  60 px patterns at bandwidth 68, about 1200 patterns/s on the GPU; 479 px patterns at
  bandwidth 123, about 130 patterns/s on the GPU and about 28 on 8 CPU workers.
- **VRAM held after a run.** Without the plan-cache patch, several GB of VRAM can stay
  allocated in the process after a GPU run, and the next call picks a tiny automatic batch
  size. Work-arounds: apply the patch, set `CHUNKSIZE` explicitly, or restart the kernel
  between runs. Also check that no other process (a game, another notebook) holds VRAM when
  a run reports out of memory.
- **Large pattern files.** `LAZY = "auto"` memory-maps files larger than a quarter of RAM.
  The operating system may show a lot of "used" memory for mapped file pages; that is cache,
  not a leak. Apply the lazy-load patch or opening such files takes minutes.
- **Accuracy vs the EMSphInx binary:** the author reports a median refined disagreement of
  about 0.3 deg, due to a pixel-stretch quirk in EMSphInx that was deliberately not copied.
  On an RKD data set the notebook matched an EMSphInx result to a median of 0.03 deg.
- **Old `.up1` files** (header version 3) do not store the map size or step; the notebook
  takes them from the `.ang` file. Hexagonal grids are not handled by that fallback.
- **Dask version.** Tested with dask 2026.1.2, numba 0.64, numpy 2.3, orix 0.14, hyperspy 2.4 on
  the laptop (commit `71a1b2f`); the re-pin to `4c7a0035` was verified with dask 2026.8, numba 0.68,
  numpy 2.5, orix 0.15, hyperspy 2.5, pyebsdindex 0.3.10.1 (CPU only).

---

## 9. References

- Fork: https://github.com/jwestraadt/kikuchipy (branch `develop`, commit `4c7a0035`).
  Spherical indexing code is in `src/kikuchipy/indexing/_spherical/`, HROSM in
  `src/kikuchipy/indexing/_hrosm/`, design notes in `specs/`, tutorials in
  `doc/tutorials/spherical_indexing.ipynb` and `doc/tutorials/hrosm.ipynb`.
- W. C. Lenthe, S. Singh, M. De Graef, "A spherical harmonic transform approach to the
  indexing of electron back-scattered diffraction patterns", Ultramicroscopy 207 (2019).
- EMSphInx: https://github.com/EMsoft-org/EMSphInx
- kikuchipy documentation: https://kikuchipy.org
