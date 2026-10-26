# GPU-accelerated spherical indexing of EBSD / RKD patterns

The **Kikuchi workbench**: a PyQt5 GUI that puts the Kikuchi Pattern Processor (`pattern_processor/`,
formerly its own repository) and GPU spherical indexing in one window, with a tab that indexes one clicked pattern and
shows the master pattern at the orientation found, and a tab that runs processing and indexing over every tile of an
EDAX APEX montage. The indexing is Johan Westraadt's Python port of **EMSphInx** (spherical harmonic EBSD
indexing, Lenthe et al. 2019) with its **CuPy GPU backend**, which lives in his fork of kikuchipy;
nothing of it is in upstream kikuchipy or on PyPI, so it is installed from a git clone (section 3).
Everything in this repository is the GUI and the functions it calls; there are no notebooks.

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
| `pattern_processor/` | The Kikuchi Pattern Processor: `gpu_pattern_viewer.py` (NLPAR through pyebsdindex, aperture, IQ range, background, stretch, CLAHE, `Apply && Save`, the montage batch API), its patched OpenCL NLPAR kernel, its own README, plan notes and HPC launch script. Tab 1 of the workbench, and still runnable on its own (`python pattern_processor/gpu_pattern_viewer.py`). Merged from `helpimlost97/kikuchi-pattern-processor` with its history. |
| `kikuchi_workbench.py`, `point_tab.py`, `live_view.py`, `sph_core.py`, `Open workbench.bat` | The GUI: the Kikuchi Pattern Processor and spherical indexing in one window, see section 6.1. `sph_core.py` holds the indexing functions the GUI calls (one per step: load, preprocess, harmonics, detector, index, save, compare, plus the single-pattern functions; usable from scripts), `point_tab.py` the *Index a point* tab, `live_view.py` the animated progress view. |
| `montage.py`, `montage_tab.py`, `STITCH_DATASET_NOTES.md`, `make_montage_sample.py`, `example_data/montage_sample/` | Batch processing and indexing of an EDAX APEX montage (a scan collected as tiles, to be stitched in OIM Analysis or MTEX afterwards), see section 6.2. `montage.py` reads the `.edaxh5` project, links each tile to its `.up1` by the GUID in the file header and writes results into a copy of the project; `montage_tab.py` is the GUI's fourth tab. The notes describe the file formats; the sample is a real project cropped to 3 x 40 points per tile by `make_montage_sample.py`. |
| `preset_RKD.txt` | Example preset file for the GUI (an RKD data set, detector parallel to the sample): the indexing parameters as `NAME = value` lines. The paths in it are placeholders and are not read by the GUI. |
| `ebsd_params.py` | Reads acquisition parameters (pattern size, projection center, map size, step) from an EDAX `.ang` or `.edaxh5` file and a `.up1`/`.up2` file. Used by the GUI's *Read scan parameters* and usable from the command line. |
| `sph_benchmark.py` | CPU vs GPU speed and parity benchmark on the nickel demo data set. Use it to verify the install. |
| `test_gpu_memory_fix.py` | Checks that VRAM is released between two GPU runs (tests one of the local patches). |
| `osc_refinement_handoff/` | Full pattern-matching refinement with a PC per map point (hours to days, CPU) as a stand-alone script for a cluster, with its own hand-off notes. |

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
does NLPAR through pyebsdindex), HROSM (`EBSD.hrosm`, not used here) and a HyperSpy 2.5 fix. The
spherical indexing code and its `spherical_indexing()` signature are unchanged, the patch applies
cleanly, and the fork's spherical indexing tests (326, with the patch) and the workbench's indexing
of the demo data were re-run against it. To try a newer commit later:

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

`update_fork.py` has already run `pip install -e kikuchipy`. Add what the GUI and scripts need
beyond the core package:

```bash
pip install pooch nlopt psutil pyqt5 pyebsdindex scipy scikit-image
```

In that line, `pooch` downloads the nickel demo data, `nlopt` is used by the refinement script,
`psutil` decides whether to load a file lazily, and the last four are the pattern processor's
packages (add `pyopencl` for NLPAR on the GPU). Alternatively
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
python sph_benchmark.py --backend both --workers 8 --compare    # from this folder
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

## 6. Index your own data with the workbench

### 6.1 The GUI: pattern processor and spherical indexing in one window

`kikuchi_workbench.py` puts the Kikuchi Pattern Processor (`pattern_processor/gpu_pattern_viewer.py`)
and the spherical indexing of `sph_core.py` into one PyQt5 window with four tabs:

1. **Pattern processor**: `gpu_pattern_viewer.py` as it is, so it also runs on its own
   (NLPAR through pyebsdindex, aperture, IQ range, background, stretch, CLAHE, `Apply && Save`).
2. **Index a point**: click a point on the map, index that one pattern and see it next to the master
   pattern simulated at the orientation found, see 6.1.1.
3. **Spherical indexing**: the indexing settings as a form on the left (drag the divider to
   resize it), the live view, the result maps and a log on the right.
4. **Montage batch**: processing and indexing for every tile of an EDAX APEX montage, see 6.2.

The settings of the indexing tab (the names are those of `sph_core.Settings` and of the preset files):

- `PC` and `PC_CONVENTION`: the projection center. Values from an `.ang` file are in the `"edax"`
  convention, which is the default; *Read scan parameters* fills them in.
- `CAMERA_TILT`: detector tilt / camera elevation in degrees (EMSphInx `thetac`). Typically 8 for a
  conventional EBSD camera, -90 for RKD (detector parallel to the sample). The fork has no -60 to
  60 deg limit on this, unlike the EMSphInx binary.
- `SAMPLE_TILT = None` takes the tilt the master pattern was simulated for, as EMSphInx does. Set
  `OVERRIDE_MASTER_SAMPLE_TILT` only if you deliberately want another value.
- `MASTER_FILE`: an EMSphInx `.sht` file or an EMsoft master pattern `.h5`. Several files, one per
  line, index several phases.
- `REMOVE_STATIC_BACKGROUND` / `REMOVE_DYNAMIC_BACKGROUND`: off for pattern files that the processor
  already background corrected or NLPAR processed (the default).
- `BANDWIDTH`: 68 is the EMSphInx default, 88 or 123 for more accuracy at lower speed.
  `kp.indexing.fast_bandwidths()` lists the efficient values.
- `BACKEND = "gpu"`, `N_WORKERS` = number of performance cores, `CHUNKSIZE = None` (automatic GPU
  batch size; set a number if a run reports out of memory).
- `TWIST`: rotation of the detector about its normal. The indexer cannot use a twisted detector, so
  the indexed orientations are rotated about the sample normal afterwards. Flip the sign if the
  IPF-X and IPF-Y maps come out rotated the wrong way.
- `TEST_SIZE`: side of the square test region, in map points. If the test maps look like noise, try
  `FLIP_PATTERNS_UD` or check the projection center and camera tilt.
- `ROI`: `(col_start, col_stop, row_start, row_stop)` to index part of the map; `REFERENCE_FILE`: an
  `.ang` of the same map from EMSphInx or OIM for *Compare with reference*.

Workflow in the indexing tab:

- **One data set at a time**: loading a file in either tab unloads the other tab's data set
  (the processor's scan and NLPAR result, or the indexer's patterns, map and last result), so two
  scans never sit in memory together. After `Apply && Save`, loading the processed file in the
  indexing tab therefore empties the processor tab; the saved file is kept.
- **One pattern file for the first three tabs**: the file loaded in the processor becomes the pattern
  file of the point and indexing tabs (one field, shown in both; *Load patterns* in either loads it),
  and the processor's *Load UP1 / UP2* dialog opens on the file set in those tabs. The path is shared,
  the patterns are loaded by the tab that needs them.
- **Pattern source**: always a file. **Load processed patterns** is greyed out until the
  processor's `Apply && Save` has written a file; it then puts that file into the pattern file box
  and loads it. Every pattern of the file is indexed: the processor leaves the patterns outside its
  IQ range out of the processing (background, stretch, NLPAR) but writes them unprocessed, so they
  are still in the file. The browse button takes any other file kikuchipy reads, e.g. a scan that
  was not processed. Big files are memory mapped and read chunk by chunk, so RAM is not a concern.
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
  per-band setup), and the Live view paints the patterns of every finished chunk (a GPU batch or a CPU
  chunk, a few times a second, in file order) with their IPF-Z colours over the grey quasi-IQ map
  the moment the indexer returns them, as EMSphInx's preview does: the refined orientation when
  *Refine* is ticked, the coarse grid orientation when it is not. The band being indexed is tinted
  orange until its patterns are painted, a pattern of that band is shown on the right (cycling
  through the band), and the status line reports elapsed time, ETA, the speed over the last few
  seconds and the speed of the current band.
  The **Results** tab then shows the score, IQ, IPF-Z and IPF-X maps. Untick *Live view* for a
  single call without animation.
- **Save .ang / .h5**, **Compare with reference**, and **Load/Save preset**. A GUI preset holds the
  indexing parameters only: master bandwidth, PC convention, sample and camera tilt, pixel size,
  twist, the two background removals, bandwidth, workers, refinement, AHE tiles, circular mask,
  Gaussian background, normalization and chunk size. Files, folders and the projection center
  values are never written to or taken from a preset. `preset_RKD.txt` is an example; other names
  in a preset file are listed in the log and ignored.

```bash
python kikuchi_workbench.py
python kikuchi_workbench.py --preset preset_RKD.txt
python kikuchi_workbench.py --no-processor                    # the point, indexing and montage tabs only
```

On Windows `Open workbench.bat` does the same (set `PYTHON` at its top). The environment is the one
of this README (fork, CuPy) plus the processor's packages: `pip install pyqt5 pyebsdindex scipy scikit-image` (and `pyopencl` for NLPAR
on the GPU). The processor's own switches (`KPP_GPU=0`, `KPP_THREADS`, `KPP_IN_MEMORY`, ...) apply
inside the workbench too.

Memory: with the background removals off (the default), the loaded patterns are indexed as they
are, memory mapped when the file is bigger than a quarter of RAM; with a removal on, the file is
read again and processed. Long steps run in a worker thread with their console output in the log
pane.

Verified headlessly on the nickel demo data written as a `.up1` (no GPU in that environment, CPU
backend): processor load and `Apply && Save`, `Load processed patterns`, quasi-IQ map, band-wise test region and full map with the live events, save. The
band-wise result is checked to be identical to a single call. The point tab was checked the same way (2026-10-25):
the mirrored fields, a load without a map, a scan file of another scan refused, the quasi-IQ map on request, a
point indexed with the indexer reused on the next point and rebuilt on a bandwidth change; its orientation is
identical to `spherical_indexing` on the same pattern (`diagnostics/point_index_check.py`).

#### 6.1.1 Index a point: one pattern against the master

The **2. Index a point** tab (`point_tab.py`) indexes a single clicked pattern and shows the master pattern
at the orientation found next to it, to check a point, the projection center, the camera tilt or the twist
sign without indexing a region.

- **Layout**: the files, the shared settings and the buttons on the left; the map in the middle, with
  the status line and the log under it; on the right the clicked pattern as the indexer sees it above
  the pattern simulated from the master. The **opacity slider** between them lays the simulated
  pattern (raw, no equalization) over the measured one: 0 % shows the measured pattern alone, 100 %
  the simulated one alone, anything between blends them so the band positions can be compared in
  place.
- **Files and settings**: the pattern file, scan file, master file(s), projection center, bandwidth,
  AHE tiles and circular mask are the same as the indexing tab's (a change in either tab is a change in
  both; adjust the PC or the bandwidth here and index the point again to see the match change, the
  indexer is rebuilt on the next click). The master bandwidth, the tilts, pixel size and twist, the
  background steps and the other indexing settings (refinement, Gaussian background, ...) are read
  from the indexing tab.
- **Load patterns** loads the file (one data set in memory at a time, as everywhere) and shows the
  map: with a scan file given, the **IQ column of the scan file**: an `.ang` (the acquisition's Hough IQ,
  or an indexing result's), an APEX `.edaxh5` (the map chosen by the map filter), an `.h5` written by
  the workbench's *Save .h5* (orix crystal map; its IQ is the spherical one) or an h5ebsd `.h5` (EDAX
  OIM, kikuchipy, EMSphInx). It must be this scan's and on a square grid.
  Without a scan file the map stays empty until **Build quasi-IQ map** computes the quasi-IQ map
  (the standard deviation of every raw pattern, one read of the file); nothing is computed on its
  own. The indexing tab still builds the quasi-IQ map by itself. **Read scan parameters** is the
  indexing tab's button, for the projection center.
- **Click a point** on the map: the pattern is shown on the right as the indexer will see it (as in
  the indexing tab, which follows the same click) and marked in red; the simulated panel empties
  until the point is indexed.
- **Index this point** indexes that pattern alone, on the CPU whatever `BACKEND` says, and fills the
  right column: the measured pattern as the indexer sees it (its pixel mask included) on top, and
  below it the pattern simulated from the master at the orientation found (the fork's
  `to_master_pattern()` of the cached harmonics, i.e. exactly what the indexer matched against,
  projected with `get_patterns()` on the same detector, shown as it comes from the master). The
  phase, the Euler angles (Bunge, with `TWIST` applied as in the saved maps), the score, the IQ, the
  normalized cross-correlation between the measured pattern and the simulated one put through the
  same equalization, and the timings are in the titles, the result line and the log. The indexer and the
  Lambert master are built on the first click (a second at bandwidth 68, longer at higher
  bandwidths; the first simulation also compiles for a few seconds) and reused until the master,
  the detector or an indexing setting changes. A simulated pattern that looks nothing like the
  measured one points at the projection center, the camera tilt or `FLIP_PATTERNS_UD`.

From a script the same steps are `sph_core.make_point_indexer`, `index_pattern`, `lambert_master` and
`simulate_pattern` (see the module docstring); `diagnostics/point_index_check.py` runs them on the nickel
demo and checks the orientation against `spherical_indexing`.

### 6.2 Montage batch: process and index every tile of an APEX montage

An APEX montage ("stitch") scan is one `.edaxh5` project holding every tile as a map (`OIM Map
1_r000c000`, `..._r000c001`, ...) with its grid, step, pattern center, phases and the Hough results,
plus one `.up1`/`.up2` per tile. The version-4 `.up1` header carries the tile's map GUID, which is
how the files are linked (the project never names the pattern files); `STITCH_DATASET_NOTES.md`
describes both formats. Nothing is stitched here: every tile is processed and indexed on its own
grid, and the stitching is done afterwards in OIM Analysis or MTEX from the output project.

The **4. Montage batch** tab:

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
  applies per tile as fractions of that tile's IQ range, as in the processor; the patterns outside it
  are written unprocessed (they take no part in the background, stretch or NLPAR). NLPAR, when enabled, runs per tile with the lambda optimized on the preview
  tile.
- **2. Batch indexing** uses the indexing tab's settings: one projection center, sample tilt,
  camera tilt and pixel size for the whole montage, as typed there (*Copy the project's PC and
  tilts to the indexing tab* fills them from the first tile; check them). *Load selected tile in
  the indexer* puts one processed tile into tab 3 for a test region. *Index all tiles* loads each
  processed tile memory mapped, indexes it band by band with the live view
  of this tab, and writes its orientation matrices, cross-correlation (`CI`), image quality
  (`IQ`, EMSphInx's discrete cosine measure, as EMSphInx's own .ang output; tick *Keep the
  project's (Hough) IQ* to leave the acquisition's IQ in place), phase and `Valid` flag into a
  copy of the project, `<project>_sph_bw<bandwidth>.edaxh5` in the output folder, right after the
  tile is done (tiles not indexed keep their Hough results). All-zero patterns, which cannot be
  indexed (processor versions before 1.7.0 wrote them outside the IQ range), are skipped; they are
  found on the raw data, before the background removal, which would rescale them. The orientation matrix convention
  of the file was verified against the Hough results of a real tile. Untick *Index the processed
  files* to index the raw files instead. Tiles without a usable file are named in the log and
  skipped, and a bad processed file shows in the table's status column.
  - *Project phase of each master*: the `Phase` column of the output must point into the project's
    phase list (printed with its numbers when the project is loaded). Blank = each master pattern is
    matched to a project phase by name (whole name or whole word, so "ferrite" matches "Ferrite (bcc)"),
    and by order when the names do not match but the counts agree; otherwise the batch refuses to
    start. Type the numbers to decide yourself, e.g. `2` or `2,1`.
  - *Resume* (on by default): an output project that already exists is kept and the tiles it already
    holds spherical results for are skipped, so a batch that was cancelled or had a failed tile
    continues where it stopped. Untick it for a fresh copy of the project and every tile again. While
    the output is partial, the log says so (the indexed tiles' IQ is on the spherical 0..1 scale, the
    others' on the Hough scale, unless the Hough IQ is kept).
- **Cancel batch** stops the batch: in processing during the shared background pass or after the
  current tile (its partial file is deleted; the tiles already written stay); in indexing after the
  current band, with the tiles already indexed in the output project. While a batch runs the tab's
  fields and the processor tab are locked, and the indexing tab refuses to load or run.

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

### 6.3 Reading scan parameters from the command line

```bash
python ebsd_params.py "scan.ang" --up1 "scan.up1"            # EDAX and EMsoft pattern centers
python ebsd_params.py "scan.edaxh5" --map "Area 1" --px 115
```

The docstring at the top of `ebsd_params.py` lists all options.

### 6.4 Correct the projection center drift

Spherical indexing (like EMSphInx) uses one projection center (PC) for the whole map, but the
true PC moves with the beam by one step length per scan step. Indexed with a fixed PC, that
shift is read as a rotation of about *beam travel / detector distance*: for a 3 mm wide map at
a detector distance of 18 mm it is 10 degrees edge to edge, and about 0.1 degrees across a 30 um
grain. Point-to-point noise is unaffected, but grain reference orientation deviations and
orientation gradients within grains are.

A projection center drift correction (every point's PC extrapolated from one anchor PC and the scan
geometry, and each orientation rotated back by its local correction, without re-indexing and without the
pattern file) exists outside this repository. What the repository holds for this is the cluster script in
`osc_refinement_handoff/` (`refine_pc_per_point.py`, with its handoff notes): the full pattern-matching
refinement with a PC per point, which also removes the band limit; it costs hours to days.

## 7. Using the API directly

The GUI is a thin layer over one method of the fork:

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

`sph_core.py` wraps these steps as functions driven by a `Settings` object (see its docstring).
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
  size. Work-arounds: apply the patch, set `CHUNKSIZE` explicitly, or restart the program
  between runs. Also check that no other process holds VRAM when
  a run reports out of memory.
- **Large pattern files.** `LAZY = "auto"` memory-maps files larger than a quarter of RAM.
  The operating system may show a lot of "used" memory for mapped file pages; that is cache,
  not a leak. Apply the lazy-load patch or opening such files takes minutes.
- **Accuracy vs the EMSphInx binary:** the author reports a median refined disagreement of
  about 0.3 deg, due to a pixel-stretch quirk in EMSphInx that was deliberately not copied.
  On an RKD data set this code matched an EMSphInx result to a median of 0.03 deg.
- **Old `.up1` files** (header version 1) do not store the map size or step; the GUI
  takes them from the `.ang` file (version 3 and later store both). Hexagonal grids are not
  handled by that fallback.
- **Dask version.** Tested with dask 2026.1.2, numba 0.64, numpy 2.3, orix 0.14, hyperspy 2.4 on
  the laptop (commit `71a1b2f`); the re-pin to `4c7a0035` was verified with dask 2026.8, numba 0.68,
  numpy 2.5, orix 0.15, hyperspy 2.5, pyebsdindex 0.3.10.1 (CPU only).

---

## 9. References

- Fork: https://github.com/jwestraadt/kikuchipy (branch `develop`, commit `4c7a0035`).
  Spherical indexing code is in `src/kikuchipy/indexing/_spherical/`, design notes in `specs/`,
  tutorial in `doc/tutorials/spherical_indexing.ipynb`.
- W. C. Lenthe, S. Singh, M. De Graef, "A spherical harmonic transform approach to the
  indexing of electron back-scattered diffraction patterns", Ultramicroscopy 207 (2019).
- EMSphInx: https://github.com/EMsoft-org/EMSphInx
- kikuchipy documentation: https://kikuchipy.org
