# Kikuchi workbench

A PyQt5 GUI for processing and spherically indexing EBSD / RKD patterns on a GPU, in one window with four tabs:
a pattern processor (NLPAR and the usual background, stretch and CLAHE steps), a tab that indexes one clicked
pattern and shows the master pattern at the orientation found, a spherical indexing tab with a live view of the
map filling in, and a batch tab that processes and indexes every tile of an EDAX APEX montage.

**The workbench is a GUI. The science in it is other people's work**, made clickable:

- **Spherical indexing** is Johan Westraadt's Python port of EMSphInx (Lenthe, Singh and De Graef, 2019) with
  his CuPy GPU backend, written in his fork of kikuchipy, https://github.com/jwestraadt/kikuchipy. It is not in
  upstream kikuchipy or on PyPI, so `requirements.txt` installs it from https://github.com/helpimlost97/kikuchipy,
  a fork of his repository pinned to one commit, carrying two small patches of our own (fast loading of large
  `.up1`/`.up2` files, VRAM released after a run). Everything the indexing tabs do is a call into his code.
- **Pattern processing** builds on pyebsdindex (Dave Rowenhorst and others), whose NLPAR (Brewick, Wright and
  Rowenhorst, 2019) does the denoising.
- **kikuchipy** (Håkon Wiik Ånes and the kikuchipy developers) and **HyperSpy** do the file reading and the
  data handling underneath.

Everything in this repository is vibe coded; read the code before trusting a number. GPLv3, see section 8.

Tested on Windows 11, Python 3.14, an RTX 5070 Laptop GPU (8 GB VRAM) with a CUDA 13 driver, 32 GB RAM. Linux
works the same way; only the path syntax in the examples changes.

**In this README**

1. [What is in this folder](#1-what-is-in-this-folder)
2. [Install](#2-install)
3. [Verify the install](#3-verify-the-install)
4. [The workbench](#4-the-workbench): the four tabs and the rules they share, then [4.1 Index a point](#41-index-a-point)
   and [4.2 Spherical indexing tab](#42-spherical-indexing-tab), each as a workflow first and a description of
   the buttons and fields after it
5. [Montage batch](#5-montage-batch): processing and indexing every tile of an EDAX APEX montage
6. [Command line and scripts](#6-command-line-and-scripts)
7. [Performance notes and known issues](#7-performance-notes-and-known-issues)
8. [References and license](#8-references-and-license)

---

## 1. What is in this folder

| File / folder | Purpose |
|---|---|
| `requirements.txt` | Everything the GUI and scripts need except CuPy (section 2). |
| `LICENSE` | GPLv3, for everything in this repository (section 8). |
| `kikuchi_workbench.py`, `point_tab.py`, `live_view.py`, `theme.py`, `sph_core.py`, `Open workbench.bat` | The GUI (section 4). `sph_core.py` holds the indexing functions the GUI calls, one per step (load, preprocess, harmonics, detector, index, save, compare, plus the single-pattern ones), usable from scripts; `point_tab.py` is the *Index a point* tab, `live_view.py` the animated progress view, `theme.py` the dark and light appearance. |
| `pattern_processor/` | The Kikuchi Pattern Processor: `gpu_pattern_viewer.py` (NLPAR through pyebsdindex, aperture, IQ range, background, stretch, CLAHE, `Apply && Save`), its patched OpenCL NLPAR kernel, its own README and notes. Tab 1 of the workbench, and still runnable on its own (`python pattern_processor/gpu_pattern_viewer.py`). |
| `montage.py`, `montage_tab.py`, `STITCH_DATASET_NOTES.md`, `make_montage_sample.py`, `example_data/montage_sample/` | Batch processing and indexing of an EDAX APEX montage (section 5). `montage.py` reads the `.edaxh5` project, links each tile to its `.up1` by the GUID in the file header and writes results into a copy of the project; the notes describe the file formats; the sample is a real project cropped to 3 x 40 points per tile. |
| `ebsd_params.py` | Reads acquisition parameters (pattern size, projection center, map size, step) from an EDAX `.ang` or `.edaxh5` and a `.up1`/`.up2`. Behind *Read scan parameters*; also a command-line tool (section 6). |
| `preset_RKD.txt` | Example preset for the GUI (an RKD data set, detector parallel to the sample): indexing parameters as `NAME = value` lines. Its paths are placeholders. |
| `sph_benchmark.py` | CPU vs GPU speed and parity benchmark on the nickel demo data set; verifies the install (section 3). |
| `test_gpu_memory_fix.py` | Checks that VRAM is released between two GPU runs. |
| `check_defaults.py` | Builds the GUI headless and checks that every widget starts at its owner's default (`sph_core.Settings`, the processor's defaults). CPU, seconds. |
| `osc_refinement_handoff/` | A stand-alone cluster script for full pattern-matching refinement with a projection center per map point (hours to days, CPU), with its hand-off notes. |

---

## 2. Install

**Requirements.** An NVIDIA GPU with a current driver (about 6 GB of free VRAM is comfortable; batches are
sized to the free memory and halved on out-of-memory, so smaller cards work, slower). A CPU with several
cores: preprocessing, back-projection and the Newton refinement run on the CPU through Dask threads, so the core
count matters in GPU mode too. Python 3.10 to 3.14 in a fresh virtual or conda environment (the kikuchipy fork
must not share an environment with another kikuchipy install). `git` on the PATH. No CUDA Toolkit install is
needed.

**The workbench and its packages.** Activate the environment, then:

```bash
git clone https://github.com/helpimlost97/kikuchi-workbench.git
cd kikuchi-workbench
pip install -r requirements.txt
```

The first line of `requirements.txt` makes pip clone the kikuchipy fork at the pinned commit and install it like
any other package; the rest is PyQt5, pyebsdindex and the other packages the GUI and scripts use. To try a newer
commit of the fork, change the hash after `@` in `requirements.txt` and run the install again. To edit the fork
in place instead, clone it into `kikuchipy/` next to this README (git-ignored) and `pip install -e kikuchipy`.

**CuPy**, separately, because its wheel depends on your driver. `nvidia-smi` prints `CUDA Version: 13.x` or
`12.x` in the top right (drivers from 580 onwards support CUDA 13). Install the matching wheel with its CUDA
pieces as pip wheels (the `[ctk]` extra; no CUDA Toolkit, no admin rights):

```bash
pip install "cupy-cuda13x[ctk]"      # driver supports CUDA 13
pip install "cupy-cuda12x[ctk]"      # driver supports CUDA 12 only
```

CuPy 14 is required on Python 3.14. Without `[ctk]`, a machine without a CUDA Toolkit fails at the first GPU
kernel with "Failed to find CUDA headers". The `CUDA path could not be detected` warning at import is harmless.
NLPAR on the GPU in the pattern processor goes through OpenCL instead and needs `pip install pyopencl`; without
it NLPAR runs on the CPU.

---

## 3. Verify the install

```bash
python -c "import kikuchipy as kp, cupy; print(kp.__version__, kp.__file__); print(cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())"
```

Expected: version `0.14.dev0`, a path in your environment, and your GPU's name. `cudaErrorNoDevice` or
`cudaErrorInsufficientDriver` means the driver is too old for the wheel you picked; the GPU backend then refuses
to run rather than silently using the CPU.

Then the benchmark on the nickel demo data set (4125 patterns of 60 x 60 px, downloaded once, a few MB):

```bash
python sph_benchmark.py --backend both --workers 8 --compare
```

`--workers` defaults to 2, as does the GUI's `N_WORKERS`; set it to about your number of performance cores. The
first call of each backend includes about 10 s of Numba compilation, excluded by a warm-up run. On the tested
laptop, bandwidth 68 with refinement: about 230 patterns/s on 8 CPU workers, about 1200 patterns/s on the GPU.
`--compare` must report identical winners and a median CPU vs GPU misorientation of 0 deg.

Optional: `python test_gpu_memory_fix.py` prints `RESULT: PASS` when the free VRAM after two runs is back to its
starting value and both runs give the same orientations.

---

## 4. The workbench

```bash
python kikuchi_workbench.py
python kikuchi_workbench.py --preset preset_RKD.txt
python kikuchi_workbench.py --no-processor         # the point, indexing and montage tabs only
```

On Windows `Open workbench.bat` does the same (set `PYTHON` at its top to your environment's interpreter). The
processor's own switches (`KPP_GPU=0`, `KPP_THREADS`, `KPP_IN_MEMORY`, ...) apply inside the workbench too. Long
steps run in a worker thread with their console output in the log pane. The **Appearance** menu switches the
whole window, figures included, between **Dark** (the default) and **Light**; the choice is remembered. At the
right end of the menu bar, the **progress line** shows the step running with its per cent, time or ETA (the
console progress bars of the step, the band summaries of a run, the batch's current tile); it is blank between
steps.

The tabs follow the order of a job: **process the patterns if they need it (tab 1), index a point or two to see
that the orientation found is the right one (tab 2), then index the full map (tab 3) or every map of a montage
(tab 4).** The four tabs:

1. **Pattern processor**: `gpu_pattern_viewer.py` as it is, so it also runs on its own. NLPAR through
   pyebsdindex, aperture, IQ range, background, stretch, CLAHE, `Apply && Save`. Its defaults live in
   `pattern_processor/processor_defaults.py`; its README is in that folder.
2. **Index a point**: click a point on the map, index that one pattern and see it next to the master pattern
   simulated at the orientation found (4.1).
3. **Spherical indexing**: the indexing settings as a form on the left (drag the divider to resize it), the live
   view, the result maps and a log on the right (4.2).
4. **Montage batch**: processing and indexing for every tile of an EDAX APEX montage (section 5).

Two rules hold across the tabs. **One data set at a time**: loading a file in any tab unloads the others' data
(the processor's scan and NLPAR result, or the indexer's patterns, map and last result), so two scans never sit
in memory together; after `Apply && Save`, loading the processed file in the indexing tab empties the processor
tab, and the saved file is kept. **One pattern file for the first three tabs**: the file loaded in the processor
becomes the pattern file of the point and indexing tabs (one field, shown in both), and the processor's *Load
UP1 / UP2* dialog opens on the file set there.

### 4.1 Index a point

The **2. Index a point** tab indexes a single clicked pattern and shows the master pattern simulated at the
orientation found next to it. It is the check between processing and a full run: index a couple of pixels,
confirm that the orientation, the projection center, the camera tilt and the twist sign are right, then go to
tab 3 or tab 4 with the same settings. Everything a basic run needs is on this tab; the fields it shares with
the indexing tab are one value shown in two places.

**Workflow**

1. **Pattern file**: browse for the `.up1`/`.up2` (or any file kikuchipy reads). A file loaded in the processor
   or in the indexing tab is already here, since the three tabs share the field.
2. **Scan file**: the `.ang` or `.edaxh5` written at acquisition. For an `.edaxh5` project with several areas,
   type part of the wanted map's name in **edaxh5 map filter** (for instance `Area 7`), so that map is read.
   **Read scan parameters** then fills the projection center, its convention and the pixel size from the file:
   no hunting for a pattern center.
3. **Load patterns**: the map appears, with the scan file's IQ column as its image.
4. **Master file(s)**: one EMsoft master pattern `.h5` per phase.
5. **Click a point** on the map and press **Index this point**. The measured pattern is shown above the pattern
   simulated from the master at the orientation found; drag the **opacity slider** to lay one over the other
   and compare the band positions in place. The result line gives the phase, the Euler angles, the score and
   the cross-correlation between the two patterns.
6. If the simulated pattern does not match: a wrong projection center, camera tilt or `FLIP_PATTERNS_UD` are
   the usual causes. Raw full-resolution patterns from a conventional detector often need **AHE tiles** (10 is
   EMSphInx's default) and the **circular mask** before the indexer finds the right orientation; tick them here,
   index the point again, and they are already set for tab 3.
7. Repeat for a second point in another grain, then move on to **3. Spherical indexing** for the whole map or
   **4. Montage batch** for a stitched project.

**The buttons and fields**

- **Layout**: files, shared settings and buttons on the left; the map in the middle with the log under it; on
  the right the clicked pattern as the indexer sees it, above the pattern simulated from the
  master. The **opacity slider** between them lays the simulated pattern over the measured one.
- **Pattern file**, **Scan file**, **edaxh5 map filter**, **Master file(s)**, **Projection center** and its
  **convention**, the **sample and camera tilts**, **pixel size**, **bandwidth**, **AHE tiles**, **circular
  mask** and **Gaussian background** are the same fields as the indexing tab's (a change in either tab is a
  change in both). The other indexing settings are read from the indexing tab.
- **Load patterns** shows the map: with a scan file, its **IQ column** (an `.ang`, an APEX `.edaxh5`, an `.h5`
  written by *Save .h5*, or an h5ebsd `.h5`; it must be this scan's, on a square grid). Without one the map is
  black until **Build quasi-IQ map**; every point is clickable either way.
- **Read scan parameters** prints the scan and pattern file headers in the log and fills the projection center
  (EDAX convention) and the pixel size (30 mm / pattern width for the EDAX/TSL convention; binning is always 1
  here). The camera elevation of the file is reported, never copied into the camera tilt.
- **Build quasi-IQ map**: the standard deviation of every raw pattern (one read of the file) as the map, for
  scans without a usable scan file.
- **Click a point**: the pattern is shown on the right and marked in red, on a fixed grey scale from 0 to the
  data's full range, as the processor shows its patterns: raw when nothing is ticked, otherwise with the ticked
  AHE tiles, circular mask and Gaussian background applied, exactly as the indexer will see it.
- **Index this point** indexes the pattern on the CPU and fills the right column with the measured pattern as
  the indexer sees it and the pattern simulated from the master at the orientation found (exactly what the
  indexer matched against, projected on the same detector). Phase, Euler angles (Bunge, `TWIST` applied as in
  the saved maps), score, IQ, the normalized cross-correlation between the two patterns and the timings go to
  the result line and the log. The indexer is built on the first click (a second at bandwidth 68) and reused
  until the master, the detector or an indexing setting changes.

### 4.2 Spherical indexing tab

**Workflow**

- **Pattern source**: always a file. **Load processed patterns** is greyed out until the processor's
  `Apply && Save` has written a file; it then loads that file. Every pattern of the file is indexed: patterns
  the processor left outside its IQ range are in the file unprocessed. The browse button takes any other file
  kikuchipy reads. Big files are memory mapped and read chunk by chunk.
- **Scan file** and **edaxh5 map filter**: the `.ang` or `.edaxh5` of the scan and, for a project with several
  areas, part of the wanted map's name. **Read scan parameters** fills the projection center from it. The
  **pixel size** is filled in as 30 mm / pattern width for the EDAX/TSL convention and stays editable; binning
  is always 1 here.
- **Load patterns** runs by itself after browsing for a file. It loads the patterns and shows a black map of the
  scan's shape in the **Live** view: every point of it can be clicked for its pattern, and a run paints its
  bands on it. **Build quasi-IQ map** (optional) puts the quasi image quality map (the standard deviation of
  every raw pattern, one read of the file) under the Live view; indexing does not need it.
- **Click anywhere on the map** before indexing to see that pattern on the right exactly as the indexer will
  see it, on a fixed grey scale from 0 to the data's full range, as the processor shows its patterns: raw when
  nothing is ticked, otherwise with the background removals, the Gaussian background and the adaptive histogram
  equalization (`N_REGIONS`) that are ticked. Changing one of those boxes refreshes the shown pattern. During a
  run the same panel cycles through the band being indexed.
- **Check geometry** shows one pattern on the sphere. **Index test region** and **Index full map** index band by
  band (whole rows, about 8 s of work each; identical results to one call) and paint every finished chunk's
  IPF-Z colours on the Live view the moment the indexer returns them, as EMSphInx's preview does. The status
  line reports elapsed time, ETA and speed; **Stop** ends a run without saving. The **Results** tab then shows
  the score, IQ, IPF-Z and IPF-X maps. Untick *Live view* for a single call without animation.
- **Save .ang / .h5**, **Compare with reference**, **Load/Save preset**. A preset holds the indexing parameters
  only (`PRESET_FIELDS` in `kikuchi_workbench.py`); files, folders and the projection center are never written
  to or taken from one. Every default of the form is the `sph_core.Settings` default, so a default is changed
  there alone (`python check_defaults.py` verifies this).

**The settings** (the names are those of `sph_core.Settings` and of the preset files):

- `SCAN_FILE` and `EDAXH5_MAP`: the acquisition file and, for an `.edaxh5` with several maps, part of the
  wanted map's name (case insensitive). Also where the IQ map and the map size of an old `.up1` come from.
- `PC` and `PC_CONVENTION`: the projection center. Values from an `.ang` are in the `"edax"` convention, the
  default; *Read scan parameters* fills them in.
- `CAMERA_TILT`: detector tilt / camera elevation in degrees (EMSphInx `thetac`). Typically 8 for a
  conventional EBSD camera, -90 for RKD (detector parallel to the sample).
- `SAMPLE_TILT = None` takes the tilt the master pattern was simulated for, as EMSphInx does. Set
  `OVERRIDE_MASTER_SAMPLE_TILT` only if you deliberately want another value.
- `MASTER_FILE`: an EMSphInx `.sht` or an EMsoft master pattern `.h5`. Several files index several phases.
- `REMOVE_STATIC_BACKGROUND` / `REMOVE_DYNAMIC_BACKGROUND`: off by default, for pattern files the processor
  already corrected.
- `N_REGIONS`, `CIRCULAR_MASK`, `GAUSSIAN_BACKGROUND`: EMSphInx's own pattern preparation inside the indexer:
  adaptive histogram equalization in `N_REGIONS` x `N_REGIONS` tiles (0 = off, 10 = EMSphInx's default), a
  circular mask over the detector, and a Gaussian background fit. Shared with the point tab.
- `BANDWIDTH`: 68 is the EMSphInx default; 88 or 123 for more accuracy at lower speed. Only the values the
  spherical harmonic transform is fast at are offered.
- `BACKEND = "gpu"`, `N_WORKERS` = Dask threads (default 2; raise to about your number of performance cores),
  `CHUNKSIZE = None` (automatic GPU batch size; set a number if a run reports out of memory).
- `TWIST`: rotation of the detector about its normal. The indexer cannot use a twisted detector, so the indexed
  orientations are rotated about the sample normal afterwards. Flip the sign if the IPF-X and IPF-Y maps come
  out rotated the wrong way.
- `TEST_SIZE`: side of the square test region, in map points. If the test maps look like noise, try
  `FLIP_PATTERNS_UD` or check the projection center and camera tilt.
- `ROI`: `(col_start, col_stop, row_start, row_stop)` to index part of the map; `REFERENCE_FILE`: an `.ang` of
  the same map from EMSphInx or OIM for *Compare with reference*.

---

## 5. Montage batch

An APEX montage ("stitch") scan is one `.edaxh5` project holding every tile as a map (`OIM Map 1_r000c000`,
`..._r000c001`, ...) with its grid, step, pattern center, phases and Hough results, plus one `.up1`/`.up2` per
tile. The version-4 `.up1` header carries the tile's map GUID, which is how the files are linked;
`STITCH_DATASET_NOTES.md` describes both formats. Nothing is stitched here: every tile is processed and indexed
on its own grid, and the stitching is done afterwards in OIM Analysis or MTEX from the output project.

- **Project**: browse for the `.edaxh5`. The table lists every map with its grid, the raw pattern file found for
  it in the *raw pattern folder* (default: the project's folder) and the processed file found in the *output
  folder* (default: `processed` next to the project). Tiles without a pattern file are listed and skipped. Files
  are matched by GUID (by name for version-3 headers) and checked against the map's grid and size. The log
  prints the first tile's PC, tilts and phases and warns when tiles differ.
- **1. Batch processing** uses the processor tab's settings. *Load selected tile in the processor* sets the
  processing up on one raw tile. *Process all tiles* computes one static background from all tiles together
  (when the processor's static background is on) and writes `<tile>_proc.up1` per tile into the output folder,
  header and GUID kept. The IQ range applies per tile; patterns outside it are written unprocessed. NLPAR runs
  per tile with the lambda optimized on the preview tile.
- **2. Batch indexing** uses the indexing tab's settings: one projection center, tilts and pixel size for the
  whole montage (*Copy the project's PC and tilts to the indexing tab* fills them from the first tile). *Load
  selected tile in the indexer* puts one processed tile into tab 3 for a test region. *Index all tiles* indexes
  each processed tile band by band with this tab's live view and writes orientation matrices, cross-correlation
  (`CI`), image quality (`IQ`, EMSphInx's measure; tick *Keep the project's (Hough) IQ* to leave the
  acquisition's), phase and `Valid` flag into a copy of the project, `<project>_sph_bw<bandwidth>.edaxh5` in the
  output folder, tile by tile. All-zero patterns are skipped. Untick *Index the processed files* to index the raw
  files. *Project phase of each master*: blank matches each master to a project phase by name, else by order
  when the counts agree; type the numbers (`2` or `2,1`) to decide yourself. *Resume* (default on) keeps an
  existing output project and skips the tiles already indexed.
- **Cancel batch** stops after the current tile (processing) or band (indexing); what is already written stays.

From a script, `montage.py` does the reading and writing (`python montage.py project.edaxh5 --patterns FOLDER`
prints the tile table); `make_montage_sample.py` cuts a few-MB sample out of a real montage, and
`example_data/montage_sample/` is one such sample.

---

## 6. Command line and scripts

```bash
python ebsd_params.py "scan.ang" --up1 "scan.up1"            # EDAX and EMsoft pattern centers
python ebsd_params.py "scan.edaxh5" --map "Area 1" --px 115
```

The docstring at the top of `ebsd_params.py` lists all options. For indexing from a script, `sph_core.py` wraps
the steps as functions driven by a `Settings` object (see its docstring); the indexing itself is one method of
the kikuchipy fork, `EBSD.spherical_indexing`, documented in the fork's tutorial
(`doc/tutorials/spherical_indexing.ipynb`).

---

## 7. Performance notes and known issues

- **What runs on the GPU**: the coarse Euler-grid cross-correlation. Preprocessing, back-projection and the
  Newton refinement stay on the CPU, which bounds the speed-up; raise `N_WORKERS`.
- **Throughput scales with pattern size and bandwidth.** On the tested laptop: 60 px patterns at bandwidth 68,
  about 1200 patterns/s on the GPU; 479 px patterns at bandwidth 123, about 130 patterns/s on the GPU and about
  28 on 8 CPU workers.
- **Out of memory on the GPU**: set `CHUNKSIZE` to a number, and check that no other process holds VRAM.
- **Large pattern files.** `LAZY = "auto"` memory-maps files larger than a quarter of RAM. The operating system
  may show a lot of "used" memory for mapped file pages; that is cache, not a leak.
- **Old `.up1` files** (header version 1) do not store the map size or step; the GUI takes them from the `.ang`
  (version 3 and later store both). Hexagonal grids are not handled by that fallback.

---

## 8. References and license

- Johan Westraadt's kikuchipy fork with spherical indexing and the GPU backend:
  https://github.com/jwestraadt/kikuchipy. The workbench installs it from https://github.com/helpimlost97/kikuchipy.
- W. C. Lenthe, S. Singh, M. De Graef, "A spherical harmonic transform approach to the indexing of electron
  back-scattered diffraction patterns", Ultramicroscopy 207 (2019). EMSphInx: https://github.com/EMsoft-org/EMSphInx
- kikuchipy: https://kikuchipy.org (H. W. Ånes et al.).
- pyebsdindex: https://pyebsdindex.readthedocs.io (D. Rowenhorst et al.). NLPAR: P. G. Brewick, S. I. Wright,
  D. J. Rowenhorst, "NLPAR: Non-local smoothing for enhanced EBSD pattern indexing", Ultramicroscopy 200 (2019).

Everything in this repository, `pattern_processor/` included, is under the GNU General Public License version 3
(`LICENSE`). The kikuchipy fork it installs is GPLv3 as well, copyright the kikuchipy developers and Johan
Westraadt; pyebsdindex, CuPy and the other dependencies are separate projects under their own licenses.
