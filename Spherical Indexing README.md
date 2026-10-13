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
| `kikuchipy/` | (Not in the repository; you create it in step 3.) Clone of https://github.com/jwestraadt/kikuchipy, branch `develop`, pinned at commit `71a1b2f`, plus the local patches below. Installed in editable mode. |
| `kikuchipy_local_patches.patch` | The uncommitted local changes to the fork (GPU memory release, fast lazy loading of `.up1` files). Apply after cloning, see step 3. |
| `build_notebook.py` | Generates the indexing notebook. Edit this, then regenerate; do not hand-edit the generated `.ipynb` unless you know you will not regenerate it. |
| `spherical_indexing_template.ipynb` | Generated notebook with default settings. Runs on kikuchipy's nickel demo data when no pattern file is given. |
| `spherical_indexing_RKD.ipynb`, `preset_RKD.txt` | Example of a notebook generated with a preset file (an RKD data set, detector parallel to the sample). The paths in both are placeholders, see section 6.1. |
| `ebsd_params.py` | Reads acquisition parameters (pattern size, projection center, map size, step) from an EDAX `.ang` or `.edaxh5` file and a `.up1`/`.up2` file. Used by the notebook's first cell and usable from the command line. |
| `sph_benchmark.py` | CPU vs GPU speed and parity benchmark on the nickel demo data set. Use it to verify the install. |
| `test_gpu_memory_fix.py` | Checks that VRAM is released between two GPU runs (tests one of the local patches). |
| `osc_refinement_handoff/` | The notebook's refinement stage (section 11) as a stand-alone script for a CPU cluster, with its own hand-off notes. |
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

Open a terminal in the folder where you want this project (any path; quote paths that
contain spaces). Activate your environment first.

```bash
git clone https://github.com/jwestraadt/kikuchipy.git
cd kikuchipy
git checkout 71a1b2f          # the commit everything here was tested with
```

`develop` keeps moving (the author has since merged further features such as HROSM). The
notebook and scripts in this folder were only run against `71a1b2f`; newer commits may work
but have not been tried.

Apply the local patches (from inside the `kikuchipy` folder, with
`kikuchipy_local_patches.patch` one level up, as in this folder's layout):

```bash
git apply ../kikuchipy_local_patches.patch
git status --short            # should list 4 modified files
```

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

Now install the fork in editable mode, still inside the `kikuchipy` folder:

```bash
pip install -e .
pip install pooch nlopt jupyterlab nbformat psutil ipywidgets
```

The second line adds what the notebook and scripts need beyond the core package: `pooch`
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
first two cells. Every `NAME` must exist in the settings cell, and `value` is Python source,
so paths are written as raw strings. `preset_RKD.txt` is a complete example; a minimal one for
conventional EBSD looks like

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
5. **Section 11, refinement** (optional, CPU only, slow). Refines each orientation against
   patterns simulated from an EMsoft master pattern `.h5` (`MASTER_PATTERN_FILE`; an `.sht`
   file cannot be used here) with a projection center fitted per map point. Expect a few
   patterns per second at `REFINE_BINNING = 2`, so run it on a region first, or use the script
   in `osc_refinement_handoff/` on a cluster.

### 6.5 Reading scan parameters without the notebook

```bash
python ebsd_params.py "scan.ang" --up1 "scan.up1"            # EDAX and EMsoft pattern centers
python ebsd_params.py "scan.edaxh5" --map "Area 1" --px 115
```

The docstring at the top of `ebsd_params.py` lists all options.

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

Signature on the pinned commit:

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
- **Dask version.** Tested with dask 2026.1.2, numba 0.64, numpy 2.3, orix 0.14, hyperspy 2.4.

---

## 9. References

- Fork: https://github.com/jwestraadt/kikuchipy (branch `develop`, commit `71a1b2f`).
  Spherical indexing code is in `src/kikuchipy/indexing/_spherical/`, design notes in `specs/`,
  tutorial in `doc/tutorials/spherical_indexing.ipynb`.
- W. C. Lenthe, S. Singh, M. De Graef, "A spherical harmonic transform approach to the
  indexing of electron back-scattered diffraction patterns", Ultramicroscopy 207 (2019).
- EMSphInx: https://github.com/EMsoft-org/EMSphInx
- kikuchipy documentation: https://kikuchipy.org
