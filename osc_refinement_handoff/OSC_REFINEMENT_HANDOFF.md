# Handoff: orientation refinement with a PC per map point, on OSC

Written 2026-10-06 for a session on a university HPC cluster ("OSC" in file names and text below refers to it).
Goal: run **only the pattern-matching refinement stage** (formerly section 11 of the spherical indexing
notebook; the project's `spherical_refinement_template.ipynb` now does a cheaper projection center drift
correction instead, and this script is the only remaining form of the full refinement) on a
2.3-million-point EBSD map, starting from an EMSphInx result that already exists. Do not build or
run spherical indexing; the starting orientations come from a file.

Files in this handoff:

- `refine_pc_per_point.py`: the complete refinement, extracted from the notebook. Treat it as the
  specification; adapt paths and settings in its SETTINGS block, keep the flow.
- `edax_binary_lazy_load.patch`: optional patch to kikuchipy's `.up1` reader, see section 4.

## 1. The data (all in one folder on the cluster)

| File | What it is |
|---|---|
| `MB6-103 Real Chi_New Sample_Area 1_EBSD+EDS Map 1_NLPAR1-Norm.up1` | patterns, NLPAR-processed, 30.9 GB, EDAX binary, uint8, 115 x 115 px per pattern |
| `MB6-103 Real Chi_New Sample_Area 1_EBSD+EDS Map 1.up1` | the same patterns, unprocessed (not needed) |
| `SphInx_TopRefined.ang` | EMSphInx result = the starting orientations (173 MB) |
| `SphInx_TopRefined.h5` | the same result as h5 (not needed) |
| `Ni-master.h5` | EMsoft master pattern of the phase |
| `MB6-103 Real Chi.edaxh5` | ignore |

From the `.ang` header (read it yourself on OSC, it is plain text):

- Map 2131 columns x 1097 rows = 2,337,707 points, square grid, step 0.5 um.
- Projection center, EDAX convention: x* 0.526609, y* 0.184287, z* 0.696815.
- One phase, Ni, fcc, a = 3.524 A, symmetry 43 (m-3m). The material name field is empty.
- Data columns: phi1, PHI, phi2, x, y, IQ, CI, phase. orix reads them as props `iq`, `ci`.
  EMSphInx writes its cross-correlation score into the CI column.

Geometry (EDAX Quattro camera, conventional EBSD): sample tilt 70 deg, camera elevation 8 deg,
detector pixel 260.87 um (30 mm wide camera at 115 px), no detector twist (`TWIST = 0`).

Beam energy: the script assumes 20 keV. Check which energies `Ni-master.h5` contains
(`kp.load(path, projection="lambert", hemisphere="both")` lists them) and what the scan was run at.

## 2. What the refinement does

This is the hybrid-indexing refinement of kikuchipy
(https://kikuchipy.org/en/stable/tutorials/hybrid_indexing.html) without the Hough and dictionary
parts. Every pattern is matched against patterns simulated from the master pattern, starting from
the EMSphInx orientation, with a PC that varies over the map:

1. **Calibration.** A `PC_GRID` of 5 x 5 blocks is laid over the map. In each block the point with
   the highest score among points inside a grain (all four neighbours within 1 deg) is picked. Those
   25 patterns are refined in orientation and PC together
   (`EBSD.refine_orientation_projection_center`, Nelder-Mead, trust region 2 deg and 0.05 in PC).
   PCs that hit the trust region, ran out of evaluations or lie far from the others are dropped.
   A plane is fitted through the remaining PCs (`EBSDDetector.fit_pc`, affine), which gives a PC for
   every map point. The sample tilt that `fit_pc` estimates is overwritten with the input tilt.
2. **Test region.** A 50 x 50 region in the middle is refined (`EBSD.refine_orientation` with a
   `navigation_mask`) to measure speed and check the result.
3. **Full map.** Same call without the mask. Saves `.h5`, `.ang` and the PC-per-point array.

Sanity checks printed by the script and what to look for:

- Calibration PCs should lie close to a plane, and the measured PC change per scan step should be
  close to the expected one (step size / detector width). If the PCs scatter by more than they
  change over the map, use more calibration patterns (larger `PC_GRID`).
- The direction of the PC shift along scan x should be near 0 deg (that is the twist check).
- Neighbour misorientation inside grains should go down after refinement. The EMSphInx result is
  already refined by EMSphInx, so expect small rotations (tenths of a degree); the gain is mostly
  from the per-point PC.

The priority is small misorientations within grains. The approach (per-point PC, no dictionary
indexing) is a settled decision; do not re-open it.

## 3. Environment: nothing needs to be rebuilt

Refinement is **upstream kikuchipy code** (`refine_orientation`,
`refine_orientation_projection_center`, `EBSDDetector.fit_pc`), not part of Johan Westraadt's
spherical indexing port. The fork installed on OSC from his `develop` branch contains it unchanged,
and any kikuchipy >= 0.11 would work too. It is CPU only: numba kernels that release the GIL,
parallelized by dask's threaded scheduler over chunks of patterns. No CuPy, no GPU.

Needed in the OSC Python env:

- `kikuchipy` (the fork already installed, or `pip install kikuchipy`)
- `nlopt` (`pip install nlopt`); the script uses the `LN_NELDERMEAD` method, which is nlopt's
- the usual `orix`, `hyperspy`, `dask`, `numba`, `h5py`, which come with kikuchipy

Laptop versions for reference: kikuchipy 0.14.dev0 (fork), orix 0.14.1, hyperspy 2.4.0,
dask 2026.1.2, numba 0.64, nlopt 2.11.0, numpy 2.3.5, Python 3.14.

## 4. What matters for speed: cores, not RAM or GPU

- **CPU cores**: the work is one independent optimisation per pattern, spread over
  `dask.config.set(num_workers=N)` threads. Set `N` to the cores of the node
  (the script reads `SLURM_CPUS_PER_TASK`). Ask for a whole node.
- **RAM**: modest. The lazy signal streams pattern chunks; per worker there is one chunk of
  115 x 115 uint8 patterns, the simulated pattern and direction cosines. The master pattern
  (210 MB file) is held once. The 2.3 M-point crystal maps and the per-point PC array
  (1097 x 2131 x 3 float64, 56 MB) are small.
- **GPU**: not used at all.

Rough expectation, to be replaced by the test-region measurement: on the laptop, refinement ran
at about 5 patterns/s with 8 threads on 239 x 239 px patterns. Cost per pattern scales with the
number of pixels, so 115 x 115 px should be about 4 x cheaper, i.e. roughly 2 to 3 patterns/s per
core. For 2.34 M patterns that is on the order of 250 core-hours: a few hours on a 96-core node,
over a day on the laptop. Always run `python refine_pc_per_point.py test` first; it prints the
measured speed and the projected time for the full map, then run `full` as a batch job.

`REFINE_MAXEVAL` (300) caps the simulated patterns per point; the script prints the mean and max
used, so if the mean is far below the cap the time estimate holds.

## 5. Loading the 30.9 GB pattern file

Two options; both give the same result.

- **Lazy (`LAZY = True`)**, recommended if the node has less than about 80 GB free. Upstream
  kikuchipy's `.up1` reader is slow for lazy loading of big files (it hashes and then copies the
  whole memory map, which on the laptop took minutes). `edax_binary_lazy_load.patch` fixes that in
  `src/kikuchipy/io/plugins/edax_binary/_api.py` (names the dask array from the file's path, size
  and mtime, and wraps the memmap so dask slices it instead of copying). Apply it with
  `git apply` in the kikuchipy clone on OSC, or make the equivalent change by hand. Measured on
  the laptop after the patch: a 32 GB file loads in 3 to 7 s.
- **Not lazy (`LAZY = False`)** on a big-memory node: kikuchipy returns a memory map, and dask's
  `from_array` copies it into RAM at the first refinement call (dask `array/core.py`, the
  `x.copy()` line in `from_array`). Peak is about twice the file size, roughly 62 GB. No patch
  needed.

The `.up1` is version 4, which stores the map size, so `kp.load` gets the 1097 x 2131 shape
from the file. If it does not, the script falls back to the shape of the `.ang`.

Use the NLPAR-processed file and leave both background removals off; "Norm" means it is already
processed. If the raw file is used instead, turn `REMOVE_STATIC_BACKGROUND` and
`REMOVE_DYNAMIC_BACKGROUND` on.

## 6. Running it

```bash
# interactive or a short job: calibration + test region, prints speed and projected time
python refine_pc_per_point.py test

# batch job on a whole node: calibration + full map + save
python refine_pc_per_point.py full
```

Outputs next to the data (or `OUTPUT_DIR`):

- `<OUTPUT_NAME>_bin1.h5`, `<OUTPUT_NAME>_bin1.ang`: refined map with props `scores`
  (refinement NCC), `num_evals`, `iq` and `scores_indexed` (the EMSphInx CI) carried over.
  The `.ang` gets `iq` as IQ and the refinement score as CI.
- `<OUTPUT_NAME>_bin1_pc.npy`: the PC per map point, Bruker convention.

The calibration is recomputed in the `full` run (it takes a minute or two), so the two stages are
independent jobs.

## 7. Pitfalls already met on the laptop

- Sliced orix `CrystalMap`s do not work as `xmap` input to kikuchipy refinement; that is why the
  test region is selected with a `navigation_mask` on the full map, not by slicing.
- `fit_pc` overwrites `sample_tilt` on the returned detector; the script resets it.
- `refine_orientation_projection_center` is called with `chunk_kwargs=dict(chunk_shape=1)` so
  that 25 calibration patterns use 25 workers instead of one.
- Binning (`REFINE_BINNING`) is done with kikuchipy's own `isig` crop, `EBSDDetector.crop` and
  `downsample`. Do not write custom functions where kikuchipy has one. At 115 px keep 1.
