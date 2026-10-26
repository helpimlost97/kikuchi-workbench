"""CPU vs GPU speed and parity benchmark of the fork's spherical indexing on the nickel demo data set.

Run it after the install (README section 5) to see the throughput of each backend and to check that the GPU
gives the same orientations as the CPU. The data set is kikuchipy's ``nickel_ebsd_large`` (4125 patterns of
60 x 60 px, downloaded once by pooch) with its small master pattern; both background removals are applied
first, as in the fork's tutorial.

    python sph_benchmark.py --backend cpu|gpu|both [--workers N] [--repeat N]
                            [--chunksize N] [--no-refine] [--compare] [--bandwidth 68]

--backend     which backend(s) to time (default: both). --compare forces both.
--workers     Dask threads (default: sph_core.Settings.N_WORKERS); raise to about the number of performance cores.
--repeat      index the full map this many times per backend and report the best time (default: 1).
--chunksize   patterns per Dask chunk and, on the GPU, the device batch size (default: Settings.CHUNKSIZE, automatic).
--bandwidth   spherical harmonic bandwidth of the indexing (default: Settings.BANDWIDTH, the EMSphInx default 68).
--no-refine   skip the Newton refinement, so only the coarse grid search is timed.
--compare     index on both backends and print the CPU vs GPU misorientation and score difference.

Every backend is first run on ten patterns so that Numba compilation and CuPy kernel loading are not part of
the timing. Prints the timing of each run and, per backend, "BEST: <s> = <patterns/s>"; writes no files.
"""
import argparse
import time

import numpy as np
import dask

import sph_core
from sph_core import Settings

p = argparse.ArgumentParser()
p.add_argument("--backend", default="both", choices=list(sph_core.BACKENDS) + ["both"])
p.add_argument("--workers", type=int, default=Settings.N_WORKERS)
p.add_argument("--repeat", type=int, default=1, help="best-of-N timing")
p.add_argument("--chunksize", type=int, default=Settings.CHUNKSIZE)
p.add_argument("--bandwidth", type=int, default=Settings.BANDWIDTH)
p.add_argument("--no-refine", action="store_true")
p.add_argument("--compare", action="store_true", help="CPU vs GPU misorientation check")
a = p.parse_args()

refine = not a.no_refine
dask.config.set(num_workers=a.workers)

# The nickel demo with both backgrounds removed, its harmonics (computed once at sph_core.DEMO_HARMONICS_BANDWIDTH;
# the indexer resizes them down to --bandwidth) and its detector with one PC for the whole map.
s, harm, det = sph_core.demo_setup()
n_pat = int(np.prod(s.axes_manager.navigation_shape))
print(f"  {s.axes_manager.navigation_shape} nav, {s.axes_manager.signal_shape} sig, {n_pat} patterns")

backends = ["cpu", "gpu"] if a.backend == "both" else [a.backend]
if a.compare and a.backend != "both":
    backends = ["cpu", "gpu"]

results = {}
for be in backends:
    kw = dict(bandwidth=a.bandwidth, refine=refine, backend=be, verbose=0)
    if a.chunksize:
        kw["chunksize"] = a.chunksize
    # warm-up (numba JIT / cupy kernels) on a small slice
    t0 = time.perf_counter()
    s.inav[:10, :1].spherical_indexing(harm, det, **kw)
    print(f"[{be}] warm-up {time.perf_counter()-t0:.1f} s", flush=True)
    best = None
    for r in range(a.repeat):
        t0 = time.perf_counter()
        xmap = s.spherical_indexing(harm, det, **kw)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
        print(f"[{be}] run {r+1}: {dt:.2f} s  -> {n_pat/dt:.0f} pat/s", flush=True)
    print(f"[{be}] BEST: {best:.2f} s = {n_pat/best:.0f} pat/s  "
          f"(workers={a.workers}, bw={a.bandwidth}, refine={refine})", flush=True)
    results[be] = xmap

if "gpu" in results:
    import cupy
    print("GPU:", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())

if a.compare and "cpu" in results and "gpu" in results:
    xc, xg = results["cpu"], results["gpu"]
    ang = xc.orientations.angle_with(xg.orientations, degrees=True)  # symmetry-reduced
    print(f"CPU vs GPU misorientation: median {np.median(ang):.4f} deg, "
          f"max {ang.max():.4f} deg, >0.01 deg: {(ang > 0.01).sum()} / {ang.size}")
    # The cross-correlation score is stored under whichever of these names the fork's version uses
    for name in ("scores", "score"):
        if name in xc.prop and name in xg.prop:
            d = np.abs(np.asarray(xc.prop[name]) - np.asarray(xg.prop[name]))
            print(f"  {name} max abs diff: {d.max():.3e}")
            break
