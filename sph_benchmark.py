"""Benchmark kikuchipy spherical indexing (jwestraadt fork) on the nickel dataset.

Usage:
  python sph_benchmark.py --backend cpu|gpu|both [--workers N] [--repeat N]
                          [--chunksize N] [--no-refine] [--compare] [--bandwidth 68]
"""
import argparse, time, sys
import numpy as np
import dask
import kikuchipy as kp

p = argparse.ArgumentParser()
p.add_argument("--backend", default="both", choices=["cpu", "gpu", "both"])
p.add_argument("--workers", type=int, default=8)
p.add_argument("--repeat", type=int, default=1, help="best-of-N timing")
p.add_argument("--chunksize", type=int, default=None)
p.add_argument("--bandwidth", type=int, default=68)
p.add_argument("--no-refine", action="store_true")
p.add_argument("--compare", action="store_true", help="CPU vs GPU misorientation check")
a = p.parse_args()

refine = not a.no_refine
dask.config.set(num_workers=a.workers)

print("Loading nickel_ebsd_large ...", flush=True)
s = kp.data.nickel_ebsd_large(allow_download=True)
s.remove_static_background()
s.remove_dynamic_background()
n_pat = int(np.prod(s.axes_manager.navigation_shape))
print(f"  {s.axes_manager.navigation_shape} nav, {s.axes_manager.signal_shape} sig, {n_pat} patterns")

mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
t0 = time.perf_counter()
harm = mp.get_spherical_harmonics(bandwidth=188)
print(f"  harmonics (bw 188) in {time.perf_counter()-t0:.1f} s")
det = s.detector.deepcopy(); det.pc = det.pc_average

backends = ["cpu", "gpu"] if a.backend == "both" else [a.backend]
if a.compare and a.backend != "both":
    backends = ["cpu", "gpu"]

results = {}
for be in backends:
    kw = dict(bandwidth=a.bandwidth, refine=refine, backend=be, verbose=0)
    if a.chunksize: kw["chunksize"] = a.chunksize
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
    for name in ("scores", "score"):
        if name in xc.prop and name in xg.prop:
            d = np.abs(np.asarray(xc.prop[name]) - np.asarray(xg.prop[name]))
            print(f"  {name} max abs diff: {d.max():.3e}")
            break
