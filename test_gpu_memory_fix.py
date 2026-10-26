"""Test that GPU memory is released after a spherical indexing run.

Checks the cuFFT plan-cache fix of the fork's workbench commit (README section 3): without it, several GB
of VRAM stay allocated after a GPU run and the next run picks a tiny automatic batch size. Runs the nickel
demo dataset twice on the GPU and checks that

1. the free VRAM after each run is back to about what it was at the start (within TOLERANCE_GB),
2. the automatic batch size of the second run equals that of the first,
3. both runs give the same orientations.

Usage:
    python test_gpu_memory_fix.py            # bandwidth 123
    python test_gpu_memory_fix.py 68         # another bandwidth

Needs CuPy and a GPU. Prints the free VRAM and batch size before and after each run and ends with
"RESULT: PASS" (exit code 0) or "RESULT: FAIL" (exit code 1). Writes no files.
"""

import sys
import time

import cupy
import dask

import kikuchipy as kp
from kikuchipy.indexing._spherical import _gpu

BANDWIDTH = int(sys.argv[1]) if len(sys.argv) > 1 else 123
TOLERANCE_GB = 0.5


def free_gb():
    """Free VRAM of device 0 in GB (1e9 bytes), as the CUDA runtime reports it."""
    return cupy.cuda.runtime.memGetInfo()[0] / 1e9


def auto_batch():
    """The device batch size the indexer would pick now for BANDWIDTH from the free VRAM.

    Uses the fork's private ``_gpu._default_batch_size(bandwidth, free_bytes)`` (a pure function of the VRAM
    model) with the current free VRAM, so a run that leaks VRAM shows up as a smaller batch size.
    """
    return _gpu._default_batch_size(BANDWIDTH, int(cupy.cuda.runtime.memGetInfo()[0]))


dask.config.set(num_workers=8)
print("kikuchipy", kp.__version__, "from", kp.__file__)
print("GPU:", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())

s = kp.data.nickel_ebsd_large(allow_download=True)
s.remove_static_background()
s.remove_dynamic_background()
mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", hemisphere="both")
harmonics = mp.get_spherical_harmonics(bandwidth=188)
# The demo detector carries a PC per pattern; the indexer needs one PC for the whole map.
det = s.detector.deepcopy()
det.pc = det.pc_average

# Touch the device once, so that the CUDA context itself is not counted as a leak
cupy.zeros(1).sum()
cupy.get_default_memory_pool().free_all_blocks()

free_start = free_gb()
print(f"\nBandwidth {BANDWIDTH}")
print(f"Free VRAM at start:    {free_start:.2f} GB, automatic batch size {auto_batch()}")

batch_sizes = []
xmaps = []
for run in (1, 2):
    batch = auto_batch()
    t0 = time.perf_counter()
    xmap = s.spherical_indexing(harmonics, det, bandwidth=BANDWIDTH, backend="gpu", verbose=1)
    dt = time.perf_counter() - t0
    free_after = free_gb()
    batch_sizes.append(batch)
    xmaps.append(xmap)
    print(
        f"Run {run}: batch size {batch}, {xmap.size / dt:.0f} patterns/s, "
        f"free VRAM after the run {free_after:.2f} GB"
    )

held = free_start - free_after
same_batch = batch_sizes[0] == batch_sizes[1]
angle = xmaps[0].orientations.angle_with(xmaps[1].orientations, degrees=True)
same_result = float(angle.max()) < 1e-6

print()
print(f"VRAM still held after the runs: {held:.2f} GB (tolerance {TOLERANCE_GB} GB)")
print(f"Same automatic batch size in both runs: {same_batch}")
print(f"Same orientations in both runs: {same_result} (max difference {angle.max():.2e} deg)")
passed = held < TOLERANCE_GB and same_batch and same_result
print("\nRESULT:", "PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
