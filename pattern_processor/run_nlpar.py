"""Headless NLPAR runner via pyebsdindex.

Runs opt_lambda then calcnlpar on a single UP1 file. Output is written next
to the input as <input>_NLPAR_l<lam>sr<sr>.up1 by pyebsdindex.

Edit the path below to point at your data.
"""
from pyebsdindex import nlpar
import time
from processor_defaults import DEFAULTS   # the viewer's defaults (run this script from its folder)
try:
    import psutil
    # pyebsdindex's _calcchunks assumes 8 B/px (float32 in+out); its default of
    # sysram/4 asks for multi-GB tiles on large scans. The 4 GB cap (float32 tiles <= 2 GB)
    # is this script's own, tighter than the viewer's DEFAULTS.chunk_cap_bytes: it has no
    # progress hook, so a smaller tile keeps the console moving.
    CHUNK_BYTES = int(min(4e9, DEFAULTS.ram_chunk_fraction * psutil.virtual_memory().available))
except Exception:
    CHUNK_BYTES = int(2e9)

# ============ EDIT THIS ============
INPUT_PATH = r'path/to/your/original.up1'
SEARCH_RADIUS = DEFAULTS.search_radius
INITIAL_LAMBDA = DEFAULTS.lam            # opt_lambda replaces it below
# ===================================

print("Initializing NLPAR...")
nlobj = nlpar.NLPAR(INPUT_PATH, lam=INITIAL_LAMBDA, searchradius=SEARCH_RADIUS)

print("Optimizing lambda...")
t0 = time.time()
nlobj.opt_lambda(chunksize=CHUNK_BYTES, automask=True, autoupdate=True, backsub=False)
print(f"Lambda optimization took {time.time() - t0:.1f}s")

print(f"\nUsing lambda = {nlobj.lam}")
print(f"Search radius = {nlobj.searchradius}")

print("\nRunning NLPAR...")
t0 = time.time()
nlobj.calcnlpar(chunksize=CHUNK_BYTES, saturation_protect=DEFAULTS.saturation_protect, automask=True, backsub=False)
print(f"NLPAR took {time.time() - t0:.1f}s")

print("\nDone.")
