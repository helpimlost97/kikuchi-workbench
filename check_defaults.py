"""Check that every GUI default comes from its owner: the indexing form, the point tab and the pattern processor
are built headless, and each widget's start value is compared with the object that owns the default
(``sph_core.Settings`` for the indexing and point tabs, ``pattern_processor.processor_defaults.DEFAULTS`` for the
processor). Prints the differences and ends with ``RESULT: PASS`` or ``RESULT: FAIL`` (exit code 1).

    python check_defaults.py

Runs on the CPU in a few seconds (Qt offscreen, ``KPP_GPU=0``), writes no files, needs no data.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KPP_GPU", "0")
sys.argv = [sys.argv[0]]  # kikuchi_workbench parses the command line at import

from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

import sph_core  # noqa: E402
import kikuchi_workbench as kw  # noqa: E402

failures: list[str] = []


def compare(what: str, got: dict, owner) -> None:
    """Record every ``name`` whose form value ``got[name]`` differs from ``getattr(owner, name)``."""
    for name, value in got.items():
        want = getattr(owner, name)
        if value != want:
            failures.append(f"{what}: {name} starts at {value!r}, its owner says {want!r}")


wb = kw.Workbench()
base = sph_core.Settings()

# 1. the indexing tab: every Settings field the form has (after the LAZY / PC normalisation of Workbench.settings)
st = wb.settings()
compare("indexing form", {n: getattr(st, n) for n in base.names() if wb.form.has(n)}, base)
missing = [n for n in base.names() if not wb.form.has(n)]
print(f"indexing form: {len(wb.form._fields)} fields checked; Settings fields without a widget: {missing or 'none'}")

# 2. the point tab: its fields start at the indexing tab's values, which start at the Settings defaults
pt = wb.point_tab.form.settings(base)
compare("point tab", {n: getattr(pt, n) for n in base.names() if wb.point_tab.form.has(n)}, base)
print(f"point tab: {len(wb.point_tab.form._fields)} fields checked")

# 3. the pattern processor, when it was found
try:
    from pattern_processor.processor_defaults import DEFAULTS
except ImportError:  # the processor has no defaults module yet
    DEFAULTS = None
if wb.proc is not None and DEFAULTS is not None:
    p = wb.proc
    got = {
        "aperture_mode": p.aperture_mode.currentText(),
        "aperture_threshold": p.aperture_thresh_spin.value(),
        "iq_range": (p.iq_min_slider.value(), p.iq_max_slider.value()),
        "npar_radius": p.npar_radius_spin.value(),
        "search_radius": p.sr_spin.value(),
        "lam": p.lam_spin.value(),
        "saturation_protect": p.sat_chk.isChecked(),
        "dthresh": p.dthresh_spin.value(),
        "static_bg": p.static_bg_chk.isChecked(),
        "bg_mode": p.bg_mode.currentText(),
        "dyn_bg": p.dyn_bg_chk.isChecked(),
        "dyn_sigma": p.dyn_sigma.value() / DEFAULTS.sigma_slider_scale,
        "per_pattern_stretch": p.normalize_chk.isChecked(),
        "global_stretch": p.global_chk.isChecked(),
        "brightness": p.bright.value(),
        "contrast": p.cont.value(),
        "clahe": p.clahe_chk.isChecked(),
        "clahe_clip": p.clahe_clip.value() / DEFAULTS.clip_slider_scale,
    }
    compare("pattern processor", got, DEFAULTS)
    print(f"pattern processor: {len(got)} widgets checked")
else:
    print("pattern processor: " + ("not loaded" if wb.proc is None else "no processor_defaults module") + ", skipped")

for f in failures:
    print("  " + f)
print("\nRESULT:", "FAIL" if failures else "PASS")
sys.exit(1 if failures else 0)
