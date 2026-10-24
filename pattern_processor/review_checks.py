"""Regression checks for the findings of the 2026-10-24 processor review (CPU only,
offscreen Qt, a few seconds; no data files needed).

    python review_checks.py

Builds a small synthetic .up1 (header grid 12 x 10, 8 full rows + 5 patterns of a partial
row recorded, 16 x 16 px) with three pattern classes of distinct IQ, drives MainWindow
headlessly and checks every finding of processor_review_2026-10-24.md that a script can.
"""
import os, sys, time, importlib.util, tempfile, traceback
os.environ['KPP_GPU'] = '0'                       # CuPy off and pyebsdindex's numba engine
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['KPP_IN_MEMORY'] = '1'
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('gpv', os.path.join(here, 'gpu_pattern_viewer.py'))
gpv = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpv)
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

# Modal boxes would block an offscreen run: record them instead.
boxes = []
for name in ('warning', 'information', 'critical'):
    setattr(gpv.QMessageBox, name, staticmethod(lambda *a, _n=name, **k: boxes.append((_n,) + tuple(a[1:3]))))

NCOLS, NROWS_HDR, PH, PW = 12, 10, 16, 16
N_FULL_ROWS, N_PARTIAL = 8, 5
N_REC = N_FULL_ROWS * NCOLS + N_PARTIAL                 # 101 recorded patterns

def make_scan(path):
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:PH, 0:PW]
    grad = 40.0 * (xx + yy) / (PH + PW - 2)
    pats = np.zeros((N_REC, PH, PW), dtype=np.float64)
    cls = np.zeros(N_REC, dtype=int)                    # 0 normal, 1 medium, 2 flat
    for p in range(N_REC):
        r, c = divmod(p, NCOLS)
        if (r, c) in {(2, 3), (2, 4), (5, 7), (6, 1)}: cls[p] = 2
        elif (r + c) % 7 == 0: cls[p] = 1
    for p in range(N_REC):
        if cls[p] == 2:
            pats[p] = 100.0 + rng.normal(0, 1.0, (PH, PW))
        else:
            s = 5.0 if cls[p] == 1 else 30.0
            pats[p] = 100.0 + grad + rng.normal(0, s, (PH, PW))
    u8 = np.clip(np.round(pats), 0, 255).astype(np.uint8)
    with open(path, 'wb') as f:
        np.asarray([3, PW, PH, 42], dtype=np.uint32).tofile(f)
        np.asarray([0], dtype=np.uint8).tofile(f)
        np.asarray([NCOLS, NROWS_HDR], dtype=np.uint32).tofile(f)
        np.asarray([0], dtype=np.uint8).tofile(f)
        np.asarray([1.0, 1.0], dtype=np.float64).tofile(f)
        u8.tofile(f)
    return u8, cls

def read_out(path):
    with open(path, 'rb') as f:
        f.seek(42); return np.fromfile(f, dtype=np.uint8).reshape(-1, PH, PW)

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

def no_processing(w):
    for chk in (w.static_bg_chk, w.dyn_bg_chk, w.clahe_chk, w.normalize_chk, w.global_chk, w.npar_chk, w.nlpar_chk):
        chk.setChecked(False)
    w.aperture_mode.setCurrentIndex(0)                  # None
    w.bright.setValue(0); w.cont.setValue(0)

tmp = tempfile.mkdtemp(prefix='kpp_check_')
scan = os.path.join(tmp, 'synthetic.up1')
u8, cls = make_scan(scan)
w = gpv.MainWindow()

try:
    # ---- load ----
    w.load_path(scan)
    check("load: truncated file recognised", w.n_actual_pats == N_REC and w.n_actual_rows == N_FULL_ROWS,
          f"{w.n_actual_pats} pats, {w.n_actual_rows} full rows")
    load_mean = w.mean_pattern.copy()
    no_processing(w)

    # ---- finding 1: click in the partial row with NPAR / NLPAR preview on ----
    w.npar_chk.setChecked(True)
    try:
        w.on_pattern_clicked(N_FULL_ROWS, 2); ok = True
    except Exception as e:
        ok = False; traceback.print_exc()
    check("F1: partial-row click with NPAR preview does not raise", ok)
    pats, mask, cr, cc = w._read_neighborhood(w.f_orig, N_FULL_ROWS, 2, 1)
    check("F1: partial-row neighbourhood is the centre alone", pats.shape[:2] == (1, 1) and
          np.array_equal(pats[0, 0], u8[N_FULL_ROWS * NCOLS + 2]))
    w.npar_chk.setChecked(False)
    w.nlpar_chk.setChecked(True)
    t0 = time.time()
    try:
        w.on_pattern_clicked(N_FULL_ROWS, 3); ok = True
    except Exception:
        ok = False; traceback.print_exc()
    check("F1: partial-row click with NLPAR preview does not raise", ok, f"{time.time()-t0:.1f}s incl. sigma map")
    w.nlpar_chk.setChecked(False)
    # a click on a point beyond the recorded patterns (row 9, header only)
    try:
        w.on_pattern_clicked(NROWS_HDR - 1, NCOLS - 1); ok = True
    except Exception:
        ok = False; traceback.print_exc()
    check("F1: click beyond the recorded patterns does not raise", ok)

    # ---- finding 9: a click on an IQ-excluded point reaches the window ----
    w.iq_min_slider.setValue(200)                       # excludes the flat patterns
    excl_rc = (2, 3)
    assert not w.iq_mask[excl_rc], "test setup: (2,3) should be excluded"
    got = []
    w.scan_canvas.pattern_clicked.connect(lambda r, c: got.append((r, c)))
    class Ev: pass
    ev = Ev(); ev.inaxes = w.scan_canvas.ax; ev.xdata = excl_rc[1]; ev.ydata = excl_rc[0]
    w.scan_canvas._on_click(ev)
    check("F9: click on an excluded point is delivered", got == [excl_rc], str(got))

    # ---- finding 3: background recompute on the truncated file ----
    w.iq_min_slider.setValue(0); w.iq_max_slider.setValue(1000)
    w.recompute_background()
    ratio = float(np.mean(w.mean_pattern / load_mean))
    check("F3: recompute with the full IQ range reproduces the load-time mean", abs(ratio - 1.0) < 1e-12, f"ratio {ratio:.6f}")
    check("F3: status counts recorded full rows only", w.bg_status.text() == f"Background: {N_FULL_ROWS*NCOLS}/{N_FULL_ROWS*NCOLS} patterns", w.bg_status.text())
    w.iq_min_slider.setValue(200)
    w.recompute_background()
    m = w._get_iq_mask()[:N_FULL_ROWS].ravel()
    manual = u8[:N_FULL_ROWS * NCOLS][m].astype(np.float64).mean(axis=0)
    check("F3: recompute with a narrowed range equals the manual mean over in-range full-row patterns",
          np.allclose(w.mean_pattern, manual), f"n={int(m.sum())}")

    # ---- finding 2: NPAR box -> NPAR-averaged file, label says so ----
    w.iq_min_slider.setValue(200)
    w.npar_chk.setChecked(True); w.npar_radius_spin.setValue(1)
    out = os.path.join(tmp, 'npar_out.up1')
    r = w.save_processed(out)
    saved = read_out(out)
    check("F2: source label names NPAR", 'NPAR' in r['source_label'], r['source_label'])
    check("F2: output has every recorded pattern", saved.shape[0] == N_REC, str(saved.shape))
    worst = 0.0
    for (rr, cc_) in [(0, 0), (0, 5), (3, 3), (2, 2), (7, 11), (7, 0), (4, 6), (1, 11)]:
        if not w.iq_mask[rr, cc_]: continue
        prev = w.process_pattern(w._compute_npar_single(rr, cc_))
        worst = max(worst, float(np.abs(prev - saved[rr * NCOLS + cc_]).max()))
    check("F2: saved NPAR patterns equal the preview within one grey level", worst <= 1.0, f"max |diff| {worst:.2f}")
    changed = np.any(saved[:N_FULL_ROWS * NCOLS] != u8[:N_FULL_ROWS * NCOLS])
    check("F2: NPAR actually changed the in-range patterns", changed)
    excl_idx = np.flatnonzero(~w.iq_mask[:N_FULL_ROWS].ravel())
    check("F2: excluded patterns are byte-identical to the originals", np.array_equal(saved[excl_idx], u8[excl_idx]), f"{excl_idx.size} excluded")
    check("F2: partial row is the original (no neighbourhood)", np.array_equal(saved[N_FULL_ROWS*NCOLS:], u8[N_FULL_ROWS*NCOLS:]))
    w.npar_chk.setChecked(False)

    # ---- finding 10 / 7: header-only file refused before the window switches ----
    hdr_only = os.path.join(tmp, 'header_only.up1')
    with open(scan, 'rb') as f: hdr = f.read(42)
    with open(hdr_only, 'wb') as f: f.write(hdr)
    boxes.clear()
    w.load_path(hdr_only)
    check("F10: header-only file is refused and the loaded data set stays", w.orig_path == scan and w.iq_map is not None
          and boxes and boxes[-1][0] == 'warning', str(boxes[-1] if boxes else None))
    check("F10: read_patterns_parallel with zero patterns returns", gpv.read_patterns_parallel(scan, 42, 0, PH, PW, np.uint8).shape == (0, PH, PW))

    # ---- finding 7: empty aperture, threshold spin in mode None, sigma busy guard (finding 6) ----
    term = w._static_background_term(True, np.zeros((PH, PW), dtype=bool))
    check("F7: static background term with an empty aperture is finite", np.all(np.isfinite(term)))
    w._ensure_sigma_map(); sig_before = w.sigma_sq_map
    w.aperture_thresh_spin.setValue(w.aperture_thresh_spin.value() + 0.01)
    check("F7: threshold spin in aperture mode None keeps the sigma map", w.sigma_sq_map is sig_before)
    w.sigma_sq_map = None; w._sigma_aperture_key = None
    w._nlpar_busy = True
    try:
        w._ensure_sigma_map()
    finally:
        w._nlpar_busy = False
    check("F6: no sigma map is cached while a batch is busy", w.sigma_sq_map is None)
    w._ensure_sigma_map()
    check("F6: sigma map is recomputed after the batch", w.sigma_sq_map is not None and w.sigma_sq_map.shape == (NROWS_HDR, NCOLS))

    # ---- finding 4 + 5: NLPAR batch, then a narrower IQ range at save time ----
    w.iq_min_slider.setValue(200)                       # flats excluded at batch time
    w.sr_spin.setValue(1); w.lam_spin.setValue(0.8); w._lambda_optimized = True
    t0 = time.time()
    w._apply_nlpar_pyebsdindex(1, 0.8, raise_errors=True)
    check("F4: NLPAR batch in RAM completed", w.f_nlpar is not None and w.nlpar_patterns is not None, f"{time.time()-t0:.1f}s")
    nl = w.nlpar_patterns.copy()
    excl_batch = np.flatnonzero(~w.iq_mask[:N_FULL_ROWS].ravel())
    check("F4: patterns excluded at batch time are the originals in the RAM result", np.array_equal(nl[excl_batch], u8[excl_batch]))
    w.iq_min_slider.setValue(500)                       # now the medium patterns leave the range too
    newly = np.setdiff1d(np.flatnonzero(~w.iq_mask[:N_FULL_ROWS].ravel()), excl_batch)
    check("F4: test setup: some patterns are newly excluded", newly.size > 0, f"{newly.size} newly excluded")
    out2 = os.path.join(tmp, 'nlpar_out.up1')
    r2 = w.save_processed(out2)
    saved2 = read_out(out2)
    check("F4: newly excluded patterns are written as the ORIGINALS, not NLPAR'd", np.array_equal(saved2[newly], u8[newly]))
    kept = np.flatnonzero(w.iq_mask[:N_FULL_ROWS].ravel())
    check("F4: in-range patterns are the NLPAR result", np.array_equal(saved2[kept], nl[kept]))
    check("F4: label is NLPAR", r2['source_label'] == 'NLPAR', r2['source_label'])
    # finding 5: a second allocation frees the first result before the RAM check
    old = w.nlpar_patterns
    ok5 = w._alloc_nlpar_ram()
    check("F5: second NLPAR allocation drops the previous result object", ok5 and w.nlpar_patterns is not old and w.f_nlpar is None)
    w.f_nlpar = None; w.nlpar_patterns = None

    # ---- findings 11, 12, 19: RAM-served reads past the end / across the row cap ----
    ram = gpv.RamUPFile(scan, w.patterns)
    ram.nRows = N_FULL_ROWS
    ram.exclude = np.zeros(N_REC, dtype=bool); ram.exclude[3] = True
    p, xy = ram.read_data(returnArrayOnly=True, patStartCount=[N_REC + 5, 10])
    check("F11: 1-D read past the RAM copy returns no patterns", p.shape[0] == 0 and xy.shape[0] == 0)
    p, xy = ram.read_data(returnArrayOnly=True, patStartCount=[N_FULL_ROWS * NCOLS - 2, 10])
    check("F12: 1-D read across the capped nRows is clamped to the grid", p.shape[0] == 2 and xy.shape[0] == 2)
    p, xy = ram.read_data(returnArrayOnly=True, patStartCount=[0, 5])
    check("F11: exclusion still applied", np.all(p[3] == 1) and not np.all(p[2] == 1))
    masked = gpv.MaskedUPFile(scan); masked.exclude = np.zeros(10, dtype=bool); masked.exclude[1] = True
    p, xy = masked.read_data(returnArrayOnly=True, patStartCount=[0, 20])
    check("F19: MaskedUPFile read beyond the exclude array does not raise", p.shape[0] == 20 and np.all(p[1] == 1))

    # ---- finding 18: canvas mask update before set_map ----
    c = gpv.ScanMapCanvas()
    try:
        c.update_mask(np.ones((2, 2), dtype=bool)); ok = True
    except Exception:
        ok = False
    check("F18: update_mask before set_map does not raise", ok)

    # ---- finding 13: kernel transform tolerant of CRLF / single-space variant ----
    try:
        from pyebsdindex.opencl import openclparam
        cl = open(os.path.join(os.path.dirname(openclparam.__file__), 'clnlpar.cl'), encoding='utf-8', newline='').read()
        lf = cl.replace('\r\n', '\n')
        _, lvl_lf = gpv._transform_clnlpar(lf)
        _, lvl_sp = gpv._transform_clnlpar(lf.replace('mask0 =  (mask0', 'mask0 = (mask0'))
        _, lvl_crlf = gpv._transform_clnlpar(lf.replace('\n', '\r\n').replace('\r\n', '\n'))
        check("F13: transform reaches 'full' on the installed kernel, its single-space variant and a normalised CRLF copy",
              (lvl_lf, lvl_sp, lvl_crlf) == ('full', 'full', 'full'), str((lvl_lf, lvl_sp, lvl_crlf)))
    except Exception as e:
        check("F13: kernel transform check", False, repr(e))

    # ---- finding 8 / unload: nothing of the data set survives ----
    w._ensure_sigma_map(); w._get_nlobj(1)
    check("F8: NLPAR input file object exists before unload", w._nl_infile is not None)
    w._on_aperture_changed()
    check("F8: aperture change drops the NLPAR input file object", w._nl_infile is None and w._nlobj is None)
    w.unload()
    check("unload: window empty", w.f_orig is None and w.patterns is None and w._nl_infile is None)

    # ---- busy guards ----
    w.load_path(scan); no_processing(w)
    boxes.clear(); w._batch_running = True
    w.apply_and_save(); w.optimize_lambda(); w.recompute_background()
    w._batch_running = False
    check("F15: GUI entry points refuse while a batch runs", len(boxes) == 3 and all(b[0] == 'information' for b in boxes), str(boxes))
    boxes.clear(); w._loading = True
    w.load_path(scan)
    w._loading = False
    check("F15: load_path refuses while another load is in flight", len(boxes) == 1 and boxes[0][1] == 'Busy', str(boxes))
finally:
    w.unload()
    import shutil; shutil.rmtree(tmp, ignore_errors=True)

n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
