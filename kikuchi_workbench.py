#!/usr/bin/env python3
"""Kikuchi workbench: the Kikuchi Pattern Processor, GPU spherical indexing and HROSM in one window.

    python kikuchi_workbench.py                              # finds the pattern processor next to this folder
    python kikuchi_workbench.py --processor-dir C:\\path\\to\\kikuchi-pattern-processor
    python kikuchi_workbench.py --preset preset_RKD.txt      # start with the settings of a notebook preset
    set KPP_DIR=C:\\path\\to\\kikuchi-pattern-processor       # alternative to --processor-dir (Windows cmd)

Three tabs:

1. **Pattern processor**: the unchanged Kikuchi Pattern Processor (gpu_pattern_viewer.py of
   https://github.com/helpimlost97/kikuchi-pattern-processor), imported from its own folder. Load a UP1/UP2,
   set the aperture, IQ range, NLPAR and the processing chain as usual. "Apply && Save" still writes a file; the
   saved file is then offered to the indexing tab automatically.
2. **Spherical indexing**: the settings of spherical_indexing_template.ipynb as a form. The patterns come either
   from a file (anything kikuchipy loads) or straight from the processor tab: the processor's processing chain
   (NLPAR result, background, stretch, CLAHE, IQ exclusion) is run on the whole data set in RAM, exactly as
   "Apply && Save" would write it, and the result is indexed without touching the disk. Check the geometry on one
   pattern, index a test region, index the full map, save .ang/.h5. Load and save notebook presets.
3. **HROSM**: re-index every grain of the last indexed map on a fine misorientation ball (EBSD.hrosm of the
   fork), with the KAM, grain, GROD and OSM maps, and save the result.

Needs the environment of README.md (the kikuchipy fork, CuPy for the GPU) plus the pattern processor's packages
(PyQt5, pyebsdindex, scipy, scikit-image). Without the processor folder the indexing and HROSM tabs still work
from files.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------------------------
# The pattern processor: found and imported before Qt is used, since its module prints what it found
# ----------------------------------------------------------------------------------------------


def find_processor_dir(explicit: str | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("KPP_DIR"):
        candidates.append(Path(os.environ["KPP_DIR"]))
    for base in (HERE, HERE.parent, Path.cwd(), Path.home()):
        candidates.append(base / "kikuchi-pattern-processor")
        candidates.append(base / "kikuchi_pattern_processor")
    candidates.append(HERE)
    for c in candidates:
        if (c / "gpu_pattern_viewer.py").is_file():
            return c.resolve()
    return None


def import_processor(directory: Path | None):
    """The gpu_pattern_viewer module, or (None, reason)."""
    if directory is None:
        return None, ("kikuchi-pattern-processor not found: put it next to this folder, or pass --processor-dir / "
                      "set KPP_DIR")
    sys.path.insert(0, str(directory))
    try:
        import gpu_pattern_viewer  # noqa: F401

        return gpu_pattern_viewer, f"{directory}"
    except Exception as e:  # noqa: BLE001
        return None, f"could not import gpu_pattern_viewer.py from {directory}: {type(e).__name__}: {e}"


_args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_args.add_argument("--processor-dir", default=None, help="folder of kikuchi-pattern-processor (gpu_pattern_viewer.py)")
_args.add_argument("--preset", default=None, help="notebook preset file (NAME = value lines) to start with")
_args.add_argument("--no-processor", action="store_true", help="do not load the pattern processor tab")
ARGS, _ = _args.parse_known_args()

if ARGS.no_processor:
    gpv, PROCESSOR_INFO = None, "pattern processor tab disabled (--no-processor)"
else:
    gpv, PROCESSOR_INFO = import_processor(find_processor_dir(ARGS.processor_dir))

from PyQt5.QtCore import Qt, QThread, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import ebsd_params  # noqa: E402
import sph_core  # noqa: E402
from sph_core import Settings  # noqa: E402

__version__ = "0.1.0"


# ----------------------------------------------------------------------------------------------
# Worker thread with its console output routed to the log
# ----------------------------------------------------------------------------------------------


class _Tee:
    """Writes to the original stream and hands complete lines (and \\r progress updates) to a callback."""

    def __init__(self, original, on_line, on_progress):
        self.original = original
        self.on_line = on_line
        self.on_progress = on_progress
        self.buffer = ""

    def write(self, text):
        try:
            self.original.write(text)
        except Exception:  # noqa: BLE001
            pass
        self.buffer += text
        while True:
            i_n, i_r = self.buffer.find("\n"), self.buffer.find("\r")
            if i_n == -1 and i_r == -1:
                break
            if i_r != -1 and (i_n == -1 or i_r < i_n):
                line, self.buffer = self.buffer[:i_r], self.buffer[i_r + 1 :]
                if line.strip():
                    self.on_progress(line.rstrip())
            else:
                line, self.buffer = self.buffer[:i_n], self.buffer[i_n + 1 :]
                if line.strip():
                    self.on_line(line.rstrip())

    def flush(self):
        try:
            self.original.flush()
        except Exception:  # noqa: BLE001
            pass

    def __getattr__(self, name):
        return getattr(self.original, name)


class Worker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(orig_out, self.log.emit, self.progress.emit)
        sys.stderr = _Tee(orig_err, self.log.emit, self.progress.emit)
        try:
            result = self.fn(self.log.emit)
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
        else:
            self.done.emit(result)
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err


# ----------------------------------------------------------------------------------------------
# Settings form helpers
# ----------------------------------------------------------------------------------------------


def _literal(text: str, default=None):
    text = text.strip()
    if text == "":
        return default
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text  # a bare path or word


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return repr(value)


class SettingsForm(QWidget):
    """Widgets for a subset of the Settings fields, in groups. settings()/set_settings() convert both ways."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields: dict[str, tuple] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._form = None

    def group(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        self._form = QFormLayout(box)
        self._form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self._layout.addWidget(box)
        return self._form

    def finish(self):
        self._layout.addStretch()

    def add_widget(self, widget):
        self._form.addRow(widget)

    # one line of a group per setting
    def add_bool(self, name, label, tip="", default=False):
        w = QCheckBox(label)
        w.setChecked(bool(default))
        w.setToolTip(tip)
        self._form.addRow(w)
        self._fields[name] = (w.isChecked, w.setChecked)
        return w

    def add_int(self, name, label, default, lo=0, hi=100000, tip=""):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(int(default))
        w.setToolTip(tip)
        self._form.addRow(label, w)
        self._fields[name] = (w.value, lambda v: w.setValue(int(v)))
        return w

    def add_float(self, name, label, default, lo=-1e9, hi=1e9, decimals=4, tip=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(float(default))
        w.setToolTip(tip)
        self._form.addRow(label, w)
        self._fields[name] = (w.value, lambda v: w.setValue(float(v)))
        return w

    def add_text(self, name, label, default=None, tip="", placeholder=""):
        """Free text, parsed as a Python literal (blank = None): numbers, tuples such as (0, 100, 0, 50), paths."""
        w = QLineEdit(_to_text(default))
        w.setToolTip(tip)
        w.setPlaceholderText(placeholder)
        self._form.addRow(label, w)
        self._fields[name] = (lambda: _literal(w.text()), lambda v: w.setText(_to_text(v)))
        return w

    def add_combo(self, name, label, options, default, tip=""):
        w = QComboBox()
        w.addItems([str(o) for o in options])
        w.setCurrentText(str(default))
        w.setToolTip(tip)
        self._form.addRow(label, w)
        self._fields[name] = (lambda: type(options[0])(w.currentText()) if options and not isinstance(options[0], str) else w.currentText(),
                              lambda v: w.setCurrentText(str(v)))
        return w

    def add_path(self, name, label, default=None, tip="", filter="All files (*)", save=False, directory=False, multi=False):
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(_to_text(default))
        edit.setToolTip(tip)
        btn = QPushButton("...")
        btn.setFixedWidth(28)

        def browse():
            start = edit.text().strip() or str(Path.cwd())
            if directory:
                path = QFileDialog.getExistingDirectory(self, label, start)
            elif multi:
                paths, _ = QFileDialog.getOpenFileNames(self, label, start, filter)
                path = repr(paths) if len(paths) > 1 else (paths[0] if paths else "")
            elif save:
                path, _ = QFileDialog.getSaveFileName(self, label, start, filter)
            else:
                path, _ = QFileDialog.getOpenFileName(self, label, start, filter)
            if path:
                edit.setText(path)

        btn.clicked.connect(browse)
        lay.addWidget(edit)
        lay.addWidget(btn)
        self._form.addRow(label, row)

        def get():
            text = edit.text().strip()
            if text == "":
                return None
            if text.startswith("["):
                return _literal(text)
            return text

        self._fields[name] = (get, lambda v: edit.setText(_to_text(v)))
        return edit

    def settings(self, base: Settings) -> Settings:
        st = dataclasses.replace(base)
        for name, (get, _) in self._fields.items():
            setattr(st, name, get())
        return st

    def set_settings(self, st: Settings):
        for name, (_, set_) in self._fields.items():
            set_(getattr(st, name))

    def has(self, name):
        return name in self._fields


# ----------------------------------------------------------------------------------------------
# Patterns taken from the processor tab, in RAM
# ----------------------------------------------------------------------------------------------


def processor_dataset_key(proc) -> tuple | None:
    """Identifies the processor's loaded data set and its processing state, to know when to re-run the chain."""
    if proc is None or proc.f_orig is None:
        return None
    return (
        proc.orig_path, id(proc.f_nlpar), id(proc.nlpar_patterns),
        proc.static_bg_chk.isChecked(), proc.bg_mode.currentText(), proc.dyn_bg_chk.isChecked(),
        proc.dyn_sigma.value(), proc.normalize_chk.isChecked(), proc.global_chk.isChecked(),
        proc.bright.value(), proc.cont.value(), proc.clahe_chk.isChecked(), proc.clahe_clip.value(),
        proc.npar_chk.isChecked(), proc.nlpar_chk.isChecked(), proc.aperture_mode.currentText(),
        proc.iq_min_slider.value(), proc.iq_max_slider.value(),
    )


def processed_patterns_from_processor(proc, status) -> tuple[np.ndarray, np.ndarray, dict]:
    """The processor's processing chain on its whole data set, as 'Apply && Save' would write it, in RAM.

    Returns (patterns (n, ph, pw) in the file's dtype, IQ mask (rows, cols) with True = kept, info dict).
    Runs in the GUI thread, since the chain reads the processor's controls and uses its GPU code; ``status``
    is called with progress text and may pump the event loop.
    """
    if proc.f_orig is None:
        raise ValueError("Load a UP1/UP2 file in the pattern processor tab first")
    if proc.nlpar_chk.isChecked() and proc.f_nlpar is None:
        raise ValueError("NLPAR preview is on but NLPAR has not been applied to the full data set: click "
                         "'Apply NLPAR to Full Dataset' in the processor tab first (or untick 'Enable Preview')")
    ph, pw = int(proc.f_orig.patternH), int(proc.f_orig.patternW)
    dtype = np.dtype(proc.f_orig.filedatatype)
    ncols, nrows = int(proc.ncols), int(proc.nrows)
    n_out = int(getattr(proc, "n_actual_pats", nrows * ncols))
    n_rows_out = (n_out + ncols - 1) // ncols
    n_full_rows = int(getattr(proc, "n_actual_rows", nrows))
    need = n_out * ph * pw * dtype.itemsize
    avail = gpv.available_ram()
    if need > 0.8 * avail:
        raise MemoryError(
            f"The processed data set needs {need / 1e9:.1f} GB of RAM but {avail / 1e9:.1f} GB are available: "
            "use 'Apply && Save' in the processor tab and index the saved file instead (pattern source = file)"
        )
    src_fobj = proc.f_nlpar if proc.f_nlpar is not None else proc.f_orig
    source = "NLPAR" if proc.f_nlpar is not None else ("NPAR" if proc.npar_chk.isChecked() else "original")
    mask = proc._get_iq_mask()
    mask_flat = mask.reshape(-1)
    # chunk size: the processor's own rule for 'Apply && Save'
    if gpv.HAS_GPU:
        import cupy as cp

        mem_per_row = 3 * ncols * ph * pw * 4
        chunk_rows = max(2, int(cp.cuda.Device(0).mem_info[1] * 0.5) // mem_per_row)
        chunk_rows = min(chunk_rows, nrows, proc._host_chunk_rows(33), proc._host_chunk_rows(18))
    else:
        chunk_rows = min(nrows, proc._host_chunk_rows(42))
    chunk_rows = max(1, int(chunk_rows))
    out = np.empty((n_out, ph, pw), dtype=dtype)
    dtype_max = float(proc.dtype_max)
    t0 = time.time()
    for row_start in range(0, n_rows_out, chunk_rows):
        rows = min(chunk_rows, n_rows_out - row_start)
        p0 = row_start * ncols
        p1 = min((row_start + rows) * ncols, n_out)
        if p1 <= n_full_rows * ncols:
            chunk = proc._read_rows(src_fobj, row_start, rows)
        else:
            parts = []
            if p0 < n_full_rows * ncols:
                parts.append(proc._read_rows(src_fobj, row_start, n_full_rows - row_start))
            q0 = max(p0, n_full_rows * ncols)
            parts.append(proc._read_patterns(proc.f_orig, q0, p1 - q0))
            chunk = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        mask_rows = mask_flat[p0:p1]

        def chunk_progress(done, n_in_chunk, _r0=row_start, _rows=rows):
            frac = _r0 + _rows * done / max(n_in_chunk, 1)
            status(f"Processing patterns in RAM: row {frac:.0f}/{n_rows_out} ({time.time() - t0:.0f} s)")

        result = proc._process_chunk_batch(chunk.astype(np.float64), mask_rows, progress=chunk_progress)
        # the same conversion as pyebsdindex's write_data (method "clip"): clip, then truncate to the integer type
        out[p0:p1] = np.clip(result, 0, dtype_max).astype(dtype)
        del chunk, result
        status(f"Processing patterns in RAM: row {row_start + rows}/{n_rows_out} ({time.time() - t0:.0f} s)")
    info = {
        "source": source, "n_patterns": n_out, "nav_shape": (nrows, ncols), "sig_shape": (ph, pw),
        "step": (float(getattr(proc.f_orig, "xStep", 0) or 0), float(getattr(proc.f_orig, "yStep", 0) or 0)),
        "seconds": time.time() - t0, "excluded": int((~mask).sum()), "file": proc.orig_path,
    }
    return out, mask, info


# ----------------------------------------------------------------------------------------------
# Map plots
# ----------------------------------------------------------------------------------------------


class MapCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(9, 4), facecolor="w")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def show_panels(self, panels, title=""):
        """panels: list of (title, 2D array or RGB image, imshow kwargs, colorbar label or None)."""
        self.figure.clear()
        n = len(panels)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        axes = self.figure.subplots(nrows, ncols, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (name, values, kw, cbar) in zip(axes.ravel(), panels):
            im = ax.imshow(values, **kw)
            ax.set_title(name, fontsize=10)
            if cbar is not None:
                self.figure.colorbar(im, ax=ax, pad=0.02, shrink=0.8, label=cbar)
        if title:
            self.figure.suptitle(title)
        self.figure.tight_layout()
        self.canvas.draw_idle()


def _grain_cmap(grain_id):
    rng = np.random.default_rng(42)
    colors = rng.uniform(0.2, 1, (int(np.nanmax(grain_id)) + 1, 3))
    colors[0] = 0
    return ListedColormap(colors)


# ----------------------------------------------------------------------------------------------
# The window
# ----------------------------------------------------------------------------------------------


class Workbench(QMainWindow):
    def __init__(self, preset: str | None = None):
        super().__init__()
        self.setWindowTitle(f"Kikuchi workbench v{__version__}: pattern processor + spherical indexing + HROSM")
        self.setGeometry(60, 60, 1500, 950)
        self.worker: Worker | None = None
        self.base = Settings()
        # prepared state of the indexing tab
        self.prepared = None      # dict: signal, scan, key, source, nav_mask, source_file
        self.harmonics = None     # list, with its key
        self.harmonics_key = None
        self.det = None
        self.last_run = None      # dict: label, signal, xmap, xmap_indexed, det, seconds, offset
        self.last_hrosm = None    # dict: xmap, xmap_indexed, label
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.proc = None
        self._build_processor_tab()
        self._build_indexing_tab()
        self._build_hrosm_tab()
        self.tabs.setCurrentIndex(1 if self.proc is None else 0)
        if preset:
            self.load_preset(preset)
        gpu_ok, gpu_text = sph_core.gpu_available()
        self.log(f"kikuchi_workbench {__version__}; pattern processor: {PROCESSOR_INFO}")
        self.log(f"GPU for spherical indexing: {gpu_text}")
        if not gpu_ok and self.form.has("BACKEND"):
            self.form._fields["BACKEND"][1]("cpu")

    # ---------------------------------------------------------------- tab 1: the processor
    def _build_processor_tab(self):
        if gpv is None:
            w = QWidget()
            lay = QVBoxLayout(w)
            lab = QLabel("The pattern processor is not available:\n" + PROCESSOR_INFO +
                         "\n\nThe indexing and HROSM tabs work from pattern files.")
            lab.setWordWrap(True)
            lay.addWidget(lab)
            lay.addStretch()
            self.tabs.addTab(w, "1. Pattern processor (not found)")
            return
        self.proc = gpv.MainWindow()
        self.tabs.addTab(self.proc, "1. Pattern processor")

        # Capture the path 'Apply && Save' writes, to offer it to the indexing tab
        base_dialog = gpv.QFileDialog
        recorder = {"path": None}

        class _RecordingFileDialog(base_dialog):
            @staticmethod
            def getSaveFileName(*a, **k):
                result = base_dialog.getSaveFileName(*a, **k)
                recorder["path"] = result[0] if result else None
                return result

        gpv.QFileDialog = _RecordingFileDialog
        original_apply_and_save = self.proc.apply_and_save

        def apply_and_save_then_register():
            recorder["path"] = None
            original_apply_and_save()
            path = recorder["path"]
            if path:
                path = os.path.splitext(path)[0] + self.proc.src_ext
                if os.path.exists(path):
                    self._processor_saved(path)

        self.proc.apply_save_btn.clicked.disconnect()
        self.proc.apply_save_btn.clicked.connect(apply_and_save_then_register)

    def _processor_saved(self, path):
        self.log(f"Processor saved {path}; set as the pattern file of the indexing tab")
        self.form._fields["PATTERN_FILE"][1](path)
        self.source_combo.setCurrentIndex(0)
        self.prepared = None

    # ---------------------------------------------------------------- tab 2: spherical indexing
    def _build_indexing_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        self.form = form = SettingsForm()

        form.group("Pattern source")
        self.source_combo = QComboBox()
        self.source_combo.addItem("Pattern file (anything kikuchipy loads: .up1/.up2, .h5, .oh5, ...)")
        if self.proc is not None:
            self.source_combo.addItem("Processor tab: its processed patterns, in RAM (as 'Apply && Save' writes them)")
        self.source_combo.setToolTip("Where the patterns to index come from")
        form._form.addRow("Source:", self.source_combo)
        form.add_path("PATTERN_FILE", "Pattern file:", tip="EDAX .up1/.up2 or any format kp.load() reads; blank = the nickel demo data",
                      filter="Pattern files (*.up1 *.up2 *.h5 *.hdf5 *.oh5 *.edaxh5 *.ebsp *.dat);;All files (*)")
        form.add_path("SCAN_FILE", "Scan file (.ang/.edaxh5):", tip="The .ang or .edaxh5 written at acquisition: gives the projection center, map size and step",
                      filter="Scan files (*.ang *.edaxh5 *.h5);;All files (*)")
        form.add_text("EDAXH5_MAP", "edaxh5 map filter:", tip="Only for .edaxh5 files with several maps: part of the map's name, e.g. 'Area 8'", placeholder="None")
        self.btn_scan = QPushButton("Read scan parameters (fills the projection center)")
        self.btn_scan.clicked.connect(self.read_scan)
        form.add_widget(self.btn_scan)
        form.add_text("NAV_SHAPE", "Map (rows, cols):", tip="Only when the pattern file does not store the map size; blank = from the file or the scan file", placeholder="None")
        form.add_text("STEP_SIZE", "Step (x, y) um:", tip="Only when the pattern file does not store it", placeholder="None")
        form.add_combo("LAZY", "Lazy loading:", ["auto", "True", "False"], "auto",
                       tip="auto = memory-map files bigger than 25 % of RAM. The in-RAM processor source is never lazy")
        form.add_text("ROI", "ROI (c0, c1, r0, r1):", tip="Index only this part of the map: (col_start, col_stop, row_start, row_stop)", placeholder="None = everything")
        form.add_bool("FLIP_PATTERNS_UD", "Flip patterns upside down", "Try this when the test maps look like noise")

        form.group("Pattern processing (file source; the processor tab does its own)")
        form.add_bool("REMOVE_STATIC_BACKGROUND", "Remove static background", "The background stored with the file, or else the mean pattern", True)
        form.add_bool("REMOVE_DYNAMIC_BACKGROUND", "Remove dynamic background", "", True)
        note = QLabel("NLPAR: use the pattern processor tab (pyebsdindex)")
        note.setStyleSheet("color: gray")
        form.add_widget(note)

        form.group("Master pattern(s)")
        form.add_path("MASTER_FILE", "Master file(s):", tip="EMSphInx .sht or EMsoft master pattern .h5; several files (one per phase) as a list ['a.sht', 'b.sht']",
                      filter="Master patterns (*.sht *.h5 *.hdf5);;All files (*)", multi=True)
        form.add_int("MASTER_BANDWIDTH", "Master bandwidth (.h5):", 384, 16, 1024, "Bandwidth of the harmonics computed from an .h5 master; keep >= 384 (see build_notebook.py)")
        form.add_text("MASTER_ENERGY", "Master energy keV (.h5):", tip="Only for .h5 masters with several energies; blank = all, weighted", placeholder="None")
        form.add_bool("OVERRIDE_MASTER_SAMPLE_TILT", "Override the master's sample tilt with the value below", "Indexing refuses a mismatch otherwise")

        form.group("Detector geometry")
        form.add_text("PC", "Projection center (x*, y*, z*):", (0.5, 0.5, 0.5), "Values from an .ang file are in the EDAX convention")
        form.add_combo("PC_CONVENTION", "PC convention:", ["edax", "tsl", "bruker", "oxford", "emsoft"], "edax")
        form.add_text("SAMPLE_TILT", "Sample tilt deg:", tip="Blank = the tilt the master pattern was simulated for (as EMSphInx does)", placeholder="None = from master")
        form.add_float("CAMERA_TILT", "Camera tilt deg:", 0.0, -180, 180, 2, "Detector tilt / camera elevation (EMSphInx thetac): 8 for EBSD, -90 for RKD")
        form.add_float("PX_SIZE", "Pixel size um:", 1.0, 1e-6, 1e6, 4, "Detector pixel size; only needed for the emsoft PC convention")
        form.add_int("BINNING", "Binning:", 1, 1, 64)
        form.add_float("TWIST", "Twist deg:", 0.0, -180, 180, 2, "Rotation of the detector about its normal, e.g. 15 for RKD; applied to the orientations afterwards")

        form.group("Indexing")
        bandwidths = [16, 32, 53, 63, 68, 74, 88, 95, 113, 123, 158, 172, 188, 263, 383]
        form.add_combo("BANDWIDTH", "Bandwidth:", bandwidths, 68, "68 = EMSphInx default; 88 or 123 for more accuracy at lower speed (fast bandwidths only)")
        form.add_combo("BACKEND", "Backend:", ["gpu", "cpu"], "gpu")
        form.add_int("N_WORKERS", "Dask workers:", max(1, (os.cpu_count() or 8) // 2), 1, 256, "About the number of performance cores")
        form.add_bool("REFINE", "Newton refinement", "", True)
        form.add_int("N_REGIONS", "AHE tiles (n_regions):", 10, 0, 64, "Adaptive histogram equalization tiles per axis; 0 = off")
        form.add_bool("CIRCULAR_MASK", "Circular mask", "Use only the inscribed circle of the detector")
        form.add_bool("GAUSSIAN_BACKGROUND", "Gaussian background", "Fit and subtract a 2D Gaussian background (EMSphInx gausbckg)")
        form.add_bool("NORMALIZE", "Normalized cross-correlation", "", True)
        form.add_text("CHUNKSIZE", "Chunk / GPU batch size:", tip="Patterns per chunk; blank = automatic from free VRAM", placeholder="None = automatic")
        form.add_int("TEST_SIZE", "Test region side:", 50, 1, 10000, "Side length in map points of the test region")

        form.group("Output")
        form.add_path("OUTPUT_DIR", "Output folder:", tip="Blank = next to the pattern file", directory=True)
        form.add_text("OUTPUT_NAME", "Output name:", tip="Blank = <pattern file>_sph_bw<bandwidth>", placeholder="None")
        form.add_bool("SAVE_ANG", "Save .ang", "", True)
        form.add_bool("SAVE_H5", "Save .h5", "", True)
        form.add_path("REFERENCE_FILE", "Reference map (optional):", tip="An .ang/.h5/.ctf of the same points to compare with",
                      filter="Crystal maps (*.ang *.h5 *.hdf5 *.ctf);;All files (*)")
        form.finish()

        scroll = QScrollArea()
        scroll.setWidget(form)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(470)
        outer.addWidget(scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        buttons = QHBoxLayout()
        self.btn_preset_load = QPushButton("Load preset...")
        self.btn_preset_load.clicked.connect(lambda: self.load_preset())
        self.btn_preset_save = QPushButton("Save preset...")
        self.btn_preset_save.clicked.connect(self.save_preset)
        self.btn_geometry = QPushButton("Check geometry")
        self.btn_geometry.setToolTip("Load the patterns and master, build the detector, show one pattern on the sphere")
        self.btn_geometry.clicked.connect(self.check_geometry)
        self.btn_test = QPushButton("Index test region")
        self.btn_test.clicked.connect(lambda: self.run_indexing(test=True))
        self.btn_full = QPushButton("Index full map")
        self.btn_full.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; }")
        self.btn_full.clicked.connect(lambda: self.run_indexing(test=False))
        self.btn_save = QPushButton("Save .ang / .h5")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_compare = QPushButton("Compare with reference")
        self.btn_compare.clicked.connect(self.compare_reference)
        for b in (self.btn_preset_load, self.btn_preset_save, self.btn_geometry, self.btn_test, self.btn_full, self.btn_save, self.btn_compare):
            buttons.addWidget(b)
        rl.addLayout(buttons)
        split = QSplitter(Qt.Vertical)
        self.maps = MapCanvas()
        split.addWidget(self.maps)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        self.progress_label = QLabel("")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        bl.addWidget(self.progress_label)
        bl.addWidget(self.log_box)
        split.addWidget(bottom)
        split.setSizes([600, 300])
        rl.addWidget(split)
        outer.addWidget(right, stretch=1)
        self.tabs.addTab(tab, "2. Spherical indexing")

    # ---------------------------------------------------------------- tab 3: HROSM
    def _build_hrosm_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        self.hform = form = SettingsForm()
        form.group("Master pattern for the simulated dictionaries")
        form.add_path("HROSM_MASTER_FILE", "EMsoft master .h5:", tip="An EMsoft master pattern .h5 of the phase (.sht cannot be used: patterns are simulated). Blank with the demo data = the demo master",
                      filter="EMsoft master patterns (*.h5 *.hdf5);;All files (*)")
        form.add_text("HROSM_ENERGY", "Energy keV:", tip="Blank = the highest energy in the file", placeholder="None")
        form.group("Grains and misorientation ball")
        form.add_float("HROSM_THRESHOLD", "KAM threshold deg:", 5.0, 0.01, 180, 2, "Neighbours whose KAM differs by less join the same grain (EMHROSM gangle)")
        form.add_float("HROSM_MAX_ANGLE", "Ball radius deg:", 5.0, 0.01, 179, 2, "Must be at least the largest GROD of the grains (EMHROSM misorang)")
        form.add_int("HROSM_N_STEPS", "Steps per half axis:", 10, 1, 60, "(2 n + 1)^3 orientations per grain: 9261 for 10, 68921 for 20 (EMHROSM default); cost grows with the cube")
        form.add_int("HROSM_KEEP_N", "Best matches kept:", 20, 1, 1000)
        form.add_int("HROSM_N_OSM", "Matches compared (OSM):", 10, 1, 1000, "At most the number kept")
        form.add_combo("HROSM_AVERAGE", "Grain reference:", ["mean", "center", "vmf", "watson"], "mean")
        form.add_int("HROSM_MIN_PIXELS", "Smallest grain (points):", 10, 1, 100000)
        form.add_bool("HROSM_DILATE", "Dilate grains into unassigned neighbours")
        form.add_text("HROSM_ROI", "ROI (c0, c1, r0, r1):", tip="Only these map points are segmented and re-indexed; blank = all", placeholder="None = all")
        form.finish()
        scroll = QScrollArea()
        scroll.setWidget(form)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(470)
        outer.addWidget(scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        buttons = QHBoxLayout()
        self.hrosm_source = QLabel("Map to re-index: none yet (index a test region or the full map in tab 2)")
        self.hrosm_source.setWordWrap(True)
        self.btn_hrosm_load = QPushButton("Load an indexed map (.h5) for the prepared patterns...")
        self.btn_hrosm_load.setToolTip("A spherical indexing .h5 of the same points as the patterns prepared in tab 2 (e.g. from the notebook)")
        self.btn_hrosm_load.clicked.connect(self.load_indexed_map)
        self.btn_hrosm = QPushButton("Run HROSM")
        self.btn_hrosm.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; }")
        self.btn_hrosm.clicked.connect(self.run_hrosm)
        self.btn_hrosm_save = QPushButton("Save HROSM .ang / .h5")
        self.btn_hrosm_save.clicked.connect(self.save_hrosm)
        buttons.addWidget(self.btn_hrosm_load)
        buttons.addWidget(self.btn_hrosm)
        buttons.addWidget(self.btn_hrosm_save)
        rl.addWidget(self.hrosm_source)
        rl.addLayout(buttons)
        self.hmaps = MapCanvas()
        rl.addWidget(self.hmaps, stretch=1)
        outer.addWidget(right, stretch=1)
        self.tabs.addTab(tab, "3. HROSM")

    # ---------------------------------------------------------------- settings, presets, log
    def settings(self) -> Settings:
        st = self.form.settings(self.base)
        st = self.hform.settings(st)
        if isinstance(st.LAZY, str) and st.LAZY != "auto":
            st.LAZY = st.LAZY == "True"
        if isinstance(st.PC, (list, tuple)) and len(st.PC) == 3:
            st.PC = tuple(float(v) for v in st.PC)
        return st

    def set_settings(self, st: Settings):
        self.form.set_settings(st)
        self.hform.set_settings(st)

    def load_preset(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load preset", str(HERE), "Presets (*.txt);;All files (*)")
            if not path:
                return
        st, ignored = sph_core.read_preset(path, self.settings())
        self.set_settings(st)
        self.prepared = None
        self.log(f"Preset loaded: {path}" + (f" (ignored: {', '.join(ignored)})" if ignored else ""))

    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", str(HERE / "preset_workbench.txt"), "Presets (*.txt)")
        if not path:
            return
        sph_core.write_preset(path, self.settings(), header="Settings of the Kikuchi workbench")
        self.log(f"Preset saved: {path}; build a notebook from it with: python build_notebook.py my_scan.ipynb \"{path}\"")

    def log(self, text: str):
        self.log_box.appendPlainText(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def set_progress(self, text: str):
        self.progress_label.setText(text[-200:])

    def status(self, text: str):
        self.set_progress(text)
        QApplication.processEvents()

    def _busy(self, busy: bool):
        for b in (self.btn_geometry, self.btn_test, self.btn_full, self.btn_save, self.btn_compare, self.btn_hrosm,
                  self.btn_hrosm_load, self.btn_hrosm_save, self.btn_preset_load, self.btn_scan):
            b.setEnabled(not busy)

    def _start(self, fn, on_done):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Wait for the running step to finish")
            return
        self._busy(True)
        self.worker = Worker(fn, self)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self.set_progress)

        def done(result):
            self._busy(False)
            self.set_progress("")
            try:
                on_done(result)
            except Exception:  # noqa: BLE001
                self.log(traceback.format_exc())
                QMessageBox.critical(self, "Error", traceback.format_exc()[-1500:])

        def failed(text):
            self._busy(False)
            self.set_progress("")
            self.log(text)
            QMessageBox.critical(self, "Error", text[-1500:])

        self.worker.done.connect(done)
        self.worker.failed.connect(failed)
        self.worker.start()

    # ---------------------------------------------------------------- scan parameters
    def read_scan(self):
        st = self.settings()
        pattern_file = st.PATTERN_FILE
        if self.source_combo.currentIndex() == 1 and self.proc is not None and self.proc.orig_path:
            pattern_file = self.proc.orig_path
        if not st.SCAN_FILE and not pattern_file:
            QMessageBox.information(self, "Scan parameters", "Give a scan file (.ang / .edaxh5) and/or a pattern file first")
            return
        try:
            scan = ebsd_params.scan_info(st.SCAN_FILE, pattern_file=pattern_file, map_filter=st.EDAXH5_MAP)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Scan parameters", f"{type(e).__name__}: {e}")
            return
        import io

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            ebsd_params.print_scan_info(scan)
        finally:
            sys.stdout = old
        self.log(buf.getvalue().rstrip())
        if scan.get("pc"):
            self.form._fields["PC"][1](tuple(round(float(v), 6) for v in scan["pc"]))
            self.form._fields["PC_CONVENTION"][1]("edax")
            self.log(f"Projection center set to {tuple(scan['pc'])} (edax convention)")
        if scan.get("camera_elevation") is not None:
            self.log(f"Camera elevation in the scan file: {scan['camera_elevation']} deg (CAMERA_TILT is not changed automatically)")
        self.prepared = None

    # ---------------------------------------------------------------- preparing patterns, master, detector
    def _prepare_key(self, st: Settings):
        source = self.source_combo.currentIndex()
        if source == 1:
            return ("processor", processor_dataset_key(self.proc), st.ROI, st.FLIP_PATTERNS_UD)
        return ("file", st.PATTERN_FILE, st.SCAN_FILE, st.EDAXH5_MAP, st.NAV_SHAPE, st.STEP_SIZE, st.LAZY, st.ROI,
                st.FLIP_PATTERNS_UD, st.REMOVE_STATIC_BACKGROUND, st.REMOVE_DYNAMIC_BACKGROUND)

    def prepare_patterns_in_gui(self, st: Settings):
        """The part that must run in the GUI thread: the processor's chain. Returns a dict for the worker, or None."""
        key = self._prepare_key(st)
        if self.prepared is not None and self.prepared["key"] == key:
            return self.prepared
        if self.source_combo.currentIndex() != 1:
            return None
        if self.proc is None or self.proc.f_orig is None:
            raise ValueError("Load a UP1/UP2 file in the pattern processor tab first, or choose the file source")
        self.log("Running the processor's processing chain on the whole data set, in RAM ...")
        patterns, mask, info = processed_patterns_from_processor(self.proc, self.status)
        self.log(
            f"Processed {info['n_patterns']} patterns ({info['source']}, {info['excluded']} excluded by the IQ range) "
            f"in {info['seconds']:.0f} s; {patterns.nbytes / 1e9:.2f} GB in RAM"
        )
        step = info["step"] if all(info["step"]) else None
        s = sph_core.signal_from_array(patterns, info["nav_shape"], step)
        s = sph_core.apply_roi_and_flip(s, st)
        nav_mask = None
        if info["excluded"]:
            nav_mask = ~mask
            if st.ROI is not None:
                c0, c1, r0, r1 = st.ROI
                nav_mask = nav_mask[r0:r1, c0:c1]
        scan = ebsd_params.scan_info(st.SCAN_FILE, pattern_file=info["file"], map_filter=st.EDAXH5_MAP) if st.SCAN_FILE else {}
        self.prepared = {"key": key, "signal": s, "scan": scan, "source": "processor", "nav_mask": nav_mask,
                         "source_file": info["file"]}
        return self.prepared

    def _prepare_in_worker(self, st: Settings, prepared, log):
        """Loads the patterns from a file when needed, then the harmonics and the detector."""
        key = self._prepare_key(st)
        if prepared is None:
            s, scan = sph_core.load_patterns(st, log=log)
            sph_core.preprocess(s, st, log)
            prepared = {"key": key, "signal": s, "scan": scan, "source": "file", "nav_mask": None,
                        "source_file": st.PATTERN_FILE}
        demo = prepared["source"] == "file" and st.PATTERN_FILE is None
        hkey = (st.MASTER_FILE, st.MASTER_BANDWIDTH, st.MASTER_ENERGY, demo)
        if self.harmonics is None or self.harmonics_key != hkey:
            harmonics = sph_core.load_harmonics(st, log, demo=demo)
        else:
            harmonics = self.harmonics
        det = sph_core.make_detector(st, prepared["signal"], harmonics, log, demo=demo)
        return prepared, harmonics, hkey, det, demo

    def _store_prepared(self, result):
        prepared, harmonics, hkey, det, demo = result
        self.prepared = prepared
        self.harmonics, self.harmonics_key = harmonics, hkey
        self.det = det
        self.demo = demo

    # ---------------------------------------------------------------- actions
    def check_geometry(self):
        st = self.settings()
        try:
            prepared = self.prepare_patterns_in_gui(st)
        except Exception as e:  # noqa: BLE001
            self.log(f"{type(e).__name__}: {e}")
            QMessageBox.critical(self, "Patterns", f"{type(e).__name__}: {e}")
            return

        def work(log):
            result = self._prepare_in_worker(st, prepared, log)
            prepared_, harmonics, hkey, det, demo = result
            s = prepared_["signal"]
            nav_shape = s.axes_manager.navigation_shape[::-1]
            idx = np.unravel_index(int(np.prod(nav_shape)) // 2, nav_shape)
            pattern = np.asarray(s.data[idx])
            seen = sph_core.preprocess_pattern_for_preview(pattern, st)
            north, window, iq = sph_core.back_project(pattern, det, st.BANDWIDTH)
            log(f"Detector: {det}")
            log(f"Image quality of the pattern at (row, col) = {tuple(int(i) for i in idx)}: {iq:.3f}")
            return result, pattern, seen, north, window, idx

        def done(out):
            result, pattern, seen, north, window, idx = out
            self._store_prepared(result)
            self.maps.show_panels([
                (f"Pattern (row, col) = {tuple(int(i) for i in idx)}", pattern, dict(cmap="gray"), None),
                ("As the indexer sees it", seen, dict(cmap="gray"), None),
                ("On the sphere (north)", north, dict(cmap="gray"), None),
                ("Window", window, dict(cmap="gray"), None),
            ], "Geometry check")

        self._start(work, done)

    def run_indexing(self, test: bool):
        st = self.settings()
        try:
            prepared = self.prepare_patterns_in_gui(st)
        except Exception as e:  # noqa: BLE001
            self.log(f"{type(e).__name__}: {e}")
            QMessageBox.critical(self, "Patterns", f"{type(e).__name__}: {e}")
            return

        def work(log):
            result = self._prepare_in_worker(st, prepared, log)
            prepared_, harmonics, hkey, det, demo = result
            s = prepared_["signal"]
            nav_mask = prepared_["nav_mask"]
            if test:
                s_run, s_warm, (c0, r0) = sph_core.test_region(s, int(st.TEST_SIZE))
                if nav_mask is not None:
                    nav_mask = nav_mask[r0 : r0 + int(st.TEST_SIZE), c0 : c0 + int(st.TEST_SIZE)]
                label, offset = "test region", (r0, c0)
            else:
                s_run, s_warm, label, offset = s, None, "full map", (0, 0)
            if nav_mask is not None and not nav_mask.any():
                nav_mask = None
            if nav_mask is not None:
                log(f"{int(nav_mask.sum())} patterns excluded by the processor's IQ range are not indexed")
            log(f"Indexing the {label}: {int(np.prod(s_run.axes_manager.navigation_shape))} patterns, bandwidth {st.BANDWIDTH}, {st.BACKEND}")
            xmap, xmap_indexed, dt = sph_core.index(s_run, harmonics, det, st, log, navigation_mask=nav_mask, warm_up=s_warm)
            if test:
                n_all = int(np.prod(s.axes_manager.navigation_shape))
                log(f"Estimated time for the full map of {n_all} patterns: {n_all / (xmap.size / dt) / 60:.1f} min")
            return result, dict(label=label, signal=s_run, xmap=xmap, xmap_indexed=xmap_indexed, det=det, seconds=dt, offset=offset)

        def done(out):
            result, run = out
            self._store_prepared(result)
            self.last_run = run
            self.show_indexing_maps(run)
            self.hrosm_source.setText(f"Map to re-index: the {run['label']} indexed just now ({run['xmap'].size} points)")

        self._start(work, done)

    def show_indexing_maps(self, run):
        xmap = run["xmap"]
        if xmap.ndim != 2:
            self.log("The map is not 2D; nothing to plot")
            return
        rgb = sph_core.ipf_rgb_map(xmap, "z")
        rgb_x = sph_core.ipf_rgb_map(xmap, "x")
        self.maps.show_panels([
            ("Correlation score", sph_core.map_data(xmap, "scores"), {}, "score"),
            ("Image quality", sph_core.map_data(xmap, "iq"), {}, "IQ"),
            ("IPF-Z", rgb, {}, None),
            ("IPF-X", rgb_x, {}, None),
        ], f"{run['label'].capitalize()}: {xmap.size} patterns in {run['seconds']:.1f} s = {xmap.size / run['seconds']:.0f} patterns/s")

    def save_result(self):
        if self.last_run is None:
            QMessageBox.information(self, "Save", "Nothing indexed yet")
            return
        st = self.settings()
        out_dir, stem = sph_core.output_location(st, self.prepared.get("source_file") if self.prepared else None)
        if self.last_run["label"] == "test region":
            stem += "_test"
        try:
            sph_core.save_xmap(self.last_run["xmap"], out_dir, stem, st.SAVE_ANG, st.SAVE_H5, log=self.log)
        except Exception as e:  # noqa: BLE001
            self.log(f"{type(e).__name__}: {e}")
            QMessageBox.critical(self, "Save", f"{type(e).__name__}: {e}")

    def compare_reference(self):
        st = self.settings()
        if self.last_run is None or not st.REFERENCE_FILE:
            QMessageBox.information(self, "Compare", "Index the full map and give a reference file first")
            return
        run = self.last_run
        s = self.prepared["signal"]
        full_nav_shape = s.axes_manager.navigation_shape[::-1]

        def work(log):
            ang = sph_core.compare_with_reference(run["xmap_indexed"], st.REFERENCE_FILE, st.ROI, full_nav_shape)
            log(f"Misorientation to the reference: median {np.median(ang):.2f} deg, p90 {np.percentile(ang, 90):.2f} deg, "
                f"fraction below 2 deg {np.mean(ang < 2):.1%}")
            return ang

        def done(ang):
            if run["xmap"].ndim == 2:
                self.maps.show_panels([("Misorientation to the reference [deg]", ang.reshape(run["xmap"].shape), dict(vmax=5), "deg")])

        self._start(work, done)

    # ---------------------------------------------------------------- HROSM
    def load_indexed_map(self):
        if self.prepared is None or self.det is None:
            QMessageBox.information(self, "HROSM", "Run 'Check geometry' in tab 2 first, so that the patterns and detector of the map are prepared")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Indexed map", str(HERE), "Crystal maps (*.h5 *.hdf5 *.ang);;All files (*)")
        if not path:
            return
        from orix import io

        st = self.settings()
        xmap = io.load(path)
        s = self.prepared["signal"]
        nav_shape = tuple(s.axes_manager.navigation_shape[::-1])
        if tuple(xmap.shape) != nav_shape:
            QMessageBox.critical(self, "HROSM", f"The map is {tuple(xmap.shape)} but the prepared patterns are {nav_shape}")
            return
        xmap_indexed = sph_core.apply_twist(xmap, -st.TWIST)  # the saved map carries TWIST; HROSM needs the orientations as indexed
        self.last_run = dict(label=f"map {Path(path).name}", signal=s, xmap=xmap, xmap_indexed=xmap_indexed, det=self.det, seconds=np.nan, offset=(0, 0))
        self.hrosm_source.setText(f"Map to re-index: {path} ({xmap.size} points)")
        self.log(f"Loaded {path} for HROSM (TWIST {st.TWIST} deg removed for the re-indexing)")

    def run_hrosm(self):
        if self.last_run is None:
            QMessageBox.information(self, "HROSM", "Index a test region or the full map in tab 2 first, or load an indexed map")
            return
        st = self.settings()
        run = self.last_run
        demo = getattr(self, "demo", False)

        def work(log):
            mp = sph_core.load_master_pattern(st, log, demo=demo)
            xmap_hrosm, xmap_hrosm_indexed, dt, messages = sph_core.run_hrosm(run["signal"], run["xmap_indexed"], mp, run["det"], st, log)
            grain_id = np.nan_to_num(sph_core.map_data(xmap_hrosm, "grain_id"), nan=0).astype(int)
            grod_hrosm = sph_core.grod_of(xmap_hrosm_indexed, grain_id) if xmap_hrosm.ndim == 2 else None
            return dict(xmap=xmap_hrosm, xmap_indexed=xmap_hrosm_indexed, label=run["label"], seconds=dt, grain_id=grain_id, grod_hrosm=grod_hrosm)

        def done(res):
            self.last_hrosm = res
            xmap = res["xmap"]
            if xmap.ndim != 2:
                return
            cmap = "viridis"
            osm = sph_core.map_data(xmap, "osm")
            panels = [
                ("KAM [deg]", sph_core.map_data(xmap, "kam"), dict(vmin=0, vmax=2, cmap=cmap), "deg"),
                ("Grains (black: none)", res["grain_id"], dict(cmap=_grain_cmap(res["grain_id"]), interpolation="nearest"), None),
                ("GROD, spherical indexing [deg]", sph_core.map_data(xmap, "grod"), dict(vmin=0, vmax=st.HROSM_MAX_ANGLE, cmap=cmap), "deg"),
                (f"HROSM (of {st.HROSM_N_OSM}; grey: not re-indexed)", np.ma.masked_invalid(osm), dict(vmin=0, vmax=st.HROSM_N_OSM, cmap=cmap), "OSM"),
                ("GROD, re-indexed [deg]", np.ma.masked_invalid(res["grod_hrosm"]), dict(vmin=0, cmap=cmap), "deg"),
                ("IPF-Z, re-indexed", sph_core.ipf_rgb_map(xmap, "z"), {}, None),
            ]
            self.hmaps.show_panels(panels, f"HROSM of the {res['label']}: {res['seconds'] / 60:.1f} min")

        self._start(work, done)

    def save_hrosm(self):
        if self.last_hrosm is None:
            QMessageBox.information(self, "Save", "No HROSM result yet")
            return
        st = self.settings()
        out_dir, stem = sph_core.output_location(st, self.prepared.get("source_file") if self.prepared else None)
        if self.last_hrosm["label"] == "test region":
            stem += "_test"
        stem += f"_hrosm_n{st.HROSM_N_STEPS}"
        try:
            xmap_out = sph_core.hrosm_output_map(self.last_hrosm["xmap"], self.last_run["xmap_indexed"])
            sph_core.save_xmap(xmap_out, out_dir, stem, st.SAVE_ANG, st.SAVE_H5, iq_prop="iq", ci_prop="best_score", log=self.log)
        except Exception as e:  # noqa: BLE001
            self.log(f"{type(e).__name__}: {e}")
            QMessageBox.critical(self, "Save", f"{type(e).__name__}: {e}")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            if QMessageBox.question(self, "Quit", "A step is still running. Quit anyway?") != QMessageBox.Yes:
                event.ignore()
                return
        if self.proc is not None:
            try:
                self.proc._remove_temp_files()
            except Exception:  # noqa: BLE001
                pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = Workbench(preset=ARGS.preset)

    def excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            win.log(text)
        except Exception:  # noqa: BLE001
            pass
        sys.__stderr__.write(text)
        QMessageBox.critical(win, "Error", text[-1500:])

    sys.excepthook = excepthook
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
