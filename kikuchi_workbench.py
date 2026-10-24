#!/usr/bin/env python3
"""Kikuchi workbench: the Kikuchi Pattern Processor and GPU spherical indexing in one window.

    python kikuchi_workbench.py                              # finds the pattern processor next to this folder
    python kikuchi_workbench.py --processor-dir C:\\path\\to\\kikuchi-pattern-processor
    python kikuchi_workbench.py --preset preset_RKD.txt      # start with the settings of a notebook preset
    set KPP_DIR=C:\\path\\to\\kikuchi-pattern-processor       # alternative to --processor-dir (Windows cmd)

Three tabs:

1. **Pattern processor**: the unchanged Kikuchi Pattern Processor (gpu_pattern_viewer.py of
   https://github.com/helpimlost97/kikuchi-pattern-processor), imported from its own folder. Load a UP1/UP2,
   set the aperture, IQ range, NLPAR and the processing chain as usual. "Apply && Save" still writes a file; the
   saved file is then offered to the indexing tab automatically.
2. **Spherical indexing**: the settings of spherical_indexing_template.ipynb as a form. The patterns come from a
   file (anything kikuchipy loads, memory-mapped when big): "Load processed patterns" takes the file the
   processor's "Apply && Save" just wrote (every pattern of it is indexed, the ones outside the processor's IQ
   range too: the processor writes them unprocessed), the browse button any other file. Check the geometry on one pattern, index a test region, index the full map, save
   .ang/.h5. Load and save notebook presets.

Needs the environment of README.md (the kikuchipy fork, CuPy for the GPU) plus the pattern processor's packages
(PyQt5, pyebsdindex, scipy, scikit-image). Without the processor folder the indexing tab still works
from files.
"""
from __future__ import annotations

import argparse
import ast
import collections
import functools
import dataclasses
import gc
import inspect
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
    QGridLayout, QLabel, QLayout, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import ebsd_params  # noqa: E402
import sph_core  # noqa: E402
from live_view import LiveView  # noqa: E402
from montage_tab import MontageTab  # noqa: E402
from sph_core import Settings  # noqa: E402

__version__ = "0.4.0"

# The EDAX detector is 30 mm wide: the pixel size of the EDAX/TSL projection center convention follows from the
# pattern width alone, which is why the GUI fills it in (ebsd_params.py uses the same default, --det-mm 30).
DETECTOR_WIDTH_UM = 30000.0
# How the map is indexed with the live view on: bands of rows that each take about this long
LIVE_BLOCK_SECONDS = 8.0
# What a GUI preset holds: the indexing parameters only, never files, folders or the projection center values
PRESET_FIELDS = [
    "MASTER_BANDWIDTH", "PC_CONVENTION", "SAMPLE_TILT", "CAMERA_TILT", "PX_SIZE", "TWIST",
    "REMOVE_STATIC_BACKGROUND", "REMOVE_DYNAMIC_BACKGROUND",
    "BANDWIDTH", "N_WORKERS", "REFINE", "N_REGIONS", "CIRCULAR_MASK", "GAUSSIAN_BACKGROUND", "NORMALIZE", "CHUNKSIZE",
]


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
    """Runs ``fn(log)`` or ``fn(log, emit_event)`` in a thread. ``emit_event(name, data)`` delivers typed events
    (block_start, progress, block_done, ...) to the GUI thread through the ``event`` signal."""

    log = pyqtSignal(str)
    progress = pyqtSignal(str)
    event = pyqtSignal(str, object)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.takes_events = len(inspect.signature(fn).parameters) >= 2

    def run(self):
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(orig_out, self.log.emit, self.progress.emit)
        sys.stderr = _Tee(orig_err, self.log.emit, self.progress.emit)
        try:
            if self.takes_events:
                result = self.fn(self.log.emit, self.event.emit)
            else:
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


def _fit(widget, min_width=60):
    """Let a field shrink to the width its column has, instead of demanding the width of its contents.

    With QSizePolicy.Ignored the layout decides the width, so long paths and long combo texts are elided or
    scroll inside the box rather than pushing the panel wider than the pane it sits in.
    """
    policy = widget.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Ignored)
    widget.setSizePolicy(policy)
    widget.setMinimumWidth(min_width)
    if isinstance(widget, QComboBox):
        widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        widget.setMinimumContentsLength(8)
    return widget


def button_rows(buttons, per_row=4) -> QGridLayout:
    """Buttons in rows of ``per_row``, each allowed to shrink below its text width, so that the pane they sit
    in never forces the settings panel narrower."""
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    for i, b in enumerate(buttons):
        b.setMinimumWidth(40)
        policy = b.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Ignored)
        b.setSizePolicy(policy)
        grid.addWidget(b, i // per_row, i % per_row)
    return grid


def side_by_side(left: QWidget, right: QWidget, left_width: int = 470) -> QSplitter:
    """A settings panel in a scroll area and a work area next to it, with a draggable divider between them."""
    scroll = QScrollArea()
    scroll.setWidget(left)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setMinimumWidth(350)
    splitter = QSplitter(Qt.Horizontal)
    splitter.addWidget(scroll)
    splitter.addWidget(right)
    splitter.setCollapsible(0, False)
    splitter.setCollapsible(1, False)
    right.setMinimumWidth(320)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([left_width, 1000])
    splitter.setHandleWidth(8)
    return splitter


class SettingsForm(QWidget):
    """Widgets for a subset of the Settings fields, in groups. settings()/set_settings() convert both ways."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields: dict[str, tuple] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        # The panel takes the width the scroll area gives it, whatever the longest label would like; the
        # fields shrink (see _fit), and only a very narrow panel clips the end of a long checkbox text.
        self._layout.setSizeConstraint(QLayout.SetNoConstraint)
        self._form = None

    def minimumSizeHint(self):  # noqa: N802 (Qt name)
        # The scroll area never makes the panel narrower than this; the layout's own minimum (the longest
        # row) would pin the panel at that width and let the splitter clip it behind the right pane.
        hint = super().minimumSizeHint()
        hint.setWidth(200)
        return hint

    def group(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        self._form = QFormLayout(box)
        self._form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # a long label goes above its field when the panel is narrow, instead of squeezing the field
        self._form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self._form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._layout.addWidget(box)
        return self._form

    def finish(self):
        self._layout.addStretch()

    def add_widget(self, widget):
        if isinstance(widget, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
            _fit(widget)
        elif isinstance(widget, QLabel):
            widget.setWordWrap(True)
        self._form.addRow(widget)

    def add_row(self, label, widget):
        self._form.addRow(label, _fit(widget))

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
        self._form.addRow(label, _fit(w))
        self._fields[name] = (w.value, lambda v: w.setValue(int(v)))
        return w

    def add_float(self, name, label, default, lo=-1e9, hi=1e9, decimals=4, tip=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(float(default))
        w.setToolTip(tip)
        self._form.addRow(label, _fit(w))
        self._fields[name] = (w.value, lambda v: w.setValue(float(v)))
        return w

    def add_text(self, name, label, default=None, tip="", placeholder="", literal=True):
        """Free text, parsed as a Python literal (blank = None): numbers, tuples such as (0, 100, 0, 50), paths.
        ``literal=False`` keeps the text as it is (blank = None), e.g. for a name filter such as '8'."""
        w = QLineEdit(_to_text(default))
        w.setToolTip(tip)
        w.setPlaceholderText(placeholder)
        self._form.addRow(label, _fit(w))
        getter = (lambda: _literal(w.text())) if literal else (lambda: (w.text().strip() or None))
        self._fields[name] = (getter, lambda v: w.setText(_to_text(v)))
        return w

    def add_combo(self, name, label, options, default, tip=""):
        w = QComboBox()
        w.addItems([str(o) for o in options])
        w.setCurrentText(str(default))
        w.setToolTip(tip)
        self._form.addRow(label, _fit(w))
        self._fields[name] = (lambda: type(options[0])(w.currentText()) if options and not isinstance(options[0], str) else w.currentText(),
                              lambda v: w.setCurrentText(str(v)))
        return w

    def add_path(self, name, label, default=None, tip="", filter="All files (*)", save=False, directory=False, multi=False,
                 start_dir=None, on_chosen=None):
        """A path field with a browse button. ``start_dir`` is a callable returning the folder the dialog opens in
        when the field is still empty (e.g. the folder of another field's file); ``on_chosen(path)`` is called
        after a file was picked in the dialog."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(_to_text(default))
        edit.setToolTip(tip)
        btn = QPushButton("...")
        btn.setFixedWidth(28)

        def browse():
            start = edit.text().strip()
            if not start and start_dir is not None:
                start = start_dir() or ""
            start = start or str(Path.cwd())
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
                if on_chosen is not None:
                    on_chosen(path)

        btn.clicked.connect(browse)
        lay.addWidget(_fit(edit))
        lay.addWidget(btn)
        self._form.addRow(label, _fit(row, 100))

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


# ----------------------------------------------------------------------------------------------
# The window
# ----------------------------------------------------------------------------------------------


class Workbench(QMainWindow):
    def __init__(self, preset: str | None = None):
        super().__init__()
        self.setWindowTitle(f"Kikuchi workbench v{__version__}: pattern processor + spherical indexing")
        self.setGeometry(60, 60, 1500, 950)
        self.worker: Worker | None = None
        self._workers: list[Worker] = []  # workers not yet finished (dropped by _drop_finished_workers)
        self.batch_running = False        # the montage tab's batch processing runs on the GUI thread
        self.base = Settings()
        # prepared state of the indexing tab
        self.loaded = None        # dict: key, signal (raw patterns), scan, iq (stretched quasi-IQ), source, source_file
        self.prepared = None      # dict: signal, scan, key, source, source_file
        self.pattern_width = None  # pixels, once a pattern file is known
        self.processed_file = None      # the file the processor's Apply && Save wrote last
        self.live_run = None      # state of the run shown in the live view
        self.harmonics = None     # list, with its key
        self.harmonics_key = None
        self.det = None
        self.last_run = None      # dict: label, signal, xmap, xmap_indexed, det, seconds, offset
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.proc = None
        self.montage_tab = None
        self._build_processor_tab()
        self._build_indexing_tab()
        self._live_view = self.live
        self.montage_tab = MontageTab(self)
        self.tabs.addTab(self.montage_tab, "3. Montage batch")
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
                         "\n\nThe indexing tab works from pattern files.")
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

        # Only one data set in memory at a time: loading into the processor drops the indexer's patterns
        original_load_path = self.proc.load_path

        def load_path_exclusive(path):
            if self.worker is not None and self.worker.isRunning():
                QMessageBox.information(self, "Busy", "Wait for the indexing tab's running step to finish before loading")
                return
            self.unload_indexing("a file is being loaded in the pattern processor")
            original_load_path(path)

        self.proc.load_path = load_path_exclusive

    def _processor_saved(self, path):
        """Called after the processor's Apply && Save: offers the written file to the indexing tab. Every pattern of
        it is indexed; the ones outside the processor's IQ range are in the file unprocessed."""
        self.processed_file = path
        self.btn_load_processed.setEnabled(True)
        self.btn_load_processed.setToolTip(f"Loads {path}")
        self.log(f"Processor saved {path}: 'Load processed patterns' in the indexing tab loads it")

    def load_processed_clicked(self):
        """Put the file Apply && Save wrote into the pattern file box and load it."""
        if self._refuse_while_busy():
            return
        if not self.processed_file or not os.path.exists(self.processed_file):
            QMessageBox.information(self, "Processed patterns", "Run 'Apply && Save' in the pattern processor tab first")
            return
        self.form._fields["PATTERN_FILE"][1](self.processed_file)
        self.prepared = None
        self.load_patterns_clicked()

    # ---------------------------------------------------------------- tab 2: spherical indexing
    def _build_indexing_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        self.form = form = SettingsForm()

        form.group("Pattern source")
        self.btn_load_processed = QPushButton("Load processed patterns")
        self.btn_load_processed.setEnabled(False)
        self.btn_load_processed.setToolTip("Greyed out until 'Apply && Save' in the pattern processor tab has written a file; "
                                           "then loads that file here")
        self.btn_load_processed.clicked.connect(self.load_processed_clicked)
        form.add_widget(self.btn_load_processed)
        self.pattern_file_edit = form.add_path(
            "PATTERN_FILE", "Pattern file:", tip="EDAX .up1/.up2 or any format kp.load() reads (an HDF5 pattern file of the h5ebsd layout, "
                                           "whatever its extension); an APEX .edaxh5 project is a scan file, its patterns are in the .up1 next "
                                           "to it. Blank = the nickel demo data",
            filter="Pattern files (*.up1 *.up2 *.h5 *.hdf5 *.oh5 *.edaxh5 *.ebsp *.dat);;All files (*)",
            on_chosen=lambda path: self.load_patterns_clicked())
        form.add_path("SCAN_FILE", "Scan file (.ang/.edaxh5):", tip="The .ang or .edaxh5 written at acquisition: gives the projection center, map size and step",
                      filter="Scan files (*.ang *.edaxh5 *.h5);;All files (*)", start_dir=self._pattern_file_dir)
        form.add_text("EDAXH5_MAP", "edaxh5 map filter:", tip="Only for .edaxh5 files with several maps: part of the map's name, e.g. 'Area 8'",
                      placeholder="None", literal=False)
        self.btn_load = QPushButton("Load patterns (quasi-IQ map)")
        self.btn_load.setToolTip("Loads the pattern file (or takes the processor's data set), shows its quasi image quality map "
                                 "in the Live view and fills the pixel size; done automatically after browsing for a file")
        self.btn_load.clicked.connect(self.load_patterns_clicked)
        form.add_widget(self.btn_load)
        self.btn_scan = QPushButton("Read scan parameters")
        self.btn_scan.setToolTip("Reads the scan and pattern file headers, prints them in the log and fills the projection center")
        self.btn_scan.clicked.connect(self.read_scan)
        form.add_widget(self.btn_scan)
        form.add_text("NAV_SHAPE", "Map (rows, cols):", tip="Only when the pattern file does not store the map size; blank = from the file or the scan file", placeholder="None")
        form.add_text("STEP_SIZE", "Step (x, y) um:", tip="Only when the pattern file does not store it", placeholder="None")
        form.add_combo("LAZY", "Lazy loading:", ["auto", "True", "False"], "auto",
                       tip="auto = memory-map files bigger than 25 % of RAM. The in-RAM processor source is never lazy")
        form.add_text("ROI", "ROI (c0, c1, r0, r1):", tip="Index only this part of the map: (col_start, col_stop, row_start, row_stop)", placeholder="None = everything")
        form.add_bool("FLIP_PATTERNS_UD", "Flip patterns upside down", "Try this when the test maps look like noise")

        form.group("Pattern processing (off for a file the processor already processed)")
        self.w_static = form.add_bool("REMOVE_STATIC_BACKGROUND", "Remove static background", "The background stored with the file, or else the mean pattern. Off by default: patterns from the processor are already corrected", False)
        self.w_dynamic = form.add_bool("REMOVE_DYNAMIC_BACKGROUND", "Remove dynamic background", "Off by default: patterns from the processor are already corrected", False)
        note = QLabel("NLPAR: use the pattern processor tab (pyebsdindex)")
        note.setStyleSheet("color: gray")
        form.add_widget(note)

        form.group("Master pattern(s)")
        form.add_path("MASTER_FILE", "Master file(s):", tip="EMSphInx .sht or EMsoft master pattern .h5; several files (one per phase) as a list ['a.sht', 'b.sht']",
                      filter="Master patterns (*.sht *.h5 *.hdf5);;All files (*)", multi=True)
        form.add_int("MASTER_BANDWIDTH", "Master bandwidth (.h5):", 384, 16, 1024, "Bandwidth of the harmonics computed from an .h5 master; keep >= 384 (see build_notebook.py)")
        form.add_text("MASTER_ENERGY", "Master energy keV (.h5):", tip="Only for .h5 masters with several energies; blank = all, weighted", placeholder="None")
        form.add_bool("OVERRIDE_MASTER_SAMPLE_TILT", "Override master sample tilt", "Use the sample tilt below even when the master pattern was simulated for another one; indexing refuses the mismatch otherwise")

        form.group("Detector geometry")
        form.add_text("PC", "Projection center (x*, y*, z*):", (0.5, 0.5, 0.5), "Values from an .ang file are in the EDAX convention")
        self.pc_convention_combo = form.add_combo("PC_CONVENTION", "PC convention:", ["edax", "tsl", "bruker", "oxford", "emsoft"], "edax")
        self.pc_convention_combo.currentTextChanged.connect(lambda *_: self.auto_px_size())
        form.add_text("SAMPLE_TILT", "Sample tilt deg:", tip="Blank = the tilt the master pattern was simulated for (as EMSphInx does)", placeholder="None = from master")
        form.add_float("CAMERA_TILT", "Camera tilt deg:", 0.0, -180, 180, 2, "Detector tilt / camera elevation (EMSphInx thetac): 8 for EBSD, -90 for RKD")
        self.px_size_spin = form.add_float("PX_SIZE", "Pixel size um:", 1.0, 1e-6, 1e6, 4,
                                           f"Detector pixel size. Filled automatically as {DETECTOR_WIDTH_UM / 1000:.0f} mm / pattern width "
                                           "for the EDAX/TSL convention once the pattern width is known; editable")
        form.add_float("TWIST", "Twist deg:", 0.0, -180, 180, 2, "Rotation of the detector about its normal, e.g. 15 for RKD; applied to the orientations afterwards")

        form.group("Indexing")
        bandwidths = [16, 32, 53, 63, 68, 74, 88, 95, 113, 123, 158, 172, 188, 263, 383]
        form.add_combo("BANDWIDTH", "Bandwidth:", bandwidths, 68, "68 = EMSphInx default; 88 or 123 for more accuracy at lower speed (fast bandwidths only)")
        form.add_combo("BACKEND", "Backend:", ["gpu", "cpu"], "gpu")
        form.add_int("N_WORKERS", "Dask workers:", max(1, (os.cpu_count() or 8) // 2), 1, 256, "About the number of performance cores")
        form.add_bool("REFINE", "Newton refinement", "", True)
        self.w_regions = form.add_int("N_REGIONS", "AHE tiles (n_regions):", 10, 0, 64, "Adaptive histogram equalization tiles per axis; 0 = off")
        form.add_bool("CIRCULAR_MASK", "Circular mask", "Use only the inscribed circle of the detector", True)
        self.w_gaussian = form.add_bool("GAUSSIAN_BACKGROUND", "Gaussian background", "Fit and subtract a 2D Gaussian background (EMSphInx gausbckg)")
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

        right = QWidget()
        rl = QVBoxLayout(right)
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
        rl.addLayout(button_rows([self.btn_preset_load, self.btn_preset_save, self.btn_geometry, self.btn_compare,
                                  self.btn_test, self.btn_full, self.btn_save]))
        self.live_check = QCheckBox("Live view: index band by band and animate the progress on the quasi-IQ map")
        self.live_check.setChecked(True)
        self.live_check.setToolTip("Off = one call for the whole map (no animation, a few per cent faster on tiny maps)")
        rl.addWidget(self.live_check)
        split = QSplitter(Qt.Vertical)
        self.view_tabs = QTabWidget()
        self.live = LiveView()
        self.live.error.connect(self.log)
        self.live.set_pattern_source(self._live_pattern)
        self.live.point_clicked.connect(self.preview_point)
        self._preview_rc = None
        for w in (self.w_static, self.w_dynamic, self.w_gaussian):
            w.toggled.connect(lambda *_: self._refresh_preview())
        self.w_regions.valueChanged.connect(lambda *_: self._refresh_preview())
        self.maps = MapCanvas()
        self.view_tabs.addTab(self.live, "Live")
        self.view_tabs.addTab(self.maps, "Results")
        split.addWidget(self.view_tabs)
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
        outer.addWidget(side_by_side(form, right))
        self.tabs.addTab(tab, "2. Spherical indexing")

    # ---------------------------------------------------------------- settings, presets, log
    def settings(self) -> Settings:
        st = self.form.settings(self.base)
        if isinstance(st.LAZY, str) and st.LAZY != "auto":
            st.LAZY = st.LAZY == "True"
        if isinstance(st.PC, (list, tuple)) and len(st.PC) == 3:
            st.PC = tuple(float(v) for v in st.PC)
        return st

    def set_settings(self, st: Settings):
        self.form.set_settings(st)

    def load_preset(self, path=None):
        if self._refuse_while_busy():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load preset", str(HERE), "Presets (*.txt);;All files (*)")
            if not path:
                return
        current = self.settings()
        st, ignored = sph_core.read_preset(path, current)
        # only the indexing parameters are taken from a preset; files, folders and the PC values stay as they are
        for name in st.names():
            if name not in PRESET_FIELDS:
                setattr(st, name, getattr(current, name))
        self.set_settings(st)
        applied = [n for n in PRESET_FIELDS if getattr(st, n) != getattr(current, n)]
        self.log(f"Preset loaded: {path}; changed: {', '.join(applied) if applied else 'nothing'}"
                 + " (files, folders and the projection center are never taken from a preset)")

    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", str(HERE / "preset_workbench.txt"), "Presets (*.txt)")
        if not path:
            return
        sph_core.write_preset(path, self.settings(), header="Indexing settings of the Kikuchi workbench (no files, folders or PC values)",
                              fields=PRESET_FIELDS)
        self.log(f"Preset saved: {path} ({len(PRESET_FIELDS)} indexing settings); a notebook can be built from it with: "
                 f"python build_notebook.py my_scan.ipynb \"{path}\"")

    def log(self, text: str):
        self.log_box.appendPlainText(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        if self.montage_tab is not None:
            self.montage_tab.log(text)

    def set_progress(self, text: str):
        self.progress_label.setText(text[-200:])
        if self.montage_tab is not None:
            self.montage_tab.set_progress(text)

    def _busy(self, busy: bool):
        for b in (self.btn_geometry, self.btn_test, self.btn_full, self.btn_save, self.btn_compare,
                  self.btn_preset_load, self.btn_scan, self.btn_load):
            b.setEnabled(not busy)
        self.btn_load_processed.setEnabled(not busy and bool(self.processed_file))

    def worker_running(self) -> bool:
        """True while a worker thread runs, or the montage tab's batch processing (GUI thread) is in progress."""
        if self.batch_running:
            return True
        try:
            return self.worker is not None and self.worker.isRunning()
        except RuntimeError:  # the QThread was deleted (deleteLater after it finished)
            self.worker = None
            return False

    def _refuse_while_busy(self) -> bool:
        """True (after telling the user) when a step is running, so that an action which would unload or
        change the loaded data is not taken half-way: the browse buttons and the montage tab are not disabled by
        ``_busy``."""
        if self.worker_running():
            QMessageBox.information(self, "Busy", "Wait for the running step to finish")
            return True
        return False

    def _drop_finished_workers(self):
        """Forget the worker threads that have finished (GUI thread, from a worker's ``finished`` signal): their
        function, whose closure holds the data set, is dropped and they are detached from the window, so Python
        frees them. (Deleting a Worker with deleteLater makes PyQt raise 'native Qt signal is not callable'.)"""
        still = []
        for w in self._workers:
            if w.isRunning():
                still.append(w)
                continue
            w.fn = None
            w.setParent(None)  # self.worker keeps pointing at the last one (scripts poll worker.isRunning())
        self._workers = still

    def _start(self, fn, on_done, on_event=None, on_failed=None) -> bool:
        """Run ``fn`` on a worker thread; False (after telling the user) when one is already running."""
        if self._refuse_while_busy():
            return False
        self._busy(True)
        worker = self.worker = Worker(fn, self)
        worker.log.connect(self.log)
        worker.progress.connect(self.set_progress)
        worker.event.connect(on_event or self.on_run_event)
        # a finished worker is deleted, and its function (whose closure holds the data set) dropped at once: as
        # children of the window the old workers would otherwise keep every run's patterns alive
        self._workers.append(worker)
        worker.finished.connect(self._drop_finished_workers)

        def release():
            worker.fn = None
            self._busy(False)
            self.set_progress("")

        def done(result):
            release()
            try:
                on_done(result)
            except Exception:  # noqa: BLE001
                self.log(traceback.format_exc())
                QMessageBox.critical(self, "Error", traceback.format_exc()[-1500:])

        def failed(text):
            release()
            if self.live_run is not None:
                self._live_view.end_run(False)
                self.live_run = None
            if on_failed is not None:
                on_failed()
            self.log(text)
            QMessageBox.critical(self, "Error", text[-1500:])

        worker.done.connect(done)
        worker.failed.connect(failed)
        worker.start()
        return True

    # ---------------------------------------------------------------- scan parameters
    def read_scan(self):
        if self._refuse_while_busy():
            return
        st = self.settings()
        pattern_file = st.PATTERN_FILE
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
        if scan.get("pattern_width"):
            self.set_pattern_width(int(scan["pattern_width"]))
        self.prepared = None

    # ---------------------------------------------------------------- pixel size, folders
    def _pattern_file_dir(self):
        """Folder of the pattern file, for the scan file dialog; None when no pattern file is set."""
        text = self.pattern_file_edit.text().strip()
        return os.path.dirname(text) if text else None

    def set_pattern_width(self, width: int):
        self.pattern_width = int(width)
        self.auto_px_size()

    def auto_px_size(self):
        """Pixel size = 30 mm / pattern width for the EDAX/TSL convention (the EDAX detector is 30 mm wide)."""
        if not self.pattern_width or self.pc_convention_combo.currentText() not in ("edax", "tsl"):
            return
        px = DETECTOR_WIDTH_UM / self.pattern_width
        if abs(self.px_size_spin.value() - px) > 1e-6:
            self.px_size_spin.setValue(round(px, 4))
            self.log(f"Pixel size set to {px:.4f} um = {DETECTOR_WIDTH_UM / 1000:.0f} mm / {self.pattern_width} px (editable)")

    # ---------------------------------------------------------------- loading patterns and the quasi-IQ map
    def _load_key(self, st: Settings):
        # Only what changes the loaded patterns themselves. The scan file is deliberately not in here: setting it
        # after loading must not throw the loaded data set (and the live map) away.
        return ("file", st.PATTERN_FILE, st.NAV_SHAPE, st.STEP_SIZE, st.LAZY, st.ROI, st.FLIP_PATTERNS_UD)

    def unload_indexing(self, why: str = "", keep_run=None):
        """Drop the indexer's loaded patterns and everything derived from them (prepared signal, last run;
        ``keep_run`` is a run of the data set about to be loaded again, which stays)."""
        had = self.loaded is not None or self.prepared is not None
        self.loaded = None
        self.prepared = None
        self.last_run = keep_run
        self._preview_rc = None
        self.live.set_base_map(None, "")
        if keep_run is None:
            self.maps.clear()
        gc.collect()
        if had:
            self.log("Indexing tab: patterns unloaded" + (f" ({why})" if why else ""))

    def _unload_processor(self) -> bool:
        """Drop the processor's data set before the indexer loads one; False if the processor is busy."""
        proc = self.proc
        if proc is None or getattr(proc, "f_orig", None) is None:
            return True
        if getattr(proc, "_nlpar_busy", False):
            QMessageBox.information(self, "Processor busy", "Wait for the NLPAR step in the pattern processor to finish before loading")
            return False
        unload = getattr(proc, "unload", None)
        if unload is not None:
            ok = unload()
        else:
            # an older pattern processor without unload(): the same resets, done from here
            ok = self._unload_processor_fallback(proc)
        if not ok:
            QMessageBox.information(self, "Processor busy", "The pattern processor could not unload its data set now")
            return False
        self.log("Pattern processor: data set unloaded (a file is being loaded in the indexing tab)")
        return True

    @staticmethod
    def _unload_processor_fallback(proc) -> bool:
        """What MainWindow.unload() does, for a processor clone that predates it (every step guarded)."""
        try:
            proc.f_nlpar = None
            proc.nlpar_patterns = None
            if hasattr(proc, "nlpar_status"):
                proc.nlpar_status.setText("No NLPAR file loaded")
            if hasattr(proc, "_remove_temp_files"):
                proc._remove_temp_files()
            for name in ("orig_path", "f_orig", "patterns", "iq_map", "iq_mask", "mean_pattern", "pixel_std", "data_max",
                         "_nlobj", "_nlobj_key", "_nl_infile", "_aperture_cache", "_aperture_cache_key"):
                if hasattr(proc, name):
                    setattr(proc, name, None)
            proc.nrows = proc.ncols = proc.current_row = proc.current_col = 0
            for name in ("_invalidate_sigma_map", "_invalidate_global_range"):
                if hasattr(proc, name):
                    getattr(proc, name)()
            proc._lambda_optimized = False
            if hasattr(proc, "file_label"):
                proc.file_label.setText("No file loaded")
            if hasattr(proc, "bg_status"):
                proc.bg_status.setText("Background: all patterns")
            for name in ("scan_canvas", "raw_canvas", "proc_canvas"):
                canvas = getattr(proc, name, None)
                if canvas is not None and hasattr(canvas, "ax"):
                    canvas.ax.clear()
                    canvas.ax.axis("off")
                    canvas.draw_idle()
            if hasattr(proc, "scan_canvas"):
                proc.scan_canvas.marker = None
                proc.scan_canvas._mask = None
            gc.collect()
            return True
        except Exception:  # noqa: BLE001
            return False

    def load_patterns_clicked(self):
        """Load the pattern file, compute and show the quasi-IQ map. The processor's data set is unloaded first."""
        if self._refuse_while_busy():
            return
        st = self.settings()
        key = self._load_key(st)
        if self.loaded is not None and self.loaded["key"] == key:
            self._show_loaded()
            return
        if not self._unload_processor():
            return
        # a result of this very data set (indexed without 'Load patterns' first) is kept: it can still be saved
        keep_run = self.last_run if self.last_run is not None and self.last_run.get("load_key") == key else None
        self.unload_indexing(keep_run=keep_run)

        def work(log, emit):
            s, scan = sph_core.load_patterns(st, log=log)
            t0 = time.perf_counter()
            iq = sph_core.quasi_iq(s, progress=lambda f: emit("progress", {"frac_index": f, "frac_refine": 1.0, "what": "iq"}))
            static_bg = sph_core.sample_mean_pattern(s)
            log(f"Quasi-IQ map (per-pattern standard deviation) in {time.perf_counter() - t0:.1f} s")
            return {"key": key, "signal": s, "scan": scan, "iq": sph_core.stretch(iq), "source": "file", "static_bg": static_bg,
                    "source_file": st.PATTERN_FILE, "nav_shape": tuple(s.axes_manager.navigation_shape[::-1])}

        def done(loaded):
            self.loaded = loaded
            self.prepared = None
            sig_shape = loaded["signal"].axes_manager.signal_shape
            self.set_pattern_width(int(sig_shape[0]))
            self._show_loaded()

        self._start(work, done)

    def preview_point(self, row, col):
        """Show the pattern at (row, col) of the loaded file as the indexer will see it: the background
        removals and the equalization that are ticked, nothing else (the raw pattern when none is)."""
        loaded = self.loaded
        if loaded is None or loaded.get("signal") is None or self.live_run is not None:
            return
        s = loaded["signal"]
        nav_shape = tuple(s.axes_manager.navigation_shape[::-1])
        if len(nav_shape) != 2 or not (0 <= row < nav_shape[0] and 0 <= col < nav_shape[1]):
            return
        st = self.settings()
        try:
            raw = np.asarray(s.data[int(row), int(col)])
            # the background removals are applied here unless they were already done in place on the data set
            shown = sph_core.preview_pattern(raw, st, loaded.get("static_bg"),
                                             apply_background=not any(loaded.get("preprocessed", (False, False))))
        except Exception as e:  # noqa: BLE001
            self.log(f"Pattern preview failed: {type(e).__name__}: {e}")
            return
        self._preview_rc = (int(row), int(col))
        self.live.show_pattern(shown, int(row), int(col))
        self.live.set_marker(int(row), int(col))
        steps = [name for name, on in (("static background", st.REMOVE_STATIC_BACKGROUND), ("dynamic background", st.REMOVE_DYNAMIC_BACKGROUND),
                                       ("Gaussian background", st.GAUSSIAN_BACKGROUND), (f"AHE {st.N_REGIONS} x {st.N_REGIONS}", bool(st.N_REGIONS)))
                 if on]
        self.set_progress(f"Pattern ({row}, {col}): " + (", ".join(steps) if steps else "raw, no processing"))

    def _refresh_preview(self):
        if self._preview_rc is not None and self.live_run is None:
            self.preview_point(*self._preview_rc)

    def _show_loaded(self):
        if self.loaded is None:
            return
        iq = self.loaded["iq"]
        name = os.path.basename(self.loaded["source_file"]) if self.loaded["source_file"] else "nickel demo"
        if iq.ndim == 2:
            self.live.set_base_map(iq, f"Quasi-IQ of {name} ({iq.shape[1]} x {iq.shape[0]} points)")
        else:
            self.live.set_base_map(None, f"{name}: 1D map, no live view")
        self.view_tabs.setCurrentWidget(self.live)
        if iq.ndim == 2 and self.loaded.get("signal") is not None:
            self.preview_point(iq.shape[0] // 2, iq.shape[1] // 2)

    # ---------------------------------------------------------------- preparing patterns, master, detector
    def _prepare_key(self, st: Settings):
        return ("file", st.PATTERN_FILE, st.SCAN_FILE, st.EDAXH5_MAP, st.NAV_SHAPE, st.STEP_SIZE, st.LAZY, st.ROI,
                st.FLIP_PATTERNS_UD, st.REMOVE_STATIC_BACKGROUND, st.REMOVE_DYNAMIC_BACKGROUND)

    def prepare_patterns_in_gui(self, st: Settings):
        """The prepared patterns of the current settings when they are still valid, else None."""
        key = self._prepare_key(st)
        if self.prepared is not None and self.prepared["key"] == key:
            return self.prepared
        return None

    def _prepare_in_worker(self, st: Settings, prepared, log):
        """Loads the patterns from a file when needed, then the harmonics and the detector."""
        key = self._prepare_key(st)
        if prepared is None:
            loaded = self.loaded
            wanted = (bool(st.REMOVE_STATIC_BACKGROUND), bool(st.REMOVE_DYNAMIC_BACKGROUND))
            if loaded is not None and loaded["source"] == "file" and loaded["key"] == self._load_key(st) and loaded["signal"] is not None \
                    and loaded.get("preprocessed", (False, False)) in ((False, False), wanted):
                # the data set already in memory: the background removals are done in place on it (one copy of
                # the patterns, not a raw one and a prepared one), and the click preview knows that
                s, scan = loaded["signal"], loaded["scan"]
                if wanted != (False, False) and loaded.get("preprocessed", (False, False)) != wanted:
                    sph_core.preprocess(s, st, log)
                    loaded["preprocessed"] = wanted
            else:
                s, scan = sph_core.load_patterns(st, log=log)
                sph_core.preprocess(s, st, log)
                if loaded is not None and loaded["source"] == "file" and loaded["key"] == self._load_key(st):
                    # the raw copy was preprocessed differently before: replace it rather than keep two
                    loaded["signal"] = s
                    loaded["preprocessed"] = wanted
            prepared = {"key": key, "signal": s, "scan": scan, "source": "file", "source_file": st.PATTERN_FILE}
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

    # ---------------------------------------------------------------- actions
    def _will_load_file(self, st: Settings, prepared) -> bool:
        """True when the worker is going to read the pattern file (nothing usable is in memory)."""
        if prepared is not None:
            return False
        loaded = self.loaded
        return loaded is None or loaded.get("signal") is None or loaded["key"] != self._load_key(st)

    def check_geometry(self):
        if self._refuse_while_busy():
            return
        st = self.settings()
        try:
            prepared = self.prepare_patterns_in_gui(st)
        except Exception as e:  # noqa: BLE001
            self.log(f"{type(e).__name__}: {e}")
            QMessageBox.critical(self, "Patterns", f"{type(e).__name__}: {e}")
            return
        if self._will_load_file(st, prepared) and not self._unload_processor():
            return  # one data set in memory at a time, as 'Load patterns' does

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
        if self._refuse_while_busy():
            return
        st = self.settings()
        try:
            prepared = self.prepare_patterns_in_gui(st)
        except Exception as e:  # noqa: BLE001
            self.log(f"{type(e).__name__}: {e}")
            QMessageBox.critical(self, "Patterns", f"{type(e).__name__}: {e}")
            return
        if self._will_load_file(st, prepared) and not self._unload_processor():
            return  # one data set in memory at a time, as 'Load patterns' does

        live = self.live_check.isChecked()
        if self.loaded is None or self.loaded["key"] != self._load_key(st):
            # the quasi-IQ map of this data set is not there yet: the live view then shows a blank map
            self.loaded = None
            self.live.set_base_map(None, "")
            if live:
                self.log("NB! Click 'Load patterns (quasi-IQ map)' first to see the map fill in during indexing")
        self.live_run = None

        def work(log, emit):
            result = self._prepare_in_worker(st, prepared, log)
            prepared_, harmonics, hkey, det, demo = result
            s = prepared_["signal"]
            if test:
                s_run, s_warm, (c0, r0) = sph_core.test_region(s, int(st.TEST_SIZE))
                label, offset = "test region", (r0, c0)
            else:
                s_run, s_warm, label, offset = s, None, "full map", (0, 0)
            nav_shape = tuple(int(v) for v in s_run.axes_manager.navigation_shape[::-1])
            n_run = int(np.prod(nav_shape))
            log(f"Indexing the {label}: {n_run} patterns, bandwidth {st.BANDWIDTH}, {st.BACKEND}")
            emit("run_start", {"r0": offset[0], "c0": offset[1], "rows": nav_shape[0], "cols": nav_shape[-1] if len(nav_shape) == 2 else 1,
                               "n_patterns": n_run, "signal": s_run, "label": label, "st": st})
            if live and len(nav_shape) == 2:
                def on_event(name, data):
                    if name == "block_done":
                        data = dict(data)
                        xb = data.pop("xmap")
                        data["rgb"] = None if xb is None else sph_core.block_rgb(xb, data, "z")
                    emit(name, data)

                # a test region is small: shorter bands, so that the animation shows several of them
                xmap_indexed, dt, speed = sph_core.index_blocks(
                    s_run, harmonics, det, st, log, on_event=on_event,
                    target_seconds=1.0 if test else LIVE_BLOCK_SECONDS,
                    first_patterns=max(nav_shape[1], min(300, max(1, n_run // 10))))
                xmap = sph_core.apply_twist(xmap_indexed, st.TWIST)
                log(f"{xmap.size} patterns in {dt:.1f} s; steady speed {speed:.0f} patterns/s ({st.BACKEND}, bandwidth {st.BANDWIDTH})")
                if st.TWIST:
                    log(f"Orientations rotated by {st.TWIST} deg about the sample normal (TWIST)")
            else:
                xmap, xmap_indexed, dt = sph_core.index(s_run, harmonics, det, st, log, warm_up=s_warm)
                speed = xmap.size / dt
            emit("run_end", {"success": True})
            if test:
                n_all = int(np.prod(s.axes_manager.navigation_shape))
                log(f"Estimated time for the full map of {n_all} patterns: {n_all / speed / 60:.1f} min at {speed:.0f} patterns/s")
            scan = prepared_.get("scan") or {}
            if st.ROI is None:
                full_nav_shape = tuple(int(v) for v in s.axes_manager.navigation_shape[::-1])
            elif scan.get("n_rows") and scan.get("n_cols"):
                full_nav_shape = (int(scan["n_rows"]), int(scan["n_cols"]))
            else:
                full_nav_shape = None
            maps = None
            if xmap.ndim == 2:  # the result maps, made here rather than on the GUI thread (seconds for a large map)
                maps = {"scores": sph_core.full_map_data(xmap, "scores"), "iq": sph_core.full_map_data(xmap, "iq"),
                        "ipf_z": sph_core.ipf_rgb_map(xmap, "z"), "ipf_x": sph_core.ipf_rgb_map(xmap, "x")}
            return result, dict(label=label, signal=s_run, xmap=xmap, xmap_indexed=xmap_indexed, det=det, seconds=dt, offset=offset,
                                source_file=st.PATTERN_FILE, load_key=self._load_key(st), full_nav_shape=full_nav_shape, maps=maps,
                                settings=st)

        def done(out):
            result, run = out
            self._store_prepared(result)
            self.last_run = run
            self.show_indexing_maps(run)
            # the Live tab keeps the finished IPF map in front; the Results tab has the score, IQ and IPF maps
            self.log("Done: the Results tab has the score, IQ, IPF-Z and IPF-X maps")

        self._start(work, done, on_event=self.on_run_event)

    # ---------------------------------------------------------------- live view plumbing
    def _live_pattern(self, row, col):
        """One pattern of the running region, for the live view (called in the GUI thread, a few times per second)."""
        run = self.live_run
        if run is None:
            return None
        s = run["signal"]
        if s.axes_manager.navigation_dimension == 1:  # a line scan or a one-row tile: shown as one row
            data = np.asarray(s.data[int(col) - run["c0"]])
        else:
            data = np.asarray(s.data[int(row) - run["r0"], int(col) - run["c0"]])
        # the prepared signal already carries the background removals; only the indexer's own steps remain
        return sph_core.preview_pattern(data, run["st"], None, apply_background=False)

    def on_run_event(self, name, data):
        if name == "run_start":
            self.live_run = {"signal": data["signal"], "st": data["st"], "r0": data["r0"], "c0": data["c0"], "n_patterns": data["n_patterns"],
                             "done": 0, "t0": time.perf_counter(), "window": collections.deque(), "block": None}
            # the live view that shows this run: the indexing tab's, or the montage tab's own
            self._live_view = data.get("live_view", self.live)
            self._live_view.begin_run(data["r0"], data["c0"], data["rows"], data["cols"], data["n_patterns"])
            if self._live_view is self.live:
                self.view_tabs.setCurrentWidget(self.live)
            return
        if name == "progress" and data.get("what") == "iq":
            self.set_progress(f"Quasi-IQ map: {100 * data['frac_index']:.0f} %")
            return
        run = self.live_run
        if run is None:
            return
        r0, c0 = run["r0"], run["c0"]
        if name == "block_start":
            run["block"] = (data["r0"], data["r1"], data["n_patterns"])
            self._live_view.block_started(r0 + data["r0"], r0 + data["r1"], c0 + data["c0"], c0 + data["c1"], data["n_patterns"])
        elif name == "progress":
            self._live_view.block_progress(data["frac_index"], data["frac_refine"])
            if run["block"] is not None:
                # patterns done so far, with the coarse search of the current band counted by its fraction
                done = run["done"] + data["frac_index"] * run["block"][2]
                self._update_speed(run, done)
        elif name == "chunk_done":
            # coloured by the live view at its next redraw tick (not here, and not on the indexing thread)
            self._live_view.chunk_done(r0 + np.asarray(data["rows"]), c0 + np.asarray(data["cols"]), packed=data["packed"],
                                       colour=functools.partial(sph_core.packed_rgb, keys=data["keys"]))
        elif name == "block_done":
            n_block = int(data.get("n_patterns", data["done_patterns"] - run["done"]))
            run["done"] = data["done_patterns"]
            run["block"] = None
            self._live_view.block_done(r0 + data["r0"], r0 + data["r1"], c0 + data["c0"], c0 + data["c1"], data["rgb"],
                                       n_block, data["seconds"])
            self._live_view.set_progress(data["done_patterns"], data["total_patterns"], data["eta_seconds"])
            self._update_speed(run, data["done_patterns"])
            self.set_progress(
                f"{data['done_patterns']} / {data['total_patterns']} patterns, {data['patterns_per_second']:.0f} patterns/s in this band, "
                f"ETA {data['eta_seconds'] / 60:.1f} min")
        elif name == "run_end":
            self._live_view.end_run(bool(data.get("success", True)))
            self.live_run = None

    def _update_speed(self, run, done_patterns, window_seconds=3.0):
        """The 'now' speed: patterns completed over the last few seconds."""
        now = time.perf_counter()
        window = run["window"]
        window.append((now, float(done_patterns)))
        while len(window) > 2 and now - window[0][0] > window_seconds:
            window.popleft()
        if len(window) >= 2 and now - window[0][0] > 0.5:
            speed = (window[-1][1] - window[0][1]) / (now - window[0][0])
            self._live_view.set_speed(max(speed, 0.0))

    def show_indexing_maps(self, run):
        xmap = run["xmap"]
        maps = run.get("maps")
        if xmap.ndim != 2 or maps is None:
            self.log("The map is not 2D; nothing to plot")
            return
        self.maps.show_panels([
            ("Correlation score", maps["scores"], {}, "score"),
            ("Image quality", maps["iq"], {}, "IQ"),
            ("IPF-Z", maps["ipf_z"], {}, None),
            ("IPF-X", maps["ipf_x"], {}, None),
        ], f"{run['label'].capitalize()}: {xmap.size} patterns in {run['seconds']:.1f} s = {xmap.size / run['seconds']:.0f} patterns/s")

    def save_result(self):
        if self.last_run is None:
            QMessageBox.information(self, "Save", "Nothing indexed yet")
            return
        st = self.settings()
        # the file the result came from, whatever has been loaded or read since
        out_dir, stem = sph_core.output_location(st, self.last_run.get("source_file"))
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
        # the run's own ROI and the whole scan's shape (not the ROI'd signal's), so a reference of the whole scan
        # is cropped to the indexed region
        roi = run["settings"].ROI if run.get("settings") is not None else st.ROI
        full_nav_shape = run.get("full_nav_shape")

        def work(log):
            ang = sph_core.compare_with_reference(run["xmap_indexed"], st.REFERENCE_FILE, roi, full_nav_shape)
            good = ang[np.isfinite(ang)]
            if good.size == 0:
                raise ValueError("No indexed point to compare")
            log(f"Misorientation to the reference ({good.size} indexed points): median {np.median(good):.2f} deg, "
                f"p90 {np.percentile(good, 90):.2f} deg, fraction below 2 deg {np.mean(good < 2):.1%}")
            return ang

        def done(ang):
            if ang.ndim == 2:
                self.maps.show_panels([("Misorientation to the reference [deg]", ang, dict(vmax=5), "deg")])

        self._start(work, done)

    def closeEvent(self, event):
        if self.worker_running():
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
