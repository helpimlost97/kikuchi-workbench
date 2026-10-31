#!/usr/bin/env python3
"""Kikuchi workbench: the Kikuchi Pattern Processor and GPU spherical indexing in one window.

    python kikuchi_workbench.py
    python kikuchi_workbench.py --preset preset_RKD.txt      # start with the indexing settings of a preset file
    python kikuchi_workbench.py --no-processor               # the indexing and montage tabs only

Four tabs:

1. **Pattern processor**: the Kikuchi Pattern Processor (pattern_processor/gpu_pattern_viewer.py, which also
   runs on its own as a stand-alone window). Load a UP1/UP2,
   set the aperture, IQ range, NLPAR and the processing chain as usual. "Apply && Save" still writes a file; the
   saved file is then offered to the indexing tab automatically.
2. **Index a point** (point_tab.py): load a pattern file, click a point on its map (the scan file's IQ column,
   or a quasi-IQ map built on request), index that one pattern on the CPU and see it next to the master pattern
   simulated at the orientation found. The file fields are shared with the indexing tab.
3. **Spherical indexing**: the settings of sph_core.Settings as a form. The patterns come from a
   file (anything kikuchipy loads, memory-mapped when big): "Load processed patterns" takes the file the
   processor's "Apply && Save" just wrote (every pattern of it is indexed, the ones outside the processor's IQ
   range too: the processor writes them unprocessed), the browse button any other file. Check the geometry on
   one pattern, index a test region, index the full map, save .ang/.h5. Load and save presets.
4. **Montage batch** (montage_tab.py): processing and indexing for every tile of an EDAX APEX montage.

Only one data set is in memory at a time: loading a file in the processor unloads the indexer's patterns and
result, loading in the indexing tab unloads the processor. The long steps (loading, indexing, comparing) run on
a worker thread (``Worker``) with their console output routed to the log pane; the GUI thread only refuses new
actions while one runs.

The processor's own switches (``KPP_GPU=0``, ``KPP_THREADS``, ``KPP_IN_MEMORY``) apply inside the workbench too.

Needs the environment of README.md (the kikuchipy fork, CuPy for the GPU) plus the pattern processor's packages
(PyQt5, pyebsdindex, scipy, scikit-image).
"""
from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import dataclasses
import functools
import gc
import inspect
import io
import os
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------------------------
# The pattern processor: found and imported before Qt is used, since its module prints what it found
# ----------------------------------------------------------------------------------------------


def import_processor():
    """Import the pattern processor, ``pattern_processor/gpu_pattern_viewer.py``: (module, info) or (None, reason).

    Importing it prints what it found (GPU, threads) to the console and sets the numba thread count, which is why
    it happens before Qt and before sph_core are imported. A failed import (a missing package, typically) leaves
    the indexing and montage tabs working and shows the reason in place of the processor tab.
    """
    try:
        from pattern_processor import gpu_pattern_viewer

        return gpu_pattern_viewer, f"pattern_processor v{gpu_pattern_viewer.__version__}"
    except Exception as e:  # noqa: BLE001
        return None, f"could not import pattern_processor/gpu_pattern_viewer.py: {type(e).__name__}: {e}"


_args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_args.add_argument("--preset", default=None, help="preset file (NAME = value lines) to start with")
_args.add_argument("--no-processor", action="store_true", help="do not load the pattern processor tab")
# parse_known_args: the scripts that import this module pass their own argv through untouched
ARGS, _ = _args.parse_known_args()

if ARGS.no_processor:
    gpv, PROCESSOR_INFO = None, "pattern processor tab disabled (--no-processor)"
else:
    gpv, PROCESSOR_INFO = import_processor()

from PyQt5.QtCore import Qt, QThread, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QAction, QActionGroup, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QGridLayout, QLabel, QLayout, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import ebsd_params  # noqa: E402
import sph_core  # noqa: E402
import theme  # noqa: E402
from live_view import LiveView  # noqa: E402
from montage_tab import MontageTab  # noqa: E402
from point_tab import PointTab  # noqa: E402
from sph_core import Settings  # noqa: E402

__version__ = "0.4.0"

# Every default of the indexing form is the ``sph_core.Settings`` default (``SettingsForm`` reads them from the
# ``Settings`` it is built with); the choices of the combo boxes and the ranges are sph_core's constants too.
# The file dialog filters, shared with the point tab:
PATTERN_FILE_FILTER = "Pattern files (*.up1 *.up2 *.h5 *.hdf5 *.oh5 *.edaxh5 *.ebsp *.dat);;All files (*)"
MASTER_FILE_FILTER = "Master patterns (*.sht *.h5 *.hdf5);;All files (*)"
# What a GUI preset holds: the indexing parameters only, never files, folders or the projection center values
PRESET_FIELDS = [
    "MASTER_BANDWIDTH", "PC_CONVENTION", "SAMPLE_TILT", "CAMERA_TILT", "PX_SIZE", "TWIST",
    "REMOVE_STATIC_BACKGROUND", "REMOVE_DYNAMIC_BACKGROUND",
    "BANDWIDTH", "N_WORKERS", "REFINE", "N_REGIONS", "CIRCULAR_MASK", "GAUSSIAN_BACKGROUND", "NORMALIZE", "CHUNKSIZE",
    "TEST_SIZE",
]


# ----------------------------------------------------------------------------------------------
# Worker thread with its console output routed to the log
# ----------------------------------------------------------------------------------------------


class _Tee:
    """A stand-in for sys.stdout / sys.stderr while a Worker runs.

    Everything written still goes to the original stream; in addition, complete lines (ended by ``\\n``) are
    handed to ``on_line`` and ``\\r``-terminated pieces (the carriage-return progress bars of tqdm and dask) to
    ``on_progress``, both without the terminator and without blank pieces. The split lets the GUI show a progress
    bar in one status label instead of appending every update to the log.

    Used from the Worker thread; the callbacks are signal ``emit`` methods, so they are thread safe.
    """

    def __init__(self, original, on_line, on_progress):
        self.original = original
        self.on_line = on_line
        self.on_progress = on_progress
        self.buffer = ""

    def write(self, text):
        """Write ``text`` to the original stream and dispatch every complete line or ``\\r`` piece in it."""
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
        """Flush the original stream; a partial line stays in the buffer until its terminator arrives."""
        try:
            self.original.flush()
        except Exception:  # noqa: BLE001
            pass

    def __getattr__(self, name):
        """Everything else (encoding, isatty, fileno, ...) is the original stream's."""
        return getattr(self.original, name)


class Worker(QThread):
    """A long step of the GUI on its own thread, with its console output and typed events sent to the GUI thread.

    ``fn`` is called as ``fn(log)`` or, when it takes two or more parameters, ``fn(log, emit)``:

    - ``log(text)`` appends a line to the log pane (the ``log`` signal);
    - ``emit(name, data)`` delivers an event (the ``event`` signal): a name such as ``run_start``, ``block_start``,
      ``progress``, ``chunk_done``, ``block_done``, ``run_end`` and a dict payload; see ``Workbench.on_run_event``
      for the payloads.

    Everything ``fn`` (or the libraries it calls) prints goes to the ``log`` signal line by line, and
    ``\\r``-updated progress bars to the ``progress`` signal (see ``_Tee``). ``fn``'s return value is sent with
    ``done``; an exception is caught and its traceback sent as text with ``failed``. ``fn`` runs off the GUI thread
    and must not touch widgets; it may read the data it was given.

    A finished Worker must not be deleted (``deleteLater`` or dropping the last reference) with PyQt5 5.15: Qt then
    raises "native Qt signal is not callable" from the connected slots. ``Workbench._drop_finished_workers``
    detaches finished workers instead.
    """

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
        """Thread body: swap the console streams for tees, call ``fn``, send ``done`` or ``failed``.

        sys.stdout / sys.stderr are process-wide, so while a worker runs the GUI thread's own prints are routed
        to the log as well; they are restored when ``fn`` returns.
        """
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
    """A form field's text as a Python value: a literal (number, tuple, list, None, ...) when it parses as one,
    else the text itself (a bare path or word); blank gives ``default``."""
    text = text.strip()
    if text == "":
        return default
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text  # a bare path or word


def _to_text(value) -> str:
    """The inverse of ``_literal`` for a form field: None is blank, a string is kept as it is, anything else is
    its repr (so that it parses back)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return repr(value)


def _fit(widget, min_width=60):
    """Let a field shrink to the width its column has, instead of demanding the width of its contents.

    With QSizePolicy.Ignored the layout decides the width, so long paths and long combo texts are elided or
    scroll inside the box rather than pushing the panel wider than the pane it sits in. Returns the widget.
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
    """A settings panel in a scroll area and a work area next to it, with a draggable divider between them.

    ``left_width`` is the panel's initial width in pixels; neither pane can be collapsed, and the work area takes
    all extra width when the window grows.
    """
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


#: ``SettingsForm.add_*`` default: take the field's default from the ``Settings`` the form was built with.
FROM_DEFAULTS = object()


class SettingsForm(QWidget):
    """Widgets for a subset of the ``Settings`` fields, in titled groups.

    Each ``add_*`` method adds one field under its ``Settings`` name and records a ``(getter, setter)`` pair in
    ``_fields`` (``getter()`` gives the field's value as a Python object, ``setter(value)`` fills the widget).
    The widget starts at the field's default in ``defaults`` (a ``Settings``; so a default is changed in
    ``sph_core.Settings`` only), unless ``default=`` is given: a form of names that are not settings (the montage
    tab's) passes its own or gets the widget's natural blank (None, False).
    ``settings(base)`` and ``set_settings(st)`` convert the whole form from and to a ``Settings`` object; other
    modules and scripts reach single fields through ``_fields[name]``. Fields are added to the group opened by the
    last ``group()`` call.
    """

    def __init__(self, defaults: Settings | None = None, parent=None):
        super().__init__(parent)
        self.defaults = defaults
        self._fields: dict[str, tuple] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        # The panel takes the width the scroll area gives it, whatever the longest label would like; the
        # fields shrink (see _fit), and only a very narrow panel clips the end of a long checkbox text.
        self._layout.setSizeConstraint(QLayout.SetNoConstraint)
        self._form = None

    def minimumSizeHint(self):  # noqa: N802 (Qt name)
        """Qt override: a fixed 200 px minimum width.

        The scroll area never makes the panel narrower than this hint; the layout's own minimum (the longest
        row) would pin the panel at that width and let the splitter clip it behind the right pane.
        """
        hint = super().minimumSizeHint()
        hint.setWidth(200)
        return hint

    def group(self, title: str) -> QFormLayout:
        """Open a new titled group; the following ``add_*`` calls put their rows into it."""
        box = QGroupBox(title)
        self._form = QFormLayout(box)
        self._form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # a long label goes above its field when the panel is narrow, instead of squeezing the field
        self._form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self._form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._layout.addWidget(box)
        return self._form

    def finish(self):
        """Close the form: the groups stay at the top when the panel is taller than they are."""
        self._layout.addStretch()

    def add_widget(self, widget):
        """Add a widget that is not a setting (a button, a note) as a full row of the current group."""
        if isinstance(widget, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
            _fit(widget)
        elif isinstance(widget, QLabel):
            widget.setWordWrap(True)
        self._form.addRow(widget)

    def _default(self, name, given, fallback=FROM_DEFAULTS):
        """The start value of field ``name``: ``given`` unless it is ``FROM_DEFAULTS``, then the ``defaults``
        object's value, then ``fallback``; a field with none of them is an error."""
        if given is not FROM_DEFAULTS:
            return given
        if self.defaults is not None and hasattr(self.defaults, name):
            return getattr(self.defaults, name)
        if fallback is FROM_DEFAULTS:
            raise ValueError(f"{name}: no default in the form's Settings and none given")
        return fallback

    # one line of a group per setting
    def add_bool(self, name, label, tip="", default=FROM_DEFAULTS):
        """A checkbox for a bool setting; returns the QCheckBox."""
        w = QCheckBox(label)
        w.setChecked(bool(self._default(name, default, False)))
        w.setToolTip(tip)
        self._form.addRow(w)
        self._fields[name] = (w.isChecked, w.setChecked)
        return w

    def add_int(self, name, label, lo=0, hi=100000, tip="", default=FROM_DEFAULTS):
        """A spin box for an int setting in ``lo..hi``; returns the QSpinBox."""
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(int(self._default(name, default)))
        w.setToolTip(tip)
        self._form.addRow(label, _fit(w))
        self._fields[name] = (w.value, lambda v: w.setValue(int(v)))
        return w

    def add_float(self, name, label, lo=-1e9, hi=1e9, decimals=4, tip="", default=FROM_DEFAULTS):
        """A spin box for a float setting in ``lo..hi`` with ``decimals`` places; returns the QDoubleSpinBox."""
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(float(self._default(name, default)))
        w.setToolTip(tip)
        self._form.addRow(label, _fit(w))
        self._fields[name] = (w.value, lambda v: w.setValue(float(v)))
        return w

    def add_text(self, name, label, tip="", placeholder="", literal=True, default=FROM_DEFAULTS):
        """Free text, parsed as a Python literal (blank = None): numbers, tuples such as (0, 100, 0, 50), paths.
        ``literal=False`` keeps the text as it is (blank = None), e.g. for a name filter such as '8'.
        Returns the QLineEdit."""
        w = QLineEdit(_to_text(self._default(name, default, None)))
        w.setToolTip(tip)
        w.setPlaceholderText(placeholder)
        self._form.addRow(label, _fit(w))
        getter = (lambda: _literal(w.text())) if literal else (lambda: (w.text().strip() or None))
        self._fields[name] = (getter, lambda v: w.setText(_to_text(v)))
        return w

    def add_combo(self, name, label, options, tip="", default=FROM_DEFAULTS):
        """A combo box over ``options``; the value has the type of the first option (so a list of ints gives an
        int, a list of strings the text). Returns the QComboBox."""
        w = QComboBox()
        w.addItems([str(o) for o in options])
        w.setCurrentText(str(self._default(name, default)))
        w.setToolTip(tip)
        self._form.addRow(label, _fit(w))
        kind = type(options[0]) if options and not isinstance(options[0], str) else None

        def get():
            return w.currentText() if kind is None else kind(w.currentText())

        self._fields[name] = (get, lambda v: w.setCurrentText(str(v)))
        return w

    def add_path(self, name, label, tip="", filter="All files (*)", save=False, directory=False, multi=False,
                 start_dir=None, on_chosen=None, default=FROM_DEFAULTS):
        """A path field with a browse button; returns the QLineEdit.

        ``directory`` browses for a folder, ``save`` for a file to write, ``multi`` for several files (the field
        then holds a Python list when more than one was picked, so the value is a list or a single path).
        ``start_dir`` is a callable returning the folder the dialog opens in when the field is still empty (e.g. the
        folder of another field's file); ``on_chosen(path)`` is called after a file was picked in the dialog, not
        when the field is typed into or set from code.
        """
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(_to_text(self._default(name, default, None)))
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
            # a path is kept as text even when it would parse as a literal; only a list of files is parsed
            text = edit.text().strip()
            if text == "":
                return None
            if text.startswith("["):
                return _literal(text)
            return text

        self._fields[name] = (get, lambda v: edit.setText(_to_text(v)))
        return edit

    def settings(self, base: Settings) -> Settings:
        """A copy of ``base`` with every field of the form set from its widget."""
        st = dataclasses.replace(base)
        for name, (get, _) in self._fields.items():
            setattr(st, name, get())
        return st

    def set_settings(self, st: Settings):
        """Fill every field of the form from ``st``."""
        for name, (_, set_) in self._fields.items():
            set_(getattr(st, name))

    def set_value(self, name: str, value):
        """Fill the one field ``name`` with ``value``."""
        self._fields[name][1](value)

    def has(self, name):
        """True when the form has a field for the setting ``name``."""
        return name in self._fields


# ----------------------------------------------------------------------------------------------
# Map plots
# ----------------------------------------------------------------------------------------------


class MapCanvas(QWidget):
    """A matplotlib figure with its navigation toolbar: the Results tab, the geometry check panels and the
    point tab's measured | simulated panels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(9, 4))   # colours from the appearance (theme.py)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def clear(self):
        """Empty the figure."""
        self.figure.clear()
        self.canvas.draw_idle()

    def show_panels(self, panels, title="", per_row=3):
        """Replace the figure by a grid of images, up to ``per_row`` per row.

        ``panels``: list of ``(title, image, imshow kwargs, colorbar label or None)``; an image is a 2D array or an
        RGB array. ``title`` is the figure's suptitle.
        """
        self.figure.clear()
        n = len(panels)
        ncols = min(n, int(per_row))
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
    """The main window: the four tabs and the state of the indexing tab (shared with the point tab).

    State of the indexing tab (all on the GUI thread; the worker closures read the dicts they were given):

    - ``loaded``: the pattern file as loaded by 'Load patterns', or None. Keys: ``key`` (``_load_key`` of the
      settings it was loaded with), ``signal`` (the kikuchipy EBSD signal, raw or, once ``preprocessed`` is set,
      with the background removals done in place), ``scan`` (the scan file's parameters, dict or None), ``iq``
      (the stretched map of the live views, 2D float array or 1D for a line scan; None until ``build_map`` made
      one), ``iq_source`` ("quasi" for the quasi-IQ map, "scan" for the scan file's IQ column, None without a
      map), ``static_bg`` (the sample mean pattern, for the click preview), ``source`` (always "file"),
      ``source_file`` (the pattern file path, None for the demo), ``nav_shape`` ((rows, cols)) and
      ``preprocessed`` ((static, dynamic) once a removal was done in place).
    - ``prepared``: the signal a step indexed, with keys ``key`` (``_prepare_key``), ``signal``, ``scan``, ``source``
      and ``source_file``; None when the next step must prepare again.
    - ``harmonics`` / ``harmonics_key``: the expanded master pattern(s) (list) and the settings they belong to
      (``sph_core.harmonics_key``), shared with the montage tab through ``harmonics_for``.
    - ``last_run``: the result of the last 'Index test region' / 'Index full map', see ``run_indexing``.
    - ``live_run``: the run the live view shows while it runs, see ``on_run_event``; None otherwise.
    - ``worker``: the last Worker started (scripts poll ``worker.isRunning()``); ``batch_running``: the montage
      tab's batch processing, which runs on the GUI thread, is in progress.
    - ``pattern_width``: pixels, once a pattern file is known; ``processed_file``: the file the processor's
      'Apply && Save' wrote last.

    Only one data set is in memory at a time: ``_unload_processor`` and ``unload_indexing`` empty the other tab
    before a file is loaded. The busy guards (``_refuse_while_busy`` for actions, ``_busy`` for the buttons) keep
    the loaded data from being changed under a running worker.
    """

    def __init__(self, preset: str | None = None):
        """Build the window; ``preset`` is a preset file to load into the indexing form at start."""
        super().__init__()
        # the appearance (dark by default, or the one chosen last) before any widget or figure is made
        theme.apply(QApplication.instance(), theme.current())
        self.setWindowTitle(f"Kikuchi workbench v{__version__}: pattern processor + spherical indexing")
        self.setGeometry(60, 60, 1500, 950)
        self.worker: Worker | None = None
        self._workers: list[Worker] = []  # workers not yet finished (detached by _drop_finished_workers)
        self.batch_running = False
        self.base = Settings()
        self.loaded = None
        self.prepared = None
        self.pattern_width = None
        self.processed_file = None
        self.live_run = None
        self.harmonics = None
        self.harmonics_key = None
        self.last_run = None
        self._preview_rc = None   # (row, col) of the pattern shown by the click preview, for refreshes
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.proc = None
        self.montage_tab = None   # log() is called before the montage and point tabs exist
        self.point_tab = None
        self._build_processor_tab()
        self._build_indexing_tab()
        self._live_view = self.live   # the live view of the current run: this tab's or the montage tab's
        self.point_tab = PointTab(self)   # after the indexing tab: its file fields are mirrored
        self.tabs.insertTab(1, self.point_tab, "2. Index a point")
        self.montage_tab = MontageTab(self)
        self.tabs.addTab(self.montage_tab, "4. Montage batch")
        self.tabs.setCurrentIndex(1 if self.proc is None else 0)
        self._build_menu()
        if preset:
            self.load_preset(preset)
        gpu_ok, gpu_text = sph_core.gpu_available()
        self.log(f"kikuchi_workbench {__version__}; pattern processor: {PROCESSOR_INFO}")
        self.log(f"GPU for spherical indexing: {gpu_text}")
        if not gpu_ok and self.form.has("BACKEND"):
            self.form.set_value("BACKEND", "cpu")

    # ---------------------------------------------------------------- menu bar
    def _build_menu(self):
        """The menu bar: *Appearance* with *Dark* (the default) and *Light*, exclusive and checkable. Choosing
        one switches every widget and figure at once (``theme.apply``) and is remembered for the next start.
        The actions are ``appearance_actions`` (mode -> QAction)."""
        menu = self.menuBar().addMenu("Appearance")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.appearance_actions = {}
        for mode, label in (("dark", "Dark"), ("light", "Light")):
            action = QAction(label, self, checkable=True)
            action.setChecked(theme.active() == mode)
            action.setToolTip(f"{label} appearance for the whole workbench; remembered for the next start")
            action.triggered.connect(lambda checked, m=mode: self.set_appearance(m))
            group.addAction(action)
            menu.addAction(action)
            self.appearance_actions[mode] = action

    def set_appearance(self, mode: str):
        """Switch to the ``"dark"`` or ``"light"`` appearance, tick its menu entry and remember the choice."""
        theme.apply(QApplication.instance(), mode, remember=True)
        theme.style_title_bar(self, mode)
        self.appearance_actions[mode].setChecked(True)
        self.log(f"Appearance: {mode}")

    def showEvent(self, event):  # noqa: N802 (Qt name)
        """Qt override: the title bar follows the appearance once the native window exists."""
        super().showEvent(event)
        theme.style_title_bar(self)

    # ---------------------------------------------------------------- tab 1: the processor
    def _build_processor_tab(self):
        """Add the processor's own MainWindow as tab 1, or a note when it was not found.

        Three of the processor's entry points are wrapped without changing its code: the file its 'Apply && Save'
        writes is captured (to offer it to the indexing tab); its ``load_path`` first unloads the indexer's data
        set and, once the file is loaded, puts its path into the pattern file field shared by the point and
        indexing tabs (not during a montage batch, which loads tile after tile); and its 'Load UP1 / UP2' dialog
        opens on that shared pattern file, so the file loaded in the other tabs is one click away here. One data
        set stays in memory at a time: the path is shared, the patterns are loaded by the tab that needs them.
        """
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

        # The processor asks for the output path with its module's QFileDialog.getSaveFileName; a subclass that
        # records the answer is put in its place, since the processor does not report the path it wrote.
        base_dialog = gpv.QFileDialog
        recorder = {"path": None}

        class _RecordingFileDialog(base_dialog):
            @staticmethod
            def getSaveFileName(*a, **k):
                result = base_dialog.getSaveFileName(*a, **k)
                recorder["path"] = result[0] if result else None
                return result

            @staticmethod
            def getOpenFileName(*a, **k):
                # the processor's 'Load UP1 / UP2' dialog (caption, blank start folder) opens on the pattern file
                # shared by the point and indexing tabs: that file pre-selected, or its folder for another type
                a = list(a)
                if len(a) >= 3 and a[1] == "Open UP1/UP2 File" and not a[2]:
                    shared = self.pattern_file_edit.text().strip()
                    if shared:
                        a[2] = shared
                return base_dialog.getOpenFileName(*a, **k)

        gpv.QFileDialog = _RecordingFileDialog
        original_apply_and_save = self.proc.apply_and_save

        def apply_and_save_then_register():
            recorder["path"] = None
            original_apply_and_save()
            path = recorder["path"]
            if path:
                # the processor writes with the extension of the source file, whatever the dialog's answer had
                path = self.proc.output_path_for(path, log=False)
                if os.path.exists(path):
                    self._processor_saved(path)

        self.proc.apply_save_btn.clicked.disconnect()
        self.proc.apply_save_btn.clicked.connect(apply_and_save_then_register)

        # Only one data set in memory at a time: loading into the processor drops the indexer's patterns
        original_load_path = self.proc.load_path

        def load_path_exclusive(path):
            # Only a running worker refuses, not worker_running(): the montage batch processing itself loads
            # every tile through load_path while batch_running is set.
            if self.worker is not None and self.worker.isRunning():
                QMessageBox.information(self, "Busy", "Wait for the indexing tab's running step to finish before loading")
                return
            self.unload_indexing("a file is being loaded in the pattern processor")
            original_load_path(path)
            # the loaded file becomes the pattern file of the point and indexing tabs (their field is shared)
            if not self.batch_running and self.proc.orig_path == path and self.pattern_file_edit.text().strip() != path:
                self.pattern_file_edit.setText(path)
                self.log(f"Pattern file of the point and indexing tabs set to {path} ('Load patterns' there loads it)")

        self.proc.load_path = load_path_exclusive

    def _processor_saved(self, path):
        """Called after the processor's Apply && Save: offers the written file to the indexing tab. Every pattern of
        it is indexed; the ones outside the processor's IQ range are in the file unprocessed."""
        self.processed_file = path
        self.btn_load_processed.setEnabled(True)
        self.btn_load_processed.setToolTip(f"Loads {path}")
        self.log(f"Processor saved {path}: 'Load processed patterns' in the indexing tab loads it")

    def load_processed_clicked(self):
        """Put the file Apply && Save wrote into the pattern file box and load it (refused while a step runs)."""
        if self._refuse_while_busy():
            return
        if not self.processed_file or not os.path.exists(self.processed_file):
            QMessageBox.information(self, "Processed patterns", "Run 'Apply && Save' in the pattern processor tab first")
            return
        self.form.set_value("PATTERN_FILE", self.processed_file)
        self.prepared = None
        self.load_patterns_clicked()

    # ---------------------------------------------------------------- tab 3: spherical indexing
    def _build_indexing_tab(self):
        """Build tab 3: the settings form on the left, the buttons, live view, result maps and log on the right.
        Keeps the widgets the point tab mirrors: the pattern, scan and master file edits (``pattern_file_edit``,
        ``scan_file_edit``, ``master_file_edit``), the projection center edit (``pc_edit``), the bandwidth combo
        (``bandwidth_combo``), the AHE tiles spin box (``w_regions``), the circular mask box (``w_circular``), the
        Gaussian background box (``w_gaussian``), the
        PC convention combo (``pc_convention_combo``), the sample tilt edit (``sample_tilt_edit``), the camera tilt
        spin box (``camera_tilt_spin``) and the pixel size spin box (``px_size_spin``); the stop button
        (``btn_stop``, enabled only while a run of ``run_indexing`` is in progress); and the tab widget as
        ``indexing_tab``."""
        tab = QWidget()
        outer = QHBoxLayout(tab)
        self.form = form = SettingsForm(self.base)

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
            filter=PATTERN_FILE_FILTER, on_chosen=lambda path: self.load_patterns_clicked())
        self.scan_file_edit = form.add_path(
            "SCAN_FILE", "Scan file (.ang/.edaxh5):", tip="The .ang or .edaxh5 written at acquisition: gives the projection center, map size and step",
            filter="Scan files (*.ang *.edaxh5 *.h5);;All files (*)", start_dir=self._pattern_file_dir)
        form.add_text("EDAXH5_MAP", "edaxh5 map filter:", tip="Only for .edaxh5 files with several maps: part of the map's name, e.g. 'Area 8'",
                      placeholder="None", literal=False)
        self.btn_load = QPushButton("Load patterns")
        self.btn_load.setToolTip("Loads the pattern file (one data set in memory at a time) and fills the pixel size; done "
                                 "automatically after browsing for a file. The Live view then shows a black map of the scan's "
                                 "shape to click on; 'Build quasi-IQ map' puts the quasi-IQ map there")
        self.btn_load.clicked.connect(lambda: self.load_patterns_clicked())
        form.add_widget(self.btn_load)
        self.btn_quasi = QPushButton("Build quasi-IQ map")
        self.btn_quasi.setToolTip("The quasi image quality map of the loaded patterns (the standard deviation of every raw "
                                  "pattern, one read of the file) as the Live view's map; not needed for indexing")
        self.btn_quasi.clicked.connect(lambda: self.build_map("quasi"))
        form.add_widget(self.btn_quasi)
        self.btn_scan = QPushButton("Read scan parameters")
        self.btn_scan.setToolTip("Reads the scan and pattern file headers, prints them in the log and fills the projection center")
        self.btn_scan.clicked.connect(self.read_scan)
        form.add_widget(self.btn_scan)
        form.add_text("NAV_SHAPE", "Map (rows, cols):", tip="Only when the pattern file does not store the map size; blank = from the file or the scan file", placeholder="None")
        form.add_text("STEP_SIZE", "Step (x, y) um:", tip="Only when the pattern file does not store it", placeholder="None")
        form.add_combo("LAZY", "Lazy loading:", sph_core.LAZY_CHOICES, tip="auto = memory-map files bigger than 25 % of RAM")
        form.add_text("ROI", "ROI (c0, c1, r0, r1):", tip="Index only this part of the map: (col_start, col_stop, row_start, row_stop)", placeholder="None = everything")
        form.add_bool("FLIP_PATTERNS_UD", "Flip patterns upside down", "Try this when the test maps look like noise")

        form.group("Pattern processing (off for a file the processor already processed)")
        self.w_static = form.add_bool("REMOVE_STATIC_BACKGROUND", "Remove static background", "The background stored with the file, or else the mean pattern. Off by default: patterns from the processor are already corrected")
        self.w_dynamic = form.add_bool("REMOVE_DYNAMIC_BACKGROUND", "Remove dynamic background", "Off by default: patterns from the processor are already corrected")
        note = QLabel("NLPAR: use the pattern processor tab (pyebsdindex)")
        note.setStyleSheet("color: gray")
        form.add_widget(note)

        form.group("Master pattern(s)")
        self.master_file_edit = form.add_path(
            "MASTER_FILE", "Master file(s):", tip="EMSphInx .sht or EMsoft master pattern .h5; several files (one per phase) as a list ['a.sht', 'b.sht']",
            filter=MASTER_FILE_FILTER, multi=True)
        form.add_int("MASTER_BANDWIDTH", "Master bandwidth (.h5):", 16, 1024,
                     f"Bandwidth of the harmonics computed from an EMsoft .h5 master pattern; keep >= {self.base.MASTER_BANDWIDTH}")
        form.add_text("MASTER_ENERGY", "Master energy keV (.h5):", tip="Only for .h5 masters with several energies; blank = all, weighted", placeholder="None")
        form.add_bool("OVERRIDE_MASTER_SAMPLE_TILT", "Override master sample tilt", "Use the sample tilt below even when the master pattern was simulated for another one; indexing refuses the mismatch otherwise")

        form.group("Detector geometry")
        self.pc_edit = form.add_text("PC", "Projection center (x*, y*, z*):", "Values from an .ang file are in the EDAX convention")
        self.pc_convention_combo = form.add_combo("PC_CONVENTION", "PC convention:", sph_core.PC_CONVENTIONS)
        self.pc_convention_combo.currentTextChanged.connect(lambda *_: self.auto_px_size())
        self.sample_tilt_edit = form.add_text("SAMPLE_TILT", "Sample tilt deg:", tip="Blank = the tilt the master pattern was simulated for (as EMSphInx does)", placeholder="None = from master")
        self.camera_tilt_spin = form.add_float("CAMERA_TILT", "Camera tilt deg:", -180, 180, 2, "Detector tilt / camera elevation (EMSphInx thetac): 8 for EBSD, -90 for RKD")
        self.px_size_spin = form.add_float("PX_SIZE", "Pixel size um:", 1e-6, 1e6, 4,
                                           f"Detector pixel size. Filled automatically as {ebsd_params.DETECTOR_WIDTH_UM / 1000:.0f} mm / pattern width "
                                           "for the EDAX/TSL convention once the pattern width is known; editable")
        form.add_float("TWIST", "Twist deg:", -180, 180, 2, "Rotation of the detector about its normal, e.g. 15 for RKD; applied to the orientations afterwards")

        form.group("Indexing")
        self.bandwidth_combo = form.add_combo("BANDWIDTH", "Bandwidth:", sph_core.FAST_BANDWIDTHS,
                                              f"{self.base.BANDWIDTH} = EMSphInx default; 88 or 123 for more accuracy at lower speed (fast bandwidths only)")
        form.add_combo("BACKEND", "Backend:", sph_core.BACKENDS)
        form.add_int("N_WORKERS", "Dask workers:", 1, 256, "Raise to about the number of performance cores")
        form.add_bool("REFINE", "Newton refinement")
        self.w_regions = form.add_int("N_REGIONS", "AHE tiles (n_regions):", 0, sph_core.N_REGIONS_MAX, "Adaptive histogram equalization tiles per axis; 0 = off")
        self.w_circular = form.add_bool("CIRCULAR_MASK", "Circular mask", "Use only the inscribed circle of the detector")
        self.w_gaussian = form.add_bool("GAUSSIAN_BACKGROUND", "Gaussian background", "Fit and subtract a 2D Gaussian background (EMSphInx gausbckg)")
        form.add_bool("NORMALIZE", "Normalized cross-correlation")
        form.add_text("CHUNKSIZE", "Chunk / GPU batch size:", tip="Patterns per chunk; blank = automatic from free VRAM", placeholder="None = automatic")
        form.add_int("TEST_SIZE", "Test region side:", 1, 10000, "Side length in map points of the test region")

        form.group("Output")
        form.add_path("OUTPUT_DIR", "Output folder:", tip="Blank = next to the pattern file", directory=True)
        form.add_text("OUTPUT_NAME", "Output name:", tip="Blank = <pattern file>_sph_bw<bandwidth>", placeholder="None")
        form.add_bool("SAVE_ANG", "Save .ang")
        form.add_bool("SAVE_H5", "Save .h5")
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
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("QPushButton:enabled { background-color: #C62828; color: white; font-weight: bold; padding: 6px; }")
        self.btn_stop.setToolTip("Stop the running indexing after the band in progress: the bands already indexed are discarded "
                                 "and nothing is saved. Greyed out when no indexing runs")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_indexing)
        self._stop_event = threading.Event()
        self.btn_save = QPushButton("Save .ang / .h5")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_compare = QPushButton("Compare with reference")
        self.btn_compare.clicked.connect(self.compare_reference)
        rl.addLayout(button_rows([self.btn_preset_load, self.btn_preset_save, self.btn_geometry, self.btn_compare,
                                  self.btn_test, self.btn_full, self.btn_stop, self.btn_save]))
        self.live_check = QCheckBox("Live view: index band by band and animate the progress on the map")
        self.live_check.setChecked(True)
        self.live_check.setToolTip("Off = one call for the whole map (no animation, a few per cent faster on tiny maps)")
        rl.addWidget(self.live_check)
        split = QSplitter(Qt.Vertical)
        self.view_tabs = QTabWidget()
        self.live = LiveView()
        self.live.error.connect(self.log)
        self.live.set_pattern_source(self._live_pattern)
        self.live.point_clicked.connect(self.preview_point)
        # the click preview shows the pattern as the indexer will see it, so it follows these boxes
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
        self.indexing_tab = tab
        self.tabs.addTab(tab, "3. Spherical indexing")

    # ---------------------------------------------------------------- settings, presets, log
    def settings(self) -> Settings:
        """The indexing settings as the form holds them now, as a ``Settings`` object.

        The LAZY combo's "True"/"False" texts become bools ("auto" stays a string) and a 3-tuple PC becomes a
        tuple of floats, as sph_core expects them.
        """
        st = self.form.settings(self.base)
        if isinstance(st.LAZY, str) and st.LAZY != "auto":
            st.LAZY = st.LAZY == "True"
        if isinstance(st.PC, (list, tuple)) and len(st.PC) == 3:
            st.PC = tuple(float(v) for v in st.PC)
        return st

    def load_preset(self, path=None):
        """Load the indexing parameters of a preset file (``PRESET_FIELDS``) into the form.

        ``path`` None opens a file dialog. Files, folders and the projection center values are never taken from a
        preset; the names that changed are listed in the log. Refused while a step runs.
        """
        if self._refuse_while_busy():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load preset", str(HERE), "Presets (*.txt);;All files (*)")
            if not path:
                return
        current = self.settings()
        st, ignored = sph_core.read_preset(path, current)
        for name in st.names():
            if name not in PRESET_FIELDS:
                setattr(st, name, getattr(current, name))
        self.form.set_settings(st)
        applied = [n for n in PRESET_FIELDS if getattr(st, n) != getattr(current, n)]
        self.log(f"Preset loaded: {path}; changed: {', '.join(applied) if applied else 'nothing'}"
                 + " (files, folders and the projection center are never taken from a preset)")
        # what the file had that was not applied: names that are not settings or whose value could not be read
        # (``ignored``), and settings a GUI preset does not carry (files, folders, PC, and the rest outside PRESET_FIELDS)
        not_carried = [n for n in sph_core.preset_keys(path) if n in st.names() and n not in PRESET_FIELDS]
        if ignored or not_carried:
            self.log("  not applied: " + ", ".join(ignored + not_carried))

    def save_preset(self):
        """Write the ``PRESET_FIELDS`` of the current settings to a preset file chosen in a dialog."""
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", str(HERE / "preset_workbench.txt"), "Presets (*.txt)")
        if not path:
            return
        sph_core.write_preset(path, self.settings(), header="Indexing settings of the Kikuchi workbench (no files, folders or PC values)",
                              fields=PRESET_FIELDS)
        self.log(f"Preset saved: {path} ({len(PRESET_FIELDS)} indexing settings)")

    def log(self, text: str):
        """Append a line to the log pane (and the point and montage tabs'), scrolled to the end. GUI thread; the
        workers reach it through their ``log`` signal."""
        self.log_box.appendPlainText(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        for tab in (self.point_tab, self.montage_tab):
            if tab is not None:
                tab.log(text)

    def set_progress(self, text: str):
        """Show ``text`` (its last 200 characters) in the status line under the views, and in the point and
        montage tabs'."""
        self.progress_label.setText(text[-200:])
        for tab in (self.point_tab, self.montage_tab):
            if tab is not None:
                tab.set_progress(text)

    def _busy(self, busy: bool):
        """Disable (``busy``) or enable the action buttons of the indexing tab. 'Load processed patterns' is only
        enabled when the processor has written a file. The browse buttons stay enabled; ``_refuse_while_busy``
        guards what they trigger."""
        for b in (self.btn_geometry, self.btn_test, self.btn_full, self.btn_save, self.btn_compare,
                  self.btn_preset_load, self.btn_scan, self.btn_load, self.btn_quasi):
            b.setEnabled(not busy)
        self.btn_load_processed.setEnabled(not busy and bool(self.processed_file))
        if self.point_tab is not None:
            self.point_tab._busy(busy)

    def worker_running(self) -> bool:
        """True while a worker thread runs, or the montage tab's batch processing (GUI thread) is in progress."""
        if self.batch_running:
            return True
        try:
            return self.worker is not None and self.worker.isRunning()
        except RuntimeError:  # the Worker's Qt object is gone (the window is closing)
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
        """Run ``fn`` on a Worker thread; False (after telling the user) when a step is already running.

        ``fn(log)`` or ``fn(log, emit)`` runs on the worker thread (see ``Worker``). On the GUI thread afterwards:
        ``on_done(result)`` with its return value, or on an exception ``on_failed()`` (if given) after the live view
        of a running run was ended, then the traceback in the log and a message box. ``on_event(name, data)``
        receives the events ``fn`` emits; default ``on_run_event``. The buttons are disabled while the worker runs
        and the status line is cleared when it ends. An exception in ``on_done`` is logged and shown as well.
        """
        if self._refuse_while_busy():
            return False
        self._busy(True)
        worker = self.worker = Worker(fn, self)
        worker.log.connect(self.log)
        worker.progress.connect(self.set_progress)
        worker.event.connect(on_event or self.on_run_event)
        # a finished worker is detached and its function (whose closure holds the data set) dropped at once: as
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
        """'Read scan parameters': print the scan and pattern file headers in the log and fill the projection
        center (edax convention) and the pattern width (hence the pixel size) from them.

        The camera elevation of the scan file is only reported, never copied into CAMERA_TILT. The prepared signal
        is dropped, since the detector it was built for may have changed. GUI thread; refused while a step runs.
        """
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
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ebsd_params.print_scan_info(scan)
        self.log(buf.getvalue().rstrip())
        if scan.get("pc"):
            self.set_pc_from_scan(scan["pc"])
        if scan.get("camera_elevation") is not None:
            self.log(f"Camera elevation in the scan file: {scan['camera_elevation']} deg (CAMERA_TILT is not changed automatically)")
        if scan.get("pattern_width"):
            self.set_pattern_width(int(scan["pattern_width"]))
        self.prepared = None

    def set_pc_from_scan(self, pc, source: str = "the scan file"):
        """Put a projection center read from an EDAX scan or project (always in the EDAX convention) into the form:
        rounded to 6 places, with ``PC_CONVENTION`` set to ``"edax"``; logged. The montage tab uses it too."""
        self.form.set_value("PC", tuple(round(float(v), 6) for v in pc))
        self.form.set_value("PC_CONVENTION", "edax")
        self.log(f"Projection center set to {tuple(pc)} (edax convention) from {source}")

    def harmonics_for(self, st: Settings, log, demo: bool = False) -> tuple[list, tuple]:
        """The harmonics of ``st``'s master pattern(s) and their key: the cached ones (``harmonics``,
        ``harmonics_key``) when the key is unchanged, else ``sph_core.load_harmonics``. The caller stores what it
        gets back (on the GUI thread) when it wants them cached."""
        hkey = sph_core.harmonics_key(st, demo)
        if self.harmonics is not None and self.harmonics_key == hkey:
            return self.harmonics, hkey
        return sph_core.load_harmonics(st, log, demo=demo), hkey

    # ---------------------------------------------------------------- pixel size, folders
    def _pattern_file_dir(self):
        """Folder of the pattern file, for the scan file dialog; None when no pattern file is set."""
        text = self.pattern_file_edit.text().strip()
        return os.path.dirname(text) if text else None

    def set_pattern_width(self, width: int):
        """Record the pattern width in pixels and fill the pixel size from it."""
        self.pattern_width = int(width)
        self.auto_px_size()

    def auto_px_size(self):
        """Pixel size = detector width / pattern width for the EDAX/TSL convention (``ebsd_params.pixel_size_um``,
        the EDAX detector being 30 mm wide).

        Nothing happens for the other conventions or before the pattern width is known; a change is logged.
        """
        if not self.pattern_width or self.pc_convention_combo.currentText() not in ("edax", "tsl"):
            return
        px = ebsd_params.pixel_size_um(self.pattern_width)
        if abs(self.px_size_spin.value() - px) > 1e-6:
            self.px_size_spin.setValue(round(px, 4))
            self.log(f"Pixel size set to {px:.4f} um = {ebsd_params.DETECTOR_WIDTH_UM / 1000:.0f} mm / {self.pattern_width} px (editable)")

    # ---------------------------------------------------------------- loading patterns and the quasi-IQ map
    def _load_key(self, st: Settings):
        """What identifies the loaded patterns: the settings that change the patterns themselves.

        The scan file is deliberately not in here: setting it after loading must not throw the loaded data set
        (and the live map) away.
        """
        return ("file", st.PATTERN_FILE, st.NAV_SHAPE, st.STEP_SIZE, st.LAZY, st.ROI, st.FLIP_PATTERNS_UD)

    def unload_indexing(self, why: str = "", keep_run=None):
        """Drop the indexer's loaded patterns and everything derived from them (prepared signal, last run;
        ``keep_run`` is a run of the data set about to be loaded again, which stays).

        Clears the live view's base map, the point tab and, unless a run is kept, the result maps; logs it with
        ``why`` when there was something to unload. Used by the processor's load and the montage tab as well.
        """
        had = self.loaded is not None or self.prepared is not None
        self.loaded = None
        self.prepared = None
        self.last_run = keep_run
        self._preview_rc = None
        self.live.set_base_map(None, "")
        if self.point_tab is not None:
            self.point_tab.unload()
        if keep_run is None:
            self.maps.clear()
        gc.collect()
        if had:
            self.log("Indexing tab: patterns unloaded" + (f" ({why})" if why else ""))

    def _unload_processor(self) -> bool:
        """Drop the processor's data set before the indexer loads one; False (after telling the user) if the
        processor is busy with NLPAR or could not unload. True at once without a processor or with nothing loaded.
        """
        proc = self.proc
        if proc is None or proc.f_orig is None:
            return True
        if proc._nlpar_busy:
            QMessageBox.information(self, "Processor busy", "Wait for the NLPAR step in the pattern processor to finish before loading")
            return False
        if not proc.unload():
            QMessageBox.information(self, "Processor busy", "The pattern processor could not unload its data set now")
            return False
        self.log("Pattern processor: data set unloaded (a file is being loaded in the indexing tab)")
        return True

    def load_patterns_clicked(self, iq_source: str | None = None):
        """'Load patterns': load the pattern file on a worker and show it; a map only when asked for.

        ``iq_source``: the map built right after the load by ``build_map``: None (the default, the indexing
        tab's button and the loads after browsing: no map, the views show a black map of the scan's shape to
        click on, 'Build quasi-IQ map' makes the quasi-IQ map later), "scan" (the IQ column of the scan file, the
        point tab's load with a scan file) or "quasi" (the quasi-IQ map). Nothing is done when the same data set
        (``_load_key``) is already loaded, except showing it again, and building the map asked for when the
        loaded one is of another source or missing. The processor's data set
        is unloaded first (one data set at a time), then the indexer's own; a result of this very data set is
        kept, so it can still be saved. On the worker: ``sph_core.load_patterns`` and the sample mean pattern
        for the click preview. Done: ``loaded`` is set (without a map), the pattern width taken from the signal,
        the data set shown and the map started. Refused while a step runs.
        """
        if self._refuse_while_busy():
            return
        st = self.settings()
        key = self._load_key(st)
        if self.loaded is not None and self.loaded["key"] == key:
            if iq_source is not None and self.loaded.get("iq_source") != iq_source:
                self.build_map(iq_source)
            else:
                self._show_loaded()
            return
        if not self._unload_processor():
            return
        # a result of this very data set (indexed without 'Load patterns' first) is kept: it can still be saved
        keep_run = self.last_run if self.last_run is not None and self.last_run.get("load_key") == key else None
        self.unload_indexing(keep_run=keep_run)

        def work(log):
            # worker thread
            s, scan = sph_core.load_patterns(st, log=log)
            static_bg, scale_max = sph_core.sample_background_and_scale(s)
            log(f"{np.dtype(s.data.dtype).name} patterns, shown on the intensity scale 0-{int(scale_max)} (as in the pattern processor)")
            return {"key": key, "signal": s, "scan": scan, "iq": None, "iq_source": None, "source": "file",
                    "static_bg": static_bg, "scale_max": scale_max, "source_file": st.PATTERN_FILE,
                    "nav_shape": tuple(s.axes_manager.navigation_shape[::-1])}

        def done(loaded):
            self.loaded = loaded
            self.prepared = None
            sig_shape = loaded["signal"].axes_manager.signal_shape
            self.set_pattern_width(int(sig_shape[0]))
            self._show_loaded()
            if iq_source is not None:
                self.build_map(iq_source)

        self._start(work, done)

    def build_map(self, iq_source: str = "quasi"):
        """Build the map of the loaded data set on a worker and show it in the live views: ``iq_source``
        "quasi" = the quasi-IQ map (``sph_core.quasi_iq``, the standard deviation of every raw pattern, one read
        of the file, progress events ``what="iq"``), "scan" = the IQ column of the scan file
        (``sph_core.scan_iq_map``, which must be this scan's). Stored as ``loaded["iq"]`` / ``loaded["iq_source"]``
        (stretched). Refused while a step runs or without a loaded data set (a message says so)."""
        if self._refuse_while_busy():
            return
        loaded = self.loaded
        if loaded is None or loaded.get("signal") is None:
            QMessageBox.information(self, "Map", "Load patterns first")
            return
        st = self.settings()
        s, nav_shape = loaded["signal"], loaded["nav_shape"]

        def work(log, emit):
            # worker thread
            if iq_source == "scan":
                return sph_core.scan_iq_map(st, nav_shape, log)
            t0 = time.perf_counter()
            iq = sph_core.quasi_iq(s, progress=lambda f: emit("progress", {"frac_index": f, "frac_refine": 1.0, "what": "iq"}))
            log(f"Quasi-IQ map (per-pattern standard deviation) in {time.perf_counter() - t0:.1f} s")
            return sph_core.stretch(iq)

        def done(iq):
            if self.loaded is not loaded:  # unloaded meanwhile (the busy guards make this unlikely)
                return
            loaded["iq"], loaded["iq_source"] = iq, iq_source
            self._show_loaded()

        self._start(work, done)

    def preview_point(self, row, col):
        """Show the pattern at (row, col) of the loaded file as the indexer will see it: the background
        removals and the equalization that are ticked, nothing else (the raw pattern when none is).

        Slot of the ``point_clicked`` of both live views (this tab's and the point tab's); ignored while a run is
        shown or nothing is loaded, or when (row, col) is outside the map. Reads one pattern from the signal on
        the GUI thread (one slice of a memory map), shows it and marks the point in both views and names the
        applied steps in the status line.
        """
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
            # a fixed grey scale (0 = black, the data's full scale = white), as the pattern processor shows its
            # patterns: a raw pattern is seen exactly as measured, never stretched per pattern
            limits = sph_core.preview_limits(st, loaded.get("scale_max") or sph_core.intensity_full_scale(raw.dtype), shown)
        except Exception as e:  # noqa: BLE001
            self.log(f"Pattern preview failed: {type(e).__name__}: {e}")
            return
        self._preview_rc = (int(row), int(col))
        for view in self._preview_views():
            view.show_pattern(shown, int(row), int(col), limits)
            view.set_marker(int(row), int(col))
        steps = [name for name, on in (("static background", st.REMOVE_STATIC_BACKGROUND), ("dynamic background", st.REMOVE_DYNAMIC_BACKGROUND),
                                       ("Gaussian background", st.GAUSSIAN_BACKGROUND), (f"AHE {st.N_REGIONS} x {st.N_REGIONS}", bool(st.N_REGIONS)))
                 if on]
        self.set_progress(f"Pattern ({row}, {col}): " + (", ".join(steps) if steps else "raw, no processing"))

    def _refresh_preview(self):
        """Show the clicked pattern again with the processing boxes as they are now (not during a run)."""
        if self._preview_rc is not None and self.live_run is None:
            self.preview_point(*self._preview_rc)

    def _preview_views(self):
        """The views that show the loaded map, the clicked pattern and its marker (``set_base_map``,
        ``show_pattern``, ``set_marker``): this tab's live view and the point tab."""
        return [self.live] + ([self.point_tab] if self.point_tab is not None else [])

    def _show_loaded(self):
        """Put the loaded data set's map into the live views (this tab's and the point tab's), bring this tab's
        Live view to the front and preview the centre pattern. Without a map yet (``build_map`` not run) the views
        get a black map of the scan's shape, so that every point can still be clicked for its pattern (and a
        run paints its bands on it); for a 1D map (a line scan) they get a note instead of a map. The centre
        pattern is previewed either way."""
        if self.loaded is None:
            return
        iq, source = self.loaded["iq"], self.loaded.get("iq_source")
        nav_shape = self.loaded["nav_shape"]
        name = os.path.basename(self.loaded["source_file"]) if self.loaded["source_file"] else "nickel demo"
        if iq is not None and iq.ndim == 2:
            base, title = iq, f"{'IQ' if source == 'scan' else 'Quasi-IQ'} of {name} ({iq.shape[1]} x {iq.shape[0]} points)"
        elif iq is None and len(nav_shape) == 2:
            # a black map of the scan's shape: clickable, and the bands of a run are painted on it
            base = np.zeros(tuple(int(v) for v in nav_shape), dtype=np.float32)
            title = f"{name} ({nav_shape[1]} x {nav_shape[0]} points): no map built, click anywhere for the pattern"
        elif iq is None:
            base, title = None, f"{name}: no map yet"
        else:
            base, title = None, f"{name}: 1D map, no live view"
        for view in self._preview_views():
            view.set_base_map(base, title)
        self.view_tabs.setCurrentWidget(self.live)
        if len(nav_shape) == 2 and self.loaded.get("signal") is not None:
            self.preview_point(nav_shape[0] // 2, nav_shape[1] // 2)

    # ---------------------------------------------------------------- preparing patterns, master, detector
    def _prepare_key(self, st: Settings):
        """What identifies a prepared signal: the load key's settings plus the scan file and the background
        removals (done in place on the signal)."""
        return ("file", st.PATTERN_FILE, st.SCAN_FILE, st.EDAXH5_MAP, st.NAV_SHAPE, st.STEP_SIZE, st.LAZY, st.ROI,
                st.FLIP_PATTERNS_UD, st.REMOVE_STATIC_BACKGROUND, st.REMOVE_DYNAMIC_BACKGROUND)

    def _begin_prepared_step(self):
        """Common start of 'Check geometry' and the indexing runs, on the GUI thread.

        Returns ``(st, prepared)``: the current settings and the prepared signal when it is still valid for them,
        else None (the worker then prepares again, see ``_prepare_in_worker``). Returns None instead when a step is
        running (after telling the user) or when the pattern file would have to be read and the processor, whose
        data set must leave memory first, cannot unload now.
        """
        if self._refuse_while_busy():
            return None
        st = self.settings()
        prepared = self.prepared
        if prepared is not None and prepared["key"] != self._prepare_key(st):
            prepared = None
        loaded = self.loaded
        will_load_file = prepared is None and (loaded is None or loaded.get("signal") is None
                                               or loaded["key"] != self._load_key(st))
        if will_load_file and not self._unload_processor():
            return None  # one data set in memory at a time, as 'Load patterns' does
        return st, prepared

    def _prepare_in_worker(self, st: Settings, prepared, log):
        """Worker thread: the signal, harmonics and detector a step needs. Returns ``(prepared, harmonics, hkey, det)``.

        ``prepared`` not None (still valid for ``st``) is used as it is. Otherwise the loaded data set is reused
        when it is the one ``st`` asks for and its background removals are either not done yet or exactly the ones
        wanted: the removals are then done in place on it (one copy of the patterns in memory, not a raw and a
        prepared one; ``loaded["preprocessed"]`` records it for the click preview). Else the file is read and
        preprocessed again, and that signal replaces the loaded one when it is the same data set. The harmonics
        are reused when ``harmonics_key`` matches, else loaded; the detector is always built anew (cheap).
        Reads ``self.loaded`` / ``self.harmonics`` but touches no widget; the GUI thread stores the result with
        ``_store_prepared`` when the step is done.
        """
        key = self._prepare_key(st)
        if prepared is None:
            loaded = self.loaded
            wanted = (bool(st.REMOVE_STATIC_BACKGROUND), bool(st.REMOVE_DYNAMIC_BACKGROUND))
            same_data_set = loaded is not None and loaded["key"] == self._load_key(st)
            if same_data_set and loaded["signal"] is not None \
                    and loaded.get("preprocessed", (False, False)) in ((False, False), wanted):
                s, scan = loaded["signal"], loaded["scan"]
                if wanted != (False, False) and loaded.get("preprocessed", (False, False)) != wanted:
                    sph_core.preprocess(s, st, log)
                    loaded["preprocessed"] = wanted
            else:
                s, scan = sph_core.load_patterns(st, log=log)
                sph_core.preprocess(s, st, log)
                if same_data_set:
                    # the raw copy was preprocessed differently before: replace it rather than keep two
                    loaded["signal"] = s
                    loaded["preprocessed"] = wanted
            prepared = {"key": key, "signal": s, "scan": scan, "source": "file", "source_file": st.PATTERN_FILE}
        demo = st.PATTERN_FILE is None
        harmonics, hkey = self.harmonics_for(st, log, demo)
        det = sph_core.make_detector(st, prepared["signal"], harmonics, log, demo=demo)
        return prepared, harmonics, hkey, det

    def _store_prepared(self, result):
        """GUI thread: keep the prepared signal and the harmonics of a finished step for the next one."""
        prepared, harmonics, hkey, _det = result
        self.prepared = prepared
        self.harmonics, self.harmonics_key = harmonics, hkey

    # ---------------------------------------------------------------- actions
    def check_geometry(self):
        """'Check geometry': prepare the patterns, master and detector on a worker and show the centre pattern raw,
        as the indexer sees it, back-projected onto the sphere (north) and the sphere window, in the Results tab.
        The detector and the pattern's image quality go to the log."""
        begun = self._begin_prepared_step()
        if begun is None:
            return
        st, prepared = begun

        def work(log):
            # worker thread
            result = self._prepare_in_worker(st, prepared, log)
            prepared_, _harmonics, _hkey, det = result
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
        """'Index test region' (``test``) or 'Index full map': index on a worker, with the live view when ticked.

        On the worker: ``_prepare_in_worker``, then ``sph_core.index_blocks`` (band by band, events to the live
        view: ``run_start``, ``block_start``, ``progress``, ``chunk_done``, ``block_done``, ``run_end``) when the
        live view is ticked and the map is 2D, else one ``sph_core.index`` call. The twist is applied afterwards
        and the result maps are computed on the worker (seconds for a large map). The result dict, kept as
        ``last_run``:

        - ``label``: "test region" or "full map"; ``signal``: the indexed signal (the ROI'd or test-region signal);
        - ``xmap``: the CrystalMap with TWIST applied (what is saved); ``xmap_indexed``: before the twist (what is
          compared with a reference); ``det``: the detector; ``seconds``: indexing time;
        - ``offset``: (row, col) of the indexed region in the loaded map;
        - ``source_file``: the pattern file (None = demo); ``load_key``: ``_load_key`` of the settings, so that
          'Load patterns' of the same file keeps the result;
        - ``full_nav_shape``: (rows, cols) of the whole scan (the signal's without an ROI, else the scan file's,
          else None), for cropping a reference map to the ROI;
        - ``maps``: {"scores", "iq", "ipf_z", "ipf_x"} arrays of the Results tab, None for a non-2D map;
        - ``settings``: the Settings the run used.

        Without the data set loaded by 'Load patterns' the live view shows a blank map (a note says so). Done: the
        result is stored and the Results tab filled. Refused while a step runs.

        The 'Stop' button (``stop_indexing``) is enabled for the run: it sets ``_stop_event``, which the worker
        checks after the preparation, before every band (``index_blocks``'s ``stop``) and after a one-call run;
        a stopped run ends with ``run_end`` (``success`` False), the worker returns no result, ``last_run`` is
        left as it was and nothing is saved.
        """
        begun = self._begin_prepared_step()
        if begun is None:
            return
        st, prepared = begun
        self._stop_event.clear()
        stop = self._stop_event.is_set

        live = self.live_check.isChecked()
        if self.loaded is None or self.loaded["key"] != self._load_key(st):
            # this data set was not loaded with 'Load patterns': the live view then shows a blank map
            self.loaded = None
            self.live.set_base_map(None, "")
            if live:
                self.log("NB! Click 'Load patterns' first to see the map fill in during indexing "
                         "('Build quasi-IQ map' puts the quasi-IQ map under it)")
        self.live_run = None
        # the live view's grey scale: the loaded data's full scale, read here on the GUI thread
        scale_max = self.loaded.get("scale_max") if self.loaded is not None else None

        def work(log, emit):
            # worker thread
            result = self._prepare_in_worker(st, prepared, log)
            if stop():
                log("Indexing stopped before it started; nothing kept")
                return result, None
            prepared_, harmonics, _hkey, det = result
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
                               "n_patterns": n_run, "signal": s_run, "label": label, "st": st,
                               "scale_max": scale_max if scale_max else sph_core.sample_background_and_scale(s_run)[1]})
            if live and len(nav_shape) == 2:
                def on_event(name, data):
                    # the band's IPF colours are computed here, on the worker, not on the GUI thread
                    if name == "block_done":
                        data = dict(data)
                        xb = data.pop("xmap")
                        data["rgb"] = None if xb is None else sph_core.block_rgb(xb, data, "z")
                    emit(name, data)

                # a test region is small: shorter bands, so that the animation shows several of them
                try:
                    xmap_indexed, dt, speed = sph_core.index_blocks(
                        s_run, harmonics, det, st, log, on_event=on_event,
                        target_seconds=1.0 if test else sph_core.LIVE_BLOCK_SECONDS, stop=stop)
                except sph_core.IndexingStopped as e:
                    emit("run_end", {"success": False})
                    log(f"Indexing stopped after {e.done_patterns} of {e.total_patterns} patterns; nothing kept, nothing saved")
                    return result, None
                xmap = sph_core.apply_twist(xmap_indexed, st.TWIST)
                log(f"{xmap.size} patterns in {dt:.1f} s; steady speed {speed:.0f} patterns/s ({st.BACKEND}, bandwidth {st.BANDWIDTH})")
                if st.TWIST:
                    log(f"Orientations rotated by {st.TWIST} deg about the sample normal (TWIST)")
            else:
                xmap, xmap_indexed, dt = sph_core.index(s_run, harmonics, det, st, log, warm_up=s_warm)
                speed = xmap.size / dt
                if stop():
                    # one call cannot be interrupted: the stop discards its result once it returns
                    emit("run_end", {"success": False})
                    log(f"Indexing stopped: the result of the {label} is discarded, nothing saved")
                    return result, None
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
            if xmap.ndim == 2:
                maps = {"scores": sph_core.full_map_data(xmap, "scores"), "iq": sph_core.full_map_data(xmap, "iq"),
                        "ipf_z": sph_core.ipf_rgb_map(xmap, "z"), "ipf_x": sph_core.ipf_rgb_map(xmap, "x")}
            return result, dict(label=label, signal=s_run, xmap=xmap, xmap_indexed=xmap_indexed, det=det, seconds=dt, offset=offset,
                                source_file=st.PATTERN_FILE, load_key=self._load_key(st), full_nav_shape=full_nav_shape, maps=maps,
                                settings=st)

        def done(out):
            self.btn_stop.setEnabled(False)
            result, run = out
            self._store_prepared(result)
            if run is None:
                return  # stopped: the previous result (if any) stays, nothing is saved
            self.last_run = run
            self.show_indexing_maps(run)
            # the Live tab keeps the finished IPF map in front; the Results tab has the score, IQ and IPF maps
            self.log("Done: the Results tab has the score, IQ, IPF-Z and IPF-X maps")

        if self._start(work, done, on_event=self.on_run_event, on_failed=lambda: self.btn_stop.setEnabled(False)):
            self.btn_stop.setEnabled(True)

    def stop_indexing(self):
        """'Stop': ask the running indexing (``run_indexing``) to end after the band in progress. The button is
        greyed out at once; the worker discards what it indexed and the run ends with ``run_end`` (not a success).
        Nothing is saved."""
        self.btn_stop.setEnabled(False)
        if not self._stop_event.is_set():
            self._stop_event.set()
            self.log("Stop requested: the band in progress finishes, then the run ends; nothing will be kept or saved")

    # ---------------------------------------------------------------- live view plumbing
    def _live_pattern(self, row, col):
        """One pattern of the running region, for the live view (called in the GUI thread, a few times per second).

        (row, col) are map coordinates of the whole map; the run's offset is subtracted. The montage tab's live
        view uses this too. Returns ``(pattern, (vmin, vmax))``, the grey limits of ``sph_core.preview_limits``
        on the run's intensity scale; None when no run is shown.
        """
        run = self.live_run
        if run is None:
            return None
        s = run["signal"]
        if s.axes_manager.navigation_dimension == 1:  # a line scan or a one-row tile: shown as one row
            data = np.asarray(s.data[int(col) - run["c0"]])
        else:
            data = np.asarray(s.data[int(row) - run["r0"], int(col) - run["c0"]])
        # the prepared signal already carries the background removals; only the indexer's own steps remain
        shown = sph_core.preview_pattern(data, run["st"], None, apply_background=False)
        return shown, sph_core.preview_limits(run["st"], run["scale_max"], shown)

    def on_run_event(self, name, data):
        """GUI thread slot for the worker's events: drives the live view, the status line and the speed estimate.

        Events and their payloads (``data`` dicts; row and column numbers of a band or chunk are relative to the
        run's region and offset by the run's ``r0``/``c0`` before they reach the live view):

        - ``run_start``: ``r0``, ``c0`` (offset of the region in the map), ``rows``, ``cols``, ``n_patterns``,
          ``signal`` (the region's signal, for ``_live_pattern``), ``label``, ``st`` (the Settings), ``scale_max``
          (the data's intensity full scale, ``sph_core.intensity_full_scale``, the live view's white) and
          optionally ``live_view`` (the montage tab's view instead of this tab's). Starts ``live_run`` and the
          view's run.
        - ``progress``: ``frac_index``, ``frac_refine`` (0..1 of the current band), ``what`` ("iq" during the
          quasi-IQ map: only the status line is updated).
        - ``block_start``: ``r0``, ``r1``, ``c0``, ``c1``, ``n_patterns`` of the band being indexed.
        - ``chunk_done``: ``rows``, ``cols`` (arrays), ``packed`` orientations and the ``keys`` to colour them (done
          by the live view at its next redraw, not here).
        - ``block_done``: the band's ``r0``, ``r1``, ``c0``, ``c1``, ``rgb`` (its IPF-Z image or None),
          ``n_patterns``, ``seconds``, ``done_patterns``, ``total_patterns``, ``patterns_per_second``,
          ``eta_seconds``.
        - ``run_end``: ``success`` (default True). Ends the view's run and clears ``live_run``.

        Events other than ``run_start`` and the quasi-IQ progress are ignored when no run is shown.
        """
        if name == "run_start":
            self.live_run = {"signal": data["signal"], "st": data["st"], "r0": data["r0"], "c0": data["c0"], "n_patterns": data["n_patterns"],
                             "scale_max": data.get("scale_max") or sph_core.intensity_full_scale(data["signal"].data.dtype),
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
        """The 'now' speed shown by the live view: patterns completed over the last ``window_seconds``.

        ``run["window"]`` holds (time, patterns done) samples; the oldest beyond the window are dropped and the
        speed is the slope over what remains, once it spans more than half a second.
        """
        now = time.perf_counter()
        window = run["window"]
        window.append((now, float(done_patterns)))
        while len(window) > 2 and now - window[0][0] > window_seconds:
            window.popleft()
        if len(window) >= 2 and now - window[0][0] > 0.5:
            speed = (window[-1][1] - window[0][1]) / (now - window[0][0])
            self._live_view.set_speed(max(speed, 0.0))

    def show_indexing_maps(self, run):
        """Fill the Results tab with the run's score, IQ, IPF-Z and IPF-X maps (``run["maps"]``); a note in the
        log for a map that is not 2D."""
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
        """'Save .ang / .h5': write ``last_run["xmap"]`` with the current output settings (folder, name, which
        formats), named after the file the result came from, whatever has been loaded or read since; a test
        region gets the suffix ``_test``. On the GUI thread (writing is quick); errors go to the log and a box."""
        if self.last_run is None:
            QMessageBox.information(self, "Save", "Nothing indexed yet")
            return
        st = self.settings()
        out_dir, stem = sph_core.output_location(st, self.last_run.get("source_file"))
        if self.last_run["label"] == "test region":
            stem += "_test"
        try:
            sph_core.save_xmap(self.last_run["xmap"], out_dir, stem, st.SAVE_ANG, st.SAVE_H5, log=self.log)
        except Exception as e:  # noqa: BLE001
            self.log(f"{type(e).__name__}: {e}")
            QMessageBox.critical(self, "Save", f"{type(e).__name__}: {e}")

    def compare_reference(self):
        """'Compare with reference': the misorientation of the last run's (untwisted) orientations to the reference
        map of the REFERENCE_FILE, computed on a worker; the statistics go to the log and the 2D angle map to the
        Results tab. The run's own ROI and the whole scan's shape are used, so a reference of the whole scan is
        cropped to the indexed region. A run with no indexed point fails."""
        st = self.settings()
        if self.last_run is None or not st.REFERENCE_FILE:
            QMessageBox.information(self, "Compare", "Index the full map and give a reference file first")
            return
        run = self.last_run
        roi = run["settings"].ROI
        full_nav_shape = run.get("full_nav_shape")

        def work(log):
            # worker thread
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
        """Qt override: ask before quitting while a step runs, and remove the processor's temporary files."""
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
    """Start the application with the command line options (``ARGS``); uncaught exceptions of the GUI thread go
    to the log, stderr and a message box instead of ending the program."""
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
