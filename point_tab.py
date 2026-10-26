"""Tab 2 of the Kikuchi workbench: spherical indexing of one clicked pattern.

Load a pattern file, click a point on its map, press *Index this point*: that one pattern is indexed on the CPU
(the fork's ``SphericalIndexer``, built once and reused) and shown next to the pattern simulated from the master
at the orientation found (``sph_core.simulate_pattern``: the band limited Lambert master synthesized from the
harmonics the indexer matched against, projected on the same detector). The map is the IQ column of the scan
file when one is given, else the quasi-IQ map built on request (*Build quasi-IQ map*), never automatically.

Three columns: the file fields, the shared settings and the buttons on the left; the map in the middle (with
the status line and the log under it); on the right the clicked pattern as the indexer sees it above the pattern
simulated from the master, with a slider between them that lays the simulated pattern over the measured one at
any opacity. The pattern, scan and master file fields, the projection center, the bandwidth, the
AHE tiles and the circular mask mirror the spherical indexing tab's widgets (a change in either tab is a change
in both); the master bandwidth, tilts, pixel size, twist, background steps and the other indexing settings are
read from that tab. ``PointTab`` is built by ``Workbench.__init__`` with the Workbench as ``wb`` and uses its
data set (``loaded``, ``prepared``, ``harmonics``), its worker (``_start``), its preparation
(``_begin_prepared_step``, ``_prepare_in_worker``, ``_store_prepared``), its click preview (``preview_point``,
which calls ``set_base_map`` / ``show_pattern`` / ``set_marker`` here as on a ``LiveView``) and its ``log``.
"""
from __future__ import annotations

import importlib
import os
import time

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy, QSlider,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

import sph_core


def _mirror(a, b):
    """Keep two widgets of the same kind equal: a change of either is copied to the other (no loop: a value
    that is already there is not set again). Line edits (text), spin boxes (value), check boxes (checked) and
    combo boxes (current text). ``a`` takes ``b``'s value first."""
    if isinstance(a, QLineEdit):
        signal, get, set_ = "textChanged", QLineEdit.text, QLineEdit.setText
    elif isinstance(a, QSpinBox):
        signal, get, set_ = "valueChanged", QSpinBox.value, QSpinBox.setValue
    elif isinstance(a, QCheckBox):
        signal, get, set_ = "toggled", QCheckBox.isChecked, QCheckBox.setChecked
    elif isinstance(a, QComboBox):
        signal, get, set_ = "currentTextChanged", QComboBox.currentText, QComboBox.setCurrentText
    else:
        raise TypeError(f"cannot mirror a {type(a).__name__}")

    def copy_to(dst):
        def on_change(value):
            if get(dst) != value:
                set_(dst, value)
        return on_change

    set_(a, get(b))
    getattr(a, signal).connect(copy_to(b))
    getattr(b, signal).connect(copy_to(a))


#: (vmin, vmax) for a grey image: ``sph_core.grey_limits`` (the same percentiles as the live view and the maps).
_grey_limits = sph_core.grey_limits


class ClickMap(QWidget):
    """The clickable map of the point tab: one matplotlib axis with the grey map, a red cross on the clicked
    point and a title. ``point_clicked(row, col)`` is emitted for a left click inside the map."""
    point_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), facecolor="w")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(200, 120)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.shape = None
        self._marker = None
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def set_base_map(self, iq, title=""):
        """Show the (rows, cols) map ``iq`` in 0..1 (grey), or clear the axis for None; ``title`` above it."""
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._marker = None
        self.shape = None
        if iq is not None:
            base = np.nan_to_num(np.asarray(iq, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            if base.ndim != 2:
                raise ValueError(f"set_base_map: expected a 2D array, got shape {base.shape}")
            self.shape = base.shape
            self.ax.imshow(base, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
            (self._marker,) = self.ax.plot([0], [0], "r+", markersize=15, markeredgewidth=2, visible=False)
        if title:
            self.ax.set_title(title, fontsize=10)
        try:
            self.figure.tight_layout()
        except Exception:  # noqa: BLE001
            pass
        self.canvas.draw_idle()

    def set_marker(self, row, col):
        """Put the red cross on map point (row, col), or hide it when either is None. No-op without a map."""
        if self._marker is None:
            return
        if row is None or col is None:
            self._marker.set_visible(False)
        else:
            self._marker.set_data([col], [row])
            self._marker.set_visible(True)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.button != 1 or event.inaxes is not self.ax or self.shape is None or event.xdata is None:
            return
        row, col = int(round(event.ydata)), int(round(event.xdata))
        if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
            self.point_clicked.emit(row, col)


class PairCanvas(QWidget):
    """The right column: the measured pattern as the indexer sees it (top), with the simulated pattern laid
    over it at the opacity of the slider under it (0 % = the measured pattern alone, 100 % = the simulated one
    alone), and the pattern simulated from the master at the orientation found (bottom). Grey with percentile
    limits, each image its own. ``ax_top`` / ``ax_bottom`` are the two axes (each in its own figure)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig_top = Figure(figsize=(4, 4), facecolor="w")
        self.canvas_top = FigureCanvas(self.fig_top)
        self.fig_bottom = Figure(figsize=(4, 4), facecolor="w")
        self.canvas_bottom = FigureCanvas(self.fig_bottom)
        for c in (self.canvas_top, self.canvas_bottom):
            c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            c.setMinimumSize(160, 120)
        self.ax_top = self.fig_top.add_subplot(111)
        self.ax_bottom = self.fig_bottom.add_subplot(111)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setToolTip("Opacity of the simulated pattern laid over the measured one: 0 % shows the measured pattern "
                               "alone, 100 % the simulated pattern alone")
        self.slider.valueChanged.connect(self._on_slider)
        self.slider_label = QLabel("")
        self.slider_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas_top, 1)
        lay.addWidget(self.slider_label)
        lay.addWidget(self.slider)
        lay.addWidget(self.canvas_bottom, 1)
        self._overlay = None          # the simulated pattern laid over the measured one, or None
        self._overlay_artist = None
        self.clear()

    def _show(self, ax, pattern, title):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        if pattern is not None:
            lo, hi = _grey_limits(pattern)
            ax.imshow(np.asarray(pattern, dtype=np.float32), cmap="gray", vmin=lo, vmax=hi, interpolation="nearest",
                      aspect="equal")
        ax.set_title(title, fontsize=10)
        fig = ax.figure
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001
            pass
        fig.canvas.draw_idle()

    def _draw_overlay(self):
        """Put the overlay image on the top axis (above the measured pattern) at the slider's opacity."""
        self._overlay_artist = None
        if self._overlay is None or not self.ax_top.images:
            self._set_slider_text()
            return
        lo, hi = _grey_limits(self._overlay)
        self._overlay_artist = self.ax_top.imshow(np.asarray(self._overlay, dtype=np.float32), cmap="gray", vmin=lo, vmax=hi,
                                                  interpolation="nearest", aspect="equal", alpha=self.slider.value() / 100.0,
                                                  zorder=2)
        self._set_slider_text()
        self.canvas_top.draw_idle()

    def _set_slider_text(self):
        v = self.slider.value()
        if self._overlay is None:
            self.slider_label.setText("Overlay of the simulated pattern on the measured one: index the point first")
        else:
            self.slider_label.setText(f"Overlay of the simulated pattern on the measured one: {v} % "
                                      + ("(measured only)" if v == 0 else "(simulated only)" if v == 100 else ""))

    def _on_slider(self, value):
        if self._overlay_artist is not None:
            self._overlay_artist.set_alpha(value / 100.0)
            self.canvas_top.draw_idle()
        self._set_slider_text()

    def show_measured(self, pattern, title):
        """The top panel; the overlay, when there is one, is drawn again on top of it."""
        self._show(self.ax_top, pattern, title)
        self._draw_overlay()

    def show_simulated(self, pattern, title):
        """The bottom panel, and the same pattern as the overlay of the top one (None clears both, the title
        stays)."""
        self._show(self.ax_bottom, pattern, title)
        self._overlay = None if pattern is None else np.asarray(pattern)
        if self.ax_top.images:
            # drop an earlier overlay (the measured image is the first artist), then draw the new one
            for im in list(self.ax_top.images[1:]):
                im.remove()
        self._draw_overlay()

    def clear(self):
        """Empty both panels and the overlay."""
        self._overlay = None
        self._show(self.ax_top, None, "Measured pattern, as the indexer sees it")
        self._show(self.ax_bottom, None, "Simulated from the master ('Index this point')")
        self._draw_overlay()


class PointTab(QWidget):
    """The single-point tab: the file fields, shared settings and buttons on the left, the clickable map
    (``ClickMap``) with the status line and log in the middle, the measured | simulated pair (``PairCanvas``)
    on the right.

    Attributes read by the workbench or by scripts: ``map_view``, ``pair``, ``indexer`` / ``indexer_key`` (the
    cached CPU ``SphericalIndexer`` and its ``sph_core.point_indexer_key``), ``masters`` / ``masters_key`` (the
    Lambert masters, one per phase, and the workbench's ``harmonics_key`` they were made from), ``last_point``
    (the result dict of the last indexed point, see ``index_point``). ``set_base_map``, ``show_pattern`` and
    ``set_marker`` are what ``Workbench.preview_point`` / ``_show_loaded`` call on a view. All methods run on
    the GUI thread except the ``work`` closure of ``index_point``.
    """

    def __init__(self, wb):
        """Build the widgets. ``wb`` is the Workbench; its module is imported by name (the workbench imports this
        module, so a plain import would be circular) for ``SettingsForm`` and the layout helpers. The workbench's
        indexing tab must exist already (its widgets are mirrored)."""
        super().__init__()
        self.wb = wb
        kw = importlib.import_module(type(wb).__module__)
        self.kw = kw
        self.indexer = None
        self.indexer_key = None
        self.masters = None
        self.masters_key = None
        self.last_point = None
        self._shown_rc = None   # (row, col) of the measured pattern on the right

        form = kw.SettingsForm(wb.base)
        self.form = form
        form.group("Pattern source (shared with the spherical indexing tab)")
        self.pattern_file_edit = form.add_path(
            "PATTERN_FILE", "Pattern file:", tip="EDAX .up1/.up2 or any format kp.load() reads; the same field as in the spherical "
                                           "indexing tab. Blank = the nickel demo data",
            filter=kw.PATTERN_FILE_FILTER, on_chosen=lambda path: self.load_patterns())
        self.scan_file_edit = form.add_path(
            "SCAN_FILE", "Scan file (.ang/.h5):", tip="The .ang or .edaxh5 of the scan, or an indexing result (.ang, the workbench's .h5): "
                                                  "its IQ column becomes the map here, and 'Read scan parameters' takes the projection "
                                                  "center from an acquisition file",
            filter="Scan files (*.ang *.edaxh5 *.h5 *.hdf5);;All files (*)", start_dir=self._pattern_file_dir)
        self.btn_load = QPushButton("Load patterns")
        self.btn_load.setToolTip("Loads the pattern file (one data set in memory at a time) and shows the scan file's IQ map; "
                                 "without a scan file the map stays empty until 'Build quasi-IQ map'")
        self.btn_load.clicked.connect(self.load_patterns)
        form.add_widget(self.btn_load)
        self.btn_scan = QPushButton("Read scan parameters")
        self.btn_scan.setToolTip("Reads the scan and pattern file headers into the log and fills the projection center")
        self.btn_scan.clicked.connect(wb.read_scan)
        form.add_widget(self.btn_scan)
        self.btn_quasi = QPushButton("Build quasi-IQ map")
        self.btn_quasi.setToolTip("The standard deviation of every raw pattern (one read of the file), when there is no scan "
                                  "file to take the IQ from")
        self.btn_quasi.clicked.connect(self.build_quasi_iq)
        form.add_widget(self.btn_quasi)
        note = QLabel("With a scan file the map is its IQ column; without one, build the quasi-IQ map.")
        note.setStyleSheet("color: gray")
        form.add_widget(note)

        form.group("Master pattern(s) (shared with the spherical indexing tab)")
        self.master_file_edit = form.add_path(
            "MASTER_FILE", "Master file(s):", tip="EMSphInx .sht or EMsoft master pattern .h5; several files (one per phase) as a "
                                             "list ['a.sht', 'b.sht']. The same field as in the spherical indexing tab",
            filter=kw.MASTER_FILE_FILTER, multi=True)

        form.group("Indexing (shared with the spherical indexing tab)")
        # the four shared fields start at the indexing tab's current values (its form holds the Settings defaults)
        self.pc_edit = form.add_text(
            "PC", "Projection center (x*, y*, z*):",
            "The same value as in the spherical indexing tab (in its PC convention): adjust it here and index the point "
            "again to see the match change; 'Read scan parameters' fills it from the scan file",
            default=wb.form._fields["PC"][0]())
        self.bandwidth_combo = form.add_combo(
            "BANDWIDTH", "Bandwidth:", sph_core.FAST_BANDWIDTHS,
            "The indexing bandwidth (fast values only), the same as in the spherical indexing tab; changing it rebuilds the "
            f"single-pattern indexer on the next click (a second or so at {wb.base.BANDWIDTH}, longer at high values)",
            default=wb.bandwidth_combo.currentText())
        self.w_regions = form.add_int("N_REGIONS", "AHE tiles (n_regions):", 0, sph_core.N_REGIONS_MAX,
                                      "Adaptive histogram equalization tiles per axis; 0 = off. The same as in the spherical indexing tab",
                                      default=wb.w_regions.value())
        self.w_circular = form.add_bool("CIRCULAR_MASK", "Circular mask", "Use only the inscribed circle of the detector. "
                                        "The same as in the spherical indexing tab", default=wb.w_circular.isChecked())
        note = QLabel("The master bandwidth, the tilts, pixel size and twist, the background steps and the other "
                      "indexing settings are those of the spherical indexing tab.")
        note.setStyleSheet("color: gray")
        form.add_widget(note)
        self.btn_index = QPushButton("Index this point")
        self.btn_index.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; }")
        self.btn_index.setToolTip("Spherically index the pattern clicked on the map (CPU) and show it above the pattern simulated "
                                  "from the master at the orientation found")
        self.btn_index.clicked.connect(self.index_point)
        form.add_widget(self.btn_index)
        self.result_label = QLabel("Click a point on the map, then 'Index this point'.")
        self.result_label.setWordWrap(True)
        form.add_widget(self.result_label)
        form.finish()

        # one value, two widgets: the indexing tab's fields and these
        _mirror(self.pattern_file_edit, wb.pattern_file_edit)
        _mirror(self.scan_file_edit, wb.scan_file_edit)
        _mirror(self.master_file_edit, wb.master_file_edit)
        _mirror(self.pc_edit, wb.pc_edit)
        _mirror(self.bandwidth_combo, wb.bandwidth_combo)
        _mirror(self.w_regions, wb.w_regions)
        _mirror(self.w_circular, wb.w_circular)

        # middle column: the map, the status line and the log; right column: the measured | simulated pair
        middle = QWidget()
        ml = QVBoxLayout(middle)
        ml.setContentsMargins(0, 0, 0, 0)
        msplit = QSplitter(Qt.Vertical)
        self.map_view = ClickMap()
        self.map_view.point_clicked.connect(wb.preview_point)
        msplit.addWidget(self.map_view)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        self.progress_label = QLabel("")
        self.progress_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        bl.addWidget(self.progress_label)
        bl.addWidget(self.log_box)
        msplit.addWidget(bottom)
        msplit.setSizes([650, 200])
        ml.addWidget(msplit)
        self.pair = PairCanvas()
        right = QSplitter(Qt.Horizontal)
        right.addWidget(middle)
        right.addWidget(self.pair)
        right.setCollapsible(0, False)
        right.setCollapsible(1, False)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        right.setSizes([600, 400])
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(kw.side_by_side(form, right))

    # ------------------------------------------------------------------------------------------ helpers
    def log(self, text: str):
        """Append a line to the tab's log pane, scrolled to the end (``Workbench.log`` mirrors every line here)."""
        self.log_box.appendPlainText(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def set_progress(self, text: str):
        """Show the last 200 characters of ``text`` in the status line (mirrored from ``Workbench.set_progress``)."""
        self.progress_label.setText(text[-200:])

    def _pattern_file_dir(self):
        """Folder of the pattern file as typed, or None: where the scan file dialog starts."""
        text = self.pattern_file_edit.text().strip()
        return os.path.dirname(text) if text else None

    def _busy(self, busy: bool):
        """Disable (``busy``) or enable the tab's buttons; called from ``Workbench._busy``."""
        for b in (self.btn_load, self.btn_scan, self.btn_quasi, self.btn_index):
            b.setEnabled(not busy)

    # the view protocol of Workbench.preview_point / _show_loaded (as on a LiveView)
    def set_base_map(self, iq, title: str = ""):
        """Show the loaded data set's map (or clear it for None) in the middle column."""
        self.map_view.set_base_map(iq, title)

    def show_pattern(self, pattern, row=None, col=None):
        """Show the clicked pattern (as the indexer sees it) in the top right panel. A new point empties the
        simulated panel until it is indexed."""
        rc = None if row is None else (int(row), int(col))
        self.pair.show_measured(pattern, f"Measured ({row}, {col}), as the indexer sees it" if rc else "Measured pattern")
        if rc != self._shown_rc or self.last_point is None:
            self.pair.show_simulated(None, "Simulated from the master ('Index this point')")
        self._shown_rc = rc

    def set_marker(self, row, col):
        """Put the red cross on the clicked map point."""
        self.map_view.set_marker(row, col)

    def unload(self):
        """Clear the map, the panels and the result line when the data set is unloaded (``Workbench.unload_indexing``).
        The cached indexer and masters stay: their keys tell when they no longer fit."""
        self.map_view.set_base_map(None, "")
        self.pair.clear()
        self.last_point = None
        self._shown_rc = None
        self.result_label.setText("Click a point on the map, then 'Index this point'.")

    # ------------------------------------------------------------------------------------------ actions
    def load_patterns(self):
        """'Load patterns': the workbench's load with the scan file's IQ column as the map, or no map without a
        scan file (built later by 'Build quasi-IQ map')."""
        st = self.wb.settings()
        self.wb.load_patterns_clicked(iq_source="scan" if st.SCAN_FILE else None)

    def build_quasi_iq(self):
        """'Build quasi-IQ map': the quasi-IQ map of the loaded data set (``Workbench.build_map``)."""
        self.wb.build_map("quasi")

    def index_point(self):
        """'Index this point': spherical indexing of the pattern clicked on the map (the workbench's
        ``_preview_rc``) on a worker with the CPU backend, and the pattern simulated from the master at the
        orientation found, shown under it.

        The settings are the indexing tab's (the shared fields included). On the worker:
        ``Workbench._prepare_in_worker`` (patterns with the background removals in place, the harmonics and the
        detector, as the runs use them), the pattern read from the prepared signal, the cached indexer and
        Lambert masters reused when their keys still fit, else built (``sph_core.make_point_indexer``,
        ``sph_core.lambert_master``), then ``sph_core.index_pattern`` and ``sph_core.simulate_pattern`` with the
        orientation as indexed (``TWIST`` is only applied to the reported angles). The measured pattern is put
        through the indexer's own preprocessing (with its pixel mask) for the top panel, the simulated one too for
        the normalized cross-correlation that is logged; the bottom panel shows the simulated pattern as it comes
        from the master. Done (GUI thread): the prepared signal, harmonics and the caches are stored, the two
        panels drawn, the result put in the result line and the log, and kept as ``last_point`` (the
        ``index_pattern`` dict plus ``row``, ``col``, ``ncc``, ``seconds_simulate``). Refused (with a message)
        without a loaded data set or a clicked point, or when the pattern settings changed since the load.
        """
        wb = self.wb
        loaded = wb.loaded
        if loaded is None or loaded.get("signal") is None or wb._preview_rc is None:
            QMessageBox.information(self, "Index this point", "Load patterns and click a point on the map first")
            return
        begun = wb._begin_prepared_step()
        if begun is None:
            return
        st, prepared = begun
        if loaded["key"] != wb._load_key(st):
            QMessageBox.information(self, "Index this point", "The pattern settings changed since the map was loaded: "
                                                              "'Load patterns' again, then click a point")
            return
        row, col = (int(v) for v in wb._preview_rc)
        # the caches are read here, on the GUI thread, and handed to the worker as plain values
        indexer, indexer_key = self.indexer, self.indexer_key
        masters, masters_key = self.masters, self.masters_key

        def work(log):
            # worker thread
            result = wb._prepare_in_worker(st, prepared, log)
            prepared_, harmonics, hkey, det = result
            pattern = np.asarray(prepared_["signal"].data[row, col])
            key = sph_core.point_indexer_key(st, det, hkey)
            idx = indexer if indexer_key == key and indexer is not None else sph_core.make_point_indexer(harmonics, det, st, log)
            if masters_key == hkey and masters is not None:
                mps = masters
            else:
                t0 = time.perf_counter()
                mps = [sph_core.lambert_master(h) for h in harmonics]
                log(f"Lambert master(s) of {len(mps)} phase(s) synthesized in {time.perf_counter() - t0:.1f} s")
            res = sph_core.index_pattern(idx, pattern, twist=st.TWIST)
            t0 = time.perf_counter()
            simulated = sph_core.simulate_pattern(mps[res["phase_id"]], res["rotation"], det)
            res["seconds_simulate"] = time.perf_counter() - t0
            seen = sph_core.preprocess_pattern_for_preview(pattern, st, idx.good_pixels)
            seen_sim = sph_core.preprocess_pattern_for_preview(simulated, st, idx.good_pixels)
            a, b = seen.astype(np.float64).ravel(), seen_sim.astype(np.float64).ravel()
            res["ncc"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
            res["row"], res["col"] = row, col
            e = res["euler_deg"]
            log(f"Point ({row}, {col}): {res['phase']}, Euler (Bunge{', TWIST applied' if st.TWIST else ''}) "
                f"{e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f} deg, score {res['score']:.4f}, IQ {res['iq']:.3f}, "
                f"NCC measured/simulated {res['ncc']:.3f}; indexed in {res['seconds']:.2f} s (cpu, bandwidth {st.BANDWIDTH}), "
                f"simulated in {res['seconds_simulate']:.2f} s")
            return result, dict(indexer=idx, key=key, masters=mps, hkey=hkey, res=res, seen=seen, simulated=simulated)

        def done(out):
            result, p = out
            wb._store_prepared(result)
            self.indexer, self.indexer_key = p["indexer"], p["key"]
            self.masters, self.masters_key = p["masters"], p["hkey"]
            res = p["res"]
            self.last_point = res
            self._shown_rc = (row, col)
            e = res["euler_deg"]
            summary = (f"({row}, {col}): {res['phase']}, Euler {e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f} deg, "
                       f"score {res['score']:.3f}, IQ {res['iq']:.3f}, NCC {res['ncc']:.3f}, {res['seconds']:.2f} s")
            self.pair.show_measured(p["seen"], f"Measured ({row}, {col}), as the indexer sees it")
            self.pair.show_simulated(p["simulated"], f"Simulated from the master: {res['phase']}, Euler {e[0]:.1f}, {e[1]:.1f}, "
                                                     f"{e[2]:.1f} deg, score {res['score']:.3f}")
            self.result_label.setText(summary)
            self.map_view.set_marker(row, col)

        wb._start(work, done)
