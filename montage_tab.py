"""Tab 3 of the Kikuchi workbench: batch processing and batch indexing of an EDAX APEX montage.

Two separate jobs, run one after the other by the user:

1. **Process all tiles** (pattern processor): a static background is computed from the patterns of all tiles
   together, then every tile is processed with the settings of the pattern processor tab and written to its own
   ``<tile><suffix>.up1`` file in the output folder, one tile at a time; the processor is unloaded after each tile
   and after the batch, so that nothing of the montage stays in RAM. The processor's ``batch_process`` runs on the
   GUI thread and pumps events itself, so the whole tab (and the workbench's entry points) are locked meanwhile.
2. **Index all tiles** (spherical indexing): every processed tile is indexed with the settings of the indexing
   tab (one projection center, sample tilt and camera tilt for the whole montage) on the workbench's ``Worker``
   thread, band by band with this tab's own live view, and its orientation table is written into a copy of the
   project, ``<project>_sph_bw<bandwidth>.edaxh5``, which OIM Analysis or MTEX can stitch like the original.
   Nothing is stitched here.

The tiles are the maps of the project; their pattern files are found by the GUID in the .up1 header (see
montage.py and STITCH_DATASET_NOTES.md). ``MontageTab`` is built by ``Workbench.__init__`` with the Workbench as
``wb`` and uses its widgets (``form``, ``tabs``, ``proc``), its locking (``_busy``, ``_refuse_while_busy``,
``batch_running``), its worker (``_start``, ``on_run_event``) and its state (``settings()``, ``harmonics``,
``unload_indexing``, ``_unload_processor``, ``log``).
"""
from __future__ import annotations

import copy
import dataclasses
import gc
import importlib
import os
import time
import traceback

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QHeaderView, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import montage
import sph_core

DEFAULT_SUFFIX = "_proc"


class _Cancelled(Exception):
    """Raised inside a batch when the user pressed Cancel batch: in the processing's progress callbacks (GUI
    thread) and in the indexing's event callback (worker thread), to leave the current step at once."""


def _has_suffix(tile: montage.Tile, suffix: str) -> bool:
    """True when the stem of the tile's matched pattern file ends with ``suffix`` (a processed file); False for
    a blank suffix."""
    return bool(suffix) and os.path.splitext(tile.pattern.name)[0].endswith(suffix)


class MontageTab(QWidget):
    """The montage batch tab: project and folders, the tile table, the two batch groups, and on the right the
    cancel button, the tab's own ``LiveView`` and its log.

    Attributes read by the workbench or by scripts: ``project`` (the loaded ``montage.Project`` or None),
    ``raw_tiles`` / ``processed_tiles`` (the tiles matched with their raw / processed files at the last rescan,
    in table order), ``processing`` (True while a batch runs), ``cancel_requested``, ``last_output_project``
    (path of the output project of the last batch indexing), ``live``, ``table``, ``form``, ``log_box``.
    All methods run on the GUI thread except the ``work`` closure of ``index_all``.
    """

    def __init__(self, wb):
        """Build the widgets. ``wb`` is the Workbench; its module is imported by name (the workbench imports this
        module, so a plain import would be circular) for ``SettingsForm``, ``LiveView``, the layout helpers and
        ``LIVE_BLOCK_SECONDS``, which scripts change at run time."""
        super().__init__()
        self.wb = wb
        kw = importlib.import_module(type(wb).__module__)
        self.kw = kw
        self.project: montage.Project | None = None
        self.raw_tiles: list = []        # tiles with their raw pattern file, from the last rescan
        self.processed_tiles: list = []  # tiles with their processed file, from the last rescan
        self.cancel_requested = False
        self.processing = False
        self.last_output_project = None

        form = kw.SettingsForm()
        self.form = form
        form.group("Montage project")
        self.project_edit = form.add_path(
            "PROJECT_FILE", "Project (.edaxh5):", tip="The APEX project of the montage: holds every tile's grid, step, pattern center and phases",
            filter="APEX projects (*.edaxh5 *.h5);;All files (*)", on_chosen=lambda path: self.load_project())
        self.btn_project = QPushButton("Load project")
        self.btn_project.clicked.connect(self.load_project)
        form.add_widget(self.btn_project)
        self.pattern_dir_edit = form.add_path("PATTERN_DIR", "Raw pattern folder:", tip="Folder with the tiles' .up1/.up2 files as written by APEX; blank = the project's folder",
                                              directory=True, start_dir=self._project_dir)
        self.output_dir_edit = form.add_path("OUTPUT_DIR", "Output folder:", tip="Where the processed tiles and the output project go; blank = 'processed' next to the project",
                                             directory=True, start_dir=self._project_dir)
        self.suffix_edit = form.add_text("SUFFIX", "Processed file suffix:", DEFAULT_SUFFIX,
                                         "Appended to the tile's file name: <tile>_proc.up1 (must not be blank)", literal=False)
        self.btn_rescan = QPushButton("Rescan folders")
        self.btn_rescan.clicked.connect(self.rescan)
        form.add_widget(self.btn_rescan)

        form.group("Tiles")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Tile", "Grid", "Raw file", "Processed file", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(220)
        self.table.verticalHeader().setVisible(False)
        form._form.addRow(self.table)
        self.geometry_label = QLabel("")
        self.geometry_label.setWordWrap(True)
        form.add_widget(self.geometry_label)

        form.group("1. Batch processing (settings of the pattern processor tab)")
        self.btn_load_proc = QPushButton("Load selected tile in the processor")
        self.btn_load_proc.setToolTip("Loads the selected tile's raw file in tab 1, to set up the processing on it")
        self.btn_load_proc.clicked.connect(self.load_selected_in_processor)
        form.add_widget(self.btn_load_proc)
        self.shared_bg_check = QCheckBox("Static background from all tiles together")
        self.shared_bg_check.setChecked(True)
        self.shared_bg_check.setToolTip("One mean pattern over every tile's patterns (streamed), used by the static background of every tile; "
                                        "off = each tile's own mean, as the processor does for a single file")
        form.add_widget(self.shared_bg_check)
        self.btn_process = QPushButton("Process all tiles")
        self.btn_process.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; }")
        self.btn_process.clicked.connect(self.process_all)
        form.add_widget(self.btn_process)

        form.group("2. Batch indexing (settings of the spherical indexing tab)")
        self.index_processed_check = QCheckBox("Index the processed files (else the raw files)")
        self.index_processed_check.setChecked(True)
        form.add_widget(self.index_processed_check)
        self.keep_iq_check = QCheckBox("Keep the project's (Hough) IQ instead of the spherical IQ")
        self.keep_iq_check.setChecked(False)
        self.keep_iq_check.setToolTip("Off (as EMSphInx writes its .ang): the IQ column of the output project becomes the spherical image "
                                      "quality, EMSphInx's discrete cosine measure of each pattern (0..1 scale), and CI the cross-correlation. "
                                      "On: the IQ column keeps the Hough IQ written at acquisition; CI is the cross-correlation either way")
        form.add_widget(self.keep_iq_check)
        self.phase_edit = form.add_text(
            "PHASE_MAP", "Project phase of each master:", None,
            "The project's phase number (1-based, as listed when the project is loaded) for each master pattern of the indexing tab, "
            "comma separated, e.g. '2' or '2,1'. Blank = matched by name, else by order when the counts agree; the batch refuses to "
            "run when no match can be made, so that the Phase column of the output is never wrong", placeholder="by name", literal=False)
        self.resume_check = QCheckBox("Resume: keep tiles already indexed in an existing output project")
        self.resume_check.setChecked(True)
        self.resume_check.setToolTip("On: an output project that already exists is kept and the tiles it already holds spherical results "
                                     "for are skipped (a cancelled or failed batch continues where it stopped). Off: a fresh copy of the "
                                     "project is made and every tile is indexed again")
        form.add_widget(self.resume_check)
        self.btn_use_geometry = QPushButton("Copy the project's PC and tilts to the indexing tab")
        self.btn_use_geometry.setToolTip("Puts the first tile's pattern center (EDAX), sample tilt and camera elevation into tab 2; "
                                         "check them there, they are your inputs")
        self.btn_use_geometry.clicked.connect(self.copy_geometry)
        form.add_widget(self.btn_use_geometry)
        self.btn_load_index = QPushButton("Load selected tile in the indexer")
        self.btn_load_index.setToolTip("Loads the selected tile (processed file when there is one) in tab 2, to test the indexing on it")
        self.btn_load_index.clicked.connect(self.load_selected_in_indexer)
        form.add_widget(self.btn_load_index)
        self.btn_index = QPushButton("Index all tiles")
        self.btn_index.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; }")
        self.btn_index.clicked.connect(self.index_all)
        form.add_widget(self.btn_index)
        self.output_label = QLabel("")
        self.output_label.setWordWrap(True)
        form.add_widget(self.output_label)
        form.finish()

        right = QWidget()
        rl = QVBoxLayout(right)
        self.btn_cancel = QPushButton("Cancel batch")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        rl.addLayout(kw.button_rows([self.btn_cancel], per_row=4))
        split = QSplitter(Qt.Vertical)
        self.live = kw.LiveView()
        self.live.error.connect(self.wb.log)
        self.live.set_pattern_source(wb._live_pattern)
        split.addWidget(self.live)
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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(kw.side_by_side(form, right, left_width=560))
        self._set_batch_buttons(False)

    # ------------------------------------------------------------------------------------------ helpers
    def log(self, text: str):
        """Append a line to the tab's log pane, scrolled to the end. ``Workbench.log`` mirrors every line of the
        main log here, so the batch's console output lands in both."""
        self.log_box.appendPlainText(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def set_progress(self, text: str):
        """Show the last 200 characters of ``text`` in the progress line above the log (mirrored from
        ``Workbench.set_progress`` as well)."""
        self.progress_label.setText(text[-200:])

    def _project_dir(self):
        """Folder of the project path as typed, or None: where the folder dialogs of the tab start."""
        text = self.project_edit.text().strip()
        return os.path.dirname(text) if text else None

    def _get(self, name):
        """Current value of form field ``name`` (its getter in ``form._fields``)."""
        return self.form._fields[name][0]()

    def pattern_dir(self) -> str | None:
        """The raw pattern folder: the field, else the project's folder, else None without a project."""
        return self._get("PATTERN_DIR") or (self.project.folder if self.project else None)

    def output_dir(self) -> str | None:
        """The output folder: the field, else ``processed`` next to the project, else None without a project."""
        out = self._get("OUTPUT_DIR")
        if out:
            return out
        return os.path.join(self.project.folder, "processed") if self.project else None

    def suffix(self) -> str:
        """The processed file suffix as typed, stripped ("" when blank)."""
        value = self._get("SUFFIX")
        return "" if value is None else str(value).strip()

    def _set_batch_buttons(self, have_project: bool):
        """Enable the per-tile and batch buttons when a project is loaded; the processing buttons stay off
        without a pattern processor."""
        for b in (self.btn_load_proc, self.btn_process, self.btn_load_index, self.btn_index, self.btn_use_geometry, self.btn_rescan):
            b.setEnabled(have_project)
        if self.wb.proc is None:
            self.btn_load_proc.setEnabled(False)
            self.btn_process.setEnabled(False)
            self.btn_process.setToolTip("The pattern processor is not available")

    def _busy(self, busy: bool):
        """Lock the tab (and the other tabs' entry points) while a batch runs. The batch processing runs on the GUI
        thread pumping events, so every control that could change the project, the folders or the processor's
        data set must be disabled, not just the batch buttons. Sets ``processing`` and ``wb.batch_running`` (which
        makes ``wb.worker_running()`` True), locks the workbench's buttons and the processor tab, and enables
        Cancel batch."""
        self.processing = busy
        self.wb.batch_running = busy
        self.form.setEnabled(not busy)  # project, folders, suffix, table, the browse buttons and both batch groups
        self._set_batch_buttons(not busy and self.project is not None)
        self.btn_project.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.wb._busy(busy)
        if self.wb.proc is not None:
            self.wb.tabs.setTabEnabled(0, not busy)

    def cancel(self):
        """Cancel batch: sets the flag the batches poll (after the current tile in processing, after the current
        band in indexing)."""
        self.cancel_requested = True
        self.log("Cancelling after the current step...")

    # ------------------------------------------------------------------------------------------ the project
    def load_project(self):
        """Read the project of the PROJECT_FILE field, log its tiles, the first tile's geometry and the phase
        numbers, and rescan the folders. A missing path or a read error shows a message box; the old project
        stays loaded on a read error."""
        path = self._get("PROJECT_FILE")
        if not path or not os.path.exists(str(path)):
            QMessageBox.information(self, "Project", "Choose the montage's .edaxh5 project first")
            return
        try:
            self.project = montage.read_project(str(path))
        except Exception as e:  # noqa: BLE001
            self.log(f"{type(e).__name__}: {e}")
            QMessageBox.critical(self, "Project", f"{type(e).__name__}: {e}")
            return
        p = self.project
        self.log(f"Project {p.name}: {len(p.tiles)} tiles ({os.path.basename(p.path)})")
        for note in p.skipped:
            self.log(f"  map group skipped (not an EBSD scan): {note}")
        if p.tiles:
            t = p.tiles[0]
            self.log(f"  first tile {t.name}: grid {t.n_cols} x {t.n_rows}, step {t.step_x} x {t.step_y} um, PC (EDAX) {t.pc}, "
                     f"sample tilt {t.sample_tilt}, camera elevation {t.camera_elevation}, {t.kv} kV; phases: "
                     + ", ".join(ph.name for ph in t.phases))
            self.log("  The indexing uses the PC, sample tilt and camera tilt of the indexing tab for every tile; compare them with these")
            self.log("  Phases of the project, by number: " + ", ".join(f"{i + 1}: {ph.name}" for i, ph in enumerate(t.phases)))
        try:
            self.rescan()
        except Exception as e:  # noqa: BLE001
            self.log(f"Rescan failed: {type(e).__name__}: {e}")
        self._set_batch_buttons(True)

    def _fresh_project(self) -> montage.Project:
        """A copy of the loaded project to match files against (the file is read once, when it is loaded;
        matching writes into the tiles)."""
        return copy.deepcopy(self.project)

    def _raw_tiles(self) -> list[montage.Tile]:
        """The project's tiles matched with their raw files in the raw pattern folder (a fresh match; the
        processed suffix is never preferred). A tile's ``status`` says why it is not ``ok``: "pattern folder not
        found", "only a processed file" (a processed copy in the raw folder is not a raw file), or what
        ``montage.match_tiles`` found wrong with the file."""
        project = self._fresh_project()
        folder = self.pattern_dir()
        if folder and os.path.isdir(folder):
            montage.pattern_folder_tiles(project, folder, prefer_suffix=None, log=self.log)
        else:
            self.log(f"Raw pattern folder not found: {folder}")
            for t in project.tiles:
                t.pattern, t.status = None, "pattern folder not found"
        suffix = self.suffix()
        for t in project.tiles:
            if t.pattern is not None and _has_suffix(t, suffix):
                t.pattern, t.status = None, "only a processed file"
        return project.tiles

    def _processed_tiles(self) -> list[montage.Tile]:
        """The project's tiles matched with their processed files in the output folder (the processed suffix
        preferred; a file without it gets status "no processed file"). A missing output folder leaves every tile
        without a file, silently: before the batch processing there is none."""
        project = self._fresh_project()
        out_dir = self.output_dir()
        suffix = self.suffix()
        if out_dir and os.path.isdir(out_dir):
            montage.pattern_folder_tiles(project, out_dir, prefer_suffix=suffix, log=self.log)
        for t in project.tiles:
            if t.pattern is not None and suffix and not _has_suffix(t, suffix):
                t.pattern, t.status = None, "no processed file"
        return project.tiles

    def rescan(self):
        """Match the tiles against both folders again and fill the table: tile, grid, raw file, processed file
        and status ("ok", the raw file's problem, or "processed file: ..." for a bad processed file, which is what
        Index all would skip; red when a file is there, grey when none is). Sets ``raw_tiles`` and
        ``processed_tiles``, the geometry warnings label and the tool tip of Index all (how many tiles it would
        take). No-op without a project."""
        if self.project is None:
            return
        raw = self._raw_tiles()
        processed = self._processed_tiles()
        self.raw_tiles, self.processed_tiles = raw, processed
        self.table.setRowCount(len(raw))
        for i, (t, tp) in enumerate(zip(raw, processed)):
            if tp.pattern is not None and not tp.ok:
                status = f"processed file: {tp.status}"
            elif t.ok or tp.pattern is None:
                status = t.status
            else:
                status = "ok"
            cells = [t.name, f"{t.n_cols} x {t.n_rows}", t.pattern.name if t.pattern else "-", tp.pattern.name if tp.pattern else "-", status]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j == 0:
                    item.setData(Qt.UserRole, t.h5_path)  # rows are told apart by the map's path, not its name
                    item.setToolTip(t.h5_path)
                if j == 4 and status != "ok":
                    item.setForeground(Qt.darkRed if t.pattern is not None or tp.pattern is not None else Qt.gray)
                self.table.setItem(i, j, item)
        n_raw, n_proc = sum(t.ok for t in raw), sum(t.ok for t in processed)
        self.log(f"{n_raw} of {len(raw)} tiles have a raw pattern file in {self.pattern_dir()}; {n_proc} have a processed file in {self.output_dir()}")
        # the geometry warnings compare the tiles' headers too, so they need the raw files matched
        project = self._fresh_project()
        montage.match_tiles(project, [t.pattern for t in raw if t.pattern is not None])
        warnings = montage.geometry_warnings(project)
        self.geometry_label.setText("\n".join("NB! " + w for w in warnings) if warnings else "All tiles share the pattern center, tilts, step and phases")
        for w in warnings:
            self.log("NB! " + w)
        self.btn_index.setEnabled(not self.processing)
        self.btn_index.setToolTip(f"{n_proc} processed tiles" if self.index_processed_check.isChecked() else f"{n_raw} raw tiles")

    def selected_tile(self, processed: bool) -> montage.Tile | None:
        """The tile of the selected table row, from ``processed_tiles`` or ``raw_tiles``, when it has a usable
        file; otherwise a message box says why and None is returned."""
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            QMessageBox.information(self, "Tiles", "Select a tile in the table first")
            return None
        tiles = self.processed_tiles if processed else self.raw_tiles
        t = tiles[rows[0].row()]
        if not t.ok:
            QMessageBox.information(self, "Tiles", f"{t.name}: {t.status}")
            return None
        return t

    def copy_geometry(self):
        """Put the first tile's PC (EDAX convention), sample tilt and camera elevation into the indexing tab's
        fields (each only when the project has it)."""
        if self.project is None or not self.project.tiles:
            return
        t = self.project.tiles[0]
        fields = self.wb.form._fields
        if t.pc is not None:
            fields["PC"][1](tuple(round(v, 6) for v in t.pc))
            fields["PC_CONVENTION"][1]("edax")
        if t.sample_tilt is not None:
            fields["SAMPLE_TILT"][1](round(float(t.sample_tilt), 3))
        if t.camera_elevation is not None:
            fields["CAMERA_TILT"][1](float(t.camera_elevation))
        self.log(f"Indexing tab: PC {t.pc} (edax), sample tilt {t.sample_tilt}, camera tilt {t.camera_elevation} from the project; check them")

    # ------------------------------------------------------------------------------------------ 1. processing
    def load_selected_in_processor(self):
        """Load the selected tile's raw file in the pattern processor tab and switch to it."""
        t = self.selected_tile(processed=False)
        if t is None or self.wb.proc is None:
            return
        self.wb.proc.load_path(t.pattern.path)
        self.wb.tabs.setCurrentIndex(0)

    def process_all(self):
        """Process all tiles: the batch processing, on the GUI thread (the processor's ``batch_process`` pumps
        events through the progress callbacks).

        Refused with a message box while a worker or the processor's NLPAR step runs, when the processor has no
        ``batch_process``, when no tile has a usable raw file, or when the suffix is blank (the output would
        overwrite the raw files). Then: the indexing tab is unloaded, the tab locked, and with *Static
        background from all tiles together* ticked and the processor's static background on, one background is
        computed over every tile's patterns before the tiles are processed one by one into
        ``montage.output_name(tile, suffix, out_dir)``. The table's status column shows the processor's stage
        ("load", "nlpar", "save", "done") per tile. Cancel stops the shared background pass at once and the tiles
        after the current one (the processor deletes its partial file). The processor is unloaded in any case
        and the tab unlocked before the results are logged, one line per tile from the processor's result dicts
        (``ok``, ``output``, ``n_saved``, ``seconds``, ``n_excluded``, ``source``, ``error``), and the folders
        rescanned.
        """
        proc = self.wb.proc
        if proc is None or self.project is None:
            return
        if self.wb._refuse_while_busy():
            return
        if proc._nlpar_busy:
            QMessageBox.information(self, "Busy", "Wait for the pattern processor's NLPAR step to finish")
            return
        all_tiles = self._raw_tiles()
        tiles = [t for t in all_tiles if t.ok]
        if not tiles:
            QMessageBox.information(self, "Tiles", "No tile has a raw pattern file")
            return
        out_dir, suffix = self.output_dir(), self.suffix()
        if not suffix:
            QMessageBox.information(self, "Suffix", "The processed file suffix must not be blank, or the output would overwrite the raw files")
            return
        skipped = [t for t in all_tiles if not t.ok]
        for t in skipped:
            self.log(f"  {t.name} is skipped: {t.status}")
        os.makedirs(out_dir, exist_ok=True)
        jobs = [(t.pattern.path, montage.output_name(t, suffix, out_dir)) for t in tiles]
        self.wb.unload_indexing("the batch processing starts")
        self.cancel_requested = False
        self._busy(True)
        self.wb.tabs.setCurrentWidget(self)
        t_start = time.perf_counter()
        results = []
        try:
            shared = None
            if self.shared_bg_check.isChecked() and proc.static_bg_chk.isChecked():
                self.log(f"Static background over all {len(tiles)} tiles...")

                def bg_progress(done, total, path):
                    self.set_progress(f"Shared background: {done} / {total} patterns ({os.path.basename(path)})")
                    QApplication.processEvents()
                    if self.cancel_requested:  # stops the pass over the tiles at once, not after all of them
                        raise _Cancelled()

                shared = proc.compute_shared_background([t.pattern.path for t in tiles], progress=bg_progress)
                if self.cancel_requested:
                    raise _Cancelled()
                self.log(f"Shared background done in {time.perf_counter() - t_start:.0f} s")
            elif self.shared_bg_check.isChecked():
                self.log("Static background is off in the processor tab: no shared background")

            def progress(i, n, path, stage):
                self.set_progress(f"Tile {i + 1} / {n}: {stage} {os.path.basename(path)} ({(time.perf_counter() - t_start) / 60:.1f} min)")
                if stage == "load":
                    self.log(f"Tile {i + 1} / {n}: {os.path.basename(path)}")
                if 0 <= i < len(tiles):
                    self._mark_row_by_path(tiles[i].h5_path, stage)
                QApplication.processEvents()

            results = proc.batch_process(jobs, shared_background=shared, progress=progress, cancel=lambda: self.cancel_requested)
        except Exception as e:  # noqa: BLE001
            results = list(proc.batch_results or [])
            # the processor's BatchCancelled is matched by name: its module is not imported here
            if type(e).__name__ in ("BatchCancelled", "_Cancelled"):
                self.log("Batch processing cancelled")
            else:
                self.log(traceback.format_exc())
                QMessageBox.critical(self, "Batch processing", f"{type(e).__name__}: {e}")
        finally:
            try:
                if proc.f_orig is not None:
                    proc.unload()
            except Exception:  # noqa: BLE001
                pass
            gc.collect()
            self._busy(False)
        n_ok = sum(1 for r in results if r.get("ok"))
        for r in results:
            if r.get("ok"):
                self.log(f"  {os.path.basename(r['output'])}: {r.get('n_saved')} patterns in {r.get('seconds', 0):.0f} s"
                         + (f", {r['n_excluded']} outside the IQ range kept unprocessed" if r.get("n_excluded") else ""))
            else:
                self.log(f"  {os.path.basename(r.get('source', '?'))}: FAILED: {r.get('error')}")
        self.log(f"Batch processing: {n_ok} of {len(jobs)} tiles written to {out_dir} in {(time.perf_counter() - t_start) / 60:.1f} min"
                 + (f" ({len(skipped)} tiles without a usable raw file skipped)" if skipped else "") + "; the processor is unloaded")
        self.set_progress("")
        self.rescan()

    # ------------------------------------------------------------------------------------------ 2. indexing
    def load_selected_in_indexer(self):
        """Load the selected tile (processed or raw file, as *Index the processed files* says) in the indexing
        tab and switch to it. The tile's grid and step come from its header, so an ROI, map shape, step, scan
        file or map filter left over from another data set is cleared first."""
        t = self.selected_tile(processed=self.index_processed_check.isChecked())
        if t is None:
            return
        if self.wb._refuse_while_busy():
            return
        fields = self.wb.form._fields
        fields["PATTERN_FILE"][1](t.pattern.path)
        for name in ("SCAN_FILE", "EDAXH5_MAP", "NAV_SHAPE", "STEP_SIZE", "ROI"):
            fields[name][1](None)
        self.wb.prepared = None
        self.wb.tabs.setCurrentIndex(1)
        self.wb.load_patterns_clicked()

    def index_all(self):
        """Index all tiles: the batch indexing on the workbench's worker thread, with this tab's live view.

        Refused (message box) while a worker runs, when no tile has a usable file of the chosen kind, or without
        a master pattern; the processor and the indexing tab are then unloaded and the tab locked. The worker's
        ``work(log, emit)`` closure never touches widgets: it loads (or reuses) the harmonics, maps the masters to
        the project's phases (``montage.phase_mapping``, which raises when it cannot), copies the project to
        ``<out_dir>/<stem>_sph_bw<BANDWIDTH>.edaxh5`` (or keeps it with *Resume*, skipping the tiles that already
        hold spherical results), and per tile: loads the patterns memory mapped with the tile's grid, finds the
        all-zero patterns on the raw data (they cannot be indexed and the background removal would rescale
        them), preprocesses, indexes band by band with ``sph_core.index_blocks``, applies TWIST and writes the
        tile into the output project right away. It frees the tile and the GPU memory after every tile, and
        stops after the current band when Cancel batch was pressed.

        Events emitted to the GUI thread, consumed by the local ``on_event``: ``"tile_start"`` (index, n_tiles,
        tile, h5_path, iq (the stretched quasi-IQ map, (rows, cols) float), one_line) sets the live view's base
        map and marks the table row "indexing"; ``"run_start"`` (r0, c0, rows, cols, n_patterns, signal, label,
        st, live_view=this tab's view) and the ``index_blocks`` events (``"block_start"``, ``"progress"``,
        ``"chunk_done"``, ``"block_done"`` with ``xmap`` replaced by ``rgb``, the band's IPF-Z image made on the
        worker) and ``"run_end"`` go to ``Workbench.on_run_event``.

        ``work`` returns a dict for ``on_done``: ``done`` and ``already`` (tile names indexed now / kept from
        the earlier run), ``failed`` ((name, error) pairs), ``n_tiles``, ``n_skipped`` (tiles without a usable
        file), ``minutes``, ``output`` (the output project's path, kept in ``last_output_project``),
        ``harmonics`` and ``hkey`` (kept on the workbench for the indexing tab and the next batch), ``cancelled``.
        ``on_done`` unlocks the tab, logs the summary and the failures, warns when the output is partial (two IQ
        scales in one project unless the Hough IQ is kept), and rescans. A failed worker (``on_failed``) unlocks
        and rescans; the workbench shows the traceback.
        """
        wb = self.wb
        if self.project is None:
            return
        if wb._refuse_while_busy():  # before anything is unloaded or locked
            return
        processed = self.index_processed_check.isChecked()
        all_tiles = self._processed_tiles() if processed else self._raw_tiles()
        tiles = [t for t in all_tiles if t.ok]
        skipped = [t for t in all_tiles if not t.ok]
        if not tiles:
            QMessageBox.information(self, "Tiles", "No tile has a " + ("processed" if processed else "raw") + " pattern file")
            return
        st = wb.settings()
        if not st.MASTER_FILE:
            QMessageBox.information(self, "Master pattern", "Set the master pattern file in the indexing tab first")
            return
        if not wb._unload_processor():
            return
        wb.unload_indexing("the batch indexing starts")
        out_dir = self.output_dir()
        os.makedirs(out_dir, exist_ok=True)
        out_project = os.path.join(out_dir, f"{self.project.stem}_sph_bw{st.BANDWIDTH}.edaxh5")
        project = self.project
        live = self.live
        self.cancel_requested = False
        wb.tabs.setCurrentWidget(self)
        kw = self.kw
        iq_prop = None if self.keep_iq_check.isChecked() else "iq"
        resume = self.resume_check.isChecked()
        phase_spec = self._get("PHASE_MAP")
        is_cancelled = lambda: self.cancel_requested  # noqa: E731  (read from the worker thread: a plain bool)
        for t in skipped:
            self.log(f"  {t.name} is skipped: {t.status}")

        def work(log, emit):
            t_all = time.perf_counter()
            hkey = (st.MASTER_FILE, st.MASTER_BANDWIDTH, st.MASTER_ENERGY, False)
            harmonics = wb.harmonics if wb.harmonics is not None and wb.harmonics_key == hkey else sph_core.load_harmonics(st, log)
            # which project phase each master is: refused when it cannot be told, so the Phase column is never wrong
            master_names = [str(getattr(getattr(h, "phase", None), "name", "") or "") for h in harmonics]
            phase_map = montage.phase_mapping(master_names, tiles[0].phases, phase_spec, log)
            already = set()
            if resume and os.path.exists(out_project):
                already = montage.indexed_tiles(out_project, tiles)
                log(f"Output project: {out_project} exists and is kept; {len(already)} of its tiles already hold spherical results "
                    "and are skipped (untick Resume to start afresh)")
            else:
                montage.copy_project(project.path, out_project, overwrite=True)
                log(f"Output project: {out_project} (a copy of the project; each tile's orientation table is replaced as it is indexed)")
            comment = (f"Spherical indexing (kikuchipy fork), bandwidth {st.BANDWIDTH}, PC {tuple(st.PC)} ({st.PC_CONVENTION}), "
                       f"sample tilt {st.SAMPLE_TILT}, camera tilt {st.CAMERA_TILT}, master {os.path.basename(str(st.MASTER_FILE))}, "
                       f"{time.strftime('%Y-%m-%d %H:%M')}")
            done, failed = [], []
            skipped_done = [t.name for t in tiles if t.h5_path in already]
            det = None
            for i, tile in enumerate(tiles):
                if is_cancelled():
                    log("Batch indexing cancelled")
                    break
                if tile.h5_path in already:
                    continue
                log(f"Tile {i + 1} / {len(tiles)}: {tile.name} ({tile.pattern.name})")
                try:
                    st_t = dataclasses.replace(st, PATTERN_FILE=tile.pattern.path, SCAN_FILE=None, EDAXH5_MAP=None, NAV_SHAPE=None,
                                               STEP_SIZE=None, ROI=None)
                    s, _ = sph_core.load_patterns(st_t, scan=montage.scan_dict(tile), log=log)
                    # all-zero patterns are found on the raw data: the background removal would rescale them
                    iq, nav_mask = montage.tile_stats(s, progress=lambda f: emit("progress", {"frac_index": f, "frac_refine": 1.0, "what": "iq"}))
                    if nav_mask is not None:
                        log(f"  {int(nav_mask.sum())} all-zero patterns are not indexed")
                    sph_core.preprocess(s, st_t, log)
                    quiet = (lambda _msg: None) if det is not None else log  # the detector is described once
                    det = sph_core.make_detector(st_t, s, harmonics, quiet)
                    nav_shape = tuple(int(v) for v in s.axes_manager.navigation_shape[::-1])
                    n_run = int(np.prod(nav_shape)) - (0 if nav_mask is None else int(nav_mask.sum()))
                    iq_map = sph_core.stretch(iq)
                    if iq_map.ndim == 1:  # a tile of one row or one column: shown as one row
                        iq_map = iq_map.reshape(1, -1)
                    emit("tile_start", {"index": i, "n_tiles": len(tiles), "tile": tile.name, "h5_path": tile.h5_path, "iq": iq_map,
                                        "one_line": len(nav_shape) == 1})
                    emit("run_start", {"r0": 0, "c0": 0, "rows": nav_shape[0] if len(nav_shape) == 2 else 1, "cols": nav_shape[-1],
                                       "n_patterns": n_run, "signal": s, "label": tile.name, "st": st_t, "live_view": live})

                    def on_block_event(name, data):
                        if name == "block_done":
                            data = dict(data)
                            xb = data.pop("xmap")
                            data["rgb"] = None if xb is None else sph_core.block_rgb(xb, data, "z")
                        emit(name, data)
                        if is_cancelled():
                            raise _Cancelled()

                    t0 = time.perf_counter()
                    xmap_indexed, dt, speed = sph_core.index_blocks(
                        s, harmonics, det, st_t, log, navigation_mask=nav_mask, on_event=on_block_event,
                        target_seconds=kw.LIVE_BLOCK_SECONDS, first_patterns=max(nav_shape[-1], min(300, max(1, n_run // 10))))
                    xmap = sph_core.apply_twist(xmap_indexed, st.TWIST)
                    counts = montage.write_tile_result(out_project, tile, xmap, iq_prop=iq_prop, comment=comment, phase_map=phase_map)
                    emit("run_end", {"success": True})
                    log(f"  {counts['n_indexed']} patterns indexed in {time.perf_counter() - t0:.0f} s ({speed:.0f} patterns/s), written to the output project")
                    done.append(tile.name)
                except _Cancelled:
                    emit("run_end", {"success": False})
                    log("Batch indexing cancelled")
                    break
                except Exception as e:  # noqa: BLE001
                    emit("run_end", {"success": False})
                    log(f"  FAILED: {type(e).__name__}: {e}")
                    failed.append((tile.name, f"{type(e).__name__}: {e}"))
                finally:
                    s = xmap = xmap_indexed = None
                    gc.collect()
                    sph_core.free_gpu_memory()
            minutes = (time.perf_counter() - t_all) / 60
            return {"done": done, "failed": failed, "already": skipped_done, "n_tiles": len(tiles), "n_skipped": len(skipped),
                    "minutes": minutes, "output": out_project, "harmonics": harmonics, "hkey": hkey, "cancelled": is_cancelled()}

        def on_done(result):
            self._busy(False)
            # the expanded master pattern(s) are kept for the indexing tab and the next batch
            wb.harmonics, wb.harmonics_key = result["harmonics"], result["hkey"]
            self.last_output_project = result["output"]
            n_have = len(result["done"]) + len(result["already"])
            self.log(f"Batch indexing: {len(result['done'])} tiles indexed in {result['minutes']:.1f} min"
                     + (f", {len(result['already'])} kept from the earlier run" if result["already"] else "")
                     + (f"; {len(result['failed'])} failed" if result["failed"] else "")
                     + ("; cancelled" if result["cancelled"] else "")
                     + (f"; {result['n_skipped']} tiles without a usable pattern file were skipped" if result["n_skipped"] else ""))
            for name, err in result["failed"]:
                self.log(f"  {name}: {err}")
            self.output_label.setText(f"Output project: {result['output']}")
            partial = n_have < result["n_tiles"] + result["n_skipped"]
            self.log(f"Output project: {result['output']}" + (" (tiles that failed or were skipped keep their Hough results)" if partial else ""))
            if partial and iq_prop is not None:
                self.log("NB! The IQ column of the indexed tiles is the spherical IQ (0..1) while the other tiles keep the Hough IQ: "
                         "two scales in one project until every tile is indexed (or tick 'Keep the project's (Hough) IQ')")
            self.set_progress("")
            self.rescan()

        def on_failed():
            self._busy(False)
            self.rescan()

        def on_event(name, data):
            if name == "tile_start":
                title = f"{data['tile']} ({data['index'] + 1} / {data['n_tiles']})" + (" shown as one row" if data.get("one_line") else "")
                self.live.set_base_map(data["iq"], title)
                self._mark_row_by_path(data["h5_path"], "indexing")
                return
            wb.on_run_event(name, data)

        if not wb._start(work, on_done, on_event, on_failed=on_failed):
            return
        self._busy(True)

    def _mark_row_by_path(self, h5_path, text):
        """Write ``text`` into the status column of the table row of the map ``h5_path`` (the batches' per-tile
        stage); no row, nothing."""
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item is not None and item.data(Qt.UserRole) == h5_path:
                self.table.setItem(i, 4, QTableWidgetItem(text))
