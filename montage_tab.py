"""Tab 3 of the Kikuchi workbench: batch processing and batch indexing of an EDAX APEX montage.

Two separate jobs, run one after the other by the user:

1. **Process all tiles** (pattern processor): a static background is computed from the patterns of all tiles
   together, then every tile is processed with the settings of the pattern processor tab and written to its own
   ``<tile><suffix>.up1`` file in the output folder, one tile at a time; the processor is unloaded after each tile
   and after the batch, so that nothing of the montage stays in RAM.
2. **Index all tiles** (spherical indexing): every processed tile is indexed with the settings of the indexing
   tab (one projection center, sample tilt and camera tilt for the whole montage), and its orientation table is
   written into a copy of the project, ``<project>_sph_bw<bandwidth>.edaxh5``, which OIM Analysis or MTEX can
   stitch like the original. Nothing is stitched here.

The tiles are the maps of the project; their pattern files are found by the GUID in the .up1 header (see
montage.py and STITCH_DATASET_NOTES.md).
"""
from __future__ import annotations

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


class MontageTab(QWidget):
    def __init__(self, wb):
        super().__init__()
        self.wb = wb
        kw = importlib.import_module(type(wb).__module__)
        self.kw = kw
        self.project: montage.Project | None = None
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
        self.suffix_edit = form.add_text("SUFFIX", "Processed file suffix:", DEFAULT_SUFFIX, "Appended to the tile's file name: <tile>_proc.up1")
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
        self.keep_iq_check = QCheckBox("Keep the project's (Hough) IQ in the output")
        self.keep_iq_check.setChecked(True)
        self.keep_iq_check.setToolTip("On: the IQ column of the output project stays the Hough IQ written at acquisition, the one OIM and MTEX use "
                                      "for stitching. Off: it is replaced by the spherical image quality (0..1 scale). The spherical score always "
                                      "goes into CI")
        form.add_widget(self.keep_iq_check)
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
        self.log_box.appendPlainText(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def set_progress(self, text: str):
        self.progress_label.setText(text[-200:])

    def _project_dir(self):
        text = self.project_edit.text().strip()
        return os.path.dirname(text) if text else None

    def _get(self, name):
        return self.form._fields[name][0]()

    def pattern_dir(self) -> str | None:
        return self._get("PATTERN_DIR") or (self.project.folder if self.project else None)

    def output_dir(self) -> str | None:
        out = self._get("OUTPUT_DIR")
        if out:
            return out
        return os.path.join(self.project.folder, "processed") if self.project else None

    def suffix(self) -> str:
        value = self._get("SUFFIX")
        return DEFAULT_SUFFIX if value is None else str(value)

    def _set_batch_buttons(self, have_project: bool):
        for b in (self.btn_load_proc, self.btn_process, self.btn_load_index, self.btn_index, self.btn_use_geometry, self.btn_rescan):
            b.setEnabled(have_project)
        if self.wb.proc is None:
            self.btn_load_proc.setEnabled(False)
            self.btn_process.setEnabled(False)
            self.btn_process.setToolTip("The pattern processor is not available")

    def _busy(self, busy: bool):
        self.processing = busy
        self._set_batch_buttons(not busy and self.project is not None)
        self.btn_project.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.wb._busy(busy)

    def cancel(self):
        self.cancel_requested = True
        self.log("Cancelling after the current step...")

    # ------------------------------------------------------------------------------------------ the project
    def load_project(self):
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
        if p.tiles:
            t = p.tiles[0]
            self.log(f"  first tile {t.name}: grid {t.n_cols} x {t.n_rows}, step {t.step_x} x {t.step_y} um, PC (EDAX) {t.pc}, "
                     f"sample tilt {t.sample_tilt}, camera elevation {t.camera_elevation}, {t.kv} kV; phases: "
                     + ", ".join(ph.name for ph in t.phases))
            self.log("  The indexing uses the PC, sample tilt and camera tilt of the indexing tab for every tile; compare them with these")
        self.rescan()
        self._set_batch_buttons(True)

    def _raw_tiles(self) -> list[montage.Tile]:
        """The tiles matched with their raw files (a fresh match; the processed suffix is never preferred)."""
        project = montage.read_project(self.project.path)
        montage.pattern_folder_tiles(project, self.pattern_dir(), prefer_suffix=None, log=self.log)
        suffix = self.suffix()
        for t in project.tiles:  # a processed copy in the raw folder is not a raw file
            if t.pattern is not None and os.path.splitext(t.pattern.name)[0].endswith(suffix) and suffix:
                t.pattern, t.status = None, "only a processed file"
        return project.tiles

    def _processed_tiles(self) -> list[montage.Tile]:
        project = montage.read_project(self.project.path)
        out_dir = self.output_dir()
        if out_dir and os.path.isdir(out_dir):
            montage.pattern_folder_tiles(project, out_dir, prefer_suffix=self.suffix(), log=self.log)
        suffix = self.suffix()
        for t in project.tiles:
            if t.pattern is not None and suffix and not os.path.splitext(t.pattern.name)[0].endswith(suffix):
                t.pattern, t.status = None, "no processed file"
        return project.tiles

    def rescan(self):
        if self.project is None:
            return
        raw = self._raw_tiles()
        processed = self._processed_tiles()
        self.raw_tiles, self.processed_tiles = raw, processed
        self.table.setRowCount(len(raw))
        for i, (t, tp) in enumerate(zip(raw, processed)):
            status = t.status if t.ok or tp.pattern is None else ("ok" if tp.ok else tp.status)
            cells = [t.name, f"{t.n_cols} x {t.n_rows}", t.pattern.name if t.pattern else "-", tp.pattern.name if tp.pattern else "-", status]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j == 4 and status != "ok":
                    item.setForeground(Qt.darkRed if t.pattern is not None or tp.pattern is not None else Qt.gray)
                self.table.setItem(i, j, item)
        n_raw, n_proc = sum(t.ok for t in raw), sum(t.ok for t in processed)
        self.log(f"{n_raw} of {len(raw)} tiles have a raw pattern file in {self.pattern_dir()}; {n_proc} have a processed file in {self.output_dir()}")
        project = montage.read_project(self.project.path)
        montage.match_tiles(project, [t.pattern for t in raw if t.pattern is not None])
        warnings = montage.geometry_warnings(project)
        self.geometry_label.setText("\n".join("NB! " + w for w in warnings) if warnings else "All tiles share the pattern center, tilts, step and phases")
        for w in warnings:
            self.log("NB! " + w)
        self.btn_index.setEnabled(self.project is not None and not self.processing)
        self.btn_index.setToolTip(f"{n_proc} processed tiles" if self.index_processed_check.isChecked() else f"{n_raw} raw tiles")

    def selected_tile(self, processed: bool) -> montage.Tile | None:
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
        t = self.selected_tile(processed=False)
        if t is None or self.wb.proc is None:
            return
        self.wb.proc.load_path(t.pattern.path)
        self.wb.tabs.setCurrentIndex(0)

    def process_all(self):
        proc = self.wb.proc
        if proc is None or self.project is None:
            return
        if self.wb.worker is not None and self.wb.worker.isRunning():
            QMessageBox.information(self, "Busy", "Wait for the running step to finish")
            return
        if getattr(proc, "_nlpar_busy", False):
            QMessageBox.information(self, "Busy", "Wait for the pattern processor's NLPAR step to finish")
            return
        if not hasattr(proc, "batch_process"):
            QMessageBox.critical(self, "Pattern processor", "This pattern processor has no batch_process(): update the kikuchi-pattern-processor folder (git pull)")
            return
        tiles = [t for t in self._raw_tiles() if t.ok]
        if not tiles:
            QMessageBox.information(self, "Tiles", "No tile has a raw pattern file")
            return
        out_dir, suffix = self.output_dir(), self.suffix()
        if not suffix:
            QMessageBox.information(self, "Suffix", "The processed file suffix must not be blank, or the output would overwrite the raw files")
            return
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
                self._mark_row(i, stage)
                QApplication.processEvents()

            results = proc.batch_process(jobs, shared_background=shared, progress=progress, cancel=lambda: self.cancel_requested)
        except Exception as e:  # noqa: BLE001
            results = list(getattr(proc, "batch_results", []) or [])
            if type(e).__name__ in ("BatchCancelled", "_Cancelled"):
                self.log("Batch processing cancelled")
            else:
                self.log(traceback.format_exc())
                QMessageBox.critical(self, "Batch processing", f"{type(e).__name__}: {e}")
        finally:
            try:
                if getattr(proc, "f_orig", None) is not None:
                    proc.unload()
            except Exception:  # noqa: BLE001
                pass
            gc.collect()
            self._busy(False)
        n_ok = sum(1 for r in results if r.get("ok"))
        for r in results:
            if r.get("ok"):
                self.log(f"  {os.path.basename(r['output'])}: {r.get('n_saved')} patterns in {r.get('seconds', 0):.0f} s"
                         + (f", {r['n_excluded']} excluded (zeros)" if r.get("n_excluded") else ""))
            else:
                self.log(f"  {os.path.basename(r.get('source', '?'))}: FAILED: {r.get('error')}")
        self.log(f"Batch processing: {n_ok} of {len(jobs)} tiles written to {out_dir} in {(time.perf_counter() - t_start) / 60:.1f} min; "
                 "the processor is unloaded")
        self.set_progress("")
        self.rescan()

    def _mark_row(self, i, text):
        if 0 <= i < self.table.rowCount():
            self.table.setItem(i, 4, QTableWidgetItem(text))

    # ------------------------------------------------------------------------------------------ 2. indexing
    def load_selected_in_indexer(self):
        t = self.selected_tile(processed=self.index_processed_check.isChecked())
        if t is None:
            return
        self.wb.form._fields["PATTERN_FILE"][1](t.pattern.path)
        self.wb.form._fields["SCAN_FILE"][1](None)
        self.wb.prepared = None
        self.wb.tabs.setCurrentIndex(1)
        self.wb.load_patterns_clicked()

    def index_all(self):
        wb = self.wb
        if self.project is None:
            return
        processed = self.index_processed_check.isChecked()
        tiles = [t for t in (self._processed_tiles() if processed else self._raw_tiles()) if t.ok]
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
        self.wb.tabs.setCurrentWidget(self)
        kw = self.kw
        iq_prop = None if self.keep_iq_check.isChecked() else "iq"
        is_cancelled = lambda: self.cancel_requested  # noqa: E731  (read from the worker thread: a plain bool)

        def work(log, emit):
            t_all = time.perf_counter()
            hkey = (st.MASTER_FILE, st.MASTER_BANDWIDTH, st.MASTER_ENERGY, False)
            harmonics = wb.harmonics if wb.harmonics is not None and wb.harmonics_key == hkey else sph_core.load_harmonics(st, log)
            montage.copy_project(project.path, out_project, overwrite=True)
            log(f"Output project: {out_project} (a copy of the project; each tile's orientation table is replaced as it is indexed)")
            comment = (f"Spherical indexing (kikuchipy fork), bandwidth {st.BANDWIDTH}, PC {tuple(st.PC)} ({st.PC_CONVENTION}), "
                       f"sample tilt {st.SAMPLE_TILT}, camera tilt {st.CAMERA_TILT}, master {os.path.basename(str(st.MASTER_FILE))}, "
                       f"{time.strftime('%Y-%m-%d %H:%M')}")
            done, failed = [], []
            det = None
            for i, tile in enumerate(tiles):
                if is_cancelled():
                    log("Batch indexing cancelled")
                    break
                log(f"Tile {i + 1} / {len(tiles)}: {tile.name} ({tile.pattern.name})")
                try:
                    st_t = dataclasses.replace(st, PATTERN_FILE=tile.pattern.path, SCAN_FILE=None, EDAXH5_MAP=None, NAV_SHAPE=None,
                                               STEP_SIZE=None, ROI=None)
                    s, _ = sph_core.load_patterns(st_t, scan=montage.scan_dict(tile), log=log)
                    sph_core.preprocess(s, st_t, log)
                    iq, nav_mask = montage.tile_stats(s, progress=lambda f: emit("progress", {"frac_index": f, "frac_refine": 1.0, "what": "iq"}))
                    if nav_mask is not None:
                        log(f"  {int(nav_mask.sum())} zero patterns (excluded by the processor) are not indexed")
                    quiet = (lambda _msg: None) if det is not None else log
                    det = sph_core.make_detector(st_t, s, harmonics, quiet)
                    nav_shape = tuple(int(v) for v in s.axes_manager.navigation_shape[::-1])
                    n_run = int(np.prod(nav_shape)) - (0 if nav_mask is None else int(nav_mask.sum()))
                    emit("tile_start", {"index": i, "n_tiles": len(tiles), "tile": tile.name, "iq": sph_core.stretch(iq)})
                    emit("run_start", {"r0": 0, "c0": 0, "rows": nav_shape[0], "cols": nav_shape[-1] if len(nav_shape) == 2 else 1,
                                       "n_patterns": n_run, "signal": s, "label": tile.name, "st": st_t, "live_view": live})

                    def on_event(name, data):
                        if name == "block_done":
                            data = dict(data)
                            xb = data.pop("xmap")
                            data["rgb"] = None if xb is None else sph_core.block_rgb(xb, data, "z")
                        emit(name, data)
                        if is_cancelled():
                            raise _Cancelled()

                    t0 = time.perf_counter()
                    xmap_indexed, dt, speed = sph_core.index_blocks(
                        s, harmonics, det, st_t, log, navigation_mask=nav_mask, on_event=on_event,
                        target_seconds=kw.LIVE_BLOCK_SECONDS, first_patterns=max(nav_shape[-1], min(300, max(1, n_run // 10))))
                    xmap = sph_core.apply_twist(xmap_indexed, st.TWIST)
                    counts = montage.write_tile_result(out_project, tile, xmap, iq_prop=iq_prop, comment=comment)
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
            return {"done": done, "failed": failed, "n_tiles": len(tiles), "minutes": minutes, "output": out_project}

        def on_done(result):
            self._busy(False)
            self.last_output_project = result["output"]
            self.log(f"Batch indexing: {len(result['done'])} of {result['n_tiles']} tiles indexed in {result['minutes']:.1f} min")
            for name, err in result["failed"]:
                self.log(f"  {name}: {err}")
            self.output_label.setText(f"Output project: {result['output']}")
            self.log(f"Output project: {result['output']}" + (" (tiles that failed or were skipped keep their Hough results)"
                                                              if len(result["done"]) < result["n_tiles"] else ""))
            self.set_progress("")

        def on_event(name, data):
            if name == "tile_start":
                self.live.set_base_map(data["iq"], f"{data['tile']} ({data['index'] + 1} / {data['n_tiles']})")
                self._mark_row_by_name(data["tile"], "indexing")
                return
            wb.on_run_event(name, data)
            if name == "run_end":
                self._mark_row_by_name(None, "")

        self._busy(True)
        wb._start(work, on_done, on_event, on_failed=lambda: self._busy(False))

    def _mark_row_by_name(self, name, text):
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item is not None and name is not None and item.text() == name:
                self.table.setItem(i, 4, QTableWidgetItem(text))


class _Cancelled(Exception):
    pass
