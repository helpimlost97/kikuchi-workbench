"""Live view of a spherical indexing run: the map fills with IPF colours band by band, next to the current pattern.

``LiveView`` is the animated progress view of the workbench's indexing tab and of the montage tab (each has its
own instance). It is a QWidget with one matplotlib figure (map on the left, pattern on the right) and a one-line
status label. Everything in this module runs on the GUI thread: the indexing runs on the ``Worker`` thread of
``kikuchi_workbench.py``, whose events reach ``Workbench.on_run_event`` through a queued signal, and that method
translates them into the calls below. The map coordinates passed in are absolute (``r0``/``c0`` of the run added).

Event protocol, in the order of a run (``sph_core.index_blocks`` emits the events, the workbench forwards them):

    view.set_base_map(iq, "Quasi image quality")   # when the data set is loaded: the grey quasi-IQ map
    view.set_pattern_source(lambda r, c: pattern)  # once: how to fetch a pattern for the right-hand panel
    view.begin_run(r0, c0, rows, cols, n_patterns)  # "run_start": outline of the region, timers on
    view.block_started(r0, r1, c0, c1, n)           # "block_start": a band of rows is being indexed, tinted orange
    view.block_progress(frac_index, frac_refine)    # "progress": left-to-right sweep inside the band
    view.chunk_done(rows, cols, packed=p, colour=fn) # "chunk_done": a finished dask chunk, painted at the next tick
    view.block_done(r0, r1, c0, c1, rgb, n, seconds) # "block_done": the band repainted whole with its IPF colours
    view.set_progress(done, total, eta); view.set_speed(pps)  # the status line's numbers
    view.end_run(True)                              # "run_end" (False when cancelled or failed): final status

During a band the patterns of every finished dask chunk are painted with their IPF colours (a GPU batch or a CPU
chunk at a time, the refined orientation when refinement is on), so the picture grows in file order several
times a second as in EMSphInx. The chunks arrive as packed result rows with a colouring function and are
coloured in one go at the next redraw tick, so the indexing thread does no colouring and the GUI colours at most
``1000 / _REDRAW_MS`` times a second. The orange left-to-right sweep is only used in a band in which no chunk has
been painted (no colour keys, e.g. harmonics without a phase).

Rendering is cheap on purpose: one RGB float32 image (``LiveView.image``) is kept and written into by slices, and
the canvas is redrawn from a timer only when something changed, however many events arrive. What is pushed to the
AxesImage is an RGBA uint8 copy of it, subsampled with a stride when the map has more pixels than the axes (a
1100 x 2100 map in a ~900 px wide axes is shown at stride 2, which is what nearest-neighbour display would show
anyway); only the rows that changed since the last redraw are converted. This keeps a redraw of a large map at a
few tens of ms instead of ~150 ms for matplotlib's float RGB path. The outline of the run rectangle and the
marker of the displayed pattern are artists updated in place.
"""
from __future__ import annotations

import math
import time

import numpy as np
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

_ORANGE = np.array([1.0, 0.55, 0.0], dtype=np.float32)
_TINT = 0.35         # weight of orange in the current band
_SWEEP_TINT = 0.70   # weight of orange in the part of the band the coarse search has passed
_REDRAW_MS = 200     # 5 Hz
_SWEEP_MS = 300      # pattern panel: one fetch per tick
_MAX_FETCHES_PER_BLOCK = 15


def _fmt_int(n) -> str:
    """``12345`` -> ``"12 345"`` (thin groups of three, as in the status line)."""
    return f"{int(round(n)):,}".replace(",", " ")


def _fmt_time(seconds) -> str:
    """Seconds -> ``m:ss`` or ``h:mm:ss``; negative input is shown as 0."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class LiveView(QWidget):
    """Map being indexed (left), current pattern (right), status line (bottom). GUI thread only.

    Public state: ``image`` is the (rows, cols, 3) float32 RGB picture of the map in 0..1 (None without a base
    map); ``n_fetched`` counts the patterns fetched for the right-hand panel during runs (for tests and
    diagnostics); ``status`` is the QLabel of the status line.
    """
    point_clicked = pyqtSignal(int, int)  # (row, col) of a left click on the map while idle
    error = pyqtSignal(str)               # a problem while fetching or colouring (once per run), for the log

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(9, 4), facecolor="w")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(200, 120)
        self.status = QLabel("No map")
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)  # long text never widens the window
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.status)

        gs = self.figure.add_gridspec(1, 2, width_ratios=[2.2, 1])
        self.ax_map = self.figure.add_subplot(gs[0, 0])
        self.ax_pat = self.figure.add_subplot(gs[0, 1])
        for ax in (self.ax_map, self.ax_pat):
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax_pat.set_title("Pattern", fontsize=10)

        # map state
        self._base: np.ndarray | None = None      # (rows, cols) float32 grey in 0..1, the quasi-IQ map
        self.image: np.ndarray | None = None      # (rows, cols, 3) float32, what is displayed
        self._map_artist = None                   # AxesImage showing _disp
        self._disp: np.ndarray | None = None      # RGBA uint8 display copy of image[::stride, ::stride]
        self._stride = 1
        self._dirty_rows = None                   # (lo, hi) rows of image changed since the last push to _disp
        self._scratch: np.ndarray | None = None   # float rows buffer for the uint8 conversion
        self._outline = None                      # Rectangle around the region being indexed
        self._marker = None                       # red cross on the map point whose pattern is shown (idle only)
        # pattern state
        self._pattern_source = None               # fn(row, col) -> 2D array or None
        self._pat_artist = None
        self._pat_shape = None
        self.n_fetched = 0
        # run state
        self._running = False
        self._run_rect = None                     # (r0, c0, rows, cols) of the run; kept after end_run
        self._run_total = 0                       # patterns of the run, from begin_run
        self._t_start = None                      # time.monotonic() at begin_run
        self._block = None                        # (r0, r1, c0, c1) of the current band, clipped to the map
        self._block_n = 0
        self._frac_index = 0.0
        self._frac_refine = 0.0
        self._sweep_col = None                    # last column painted with the bright sweep tint (absolute)
        self._chunks_painted = 0                  # patterns painted by chunk_done in the current band
        self._queued = []                         # (rows, cols, packed, colour) of chunks not yet coloured and painted
        self._errors_reported = 0
        self._sweep_pos = 0                       # position of the pattern panel's cycle inside the band
        self._sweep_offset = 0
        self._speed_now = None                    # from set_speed, patterns/s
        self._speed_block = None                  # patterns/s of the last finished band, from block_done
        self._done = None                         # from set_progress
        self._total = None
        self._eta = None
        self._blocks_done_patterns = 0            # patterns of the bands finished so far, from block_done

        self._dirty = False                       # something changed since the canvas was last redrawn
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(_REDRAW_MS)
        self._redraw_timer.timeout.connect(self._on_redraw_tick)
        self._sweep_timer = QTimer(self)
        self._sweep_timer.setInterval(_SWEEP_MS)
        self._sweep_timer.timeout.connect(self._on_sweep_tick)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------------------------------
    # Base map
    # ------------------------------------------------------------------------------------------

    def set_base_map(self, iq, title: str = ""):
        """Show a new greyscale map, or clear the view.

        ``iq``: (rows, cols) array of any float dtype in 0..1 (NaN -> 0, +inf -> 1, clipped), or None to clear
        the map and the pattern panel. ``title`` goes above the map when given. Resets the map artists, the
        queued chunks and the sweep. The outline of an earlier run belongs to the map it was run on and is
        dropped; during a run (begun on a blank map when the quasi-IQ map was not loaded) the outline stays and
        the current band is clipped and tinted on the new map. Raises ValueError for an array that is not 2D.
        """
        self.ax_map.clear()
        self.ax_map.set_xticks([])
        self.ax_map.set_yticks([])
        self._map_artist = self._outline = self._marker = None
        self._sweep_col = None
        self._queued.clear()
        if not self._running:
            self._run_rect = None  # the outline of an earlier run belongs to the map it was run on
        if iq is None:
            self._base = None
            self.image = None
            self._disp = None
            self._dirty_rows = None
            self._clear_pattern()
            if title:
                self.ax_map.set_title(title, fontsize=10)
            self.status.setText("No map")
        else:
            base = np.nan_to_num(np.asarray(iq, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            if base.ndim != 2:
                raise ValueError(f"set_base_map: expected a 2D array, got shape {base.shape}")
            np.clip(base, 0.0, 1.0, out=base)
            self._base = base
            self.image = np.empty(base.shape + (3,), dtype=np.float32)
            self.image[...] = base[..., None]
            self._stride = 1
            self._rebuild_display()
            h, w = base.shape
            self._map_artist = self.ax_map.imshow(self._disp, interpolation="nearest", aspect="equal",
                                                  extent=(-0.5, w - 0.5, h - 0.5, -0.5))
            self._outline = Rectangle((0, 0), 1, 1, fill=False, edgecolor="#ff8c00", linewidth=1.2,
                                      visible=False)
            self.ax_map.add_patch(self._outline)
            # the same red cross as the pattern processor's scan map
            (self._marker,) = self.ax_map.plot([0], [0], "r+", markersize=15, markeredgewidth=2, visible=False)
            if title:
                self.ax_map.set_title(title, fontsize=10)
            if self._run_rect is not None:
                self._show_outline()
            if self._block is not None and self._running:
                self._block = self._clip_block(*self._block)
                self._paint_band_tint()
            if not self._running:
                self.status.setText(f"Map {base.shape[0]} × {base.shape[1]}")
        try:
            self.figure.tight_layout()
        except Exception:
            pass
        if self._update_stride():
            self._map_artist.set_data(self._disp)
        self._dirty = False
        self.canvas.draw_idle()

    # ------------------------------------------------------------------------------------------
    # Pattern
    # ------------------------------------------------------------------------------------------

    def set_pattern_source(self, fn):
        """``fn(row, col) -> 2D array or None``, polled on the GUI thread every ``_SWEEP_MS`` during a run to
        fill the pattern panel. None disables the panel's cycling. Exceptions of ``fn`` are reported once per run
        through ``error`` and never stop the run."""
        self._pattern_source = fn

    def show_pattern(self, pattern, row=None, col=None):
        """Display one pattern on the right, grey, with the colour limits at its 1st and 99th percentile.

        ``pattern``: 2D array of any dtype (None, a non-2D or empty array is ignored). ``row``/``col`` go into the
        panel's title. Does not move the marker (``set_marker``). Redrawn at the next redraw tick.
        """
        if pattern is None:
            return
        p = np.asarray(pattern)
        if p.ndim != 2 or p.size == 0:
            return
        pf = p.astype(np.float32, copy=False)
        finite = pf[np.isfinite(pf)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
        else:
            lo, hi = 0.0, 1.0
        if not hi > lo:
            hi = lo + 1.0
        if self._pat_artist is None or self._pat_shape != p.shape:
            if self._pat_artist is not None:
                self._pat_artist.remove()
            self._pat_artist = self.ax_pat.imshow(pf, cmap="gray", interpolation="nearest", aspect="equal",
                                                  vmin=lo, vmax=hi)
            self._pat_shape = p.shape
            self.ax_pat.set_xticks([])
            self.ax_pat.set_yticks([])
        else:
            self._pat_artist.set_data(pf)
            self._pat_artist.set_clim(lo, hi)
        self.ax_pat.set_title(f"({row}, {col})" if row is not None else "Pattern", fontsize=10)
        self._mark_dirty()

    def _clear_pattern(self):
        """Empty the pattern panel (the pattern of a file that is no longer loaded must not stay there)."""
        if self._pat_artist is not None:
            self._pat_artist.remove()
            self._pat_artist = None
        self._pat_shape = None
        self.ax_pat.clear()
        self.ax_pat.set_xticks([])
        self.ax_pat.set_yticks([])
        self.ax_pat.set_title("Pattern", fontsize=10)

    def set_marker(self, row, col):
        """Put the red cross on map point (row, col), or hide it when either is None. No-op without a base map.
        The workbench calls it after a click while idle; during a run the marker is hidden."""
        if self._marker is None:
            return
        if row is None or col is None:
            self._marker.set_visible(False)
        else:
            self._marker.set_data([col], [row])
            self._marker.set_visible(True)
        self._mark_dirty()

    # ------------------------------------------------------------------------------------------
    # Run events
    # ------------------------------------------------------------------------------------------

    def begin_run(self, r0, c0, rows, cols, n_patterns):
        """Start of a run (the workbench's "run_start" event): the region ``rows`` x ``cols`` with its top-left
        map point at (r0, c0) is outlined, the run state and the status numbers are reset, the marker hidden,
        and the redraw and sweep timers started. ``n_patterns``: patterns the run will index (the total of the
        status line until ``set_progress`` gives one)."""
        self._running = True
        self._run_rect = (int(r0), int(c0), int(rows), int(cols))
        self._run_total = int(n_patterns)
        self._t_start = time.monotonic()
        self._block = None
        self._block_n = 0
        self._frac_index = self._frac_refine = 0.0
        self._sweep_col = None
        self._speed_now = self._speed_block = None
        self._done, self._total, self._eta = 0, int(n_patterns), None
        self._blocks_done_patterns = 0
        self._queued.clear()
        self._errors_reported = 0
        self._show_outline()
        if self._marker is not None:
            self._marker.set_visible(False)
        self._update_status()
        self._mark_dirty()
        self._redraw_timer.start()
        self._sweep_timer.start()

    def block_started(self, r0, r1, c0, c1, n_patterns):
        """A band of rows ``r0:r1`` and columns ``c0:c1`` (absolute map coordinates, end exclusive) is being
        indexed (the "block_start" event): it is tinted orange, the sweep and the pattern panel's cycle start at
        its first column. A band still tinted from before (no ``block_done`` for it) is restored to grey first.
        ``n_patterns``: patterns of the band that are indexed (masked points excluded)."""
        if self._block is not None:
            self._untint_band()
        self._block = self._clip_block(r0, r1, c0, c1)
        self._block_n = int(n_patterns)
        self._frac_index = self._frac_refine = 0.0
        self._sweep_col = None
        self._chunks_painted = 0
        self._sweep_pos = 0
        self._sweep_offset = 0
        self._paint_band_tint()
        self._mark_dirty()

    def block_progress(self, frac_index, frac_refine):
        """Progress inside the current band (the "progress" event): ``frac_index`` of the coarse search and
        ``frac_refine`` of the refinement, 0..1 (clipped; non-numeric values are ignored). Advances the bright
        sweep across the band as long as no chunk of it has been painted, and the "block: ..." part of the
        status line."""
        try:
            fi = float(np.clip(frac_index, 0.0, 1.0))
            fr = float(np.clip(frac_refine, 0.0, 1.0))
        except (TypeError, ValueError):
            return
        self._frac_refine = fr
        if fi != self._frac_index:
            self._frac_index = fi
            if not self._chunks_painted:  # the painted chunks show the progress themselves
                self._paint_sweep()
        self._mark_dirty()

    def chunk_done(self, rows, cols, rgb=None, packed=None, colour=None):
        """The patterns at map coordinates (``rows[i]``, ``cols[i]``) of a finished chunk (the "chunk_done"
        event), painted as EMSphInx paints each pattern the moment it is indexed.

        ``rows``, ``cols``: int arrays of the same length (absolute map coordinates; points outside the map are
        ignored). Either ``rgb`` (float (n, 3) in 0..1) is painted now, or ``packed`` (the chunk's result rows,
        (n, >= 5)) is queued and ``colour(packed) -> rgb`` is called for all queued chunks of the same function
        at the next redraw tick, so colouring costs the GUI at most one call per tick and the indexing thread
        nothing. Nothing happens without a base map. The band's ``block_done`` repaints the whole band afterwards.
        """
        if self.image is None:
            return
        try:
            rows = np.asarray(rows, dtype=int).ravel()
            cols = np.asarray(cols, dtype=int).ravel()
        except (TypeError, ValueError):
            return
        if rows.size != cols.size or rows.size == 0:
            return
        if rgb is None:
            if packed is None or colour is None:
                return
            self._queued.append((rows, cols, np.asarray(packed), colour))
            self._mark_dirty()
            return
        self._paint_points(rows, cols, rgb)

    def _paint_points(self, rows, cols, rgb):
        """Write ``rgb`` (float-convertible (n, >= 3), NaN -> 0, clipped to 0..1) into ``image`` at the points
        (rows[i], cols[i]), dropping points outside the map; a shape mismatch paints nothing. The first chunk of a
        band takes the bright sweep back to the plain band tint (the sweep would otherwise stay frozen on the
        rows no chunk has reached yet) and counts the painted patterns in ``_chunks_painted``."""
        try:
            a = np.asarray(rgb, dtype=np.float32)
        except (TypeError, ValueError):
            return
        if a.ndim != 2 or a.shape[1] < 3 or a.shape[0] != rows.size:
            return
        h, w = self.image.shape[:2]
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        if not ok.all():
            rows, cols, a = rows[ok], cols[ok], a[ok]
            if rows.size == 0:
                return
        if not self._chunks_painted and self._block is not None and self._sweep_col is not None:
            r0, r1, c0, c1 = self._block
            if self._sweep_col > c0:
                self._tint(r0, r1, c0, self._sweep_col, _TINT)
            self._sweep_col = c0
        self.image[rows, cols] = np.clip(np.nan_to_num(a[:, :3], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        self._touch(int(rows.min()), int(rows.max()) + 1)
        self._chunks_painted += int(rows.size)
        self._mark_dirty()

    def _paint_queued(self):
        """Colour and paint the queued chunks, grouped by colouring function (one call each). A colouring error
        is reported once per run through ``error`` and the chunk is dropped: the preview must never stop the
        run."""
        if not self._queued or self.image is None:
            self._queued.clear()
            return
        queued, self._queued = self._queued, []
        groups: dict = {}
        for rows, cols, packed, colour in queued:
            groups.setdefault(id(colour), (colour, []))[1].append((rows, cols, packed))
        for colour, items in groups.values():
            rows = np.concatenate([r for r, _, _ in items])
            cols = np.concatenate([c for _, c, _ in items])
            try:
                rgb = colour(np.concatenate([p for _, _, p in items], axis=0))
            except Exception as e:  # noqa: BLE001
                self._report_error(f"Live preview: colouring failed ({type(e).__name__}: {e})")
                continue
            self._paint_points(rows, cols, rgb)

    def _report_error(self, text: str):
        """Emit ``error`` for the first problem of a run only; a failing source would otherwise flood the log
        several times a second."""
        if self._errors_reported == 0:
            self.error.emit(text)
        self._errors_reported += 1

    def block_done(self, r0, r1, c0, c1, rgb, n_patterns, seconds):
        """A band is finished (the "block_done" event): rows ``r0:r1``, columns ``c0:c1`` (absolute, end
        exclusive) are repainted from ``rgb``, the band's IPF image of shape (r1 - r0, c1 - c0, 3) with
        ``rgb[0, 0]`` at (r0, c0): float in 0..1, or integers (0..255 when any value exceeds 1); None leaves the
        band grey (a band masked out completely). The band is restored to grey first, so an ``rgb`` smaller than
        the band leaves no tint behind. Chunks still queued are dropped (the band is painted whole). ``n_patterns``
        and ``seconds`` give the band's speed for the status line and add to the patterns done."""
        band = self._clip_block(r0, r1, c0, c1)
        self._queued.clear()
        if self.image is not None and band is not None:
            br0, br1, bc0, bc1 = band
            self._tint(br0, br1, bc0, bc1, 0.0)
            if rgb is not None:
                self._paint_rgb(int(r0), int(c0), band, rgb)
        if self._block is not None and band is not None and self._block == band:
            self._block = None
            self._sweep_col = None
            self._chunks_painted = 0
        try:
            if n_patterns and seconds and seconds > 0:
                self._speed_block = float(n_patterns) / float(seconds)
        except (TypeError, ValueError):
            pass
        try:
            self._blocks_done_patterns += int(n_patterns or 0)
        except (TypeError, ValueError):
            pass
        self._mark_dirty()

    def set_speed(self, now_pps):
        """The current speed in patterns/s for the status line ("... pat/s now"), or None to leave it out. The
        workbench computes it over the last few seconds from the progress events."""
        self._speed_now = None if now_pps is None else float(now_pps)
        self._mark_dirty()

    def set_progress(self, done_patterns, total_patterns, eta_seconds):
        """Patterns done and total, and the ETA in seconds, for the status line (each may be None). The patterns
        done are shown as the larger of this and the sum of the finished bands."""
        self._done = None if done_patterns is None else int(done_patterns)
        self._total = None if total_patterns is None else int(total_patterns)
        self._eta = None if eta_seconds is None else float(eta_seconds)
        self._mark_dirty()

    def end_run(self, success: bool):
        """End of the run (the "run_end" event, or the workbench when the worker failed): timers stopped, the
        chunks still queued painted, a band still tinted restored to grey, the final status written ("Done in
        ..." with the average speed, or "Stopped after ..." with the patterns done when ``success`` is False),
        and the canvas redrawn at once. The outline of the run stays. Safe to call without a run."""
        self._sweep_timer.stop()
        self._redraw_timer.stop()
        self._paint_queued()
        if self._block is not None:
            self._untint_band()
            self._block = None
            self._mark_dirty()
        self._sweep_col = None
        was_running = self._running
        self._running = False
        elapsed = (time.monotonic() - self._t_start) if (self._t_start is not None and was_running) else 0.0
        done = self._done_patterns()
        if success:
            avg = done / elapsed if elapsed > 0 else 0.0
            text = f"Done in {_fmt_time(elapsed)} · {_fmt_int(done)} patterns · average {_fmt_int(avg)} pat/s"
        else:
            total = self._total or self._run_total
            tail = f" · {_fmt_int(done)} / {_fmt_int(total)} patterns" if total else ""
            text = f"Stopped after {_fmt_time(elapsed)}{tail}"
        self.status.setText(text)
        self._flush()

    # ------------------------------------------------------------------------------------------
    # Internals: painting
    # ------------------------------------------------------------------------------------------

    def _clip_block(self, r0, r1, c0, c1):
        """The band (r0, r1, c0, c1) as ints, clipped to the base map; None when nothing of it is left. Without
        a base map the band is only checked to be non-empty (it is clipped when the map arrives)."""
        r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
        if self._base is None:
            return (r0, r1, c0, c1) if (r1 > r0 and c1 > c0) else None
        nr, nc = self._base.shape
        r0, r1 = max(0, min(r0, nr)), max(0, min(r1, nr))
        c0, c1 = max(0, min(c0, nc)), max(0, min(c1, nc))
        if r1 <= r0 or c1 <= c0:
            return None
        return (r0, r1, c0, c1)

    def _tint(self, r0, r1, c0, c1, weight):
        """Repaint ``image[r0:r1, c0:c1]`` from the grey base map mixed with ``weight`` (0..1) of orange; 0 is
        plain grey. Marks the rows for the next display sync."""
        base = self._base[r0:r1, c0:c1, None]
        self.image[r0:r1, c0:c1] = base * (1 - weight) + _ORANGE * weight if weight else base
        self._touch(r0, r1)

    def _paint_band_tint(self):
        """Tint the current band with the plain band tint and start the sweep at its first column."""
        if self.image is None or self._block is None:
            return
        r0, r1, c0, c1 = self._block
        self._tint(r0, r1, c0, c1, _TINT)
        self._sweep_col = c0
        self._paint_sweep()

    def _paint_sweep(self):
        """Bright tint on the columns c0 .. c0 + frac_index * width of the current band, incrementally from the
        last painted column; a fraction that went back (should not happen) repaints the difference plain."""
        if self.image is None or self._block is None:
            return
        r0, r1, c0, c1 = self._block
        target = c0 + int(round(self._frac_index * (c1 - c0)))
        start = c0 if self._sweep_col is None else self._sweep_col
        if target > start:
            self._tint(r0, r1, start, target, _SWEEP_TINT)
        elif target < start:
            self._tint(r0, r1, target, start, _TINT)
        self._sweep_col = target

    def _untint_band(self):
        """Restore the whole current band to grey: its chunks painted so far go too, which is right for a band
        that never gets its ``block_done`` (a cancelled run)."""
        if self.image is None or self._block is None:
            return
        r0, r1, c0, c1 = self._block
        self._tint(r0, r1, c0, c1, 0.0)

    def _paint_rgb(self, r0_req, c0_req, band, rgb):
        """Write the band image ``rgb`` (see ``block_done``) into ``image`` at ``band`` = (r0, r1, c0, c1), the
        band clipped to the map; ``rgb[0, 0]`` sits at the requested (r0_req, c0_req), which may lie outside.
        Does not mark rows dirty (``block_done`` does)."""
        a = np.asarray(rgb)
        if a.ndim != 3 or a.shape[2] < 3:
            return
        a = a[..., :3]
        if np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.float32) / (255.0 if a.max(initial=0) > 1 else 1.0)
        else:
            a = np.nan_to_num(a.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        br0, br1, bc0, bc1 = band
        sr0, sc0 = br0 - r0_req, bc0 - c0_req
        sub = a[sr0:sr0 + (br1 - br0), sc0:sc0 + (bc1 - bc0)]
        h, w = sub.shape[:2]
        if h == 0 or w == 0:
            return
        self.image[br0:br0 + h, bc0:bc0 + w] = np.clip(sub, 0.0, 1.0)

    def _touch(self, r0, r1):
        """Widen the range of image rows to convert at the next display sync to include ``r0:r1``."""
        lo, hi = (r0, r1) if self._dirty_rows is None else (min(r0, self._dirty_rows[0]), max(r1, self._dirty_rows[1]))
        self._dirty_rows = (lo, hi)

    def _rebuild_display(self):
        """(Re)allocate the RGBA uint8 display array for the current stride and fill it from the whole image."""
        s = self._stride
        sub = self.image[::s, ::s]
        self._disp = np.empty(sub.shape[:2] + (4,), dtype=np.uint8)
        self._disp[..., 3] = 255
        self._dirty_rows = (0, self.image.shape[0])
        self._sync_display()

    def _sync_display(self):
        """Convert the changed rows of the float image into the display array (rounded to uint8)."""
        if self._dirty_rows is None or self.image is None or self._disp is None:
            return
        s = self._stride
        lo, hi = self._dirty_rows
        a, b = -(-lo // s), -(-hi // s)  # display rows whose source row (k * s) lies in [lo, hi)
        if b > a:
            buf = self._scratch_rows(b - a)
            np.multiply(self.image[a * s:b * s:s, ::s], 255.0, out=buf)
            buf += 0.5
            self._disp[a:b, :, :3] = buf  # float -> uint8 truncates: rounded
        self._dirty_rows = None

    def _scratch_rows(self, n):
        """A reused float32 buffer of ``n`` display rows for ``_sync_display``, reallocated when too small or
        when the display width changed."""
        buf = self._scratch
        if buf is None or buf.shape[1:] != (self._disp.shape[1], 3) or buf.shape[0] < n:
            buf = self._scratch = np.empty((max(n, 1), self._disp.shape[1], 3), dtype=np.float32)
        return buf[:n]

    def _update_stride(self) -> bool:
        """Stride from the map size vs the axes size in pixels; True if it changed (display rebuilt)."""
        if self.image is None:
            return False
        try:
            bb = self.ax_map.get_window_extent()
            h, w = self.image.shape[:2]
            s = max(1, int(min(h / max(bb.height, 1.0), w / max(bb.width, 1.0))))
        except Exception:
            s = 1
        if s == self._stride and self._disp is not None:
            return False
        self._stride = s
        self._rebuild_display()
        return True

    def _on_click(self, event):
        """A left click on the map while no run is active emits point_clicked(row, col)."""
        if self._running or self._map_artist is None or event.inaxes is not self.ax_map or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        h, w = self.image.shape[:2]
        row, col = int(round(event.ydata)), int(round(event.xdata))
        if 0 <= row < h and 0 <= col < w:
            self.point_clicked.emit(row, col)

    def _on_resize(self, _event=None):
        """Canvas resized: the stride may change, and then the whole display copy is rebuilt."""
        if self._map_artist is not None and self._update_stride():
            self._map_artist.set_data(self._disp)
            self.canvas.draw_idle()

    def _show_outline(self):
        """Place the outline rectangle around the run region (no-op without a map or a run)."""
        if self._outline is None or self._run_rect is None:
            return
        r0, c0, rows, cols = self._run_rect
        self._outline.set_bounds(c0 - 0.5, r0 - 0.5, cols, rows)
        self._outline.set_visible(True)

    # ------------------------------------------------------------------------------------------
    # Internals: timers
    # ------------------------------------------------------------------------------------------

    def _mark_dirty(self):
        """Note that the canvas needs a redraw; outside a run the redraw timer is started for one tick."""
        self._dirty = True
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def _flush(self):
        """Push the changed rows to the display copy and redraw now if anything changed."""
        if self._dirty:
            if self._map_artist is not None and self._dirty_rows is not None:
                self._sync_display()
                self._map_artist.set_data(self._disp)
            self.canvas.draw_idle()
            self._dirty = False

    def _on_redraw_tick(self):
        """Redraw timer: status line, queued chunks, canvas. The timer stops itself after the tick when no run
        is going on (a ``_mark_dirty`` while idle starts it for one tick)."""
        self._update_status()
        self._paint_queued()
        self._flush()
        if not self._running:
            self._redraw_timer.stop()

    def _on_sweep_tick(self):
        """Sweep timer: fetch and show the next pattern of the current band, row-major with a stride so that
        about ``_MAX_FETCHES_PER_BLOCK`` fetches cover the band; a further pass is shifted to show other points.
        A failing fetch or display is reported once per run through ``error``. No marker is moved during a run:
        the band tint says where the indexer is."""
        if self._pattern_source is None or self._block is None:
            return
        r0, r1, c0, c1 = self._block
        w = c1 - c0
        n = (r1 - r0) * w
        if n <= 0:
            return
        stride = max(1, math.ceil(n / _MAX_FETCHES_PER_BLOCK))
        if w > 1 and stride % w == 0:   # avoid visiting a single column
            stride += 1
        i = self._sweep_pos
        if i >= n:
            self._sweep_offset = (self._sweep_offset + max(1, stride // 2)) % stride
            i = self._sweep_offset
        self._sweep_pos = i + stride
        row, col = r0 + i // w, c0 + i % w
        try:
            pattern = self._pattern_source(row, col)
        except Exception as e:  # noqa: BLE001
            self._report_error(f"Live view: the pattern at ({row}, {col}) could not be fetched "
                               f"({type(e).__name__}: {e})")
            return
        if pattern is None:
            return
        self.n_fetched += 1
        try:
            self.show_pattern(pattern, row, col)
        except Exception as e:  # noqa: BLE001
            self._report_error(f"Live view: the pattern at ({row}, {col}) could not be shown ({type(e).__name__}: {e})")

    # ------------------------------------------------------------------------------------------
    # Internals: status
    # ------------------------------------------------------------------------------------------

    def _done_patterns(self) -> int:
        """Patterns done so far: the larger of the last ``set_progress`` and the sum of the finished bands."""
        return max(self._done or 0, self._blocks_done_patterns)

    def _update_status(self):
        """Rewrite the status line during a run: per cent, patterns, elapsed, ETA, the two speeds and the
        progress inside the band (when known). Outside a run the line is left as it is."""
        if not self._running:
            return
        elapsed = time.monotonic() - self._t_start if self._t_start is not None else 0.0
        total = self._total or self._run_total
        done = self._done_patterns()
        parts = []
        pct = f"{100.0 * done / total:.0f} %" if total else "…"
        parts.append(f"Indexing: {pct}")
        if total:
            parts.append(f"{_fmt_int(done)} / {_fmt_int(total)} patterns")
        parts.append(f"elapsed {_fmt_time(elapsed)}")
        if self._eta is not None and np.isfinite(self._eta):
            parts.append(f"ETA {_fmt_time(self._eta)}")
        if self._speed_now is not None and np.isfinite(self._speed_now):
            parts.append(f"{_fmt_int(self._speed_now)} pat/s now")
        if self._speed_block is not None:
            parts.append(f"{_fmt_int(self._speed_block)} pat/s this block")
        if self._block is not None and (self._frac_index > 0 or self._frac_refine > 0):
            # the fork refines inside the search tasks, so a separate refinement fraction only exists (and is
            # shown) while it is still below 100 %
            text_block = f"block: {100 * self._frac_index:.0f} %"
            if self._frac_refine < 1.0:
                text_block = f"block: search {100 * self._frac_index:.0f} %, refine {100 * self._frac_refine:.0f} %"
            parts.append(text_block)
        text = " · ".join(parts)
        if text != self.status.text():
            self.status.setText(text)
