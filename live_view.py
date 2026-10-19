"""Live view of a spherical indexing run: the map fills with IPF colours block by block, next to the current pattern.

``LiveView`` is a QWidget with one matplotlib figure (map on the left, pattern on the right) and a one-line status
label. The caller (the main window) forwards the indexing thread's events to it on the GUI thread:

    view.set_base_map(iq, "Quasi image quality")
    view.set_pattern_source(lambda r, c: patterns[r, c])
    view.begin_run(r0, c0, rows, cols, n_patterns)
    view.block_started(r0, r1, c0, c1, n)      # a band of rows is being indexed: tinted orange
    view.block_progress(frac_index, frac_refine)  # left-to-right sweep inside the band
    view.block_done(r0, r1, c0, c1, rgb, n, seconds)  # band painted with its IPF colours
    view.set_speed(pps); view.set_progress(done, total, eta)
    view.end_run(True)

Rendering is cheap on purpose: one RGB float32 image (``LiveView.image``) is kept and written into by slices, and the
canvas is redrawn from a 5 Hz timer only when something changed, however many events arrive. What is pushed to the
AxesImage is an RGBA uint8 copy of it, subsampled with a stride when the map has more pixels than the axes (a
1100 x 2100 map in a ~900 px wide axes is shown at stride 2, which is what nearest-neighbour display would show
anyway); only the rows that changed since the last redraw are converted. This keeps a redraw of a large map at a few
tens of ms instead of ~150 ms for matplotlib's float RGB path. The outline of the run rectangle and the
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
_SWEEP_MS = 300
_MAX_FETCHES_PER_BLOCK = 15


def _fmt_int(n) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


def _fmt_time(seconds) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class LiveView(QWidget):
    """Map being indexed (left), current pattern (right), status line (bottom). GUI thread only."""
    point_clicked = pyqtSignal(int, int)  # (row, col) of a left click on the map while idle


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
        self._base: np.ndarray | None = None      # (rows, cols) float32 grey
        self.image: np.ndarray | None = None      # (rows, cols, 3) float32, what is displayed
        self._map_artist = None
        self._disp: np.ndarray | None = None      # RGBA uint8 display copy of image[::stride, ::stride]
        self._stride = 1
        self._dirty_rows = None                   # (lo, hi) rows of image changed since the last push
        self._scratch: np.ndarray | None = None   # float rows buffer for the conversion
        self._outline = None
        self._marker = None
        self._map_title = ""
        # pattern state
        self._pattern_source = None
        self._pat_artist = None
        self._pat_shape = None
        self.n_fetched = 0                        # patterns fetched from the source (for tests / diagnostics)
        # run state
        self._running = False
        self._run_rect = None                     # (r0, c0, rows, cols)
        self._run_total = 0
        self._t_start = None
        self._t_end = None
        self._block = None                        # (r0, r1, c0, c1), clipped to the map when there is one
        self._block_n = 0
        self._frac_index = 0.0
        self._frac_refine = 0.0
        self._sweep_col = None                    # last column painted with the bright sweep tint (absolute)
        self._sweep_pos = 0
        self._sweep_offset = 0
        self._speed_now = None
        self._speed_block = None
        self._done = None                         # from set_progress
        self._total = None
        self._eta = None
        self._blocks_done_patterns = 0
        self._final_text = None

        self._dirty = False
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
        """Greyscale map in [0, 1] (rows, cols), or None to clear. Resets any overlay."""
        self._map_title = title or ""
        self.ax_map.clear()
        self.ax_map.set_xticks([])
        self.ax_map.set_yticks([])
        self._map_artist = self._outline = self._marker = None
        self._sweep_col = None
        if iq is None:
            self._base = None
            self.image = None
            self._disp = None
            self._dirty_rows = None
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
            (self._marker,) = self.ax_map.plot([0], [0], "o", ms=7, mfc="none", mec="#00e5ff", mew=1.6,
                                               visible=False)
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
        """fn(row, col) -> 2D array or None; polled during a run (about every 300 ms). None disables it."""
        self._pattern_source = fn

    def show_pattern(self, pattern, row=None, col=None):
        """Display one pattern on the right (clim from its 1st and 99th percentile)."""
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

    def set_marker(self, row, col):
        """Mark the map point whose pattern is displayed (no-op without a base map)."""
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
        self._running = True
        self._run_rect = (int(r0), int(c0), int(rows), int(cols))
        self._run_total = int(n_patterns)
        self._t_start = time.monotonic()
        self._t_end = None
        self._block = None
        self._block_n = 0
        self._frac_index = self._frac_refine = 0.0
        self._sweep_col = None
        self._speed_now = self._speed_block = None
        self._done, self._total, self._eta = 0, int(n_patterns), None
        self._blocks_done_patterns = 0
        self._final_text = None
        self._show_outline()
        if self._marker is not None:
            self._marker.set_visible(False)
        self._update_status()
        self._mark_dirty()
        self._redraw_timer.start()
        self._sweep_timer.start()

    def block_started(self, r0, r1, c0, c1, n_patterns):
        if self._block is not None:
            self._untint_band()
        self._block = self._clip_block(r0, r1, c0, c1)
        self._block_n = int(n_patterns)
        self._frac_index = self._frac_refine = 0.0
        self._sweep_col = None
        self._sweep_pos = 0
        self._sweep_offset = 0
        self._paint_band_tint()
        self._mark_dirty()

    def block_progress(self, frac_index, frac_refine):
        try:
            fi = float(np.clip(frac_index, 0.0, 1.0))
            fr = float(np.clip(frac_refine, 0.0, 1.0))
        except (TypeError, ValueError):
            return
        self._frac_refine = fr
        if fi != self._frac_index:
            self._frac_index = fi
            self._paint_sweep()
        self._mark_dirty()

    def block_done(self, r0, r1, c0, c1, rgb, n_patterns, seconds):
        band = self._clip_block(r0, r1, c0, c1)
        if self.image is not None and band is not None:
            br0, br1, bc0, bc1 = band
            # restore grey first so an rgb that does not cover the band leaves no tint behind
            self.image[br0:br1, bc0:bc1] = self._base[br0:br1, bc0:bc1, None]
            if rgb is not None:
                self._paint_rgb(int(r0), int(c0), band, rgb)
            self._touch(br0, br1)
        if self._block is not None and band is not None and self._block == band:
            self._block = None
            self._sweep_col = None
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
        self._speed_now = None if now_pps is None else float(now_pps)
        self._mark_dirty()

    def set_progress(self, done_patterns, total_patterns, eta_seconds):
        self._done = None if done_patterns is None else int(done_patterns)
        self._total = None if total_patterns is None else int(total_patterns)
        self._eta = None if eta_seconds is None else float(eta_seconds)
        self._mark_dirty()

    def end_run(self, success: bool):
        self._sweep_timer.stop()
        self._redraw_timer.stop()
        if self._block is not None:
            self._untint_band()
            self._block = None
        self._sweep_col = None
        was_running = self._running
        self._running = False
        self._t_end = time.monotonic()
        elapsed = (self._t_end - self._t_start) if (self._t_start is not None and was_running) else 0.0
        done = self._done_patterns()
        if success:
            avg = done / elapsed if elapsed > 0 else 0.0
            self._final_text = (f"Done in {_fmt_time(elapsed)} · {_fmt_int(done)} patterns · "
                                f"average {_fmt_int(avg)} pat/s")
        else:
            total = self._total or self._run_total
            tail = f" · {_fmt_int(done)} / {_fmt_int(total)} patterns" if total else ""
            self._final_text = f"Stopped after {_fmt_time(elapsed)}{tail}"
        self.status.setText(self._final_text)
        self._flush()

    # ------------------------------------------------------------------------------------------
    # Internals: painting
    # ------------------------------------------------------------------------------------------

    def _clip_block(self, r0, r1, c0, c1):
        r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
        if self._base is None:
            return (r0, r1, c0, c1) if (r1 > r0 and c1 > c0) else None
        nr, nc = self._base.shape
        r0, r1 = max(0, min(r0, nr)), max(0, min(r1, nr))
        c0, c1 = max(0, min(c0, nc)), max(0, min(c1, nc))
        if r1 <= r0 or c1 <= c0:
            return None
        return (r0, r1, c0, c1)

    def _paint_band_tint(self):
        if self.image is None or self._block is None:
            return
        r0, r1, c0, c1 = self._block
        self.image[r0:r1, c0:c1] = self._base[r0:r1, c0:c1, None] * (1 - _TINT) + _ORANGE * _TINT
        self._touch(r0, r1)
        self._sweep_col = c0
        self._paint_sweep()

    def _paint_sweep(self):
        """Bright tint on the columns c0 .. c0 + frac_index * width of the current band (incremental)."""
        if self.image is None or self._block is None:
            return
        r0, r1, c0, c1 = self._block
        target = c0 + int(round(self._frac_index * (c1 - c0)))
        start = c0 if self._sweep_col is None else self._sweep_col
        if target > start:
            self.image[r0:r1, start:target] = (self._base[r0:r1, start:target, None] * (1 - _SWEEP_TINT)
                                               + _ORANGE * _SWEEP_TINT)
        elif target < start:  # the fraction went back (should not happen): repaint with the plain tint
            self.image[r0:r1, target:start] = (self._base[r0:r1, target:start, None] * (1 - _TINT)
                                               + _ORANGE * _TINT)
        if target != start:
            self._touch(r0, r1)
        self._sweep_col = target

    def _untint_band(self):
        if self.image is None or self._block is None:
            return
        r0, r1, c0, c1 = self._block
        # only pixels still showing the tint: done blocks were painted by block_done already
        self.image[r0:r1, c0:c1] = self._base[r0:r1, c0:c1, None]
        self._touch(r0, r1)

    def _paint_rgb(self, r0_req, c0_req, band, rgb):
        a = np.asarray(rgb)
        if a.ndim != 3 or a.shape[2] < 3:
            return
        a = a[..., :3]
        if np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.float32) / (255.0 if a.max(initial=0) > 1 else 1.0)
        else:
            a = np.nan_to_num(a.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        br0, br1, bc0, bc1 = band
        # rgb[0, 0] is at (r0_req, c0_req); the band may have been clipped to the map
        sr0, sc0 = br0 - r0_req, bc0 - c0_req
        sub = a[sr0:sr0 + (br1 - br0), sc0:sc0 + (bc1 - bc0)]
        h, w = sub.shape[:2]
        if h == 0 or w == 0:
            return
        self.image[br0:br0 + h, bc0:bc0 + w] = np.clip(sub, 0.0, 1.0)

    def _touch(self, r0, r1):
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
        """Convert the changed rows of the float image into the display array."""
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
        if self._map_artist is not None and self._update_stride():
            self._map_artist.set_data(self._disp)
            self.canvas.draw_idle()

    def _show_outline(self):
        if self._outline is None or self._run_rect is None:
            return
        r0, c0, rows, cols = self._run_rect
        self._outline.set_bounds(c0 - 0.5, r0 - 0.5, cols, rows)
        self._outline.set_visible(True)

    # ------------------------------------------------------------------------------------------
    # Internals: timers
    # ------------------------------------------------------------------------------------------

    def _mark_dirty(self):
        self._dirty = True
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def _flush(self):
        """Push the image and redraw now if anything changed (also used at the end of a run)."""
        if self._dirty:
            if self._map_artist is not None and self._dirty_rows is not None:
                self._sync_display()
                self._map_artist.set_data(self._disp)
            self.canvas.draw_idle()
            self._dirty = False

    def _on_redraw_tick(self):
        if self._running:
            self._update_status()
        self._flush()
        if not self._running:
            self._redraw_timer.stop()

    def _on_sweep_tick(self):
        """Fetch the next pattern of the current block (row-major, strided: ~15 fetches cover a block)."""
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
        if i >= n:  # block covered: start again, shifted so the next pass shows other points
            self._sweep_offset = (self._sweep_offset + max(1, stride // 2)) % stride
            i = self._sweep_offset
        self._sweep_pos = i + stride
        row, col = r0 + i // w, c0 + i % w
        try:
            pattern = self._pattern_source(row, col)
        except Exception:
            return
        if pattern is None:
            return
        self.n_fetched += 1
        try:
            self.show_pattern(pattern, row, col)
        except Exception:
            return
        self.set_marker(row, col)

    # ------------------------------------------------------------------------------------------
    # Internals: status
    # ------------------------------------------------------------------------------------------

    def _done_patterns(self) -> int:
        return max(self._done or 0, self._blocks_done_patterns)

    def _update_status(self):
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
            parts.append(f"block: search {100 * self._frac_index:.0f} %, refine {100 * self._frac_refine:.0f} %")
        text = " · ".join(parts)
        if text != self.status.text():
            self.status.setText(text)
