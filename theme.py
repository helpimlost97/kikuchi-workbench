"""Dark (true black) and light appearance of the Kikuchi workbench.

``apply(app, mode)`` puts the whole application, the embedded pattern processor included, into ``"dark"`` or
``"light"``: the Fusion widget style with a matching palette, and matplotlib's colours (figure and axes
background, titles, ticks, labels) for the figures still to be created and for every figure already on screen.
Dark is true black (``#000000`` window, inputs, log panes and figure backgrounds, for OLED displays) with buttons
and splitter handles lifted just enough to be found, soft grey text, and check-box frames in that grey (a
per-class palette, see ``_indicator_palette``). ``style_title_bar(window, mode)`` makes
the Windows title bar follow (black caption, border and buttons; nothing elsewhere). ``current()`` is the mode
to start in: the one chosen last in the *Appearance* menu (kept in ``QSettings``), else ``DEFAULT``. Nothing
here changes the data shown: images keep their colour maps and grey scales.
"""

from __future__ import annotations

import sys

import matplotlib
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

MODES = ("dark", "light")
DEFAULT = "dark"

_ORG, _APP, _KEY = "kikuchi-workbench", "kikuchi-workbench", "appearance"

# background / text / disabled text of the two modes, shared by the palette, the figures and the title bar
_COLORS = {
    "dark": {"window": "#000000", "base": "#000000", "alt": "#0d0d0d", "button": "#1c1c1c", "handle": "#242424",
             "text": "#d0d0d0", "disabled": "#6a6a6a", "highlight": "#1f5fa8", "highlight_text": "#ffffff",
             "link": "#5c9ce6", "tooltip": "#101010"},
    "light": {"window": "#efefef", "base": "#ffffff", "alt": "#f5f5f5", "button": "#efefef", "handle": "#d0d0d0",
              "text": "#000000", "disabled": "#808080", "highlight": "#308cc6", "highlight_text": "#ffffff",
              "link": "#0000ff", "tooltip": "#ffffdc"},
}

#: Red for a warning in a table cell, readable on both backgrounds (``Qt.darkRed`` is not, on black).
WARNING_RED = "#e05252"

_current = None


def current() -> str:
    """The mode to start in: the last one chosen (``QSettings``), else ``DEFAULT``."""
    mode = QSettings(_ORG, _APP).value(_KEY, DEFAULT)
    return mode if mode in MODES else DEFAULT


def active() -> str | None:
    """The mode ``apply`` last set in this process, None before the first call."""
    return _current


def _palette(mode: str) -> QPalette:
    c = _COLORS[mode]
    p = QPalette()
    for role, key in ((QPalette.Window, "window"), (QPalette.Base, "base"), (QPalette.AlternateBase, "alt"),
                      (QPalette.Button, "button"), (QPalette.ToolTipBase, "tooltip")):
        p.setColor(role, QColor(c[key]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText, QPalette.ToolTipText):
        p.setColor(role, QColor(c["text"]))
        p.setColor(QPalette.Disabled, role, QColor(c["disabled"]))
    p.setColor(QPalette.PlaceholderText, QColor(c["disabled"]))
    p.setColor(QPalette.BrightText, QColor("#ff5555"))
    p.setColor(QPalette.Link, QColor(c["link"]))
    p.setColor(QPalette.Highlight, QColor(c["highlight"]))
    p.setColor(QPalette.HighlightedText, QColor(c["highlight_text"]))
    p.setColor(QPalette.Disabled, QPalette.Highlight, QColor(c["button"]))
    if mode == "dark":
        # Fusion draws frames and 3D edges from these; on black they vanish without a lift
        p.setColor(QPalette.Light, QColor("#2e2e2e"))
        p.setColor(QPalette.Midlight, QColor("#242424"))
        p.setColor(QPalette.Mid, QColor("#181818"))
        p.setColor(QPalette.Dark, QColor("#0a0a0a"))
        p.setColor(QPalette.Shadow, QColor("#000000"))
    return p


def _indicator_palette(mode: str) -> QPalette:
    """The palette of check boxes (and radio buttons) only, set per class with ``QApplication.setPalette(p,
    "QCheckBox")``. Fusion draws the frame of a check box from the *Window* colour (darkened by 140 then
    lightened by 110), which on true black is black: the box vanishes. A white Window for these classes alone
    gives a ``#c8c8c8`` frame, the soft grey of the text; nothing else of a check box uses Window (the box is
    filled from Base, the mark from Text, and the widget does not paint its own background), so the black stays."""
    p = _palette(mode)
    if mode == "dark":
        p.setColor(QPalette.Window, QColor("#ffffff"))
    return p


def _stylesheet(mode: str) -> str:
    """The few things the palette does not reach: splitter handles and the frame of inputs on black."""
    c = _COLORS[mode]
    if mode != "dark":
        return ""
    return (f"QSplitter::handle {{ background: {c['handle']}; }}"
            f"QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QAbstractItemView"
            f" {{ border: 1px solid {c['handle']}; }}"
            f"QHeaderView::section {{ background: {c['button']}; color: {c['text']}; border: 1px solid {c['handle']}; }}")


def _rc(mode: str) -> None:
    """matplotlib's defaults for figures created from now on."""
    c = _COLORS[mode]
    bg, fg = c["window"], c["text"]
    matplotlib.rcParams.update({
        "figure.facecolor": bg, "figure.edgecolor": bg, "axes.facecolor": bg, "axes.edgecolor": fg,
        "axes.labelcolor": fg, "axes.titlecolor": fg, "xtick.color": fg, "ytick.color": fg, "text.color": fg,
        "grid.color": c["disabled"], "legend.facecolor": c["base"], "legend.edgecolor": fg,
    })


def restyle_figure(fig, mode: str | None = None) -> None:
    """Recolour one existing matplotlib figure (background, axes, titles, ticks, labels, suptitle) for ``mode``
    (default: the active one). Images, lines and markers are left alone."""
    c = _COLORS[mode or _current or DEFAULT]
    bg, fg = c["window"], c["text"]
    fig.set_facecolor(bg)
    fig.set_edgecolor(bg)
    for text in fig.texts:
        text.set_color(fg)
    for ax in fig.axes:
        ax.set_facecolor(bg)
        ax.title.set_color(fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.tick_params(colors=fg, which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(c["base"])
            legend.get_frame().set_edgecolor(fg)
            for text in legend.get_texts():
                text.set_color(fg)


def style_title_bar(window, mode: str | None = None) -> None:
    """Make a top-level window's title bar follow ``mode`` on Windows: dark caption buttons (the
    ``DWMWA_USE_IMMERSIVE_DARK_MODE`` attribute, 20 on current builds, 19 on builds before 2004) and, on
    Windows 11, the caption, its text and the border in the theme's exact colours (``DWMWA_CAPTION_COLOR`` 35,
    ``DWMWA_TEXT_COLOR`` 36, ``DWMWA_BORDER_COLOR`` 34; ``0xFFFFFFFF`` restores the system default). Light mode
    restores the defaults. A no-op on other platforms, on Windows 10 for the colours, and when the call fails
    (an offscreen window, for instance). Call it after the window exists; ``winId()`` creates it if needed."""
    if sys.platform != "win32":
        return
    mode = mode or _current or DEFAULT
    try:
        import ctypes

        dwm = ctypes.windll.dwmapi
        hwnd = ctypes.c_void_p(int(window.winId()))
        dark = ctypes.c_int(1 if mode == "dark" else 0)
        for attribute in (20, 19):
            if dwm.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(dark), ctypes.sizeof(dark)) == 0:
                break
        c = _COLORS[mode]
        def colorref(hex_color):  # COLORREF is 0x00BBGGRR
            q = QColor(hex_color)
            return ctypes.c_uint((q.blue() << 16) | (q.green() << 8) | q.red())
        default = ctypes.c_uint(0xFFFFFFFF)
        for attribute, value in ((35, colorref(c["window"]) if mode == "dark" else default),
                                 (36, colorref(c["text"]) if mode == "dark" else default),
                                 (34, colorref(c["window"]) if mode == "dark" else default)):
            dwm.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value))
    except (OSError, AttributeError, ValueError):
        pass


def apply(app: QApplication, mode: str, remember: bool = False) -> None:
    """Switch the application to ``mode`` (``"dark"`` or ``"light"``): widget style, palette and stylesheet,
    matplotlib defaults, every figure already shown (redrawn) and the title bars of the top-level windows.
    ``remember`` stores the choice for the next start."""
    global _current
    if mode not in MODES:
        raise ValueError(f"appearance must be one of {MODES}, not {mode!r}")
    app.setStyle("Fusion")
    app.setPalette(_palette(mode))
    for cls in ("QCheckBox", "QRadioButton"):   # visible frames on black; class palettes survive setPalette
        app.setPalette(_indicator_palette(mode), cls)
    app.setStyleSheet(_stylesheet(mode))
    _rc(mode)
    _current = mode
    for widget in app.allWidgets():
        fig = getattr(widget, "figure", None)
        if fig is not None and hasattr(widget, "draw_idle") and hasattr(fig, "axes"):
            restyle_figure(fig, mode)
            widget.draw_idle()
    for window in app.topLevelWidgets():
        if window.isWindow() and window.testAttribute(Qt.WA_WState_Created):  # only windows that exist already
            style_title_bar(window, mode)
    if remember:
        QSettings(_ORG, _APP).setValue(_KEY, mode)
