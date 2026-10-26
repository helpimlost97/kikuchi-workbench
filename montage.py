"""EDAX APEX montage ("stitch") projects: the .edaxh5 project file and its per-tile .up1/.up2 pattern files.

An APEX project (.edaxh5) holds every map ("tile") of a montage: its grid, step, pattern center, phases, stage
position and the Hough-indexed orientation table (``EBSD/ANG/DATA/DATA``), but no patterns. The patterns of a
tile are in a .up1/.up2 file whose version-4 header carries the tile's map GUID (the map group's ``Id``
attribute), which is how the two are linked; the project never names its pattern files. STITCH_DATASET_NOTES.md
describes both formats.

This module is the file layer of the workbench's montage batch tab (montage_tab.py) and has no Qt in it. It
reads a project (:func:`read_project`), finds and checks the pattern file of every tile
(:func:`read_pattern_header`, :func:`find_pattern_files`, :func:`match_tiles`, :func:`pattern_folder_tiles`),
builds the ``scan`` dict sph_core needs to load a tile (:func:`scan_dict`), computes a tile's quasi-IQ and
all-zero mask (:func:`tile_stats`), maps master patterns to the project's phases (:func:`phase_mapping`) and
writes the spherical indexing result of a tile into a copy of the project (:func:`copy_project`,
:func:`write_tile_result`, :func:`indexed_tiles` for Resume), so that OIM Analysis or MTEX can stitch the montage
as they would the Hough results. Nothing here is stitched: every tile keeps its own grid.

HDF5 layout relied on (paths relative to a map group, which is any group with the attribute ``GroupType`` 3):
``Sample/Number Of Rows|Columns``, ``Sample/Step X|Y`` (um), ``Sample/Grid Type``, ``Sample/Sample Tilt`` (deg),
``Camera/Elevation Angle|Azimuthal Angle`` (deg), ``Column/SemKV|SemWD`` (kV, mm), ``Stage/X|Y|Z`` (mm),
``EBSD/ANG/HEADER/Pattern Center Calibration/X-Star|Y-Star|Z-Star`` (EDAX convention),
``EBSD/ANG/HEADER/Phase/<n>/...`` (1-based phase numbers), ``EBSD/ANG/HEADER/Comments`` and the compound table
``EBSD/ANG/DATA/DATA`` with one row per grid point in row-major order (fields ``Orientations`` (9 f8), ``IQ``,
``CI``, ``Phase``, ``Fit``, ``Valid``, ...). APEX stores every scalar as a length-1 array.

The orientation table's ``Orientations`` field is the 3x3 rotation matrix (row-major) of the orientation in the
same convention as a kikuchipy/orix rotation's ``to_matrix()``, checked against the Hough results of a real tile
(median misorientation 0.9 deg with a nickel master on austenite; the transpose gives 40 deg).

Command line: ``python montage.py project.edaxh5 [--patterns FOLDER] [--suffix SUFFIX]`` prints the tile table.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np

import sph_core

#: How the montage batch names its output: processed tiles get this suffix before the extension
#: (``<tile>_proc.up1``), and they and the output project go into this folder next to the project unless the
#: GUI's output folder says otherwise. The montage tab and the command line below read these.
DEFAULT_SUFFIX = "_proc"
DEFAULT_OUTPUT_SUBDIR = "processed"

# values of the ``GroupType`` attribute that sets the level of a group in an APEX project
GROUP_PROJECT, GROUP_SPECIMEN, GROUP_AREA, GROUP_MAP = 0, 1, 2, 3
# the tile's position in the montage, at the end of its map name ("OIM Map 1_r000c003")
_TILE_RC = re.compile(r"_r(\d+)c(\d+)$")
PATTERN_EXTENSIONS = (".up1", ".up2")
Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


def _first(value, default=None):
    """The first element of an APEX value (scalars are stored as length-1 arrays) as a Python object.

    Bytes are decoded as UTF-8, numpy scalars become Python scalars; ``default`` for an empty value.
    """
    value = np.asarray(value).ravel()
    if value.size == 0:
        return default
    value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar(group: h5py.Group, path: str, default=None):
    """The dataset ``path`` of ``group`` as a Python scalar, or ``default`` when it does not exist."""
    try:
        value = group[path][()]
    except KeyError:
        return default
    return _first(value, default)


def _attr(obj, name: str, default=None):
    """The attribute ``name`` of a group or dataset as a Python scalar, or ``default`` when it does not exist."""
    if name not in obj.attrs:
        return default
    return _first(obj.attrs[name], default)


# ----------------------------------------------------------------------------------------------
# Pattern files
# ----------------------------------------------------------------------------------------------


@dataclass
class PatternHeader:
    """The header of an EDAX .up1/.up2 file, as :func:`read_pattern_header` reads it.

    Little-endian layout: bytes 0:16 are version, pattern width, pattern height and data offset (uint32 each).
    Version 3 and later add n_cols (17:21), n_rows (21:25), a hexagonal-grid flag (25) and the step in um
    (26:42, two float64); version 4 adds the map GUID in .NET byte order (42:58). Version 1 has no grid, so
    ``n_cols``, ``n_rows``, ``step_x``, ``step_y`` are None, as is ``guid`` before version 4. The patterns follow
    at ``offset``, row-major, ``bytes_per_pixel`` 1 (.up1) or 2 (.up2).
    """

    path: str
    version: int
    width: int
    height: int
    offset: int
    n_cols: int | None
    n_rows: int | None
    hex_grid: bool
    step_x: float | None
    step_y: float | None
    guid: str | None
    bytes_per_pixel: int
    file_size: int

    @property
    def n_patterns_in_file(self) -> int:
        """How many whole patterns the file holds after the header (from its size, not the grid)."""
        return max(0, (self.file_size - self.offset) // (self.width * self.height * self.bytes_per_pixel))

    @property
    def n_patterns_expected(self) -> int | None:
        """n_cols x n_rows of the header, None without a grid."""
        return None if self.n_cols is None or self.n_rows is None else self.n_cols * self.n_rows

    @property
    def size_ok(self) -> bool:
        """True when the file holds exactly the patterns of its grid (no truncation, no extra patterns)."""
        expected = self.n_patterns_expected
        if expected is None:
            return False
        return self.file_size == self.offset + expected * self.width * self.height * self.bytes_per_pixel

    @property
    def name(self) -> str:
        """The file name without its folder."""
        return os.path.basename(self.path)


def read_pattern_header(path: str | os.PathLike) -> PatternHeader:
    """Read the header of a .up1/.up2 file (see :class:`PatternHeader` for the layout).

    The grid fields are read from version 3 on, the map GUID from version 4 on (and only when the data offset
    leaves room for it). Raises ValueError for a file shorter than the 16 fixed header bytes; OSError when the
    file cannot be read.
    """
    path = str(path)
    bpp = 2 if path.lower().endswith(".up2") else 1
    with open(path, "rb") as f:
        head = f.read(64)
    if len(head) < 16:
        raise ValueError(f"{path}: not an EDAX pattern file (header too short)")
    version, width, height, offset = struct.unpack_from("<IIII", head, 0)
    n_cols = n_rows = step_x = step_y = guid = None
    hex_grid = False
    if version >= 3 and len(head) >= 42:
        n_cols, n_rows = struct.unpack_from("<II", head, 17)
        hex_grid = bool(head[25])
        step_x, step_y = struct.unpack_from("<dd", head, 26)
    if version >= 4 and offset >= 58:
        guid = str(uuid.UUID(bytes_le=head[42:58]))
    return PatternHeader(path, int(version), int(width), int(height), int(offset), n_cols, n_rows, hex_grid,
                         step_x, step_y, guid, bpp, os.path.getsize(path))


def find_pattern_files(folder: str | os.PathLike, recursive: bool = False) -> list[PatternHeader]:
    """The headers of every .up1/.up2 file in ``folder``, in sorted path order (unreadable files are skipped)."""
    folder = Path(folder)
    paths = folder.rglob("*") if recursive else folder.iterdir()
    headers = []
    for p in sorted(paths):
        if p.is_file() and p.suffix.lower() in PATTERN_EXTENSIONS:
            try:
                headers.append(read_pattern_header(p))
            except (OSError, ValueError):
                continue
    return headers


# ----------------------------------------------------------------------------------------------
# The project
# ----------------------------------------------------------------------------------------------


@dataclass
class Phase:
    """One entry of a map's phase list (``EBSD/ANG/HEADER/Phase/<n>``); ``lattice`` is (a, b, c, alpha, beta,
    gamma) in Angstrom and degrees, ``symmetry`` the TSL symmetry number."""

    name: str
    formula: str
    lattice: tuple
    laue_group: str
    point_group: str
    symmetry: int | None


@dataclass
class Tile:
    """One map of the project and, once matched, its pattern file.

    ``h5_path`` is the map group's path in the file without the leading slash; ``row``/``col`` the montage
    position parsed from the name (None when the name has none); ``n_rows``/``n_cols`` the scan grid; the steps
    are in um, the stage position in mm, the tilts in degrees, ``pc`` is (x*, y*, z*) in the EDAX convention.
    ``n_table_rows`` is the length of the orientation table. ``pattern`` and ``status`` are set by
    :func:`match_tiles`: ``status`` is "ok", "no pattern file" or what is wrong with the matched file.
    """

    name: str
    h5_path: str
    guid: str | None
    row: int | None
    col: int | None
    n_rows: int
    n_cols: int
    step_x: float | None
    step_y: float | None
    grid_type: str | None
    sample_tilt: float | None
    camera_elevation: float | None
    camera_azimuth: float | None
    kv: float | None
    working_distance: float | None
    pc: tuple | None
    stage: tuple
    phases: list[Phase] = field(default_factory=list)
    n_table_rows: int | None = None
    pattern: PatternHeader | None = None
    status: str = "no pattern file"

    @property
    def ok(self) -> bool:
        """True when the tile has a pattern file that passed every check of :func:`match_tiles`."""
        return self.pattern is not None and self.status == "ok"


@dataclass
class Project:
    """An APEX project: its file, its name (the project group's ``Name`` attribute) and its tiles in montage order.

    ``skipped`` lists the map groups that are not EBSD scans (no grid, e.g. EDS-only maps) and are not tiles.
    """

    path: str
    name: str
    tiles: list[Tile]
    skipped: list[str] = field(default_factory=list)

    @property
    def folder(self) -> str:
        """The absolute folder of the project file (the default place of its pattern files)."""
        return os.path.dirname(os.path.abspath(self.path))

    @property
    def stem(self) -> str:
        """The project file name without folder and extension."""
        return os.path.splitext(os.path.basename(self.path))[0]


def _read_phases(m: h5py.Group) -> list[Phase]:
    """The phase list of a map group, in the order of its phase numbers (empty when the map has none)."""
    phases = []
    if "EBSD/ANG/HEADER/Phase" not in m:
        return phases
    g = m["EBSD/ANG/HEADER/Phase"]
    # the keys are the phase numbers as strings: sort "10" after "9"
    for key in sorted(g, key=lambda k: (len(k), k)):
        p = g[key]
        if not isinstance(p, h5py.Group):
            continue
        lattice = tuple(_scalar(p, f"Lattice Constant {k}") for k in ("A", "B", "C", "Alpha", "Beta", "Gamma"))
        phases.append(Phase(str(_scalar(p, "Material Name", key)), str(_scalar(p, "Formula", "")), lattice,
                            str(_scalar(p, "Laue Group", "")), str(_scalar(p, "Point Group", "")),
                            _scalar(p, "Symmetry")))
    return phases


def _read_tile(m: h5py.Group) -> Tile:
    """A :class:`Tile` from a map group. Raises ValueError when the group has no scan grid."""
    name = str(_attr(m, "Name", m.name.rsplit("/", 1)[-1]))
    rc = _TILE_RC.search(name)
    row, col = (int(rc.group(1)), int(rc.group(2))) if rc else (None, None)
    n_rows, n_cols = _scalar(m, "Sample/Number Of Rows"), _scalar(m, "Sample/Number Of Columns")
    if n_rows is None or n_cols is None:
        raise ValueError(f"{m.name}: no Sample/Number Of Rows|Columns")
    pc_group = "EBSD/ANG/HEADER/Pattern Center Calibration"
    pc = tuple(_scalar(m, f"{pc_group}/{k}") for k in ("X-Star", "Y-Star", "Z-Star"))
    pc = None if any(v is None for v in pc) else tuple(float(v) for v in pc)
    stage = tuple(_scalar(m, f"Stage/{k}") for k in ("X", "Y", "Z"))
    stage = tuple(None if v is None else float(v) for v in stage)
    n_table = m["EBSD/ANG/DATA/DATA"].shape[0] if "EBSD/ANG/DATA/DATA" in m else None
    return Tile(
        name=name, h5_path=m.name.lstrip("/"), guid=_attr(m, "Id"), row=row, col=col,
        n_rows=int(n_rows), n_cols=int(n_cols), step_x=_scalar(m, "Sample/Step X"), step_y=_scalar(m, "Sample/Step Y"),
        grid_type=_scalar(m, "Sample/Grid Type"), sample_tilt=_scalar(m, "Sample/Sample Tilt"),
        camera_elevation=_scalar(m, "Camera/Elevation Angle"), camera_azimuth=_scalar(m, "Camera/Azimuthal Angle"),
        kv=_scalar(m, "Column/SemKV"), working_distance=_scalar(m, "Column/SemWD"), pc=pc, stage=stage,
        phases=_read_phases(m), n_table_rows=n_table,
    )


def read_project(path: str | os.PathLike) -> Project:
    """Read an APEX .edaxh5 project: every map group (``GroupType`` 3) anywhere in the file becomes a tile.

    The file is walked, not a fixed ``Project/Specimen/Area`` path, because a project can hold several specimens
    and areas. Tiles are sorted by (row, col), tiles without a position last, by their path. Map groups without a
    scan grid go to ``Project.skipped`` with a note. The project name is the ``Name`` attribute of the top-level
    group with ``GroupType`` 0, else the file name. Raises OSError when the file cannot be opened.
    """
    path = str(path)
    tiles = []
    with h5py.File(path, "r") as f:
        name = os.path.splitext(os.path.basename(path))[0]
        for key in f:
            if isinstance(f[key], h5py.Group) and _attr(f[key], "GroupType") == GROUP_PROJECT:
                name = str(_attr(f[key], "Name", name))
                break

        skipped = []

        def visit(_name, obj):
            if isinstance(obj, h5py.Group) and _attr(obj, "GroupType") == GROUP_MAP:
                if "Sample/Number Of Rows" in obj and "Sample/Number Of Columns" in obj:
                    tiles.append(_read_tile(obj))
                else:
                    skipped.append(f"{str(_attr(obj, 'Name', obj.name))} ({obj.name}): "
                                   "no Sample/Number Of Rows|Columns")

        f.visititems(visit)
    tiles.sort(key=lambda t: (t.row if t.row is not None else 1 << 30, t.col if t.col is not None else 1 << 30,
                              t.h5_path))
    return Project(path, name, tiles, skipped)


def match_tiles(project: Project, headers: list[PatternHeader], prefer_suffix: str | None = None,
                log: Log | None = None) -> None:
    """Assign a pattern file to every tile of the project, in place (``Tile.pattern`` and ``Tile.status``).

    A file matches a tile by the GUID in its version-4 header; a file without a GUID matches by name (its stem
    ends with the tile's map name, or with the map name plus ``prefer_suffix``). When several files match one
    tile (e.g. the raw file and a processed copy, which keeps the header), the one whose stem ends with
    ``prefer_suffix`` wins, else the one with the shortest name (the raw file). The grid of the file must equal
    the tile's (``status`` says what is wrong otherwise), and the file must have exactly the patterns of its grid.
    Every tile is reset first, so a tile without a match ends with ``status`` "no pattern file". ``log`` gets a
    line for every file that could not be assigned and for every tile with more than one candidate.
    """
    by_guid: dict[str, list[PatternHeader]] = {}
    without_guid: list[PatternHeader] = []
    for h in headers:
        if h.guid:
            by_guid.setdefault(h.guid.lower(), []).append(h)
        else:
            without_guid.append(h)

    def name_matches(h: PatternHeader, tile: Tile) -> bool:
        stem = os.path.splitext(h.name)[0]
        return stem.endswith(tile.name) or bool(prefer_suffix and stem.endswith(tile.name + prefer_suffix))

    def rank(h: PatternHeader):
        stem = os.path.splitext(h.name)[0]
        preferred = bool(prefer_suffix) and stem.endswith(prefer_suffix)
        return (not preferred, len(h.name), h.name)

    # a file without a GUID is matched by name only when exactly one tile has that name (a project with several
    # areas can hold the same map name twice; such files cannot be told apart and are left unassigned)
    ambiguous = set()
    for h in without_guid:
        n_tiles = sum(1 for t in project.tiles if name_matches(h, t))
        if n_tiles > 1:
            ambiguous.add(h.path)
            if log is not None:
                log(f"{h.name}: no GUID in its header and {n_tiles} tiles of the project match its name; not assigned")
    for tile in project.tiles:
        candidates = list(by_guid.get((tile.guid or "").lower(), []))
        for h in without_guid:
            if h.path not in ambiguous and name_matches(h, tile):
                candidates.append(h)
        tile.pattern, tile.status = None, "no pattern file"
        if not candidates:
            continue
        candidates.sort(key=rank)
        h = candidates[0]
        tile.pattern = h
        problems = []
        if h.n_cols is None:
            problems.append(f"header version {h.version} has no grid")
        else:
            if (h.n_rows, h.n_cols) != (tile.n_rows, tile.n_cols):
                problems.append(f"file grid {h.n_cols} x {h.n_rows} differs from the map's "
                                f"{tile.n_cols} x {tile.n_rows}")
            step_differs = (tile.step_x is not None and h.step_x is not None
                            and abs(h.step_x - tile.step_x) > 1e-6 * max(1.0, abs(tile.step_x)))
            if step_differs:
                problems.append(f"file step {h.step_x} differs from the map's {tile.step_x}")
            if not h.size_ok:
                problems.append(f"file has {h.n_patterns_in_file} patterns, the grid needs {h.n_patterns_expected}")
        if h.hex_grid:
            problems.append("hexagonal grid")
        tile.status = "ok" if not problems else "; ".join(problems)
        if log is not None and len(candidates) > 1:
            log(f"{tile.name}: {len(candidates)} pattern files carry its GUID, using {h.name}")


def pattern_folder_tiles(project: Project, folder: str | os.PathLike, prefer_suffix: str | None = None,
                         log: Log | None = None) -> list[Tile]:
    """Match the tiles against the pattern files of ``folder`` (see :func:`match_tiles`); returns the project's
    tiles. Raises OSError when the folder cannot be listed."""
    match_tiles(project, find_pattern_files(folder), prefer_suffix=prefer_suffix, log=log)
    return project.tiles


def geometry_warnings(project: Project) -> list[str]:
    """Differences between the tiles in what one set of indexing parameters assumes to be shared: pattern center,
    sample tilt, camera elevation, step, phase names, kV and the pattern width of the matched files. One warning
    line per differing quantity (the first three differing tiles are named), empty when the tiles agree."""
    tiles = project.tiles
    if not tiles:
        return ["the project has no maps"]
    out = []
    first = tiles[0]
    for label, get in (("pattern center", lambda t: t.pc), ("sample tilt", lambda t: t.sample_tilt),
                       ("camera elevation", lambda t: t.camera_elevation), ("step", lambda t: (t.step_x, t.step_y)),
                       ("phases", lambda t: tuple(p.name for p in t.phases)), ("kV", lambda t: t.kv)):
        values = {t.name: get(t) for t in tiles}
        differing = [n for n, v in values.items() if v != get(first)]
        if differing:
            out.append(f"{label} differs between tiles: {first.name} has {get(first)}, " +
                       ", ".join(f"{n} has {values[n]}" for n in differing[:3]) +
                       (" ..." if len(differing) > 3 else ""))
    widths = {t.pattern.width for t in tiles if t.pattern is not None}
    if len(widths) > 1:
        out.append(f"pattern width differs between the pattern files: {sorted(widths)}")
    return out


def tile_table(project: Project) -> str:
    """The tiles as an aligned text table: name, row, col, grid, pattern file name ("-" if none) and status."""
    rows = [("tile", "row", "col", "grid (cols x rows)", "pattern file", "status")]
    for t in project.tiles:
        rows.append((t.name, "" if t.row is None else str(t.row), "" if t.col is None else str(t.col),
                     f"{t.n_cols} x {t.n_rows}", t.pattern.name if t.pattern else "-", t.status))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def scan_dict(tile: Tile) -> dict:
    """The ``scan`` dict ``sph_core.load_patterns`` takes (the keys of ``ebsd_params.scan_info`` it reads), from
    the tile's map and pattern header, so that no .ang is needed. The pattern size is None without a pattern
    file."""
    h = tile.pattern
    return {"n_rows": tile.n_rows, "n_cols": tile.n_cols, "step_x": tile.step_x, "step_y": tile.step_y,
            "grid": tile.grid_type, "pc": tile.pc, "sample_tilt": tile.sample_tilt,
            "camera_elevation": tile.camera_elevation,
            "up_has_map_size": h is not None and h.n_cols is not None, "pattern_width": None if h is None else h.width,
            "pattern_height": None if h is None else h.height, "phases": [p.name for p in tile.phases]}


def output_name(tile: Tile, suffix: str, folder: str | os.PathLike | None = None) -> str:
    """``<pattern file stem><suffix><extension>`` in ``folder`` (default: the pattern file's folder), the name of
    the tile's processed file. Raises ValueError when the tile has no pattern file."""
    if tile.pattern is None:
        raise ValueError(f"{tile.name} has no pattern file")
    src = Path(tile.pattern.path)
    out_dir = Path(folder) if folder else src.parent
    return str(out_dir / f"{src.stem}{suffix}{src.suffix}")


# ----------------------------------------------------------------------------------------------
# Results into a copy of the project
# ----------------------------------------------------------------------------------------------


def copy_project(project_path: str | os.PathLike, out_path: str | os.PathLike, overwrite: bool = False) -> str:
    """Copy the project file to ``out_path`` (folders are created) and return its absolute path.

    The source is never modified; the results are written into the copy. An existing copy is kept unless
    ``overwrite`` (Resume relies on this). Raises ValueError when both paths name the same file.
    """
    project_path, out_path = os.path.abspath(str(project_path)), os.path.abspath(str(out_path))
    if os.path.normcase(project_path) == os.path.normcase(out_path):
        raise ValueError("The output project must not be the input project")
    if os.path.exists(out_path) and not overwrite:
        return out_path
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.copyfile(project_path, out_path)
    return out_path


def tile_stats(s, progress: Callable[[float], None] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """One pass over a tile's patterns (a kikuchipy EBSD signal, lazy or not): the quasi-IQ map and the all-zero
    mask.

    Returns the per-pattern standard deviation as float32 of the navigation shape (rows, cols), and the bool
    mask of the same shape of the patterns that are all zero, or None when there is none. All-zero patterns
    cannot be indexed (processor versions before 1.7.0 wrote zeros for the patterns outside the IQ range; later
    ones keep them unprocessed), so the caller excludes them. A dask signal is reduced chunk by chunk in blocks
    of ``sph_core.SLAB_BYTES`` (``progress`` gets the fraction done, 0..1, through sph_core's dask callback); an
    in-memory signal in slabs of that size. The IQ is ``sph_core.pattern_std``, as everywhere.
    """
    nav_shape = tuple(int(v) for v in s.axes_manager.navigation_shape[::-1])
    sig_shape = tuple(int(v) for v in s.axes_manager.signal_shape[::-1])
    data = s.data
    try:
        import dask
        import dask.array as da
    except ImportError:
        da = None
    if da is not None and isinstance(data, da.Array):
        flat = data.reshape((-1,) + sig_shape)
        flat = flat.rechunk({0: "auto", 1: -1, 2: -1}, block_size_limit=sph_core.SLAB_BYTES)
        std = sph_core.pattern_std(flat)
        mx = flat.max(axis=(-2, -1))
        if progress is None:
            std, mx = dask.compute(std, mx)
        else:
            with sph_core._FractionCallback(progress):
                std, mx = dask.compute(std, mx)
            progress(1.0)
        std, mx = np.asarray(std), np.asarray(mx)
    else:
        arr = np.asarray(data).reshape((-1,) + sig_shape)
        n = arr.shape[0]
        slab = max(1, sph_core.SLAB_BYTES // max(1, int(np.prod(sig_shape)) * 4))
        std = np.empty(n, np.float32)
        mx = np.empty(n, arr.dtype)
        for i0 in range(0, n, slab):
            block = arr[i0 : i0 + slab]
            std[i0 : i0 + slab] = sph_core.pattern_std(block)
            mx[i0 : i0 + slab] = block.max(axis=(-2, -1))
            if progress is not None:
                progress(min(1.0, (i0 + slab) / n))
    iq = np.asarray(std, dtype=np.float32).reshape(nav_shape)
    zero = np.asarray(mx == 0).reshape(nav_shape)
    return iq, (zero if zero.any() else None)


def _full_grid_matrices(xmap) -> tuple[np.ndarray, np.ndarray]:
    """(n, 3, 3) rotation matrices of every point of the crystal map's grid (identity where the point is not in
    the data, i.e. was not indexed) and the (n,) bool in-data mask, both in row-major grid order. With several
    orientations per point only the best (first) one is taken."""
    in_data = np.asarray(xmap.is_in_data, dtype=bool).ravel()
    n = in_data.size
    mats = np.tile(np.eye(3), (n, 1, 1))
    rot = xmap.rotations
    if rot.ndim > 1:
        rot = rot[:, 0]
    mats[in_data] = rot.to_matrix().reshape(-1, 3, 3)
    return mats, in_data


def _full_grid_prop(xmap, name: str, in_data: np.ndarray, fill=0.0) -> np.ndarray:
    """The crystal map property ``name`` spread over the full grid as float64 (n,): ``fill`` where the point is
    not in the data, and everywhere when the map has no such property. With several values per point (one per
    orientation) the first one is taken."""
    out = np.full(in_data.size, fill, dtype=np.float64)
    if name in xmap.prop:
        values = np.asarray(xmap.prop[name], dtype=np.float64)
        if values.ndim > 1:
            values = values[:, 0]
        out[in_data] = values
    return out


def write_tile_result(out_path: str | os.PathLike, tile: Tile, xmap, iq_prop: str | None = None,
                      ci_prop: str = "scores", comment: str | None = None,
                      phase_map: dict[int, int] | None = None) -> dict:
    """Replace the orientation table of ``tile`` in the output project with the spherical indexing result.

    ``xmap`` is the tile's crystal map (shape (rows, cols); TWIST applied, as saved to .ang). Written into
    ``<tile>/EBSD/ANG/DATA/DATA`` of ``out_path`` (a :func:`copy_project` copy, opened read-write): the
    orientation matrices, ``CI`` (``ci_prop``; -1 for points not indexed), ``Phase`` (0-based index into the
    project's phase list: ``phase_map`` maps the crystal map's phase ids, default the same index), ``Fit`` 0 and
    ``Valid`` 1 for indexed points (0 otherwise). ``IQ`` keeps the project's (Hough) image quality unless
    ``iq_prop`` names a property of the crystal map (e.g. "iq", the spherical image quality on a 0..1 scale) to
    replace it with. The other fields (SEM signal, PRIAS strips) are kept. ``comment`` goes into
    ``EBSD/ANG/HEADER/Comments`` (when the dataset exists and accepts it); :func:`indexed_tiles` recognizes an
    indexed tile by it when it starts with :data:`RESULT_MARKER`. Returns ``{"n_points", "n_indexed",
    "n_not_indexed"}``.

    A tile of one row or one column is a 1-D crystal map; its points are the grid in row-major order too, so it
    is accepted. Raises ValueError when the crystal map's grid or the table's length is not the tile's grid.
    """
    from sph_core import full_shape

    shape = full_shape(xmap)
    one_line = min(tile.n_rows, tile.n_cols) == 1 and shape == (tile.n_rows * tile.n_cols,)
    if shape != (tile.n_rows, tile.n_cols) and not one_line:
        raise ValueError(f"{tile.name}: crystal map grid {shape} is not the tile's grid ({tile.n_rows}, {tile.n_cols})")
    mats, in_data = _full_grid_matrices(xmap)
    iq = None if iq_prop is None else _full_grid_prop(xmap, iq_prop, in_data, 0.0)
    ci = _full_grid_prop(xmap, ci_prop, in_data, -1.0)
    phase = np.zeros(in_data.size, dtype=np.int32)
    ids = np.asarray(xmap.phase_id).ravel()
    if phase_map:
        ids = np.fromiter((phase_map.get(int(i), int(i)) for i in ids), dtype=np.int64, count=ids.size)
    phase[in_data] = np.maximum(ids, 0).astype(np.int32)
    with h5py.File(out_path, "r+") as f:
        m = f[tile.h5_path]
        d = m["EBSD/ANG/DATA/DATA"]
        n = d.shape[0]
        if n != in_data.size:
            raise ValueError(f"{tile.name}: the orientation table has {n} rows, the tile grid {in_data.size} points")
        rec = d[()]
        rec["Orientations"] = mats.reshape(n, 9)
        if iq is not None:
            rec["IQ"] = iq.astype(rec["IQ"].dtype)
        rec["CI"] = ci.astype(rec["CI"].dtype)
        rec["Phase"] = phase
        if "Fit" in rec.dtype.names:
            rec["Fit"] = 0
        if "Valid" in rec.dtype.names:
            rec["Valid"] = in_data.astype(rec["Valid"].dtype)
        d[...] = rec
        if comment is not None and "EBSD/ANG/HEADER/Comments" in m:
            try:
                m["EBSD/ANG/HEADER/Comments"][0] = comment
            except (TypeError, ValueError, OSError):
                pass
    return {"n_points": int(n), "n_indexed": int(in_data.sum()), "n_not_indexed": int((~in_data).sum())}


# start of the comment the montage tab passes to write_tile_result; indexed_tiles (Resume) looks for it
RESULT_MARKER = "Spherical indexing"


def indexed_tiles(out_path: str | os.PathLike, tiles: Sequence[Tile]) -> set[str]:
    """The ``h5_path`` of every tile of ``tiles`` whose table in the output project was written by
    :func:`write_tile_result` (its ``EBSD/ANG/HEADER/Comments`` starts with :data:`RESULT_MARKER`). Empty when
    the output project does not exist; tiles without a readable comment are not counted."""
    done = set()
    if not os.path.exists(str(out_path)):
        return done
    with h5py.File(out_path, "r") as f:
        for t in tiles:
            key = f"{t.h5_path}/EBSD/ANG/HEADER/Comments"
            if key not in f:
                continue
            try:
                value = f[key][0]
            except (IndexError, TypeError, ValueError, OSError):
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            if str(value).startswith(RESULT_MARKER):
                done.add(t.h5_path)
    return done


def phase_mapping(master_names: Sequence[str], project_phases: Sequence[Phase], spec: str | None = None,
                  log: Log | None = None) -> dict[int, int]:
    """Which phase of the project each master pattern is: a dict master index -> 0-based index into the project's
    phase list (the ``Phase`` column of the output table).

    ``spec`` is the user's choice, the 1-based project phase number of every master, comma separated (e.g.
    "2,1"). Without it the masters are matched to the project's phases by name (one name containing the other,
    case insensitive), and when that is not unique for every master, by order when there are as many masters as
    phases. ``log`` gets one line saying which mapping was taken. Raises ValueError when no mapping can be made
    (bad or incomplete ``spec``, a phase number outside 1..n, a project without phases, or unmatched names with
    different counts), so that a wrong phase is never written.
    """
    n_m, n_p = len(master_names), len(project_phases)
    if n_p == 0:
        raise ValueError("The project has no phase list (EBSD/ANG/HEADER/Phase)")
    names_p = [str(p.name) for p in project_phases]
    if spec is not None and str(spec).strip():
        parts = [s.strip() for s in str(spec).replace(";", ",").split(",") if s.strip()]
        if len(parts) != n_m:
            raise ValueError(f"The phase assignment {spec!r} has {len(parts)} entries, "
                             f"but there are {n_m} master patterns")
        mapping = {}
        for i, part in enumerate(parts):
            try:
                k = int(part)
            except ValueError:
                raise ValueError(f"The phase assignment {spec!r}: {part!r} is not a phase number (1..{n_p})") from None
            if not 1 <= k <= n_p:
                raise ValueError(f"The phase assignment {spec!r}: phase {k} does not exist in the project (1..{n_p}: "
                                 + ", ".join(names_p) + ")")
            mapping[i] = k - 1
        if log is not None:
            log("Phases: " + ", ".join(f"master {i + 1} ({master_names[i] or '?'}) -> "
                                       f"project phase {k + 1} ({names_p[k]})" for i, k in mapping.items()))
        return mapping

    def words(text: str) -> set[str]:
        return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w}

    # a master matches a project phase whose name is the same, or whose name has the master's name as a whole
    # word (or the other way round): 'ferrite' matches 'Ferrite (bcc)', while 'ni' must not match 'austenite'
    mapping = {}
    for i, name in enumerate(master_names):
        n = (name or "").strip().lower()
        hits = [k for k, pn in enumerate(names_p)
                if n and (n == pn.lower() or n in words(pn) or pn.lower() in words(n))]
        if len(hits) == 1:
            mapping[i] = hits[0]
    if len(mapping) == n_m and len(set(mapping.values())) == n_m:
        if log is not None:
            log("Phases matched by name: " + ", ".join(f"master {i + 1} ({master_names[i]}) -> "
                                                       f"project phase {k + 1} ({names_p[k]})"
                                                       for i, k in mapping.items()))
        return mapping
    if n_m == n_p:
        if log is not None:
            log(f"NB! The master pattern names ({', '.join(m or '?' for m in master_names)}) could not all be "
                f"matched to the project's phases ({', '.join(names_p)}) by name: taken in order. "
                "Set the phase assignment to change that")
        return {i: i for i in range(n_m)}
    raise ValueError(
        f"{n_m} master pattern(s) ({', '.join(m or '?' for m in master_names)}) but the project has {n_p} phase(s) "
        f"({', '.join(names_p)}), and the names do not match: give the project phase number of each master in the "
        "phase assignment field (e.g. '2' or '2,1')"
    )


def read_tile_result(path: str | os.PathLike, tile: Tile) -> dict:
    """The orientation table of ``tile`` in a project (the source or an output copy) as a dict: ``matrices``
    (n, 3, 3) float64, ``iq``, ``ci``, ``phase`` (the table's own dtypes) and, when the table has the field,
    ``valid``; n rows in row-major grid order."""
    with h5py.File(path, "r") as f:
        rec = f[tile.h5_path]["EBSD/ANG/DATA/DATA"][()]
    out = {"matrices": rec["Orientations"].reshape(-1, 3, 3), "iq": rec["IQ"], "ci": rec["CI"], "phase": rec["Phase"]}
    if "Valid" in rec.dtype.names:
        out["valid"] = rec["Valid"]
    return out


def main():
    """Command line: print the tile table of a project, the first tile's parameters and the geometry warnings."""
    ap = argparse.ArgumentParser(description="List the tiles of an APEX montage project and their pattern files")
    ap.add_argument("project", help=".edaxh5 project")
    ap.add_argument("--patterns", help="folder with the .up1/.up2 files (default: the project's folder)")
    ap.add_argument("--suffix", help=f"prefer pattern files whose name ends with this, e.g. {DEFAULT_SUFFIX}")
    a = ap.parse_args()
    project = read_project(a.project)
    pattern_folder_tiles(project, a.patterns or project.folder, prefer_suffix=a.suffix, log=_print)
    print(f"{project.name}: {len(project.tiles)} tiles, {sum(t.ok for t in project.tiles)} with a pattern file")
    print(tile_table(project))
    first = project.tiles[0] if project.tiles else None
    if first is not None:
        print(f"\nPC (EDAX) {first.pc}, sample tilt {first.sample_tilt}, camera elevation {first.camera_elevation}, "
              f"step {first.step_x} x {first.step_y} um, phases: " + ", ".join(p.name for p in first.phases))
    for w in geometry_warnings(project):
        print("NB!", w)


if __name__ == "__main__":
    main()
