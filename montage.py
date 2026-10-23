"""EDAX APEX montage ("stitch") projects: the .edaxh5 project file and its per-tile .up1/.up2 pattern files.

An APEX project (.edaxh5) holds every map ("tile") of a montage: its grid, step, pattern center, phases, stage
position and the Hough-indexed orientation table (``EBSD/ANG/DATA/DATA``), but no patterns. The patterns of a
tile are in a .up1/.up2 file whose version-4 header carries the tile's map GUID (the map group's ``Id``
attribute), which is how the two are linked. See STITCH_DATASET_NOTES.md.

This module reads a project (:func:`read_project`), finds and checks the pattern file of every tile
(:func:`read_pattern_header`, :func:`match_tiles`) and writes spherical indexing results of the tiles into a copy of
the project (:func:`copy_project`, :func:`write_tile_result`), so that OIM Analysis or MTEX can stitch the montage
as they would the Hough results. Nothing here is stitched: every tile keeps its own grid.

The orientation table's ``Orientations`` field is the 3x3 rotation matrix (row-major) of the orientation in the
same convention as a kikuchipy/orix rotation's ``to_matrix()``, checked against the Hough results of a real tile
(median misorientation 0.9 deg with a nickel master on austenite; the transpose gives 40 deg).

Command line: ``python montage.py project.edaxh5 [--patterns FOLDER] [--suffix _proc]`` prints the tile table.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shutil
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

GROUP_PROJECT, GROUP_SPECIMEN, GROUP_AREA, GROUP_MAP = 0, 1, 2, 3
_TILE_RC = re.compile(r"_r(\d+)c(\d+)$")
PATTERN_EXTENSIONS = (".up1", ".up2")
Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


def _scalar(group: h5py.Group, path: str, default=None):
    """A scalar dataset of APEX (stored as a length-1 array); bytes are decoded."""
    try:
        value = group[path][()]
    except KeyError:
        return default
    value = np.asarray(value).ravel()
    if value.size == 0:
        return default
    value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attr(obj, name: str, default=None):
    if name not in obj.attrs:
        return default
    value = np.asarray(obj.attrs[name]).ravel()
    if value.size == 0:
        return default
    value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


# ----------------------------------------------------------------------------------------------
# Pattern files
# ----------------------------------------------------------------------------------------------


@dataclass
class PatternHeader:
    """The header of an EDAX .up1/.up2 file (version 3 or 4; version 1 has no grid)."""

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
        return max(0, (self.file_size - self.offset) // (self.width * self.height * self.bytes_per_pixel))

    @property
    def n_patterns_expected(self) -> int | None:
        return None if self.n_cols is None or self.n_rows is None else self.n_cols * self.n_rows

    @property
    def size_ok(self) -> bool:
        expected = self.n_patterns_expected
        return expected is not None and self.file_size == self.offset + expected * self.width * self.height * self.bytes_per_pixel

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def read_pattern_header(path: str | os.PathLike) -> PatternHeader:
    """Read the header of a .up1/.up2 file. The map GUID is read from version-4 headers (bytes 42:58)."""
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
    """The headers of every .up1/.up2 file in ``folder`` (unreadable files are skipped)."""
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
    name: str
    formula: str
    lattice: tuple  # a, b, c, alpha, beta, gamma
    laue_group: str
    point_group: str
    symmetry: int | None


@dataclass
class Tile:
    """One map of the project and, once matched, its pattern file."""

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
    pc: tuple | None  # (x*, y*, z*), EDAX convention
    stage: tuple  # (x, y, z) mm
    phases: list[Phase] = field(default_factory=list)
    n_table_rows: int | None = None  # rows of the orientation table
    pattern: PatternHeader | None = None
    status: str = "no pattern file"

    @property
    def n_patterns(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def ok(self) -> bool:
        return self.pattern is not None and self.status == "ok"

    @property
    def pattern_file(self) -> str | None:
        return None if self.pattern is None else self.pattern.path


@dataclass
class Project:
    path: str
    name: str
    tiles: list[Tile]

    @property
    def folder(self) -> str:
        return os.path.dirname(os.path.abspath(self.path))

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    def tile(self, h5_path: str) -> Tile:
        for t in self.tiles:
            if t.h5_path == h5_path:
                return t
        raise KeyError(h5_path)


def _read_phases(m: h5py.Group) -> list[Phase]:
    phases = []
    if "EBSD/ANG/HEADER/Phase" not in m:
        return phases
    g = m["EBSD/ANG/HEADER/Phase"]
    for key in sorted(g, key=lambda k: (len(k), k)):
        p = g[key]
        if not isinstance(p, h5py.Group):
            continue
        lattice = tuple(_scalar(p, f"Lattice Constant {k}") for k in ("A", "B", "C", "Alpha", "Beta", "Gamma"))
        phases.append(Phase(str(_scalar(p, "Material Name", key)), str(_scalar(p, "Formula", "")), lattice,
                            str(_scalar(p, "Laue Group", "")), str(_scalar(p, "Point Group", "")), _scalar(p, "Symmetry")))
    return phases


def _read_tile(m: h5py.Group) -> Tile:
    name = str(_attr(m, "Name", m.name.rsplit("/", 1)[-1]))
    rc = _TILE_RC.search(name)
    row, col = (int(rc.group(1)), int(rc.group(2))) if rc else (None, None)
    n_rows, n_cols = _scalar(m, "Sample/Number Of Rows"), _scalar(m, "Sample/Number Of Columns")
    if n_rows is None or n_cols is None:
        raise ValueError(f"{m.name}: no Sample/Number Of Rows|Columns")
    pc_group = "EBSD/ANG/HEADER/Pattern Center Calibration"
    pc = tuple(_scalar(m, f"{pc_group}/{k}") for k in ("X-Star", "Y-Star", "Z-Star"))
    if any(v is None for v in pc):
        pc = None
    else:
        pc = tuple(float(v) for v in pc)
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
    """Every map (GroupType 3) of an APEX .edaxh5 project, in file order, with its acquisition parameters."""
    path = str(path)
    tiles = []
    with h5py.File(path, "r") as f:
        name = os.path.splitext(os.path.basename(path))[0]
        for key in f:
            if isinstance(f[key], h5py.Group) and _attr(f[key], "GroupType") == GROUP_PROJECT:
                name = str(_attr(f[key], "Name", name))
                break

        def visit(_name, obj):
            if isinstance(obj, h5py.Group) and _attr(obj, "GroupType") == GROUP_MAP:
                tiles.append(_read_tile(obj))

        f.visititems(visit)
    tiles.sort(key=lambda t: (t.row if t.row is not None else 1 << 30, t.col if t.col is not None else 1 << 30, t.h5_path))
    return Project(path, name, tiles)


def match_tiles(project: Project, headers: list[PatternHeader], prefer_suffix: str | None = None,
                log: Log | None = None) -> None:
    """Assign a pattern file to every tile of the project, in place.

    A file matches a tile by the GUID in its version-4 header; a file without a GUID matches by name (its stem
    ends with the tile's map name). When several files match one tile (e.g. the raw file and a processed copy,
    which keeps the header), the one whose stem ends with ``prefer_suffix`` wins, else the one with the shortest
    name (the raw file). The grid of the file must equal the tile's (``status`` says what is wrong otherwise),
    and the file must have exactly the patterns of its grid.
    """
    by_guid: dict[str, list[PatternHeader]] = {}
    without_guid: list[PatternHeader] = []
    for h in headers:
        if h.guid:
            by_guid.setdefault(h.guid.lower(), []).append(h)
        else:
            without_guid.append(h)
    for tile in project.tiles:
        candidates = list(by_guid.get((tile.guid or "").lower(), []))
        for h in without_guid:
            stem = os.path.splitext(h.name)[0]
            if stem.endswith(tile.name) or (prefer_suffix and stem.endswith(tile.name + prefer_suffix)):
                candidates.append(h)
        tile.pattern, tile.status = None, "no pattern file"
        if not candidates:
            continue

        def rank(h: PatternHeader):
            stem = os.path.splitext(h.name)[0]
            preferred = bool(prefer_suffix) and stem.endswith(prefer_suffix)
            return (not preferred, len(h.name), h.name)

        candidates.sort(key=rank)
        h = candidates[0]
        tile.pattern = h
        problems = []
        if h.n_cols is None:
            problems.append(f"header version {h.version} has no grid")
        else:
            if (h.n_rows, h.n_cols) != (tile.n_rows, tile.n_cols):
                problems.append(f"file grid {h.n_cols} x {h.n_rows} differs from the map's {tile.n_cols} x {tile.n_rows}")
            if tile.step_x is not None and h.step_x is not None and abs(h.step_x - tile.step_x) > 1e-6 * max(1.0, abs(tile.step_x)):
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
    """Match the tiles against the pattern files of ``folder``; returns the project's tiles."""
    match_tiles(project, find_pattern_files(folder), prefer_suffix=prefer_suffix, log=log)
    return project.tiles


def geometry_warnings(project: Project) -> list[str]:
    """Differences between the tiles in what one set of indexing parameters assumes to be shared."""
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
                       ", ".join(f"{n} has {values[n]}" for n in differing[:3]) + (" ..." if len(differing) > 3 else ""))
    widths = {t.pattern.width for t in tiles if t.pattern is not None}
    if len(widths) > 1:
        out.append(f"pattern width differs between the pattern files: {sorted(widths)}")
    return out


def tile_table(project: Project) -> str:
    rows = [("tile", "row", "col", "grid (cols x rows)", "pattern file", "status")]
    for t in project.tiles:
        rows.append((t.name, "" if t.row is None else str(t.row), "" if t.col is None else str(t.col),
                     f"{t.n_cols} x {t.n_rows}", t.pattern.name if t.pattern else "-", t.status))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def scan_dict(tile: Tile) -> dict:
    """The ``scan`` dict sph_core.load_patterns takes, from the tile (no .ang needed)."""
    h = tile.pattern
    return {"n_rows": tile.n_rows, "n_cols": tile.n_cols, "step_x": tile.step_x, "step_y": tile.step_y,
            "grid": tile.grid_type, "pc": tile.pc, "sample_tilt": tile.sample_tilt, "camera_elevation": tile.camera_elevation,
            "up_has_map_size": h is not None and h.n_cols is not None, "pattern_width": None if h is None else h.width,
            "pattern_height": None if h is None else h.height, "phases": [p.name for p in tile.phases]}


def output_name(tile: Tile, suffix: str, folder: str | os.PathLike | None = None) -> str:
    """``<pattern file stem><suffix><extension>`` in ``folder`` (default: the pattern file's folder)."""
    if tile.pattern is None:
        raise ValueError(f"{tile.name} has no pattern file")
    src = Path(tile.pattern.path)
    out_dir = Path(folder) if folder else src.parent
    return str(out_dir / f"{src.stem}{suffix}{src.suffix}")


# ----------------------------------------------------------------------------------------------
# Results into a copy of the project
# ----------------------------------------------------------------------------------------------


def copy_project(project_path: str | os.PathLike, out_path: str | os.PathLike, overwrite: bool = False) -> str:
    """A copy of the project file to write the results into (the source is never modified)."""
    project_path, out_path = os.path.abspath(str(project_path)), os.path.abspath(str(out_path))
    if os.path.normcase(project_path) == os.path.normcase(out_path):
        raise ValueError("The output project must not be the input project")
    if os.path.exists(out_path) and not overwrite:
        return out_path
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.copyfile(project_path, out_path)
    return out_path


def zero_pattern_mask(s) -> np.ndarray | None:
    """(rows, cols) True where a pattern is all zeros (the processor writes zeros for patterns it excluded), or
    None when there is no such pattern. Works for lazy signals."""
    nav_shape = tuple(int(v) for v in s.axes_manager.navigation_shape[::-1])
    sig_shape = tuple(int(v) for v in s.axes_manager.signal_shape[::-1])
    data = s.data
    try:
        import dask.array as da

        if isinstance(data, da.Array):
            flat = data.reshape((-1,) + sig_shape).max(axis=(-2, -1)).compute()
        else:
            flat = np.asarray(data).reshape((-1,) + sig_shape).max(axis=(-2, -1))
    except Exception:  # noqa: BLE001  (a layout that cannot be reshaped: go through numpy)
        flat = np.asarray(data).reshape((-1,) + sig_shape).max(axis=(-2, -1))
    mask = np.asarray(flat == 0).reshape(nav_shape)
    return mask if mask.any() else None


def tile_stats(s, progress: Callable[[float], None] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """One pass over a tile's patterns: the quasi-IQ map (per-pattern standard deviation, float32 (rows, cols))
    and the mask of all-zero patterns (None when there is none), e.g. the patterns the processor excluded."""
    nav_shape = tuple(int(v) for v in s.axes_manager.navigation_shape[::-1])
    sig_shape = tuple(int(v) for v in s.axes_manager.signal_shape[::-1])
    data = s.data
    try:
        import dask
        import dask.array as da

        if isinstance(data, da.Array):
            flat = data.reshape((-1,) + sig_shape)
            flat = flat.rechunk({0: "auto", 1: -1, 2: -1}, block_size_limit=64 * 1024 ** 2)
            std = flat.astype(np.float32).std(axis=(-2, -1))
            mx = flat.max(axis=(-2, -1))
            if progress is None:
                std, mx = dask.compute(std, mx)
            else:
                from sph_core import _FractionCallback

                with _FractionCallback(progress):
                    std, mx = dask.compute(std, mx)
                progress(1.0)
            std, mx = np.asarray(std), np.asarray(mx)
        else:
            raise TypeError
    except (ImportError, TypeError):
        arr = np.asarray(data).reshape((-1,) + sig_shape)
        n = arr.shape[0]
        slab = max(1, (64 * 1024 ** 2) // max(1, int(np.prod(sig_shape)) * 4))
        std = np.empty(n, np.float32)
        mx = np.empty(n, arr.dtype)
        for i0 in range(0, n, slab):
            block = arr[i0 : i0 + slab]
            std[i0 : i0 + slab] = block.astype(np.float32).std(axis=(-2, -1))
            mx[i0 : i0 + slab] = block.max(axis=(-2, -1))
            if progress is not None:
                progress(min(1.0, (i0 + slab) / n))
    iq = np.asarray(std, dtype=np.float32).reshape(nav_shape)
    zero = np.asarray(mx == 0).reshape(nav_shape)
    return iq, (zero if zero.any() else None)


def _full_grid_matrices(xmap) -> tuple[np.ndarray, np.ndarray]:
    """(n, 3, 3) rotation matrices of every point of the map's grid (identity where not in the data) and the
    (n,) in-data mask."""
    in_data = np.asarray(xmap.is_in_data, dtype=bool).ravel()
    n = in_data.size
    mats = np.tile(np.eye(3), (n, 1, 1))
    rot = xmap.rotations
    if rot.ndim > 1:  # several best orientations per point: the first one
        rot = rot[:, 0]
    mats[in_data] = rot.to_matrix().reshape(-1, 3, 3)
    return mats, in_data


def _full_grid_prop(xmap, name: str, in_data: np.ndarray, fill=0.0) -> np.ndarray:
    out = np.full(in_data.size, fill, dtype=np.float64)
    if name in xmap.prop:
        values = np.asarray(xmap.prop[name], dtype=np.float64)
        if values.ndim > 1:
            values = values[:, 0]
        out[in_data] = values
    return out


def write_tile_result(out_path: str | os.PathLike, tile: Tile, xmap, iq_prop: str = "iq", ci_prop: str = "scores",
                      comment: str | None = None, phase_map: dict[int, int] | None = None) -> dict:
    """Replace the orientation table of ``tile`` in the output project with the spherical indexing result.

    ``xmap`` is the tile's crystal map (shape (rows, cols); TWIST applied, as saved to .ang). Written: the
    orientation matrices, ``IQ`` (``iq_prop``), ``CI`` (``ci_prop``; -1 for points not indexed), ``Phase``
    (0-based index into the project's phase list: ``phase_map`` maps the crystal map's phase ids, default the
    same index), ``Fit`` 0 and ``Valid`` 1 for indexed points. The other fields (SEM signal, PRIAS strips) are
    kept. ``comment`` goes into ``EBSD/ANG/HEADER/Comments``. Returns counts.
    """
    if tuple(int(v) for v in xmap.shape) != (tile.n_rows, tile.n_cols):
        raise ValueError(f"{tile.name}: crystal map shape {tuple(xmap.shape)} is not the tile's grid ({tile.n_rows}, {tile.n_cols})")
    mats, in_data = _full_grid_matrices(xmap)
    iq = _full_grid_prop(xmap, iq_prop, in_data, 0.0)
    ci = _full_grid_prop(xmap, ci_prop, in_data, -1.0)
    ci[~in_data] = -1.0
    phase = np.zeros(in_data.size, dtype=np.int32)
    ids = np.asarray(xmap.phase_id).ravel()
    if phase_map:
        ids = np.vectorize(lambda i: phase_map.get(int(i), int(i)))(ids) if ids.size else ids
    phase[in_data] = np.maximum(ids, 0).astype(np.int32)
    with h5py.File(out_path, "r+") as f:
        m = f[tile.h5_path]
        d = m["EBSD/ANG/DATA/DATA"]
        n = d.shape[0]
        if n != in_data.size:
            raise ValueError(f"{tile.name}: the orientation table has {n} rows, the tile grid {in_data.size} points")
        rec = d[()]
        rec["Orientations"] = mats.reshape(n, 9)
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


def read_tile_result(path: str | os.PathLike, tile: Tile) -> dict:
    """The orientation table of ``tile`` in a project: matrices (n, 3, 3), iq, ci, phase, valid."""
    with h5py.File(path, "r") as f:
        rec = f[tile.h5_path]["EBSD/ANG/DATA/DATA"][()]
    out = {"matrices": rec["Orientations"].reshape(-1, 3, 3), "iq": rec["IQ"], "ci": rec["CI"], "phase": rec["Phase"]}
    if "Valid" in rec.dtype.names:
        out["valid"] = rec["Valid"]
    return out


def main():
    ap = argparse.ArgumentParser(description="List the tiles of an APEX montage project and their pattern files")
    ap.add_argument("project", help=".edaxh5 project")
    ap.add_argument("--patterns", help="folder with the .up1/.up2 files (default: the project's folder)")
    ap.add_argument("--suffix", help="prefer pattern files whose name ends with this, e.g. _proc")
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
