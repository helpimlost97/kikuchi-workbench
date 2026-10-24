#!/usr/bin/env python3
"""Print the acquisition parameters of an EBSD scan, from either an EDAX
.edaxh5 (OIM/APEX HDF5) file or an EDAX .ang (TSL OIM Analysis text) file.
Which parser runs is chosen from the file extension.

    python ebsd_params.py "LBW 304L.edaxh5"                      # every OIM map in the file
    python ebsd_params.py "LBW 304L.edaxh5" --map "Base Metal"   # only maps whose path contains this
    python ebsd_params.py "LBW 304L.edaxh5" --all                # every scalar dataset in the file, raw
    python ebsd_params.py "scan.ang"                             # ang header parameters
    python ebsd_params.py "LBW 304L.edaxh5" --px 115             # + EMsoft pattern center for 115x115 px patterns
    python ebsd_params.py "scan.ang" --up1 scan.up1 --delta 55   # pattern size from the .up1 header, RKD pixel size

Pattern center: every map/.ang prints the EDAX pattern center (x*, y*, z*) and, when
the pattern size in pixels is known (--px W[xH] or --up1 FILE), the EMsoft one:
    xpc = W (x* - 1/2)      ypc = W y* - H/2      L = W * delta * z*      [px, px, um]
delta (um per pattern pixel) comes from --delta, else from --det-mm (detector width,
default 30 mm = the EDAX detector on the Quattro SEM): delta = det_mm * 1000 / W.
RKD data: pass --delta 55.

Needs h5py and numpy for .edaxh5 files. .ang files need only the standard library.
Activate the environment that has h5py and numpy first, e.g. `source /path/to/venv/bin/activate`.

Windows (PowerShell or cmd), same options; quote paths which contain spaces:
    python "C:/path/to/ebsd_params.py" "C:/data/scan.ang" --up1 "C:/data/scan.up1" --delta 55

From Python:
    import ebsd_params
    scan = ebsd_params.scan_info("scan.ang", pattern_file="scan.up1")
    ebsd_params.print_scan_info(scan)
    scan["pc"], scan["pattern_width"], scan["pattern_height"]

--- .edaxh5 ---
For each EBSD map group ("OIM Map N", "EBSD+EDS Map N", ...) it prints the sections that matter for indexing:
Camera (elevation/azimuth, model), Column (kV, WD, magnification), Sample (tilt, step,
grid), the ANG header (pattern center x*/y*/z*, phases with lattice constants and
symmetry), and the Hough (OHP) header. Big arrays (orientation data, images, HKL
tables) are shown as shape/dtype only.

--- .ang ---
Parses the '#' header lines (both "# KEY value" and "# KEY: value" forms). Prints,
labelled, in a fixed order: TEM_PIXperUM, x-star, y-star, z-star, WorkingDistance,
SampleTiltAngle, CameraElevationAngle, CameraAzimuthalAngle, GRID, XSTEP, YSTEP,
NCOLS_ODD, NCOLS_EVEN, NROWS, OPERATOR, SAMPLEID, SCANID; every Phase block
(MaterialName, Formula, Info, Symmetry, LatticeConstants, NumberFamilies); the
count of data rows (lines not starting with '#', counted by streaming -- the data
section is never loaded into memory); and, finally, any '#' header lines that were
not recognized ("other header lines"), so nothing in the header is silently dropped.
--map/--all are ignored for .ang files (they only apply to .edaxh5).

Note: pattern pixel size (binning/pattern resolution) is NOT stored in either an
.edaxh5 map's headers or an .ang header -- it lives in the raw pattern file's
(.up1/.up2) header instead. A pattern processor, or EMSphInx's EBSPDims utility,
will report it from the .up1/.up2 file directly.
"""
import argparse
import os
import re
import struct
import sys

try:  # only needed for .edaxh5 files; .ang files and .up1 headers need the standard library only
    import h5py
    import numpy as np
except ImportError:
    h5py = None
    np = None

# ----------------------------------------------------------------------------
# .edaxh5 support
# ----------------------------------------------------------------------------

# the sections of a map group printed for indexing, in this order
SECTIONS = ["Camera", "Column", "Sample", "EBSD/ANG/HEADER", "EBSD/OHP/HEADER"]
# datasets and groups whose name contains one of these are large tables or images: shown as shape/dtype only
SKIP = ("HKL Families", "DATA", "Annotations", "Images", "FOVIMAGE")

def fmt(v):
    """A dataset value as text: a scalar (APEX stores scalars as length-1 arrays) or a short list as is,
    bytes decoded, anything with more than 12 elements as ``<shape dtype>``."""
    v = np.asarray(v)
    if v.dtype.kind == "S":
        v = np.char.decode(v, errors="replace")
    if v.size == 1:
        v = v.reshape(-1)[0]
        return v.decode(errors="replace") if isinstance(v, bytes) else str(v)
    if v.size <= 12:
        return "[" + ", ".join(str(x.decode(errors="replace") if isinstance(x, bytes) else x)
                               for x in v.reshape(-1)) + "]"
    return f"<{v.shape} {v.dtype}>"

def _ebsd_map_groups(f):
    """The paths of every EBSD map group of an open .edaxh5 file, in file order: a group whose own name
    contains " Map" ("OIM Map 1", "EBSD+EDS Map 2", ...) and which holds an ``EBSD`` group."""
    maps = []
    f.visititems(lambda n, o: maps.append(n)
                 if isinstance(o, h5py.Group) and " Map" in n.split("/")[-1] and "EBSD" in o else None)
    return maps

def read_up_dims(path):
    """(width, height) in pixels of the patterns of a .up1/.up2 file (from its header)."""
    h = read_up_header(path)
    return int(h["pattern_width"]), int(h["pattern_height"])

def read_up_header(path):
    """Header of an EDAX .up1/.up2 file as a dict: version, pattern_width, pattern_height,
    offset, n_patterns, size_gb and, for version 3 and later, n_cols, n_rows, grid, step_x,
    step_y. Version 1 files store neither the map size nor the step size.

    Little-endian layout: version, width, height and data offset as uint32 at bytes 0:16; from version 3 on
    n_cols and n_rows (uint32) at 17:25, the hexagonal-grid flag (uint8) at 25 and the steps in um (float64)
    at 26:42. n_patterns is what the file size holds after the offset, not the grid; size_gb is the file size
    in GB. Raises OSError when the file cannot be read, struct.error when it is shorter than its header."""
    path = os.path.expanduser(str(path))
    itemsize = 2 if path.lower().endswith(".up2") else 1
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        version, w, h, offset = struct.unpack("<IIII", f.read(16))
        info = {"version": version, "pattern_width": w, "pattern_height": h, "offset": offset}
        if version >= 3:
            f.seek(17)
            n_cols, n_rows = struct.unpack("<II", f.read(8))
            is_hex = struct.unpack("<B", f.read(1))[0]
            step_x, step_y = struct.unpack("<dd", f.read(16))
            info.update(n_cols=n_cols, n_rows=n_rows, step_x=step_x, step_y=step_y,
                        grid="HexGrid" if is_hex else "SqrGrid")
    info["n_patterns"] = (size - offset) // (w * h * itemsize) if w and h else 0
    info["size_gb"] = size / 1e9
    return info

def pattern_size(a):
    """(W, H) in pixels from the parsed arguments: ``--px W[xH]`` (H defaults to W), else the ``--up1`` header,
    else None. Raises ValueError for a ``--px`` that is not a number."""
    if a.px:
        parts = a.px.lower().replace("×", "x").split("x")
        w = int(parts[0])
        h = int(parts[1]) if len(parts) > 1 and parts[1] else w
        return w, h
    if a.up1:
        return read_up_dims(a.up1)
    return None

def print_pattern_center(xs, ys, zs, a, indent="  "):
    """Print the EDAX pattern center (x*, y*, z*) and, when the pattern size is known from the arguments, the
    EMsoft one: xpc = W (x* - 1/2), ypc = W y* - H/2 in pixels and L = W delta z* in um, with delta (um per
    pattern pixel) from ``--delta`` or ``--det-mm`` (detector width in mm) / W."""
    print(f"{indent}EDAX   x* = {xs:.6f}   y* = {ys:.6f}   z* = {zs:.6f}")
    dims = pattern_size(a)
    if dims is None:
        print(f"{indent}EMsoft xpc, ypc, L: need the pattern size in pixels -> add --px 115 (or --up1 <patterns.up1>)")
        return
    W, H = dims
    delta = a.delta if a.delta else a.det_mm * 1000.0 / W
    src = f"--delta {a.delta:g}" if a.delta else f"--det-mm {a.det_mm:g} / {W} px"
    xpc = W * (xs - 0.5)
    ypc = W * ys - H / 2.0
    L = W * delta * zs
    print(f"{indent}EMsoft xpc = {xpc:.3f} px   ypc = {ypc:.3f} px   L = {L:.1f} um"
          f"   [{W}x{H} px, delta = {delta:.4f} um/px from {src}]")

def dump_group(g, indent, out):
    """Append one line per dataset of ``g`` (recursively, subgroups as ``[name]`` blocks) to the list ``out``:
    ``name = value``, or ``name: <shape dtype>`` for the large datasets named in SKIP, which are not read."""
    for k in sorted(g.keys()):
        obj = g[k]
        if any(s in k for s in SKIP):
            if isinstance(obj, h5py.Dataset):
                out.append(f"{indent}{k}: <{obj.shape} {obj.dtype}>")
            continue
        if isinstance(obj, h5py.Group):
            out.append(f"{indent}[{k}]")
            dump_group(obj, indent + "  ", out)
        else:
            try:
                out.append(f"{indent}{k} = {fmt(obj[()])}")
            except Exception as e:
                out.append(f"{indent}{k} = <unreadable: {e}>")

def run_edaxh5(a):
    """The command-line report of a .edaxh5 file: with ``--all`` every dataset of the file, raw; otherwise the
    sections of every EBSD map (filtered by ``--map``) and its pattern center. Exits with a message when h5py is
    missing or no map matches."""
    if h5py is None:
        sys.exit("h5py and numpy are needed to read .edaxh5 files: python -m pip install h5py numpy")
    with h5py.File(a.file, "r") as f:
        if a.all:
            def walk(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"{name} = {fmt(obj[()]) if obj.size <= 12 else f'<{obj.shape} {obj.dtype}>'}")
            f.visititems(walk)
            return
        maps = _ebsd_map_groups(f)
        if a.map:
            maps = [m for m in maps if a.map.lower() in m.lower()]
        if not maps:
            sys.exit(f"no EBSD map groups found{' matching ' + repr(a.map) if a.map else ''} in {a.file}")
        for m in maps:
            print("=" * 100)
            print(m)
            print("=" * 100)
            g = f[m]
            # small datasets directly under the map (pattern dims, binning, etc. if present)
            out = []
            for k in sorted(g.keys()):
                if isinstance(g[k], h5py.Dataset) and not any(s in k for s in SKIP):
                    out.append(f"  {k} = {fmt(g[k][()])}")
            if out:
                print("\n".join(out))
            for sec in SECTIONS:
                if sec in g:
                    print(f"--- {sec}")
                    out = []
                    dump_group(g[sec], "  ", out)
                    print("\n".join(out))
            pc = "EBSD/ANG/HEADER/Pattern Center Calibration"
            if pc in g:
                try:
                    xs, ys, zs = (float(np.asarray(g[pc][k][()]).reshape(-1)[0])
                                  for k in ("X-Star", "Y-Star", "Z-Star"))
                    print("--- Pattern center")
                    print_pattern_center(xs, ys, zs, a)
                except Exception as e:
                    print(f"--- Pattern center: <unreadable: {e}>")
            print()

# ----------------------------------------------------------------------------
# .ang support
# ----------------------------------------------------------------------------

ANG_TOP_LEVEL_KEYS = [
    "TEM_PIXperUM", "x-star", "y-star", "z-star", "WorkingDistance",
    "SampleTiltAngle", "CameraElevationAngle", "CameraAzimuthalAngle",
    "GRID", "XSTEP", "YSTEP", "NCOLS_ODD", "NCOLS_EVEN", "NROWS",
    "OPERATOR", "SAMPLEID", "SCANID",
]
ANG_PHASE_KEYS = ["MaterialName", "Formula", "Info", "Symmetry", "LatticeConstants", "NumberFamilies"]

# a header line after the '#': the key, then whatever follows (possibly nothing)
_KV_RE = re.compile(r"^(\S+)\s*(.*)$")

def _parse_ang_header_line(stripped):
    """(key, value) of a '#' header line with the leading '#' and surrounding
    whitespace already removed. Handles both 'KEY value' and 'KEY: value'
    forms, and keys with no value at all (value ""). Never raises."""
    m = _KV_RE.match(stripped)
    if not m:
        return stripped, ""
    key = m.group(1).rstrip(":")
    value = m.group(2).strip()
    return key, value

def parse_ang(path):
    """Stream-parse an .ang file. Returns (top, phases, other, n_data_rows): the top-level header values by key
    of ANG_TOP_LEVEL_KEYS (as strings), one dict per Phase block (its "number" and the ANG_PHASE_KEYS, None when
    absent), the header lines recognized as neither ("key: value" strings) and the count of data rows. Keys are
    matched case insensitively. Data rows are only counted, never loaded, so this is safe on a 10+ MB data
    section. Raises OSError when the file cannot be read."""
    top = {}
    phases = []
    other = []
    current_phase = None
    n_data_rows = 0
    in_header = True

    with open(path, "r", errors="replace") as f:
        for line in f:
            if in_header:
                if line.startswith("#"):
                    stripped = line[1:].strip()
                    if not stripped:
                        continue
                    key, value = _parse_ang_header_line(stripped)
                    lower = key.lower()

                    if lower == "phase":
                        current_phase = {"number": value or "?"}
                        for pk in ANG_PHASE_KEYS:
                            current_phase[pk] = None
                        phases.append(current_phase)
                        continue

                    matched_top = next((k for k in ANG_TOP_LEVEL_KEYS if k.lower() == lower), None)
                    if matched_top:
                        top[matched_top] = value
                        current_phase = None
                        continue

                    if current_phase is not None:
                        matched_phase_key = next((k for k in ANG_PHASE_KEYS if k.lower() == lower), None)
                        if matched_phase_key:
                            current_phase[matched_phase_key] = value
                            continue
                        other.append(f"[Phase {current_phase['number']}] {key}: {value}")
                        continue

                    other.append(f"{key}: {value}")
                    continue
                elif not line.strip():
                    continue
                else:
                    in_header = False  # fall through: this line is the first data row

            if line.strip() and not line.startswith("#"):
                n_data_rows += 1

    return top, phases, other, n_data_rows

def run_ang(a):
    """The command-line report of an .ang file: the top-level header values in a fixed order, the pattern
    center, every Phase block, the number of data rows and the header lines not recognized."""
    if a.map or a.all:
        print("note: --map/--all only apply to .edaxh5 files; ignoring for .ang input", file=sys.stderr)
    top, phases, other, n_data_rows = parse_ang(a.file)

    print("=" * 100)
    print(a.file)
    print("=" * 100)
    for k in ANG_TOP_LEVEL_KEYS:
        print(f"{k}: {top.get(k, '(not found)')}")
    print()
    try:
        xs, ys, zs = (float(top[k]) for k in ("x-star", "y-star", "z-star"))
        print("Pattern center")
        print_pattern_center(xs, ys, zs, a)
        print()
    except (KeyError, ValueError):
        print("Pattern center: x-star/y-star/z-star not found in header")
        print()

    if phases:
        for ph in phases:
            print(f"Phase {ph['number']}")
            for pk in ANG_PHASE_KEYS:
                v = ph.get(pk)
                print(f"  {pk}: {'(not found)' if v is None else v}")
            print()
    else:
        print("(no Phase blocks found)")
        print()

    print(f"data rows: {n_data_rows}")

    if other:
        print()
        print("other header lines:")
        for o in other:
            print(f"  {o}")

# ----------------------------------------------------------------------------
# Importable interface (used by sph_core.py and the workbench)
# ----------------------------------------------------------------------------

SCAN_INFO_KEYS = [
    "scan_file", "pattern_file", "pattern_width", "pattern_height", "up_version", "n_patterns",
    "size_gb", "up_has_map_size", "n_cols", "n_rows", "grid", "step_x", "step_y", "pc",
    "sample_tilt", "camera_elevation", "camera_azimuth", "working_distance", "phases", "map", "notes",
]

def _to_float(value):
    """The first word of ``value`` as a float ("2.0 um" -> 2.0), None when there is none or it is not a number."""
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError, TypeError):
        return None

def _to_int(value):
    """:func:`_to_float` truncated to an int, None when it is None."""
    number = _to_float(value)
    return None if number is None else int(number)

def _scan_info_ang(path, info):
    """Fill the scan-file keys of ``info`` (see :func:`scan_info`) from an .ang header. Only NCOLS_ODD is taken
    as the column count. A note is added when the file looks like an EMSphInx result rather than the acquisition's."""
    top, phases, _, _ = parse_ang(path)
    pc = tuple(_to_float(top.get(k)) for k in ("x-star", "y-star", "z-star"))
    info["pc"] = None if None in pc else pc
    info["working_distance"] = _to_float(top.get("WorkingDistance"))
    info["sample_tilt"] = _to_float(top.get("SampleTiltAngle"))
    info["camera_elevation"] = _to_float(top.get("CameraElevationAngle"))
    info["camera_azimuth"] = _to_float(top.get("CameraAzimuthalAngle"))
    info["grid"] = top.get("GRID") or None
    info["n_cols"] = _to_int(top.get("NCOLS_ODD"))
    info["n_rows"] = _to_int(top.get("NROWS"))
    info["step_x"] = _to_float(top.get("XSTEP"))
    info["step_y"] = _to_float(top.get("YSTEP"))
    info["phases"] = [ph.get("MaterialName") or "?" for ph in phases]
    if "sphinx" in os.path.basename(path).lower() or any("EMSphInx" in str(ph.get("Info")) for ph in phases):
        info["notes"].append("this .ang looks like an indexing result (EMSphInx), not the one from data collection")

def _scan_info_edaxh5(path, info, map_filter):
    """Fill the scan-file keys of ``info`` (see :func:`scan_info`) from the first EBSD map of a .edaxh5 file
    whose path contains ``map_filter`` (case insensitive; every map when None).

    Best effort: the pattern center is read from its known place
    (``EBSD/ANG/HEADER/Pattern Center Calibration/X-Star|Y-Star|Z-Star``), the phases from
    ``EBSD/ANG/HEADER/Phase/<n>/Material Name``; everything else is looked up by dataset name, ignoring case,
    spaces and underscores, in the sections EBSD/ANG/HEADER, Sample, Camera and Column (first hit wins), and a
    note says so. Raises ImportError without h5py."""
    if h5py is None:
        raise ImportError("h5py and numpy are needed to read .edaxh5 files")
    with h5py.File(path, "r") as f:
        maps = _ebsd_map_groups(f)
        if map_filter:
            maps = [m for m in maps if map_filter.lower() in m.lower()]
        if not maps:
            info["notes"].append("no EBSD map groups found" + (f" matching {map_filter!r}" if map_filter else ""))
            return
        if len(maps) > 1:
            info["notes"].append(f"{len(maps)} maps in the file, using the first; choose one with the map filter: "
                                 + "; ".join(maps))
        g = f[maps[0]]
        info["map"] = maps[0]

        def scalar(obj):
            try:
                v = np.asarray(obj[()]).reshape(-1)[0]
                return v.decode(errors="replace") if isinstance(v, bytes) else v
            except Exception:
                return None

        pc = "EBSD/ANG/HEADER/Pattern Center Calibration"
        if pc in g:
            values = tuple(_to_float(scalar(g[pc][k])) if k in g[pc] else None for k in ("X-Star", "Y-Star", "Z-Star"))
            info["pc"] = None if None in values else values

        # every scalar of the sections which matter, by lower case name without spaces, first hit wins
        found = {}
        def collect(group):
            for k in group.keys():
                obj = group[k]
                if any(s in k for s in SKIP):
                    continue
                if isinstance(obj, h5py.Group):
                    collect(obj)
                elif obj.size == 1:
                    found.setdefault(k.lower().replace(" ", "").replace("_", ""), scalar(obj))
        for sec in ("EBSD/ANG/HEADER", "Sample", "Camera", "Column"):
            if sec in g:
                collect(g[sec])

        def lookup(*names):
            for name in names:
                if found.get(name) is not None:
                    return found[name]
            return None

        info["sample_tilt"] = _to_float(lookup("sampletiltangle", "sampletilt", "tiltangle", "tilt"))
        info["camera_elevation"] = _to_float(lookup("cameraelevationangle", "elevationangle", "elevation"))
        info["camera_azimuth"] = _to_float(lookup("cameraazimuthalangle", "azimuthalangle", "azimuth"))
        info["working_distance"] = _to_float(lookup("workingdistance", "semwd", "wd"))
        info["step_x"] = _to_float(lookup("stepx", "xstep"))
        info["step_y"] = _to_float(lookup("stepy", "ystep"))
        info["n_cols"] = _to_int(lookup("numberofcolumns", "ncolumns", "ncolsodd", "ncols"))
        info["n_rows"] = _to_int(lookup("numberofrows", "nrows"))
        grid = lookup("gridtype", "grid")
        info["grid"] = None if grid is None else str(grid)
        phase_group = "EBSD/ANG/HEADER/Phase"
        if phase_group in g:
            # the keys are the phase numbers: numeric order, "10" after "9"; non-numeric keys last
            names = sorted(g[phase_group].keys(), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
            info["phases"] = [scalar(g[phase_group][k]["Material Name"]) for k in names
                              if "Material Name" in g[phase_group][k]]
        info["notes"].append("values other than the pattern center were looked up by name in the .edaxh5 file; "
                             "check them")

def scan_info(scan_file=None, pattern_file=None, map_filter=None):
    """Acquisition parameters of a scan as a dict with the keys SCAN_INFO_KEYS (None = unknown).

    scan_file     .ang or .edaxh5/.h5 file from the data collection, or None
    pattern_file  .up1/.up2 pattern file, or None; gives the pattern width and height
    map_filter    .edaxh5 only: part of the name of the map to use

    The pattern center "pc" is (x*, y*, z*) in the EDAX convention; the tilts are in degrees, the
    working distance in mm, the steps in um, "grid" is "SqrGrid" or "HexGrid". The map size and
    step size are those of the pattern file when it stores them (version 3 and later; then
    "up_has_map_size" is True), else of the scan file. "notes" lists what to check (a map size that
    differs between the files, values looked up by name, an unknown extension). Raises OSError when
    a file cannot be read, ImportError for a .edaxh5 without h5py.
    """
    info = dict.fromkeys(SCAN_INFO_KEYS)
    info["notes"] = []
    info["phases"] = []
    info["up_has_map_size"] = False

    if scan_file:
        scan_file = os.path.expanduser(str(scan_file))
        info["scan_file"] = scan_file
        ext = os.path.splitext(scan_file)[1].lower()
        if ext == ".ang":
            _scan_info_ang(scan_file, info)
        elif ext in (".edaxh5", ".h5", ".hdf5"):
            _scan_info_edaxh5(scan_file, info, map_filter)
        else:
            info["notes"].append(f"unrecognized extension {ext!r}: expected .ang or .edaxh5")

    if pattern_file:
        pattern_file = os.path.expanduser(str(pattern_file))
        info["pattern_file"] = pattern_file
        if os.path.splitext(pattern_file)[1].lower() in (".up1", ".up2"):
            up = read_up_header(pattern_file)
            info["pattern_width"] = up["pattern_width"]
            info["pattern_height"] = up["pattern_height"]
            info["up_version"] = up["version"]
            info["n_patterns"] = up["n_patterns"]
            info["size_gb"] = up["size_gb"]
            if "n_cols" in up:
                info["up_has_map_size"] = True
                scan_size = (info["n_cols"], info["n_rows"])
                if None not in scan_size and scan_size != (up["n_cols"], up["n_rows"]):
                    info["notes"].append(
                        f"the map size of the scan file ({scan_size[0]} x {scan_size[1]}) differs from the one of the "
                        f"pattern file ({up['n_cols']} x {up['n_rows']}); are they from the same scan? "
                        "Using the pattern file's"
                    )
                for k in ("n_cols", "n_rows", "grid", "step_x", "step_y"):
                    info[k] = up[k]
        else:
            info["notes"].append("the pattern size is only read from .up1/.up2 files")
    return info

def print_scan_info(info):
    """Print the dict returned by scan_info() as the labelled summary the workbench shows: pattern file, scan
    file, map size and step, then every note as an "NB!" line."""
    def show(label, value, unit=""):
        print(f"  {label:<26}{'(not found)' if value is None else str(value) + unit}")

    if info.get("pattern_file"):
        print(f"Pattern file: {os.path.basename(info['pattern_file'])}")
        if info.get("pattern_width") is not None:
            show("Pattern width x height:", f"{info['pattern_width']} x {info['pattern_height']}", " px")
            show("Number of patterns:", info["n_patterns"])
            show("File:", f"{info['size_gb']:.1f} GB, version {info['up_version']}")
    else:
        print("Pattern file: none given")
    print()
    if info.get("scan_file"):
        print(f"Scan file: {os.path.basename(info['scan_file'])}")
        if info.get("map"):
            show("Map:", info["map"])
        pc = info.get("pc")
        show("Pattern center (EDAX):", None if pc is None else "x* = {:g}, y* = {:g}, z* = {:g}".format(*pc))
        show("Sample tilt:", info.get("sample_tilt"), " deg")
        show("Camera elevation:", info.get("camera_elevation"), " deg")
        show("Working distance:", info.get("working_distance"), " mm")
        show("Phases:", ", ".join(str(p) for p in info["phases"]) if info.get("phases") else None)
    else:
        print("Scan file: none given")
    print()
    print("Map" + (" (from the pattern file)" if info.get("up_has_map_size")
                   else " (from the scan file)" if info.get("n_cols") else ""))
    size = None if info.get("n_cols") is None else f"{info['n_cols']} x {info['n_rows']}"
    show("Columns x rows:", size)
    show("Grid:", info.get("grid"))
    step = None if info.get("step_x") is None else f"{info['step_x']:g}, {info['step_y']:g}"
    show("Step size x, y:", step, " um")
    for note in info.get("notes") or []:
        print(f"\nNB! {note}")

# ----------------------------------------------------------------------------

def main():
    """Command line (see the module docstring): the report of one .ang or .edaxh5 file, chosen by extension."""
    # Windows consoles are often not UTF-8: never fail on a character they cannot show
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--map", default=None, help="substring filter on the map path (.edaxh5 only)")
    ap.add_argument("--all", action="store_true", help="dump every small dataset in the file, raw (.edaxh5 only)")
    ap.add_argument("--px", default=None, metavar="W[xH]",
                    help="pattern size in pixels, e.g. 115 or 479x479 (enables the EMsoft pattern center)")
    ap.add_argument("--up1", default=None, metavar="FILE",
                    help="read the pattern size from this .up1/.up2 header instead of --px")
    ap.add_argument("--delta", type=float, default=None, metavar="UM",
                    help="pixel size in um per pattern pixel (RKD: 55); overrides --det-mm")
    ap.add_argument("--det-mm", type=float, default=30.0, metavar="MM",
                    help="detector width in mm used to derive delta when --delta is absent "
                         "(default 30 = Quattro EDAX detector)")
    a = ap.parse_args()
    a.file = os.path.expanduser(a.file)   # '~/...' inside quotes reaches us unexpanded

    ext = os.path.splitext(a.file)[1].lower()
    if ext == ".ang":
        run_ang(a)
    elif ext in (".edaxh5", ".h5", ".hdf5"):
        run_edaxh5(a)
    else:
        sys.exit(f"unrecognized extension {ext!r} for {a.file}: expected .edaxh5 or .ang")

if __name__ == "__main__":
    main()
