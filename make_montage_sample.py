#!/usr/bin/env python3
"""Cut a small, self-consistent sample out of an EDAX APEX montage (.edaxh5 + .up1/.up2 tiles) for development.

    python make_montage_sample.py "C:\\path\\to\\montage.edaxh5" ^
        --out example_data\\montage_sample --rows 3 --cols 40

Writes into --out:

* a copy of the .edaxh5 with the complete group hierarchy and every attribute, in which each map
  (GroupType 3) is cropped to the first --rows scan rows and --cols scan columns: ``Sample/Number Of
  Rows`` and ``Number Of Columns`` are set to the cropped size, every per-pattern table (N rows, N =
  rows x cols, row-major) keeps the patterns of the cropped grid only, and other large datasets keep
  their first --rows x --cols elements. The original grid size of every map is kept in the map's
  ``SampleOriginalRows`` / ``SampleOriginalCols`` attributes. Attributes whose name starts with
  ``User`` are blanked.
* every ``*.up1`` / ``*.up2`` found next to the .edaxh5 (or in --patterns) cropped to the same grid,
  with the header copied byte for byte (version, data offset, GUID) and only n_cols / n_rows patched.

With rows 3 and cols 40 a 115 x 115 px tile is 1.6 MB, so the sample of a 12-tile montage with three
pattern files fits comfortably in the git repository. Requires h5py and numpy.
"""
import argparse
import shutil
import struct
import uuid
from pathlib import Path

import h5py
import numpy as np


def _is_map(obj) -> bool:
    """True for a map group of an APEX project (attribute ``GroupType`` 3, one map per montage tile)."""
    # APEX stores scalars as length-1 arrays; numpy 2 refuses int() on those, hence the reshape
    return isinstance(obj, h5py.Group) and int(np.asarray(obj.attrs.get("GroupType", -1)).reshape(-1)[0]) == 3


def map_groups(f: h5py.File) -> list[str]:
    """The paths of every map group of the project, in file order."""
    out = []

    def visit(name, obj):
        if _is_map(obj):
            out.append(name)

    f.visititems(visit)
    return out


def copy_attrs(src, dst):
    """Copy every attribute of ``src`` to ``dst``. Attributes whose name starts with ``User`` (user names,
    stored by APEX as strings or string arrays) are blanked: empty strings of the same shape, zeros for
    numbers. An attribute h5py refuses to write is reported and skipped, never fatal."""
    for k, v in src.attrs.items():
        if k.startswith("User"):
            a = np.asarray(v)
            if isinstance(v, (str, bytes)):
                v = ""
            elif a.dtype.kind in "OSU":
                v = np.array([""] * a.size, dtype=h5py.string_dtype()).reshape(a.shape)
            else:
                v = np.zeros_like(a)
        try:
            dst.attrs[k] = v
        except Exception as e:  # noqa: BLE001
            print(f"  attr {src.name}@{k} not copied: {e}")


def crop_table(data: np.ndarray, n_cols: int, rows: int, cols: int) -> np.ndarray:
    """First `rows` x `cols` entries of a row-major (n_rows x n_cols) per-pattern table, as a (rows x cols,)
    table in the same order."""
    idx = (np.arange(rows)[:, None] * n_cols + np.arange(cols)[None, :]).ravel()
    return data[idx]


def copy_cropped(src: h5py.Group, dst: h5py.Group, grid: tuple[int, int] | None, rows: int, cols: int):
    """Copy the group ``src`` into ``dst`` recursively, cropping every map to its first ``rows`` x ``cols``.

    ``grid`` is (n_rows, n_cols) of the map this group belongs to, None outside maps; it is picked up from
    ``Sample/Number Of Rows|Columns`` when a map group is entered, and the map's original grid is kept in its
    ``SampleOriginalRows`` / ``SampleOriginalCols`` attributes (-1 when it has none). Inside a map,
    ``Sample/Number Of Rows|Columns`` are set to the cropped size, a dataset with exactly n_rows x n_cols
    entries (an orientation table, a per-pattern image) keeps the cropped grid's entries, and any other dataset
    over 4 MB keeps its first rows x cols elements. Scalars keep their length-1 shape and dtype. Datasets over
    64 kB are gzip compressed. Attributes are copied with :func:`copy_attrs`.
    """
    for name, obj in src.items():
        if isinstance(obj, h5py.Group):
            g = dst.create_group(name)
            copy_attrs(obj, g)
            this_grid = grid
            if _is_map(obj):
                try:
                    this_grid = (int(np.asarray(obj["Sample/Number Of Rows"][()]).reshape(-1)[0]),
                                 int(np.asarray(obj["Sample/Number Of Columns"][()]).reshape(-1)[0]))
                except KeyError:
                    this_grid = None
                    print(f"  {obj.name}: no Sample/Number Of Rows|Columns, datasets only truncated")
                g.attrs["SampleOriginalRows"] = this_grid[0] if this_grid else -1
                g.attrs["SampleOriginalCols"] = this_grid[1] if this_grid else -1
            copy_cropped(obj, g, this_grid, rows, cols)
            continue
        data = obj[()]
        if obj.shape == () or obj.size <= 1:
            # APEX stores every scalar as a length-1 array; keep the shape and dtype
            if grid is not None and src.name.endswith("/Sample") and name in ("Number Of Rows", "Number Of Columns"):
                keep = min(rows, grid[0]) if name == "Number Of Rows" else min(cols, grid[1])
                data = np.full_like(np.asarray(data), keep)
            d = dst.create_dataset(name, data=data)
        else:
            n = obj.shape[0]
            if grid is not None and n == grid[0] * grid[1]:
                data = crop_table(data, grid[1], min(rows, grid[0]), min(cols, grid[1]))
            elif obj.nbytes > 4 * 1024 * 1024:
                data = data[: rows * cols]
            d = dst.create_dataset(name, data=data, compression="gzip" if data.nbytes > 65536 else None)
        copy_attrs(obj, d)


def crop_pattern_file(src: Path, dst: Path, rows: int, cols: int):
    """Write the first ``rows`` x ``cols`` patterns of the .up1/.up2 file ``src`` to ``dst``.

    The header (version, pattern size, data offset at bytes 0:16; n_cols, n_rows at 17:25; the GUID at 42:58 in
    version 4) is copied byte for byte up to the data offset, and only n_cols / n_rows are patched to the
    cropped grid, so the file keeps its GUID and links to the cropped project. The patterns are read row by row
    from the source grid (row-major, width x height x 1 or 2 bytes each). Prints one line per file. Returns the
    GUID as a string ("" for a header without one), or None for a header before version 3, which has no grid
    and is skipped.
    """
    with open(src, "rb") as f:
        head = f.read(64)
        version, w, h, offset = struct.unpack_from("<IIII", head, 0)
        if version < 3:
            print(f"  {src.name}: header version {version} has no grid size, skipped")
            return None
        n_cols, n_rows = struct.unpack_from("<II", head, 17)
        bpp = 2 if src.suffix.lower() == ".up2" else 1
        header = head[:offset] if offset <= 64 else head + f.read(offset - 64)
        keep_r, keep_c = min(rows, n_rows), min(cols, n_cols)
        header = bytearray(header)
        struct.pack_into("<II", header, 17, keep_c, keep_r)
        pat = w * h * bpp
        with open(dst, "wb") as g:
            g.write(header)
            for r in range(keep_r):
                f.seek(offset + r * n_cols * pat)
                g.write(f.read(keep_c * pat))
    guid = ""
    if offset >= 58:
        guid = str(uuid.UUID(bytes_le=bytes(header[42:58])))
    print(f"  {src.name}: v{version} {w}x{h} px, {n_cols}x{n_rows} -> {keep_c}x{keep_r}, GUID {guid or '-'}")
    return guid


def main():
    """Command line (see the module docstring): write the cropped project, the cropped pattern files and a copy
    of every .ang next to them into --out."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edaxh5")
    ap.add_argument("--out", required=True, help="output folder (created)")
    ap.add_argument("--patterns", help="folder with the .up1/.up2 tiles (default: next to the .edaxh5)")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=40)
    a = ap.parse_args()

    src = Path(a.edaxh5)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pat_dir = Path(a.patterns) if a.patterns else src.parent

    print(f"{src.name}: cropping every map to {a.rows} rows x {a.cols} columns")
    with h5py.File(src, "r") as f, h5py.File(out / src.name, "w") as g:
        maps = map_groups(f)
        print(f"  {len(maps)} maps: " + ", ".join(m.rsplit('/', 1)[-1] for m in maps))
        copy_attrs(f, g)
        copy_cropped(f, g, None, a.rows, a.cols)
    print(f"  written {out / src.name} ({(out / src.name).stat().st_size / 1e6:.1f} MB)")

    files = sorted(p for p in pat_dir.iterdir() if p.suffix.lower() in (".up1", ".up2"))
    print(f"{len(files)} pattern files in {pat_dir}")
    for p in files:
        crop_pattern_file(p, out / p.name, a.rows, a.cols)
    for extra in pat_dir.glob("*.ang"):
        shutil.copy(extra, out / extra.name)
        print(f"  copied {extra.name}")
    print(f"\nDone. Add the folder to the repository:\n  git add \"{out}\"\n  git commit -m \"Montage sample\"\n"
          "  git push")


if __name__ == "__main__":
    main()
