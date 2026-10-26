# EDAX montage ("Stitch") dataset: how .edaxh5 and .up1 files relate

Reference dataset: an APEX montage project, `GMAW 304L 309.edaxh5` plus one `.up1` per tile; a copy
cropped to a few points per tile is in `example_data/montage_sample/`
(collected 2026-09-28, EDAX APEX `PRODUCT_VERSION V3.1.2007.0001`, Velocity Gen1 camera).

Purpose: give the information needed to add **batch pattern processing + batch spherical
indexing** to Kikuchi Workbench for montage scans. Workbench does no stitching. It processes and
indexes each tile on its own and writes per-tile outputs that an external stitcher can use.

## Files

| File | Size | Contents |
|---|---|---|
| `GMAW 304L 309.edaxh5` | 481 MB | HDF5 project: metadata + Hough-indexed ANG data for **all 12 tiles**. **No patterns.** |
| `GMAW 304L 309_Stitch_Area 1_OIM Map 1_r000c000.up1` | 2.24 GB | Patterns for tile r000c000 |
| `..._r000c001.up1` | 2.24 GB | Patterns for tile r000c001 |
| `..._r000c002.up1` | 2.24 GB | Patterns for tile r000c002 |

The montage has 12 tiles (c000 to c011), but only 3 have .up1 files here. Batch code must handle
tiles listed in the .edaxh5 that have no pattern file: skip them and report them. Do not fail.

## .edaxh5 hierarchy

Every group has a `GroupType` attribute (int32), and that attribute sets the level:

```
/GMAW 304L 309                      GroupType=0  Project   (attrs: Name, Id, PRODUCT_VERSION, User*)
  /Stitch                           GroupType=1  Specimen  (attrs: Name, Id, Thickness...)
    /Area 1                         GroupType=2  Area      (attrs: Name, Id, NoOfSlices)
      FOVIMAGE (400x512 u8)              overview SE image of the area
      FOVIMAGECOLLECTIONPARAMS, FOVIPR   overview image params (mmFieldWidth/Height = 1.065 x 0.835 mm at 200x)
      HOSTPARAMS                         stage/SEM state at area origin
      /OIM Map 1_r000c000           GroupType=3  Map (one per tile)
      /OIM Map 1_r000c001
      ...
      /OIM Map 1_r000c011
```

For discovery, walk the file and collect every group with `GroupType == 3`. Do not hard-code
`Stitch`/`Area 1`: a project can hold several specimens and areas.

### Per-map (tile) group contents (paths relative to the map group)

| Path | Meaning | This dataset |
|---|---|---|
| attr `Id` | **Map GUID**. Key that links to the .up1 | e.g. `4aba8a87-6255-4ff5-96ee-4f9b5c4f9010` |
| attr `Name` | `OIM Map 1_rRRRcCCC`, the tile's row/col in the montage | `_r000c000` ... `_r000c011` |
| attr `TimeStamp`, `ScanEndTime` | .NET ticks (100 ns since 0001-01-01) | ~112 s apart per tile |
| attr `IsScanSuccessful` | **0 for every tile even though the data is fine.** Don't filter on it | 0 |
| `Sample/Number Of Columns`, `Number Of Rows` | Grid size | 414 x 409 (**last tile c011: 301 x 409**) |
| `Sample/Step X`, `Step Y` | um | 2.0, 2.0 |
| `Sample/Grid Type` | | `SqrGrid` |
| `Sample/Sample Tilt` | deg | 69.9 |
| `Stage/X`, `Stage/Y`, `Stage/Z` | **Per-tile stage position, mm.** Use these | X goes from -21.408 to -13.1415 in **+0.7515 mm** steps; Y = -40.253; Z = 28.02 |
| `HOSTPARAMS` (compound) | **Stale.** Every tile shows tile c000's stage XY. Don't use it for position | |
| `EBSD/ANG/HEADER/Pattern Center Calibration/X-Star, Y-Star, Z-Star` | EDAX PC (x*, y*, z*) | 0.52648, 0.17346, 0.69487. **Same for all tiles** |
| `.../XStarAdjustCoeffs` etc. | PC beam-shift correction coeffs | all zeros |
| `EBSD/ANG/HEADER/Phase/<n>/...` | Phases: name, lattice, Laue/point group, HKL families | 1 phase: Fe (gamma, austenite, fcc), a = 3.65 Å, m3m |
| `EBSD/ANG/DATA/DATA` (compound, N rows) | Hough indexing: `Orientations` (3x3 matrix, 9 f8), IQ, CI, Phase, SEM Signal, Fit, PRIAS strips, Valid, Custom | N = rows x cols, row-major |
| `EBSD/OHP/HEADER/*` | Hough settings (Circle Size 96, bands 3 to 8, ...) | |
| `Camera/*` | Elevation 8°, azimuth 0°, model VelocityGen1 | |
| `Column/*` | kV 20, mag 200, WD 28.02 | |
| `SEM-PRIAS Images/DATA/SEM` | Per-tile SE image 400x512 | |

**Montage geometry (useful to write into outputs; not needed for indexing):**
- Single row of 12 tiles. Stage X increases with column index. Stage Y is constant.
- Tile pitch is 751.5 um, which is 0.9 x the 835 um FOV height, so the scan was set up with
  10% overlap. One tile scan is 414 x 2 = 828 um wide, so neighboring tiles overlap by
  **76.5 um (about 38 columns)**.
- The last tile is cut to 301 columns. Total width is 11 x 751.5 + 602 = 8868.5 um.
- All tiles share one PC, phase list, step, tilt, kV, WD and camera setup. In this dataset one
  set of indexing parameters is valid for the whole montage. A batch run should still read
  them per tile and warn if they differ.

## .up1 file format (header version 4)

Little-endian. The data offset is stored in the header. Here it is **58 bytes**.

| Byte offset | Type | Field | Value (c000) |
|---|---|---|---|
| 0 | uint32 | version | 4 |
| 4 | uint32 | pattern width (px) | 115 |
| 8 | uint32 | pattern height (px) | 115 |
| 12 | uint32 | **data offset** | 58 |
| 16 | uint8 | extra-patterns flag | 0 |
| 17 | uint32 | n_cols | 414 |
| 21 | uint32 | n_rows | 409 |
| 25 | uint8 | hex grid flag | 0 |
| 26 | float64 | step x (um) | 2.0 |
| 34 | float64 | step y (um) | 2.0 |
| **42** | **16 bytes** | **Map GUID in .NET byte order** (new in v4; v3 headers end at 42) | `4aba8a87-...-4f9b5c4f9010` |
| 58 | uint8[N][h][w] | patterns, row-major (x fastest), N = n_rows x n_cols | |

Check: 58 + 169326 x 115 x 115 = 2,239,336,408 bytes, the exact file size. (`.up2` uses the same layout with uint16 pixels.)

Decode the GUID with `uuid.UUID(bytes_le=hdr[42:58])`. Data1, Data2 and Data3 are
little-endian. Verified on all 3 files:

| up1 file | GUID in header | Matches .edaxh5 map |
|---|---|---|
| `..._r000c000.up1` | 4aba8a87-6255-4ff5-96ee-4f9b5c4f9010 | `OIM Map 1_r000c000` attr `Id` ✓ |
| `..._r000c001.up1` | ad0713cf-916d-4cec-ad6e-2c4a278ff23f | `OIM Map 1_r000c001` attr `Id` ✓ |
| `..._r000c002.up1` | b30e50ed-1736-4417-8f88-7e46fb592878 | `OIM Map 1_r000c002` attr `Id` ✓ |

The .edaxh5 has **no reference to .up1 file names or paths** (checked with a byte search).
The link runs one way: the .up1 header points to the map GUID.

## Linking rules (most to least reliable)

1. **GUID (v4+ headers):** header bytes 42 to 58 equal the map group's `Id` attribute.
2. **Filename convention:** `{Project}_{Specimen}_{Area}_{MapName}.up1`, which here is
   `GMAW 304L 309` + `_Stitch` + `_Area 1` + `_OIM Map 1_r000c000`. Parse the tile position with
   `_r(\d+)c(\d+)\.up[12]$`. Use this only as a fallback for v3 headers with no GUID. Names
   contain spaces and underscores, so match against map names you already know rather than
   splitting on `_`.
3. **Always check after matching:** header n_cols/n_rows/step must equal `Sample/*` of the
   matched map, and file size must equal offset + n_cols x n_rows x w x h x bytes-per-pixel.
   That catches truncated files and a wrong pairing.

## What batch processing/indexing needs per tile

From the .up1: width, height, offset, n_cols, n_rows, step, GUID.
From the matched .edaxh5 map: PC (x*, y*, z*), sample tilt, camera elevation/azimuth, kV,
phase list, step, grid size, stage X/Y (pass these to the output for the external stitcher),
and the tile name `rRRRcCCC`.

Suggested flow:
1. User picks a .edaxh5 (or a folder). List all GroupType=3 maps.
2. Scan the folder for `*.up1` / `*.up2`, read headers, and match each file by GUID to a map.
3. Show a table of tiles: name, row/col, grid size, pattern file found or missing, PC.
   Tiles with no pattern file are greyed out.
4. Run processing, then spherical indexing, on each tile in turn with shared settings. Pick up
   per-tile PC/geometry from the .edaxh5. Don't recompute anything shared (master pattern, SHT
   tables) for each tile.
5. Write one output per tile, named after the source map, e.g.
   `GMAW 304L 309_Stitch_Area 1_OIM Map 1_r000c000_<suffix>.up1` / `.ang`. Keep the tile's
   grid and step unchanged. Put stage X/Y and row/col in the header/comments so stitching
   software can place the tile.

Edge cases this dataset already shows: a missing pattern file (9 of 12 tiles), an edge tile
with a different grid size (c011 is 301 columns), `IsScanSuccessful=0` on good data, stale
`HOSTPARAMS`, and 2.2 GB per tile (stream or memory-map; never load the whole montage).
