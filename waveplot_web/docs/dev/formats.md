# WAVE output file formats (as read by WavePlot)

Verified against two real datasets (`Waveplot_exe/mt_slip5`, `cg11`) and transcribed from the
v407 Pascal sources (`DATA_RSC.PAS`, `DSPL.PAS`, `DSPSNAP.PAS`, `DSPGEOM.PAS`). Items marked
**unverified** have no sample file and are implemented from the Pascal only.

## Global conventions

* All floating-point values are IEEE-754 **float32, little-endian** (`<f4`). No file uses the
  Turbo Pascal 6-byte `Real`.
* Integers are int32 little-endian. Records are byte-packed (floats may be unaligned).
* WAVE writes Fortran-style unformatted records by hand: every record is
  `[int32 nbytes][payload][int32 nbytes]`.
* Offsets into `.snp/.hst/.dmp` count float32 values; offsets into `.geo/.crk` count bytes. All
  start at 0 and accumulate per file type in `.map` order.

## `.map` — index

128-byte records: `[i4 120][i4 PlType][116-byte body][i4 120]`.

| PlType | Kind | `struct` format of `rec[8:8+N]` | N | Fields |
|---|---|---|---|---|
| 0 | snapshot | `<iff6i2c2f2i2i2f2i` | 78 | var_raw, t1, time, i1, i2, j1, j2, k1, k2, ax1, ax2, dx, dy, sx, sy, xqty, yqty, max, min, gnum, gid |
| 1 | history | `<iff6icfii2f2i` | 65 | var_raw, t1, t2, i1…k2, ax1, dx, sx, xqty, max, min, gnum, gid |
| 2 | dump | `<f6i2c2f2i2ii2if2i` | 78 | time, i1…k2, ax1, ax2, dx, dy, sx, sy, iqty, jqty, vqty, gnum, gid, dz, sz, kqty |
| 3 | geometry | `<2i4ifi6i3fi` | 72 | gnum, gid, proptot, mattot, sourcetot, stopetot, time, ntot, i1…k2, dx, dy, dz, grlen |
| 4 | crack dump | `<f2i2i3ii` | 36 | time, ic1, ic2, gnum, gid, stqty, tqty, vqty, crlen |

* `var = var_raw mod 65535`, `func = var_raw div 65535` (v407 uses `$FFFF`; configurable via
  `func_base`, unverified because no sample uses a function code). Function names:
  `'' MX mn Avg Abs AMX`.
* Offsets: snapshot `+= xqty·(yqty+2)`; history `+= 3·xqty`; dump `+= iqty·jqty·kqty·vqty + 2`
  (if `kqty == 0` then `kqty = 1`); geometry `+= grlen` bytes (if `grlen == 0` then
  `(prop+mat+src+stope)·128`); crack `+= crlen` bytes.
* Leftover body bytes are ASCII `z` padding.

## `.snp` — snapshots

Snapshot = `xqty` Fortran records, one per index along `ax1`, each holding `yqty` values along
`ax2`: `np.memmap(f, '<f4', offset=4*off, shape=(xqty, yqty+2))[:, 1:-1]`.

* Axes are `i`, `j`, `k` (grid) or `t` (time; **unverified**). The collapsed grid index is the
  plane position (e.g. `k1 == k2 == 61` → i-j plane at k = 61).
* Grid index of row `r`: `lo + ((hi-lo) mod s) div 2 + r·s` with stride `s = sx`.
* Physical coordinate: zone `i` spans `[(i-1)·dx, i·dx]`; values sit at the zone centre, shifted
  half a zone along staggered axes (`VarPositions` bitmask: XVEL 1, YVEL 2, ZVEL 4, S12 3, S31 5,
  S23 6; stope codes 8/9/10 resolved per plane).
* Display: `ax1` horizontal, `ax2` vertical; a vertical `j` axis points up, others down.

## `.hst` — histories

Each sample is its own Fortran record: `memmap('<f4', offset=4*off, shape=(xqty, 3))[:, 1]`.
If `ax1` is `i/j/k` the history is a spatial profile with `x0 = (lo-1)·dx`, `x1 = (hi-1)·dx`;
otherwise `x0 = t1`, `x1 = t2`. Always `dt = (x1-x0)/(n-1)` — the header `dx` is the *model*
timestep, not the sampling interval. Position `(i1, j1, k1)`; global quantities (MaxVel) have
`i = j = k = 0`.

## `.dmp` — dumps

One Fortran record per dump; data at float offset `off + 1`, shape `(X, Y, vqty)` with the
component fastest. Plane axes come from the counts, not from the header `ax1/ax2`:
`X = k if (kqty > 1 and jqty > 1) else i`, `Y = j if jqty > 1 else k`.

* `vqty = 9` (3-D): `v1 v2 v3 s11 s22 s33 s12 s23 s13`; `vqty = 5` (2-D, **unverified**):
  `v1 v2 s11 s22 s12`.
* In-plane selection: i-j → (v1, v2, s11, s22, s12); i-k → (v1, v3, s11, s33, s13);
  k-j → (v3, v2, s33, s22, s23).

## `.geo` — geometry

Byte-addressed 128-byte Fortran records `[i4 120][120-byte payload][i4 120]` starting at the
geometry offset:

1. state blocks (first payload word `0x01FEDCB1`; words 7–12 = grid node extents),
2. boundary blocks (`0x01FEDCB2`; first one holds index ranges + physical size, second one
   boundary stresses `s11 s22 s33 s12 s23 s13`),
3. `proptot` × `K, G, ρ` (float32),
4. `mattot` × `matnum, i1, i2, j1, j2, k1, k2` (zone ranges),
5. `sourcetot` × `stype, i1…k2` (+ wave parameters; `stype` = WAVE variable number of the
   driven component), stored in reverse order,
6. `stopetot` × `stopID, stype, i1…k2` (+ `kn ks … friction cohesion tan(friction) …`),
   `LayoutSz` at payload byte 116, then an optional byte layout read from the next block
   start + 4 and padded to 128 bytes (**unverified**), stored in reverse order.

`Vp = sqrt((K + 4G/3)/ρ)`, `Vs = sqrt(G/ρ)`.

## `.crk` — crack dumps (**unverified**)

Per crack: a 128-byte header block (at +4: `stid iqty i1 i2 j1 j2 k1 k2`), then `iqty` element
records of `vqty + 2` floats. Element fields: `vNs1 vNs0 vT1s1 vT1s0 uNs1 uNs0 uT1s1 uT1s0 sNs1
sT1s1 sT1s0 sT1sh T1slp Taslp Nbnd Shbnd`.

## Text side files

* `.m` — WAVE input script (`gr`, `prop`, `mat`, `so`, `stope`, `his`, `pl`, `cy`, …).
* `.log` — run log (`(Min.) timestep = …`, `Cycle N, time = T`).
* `.otl` — stope outline: point count, then `x y` integer pairs.
* `form.N` — v407 formula for graph slot N (first line).
