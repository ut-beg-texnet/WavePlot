# WavePlot v407 behaviour: findings and how the port handles them

All references are to `Turbo pascal version/`. "Reproduced" means available in `mode="v407"`
(with a provenance quirk key); standard mode uses the corrected behaviour.

## Numerics (reproduced in v407 mode)

| quirk key | v407 behaviour | source |
|---|---|---|
| `ess_shear_sign` | ESS shear term +½(s11−s22)·sin2θ: shear and normal stress on different planes | DATA_RSC.PAS:1171-1175 |
| `fft_sign` | forward FFT uses e^{+i} (phase sign opposite to NumPy) | FOURIER.PAS:118-165 |
| `arctan0` | atan2(0,0) = −π/2 → empty spectral bins show 270° | FUNCT.PAS:113-130 |
| `phase_monotone` | phase forced to increase monotonically | DSPFORM.PAS:350-361 |
| `fft_cap` | FFT length capped at 4096 | FOURIER.PAS:8 |
| `taper_offset` | spectral taper indices off by one sample | DSPFORM.PAS:302-324 |
| `mirror_bin` | FFT filters / Q modify bin N−1−i instead of N−i | FOURIER.PAS:182-224, DSPFORM.PAS:245-253 |
| `butter_state` | zero-phase Butterworth continues the forward-pass state in the backward pass | FILTERS.PAS:90-110 |
| `integrate_left_rect` | integration y[t] = y[t−1] + x[t−1]·dt | DSPFORM.PAS:753-767 |
| `diff_forward` | forward-difference derivative, last value repeated | DSPFORM.PAS:768-782 |
| `decimate_no_aa` | `&D` keeps samples q, 2q, … without anti-alias filtering | DSPFORM.PAS:1084-1098 |
| `pow_sign` | `POW` keeps the sign only for odd round(p) or odd round(1/p) | FUNCT.PAS:393-419 |
| `accel_sign` | acceleration trajectories (v_prev − v)/dt | DATA_RSC.PAS:1698 |
| `level_shrink` | colour-level range ×(Q−0.5)/Q | DSPSNAP.PAS:71-77 |
| `cosine_levels` | cosine distribution ½(1−cos πx)⁴ reaches 8× the range | DSPSNAP.PAS:97-98 |
| `zero_level` / `band_toward_zero` | level nearest 0 forced to 0 (background); bands round toward zero | DSPSNAP.PAS:102-205 |
| `dump_sampling` | glyph grid ROUND((p−0.49)·ratio) | DATA_RSC.PAS:1141 |
| `index_rotation` | `&R` rotation uses grid indices instead of metres | DSPFORM.PAS:1171-1172 |
| `binary_right_time` | binary operations are index-wise and take the right operand's axis | DSPFORM.PAS:856-906 |
| `hist_truncate` | histories truncated to 4096 samples | DATA_RSC.PAS:729-730 |
| `v407_contours` | 4-triangle centre-point contour segments | DSPSNAP.PAS:232-366 |

## Formula language semantics (always reproduced by the legacy parser)

* The accumulator starts with the slot's history (`self`, DSPSEISM.PAS:265-269).
* Binary operators parse the rest of the string as their right operand (right-recursive, no
  precedence): `[1]*[2]+[3]` = `h1*(h2+h3)`.
* Parameters: up to 5, `Params[k] = 0` when not given; the number scanner accepts `e/E`, so `M,2E`
  fails to parse (now reported).
* `J`/`&W` with 5 parameters takes the decimation factor from the taper value (`Params[4]`).
* `V W Z A B` are no-ops (body commented out); `$H` does nothing but consumes the rest.

## Corrected silently (not reproduced)

* Capacity limits (250 snapshots, 510 histories, 50 dumps — cg11 has 61 —, 4096 samples, 101×101
  display grid, 24 contours, 8 traces, 6×6 panels).
* Principal-stress glyph direction: v407 stored the mirrored angle and drew at −Ang on a flipped
  screen axis; the port computes the physical direction directly (verified as an eigenvector).
* Uninitialised buffers (`$` operators, right operands without `[n]`) are zeros (v407 mode, with a
  warning) or an error (standard mode).
* Stale loop indices in `DI_ReadGeom` (`GeomMap^[i]`, `Stopes^[i].Outline`).
* `.ASC` export of traces with different lengths (standard: padded columns; v407 mode: trace 1's
  length, as before).
* UI defects: placeholder help pages, misleading hint texts, `<` driving the level count to 0,
  Esc not leaving the main menu, zoom persisted across sessions.

## Unverified formats (no sample files)

`.crk` crack dumps, 2-D (5-component) dumps, time-axis (`t`) snapshots, spatial line histories,
function codes (MX/mn/Avg/Abs/AMX packing base 65535 vs 65536) and stope layouts are implemented
from the Pascal and tested with synthetic files only.
