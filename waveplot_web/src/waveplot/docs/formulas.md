# Formulas and processing pipelines

Every history trace is defined by an **expression**. The simplest is a history number:

    h12          history 12
    self         the trace slot's own history (used by legacy formulas)

Operations taking one input are chained with `|` (read "then"):

    h2 | integrate | bandpass(f_lo=0.5, f_hi=10)
    h7 | window(units='absolute', t1=1.0, t2=3.5, taper=0.2) | spectrum(kind='amp')

Arithmetic uses normal precedence: `(h3 + h4) / 2`, `h1 * (h2 + h3)`, `-h5`, `h2 ** 2`.
Operations with several inputs are called like functions: `rotate(h5, h6, theta=30)`,
`vector_magnitude(h=4)` (histories 4, 5, 6 as x, y, z).

Binary operations on traces with different sampling interpolate the second trace onto the first
(Standard mode) — v407 combined them sample by sample and took the second trace's time axis.

## Legacy v407 formulas

The original formula language is still accepted (and `form.N` files can be loaded). WavePlot shows
the equivalent modern expression. Important v407 rules:

* the formula starts from the slot's own history; `[n]` loads history *n*;
* operators apply to everything before them; parameters follow as `,value` (at most 5);
* a binary operator (`+ - * / C R r`, `$…`) takes **the whole rest** of the formula as its right
  operand: `[1]*[2]+[3]` means `h1 * (h2 + h3)`.

| v407 | modern | meaning |
|---|---|---|
| `[2]I` | `h2 \| integrate` | integral |
| `[2]D` | `h2 \| differentiate` | derivative |
| `[2]M,3` / `[2]#,1` | `h2 \| scale(c=3)` / `h2 \| offset(c=1)` | multiply / add |
| `[2]F` / `[2]P` | `h2 \| spectrum(kind='amp')` / `kind='phase'` | Fourier amplitude / phase |
| `[2]f,5,50` | `spectrum(kind='amp', filter='stop', f1=5, f2=50)` | spectrum after an FFT stop band (% of Nyquist) |
| `[2]t,0,10` | `spectrum(kind='inverse', filter='stop', f1=0, f2=10)` | waveform after an FFT stop band |
| `[2]&F,1,10` | `spectrum(kind='inverse', filter='pass', f1=1, f2=10)` | waveform after an FFT pass band (Hz) |
| `[2]&f,1,10,4,1` | `bandpass(f_lo=1, f_hi=10, sections=4, zero_phase=1)` | Butterworth band-pass |
| `[2]J,1,0.5,2,0.1` | `window(units='absolute', t1=0.5, t2=2, taper=0.1)` | extract with Gaussian tapers |
| `[2]&D,4` | `decimate(q=4)` | resample |
| `[2]&q,50,3000,1000` | `attenuate(Q=50, c=3000, L=1000)` | constant-Q attenuation |
| `&V,4` | `vector_magnitude(h=4)` | √(x²+y²+z²) of h4..h6 |
| `&R,0,4,250,50,250` | `rotate_to_source(component=0, h=4, i=250, j=50, k=250)` | radial component |
| `[2]&t,0,,5` | `tau_p(kind=0, ds=0, window=5)` | predominant period τp |
| `[1]C,30[2]` | `rotate(h1, h2, theta=30)` | x cosθ + y sinθ |
| `[1]R[2]` / `[1]r[2]` | `hypot(h1, h2)` / `angle(h1, h2)` | magnitude / direction |
| `G([2]F)([2]P)` | `from_spectrum(…, …)` | waveform from amplitude and phase |

The phase/group-velocity functions `V W Z A B` were disabled in v407 and are reported as such.
