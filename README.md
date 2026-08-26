# DSSim
Double star simulator

# Double Star Simulator
 
A command-line tool that takes a real single-star **SER** or **FITS** image cube and generates a synthetic **double-star** sequence by superimposing each frame with a sub-pixel-shifted copy of itself.
 
This is useful for testing and validating double-star measurement / astrometry pipelines (e.g. speckle reduction, PSF fitting, `rho`/`theta` measurement tools) against a known, ground-truth separation and position angle.
 
## How it works
 
For every frame `F` in the input sequence, the tool produces:
 
```
F_out = F + alpha * shift(F, rho, theta)
```
 
- **`rho`** — the separation between the two stars, in pixels (sub-pixel precision supported)
- **`theta`** — the position angle in degrees, measured counter-clockwise from the positive X axis
- **`alpha`** — the flux ratio of the secondary star relative to the primary (`1.0` = equal brightness, `0.5` = secondary is half as bright, etc.)
The shift is performed with sub-pixel accuracy using cubic spline interpolation (`scipy.ndimage.shift`), and the result is clipped back to the original bit depth so the output looks like a real capture.
 
## Features
 
- Reads and writes both **SER** (`.ser`) and **FITS** (`.fits`, `.fit`, `.fts`) formats
- Auto-detects the input format from the file extension or magic bytes
- Supports 8-bit and 16-bit mono data
- Preserves SER metadata (observer, instrument, telescope, color ID, pixel depth) from the input file when writing SER output
- Sub-pixel shifting via spline interpolation, with a simple integer-roll fallback if SciPy isn't available
- Converts freely between formats — feed in a SER file and get FITS out, or vice versa
## Requirements
 
- Python 3.7+
- [NumPy](https://numpy.org/)
- [Astropy](https://www.astropy.org/) — required for FITS reading/writing
- [SciPy](https://scipy.org/) — required for sub-pixel (spline) shifting; without it, shifts are rounded to the nearest whole pixel
Install dependencies:
 
```bash
pip install numpy astropy scipy
```
 
## Usage
 
```bash
python doublestar_simulator.py <input> --rho <pixels> --theta <degrees> --alpha <ratio> --output <output>
```
 
### Arguments
 
| Flag | Short | Required | Description |
|---|---|---|---|
| `input` | | Yes | Input SER or FITS file containing a single star |
| `--rho` | `-r` | Yes | Separation in pixels (float, supports sub-pixel values) |
| `--theta` | `-t` | Yes | Position angle in degrees, CCW from the positive X axis |
| `--alpha` | `-a` | No | Flux ratio of the secondary star (default: `1.0`) |
| `--output` | `-o` | Yes | Output file path (`.ser` or `.fits`/`.fit`) |
 
### Examples
 
Create a synthetic double star with 5.3 px separation at PA 45°, secondary at 70% the brightness of the primary:
 
```bash
python doublestar_simulator.py input.ser --rho 5.3 --theta 45 --alpha 0.7 --output output.fits
```
 
Convert a FITS cube into an equal-brightness pair at 3.0 px / PA 120° and save back to SER:
 
```bash
python doublestar_simulator.py input.fits --rho 3.0 --theta 120 --alpha 1.0 --output output.ser
```
 
## Notes on conventions
 
- **Separation (`rho`)** is defined in pixels, not arcseconds — convert using your setup's plate scale if you need a specific real-world separation.
- **Position angle (`theta`)** is measured CCW from the +X (column) axis in pixel space, *not* from celestial north — keep this in mind if you're benchmarking against catalog PA values, which conventionally use north through east.
- Output pixel values are clipped to the input's dynamic range (e.g. 0–255 for 8-bit, 0–65535 for 16-bit) to avoid overflow when the two stars overlap.
## Output
 
The script prints a short summary as it runs, including the computed shift vector, frame count, and dimensions, and confirms the separation, angle, flux ratio, and frame count once the output file has been written.
 

