#!/usr/bin/env python3
"""
Double Star Simulator
=====================
Takes a real single-star SER or FITS cube and produces a synthetic double-star
sequence by superimposing each frame with a shifted (sub-pixel) copy.

Usage:
    python doublestar_simulator.py input.ser --rho 5.3 --theta 45 --alpha 0.7 --output output.fits
    python doublestar_simulator.py input.fits --rho 3.0 --theta 120 --alpha 1.0 --output output.ser

Parameters:
    --rho    : separation in pixels (float, sub-pixel precision)
    --theta  : angle in degrees, measured from X axis counter-clockwise
    --alpha  : flux ratio of the secondary (1.0 = equal brightness, 0.5 = half)
    --output : output file (.ser or .fits)
"""

import argparse
import os
import struct
import numpy as np
from datetime import datetime, timezone

try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    from scipy.ndimage import shift as scipy_shift
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# SER FORMAT CONSTANTS (standard SER spec)
# =============================================================================
SER_HEADER_SIZE = 178
SER_MAGIC = b"LUCAM-RECORDER"

# Color IDs
MONO       = 0
BAYER_RGGB = 8
BAYER_GRBG = 9
BAYER_GBRG = 10
BAYER_BGGR = 11
RGB        = 100
BGR        = 101


# =============================================================================
# SER READER
# =============================================================================
class SERFile:
    """Read and write SER files following the standard spec (178-byte header)."""

    def __init__(self):
        self.file_id       = "LUCAM-RECORDER"
        self.lu_id         = 0
        self.color_id      = MONO
        self.little_endian = 0          # 0 = big-endian for >8bit, but we store as written
        self.image_width   = 0
        self.image_height  = 0
        self.pixel_depth   = 8          # bits per pixel per plane
        self.frame_count   = 0
        self.observer      = ""
        self.instrument    = ""
        self.telescope     = ""
        self.date_time     = 0          # Windows FILETIME (100-ns ticks since 1601)
        self.date_time_utc = 0
        self.frames        = []         # list of 2-D numpy arrays

    # ------------------------------------------------------------------
    @staticmethod
    def _str40(s):
        """Encode a string into 40 bytes (padded / truncated)."""
        b = s.encode("ascii", errors="replace")
        return b[:40].ljust(40, b"\x00")

    @staticmethod
    def _decode40(b):
        return b.rstrip(b"\x00").decode("ascii", errors="replace")

    @staticmethod
    def _windows_filetime_now():
        """Return current UTC time as Windows FILETIME (100-ns ticks since 1601-01-01)."""
        epoch_1601 = datetime(1601, 1, 1, tzinfo=timezone.utc)
        now        = datetime.now(tz=timezone.utc)
        delta      = now - epoch_1601
        return int(delta.total_seconds() * 1e7)

    # ------------------------------------------------------------------
    def read(self, path):
        """Load a SER file from disk."""
        with open(path, "rb") as f:
            raw = f.read()

        if len(raw) < SER_HEADER_SIZE:
            raise ValueError(f"File too small to be a valid SER: {path}")

        # --- parse header -------------------------------------------------
        file_id        = raw[0:14]
        if file_id != SER_MAGIC:
            raise ValueError(f"Not a valid SER file (bad magic): {file_id}")

        (lu_id,
         color_id,
         little_endian,
         image_width,
         image_height,
         pixel_depth,
         frame_count)  = struct.unpack_from("<iiiiiii", raw, 14)

        observer        = self._decode40(raw[42:82])
        instrument      = self._decode40(raw[82:122])
        telescope       = self._decode40(raw[122:162])
        date_time       = struct.unpack_from("<q", raw, 162)[0]
        date_time_utc   = struct.unpack_from("<q", raw, 170)[0]

        self.lu_id         = lu_id
        self.color_id      = color_id
        self.little_endian = little_endian
        self.image_width   = image_width
        self.image_height  = image_height
        self.pixel_depth   = pixel_depth
        self.frame_count   = frame_count
        self.observer      = observer
        self.instrument    = instrument
        self.telescope     = telescope
        self.date_time     = date_time
        self.date_time_utc = date_time_utc

        # --- determine numpy dtype ----------------------------------------
        bytes_per_pixel = 1 if pixel_depth <= 8 else 2
        dt = np.uint8 if bytes_per_pixel == 1 else np.uint16

        # Number of planes (1 for mono, 3 for colour)
        if color_id in (RGB, BGR):
            planes = 3
        elif color_id in (BAYER_RGGB, BAYER_GRBG, BAYER_GBRG, BAYER_BGGR):
            planes = 1
        else:
            planes = 1

        frame_bytes = image_width * image_height * planes * bytes_per_pixel

        # --- read frames --------------------------------------------------
        offset = SER_HEADER_SIZE
        self.frames = []
        for i in range(frame_count):
            chunk = raw[offset: offset + frame_bytes]
            if len(chunk) < frame_bytes:
                print(f"Warning: truncated file, got {i} frames instead of {frame_count}")
                break
            arr = np.frombuffer(chunk, dtype=dt).reshape(image_height, image_width)
            # Handle endianness for 16-bit
            if bytes_per_pixel == 2:
                if little_endian == 0:
                    arr = arr.byteswap()
                arr = arr.astype(np.uint16)
            self.frames.append(arr.copy())
            offset += frame_bytes

        self.frame_count = len(self.frames)
        return self

    # ------------------------------------------------------------------
    def write(self, path):
        """Save frames to a SER file."""
        if not self.frames:
            raise ValueError("No frames to write.")

        frame_count   = len(self.frames)
        sample        = self.frames[0]
        image_height, image_width = sample.shape[:2]
        pixel_depth   = self.pixel_depth if self.pixel_depth else (8 if sample.dtype == np.uint8 else 16)
        bytes_per_pixel = 1 if pixel_depth <= 8 else 2
        dt            = np.uint8 if bytes_per_pixel == 1 else np.uint16
        little_endian = 1          # we always write little-endian 16-bit

        ts = self._windows_filetime_now()

        header = struct.pack(
            "<14siiiiiii40s40s40sqq",
            SER_MAGIC,
            self.lu_id,
            self.color_id,
            little_endian,
            image_width,
            image_height,
            pixel_depth,
            frame_count,
            self._str40(self.observer),
            self._str40(self.instrument),
            self._str40(self.telescope),
            ts,
            ts,
        )
        assert len(header) == SER_HEADER_SIZE, f"Header size mismatch: {len(header)}"

        with open(path, "wb") as f:
            f.write(header)
            for frame in self.frames:
                arr = frame.astype(dt)
                if bytes_per_pixel == 2:
                    arr = arr.astype("<u2")   # ensure little-endian
                f.write(arr.tobytes())

        print(f"SER written: {path}  ({frame_count} frames, {image_width}x{image_height}, {pixel_depth}-bit)")


# =============================================================================
# FITS READER / WRITER
# =============================================================================
def read_fits(path):
    """Return list of 2-D numpy arrays from a FITS cube (first image HDU)."""
    if not ASTROPY_AVAILABLE:
        raise ImportError("astropy is required to read FITS files. Install with: pip install astropy")
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                data = hdu.data
                if data.ndim == 2:
                    return [data.copy()]
                elif data.ndim == 3:
                    return [data[i].copy() for i in range(data.shape[0])]
    raise ValueError(f"No image data found in FITS file: {path}")


def write_fits(path, frames, original_header=None):
    """Write list of 2-D numpy arrays as a FITS cube."""
    if not ASTROPY_AVAILABLE:
        raise ImportError("astropy is required to write FITS files. Install with: pip install astropy")
    cube = np.stack(frames, axis=0)
    hdu  = fits.PrimaryHDU(data=cube)
    if original_header:
        for key, val in original_header.items():
            try:
                hdu.header[key] = val
            except Exception:
                pass
    hdu.header["NAXIS"]  = 3
    hdu.header["HISTORY"] = "Generated by doublestar_simulator.py"
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=True)
    print(f"FITS written: {path}  ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


# =============================================================================
# CORE: SUB-PIXEL SHIFT
# =============================================================================
def subpixel_shift(frame, dy, dx):
    """
    Shift a 2-D frame by (dy, dx) with sub-pixel accuracy using
    scipy.ndimage.shift (spline interpolation order=3).
    Falls back to numpy roll (integer only) if scipy is unavailable.
    """
    if SCIPY_AVAILABLE:
        return scipy_shift(frame.astype(np.float64), shift=[dy, dx], order=3, mode="reflect")
    else:
        # integer fallback
        idy, idx = int(round(dy)), int(round(dx))
        return np.roll(np.roll(frame, idy, axis=0), idx, axis=1).astype(np.float64)


# =============================================================================
# CORE: SIMULATE DOUBLE STAR
# =============================================================================
def simulate_double_star(frames, rho, theta_deg, alpha):
    """
    For each frame F, produce:
        F_out = F + alpha * shift(F, rho, theta)

    rho       : separation in pixels (float)
    theta_deg : angle in degrees, measured CCW from the positive X axis
    alpha     : flux ratio of the secondary component
    """
    theta_rad = np.deg2rad(theta_deg)
    dx =  rho * np.cos(theta_rad)   # column shift
    dy = -rho * np.sin(theta_rad)   # row shift (Y axis inverted in image coords)

    print(f"Shift vector: dx={dx:+.3f} px, dy={dy:+.3f} px  (rho={rho}, theta={theta_deg}°, alpha={alpha})")

    original_dtype = frames[0].dtype
    max_val = np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else 1.0

    result = []
    for i, frame in enumerate(frames):
        f      = frame.astype(np.float64)
        f_shift = subpixel_shift(f, dy, dx)
        f_out  = f + alpha * f_shift
        # Clip to original dynamic range and cast back
        f_out  = np.clip(f_out, 0, max_val)
        result.append(f_out.astype(original_dtype))

    return result


# =============================================================================
# AUTO-DETECT INPUT FORMAT
# =============================================================================
def detect_format(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ser":
        return "ser"
    elif ext in (".fits", ".fit", ".fts"):
        return "fits"
    else:
        # Peek at magic bytes
        with open(path, "rb") as f:
            magic = f.read(14)
        if magic == SER_MAGIC:
            return "ser"
        raise ValueError(f"Cannot determine file format for: {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Double Star Simulator — create a synthetic double-star sequence from a single-star cube.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("input",          help="Input SER or FITS file (single star)")
    parser.add_argument("--rho",   "-r",  type=float, required=True,
                        help="Separation in pixels (float, sub-pixel precision)")
    parser.add_argument("--theta", "-t",  type=float, required=True,
                        help="Angle in degrees, CCW from positive X axis")
    parser.add_argument("--alpha", "-a",  type=float, default=1.0,
                        help="Flux ratio of the secondary (default: 1.0 = equal brightness)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output file (.ser or .fits/.fit)")

    args = parser.parse_args()

    # --- sanity checks ----------------------------------------------------
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.rho <= 0:
        raise ValueError("--rho must be positive.")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive.")

    # --- read input -------------------------------------------------------
    in_fmt  = detect_format(args.input)
    out_fmt = detect_format(args.output) if os.path.splitext(args.output)[1].lower() in (".ser",".fits",".fit",".fts") \
              else os.path.splitext(args.output)[1].lower().lstrip(".")

    print(f"Reading {in_fmt.upper()}: {args.input}")

    ser_obj = None
    if in_fmt == "ser":
        ser_obj = SERFile().read(args.input)
        frames  = ser_obj.frames
        print(f"  {len(frames)} frames, {ser_obj.image_width}x{ser_obj.image_height}, {ser_obj.pixel_depth}-bit")
    else:
        frames = read_fits(args.input)
        print(f"  {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, dtype={frames[0].dtype}")

    # --- simulate ---------------------------------------------------------
    print(f"\nSimulating double star: rho={args.rho} px, theta={args.theta}°, alpha={args.alpha}")
    out_frames = simulate_double_star(frames, args.rho, args.theta, args.alpha)

    # --- write output -----------------------------------------------------
    out_ext = os.path.splitext(args.output)[1].lower()
    print(f"\nWriting output: {args.output}")

    if out_ext == ".ser":
        out_ser = SERFile()
        # Copy metadata from input if available
        if ser_obj:
            out_ser.lu_id        = ser_obj.lu_id
            out_ser.color_id     = ser_obj.color_id
            out_ser.pixel_depth  = ser_obj.pixel_depth
            out_ser.observer     = ser_obj.observer
            out_ser.instrument   = ser_obj.instrument
            out_ser.telescope    = ser_obj.telescope
        else:
            out_ser.pixel_depth  = 16 if out_frames[0].dtype == np.uint16 else 8
        out_ser.frames = out_frames
        out_ser.write(args.output)
    else:
        write_fits(args.output, out_frames)

    print("\nDone.")
    print(f"  Separation : {args.rho} px")
    print(f"  Angle      : {args.theta}°")
    print(f"  Flux ratio : {args.alpha}")
    print(f"  Frames     : {len(out_frames)}")


if __name__ == "__main__":
    main()
