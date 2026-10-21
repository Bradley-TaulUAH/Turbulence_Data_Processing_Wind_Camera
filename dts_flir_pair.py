import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import center_of_mass, gaussian_filter1d
from scipy.stats import gaussian_kde
import os
import json
import csv

try:
    import PySpin
    PYSPIN_AVAILABLE = True
except ImportError:
    PYSPIN_AVAILABLE = False
    print("[WARNING] PySpin not found. Live acquisition (record_flir_stack) disabled; "
          "analysis of already-saved .npy stacks still works.")

# SECTION 0: FLIR ACQUISITION (legacy path, run at the camera; not called by analysis)

def record_flir_stack(save_path, num_frames, exposure_us=None, gain_db=None,
                       camera_index=0):
    if not PYSPIN_AVAILABLE:
        raise ImportError("PySpin (Spinnaker SDK) required for FLIR acquisition")

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = cam_list.GetByIndex(camera_index)
    cam.Init()
    nodemap = cam.GetNodeMap()

    gamma_node = PySpin.CBooleanPtr(nodemap.GetNode('GammaEnable'))
    if PySpin.IsAvailable(gamma_node) and PySpin.IsWritable(gamma_node):
        gamma_node.SetValue(False)

    pf_node = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    pf_node.SetIntValue(pf_node.GetEntryByName('Mono12').GetValue())

    chunk_mode = PySpin.CBooleanPtr(nodemap.GetNode('ChunkModeActive'))
    chunk_mode.SetValue(True)
    chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode('ChunkSelector'))
    chunk_selector.SetIntValue(chunk_selector.GetEntryByName('Timestamp').GetValue())
    PySpin.CBooleanPtr(nodemap.GetNode('ChunkEnable')).SetValue(True)

    if exposure_us is not None:
        PySpin.CFloatPtr(nodemap.GetNode('ExposureTime')).SetValue(exposure_us)
    if gain_db is not None:
        PySpin.CFloatPtr(nodemap.GetNode('Gain')).SetValue(gain_db)

    cam.BeginAcquisition()
    frames = []
    timestamps_ns = []

    for i in range(num_frames):
        img = cam.GetNextImage(2000)
        if img.IsIncomplete():
            img.Release()
            continue
        frames.append(np.array(img.GetNDArray(), dtype=np.uint16))
        timestamps_ns.append(int(img.GetTimeStamp()))
        img.Release()
        if i % 500 == 0:
            print(f"[INFO] Captured {i}/{num_frames}")

    cam.EndAcquisition()

    exposure_actual = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime')).GetValue()
    gain_actual = PySpin.CFloatPtr(nodemap.GetNode('Gain')).GetValue()
    gamma_actual = gamma_node.GetValue() if PySpin.IsAvailable(gamma_node) else None

    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()

    frame_stack = np.stack(frames, axis=0)
    npy_path = save_path if save_path.endswith('.npy') else save_path + '.npy'
    np.save(npy_path, frame_stack)

    import time
    meta = {
        'timestamps_ns': timestamps_ns,
        'recording_start_utc': time.time(),
        'exposure_us': float(exposure_actual),
        'gain_db': float(gain_actual),
        'gamma_enabled': bool(gamma_actual) if gamma_actual is not None else None,
        'pixel_format': 'Mono12',
        'n_frames': len(frames),
    }
    base, _ = os.path.splitext(npy_path)
    with open(base + '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved {len(frames)} frames -> {npy_path}")
    print(f"[INFO] Gamma enabled at capture: {gamma_actual}  (MUST be False for valid SI)")
    return frame_stack, meta


# SECTION 1: FLIR FILE I/O

def select_file_dialog(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def get_experiment_folder():
    exp_name = input(
        "Enter experiment name/folder (or press Enter for 'scintillation_analysis'): "
    ).strip()
    if not exp_name:
        exp_name = "scintillation_analysis"
    dirname = exp_name.replace(" ", "_")
    os.makedirs(dirname, exist_ok=True)
    return dirname


def load_flir_stack(file_path):
    """
    Accepts EITHER a flir_acquire.py session JSON (flir_session_meta_*.json)
    OR a legacy record_flir_stack() .npy + _meta.json pair.
    """
    if file_path.lower().endswith('.json'):
        return _load_flir_session_json(file_path)
    return _load_flir_legacy_npy(file_path)


def _load_flir_session_json(session_json_path):
    with open(session_json_path, 'r') as f:
        session_meta = json.load(f)

    json_dir = os.path.dirname(os.path.abspath(session_json_path))

    def resolve(p):
        # Try the path as stored (relative to json_dir) first.
        candidate = p if os.path.isabs(p) else os.path.join(json_dir, p)
        if os.path.exists(candidate):
            return candidate

        # Fallback: the JSON may already live inside the same folder the
        # stored path's prefix refers to (e.g. json_dir is itself
        # ".../flir_captures" and frames_file is "flir_captures/x.npy"),
        # which doubles the folder name when joined directly. Try just the
        # basename in json_dir instead.
        basename_candidate = os.path.join(json_dir, os.path.basename(p))
        if os.path.exists(basename_candidate):
            print(f"[INFO] Resolved '{p}' as '{basename_candidate}' "
                  f"(direct join would have duplicated the folder name)")
            return basename_candidate

        raise FileNotFoundError(
            f"Could not locate '{p}' relative to session JSON at "
            f"'{session_json_path}'. Tried:\n  {candidate}\n  {basename_candidate}\n"
            f"Check that the .npy/.npz files are in the same folder as the "
            f"session JSON, or update 'frames_file'/'chunk_meta_file' in the "
            f"JSON to the correct relative or absolute path."
        )

    frames_path = resolve(session_meta['frames_file'])
    meta_path = resolve(session_meta['chunk_meta_file'])

    frame_stack = np.load(frames_path)
    frames = [f.astype(np.float32) for f in frame_stack]
    print(f"[INFO] FLIR stack loaded: {len(frames)} frames, shape {frames[0].shape}, "
          f"dtype was {frame_stack.dtype}, condition='{session_meta.get('condition')}'")

    if session_meta.get('pixel_format') != 'Mono16':
        print(f"[WARNING] Session pixel_format is '{session_meta.get('pixel_format')}', "
              f"not Mono16 -- sigma_I computed from this file may not be "
              f"physically valid if it isn't a linear detector response.")

    dts_us = None
    fps_measured = None
    recording_start_utc = None  # flir_acquire.py doesn't log this yet

    chunk = np.load(meta_path)
    timestamps_ns = chunk['timestamp_ns'].astype(np.float64)
    valid_ts = timestamps_ns[timestamps_ns > 0]

    if len(valid_ts) > 1:
        dts_us = valid_ts * 1e-3
        dt_sec = np.diff(dts_us) * 1e-6
        fps_measured = 1.0 / np.median(dt_sec)
        print(f"[INFO] Measured frame rate: {fps_measured:.2f} fps, "
              f"jitter std: {np.std(dt_sec)*1e6:.2f} us")
    else:
        print("[WARNING] No valid timestamps in chunk metadata -- timing "
              "diagnostics and DTS time-matching unavailable for this session.")

    return frames, fps_measured, dts_us, recording_start_utc


def _load_flir_legacy_npy(file_path):
    base, _ = os.path.splitext(file_path)
    meta_path = base + '_meta.json'

    frame_stack = np.load(file_path)
    frames = [f.astype(np.float32) for f in frame_stack]
    print(f"[INFO] FLIR stack loaded: {len(frames)} frames, shape {frames[0].shape}, "
          f"dtype was {frame_stack.dtype}")

    dts_us = None
    fps_measured = None
    recording_start_utc = None

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        gamma_flag = meta.get('gamma_enabled', None)
        if gamma_flag:
            print("[WARNING] Metadata says GammaEnable=True at capture time. "
                  "sigma_I computed from this file is NOT physically valid.")
        elif gamma_flag is None:
            print("[WARNING] No gamma flag in metadata - cannot confirm linearity.")

        recording_start_utc = meta.get('recording_start_utc', None)

        ts_ns = np.array(meta['timestamps_ns'], dtype=np.float64)
        dts_us = ts_ns * 1e-3
        dt_sec = np.diff(dts_us * 1e-6)
        if len(dt_sec) > 0:
            fps_measured = 1.0 / np.median(dt_sec)
            print(f"[INFO] Measured frame rate: {fps_measured:.2f} fps, "
                  f"jitter std: {np.std(dt_sec)*1e6:.2f} us")
    else:
        print(f"[WARNING] No metadata file at {meta_path}. "
              "Timing diagnostics, real-time axis, and DTS time-matching unavailable.")

    return frames, fps_measured, dts_us, recording_start_utc


def select_flir_session():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="Select FLIR session JSON (flir_session_meta_*.json)",
        filetypes=[("FLIR session", "flir_session_meta_*.json"),
                   ("JSON files", "*.json"),
                   ("FLIR stack (legacy .npy)", "*.npy"),
                   ("All files", "*.*")]
    )
    root.destroy()
    return path


# SECTION 1b: DARK-FRAME / FLAT-FIELD CALIBRATION

def build_camera_noise_floor(dark_stack):
    dark_stack = np.asarray(dark_stack, dtype=np.float64)
    mean_dn = dark_stack.mean()
    var_dn = dark_stack.var()
    si_floor = var_dn / (mean_dn**2 + 1e-9)
    print(f"[INFO] Camera noise floor: mean={mean_dn:.2f} DN, SI_floor={si_floor:.6f}")
    return si_floor


def build_flatfield(uniform_stack):
    ff = np.mean(np.asarray(uniform_stack, dtype=np.float64), axis=0)
    ff = ff / ff.max()
    ff[ff < 0.05] = 1.0
    return ff.astype(np.float32)


def apply_flatfield(frame, flatfield):
    return frame / flatfield


# SECTION 2: PREPROCESSING & ROI

def extract_roi_intensity(frame, roi_size=20):
    frame = frame.astype(np.float32)
    frame_u16 = np.clip(frame, 0, 65535).astype(np.uint16)
    frame_denoised = cv2.medianBlur(frame_u16, 3)
    threshold = np.percentile(frame_denoised, 90)
    mask = frame_denoised > threshold
    if np.any(mask):
        cy, cx = center_of_mass(mask)
    else:
        cy, cx = frame.shape[0] // 2, frame.shape[1] // 2

    y1 = int(cy) - roi_size
    y2 = int(cy) + roi_size
    x1 = int(cx) - roi_size
    x2 = int(cx) + roi_size
    roi = frame[max(0, y1):min(frame.shape[0], y2),
                max(0, x1):min(frame.shape[1], x2)]
    return float(np.sum(roi))


def extract_roi_intensity_and_centroid(frame, roi_size=20):
    """
    Same centroid/threshold logic as extract_roi_intensity, but also
    returns the per-frame (cx, cy) used to place the ROI. Used to check
    whether intensity-trace fluctuations are due to real illumination
    changes (centroid stays put, intensity still varies) or beam
    pointing/wander (centroid moves, and a FIXED-size ROI window clips
    a moving beam differently frame to frame, which inflates the
    intensity variance with a pointing artifact rather than true
    scintillation or absorption).
    """
    frame = frame.astype(np.float32)
    frame_u16 = np.clip(frame, 0, 65535).astype(np.uint16)
    frame_denoised = cv2.medianBlur(frame_u16, 3)
    threshold = np.percentile(frame_denoised, 90)
    mask = frame_denoised > threshold
    if np.any(mask):
        cy, cx = center_of_mass(mask)
    else:
        cy, cx = frame.shape[0] // 2, frame.shape[1] // 2

    y1 = int(cy) - roi_size
    y2 = int(cy) + roi_size
    x1 = int(cx) - roi_size
    x2 = int(cx) + roi_size
    roi = frame[max(0, y1):min(frame.shape[0], y2),
                max(0, x1):min(frame.shape[1], x2)]
    return float(np.sum(roi)), float(cx), float(cy)


def filter_dark_frames(preloaded_frames, dark_threshold=5000):
    print(f"[INFO] Scanning {len(preloaded_frames)} frames for dark frame filtering...")
    intensity_trace = []
    frame_numbers_valid = []

    for i, frame in enumerate(preloaded_frames):
        intensity = extract_roi_intensity(frame)
        if intensity >= dark_threshold:
            intensity_trace.append(intensity)
            frame_numbers_valid.append(i)

    print(f"[INFO] {len(frame_numbers_valid)} valid (non-dark) frames found")
    return frame_numbers_valid, intensity_trace


def track_beam_centroid(preloaded_frames, frame_numbers_valid, roi_size=20):
    """
    Recompute per-frame centroid (cx, cy) for each valid frame, alongside
    the ROI intensity, so beam-pointing drift can be checked against the
    intensity trace. See extract_roi_intensity_and_centroid docstring for
    why this distinction matters -- a wandering beam under a FIXED-size ROI
    window inflates the measured intensity variance with a pointing
    artifact, which would corrupt SI if mistaken for real scintillation.
    """
    cx_arr = np.empty(len(frame_numbers_valid))
    cy_arr = np.empty(len(frame_numbers_valid))
    intensity_arr = np.empty(len(frame_numbers_valid))

    for idx, fnum in enumerate(frame_numbers_valid):
        intensity, cx, cy = extract_roi_intensity_and_centroid(
            preloaded_frames[fnum], roi_size=roi_size
        )
        cx_arr[idx] = cx
        cy_arr[idx] = cy
        intensity_arr[idx] = intensity

    return cx_arr, cy_arr, intensity_arr


def find_ramp_blocks(block_means, threshold_ratio=1.5, min_step=30000):
    diffs = np.diff(block_means)
    ramp_start = ramp_end = None

    for i, d in enumerate(diffs):
        baseline = np.median(block_means[:max(1, i+1)])
        if d > min_step and block_means[i+1] > threshold_ratio * baseline:
            ramp_start = i + 1
            break

    if ramp_start is not None:
        ramp_range_diffs = np.abs(diffs[max(0, ramp_start-5):ramp_start+1])
        ramp_magnitude = np.median(ramp_range_diffs) if len(ramp_range_diffs) > 0 \
                          else np.abs(diffs[ramp_start])
        for j in range(ramp_start + 1, len(diffs)):
            if np.abs(diffs[j]) < 0.1 * ramp_magnitude:
                ramp_end = j + 1
                break

    return ramp_start, ramp_end


def compute_intensity_weighted_center(frame, w, h):
    total = np.sum(frame)
    if total > 0:
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        cy = int(np.clip(np.sum(frame * y_coords) / total, 0, h - 1))
        cx = int(np.clip(np.sum(frame * x_coords) / total, 0, w - 1))
        center = (cx, cy)
    else:
        center = (w // 2, h // 2)
    print(f"[INFO] Intensity-weighted beam centroid: {center}")
    return center


def detect_beam_edge(frame, center, w, h):
    Y, X = np.ogrid[:h, :w]
    r_map = np.sqrt((X - center[0])**2 + (Y - center[1])**2).astype(int)

    corners = [
        np.sqrt(center[0]**2 + center[1]**2),
        np.sqrt((w - center[0])**2 + center[1]**2),
        np.sqrt(center[0]**2 + (h - center[1])**2),
        np.sqrt((w - center[0])**2 + (h - center[1])**2),
    ]
    r_max = max(1, min(int(max(corners)), int(np.sqrt(w**2 + h**2))))

    radial_profile = np.zeros(r_max)
    radial_counts = np.zeros(r_max)
    for radius in range(r_max):
        mask = (r_map == radius)
        cnt = np.sum(mask)
        if cnt > 0:
            radial_profile[radius] = np.mean(frame[mask])
            radial_counts[radius] = cnt

    valid = radial_counts > 10
    if not np.any(valid):
        return int(min(center[0], w - center[0], center[1], h - center[1]) * 0.9)

    radial_profile /= (np.max(radial_profile[valid]) + 1e-6)
    kernel = min(max(5, r_max // 20), 30)
    smoothed = np.convolve(radial_profile, np.ones(kernel) / kernel, mode='same')

    low = np.where((smoothed < 0.15) & valid)[0]
    drop_r = low[0] if len(low) > 0 else int(np.where(valid)[0][-1] * 0.95)
    shaved = max(10, drop_r - 3)
    print(f"[INFO] Beam edge at r={drop_r} px -> using r={shaved} px (shaved)")
    return shaved


# SECTION 3: CAMERA SCINTILLATION INDEX

def compute_si_from_stack(frame_stack, circular_mask, si_floor=0.0):
    mean_map = np.mean(frame_stack, axis=0)
    var_map  = np.var(frame_stack, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        si_map = var_map / (mean_map**2 + 1e-6)
        si_map[~np.isfinite(si_map)] = 0.0

    if si_floor > 0:
        si_map = np.clip(si_map - si_floor, 0.0, None)

    raw_vals = si_map[circular_mask]
    raw_vals = raw_vals[raw_vals > 0]

    if len(raw_vals) == 0:
        return si_map, np.array([0.0]), {'mean': 0.0, 'std': 0.0, 'median': 0.0}

    clip_hi = np.percentile(raw_vals, 97)
    si_values = raw_vals[raw_vals < clip_hi]

    stats = {
        'mean':   float(np.mean(si_values)),
        'std':    float(np.std(si_values)),
        'median': float(np.median(si_values)),
        'n_pixels': int(len(si_values)),
    }
    return si_map, si_values, stats


def compute_rolling_si(intensity_trace, fps, window_seconds=0.5):
    """
    Rolling (local-window) SI from the aperture-summed intensity trace.
    Expected to read 1-2 orders of magnitude LOWER than the per-pixel SI
    map's mean (aperture averaging, Andrews & Phillips Ch. 8) -- not a bug.
    """
    window_frames = max(3, int(round(fps * window_seconds))) if fps else 30
    s = pd.Series(np.asarray(intensity_trace, dtype=np.float64))
    roll_mean = s.rolling(window_frames, center=True, min_periods=window_frames//2).mean()
    roll_sqmean = (s**2).rolling(window_frames, center=True, min_periods=window_frames//2).mean()
    roll_var = roll_sqmean - roll_mean**2
    rolling_si = (roll_var / (roll_mean**2 + 1e-9)).values
    return rolling_si, window_frames


# SECTION 4: DTS -> Cn2 -> THEORETICAL SI

def load_dts_csv(csv_path, dt1_col='DT1', dt2_col='DT2', time_col=None):
    """
    The BlueHalo DTS CSV header is comma-space-separated, so pandas reads
    every column except the first with a leading space (' DT1', ' Pressure',
    etc). Stripped here so dt1_col='DT1' matches correctly.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if dt1_col not in df.columns or dt2_col not in df.columns:
        raise ValueError(
            f"Expected columns '{dt1_col}' and '{dt2_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if time_col is not None and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
    return df


def estimate_cn2_from_ri_variance(dts_df, ri_col='RI', outer_scale_m=1.0):
    """
    Independent Cn2 cross-check from RI = (n-1)*1e6 temporal variance.
    NOTE: dominated by slow thermal/HVAC drift, not turbulence (no
    differencing between two points the way DT1/DT2 has). Expect large
    disagreement with the DT1/DT2 estimate -- diagnostic only.
    """
    RI = dts_df[ri_col].to_numpy(dtype=np.float64)
    RI_std = float(np.std(RI))
    sigma_n_sq = (RI_std / 1e6) ** 2
    Cn2_est = sigma_n_sq / (outer_scale_m ** (2.0 / 3.0))
    return {
        'RI_mean': float(np.mean(RI)),
        'RI_std': RI_std,
        'sigma_n_sq': sigma_n_sq,
        'outer_scale_assumed_m': outer_scale_m,
        'Cn2_from_RI': Cn2_est,
    }


def match_dts_window(dts_df, recording_start_utc, duration_s, time_col=None,
                      time_zero_utc=None):
    """
    Requires a manually-supplied time_zero_utc since the BlueHalo unit's GPS
    fields are zeroed with no satellite lock indoors -- there is no wall
    clock column in the CSV to read automatically.
    """
    if time_col is None or time_col not in dts_df.columns or time_zero_utc is None:
        print("[WARNING] DTS time window not matched to camera recording - "
              "using the full DTS file for this run. Provide time_col and "
              "time_zero_utc to align them properly.")
        return dts_df

    t0 = pd.to_datetime(time_zero_utc, unit='s') + pd.to_timedelta(
        recording_start_utc - time_zero_utc, unit='s')
    t1 = t0 + pd.to_timedelta(duration_s, unit='s')
    window = dts_df[(dts_df[time_col] >= t0) & (dts_df[time_col] <= t1)]
    print(f"[INFO] DTS window matched: {len(window)} of {len(dts_df)} rows "
          f"between {t0} and {t1}")
    return window


def compute_cn2_from_dts(dts_df, dt1_col='DT1', dt2_col='DT2',
                          r_separation_m=0.381, P_hPa=None,
                          temp_valid_range=(15.0, 30.0)):
    """
    Cn2 from paired-fiber temperature structure function (Obukhov-Corrsin ->
    Gladstone-Dale, Andrews & Phillips Ch. 3 Eq. 14). At 38.1 cm separation,
    likely EXCEEDS the local wake outer scale (~3-6 cm) -- treat as an
    approximate two-point variance, not a validated inertial-subrange
    measurement, per prior discussion.

    P_hPa: pressure in MILLIBARS/hPa, NOT Pascals. The empirical 79e-6
    coefficient is calibrated for P in mb -- passing Pa here (a 100x
    difference) inflates Cn2 by 100^2 = 10,000x. The DTS CSV's own
    'Pressure' column is already in hPa (~1000-1015 typical), so pass it
    directly without converting to Pa. If P_hPa is None, defaults to
    standard sea-level 1013.25 hPa.

    CT2 uses np.var(), not np.mean(diff**2) -- variance subtracts the mean
    difference automatically, which matters because DT1/DT2 typically carry
    a small fixed calibration offset between channels (observed ~0.3-0.5 C
    in this setup). That offset is an instrument bias, not turbulence, and
    squaring the raw (uncentered) difference lets it dominate the structure
    function.

    temp_valid_range: rows where DT1 or DT2 fall outside this band are
    dropped before computing the structure function. The BlueHalo DTS unit
    produces a small number of known corrupted init-spike readings (e.g.
    ~+70 C and ~-25 C observed in this setup) that are instrument artifacts,
    not real temperature. Even ONE such row can inflate CT2/Cn2 by one to
    two orders of magnitude, since the structure function squares the
    difference -- a single ~90 C spike-vs-normal difference contributes as
    much as thousands of normal-sized fluctuations combined. This filter is
    NOT a substitute for inspecting your data; if you see it drop more than
    a handful of rows out of thousands, something else may be wrong and is
    worth checking by hand rather than trusting the filter blindly.
    """
    if P_hPa is None:
        P_hPa = 1013.25

    lo, hi = temp_valid_range
    valid = (dts_df[dt1_col] >= lo) & (dts_df[dt1_col] <= hi) & \
            (dts_df[dt2_col] >= lo) & (dts_df[dt2_col] <= hi)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f"[INFO] compute_cn2_from_dts: dropped {n_dropped} row(s) with "
              f"DT1/DT2 outside [{lo}, {hi}] C (likely sensor init-spike "
              f"artifacts) out of {len(dts_df)} total rows.")
        if n_dropped > 0.01 * len(dts_df):
            print(f"[WARNING] Dropped rows are >1% of the dataset -- this is "
              f"more than the handful of known init-spike artifacts typically "
              f"seen. Inspect the DTS file manually before trusting this Cn2.")
    df_clean = dts_df.loc[valid]

    T1 = df_clean[dt1_col].to_numpy(dtype=np.float64)
    T2 = df_clean[dt2_col].to_numpy(dtype=np.float64)

    T_mean_K = np.mean(np.concatenate([T1, T2])) + 273.15
    CT2 = np.var(T2 - T1) / (r_separation_m ** (2.0 / 3.0))
    Cn2 = (79e-6 * P_hPa / T_mean_K**2) ** 2 * CT2

    return {
        'T_mean_K': float(T_mean_K),
        'CT2': float(CT2),
        'Cn2': float(Cn2),
        'n_rows_dropped': n_dropped,
    }


def compute_theoretical_si(Cn2, wavelength_m=632.8e-9, L_m=140.0):
    """Extended-Rytov SI (Andrews & Phillips Eq. 9.47-type form)."""
    k = 2 * np.pi / wavelength_m
    sigma_R2 = 1.23 * Cn2 * k**(7.0/6.0) * L_m**(11.0/6.0)

    si_theory = np.exp(
        0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(12.0/5.0))**(7.0/6.0) +
        0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(12.0/5.0))**(5.0/6.0)
    ) - 1

    regime = ('weak' if sigma_R2 < 0.3 else
              'moderate' if sigma_R2 < 1.0 else
              'strong')

    return {
        'sigma_R2': float(sigma_R2),
        'SI_theoretical': float(si_theory),
        'regime': regime,
    }


def find_si_turnover(wavelength_m=632.8e-9, L_m=140.0,
                      cn2_scan_lo=1e-18, cn2_scan_hi=1e-9, n_scan=400):
    """
    The extended-Rytov SI_theoretical(Cn2) curve is NOT monotonic -- it
    rises with Cn2 up to a peak (scintillation saturation, Andrews &
    Phillips Ch. 9-10) and then DECREASES at higher Cn2. Any Cn2-from-SI
    inversion is only unique below this turnover point; above it, a single
    SI value corresponds to two different Cn2 values (or none, if it
    exceeds the peak). This directly matters for the payload-mass
    invertibility claim -- the whole method requires staying below this
    turnover.

    Returns (cn2_at_peak, si_at_peak) found by coarse log-spaced scan.
    Not high precision -- good enough to bracket a root-find or to warn
    when a measured SI is close to or above the peak.
    """
    cn2_grid = np.logspace(np.log10(cn2_scan_lo), np.log10(cn2_scan_hi), n_scan)
    si_grid = np.array([compute_theoretical_si(c, wavelength_m, L_m)['SI_theoretical']
                         for c in cn2_grid])
    peak_idx = np.argmax(si_grid)
    return float(cn2_grid[peak_idx]), float(si_grid[peak_idx])


def invert_cn2_from_si(si_measured, wavelength_m=632.8e-9, L_m=140.0,
                        cn2_scan_lo=1e-18):
    """
    Inverts the extended-Rytov SI_theoretical(Cn2) relation to find the Cn2
    that WOULD produce a given measured camera SI, restricted to the
    monotonic (weak-to-moderate) branch below the scintillation-saturation
    peak (see find_si_turnover). This lets you compare a camera-measured SI
    directly against a DTS-measured Cn2 in the same units, as an
    independent cross-check between the two instruments.

    Returns dict with 'Cn2_camera_implied' (None if si_measured exceeds the
    monotonic branch's peak -- no unique solution exists) and
    'near_turnover' (bool, True if si_measured is within 10% of the peak,
    meaning the inversion is numerically fragile even if technically
    solvable, and the payload-inference method itself would be operating
    close to its own invertibility limit at this condition).
    """
    from scipy.optimize import brentq

    cn2_peak, si_peak = find_si_turnover(wavelength_m, L_m, cn2_scan_lo=cn2_scan_lo)

    if si_measured >= si_peak:
        return {
            'Cn2_camera_implied': None,
            'si_peak': si_peak,
            'cn2_at_peak': cn2_peak,
            'near_turnover': True,
        }

    def f_log(log10_cn2):
        return compute_theoretical_si(10**log10_cn2, wavelength_m, L_m)['SI_theoretical'] - si_measured

    try:
        # solve in log10(Cn2) space -- Cn2 itself is ~1e-14 to 1e-9, far
        # below scipy's default absolute xtol (2e-12), which caused brentq
        # to falsely "converge" near the lower bracket bound when solving
        # directly in linear Cn2 space. Working in log10 keeps the search
        # variable O(1)-O(10), where the default tolerance is meaningful.
        log10_lo = np.log10(cn2_scan_lo)
        log10_hi = np.log10(cn2_peak)
        log10_root = brentq(f_log, log10_lo, log10_hi)
        cn2_implied = 10 ** log10_root
    except ValueError:
        # measured SI below what even cn2_scan_lo produces -- no positive
        # root in range; report None rather than a misleading extrapolation
        cn2_implied = None

    near_turnover = si_measured > 0.9 * si_peak

    return {
        'Cn2_camera_implied': cn2_implied,
        'si_peak': si_peak,
        'cn2_at_peak': cn2_peak,
        'near_turnover': near_turnover,
    }


# SECTION 5: PLOTTING

def plot_timing_diagnostics(dts_us, intensity_trace, valid_frame_indices,
                             fps_measured, save_dir, condition_label):
    if dts_us is None or len(dts_us) < 2:
        print("[INFO] No timestamp data available - skipping timing diagnostic plots.")
        return

    dts_sec = dts_us * 1e-6
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Timing Diagnostics - {condition_label}', fontsize=14, fontweight='bold')

    dt_us_diff = np.diff(dts_us)
    axes[0].plot(dt_us_diff, color='steelblue', linewidth=0.8, alpha=0.8)
    axes[0].axhline(np.median(dt_us_diff), color='red', linewidth=1.5,
                    linestyle='--', label=f'Median dt = {np.median(dt_us_diff):.1f} us')
    axes[0].set_xlabel('Frame Index'); axes[0].set_ylabel('dt between frames (us)')
    axes[0].set_title(f'Frame Interval - Measured fps ~ {fps_measured:.1f}'
                      if fps_measured else 'Frame Interval')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if len(intensity_trace) > 0 and len(valid_frame_indices) > 0:
        valid_times = []
        for vi in valid_frame_indices:
            if vi < len(dts_sec):
                valid_times.append(dts_sec[vi] - dts_sec[0])
            else:
                valid_times.append(vi / (fps_measured if fps_measured else 1000))
        axes[1].plot(valid_times[:len(intensity_trace)], intensity_trace,
                     color='darkorange', linewidth=0.8)
        axes[1].set_xlabel('Time from recording start (s)')
        axes[1].set_ylabel('ROI Summed Intensity (DN)')
        axes[1].set_title('Laser Intensity Trace vs. Real Time')
        axes[1].grid(True, alpha=0.3)

    axes[2].hist(dt_us_diff, bins=60, color='steelblue', edgecolor='white',
                 linewidth=0.4, density=True)
    axes[2].set_xlabel('Frame Interval dt (us)'); axes[2].set_ylabel('Probability Density')
    axes[2].set_title(f'Frame Timing Jitter  sigma = {np.std(dt_us_diff):.2f} us')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, f'timing_diagnostics_{condition_label.replace(" ", "_")}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show()
    print(f"[INFO] Timing diagnostics saved: {out}")


def plot_centroid_diagnostics(cx_arr, cy_arr, intensity_arr, fps, condition_label, save_dir):
    """
    Checks whether intensity-trace fluctuations correlate with beam
    centroid position -- if they do, a fixed-size ROI window is clipping a
    physically wandering beam differently frame to frame, which inflates
    the measured intensity variance with a pointing artifact rather than
    true scintillation, illumination, or absorption changes. If centroid
    position stays essentially flat while intensity still swings, the
    fluctuation is a genuine illumination-level effect, not beam wander.
    """
    t = np.arange(len(intensity_arr)) / (fps if fps else 1.0)

    r = np.hypot(cx_arr - np.mean(cx_arr), cy_arr - np.mean(cy_arr))
    if np.std(intensity_arr) > 0 and np.std(r) > 0:
        corr = np.corrcoef(r, intensity_arr)[0, 1]
    else:
        corr = 0.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f'Beam Centroid vs. Intensity - {condition_label}\n'
                 f'corr(radial centroid offset, intensity) = {corr:.3f}',
                 fontsize=13, fontweight='bold', y=1.02)

    axes[0].plot(t, cx_arr - np.mean(cx_arr), color='steelblue', lw=0.8, label='cx - mean')
    axes[0].plot(t, cy_arr - np.mean(cy_arr), color='indianred', lw=0.8, label='cy - mean')
    axes[0].set_ylabel('Centroid offset (px)')
    axes[0].set_title('Beam centroid position over time (relative to mean)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, r, color='darkorange', lw=0.8)
    axes[1].set_ylabel('Radial offset from mean (px)')
    axes[1].set_title('Centroid distance from mean position')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, intensity_arr, color='mediumseagreen', lw=0.8)
    axes[2].set_ylabel('ROI Summed Intensity (DN)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('ROI intensity (for comparison against panels above)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(save_dir, f'centroid_diagnostics_{condition_label.replace(" ", "_")}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show()

    if abs(corr) > 0.5:
        print(f"[WARNING] |corr|={abs(corr):.2f} > 0.5 between centroid offset and "
              f"intensity -- beam pointing/wander likely contaminates the intensity "
              f"trace and any SI computed from it. Consider a larger fixed ROI, or "
              f"track the centroid and use a per-frame recentered window instead of "
              f"a single global ROI.")
    else:
        print(f"[INFO] |corr|={abs(corr):.2f} <= 0.5 -- intensity fluctuations are "
              f"not strongly explained by centroid wander; more likely a genuine "
              f"illumination-level effect (laser power, absorption, ambient turbulence).")

    print(f"[INFO] Centroid diagnostics saved: {out}")
    return corr


def plot_roi_visualization(cropped_first, center, shaved_radius, inner_radius,
                           condition_label, save_dir):
    vis = cv2.normalize(cropped_first, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.circle(vis_color, center, 5, (0, 255, 0), -1)
    cv2.circle(vis_color, center, inner_radius, (0, 255, 255), 2)
    cv2.circle(vis_color, center, shaved_radius, (0, 0, 255), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Analysis Region - {condition_label}', fontsize=13, fontweight='bold')
    axes[0].imshow(vis_color)
    axes[0].set_title('ROI Overlay\nGreen=Center | Yellow=Analysis | Red=Beam Edge')
    axes[0].axis('off')

    im = axes[1].imshow(cropped_first, cmap='inferno',
                        norm=LogNorm(vmin=max(1, np.percentile(cropped_first, 5)),
                                     vmax=np.percentile(cropped_first, 99.5)))
    plt.colorbar(im, ax=axes[1], label='Intensity (DN)')
    theta = np.linspace(0, 2*np.pi, 360)
    axes[1].plot(center[0] + inner_radius * np.cos(theta),
                 center[1] + inner_radius * np.sin(theta), 'cyan', lw=1.5)
    axes[1].plot(center[0] + shaved_radius * np.cos(theta),
                 center[1] + shaved_radius * np.sin(theta), 'red', lw=1, ls='--')
    axes[1].set_title('Log-Scale Irradiance')

    plt.tight_layout()
    out = os.path.join(save_dir, f'ROI_visualization_{condition_label.replace(" ", "_")}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show()
    print(f"[INFO] ROI visualization saved: {out}")


def plot_si_map(si_map, circular_mask, center, inner_radius, shaved_radius,
                condition_label, save_dir):
    masked_si = np.where(circular_mask, si_map, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Pixel-wise SI Map - {condition_label}', fontsize=13, fontweight='bold')

    vmax = np.nanpercentile(masked_si[circular_mask], 95)
    im0 = axes[0].imshow(masked_si, cmap='hot', vmin=0, vmax=vmax)
    plt.colorbar(im0, ax=axes[0], label='sigma^2_I')
    axes[0].set_title('SI Spatial Map (linear scale)')

    safe = np.where(circular_mask & (si_map > 0), si_map, np.nan)
    vmin_log = np.nanpercentile(safe[circular_mask & (si_map > 0)], 5)
    im1 = axes[1].imshow(safe, cmap='hot',
                         norm=LogNorm(vmin=max(1e-6, vmin_log), vmax=vmax))
    plt.colorbar(im1, ax=axes[1], label='sigma^2_I (log)')
    axes[1].set_title('SI Spatial Map (log scale)')

    for ax in axes:
        theta = np.linspace(0, 2*np.pi, 360)
        ax.plot(center[0] + inner_radius * np.cos(theta),
                center[1] + inner_radius * np.sin(theta), 'cyan', lw=1)

    plt.tight_layout()
    out = os.path.join(save_dir, f'SI_map_{condition_label.replace(" ", "_")}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show()
    print(f"[INFO] SI spatial map saved: {out}")


def plot_rolling_si(rolling_si, window_frames, fps, condition_label, save_dir):
    t = np.arange(len(rolling_si)) / (fps if fps else 1.0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, rolling_si, color='mediumseagreen', lw=0.8)
    valid = np.isfinite(rolling_si)
    if np.any(valid):
        ax.axhline(np.nanmean(rolling_si), color='red', lw=1.5, ls='--',
                   label=f'mean = {np.nanmean(rolling_si):.5f}')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Rolling sigma^2_I')
    ax.set_title(f'{condition_label} - Rolling SI (window = {window_frames} frames)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, f'rolling_SI_{condition_label.replace(" ", "_")}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show()
    print(f"[INFO] Rolling SI plot saved: {out}")


def plot_period_mean_evolution(period_data, overall_stats, condition_label, save_dir):
    means = [p['mean_si'] for p in period_data]
    stds  = [p['std_si']  for p in period_data]
    xs    = np.arange(1, len(means) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(xs, means, yerr=stds, fmt='o-', color='royalblue',
                ecolor='skyblue', capsize=4, lw=2, markersize=7,
                label='Period mean sigma^2_I +/- sigma')
    ax.axhline(overall_stats['mean'], color='red', lw=2, ls='--',
               label=f'Overall mean = {overall_stats["mean"]:.5f}')
    ax.fill_between(xs,
                    [overall_stats['mean'] - overall_stats['std']] * len(xs),
                    [overall_stats['mean'] + overall_stats['std']] * len(xs),
                    alpha=0.15, color='red', label='+/-1sigma overall')

    cv = (np.std(means) / np.mean(means)) * 100 if np.mean(means) > 0 else 0
    ax.set_title(f'{condition_label}  -  SI Evolution Across Periods\n'
                 f'Coefficient of Variation = {cv:.1f}%  (< 10% = stationary)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Period'); ax.set_ylabel('Mean sigma^2_I'); ax.set_xticks(xs)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, f'SI_period_evolution_{condition_label.replace(" ", "_")}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show()
    print(f"[INFO] Period evolution plot saved: {out}")
    return cv


# SECTION 6: CSV RESULTS TABLE

RESULTS_CSV_COLUMNS = [
    'date', 'condition', 'video_index', 'frame_rate', 'n_frames',
    'SI_camera_global', 'SI_camera_rolling_mean', 'SI_camera_rolling_std',
    'centroid_intensity_corr',
    'T_mean_K', 'CT2', 'Cn2_DTS', 'Cn2_camera_implied', 'Cn2_camera_to_DTS_ratio',
    'near_si_turnover', 'Cn2_from_RI', 'RI_to_DT_ratio',
    'sigma_R2', 'SI_theoretical', 'regime',
    'stationarity_cv_pct', 'quality_pass',
]


def append_result_row(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, '') for k in RESULTS_CSV_COLUMNS})
    print(f"[INFO] Row appended to {csv_path}")


# SECTION 7: MAIN PAIRED ANALYSIS

def run_paired_analysis(
    flir_npy_path,
    dts_csv_path,
    condition_label="Condition",
    video_index=1,
    date_str=None,
    dt1_col='DT1',
    dt2_col='DT2',
    r_separation_m=0.381,
    wavelength_m=632.8e-9,
    L_m=140.0,
    P_hPa=None,  # None -> auto-use DTS CSV's own Pressure column (hPa); see
                 # compute_cn2_from_dts docstring re: units (mb, NOT Pa)
    skip_initial_frames=300,
    dark_threshold=5000,
    exclude_edge_ring=True,
    edge_ring_percent=15,
    detect_ramp=True,
    block_size=30,
    manual_center=None,
    manual_radius=None,
    num_periods=10,
    dark_frame_path=None,
    flatfield_path=None,
    ri_outer_scale_m=0.045,
    results_csv='scintillation_results.csv',
):
    save_dir = get_experiment_folder()

    si_floor = 0.0
    if dark_frame_path is not None:
        si_floor = build_camera_noise_floor(np.load(dark_frame_path))

    flatfield = None
    if flatfield_path is not None:
        flatfield = build_flatfield(np.load(flatfield_path))

    # CAMERA SIDE
    preloaded_frames, fps_measured, dts_us, recording_start_utc = load_flir_stack(flir_npy_path)

    if skip_initial_frames >= len(preloaded_frames):
        print(f"[WARNING] skip_initial_frames={skip_initial_frames} >= "
              f"{len(preloaded_frames)} total frames -- reducing to 0 so "
              f"some data remains for analysis. This session may be too "
              f"short for the skip you requested.")
        skip_initial_frames = 0

    preloaded_frames = preloaded_frames[skip_initial_frames:]
    dts_us = dts_us[skip_initial_frames:] if dts_us is not None else None

    frame_numbers_valid, intensity_trace = filter_dark_frames(preloaded_frames, dark_threshold)
    if len(frame_numbers_valid) == 0:
        print("[ERROR] No valid frames found. Check dark_threshold.")
        return None

    cx_arr, cy_arr, _ = track_beam_centroid(preloaded_frames, frame_numbers_valid)
    centroid_corr = plot_centroid_diagnostics(
        cx_arr, cy_arr, np.asarray(intensity_trace), fps_measured,
        condition_label, save_dir
    )

    plot_timing_diagnostics(dts_us, intensity_trace, frame_numbers_valid,
                            fps_measured, save_dir, condition_label)

    if detect_ramp and len(intensity_trace) > block_size * 2:
        block_means = np.array([
            np.mean(intensity_trace[i:i+block_size])
            for i in range(0, len(intensity_trace) - block_size + 1, block_size)
        ])
        ramp_start, _ = find_ramp_blocks(block_means)
        if ramp_start is not None:
            ramp_start_idx = ramp_start * block_size
            print(f"[INFO] Laser ramp detected - discarding {ramp_start_idx} frames")
            frame_numbers_valid = frame_numbers_valid[ramp_start_idx:]
            intensity_trace = intensity_trace[ramp_start_idx:]

    print(f"[INFO] Using {len(frame_numbers_valid)} frames for SI analysis")

    first_frame = preloaded_frames[frame_numbers_valid[0]].astype(np.float32)
    if flatfield is not None:
        first_frame = apply_flatfield(first_frame, flatfield)
    h, w = first_frame.shape[:2]
    cropped_first = first_frame

    center = manual_center if manual_center is not None else \
             compute_intensity_weighted_center(cropped_first, w, h)
    shaved_radius = manual_radius if manual_radius is not None else \
                    detect_beam_edge(cropped_first, center, w, h)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    inner_radius = int((1.0 - edge_ring_percent/100.0) * shaved_radius) \
                   if exclude_edge_ring else shaved_radius
    circular_mask = (dist <= inner_radius) & (dist >= 5)
    print(f"[INFO] Analysis mask: {np.sum(circular_mask)} pixels, "
          f"inner_radius={inner_radius} px, shaved_radius={shaved_radius} px")

    plot_roi_visualization(cropped_first, center, shaved_radius, inner_radius,
                           condition_label, save_dir)

    frame_list = []
    print(f"[INFO] Loading {len(frame_numbers_valid)} frames "
          f"(flat-field {'ON' if flatfield is not None else 'OFF'}, "
          f"noise floor {'ON' if si_floor > 0 else 'OFF'})...")
    for idx, fnum in enumerate(frame_numbers_valid):
        if idx % 200 == 0:
            print(f"  Frame {idx+1}/{len(frame_numbers_valid)}")
        frame = preloaded_frames[fnum].astype(np.float32)
        if flatfield is not None:
            frame = apply_flatfield(frame, flatfield)
        frame_list.append(frame)

    frame_stack = np.array(frame_list, dtype=np.float32)
    print(f"[INFO] Frame stack: {frame_stack.shape}")

    overall_si_map, overall_si_vals, overall_stats = compute_si_from_stack(
        frame_stack, circular_mask, si_floor=si_floor
    )
    print(f"[INFO] Camera SI_global (sigma^2_I): {overall_stats['mean']:.6f}")
    plot_si_map(overall_si_map, circular_mask, center, inner_radius, shaved_radius,
                condition_label, save_dir)

    rolling_si, window_frames = compute_rolling_si(intensity_trace, fps_measured)
    plot_rolling_si(rolling_si, window_frames, fps_measured, condition_label, save_dir)
    rolling_si_valid = rolling_si[np.isfinite(rolling_si)]

    frames_per_period = len(frame_list) // num_periods
    period_data = []
    for pi in range(num_periods):
        s_idx = pi * frames_per_period
        e_idx = s_idx + frames_per_period if pi < num_periods - 1 else len(frame_list)
        _, si_vals, stats = compute_si_from_stack(frame_stack[s_idx:e_idx], circular_mask,
                                                    si_floor=si_floor)
        period_data.append({'period': pi+1, 'mean_si': stats['mean'], 'std_si': stats['std']})
    cv_val = plot_period_mean_evolution(period_data, overall_stats, condition_label, save_dir)

    # DTS SIDE
    dts_df = load_dts_csv(dts_csv_path, dt1_col=dt1_col, dt2_col=dt2_col)
    duration_s = (dts_us[-1] - dts_us[0]) * 1e-6 if dts_us is not None else None
    dts_window = match_dts_window(dts_df, recording_start_utc, duration_s)

    if 'Pressure' in dts_window.columns:
        P_hPa_used = float(dts_window['Pressure'].mean())
        print(f"[INFO] Using measured mean pressure from DTS CSV: "
              f"{P_hPa_used:.2f} hPa")
    else:
        P_hPa_used = P_hPa if P_hPa is not None else 1013.25
        print(f"[INFO] No Pressure column found -- using default P_hPa={P_hPa_used}")

    cn2_result = compute_cn2_from_dts(dts_window, dt1_col=dt1_col, dt2_col=dt2_col,
                                       r_separation_m=r_separation_m, P_hPa=P_hPa_used)
    si_theory_result = compute_theoretical_si(cn2_result['Cn2'], wavelength_m=wavelength_m,
                                               L_m=L_m)

    print(f"[INFO] DTS: T_mean={cn2_result['T_mean_K']:.2f} K, "
          f"Cn2={cn2_result['Cn2']:.3e} m^-2/3, "
          f"sigma_R^2={si_theory_result['sigma_R2']:.4f} ({si_theory_result['regime']}), "
          f"SI_theoretical={si_theory_result['SI_theoretical']:.6f}")

    # Invert the camera's measured SI to an implied Cn2, for direct
    # apples-to-apples comparison against Cn2_DTS -- two independent
    # instruments, same units. Only valid below the scintillation-
    # saturation turnover; see invert_cn2_from_si docstring.
    inv_result = invert_cn2_from_si(overall_stats['mean'], wavelength_m=wavelength_m, L_m=L_m)
    cn2_camera_implied = inv_result['Cn2_camera_implied']
    cn2_ratio = ''
    if cn2_camera_implied is not None:
        cn2_ratio = cn2_camera_implied / cn2_result['Cn2']
        print(f"[INFO] Cn2 implied by camera SI: {cn2_camera_implied:.3e} m^-2/3 "
              f"(vs. DTS: {cn2_result['Cn2']:.3e}, ratio={cn2_ratio:.3f})")
    else:
        print(f"[WARNING] Camera SI ({overall_stats['mean']:.4f}) is at or above "
              f"the scintillation-saturation turnover (peak SI={inv_result['si_peak']:.4f} "
              f"at Cn2={inv_result['cn2_at_peak']:.3e}) -- no unique Cn2 inversion exists "
              f"at this operating point.")
    if inv_result['near_turnover'] and cn2_camera_implied is not None:
        print(f"[WARNING] Measured SI is within 10% of the saturation peak -- "
              f"inversion is numerically fragile here, and this condition is "
              f"operating close to the payload-inference method's own "
              f"invertibility limit.")

    ri_result = None
    ratio_ri_to_dt = ''
    if 'RI' in dts_window.columns:
        ri_result = estimate_cn2_from_ri_variance(dts_window, outer_scale_m=ri_outer_scale_m)
        ratio_ri_to_dt = ri_result['Cn2_from_RI'] / (cn2_result['Cn2'] + 1e-30)
        print(f"[INFO] Cn2 (RI variance, L0={ri_outer_scale_m} m): "
              f"{ri_result['Cn2_from_RI']:.3e} m^-2/3  "
              f"(ratio to DT1/DT2 estimate: {ratio_ri_to_dt:.2f}x -- "
              f"expect large disagreement, dominated by thermal drift, "
              f"not a validation check)")

    quality_pass = (cv_val < 10.0) and (overall_stats['n_pixels'] > 0)

    row = {
        'date': date_str or '',
        'condition': condition_label,
        'video_index': video_index,
        'frame_rate': round(fps_measured, 2) if fps_measured else '',
        'n_frames': len(frame_list),
        'SI_camera_global': overall_stats['mean'],
        'SI_camera_rolling_mean': float(np.nanmean(rolling_si_valid)) if len(rolling_si_valid) else '',
        'SI_camera_rolling_std': float(np.nanstd(rolling_si_valid)) if len(rolling_si_valid) else '',
        'centroid_intensity_corr': float(centroid_corr),
        'T_mean_K': cn2_result['T_mean_K'],
        'CT2': cn2_result['CT2'],
        'Cn2_DTS': cn2_result['Cn2'],
        'Cn2_camera_implied': cn2_camera_implied if cn2_camera_implied is not None else '',
        'Cn2_camera_to_DTS_ratio': cn2_ratio,
        'near_si_turnover': inv_result['near_turnover'],
        'Cn2_from_RI': ri_result['Cn2_from_RI'] if ri_result else '',
        'RI_to_DT_ratio': ratio_ri_to_dt,
        'sigma_R2': si_theory_result['sigma_R2'],
        'SI_theoretical': si_theory_result['SI_theoretical'],
        'regime': si_theory_result['regime'],
        'stationarity_cv_pct': cv_val,
        'quality_pass': quality_pass,
    }
    append_result_row(results_csv, row)

    json_path = os.path.join(save_dir,
                f'combined_results_{condition_label.replace(" ", "_")}_{video_index}.json')
    with open(json_path, 'w') as f:
        json.dump(row, f, indent=2)
    print(f"[INFO] Per-run JSON saved: {json_path}")

    return row


# ENTRY POINT

if __name__ == "__main__":
    flir_path = select_flir_session()
    if not flir_path:
        print("[ERROR] No FLIR file selected. Exiting.")
        raise SystemExit

    dts_path = select_file_dialog(
        title="Select matching DTS CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not dts_path:
        print("[ERROR] No DTS file selected. Exiting.")
        raise SystemExit

    print(f"[INFO] FLIR file: {flir_path}")
    print(f"[INFO] DTS file:  {dts_path}")

    result = run_paired_analysis(
        flir_npy_path=flir_path,
        dts_csv_path=dts_path,
        condition_label="Background",
        video_index=1,
        r_separation_m=0.381,
        wavelength_m=632.8e-9,
        L_m=140.0,
        manual_center=None,
        manual_radius=None,
        dark_frame_path=None,
        flatfield_path=None,
        skip_initial_frames=0, 
        results_csv='scintillation_results.csv',
    )