"""
FLIR Blackfly BFLY-U3-13S2C acquisition script.

Runs the camera directly via PySpin (NOT SpinView) so that per-frame
chunk data (FrameID, hardware Timestamp) can be captured alongside the
raw Mono16 pixel data. SpinView cannot do this — see project notes.

Output per session:
    <save_dir>/flir_frames.npy        - raw stack, shape (N, H, W), dtype uint16
    <save_dir>/flir_chunk_meta.npz    - frame_id[N], timestamp_ns[N]
    <save_dir>/flir_session_meta.json - width, height, pixel format, exposure,
                                         gain, fps target, dropped-frame report

IMPORTANT: Close SpinView completely before running this script. Only one
process can hold the camera connection at a time.
"""

import PySpin
import numpy as np
import json
import os
import time


# Validated settings for 632.8 nm laser work, ND0.3 filter removed
# (full transmission, OD 0), moderate gain to protect temporal resolution
# at 35 fps rather than pushing exposure toward the frame period limit.
DEFAULT_EXPOSURE_US = 12000.0   # midpoint of validated 10,000-15,000 us range
DEFAULT_GAIN_DB = 13.0
DEFAULT_FPS_TARGET = 30.0       # hardware ceiling on this camera at Mono16:
                                 # Device Link Throughput Limit (74,880,000 B/s)
                                 # / frame size (1288*964*2 B) ~= 30.15 fps.
                                 # 35 fps was the original target but is not
                                 # achievable without raising the throughput
                                 # limit (untested for USB bus headroom) --
                                 # 30 fps accepted as the validated rate.


def configure_camera(cam, exposure_us=None, gain_db=None,
                      fps_target=DEFAULT_FPS_TARGET):
    """
    Lock in the camera settings validated for 632.8 nm laser work.

    Physics rationale (from prior FLIR characterization):
    - Mono16: full linear sensor readout, no color processing
    - GammaEnable off: gamma correction is a nonlinear tone curve applied
      by the sensor ISP; any nonlinearity here corrupts the linear
      relationship between photon count and DN that scintillation
      statistics (Var[I]/<I>^2) depend on.
    - SharpeningEnable off: sharpening is a spatial high-pass filter —
      it manufactures fake local contrast, which is exactly what
      spatial SI measurement would misread as scintillation.
    - ExposureAuto/GainAuto off: auto-exposure or auto-gain would silently
      renormalize intensity frame-to-frame, destroying the very
      fluctuations you're trying to measure. This also guards against
      the previously-found bug where Exposure Compensation Auto silently
      overrode manual settings.
    """
    cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)

    try:
        cam.GammaEnable.SetValue(False)
    except PySpin.SpinnakerException:
        print("[WARN] GammaEnable not accessible on this camera/firmware")

    try:
        cam.SharpeningEnable.SetValue(False)
    except PySpin.SpinnakerException:
        print("[WARN] SharpeningEnable not accessible on this camera/firmware")

    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)

    # Exposure Compensation Auto is a SEPARATE auto-brightness mechanism
    # from ExposureAuto -- it can apply an automatic EV offset on top of
    # a manually fixed exposure/gain. If left on Continuous, it would
    # silently rescale intensity frame-to-frame or scene-to-scene,
    # independent of real scintillation -- directly corrupting
    # Var[I]/<I>^2 regardless of how well exposure/gain/fps are locked
    # down. Force it off; some firmware exposes this as a plain boolean
    # (ExposureCompensationAuto) rather than an enumeration, so try both.
    try:
        ec_node = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('ExposureCompensationAuto'))
        if PySpin.IsAvailable(ec_node) and PySpin.IsWritable(ec_node):
            current = ec_node.GetCurrentEntry().GetSymbolic()
            if current != 'Off':
                ec_node.SetIntValue(ec_node.GetEntryByName('Off').GetValue())
                print(f"[INFO] ExposureCompensationAuto forced Off (was {current})")
            else:
                print("[INFO] ExposureCompensationAuto already Off")
        else:
            print("[WARN] ExposureCompensationAuto node not writable/available "
                  "as an enumeration on this firmware.")
    except PySpin.SpinnakerException as e:
        print(f"[WARN] Could not access ExposureCompensationAuto as enum ({e}); "
              "trying as boolean node instead.")
        try:
            ec_bool = PySpin.CBooleanPtr(cam.GetNodeMap().GetNode('ExposureCompensationAuto'))
            if PySpin.IsAvailable(ec_bool) and PySpin.IsWritable(ec_bool):
                ec_bool.SetValue(False)
                print("[INFO] ExposureCompensationAuto (boolean) forced False")
        except PySpin.SpinnakerException as e2:
            print(f"[WARN] ExposureCompensationAuto not accessible at all on this "
                  f"camera/firmware ({e2}). If SpinView shows it locked/greyed out "
                  f"under Exposure Mode=Timed, it may already be inert -- verify by "
                  f"capturing two frames of a static scene at different times and "
                  f"confirming their means match; drifting means would indicate "
                  f"it's still active despite being unwritable from here.")

    exposure_us = exposure_us if exposure_us is not None else DEFAULT_EXPOSURE_US
    gain_db = gain_db if gain_db is not None else DEFAULT_GAIN_DB

    cam.ExposureTime.SetValue(float(exposure_us))
    cam.Gain.SetValue(float(gain_db))

    # Frame period budget check: exposure must leave enough headroom in
    # each frame period (1/fps) for sensor readout + transfer, or the
    # camera will silently cap the achievable frame rate below fps_target.
    frame_period_us = 1e6 / fps_target
    duty = exposure_us / frame_period_us
    print(f"[INFO] PixelFormat={cam.PixelFormat.GetCurrentEntry().GetSymbolic()}")
    print(f"[INFO] ExposureTime={cam.ExposureTime.GetValue():.1f} us")
    print(f"[INFO] Gain={cam.Gain.GetValue():.2f} dB")
    print(f"[INFO] Target fps={fps_target:.1f} -> frame period={frame_period_us:.1f} us, "
          f"exposure uses {duty*100:.1f}% of that period")
    if duty > 0.8:
        print("[WARN] Exposure is using >80% of the frame period at this fps — "
              "achieved fps may fall below target. Verify actual fps from "
              "chunk timestamps after capture, not just the fps you set.")

    try:
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(float(fps_target))
        print(f"[INFO] AcquisitionFrameRate set to {fps_target:.1f} fps")
    except PySpin.SpinnakerException:
        print("[WARN] Could not set AcquisitionFrameRate directly on this "
              "camera/firmware — frame rate will follow exposure/readout timing. "
              "Rely on chunk timestamps to know the actual achieved fps.")


def enable_chunk_data(cam):
    """
    Chunk data (ChunkModeActive) is producing ImageStatus 11
    (CHUNK_DATA_INVALID) on every frame with this camera/SDK combination,
    even with a single clean selector (Timestamp only) and even after a
    full reset-and-reconfigure pass. That points to an incompatibility
    between this firmware/Spinnaker version and chunk mode itself under
    Mono16 streaming, not a selector-ordering issue.

    Chunk mode is disabled entirely. Per-frame timing instead comes from
    img.GetTimeStamp() in acquire_stack(), which is a built-in Image
    method that does NOT require ChunkModeActive -- it reads the same
    underlying camera clock without going through the chunk-data payload
    path that's failing here. You lose FrameID-based hardware drop
    detection (chunk-only), but keep real per-frame timestamps for
    jitter/fps diagnostics.
    """
    try:
        cam.ChunkModeActive.SetValue(False)
    except PySpin.SpinnakerException:
        pass
    print("[INFO] Chunk data disabled (ImageStatus 11 on this camera/firmware). "
          "Using img.GetTimeStamp() for per-frame timing instead -- "
          "no hardware FrameID counter available for drop detection.")
    return []


def acquire_stack(cam, n_frames, save_dir, condition_label="condition",
                   frameid_available=True):
    """
    Acquire n_frames raw Mono16 frames with chunk metadata.

    Returns
    -------
    frames        : np.ndarray (N, H, W) uint16 — raw linear DN, unprocessed
    frame_ids     : np.ndarray (N,) int64
    timestamps_ns : np.ndarray (N,) int64
    """
    h = cam.Height.GetValue()
    w = cam.Width.GetValue()
    print(f"[INFO] Frame size: {w} x {h}")

    frames = np.empty((n_frames, h, w), dtype=np.uint16)
    frame_ids = np.full(n_frames, -1, dtype=np.int64)
    timestamps_ns = np.full(n_frames, -1, dtype=np.int64)

    cam.BeginAcquisition()
    t_wall_start = time.time()

    n_incomplete = 0
    n_captured = 0

    try:
        for i in range(n_frames):
            if i % 100 == 0:
                print(f"  Capturing frame {i+1}/{n_frames}")

            img = cam.GetNextImage(5000)  # 5 s timeout

            if img.IsIncomplete():
                print(f"[WARN] Incomplete frame at index {i} "
                      f"(status: {img.GetImageStatus()})")
                n_incomplete += 1
                img.Release()
                continue

            chunk = None
            if frameid_available:
                try:
                    chunk = img.GetChunkData()
                    frame_ids[i] = chunk.GetFrameID()
                except PySpin.SpinnakerException:
                    frame_ids[i] = -1

            # GetTimeStamp() is a plain Image method, independent of
            # ChunkModeActive/chunk data -- reads the same underlying
            # camera clock without requiring the chunk payload path.
            timestamps_ns[i] = img.GetTimeStamp()

            # GetNDArray() on a Mono16 image returns the raw uint16 array
            # directly -- no conversion, no normalization, no display
            # pipeline processing.
            frames[i] = img.GetNDArray().copy()
            n_captured += 1

            img.Release()

    finally:
        cam.EndAcquisition()

    t_wall_elapsed = time.time() - t_wall_start
    print(f"[INFO] Acquisition complete: {n_captured}/{n_frames} good frames, "
          f"{n_incomplete} incomplete, wall time {t_wall_elapsed:.2f} s")

    # ── Dropped-frame check via FrameID continuity (only meaningful if
    #    FrameID chunk data is actually available; currently disabled on
    #    this camera/firmware, so frame_ids stays at -1 placeholder) ───────
    id_valid_mask = frame_ids >= 0
    valid_ids = frame_ids[id_valid_mask]
    if len(valid_ids) > 1:
        gaps = np.diff(valid_ids)
        dropped_at = np.where(gaps != 1)[0]
        if len(dropped_at) > 0:
            total_dropped = int(np.sum(gaps[dropped_at] - 1))
            print(f"[WARN] {len(dropped_at)} gap(s) in FrameID sequence, "
                  f"~{total_dropped} frame(s) dropped by camera/USB "
                  f"(not just incomplete transfers)")
        else:
            print("[INFO] FrameID sequence continuous — no drops detected")
    else:
        print("[INFO] No FrameID chunk data available (disabled on this "
              "camera/firmware) — drop detection relies on timestamp gaps "
              "below instead of an explicit frame counter.")

    # ── Timestamp jitter / fps check — this is independent of FrameID and
    #    works via img.GetTimeStamp(), which is populated regardless of
    #    chunk-data status ─────────────────────────────────────────────────
    ts_valid_mask = timestamps_ns > 0
    valid_ts = timestamps_ns[ts_valid_mask]
    if len(valid_ts) > 1:
        dt_ns = np.diff(valid_ts)
        dt_ms = dt_ns / 1e6
        fps_measured = 1e9 / np.median(dt_ns)
        print(f"[INFO] Measured fps from hardware timestamps: {fps_measured:.2f}")
        print(f"[INFO] Frame interval jitter (std): {np.std(dt_ms):.4f} ms")

        # indirect drop detection: without FrameID, a genuine dropped frame
        # shows up as a dt_ns outlier roughly >=2x the median interval,
        # since the camera clock keeps ticking even if a frame is lost.
        outlier_thresh = 1.8 * np.median(dt_ns)
        gap_indices = np.where(dt_ns > outlier_thresh)[0]
        if len(gap_indices) > 0:
            print(f"[WARN] {len(gap_indices)} timestamp gap(s) >1.8x median "
                  f"interval detected -- possible dropped frame(s) at index "
                  f"{gap_indices.tolist()} (indirect check; no FrameID counter "
                  f"available to confirm exact count)")
        else:
            print("[INFO] No timestamp gaps >1.8x median interval detected.")

    return frames, frame_ids, timestamps_ns, n_incomplete


def save_session(frames, frame_ids, timestamps_ns, cam, save_dir,
                  condition_label, n_incomplete):
    os.makedirs(save_dir, exist_ok=True)

    frames_path = os.path.join(save_dir, f'flir_frames_{condition_label}.npy')
    meta_path = os.path.join(save_dir, f'flir_chunk_meta_{condition_label}.npz')
    session_path = os.path.join(save_dir, f'flir_session_meta_{condition_label}.json')

    # np.save embeds shape + dtype in its header -- self-describing,
    # unlike a bare .raw file, while still storing unprocessed linear DN.
    np.save(frames_path, frames)
    np.savez(meta_path, frame_id=frame_ids, timestamp_ns=timestamps_ns)

    valid_ts = timestamps_ns[timestamps_ns > 0]
    fps_measured = float(1e9 / np.median(np.diff(valid_ts))) if len(valid_ts) > 1 else None

    session_meta = {
        'condition': condition_label,
        'width': int(cam.Width.GetValue()),
        'height': int(cam.Height.GetValue()),
        'pixel_format': cam.PixelFormat.GetCurrentEntry().GetSymbolic(),
        'exposure_us': float(cam.ExposureTime.GetValue()),
        'gain_db': float(cam.Gain.GetValue()),
        'n_frames_requested': int(len(frame_ids)),
        'n_frames_incomplete': int(n_incomplete),
        'fps_measured_from_timestamps': fps_measured,
        'frames_file': frames_path,
        'chunk_meta_file': meta_path,
    }
    with open(session_path, 'w') as f:
        json.dump(session_meta, f, indent=2)

    print(f"[INFO] Saved: {frames_path}")
    print(f"[INFO] Saved: {meta_path}")
    print(f"[INFO] Saved: {session_path}")


def main():
    condition_label = input(
        "Condition label (e.g. 'background', 'r5cm_maxthrottle'): "
    ).strip() or "condition"
    n_frames = int(input("Number of frames to capture (min 300 recommended): ") or 300)
    exposure_us = input(
        f"Exposure time in us (Enter for validated default {DEFAULT_EXPOSURE_US:.0f}): "
    ).strip()
    exposure_us = float(exposure_us) if exposure_us else None
    gain_db = input(
        f"Gain in dB (Enter for validated default {DEFAULT_GAIN_DB:.0f}): "
    ).strip()
    gain_db = float(gain_db) if gain_db else None
    fps_target = input(
        f"Target fps (Enter for validated default {DEFAULT_FPS_TARGET:.0f}): "
    ).strip()
    fps_target = float(fps_target) if fps_target else DEFAULT_FPS_TARGET
    save_dir = input(
        "Save folder (Enter for 'flir_captures'): "
    ).strip() or "flir_captures"

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("[ERROR] No FLIR camera detected. Is SpinView fully closed?")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    cam = cam_list.GetByIndex(0)

    try:
        cam.Init()
        configure_camera(cam, exposure_us, gain_db, fps_target)
        enabled_chunks = enable_chunk_data(cam)

        frames, frame_ids, timestamps_ns, n_incomplete = acquire_stack(
            cam, n_frames, save_dir, condition_label,
            frameid_available=('FrameID' in enabled_chunks)
        )

        save_session(frames, frame_ids, timestamps_ns, cam, save_dir,
                     condition_label, n_incomplete)

    finally:
        try:
            cam.DeInit()
        except PySpin.SpinnakerException as e:
            print(f"[WARN] DeInit failed (camera may already be released): {e}")
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        print("[INFO] Camera released.")


if __name__ == "__main__":
    main()