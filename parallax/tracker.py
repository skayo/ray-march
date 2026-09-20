"""Webcam head tracker for the parallax demo.

Finds your eyes with MediaPipe's Face Landmarker and sends their midpoint to
the renderer over UDP, one datagram per camera frame, as the text "x y z":
your eye position relative to the screen center in screen half-widths, with
+x to your right, +y up and +z toward you (away from the screen). The
renderer (main.odin) does the smoothing, the re-centering and the fallback to
the mouse, so this script only measures.

Geometry, all pinhole camera: the two iris centers are about IPD_MM apart in
the real world, so their pixel separation gives the distance, and the pixel
offset of their midpoint from the image center, scaled by that distance,
gives the lateral position. The camera's horizontal field of view is the one
number that sets the scale; the defaults suit a MacBook's built-in camera
with a 480 px window on the built-in display. Camera placement relative to
the screen center does not need to be known: the renderer treats your
position when it first hears from us (or when C is pressed) as straight
ahead.

Run from this directory:   .venv/bin/python tracker.py
Options:                   .venv/bin/python tracker.py --help
"""
import argparse
import math
import socket
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

# Face mesh landmark indices. The landmarker model includes the iris points,
# so 468 and 473 are the pupil centers; the eye corners are the fallback.
LEFT_IRIS, RIGHT_IRIS = 468, 473
LEFT_EYE_CORNERS, RIGHT_EYE_CORNERS = (33, 133), (362, 263)


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading face landmarker model to {MODEL_PATH} ...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def eye_points(landmarks, width, height):
    """Pixel coordinates of the left and right pupil (or eye center)."""
    def px(i):
        return landmarks[i].x * width, landmarks[i].y * height

    def mid(a, b):
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    if len(landmarks) > RIGHT_IRIS:
        return px(LEFT_IRIS), px(RIGHT_IRIS)
    return mid(*map(px, LEFT_EYE_CORNERS)), mid(*map(px, RIGHT_EYE_CORNERS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=60.0, help="camera horizontal field of view, degrees")
    ap.add_argument("--ipd-mm", type=float, default=63.0, help="assumed distance between your pupils")
    ap.add_argument("--screen-half-mm", type=float, default=50.0,
                    help="half the physical width of the render window, in mm (480 px on a MacBook display is about 100 mm)")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="multiply lateral motion; 1.0 is geometrically correct, 1.5-2 exaggerates the effect at desk distance")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--hold", type=float, default=2.0,
                    help="keep sending the last position this many seconds after the face is lost")
    ap.add_argument("--preview", action="store_true", help="show the camera image with the eye points")
    ap.add_argument("--seconds", type=float, default=0, help="exit after this long (0 = run until Ctrl-C)")
    args = ap.parse_args()

    model = ensure_model()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        print("camera: no frames. On macOS, allow camera access for your terminal app in "
              "System Settings > Privacy & Security > Camera, then run this again.", file=sys.stderr)
        return 1
    height, width = frame.shape[:2]
    focal_px = (width / 2) / math.tan(math.radians(args.hfov) / 2)
    print(f"camera {args.camera}: {width}x{height}, focal {focal_px:.0f} px; sending to udp://127.0.0.1:{args.port}", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("127.0.0.1", args.port)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    start = time.monotonic()
    last_seen = None      # when we last saw a face
    last_msg = None       # last "x y z" sent
    frames = detections = 0
    last_status = start
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera: frame read failed", file=sys.stderr)
                return 1
            now = time.monotonic()
            frames += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int((now - start) * 1000))

            if result.face_landmarks:
                detections += 1
                left, right = eye_points(result.face_landmarks[0], width, height)
                cx, cy = (left[0] + right[0]) / 2, (left[1] + right[1]) / 2
                ipd_px = math.hypot(left[0] - right[0], left[1] - right[1])
                if ipd_px > 1:
                    z_mm = focal_px * args.ipd_mm / ipd_px
                    # The camera is not mirrored: your right appears on the image's
                    # left, hence the sign on x. Image y grows downward, hence y.
                    x_mm = -(cx - width / 2) / focal_px * z_mm
                    y_mm = -(cy - height / 2) / focal_px * z_mm
                    s = args.screen_half_mm
                    g = args.gain
                    last_msg = f"{x_mm * g / s:.4f} {y_mm * g / s:.4f} {z_mm / s:.4f}"
                    last_seen = now
                if args.preview:
                    for p in (left, right):
                        cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

            # Keep the renderer fed briefly through a dropout so it doesn't snap
            # back to the mouse every time the detector blinks.
            if last_msg is not None and last_seen is not None and now - last_seen <= args.hold:
                sock.sendto(last_msg.encode(), dest)

            if now - last_status >= 2.0:
                fps = frames / (now - last_status)
                state = f"eye {last_msg}" if last_seen is not None and now - last_seen <= args.hold else "no face"
                print(f"{fps:5.1f} fps, {detections}/{frames} frames with a face, {state}", flush=True)
                frames = detections = 0
                last_status = now

            if args.preview:
                cv2.putText(frame, last_msg or "no face", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("head tracker (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.seconds and now - start >= args.seconds:
                break
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
