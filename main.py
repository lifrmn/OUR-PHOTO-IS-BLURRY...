import os
import sys
import warnings
import threading

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import mediapipe as mp

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
result_lock = threading.Lock()


def result_callback(result, output_image, timestamp_ms):
    del output_image, timestamp_ms
    global latest_result
    with result_lock:
        latest_result = result


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    result_callback=result_callback,
)


def is_hand_raised(hand_landmarks):
    """
    Tangan dianggap NAIK jika wrist (titik 0) berada di 65% bagian atas frame.
    Koordinat y dinormalisasi: 0 = atas layar, 1 = bawah layar.
    """
    wrist = hand_landmarks[0]
    return wrist.y < 0.65


def main():
    display_width = 720
    display_height = 405

    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Webcam tidak bisa dibuka. Pastikan kamera aktif dan tidak dipakai aplikasi lain.")
        cap.release()
        return 1

    headless = sys.platform != "win32" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    writer = None
    window_name = "Both Hands Blur"

    if headless:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 20.0, (display_width, display_height))
        print("Mode headless: hasil disimpan ke output.mp4")
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        print("=" * 50)
        print("  Angkat KEDUA tangan  ->  layar BLUR")
        print("  Turunkan tangan      ->  layar NORMAL")
        print("  Tekan 'q' untuk keluar")
        print("=" * 50)

    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolusi kamera: {native_width}x{native_height}")

    frame_timestamp = 0

    # Smoothing: blur hanya aktif setelah N frame berturut-turut terdeteksi
    SMOOTH_FRAMES = 3
    raised_counter = 0
    blur_active = False

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (display_width, display_height))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp += 33
            landmarker.detect_async(mp_image, frame_timestamp)

            with result_lock:
                result = latest_result

            # Kedua tangan terdeteksi DAN keduanya dalam posisi naik
            both_raised = (
                result is not None
                and result.hand_landmarks is not None
                and len(result.hand_landmarks) >= 2
                and all(is_hand_raised(h) for h in result.hand_landmarks)
            )

            # Smoothing: hindari blur kedip-kedip
            if both_raised:
                raised_counter = min(raised_counter + 1, SMOOTH_FRAMES)
            else:
                raised_counter = max(raised_counter - 1, 0)

            blur_active = raised_counter >= SMOOTH_FRAMES

            display_frame = frame.copy()
            if blur_active:
                display_frame = cv2.GaussianBlur(display_frame, (55, 55), 15)

            # Tampilkan jumlah tangan & status di sudut layar
            hand_count = len(result.hand_landmarks) if result and result.hand_landmarks else 0
            status_text = "BLUR AKTIF" if blur_active else "Normal"
            status_color = (0, 0, 255) if blur_active else (0, 255, 0)
            cv2.putText(
                display_frame,
                f"Tangan: {hand_count}/2  |  {status_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )

            if headless:
                writer.write(display_frame)
            else:
                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())