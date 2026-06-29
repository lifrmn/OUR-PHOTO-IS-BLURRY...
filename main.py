import os
import sys

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


def result_callback(result, output_image, timestamp_ms):
    del output_image, timestamp_ms

    global latest_result
    latest_result = result


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback,
)


def is_peace_sign(hand_landmarks):
    lm = hand_landmarks

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y

    return index_up and middle_up and ring_down and pinky_down


def main():
    display_width = 720
    display_height = 405

    # Sumber video: argumen pertama bisa berupa path file video,
    # jika kosong gunakan webcam default (index 0).
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        if source == 0:
            print("Webcam tidak tersedia pada device 0.")
            print("Jika menjalankan dari container/VM/Codespaces, akses webcam biasanya tidak tersedia.")
            print("Jalankan script ini di mesin lokal yang punya webcam aktif,")
            print("atau jalankan: python main.py path/ke/video.mp4")
        else:
            print(f"Tidak bisa membuka sumber video: {source}")
        cap.release()
        return 1

    headless = not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    writer = None
    window_name = "Peace Sign Blur v2"

    if headless:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 20.0, (display_width, display_height))
        print("Tidak ada display, mode headless: hasil disimpan ke output.mp4")
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        print("=" * 50)
        print("  TREND: Peace Sign Blur! ✌️")
        print("  Tunjukkan peace → layar blur")
        print("  Lepaskan → layar normal")
        print("  Tekan 'q' untuk keluar")
        print("=" * 50)

    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolusi native: {native_width}x{native_height}")
    print(f"Ukuran proses: {display_width}x{display_height}")

    frame_timestamp = 0
    printed_actual_size = False

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (display_width, display_height))

            if not printed_actual_size:
                height, width = frame.shape[:2]
                print(f"Ukuran frame setelah resize: {width}x{height}")
                printed_actual_size = True

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp += 33
            landmarker.detect_async(mp_image, frame_timestamp)

            peace_detected = False
            if latest_result and latest_result.hand_landmarks:
                for hand_lm in latest_result.hand_landmarks:
                    if is_peace_sign(hand_lm):
                        peace_detected = True
                        break

            if peace_detected:
                frame = cv2.GaussianBlur(frame, (35, 35), 10)

            if headless:
                writer.write(frame)
            else:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())