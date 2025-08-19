from ultralytics import YOLO
import cv2
import argparse

def detect_image(model, image_path):
    results = model(image_path)
    result = results[0]
    img = result.orig_img.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Image Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # ✅ Get FPS of the video and calculate correct delay
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # fallback if FPS not available
        fps = 60
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Video Detection", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--video", type=str, help="Path to video file (or 0 for webcam)")
    args = parser.parse_args()

    # model = YOLO("yolov8m.pt")   # or yolov8l.pt
    model = YOLO("yolov8n.pt")   # Nano version, much faster


    if args.image:
        detect_image(model, args.image)
    elif args.video:
        if args.video == "0":
            detect_video(model, 0)  # webcam
        else:
            detect_video(model, args.video)
    else:
        print("Please provide either --image <path> or --video <path/0>")
