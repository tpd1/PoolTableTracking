from ultralytics import YOLO
import cv2
import math

model = YOLO("Models/pool_ball_weights.pt")

cap = cv2.VideoCapture("Input_Videos/pool_ball_test.mp4")
frame_width = 1280
frame_height = 720
fps = 30

classNames = ['black', 'blue', 'cor_pocket', 'cue_ball', 'mid_pocket', 'yellow']
my_colour_red = (0, 0, 255)
my_colour_blue = (255, 0, 0)

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

while True:
    success, img = cap.read()

    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), my_colour_blue, 1)

            # Confidence
            conf = math.ceil((box.conf[0] * 100))

            # Class Name
            cls = int(box.cls[0])
            cv2.putText(img, f'{classNames[cls]} {conf}%', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=my_colour_red, thickness=1)

    out.write(img)

    # cv2.imshow("Image", img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
