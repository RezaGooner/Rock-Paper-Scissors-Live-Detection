from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

print(" >> Developed and built by RezaGooner << ")
print(" --------------------------------------- ")
print("           Github.com/RezaGooner         ")
print(" --------------------------------------- ")

if not cap.isOpened():
    print("Error >> Camera cannot be opened")
    exit()

print("Camera activated >> Press 'q' to exit")
print("Press 'space' to pause/resume the video capture")

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Reading frames from the camera stopped")
            break

        frame_resized = cv2.resize(frame, (640, 320))

        results = model(frame_resized)

        annotated_frame = results[0].plot()

        display_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
    else:
        display_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

    cv2.imshow('Rock Paper Scissors - Live Detection', display_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
        status = "Paused" if paused else "Resumed"
        print(f"Status: {status}")

cap.release()
cv2.destroyAllWindows()
