from ultralytics import YOLO
import cv2
import random

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

choices = ['rock', 'paper', 'scissors']
user_score = 0
ai_score = 0
round_result = ""
current_ai_choice = random.choice(choices)
wait_for_next_round = False

def get_user_choice(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    if boxes.shape[0] == 0:
        return None
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    idx = areas.argmax()
    label = result.names[int(classes[idx])]
    return label.strip().lower()

def determine_winner(user, ai):
    if user is None or ai is None:
        return None
    user = user.strip().lower()
    ai = ai.strip().lower()
    if user == ai:
        return 0 
    if (user == 'rock' and ai == 'scissors') or \
       (user == 'scissors' and ai == 'paper') or \
       (user == 'paper' and ai == 'rock'):
        return 1 
    else:
        return -1 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Reading frames from the camera stopped")
        break

    frame_resized = cv2.resize(frame, (640, 320))
    results = model(frame_resized)
    annotated_frame = results[0].plot()
    display_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

    info_panel = display_frame.copy()
    cv2.rectangle(info_panel, (0, 0), (frame.shape[1], 90), (0,0,0), -1)
    cv2.putText(info_panel, f"Your Score: {user_score}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(info_panel, f"AI Score: {ai_score}", (frame.shape[1]//2 + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(info_panel, round_result, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160,220,255), 2)

    if not wait_for_next_round:
        user_choice = get_user_choice(results[0])
        print(f"[DEBUG] user_choice: {user_choice} | ai_choice: {current_ai_choice}")
        if user_choice is not None:
            result = determine_winner(user_choice, current_ai_choice)
            round_result = f"You: {user_choice.upper()} | AI: {current_ai_choice.upper()} --> "
            if result == 1:
                user_score += 1
                round_result += "You WIN!"
            elif result == -1:
                ai_score += 1
                round_result += "AI WINS!"
            elif result == 0:
                round_result += "Draw "
            else:
                round_result += "Cannot Detect"
            wait_for_next_round = True
        else:
            round_result = "Show your hand in front of camera..."

    else:
        cv2.putText(info_panel, "Press 'n' for next round...", (20, frame.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,216,255), 2)

    cv2.imshow('Rock Paper Scissors - Live Detection', info_panel)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        user_score = 0
        ai_score = 0
        round_result = "Game reset!"
        wait_for_next_round = False
        current_ai_choice = random.choice(choices)
    elif wait_for_next_round and key == ord('n'):
        round_result = ""
        wait_for_next_round = False
        current_ai_choice = random.choice(choices)

cap.release()
cv2.destroyAllWindows()


# >> Github.com/RezaGooner
