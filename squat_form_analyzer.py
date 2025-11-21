import cv2 as cv
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    # 
    # Utility Functions for Program
    # 

    def resized_window(default_window_height: int, default_window_width: int, a: int) -> list[int]:
        return [int(default_window_height * 1.5 + a), int(default_window_width * 1.5 + a)]

    def chosen_knee(a, b):
        choice = None
        side = None

        if a.z < b.z and abs(a.z - b.z) > 0.05:
            choice = a
            side = "left"

        if b.z < a.z and abs(a.z - b.z) > 0.05:
            choice = b
            side = "right"

        if choice is None:
            if a.visibility > b.visibility:
                choice = a
                side = "left"
            if b.visibility > a.visibility:
                choice = b
                side = "right"

        if choice is None:
            side, choice = last_state

        return side, choice

    def detect_valgus(hip_width, knee, midline, side, threshold=0.13):
        direction_inward = False
        severity = abs(knee.x - midline) / hip_width

        lowerBound = midline - threshold
        upperBound = midline + threshold

        if side == "left" and midline < knee.x < upperBound:
            direction_inward = True
        if side == "right" and lowerBound < knee.x < midline:
            direction_inward = True

        valgus = direction_inward and severity > threshold
        return valgus, severity

    def calculate_angles(a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        ba = a - b
        bc = c - b

        cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        theta = np.degrees(np.arccos(cos_theta))
        return theta

    def squatform(theta, currentknee):
        if 60 < theta < 95:
            Put_text(image, "Good Squat",
                     tuple(np.multiply([currentknee.x, currentknee.y], [200, 100]).astype(int)))

    def drawbox(img, text_lines, position, padding=10):
        max_width = 0
        total_height = 0

        for text in text_lines:
            (text_width, text_height), baseline = cv.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
            max_width = max(max_width, text_width)
            total_height += text_height + baseline

        box_x, box_y = position
        box_width = max_width + padding * 2
        box_height = total_height + padding * 2

        cv.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)

        y = box_y + padding
        for text in text_lines:
            (text_width, text_height), baseline = cv.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
            cv.putText(img, text, (box_x + padding, y + text_height),
                       FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            y += text_height + baseline

    def get_state(angle):
        if angle >= 160:
            return "s1"
        elif angle >= 120:
            return "s2"
        return "s3"

    def Put_text(img, text, position):
        cv.putText(img, str(text), position,
                   cv.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (225, 225, 225), 2, cv.LINE_AA)

    def detect_mode(leftShoulder, rightShoulder):
        shoulder_width = abs(leftShoulder.x - rightShoulder.x)
        if shoulder_width > 0.11:
            return "Front Mode"
        return "Side Mode"

    
    # Global Variables
    
    # FONT Variables
    FONT = cv.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    FONT_COLOR = (255, 255, 255)
     #Squat logic variables
    last_state = None
    rep_count = 0
    bad_rep_count = 0
    squat_stage = []
    last_stage = None
    Red_Warning_Color = (255,0,0)
    Green_Safe_Warning = (0,225,0)

    capture = cv.VideoCapture(0)

    
    # Main Loop
    

    while capture.isOpened():

        isTrue, frame = capture.read()
        height, width = frame.shape[:2]
        chged_height, chged_width = resized_window(height, width, 0)
        frame_dims = [chged_width, chged_height]

        frame = cv.resize(frame, (chged_width, chged_height))
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            landmarks = results.pose_landmarks.landmark

            # Joints
            leftHip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            leftKnee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            leftAnkle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            leftShoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            rightHip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            rightKnee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            rightAnkle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            rightShoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            #DETECT MODE
            mode = detect_mode(leftShoulder, rightShoulder)

            #SHARED SIDE LOGIC

            side, current_knee = chosen_knee(leftKnee, rightKnee)
            last_state = (side, current_knee)

            if side == "left":
                hip = leftHip
                ankle = leftAnkle
                shoulder = leftShoulder
            else:
                hip = rightHip
                ankle = rightAnkle
                shoulder = rightShoulder

            theta = calculate_angles(hip, current_knee, ankle)
            current_stage = get_state(theta)

            # store states except s1
            if current_stage != last_stage and current_stage != "s1":
                squat_stage.append(current_stage)

            # evaluate reps on return to s1
            if current_stage == "s1":
                if squat_stage == []:
                    pass
                else:
                    if mode == "Front Mode":
                        rep_count = 0
                        bad_rep_count = 0
                    else:
                        if "s2" in squat_stage and "s3" not in squat_stage:
                            bad_rep_count += 1
                        else:
                            first_s2 = squat_stage.index("s2")
                            first_s3 = squat_stage.index("s3")
                            if first_s3 > first_s2 and "s2" in squat_stage[first_s3 + 1]:
                                rep_count += 1
                            else:
                                bad_rep_count += 1
                squat_stage = []

            last_stage = current_stage

            #SIDE MODE
            if mode == "Side Mode":

                Put_text(
                    image,
                    round(theta, 2),
                    tuple(np.multiply([current_knee.x, current_knee.y], frame_dims).astype(int))

                )

                

               #  theta2 = calculate_angles(shoulder, hip, current_knee)
               #  Put_text(
               #      image,
               #      round(theta2, 2),
               #      tuple(np.multiply([hip.x, hip.y], frame_dims).astype(int))
               #  )

                drawbox(
                    image,
                    [
                        "SquatCoach AI",
                        f"Mode: {mode}",
                        f"Good Reps: {rep_count}, Bad reps {bad_rep_count}",
                        f"Stage List: {squat_stage}"
                    ],
                    (0, 0)
                )

            #FRONT MODE
            elif mode == "Front Mode":

                hip_width = abs(rightHip.x - leftHip.x)
                midline = (leftHip.x + rightHip.x) / 2

                valgus_left, l_severity = detect_valgus(hip_width, leftKnee, midline, "left")
                valgus_right, r_severity = detect_valgus(hip_width, rightKnee, midline, "right")

                knee_pixels_coords_l = (int(leftKnee.x * chged_width),  int(leftKnee.y * chged_height))
                cv.circle(image, knee_pixels_coords_l, 12, Green_Safe_Warning, -1)

                knee_pixels_coords_r = (int(rightKnee.x * chged_width), int(rightKnee.y * chged_height))
                cv.circle(image, knee_pixels_coords_r, 12, Green_Safe_Warning, -1)

                if valgus_left and current_stage == "s3":
                    trueValgus_left = True
                    cv.circle(image, knee_pixels_coords_l, 12, Red_Warning_Color, -1)
                else:
                    trueValgus_left = False
                    
                if valgus_right and current_stage == "s3":
                    trueValgus_right = True
                    cv.circle(image, knee_pixels_coords_r, 12, Red_Warning_Color, -1)
                else:
                    trueValgus_right = False
                    






                drawbox(
                    image,
                    [
                        "SquatCoach AI",
                        f"Mode: {mode}",
                        f"Valgus_L: {trueValgus_left}",
                        f"Valgus_R: {trueValgus_right}"
                    ],
                    (0, 0)
                )
 
               

        except:
            pass

        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imshow("Webcam", image)

        if cv.waitKey(10) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()
