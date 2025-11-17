import cv2 as cv # reads BGR (blue- green0 red)
import mediapipe as mp # reads RGB
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

for landmark in mp_pose.PoseLandmark:
            print(landmark)

# Video Feed
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    # default_window_height = 640
    # default_window_width = 480


    def resized_window(default_window_height:int, default_window_width: int,  a: int) -> list[int]:
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
         if choice == None:
              if a.visibility > b.visibility: 
                   choice = a
                   side = "left"
              if b.visibility > a.visibility:
                   choice = b
                   side = "right"
         if choice == None:
              side, choice = last_state
          
         return side, choice 
    

    

    def detect_valgus(hip_width, knee, midline,side, threshold = 0.13,):
                 valgus = False
                 direction_inward = False
                 severity = abs(knee.x - midline)/ hip_width
                 lowerBound = (midline - threshold)
                 upperBound = (midline + threshold)
                 if side == "left" and midline < knee.x < upperBound:
                           direction_inward = True
                    
                 if side == "right" and lowerBound < knee.x < midline:
                           direction_inward = True
                 
                 valgus = direction_inward and (severity > threshold)

                          

                 return valgus, severity  
    
         
    
    
     
          
          
              
              
              
         
         


         
     
    def calculate_angles(a,b,c):
            a = np.array([a.x,a.y])
            b = np.array([b.x,b.y])
            c = np.array([c.x,c.y])

            ba = a-b
            bc = c-b

            magba = np.linalg.norm(ba)
            magbc = np.linalg.norm(bc)

            dot_product = np.dot(ba,bc)


            cos_theta = dot_product / (magba * magbc)
            theta = np.degrees(np.arccos(cos_theta))
            

            return theta


    def squatform(theta, currentknee):
        if 60< theta < 95:
             
             Put_text(image,"Good Squat", tuple(np.multiply([currentknee.x,currentknee.y],[200,100]).astype(int)))
            # cv.putText(
            #      image,
            #             str("Good Squat"),
            #            tuple(np.multiply([leftKnee.x,leftKnee.y],[640 + 100 ,480 + 100]).astype(int)),
            #            cv.FONT_HERSHEY_SCRIPT_COMPLEX,
            #            1,
            #            (225,225,225),
            #            2,
            #            cv.LINE_AA
            # )

    def gts():
         time.wait(100)

    def drawbox(img: np.ndarray,
                text_lines: list[str],
                position:tuple[int,int],
                padding: int = 10):
          
          max_width = 0
          total_height = 0

          for text in text_lines:
               (text_width,text_height), baseline = cv.getTextSize(text,FONT,FONT_SCALE,FONT_THICKNESS)
               max_width = max(max_width, text_width)
               total_height += text_height + baseline

          box_x, box_y = position
          box_width = max_width + (padding * 2)
          box_height = total_height + (padding * 2)

          color = (0,0,0)
          thickness = -1

          cv.rectangle(img,(box_x,box_y), (box_x + box_width, box_y + box_height),color,thickness )

          y = box_y + padding
          for text in text_lines:
               (text_width,text_height), baseline = cv.getTextSize(text, FONT,FONT_SCALE,FONT_THICKNESS)
               cv.putText(img,text, (box_x + padding, y + text_height),FONT,FONT_SCALE,FONT_COLOR,FONT_THICKNESS)
                     
               y += text_height + baseline
                     





          
          
          

        
           
    def Put_text(img, text,position):
         cv.putText(
              img,
              str(text),
              position,
              cv.FONT_HERSHEY_SCRIPT_COMPLEX,
              1,
              (225,225,225),
              2,
              cv.LINE_AA

         )

     

     

    FONT = cv.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    FONT_COLOR = (255, 255, 255)
 
    last_state = None 
    counter = 0        
    rep_count = 0
    stage = "Up"
    def rep_counter(theta_in,stage_in,rep_count_in):
         if theta_in < 95 and stage_in == "Up":
                 stage_in = "Down"
         if theta_in > 160 and stage_in == "Down":
                 rep_count_in += 1
                 stage_in = "Up"
         return stage_in,rep_count_in
    
    
         
         
    capture = cv.VideoCapture(0)   

    while capture.isOpened():  # boolean that returns true if 
        isTrue, frame = capture.read()
        height, width = frame.shape[:2]
        chged_height, chged_width = resized_window(height, width, 0)
        frame_dims = [chged_width, chged_height]
        # if isTrue and counter == 0:
             
        #      print(f"Height : {height}, width: {width}")
        #      print(f"New height: {chged_height}, new width: {chged_width}")
        #      counter += 1
        frame = cv.resize(frame, (chged_width, chged_height))

        # processing the frame
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #converting so mediapipe understands, laster wew reverse
        image.flags.writeable = False

        results = pose.process(image)
        
        image.flags.writeable = True
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            
        )
        try:
            landmarks = results.pose_landmarks.landmark

            #getting coordinates
            leftHip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            leftKnee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            
            leftAnkle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            leftShoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            rightKnee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            rightHip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            rightAnkle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            rightShoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            

            #Analyzing hte closest knee
            side, current_knee = chosen_knee(leftKnee,rightKnee)
            last_state = side, current_knee
            if side == "left":
                 hip = leftHip
                 ankle = leftAnkle
                 shoulder = leftShoulder
            if side == "right":
                 hip = rightHip
                 ankle = rightAnkle
                 shoulder = rightShoulder
            theta = calculate_angles(hip, current_knee, ankle)
            hip_width = abs(rightHip.x - leftHip.x)
            midline = (leftHip.x +rightHip.x)/ 2 

          
            
                      
               
                 
                         
               
                 
                 
                 
               
          
          
            valgus_left, l_severity = detect_valgus(hip_width,leftKnee,midline,"left")
            valgus_right,r_severity = detect_valgus(hip_width,rightKnee, midline,"right")

            form = squatform(theta, current_knee)

            
            #Rep Counter
            
                 
            stage, rep_count = rep_counter(theta,stage,rep_count)     
                 
            theta2 = calculate_angles(shoulder, hip, current_knee )
            #Displaying text next to the closest knee
            c_kneetext_x = current_knee.x
            c_kneetext_y = current_knee.y
            c_hiptext_x = hip.x
            c_hiptext_y = hip.y


            # cv.putText(image,
            #             str(round(theta,2)),
            #            tuple(np.multiply([leftKnee.x,leftKnee.y],[640,480]).astype(int)),
            #            cv.FONT_HERSHEY_SCRIPT_COMPLEX,
            #            1,
            #            (225,225,225),
            #            2,
            #            cv.LINE_AA

             
            # )

            Put_text(
                    image,
                    round(theta,2),
                    tuple(np.multiply([c_kneetext_x,c_kneetext_y],frame_dims).astype(int))
                    )
            
            Put_text(
                    image,
                    round(theta2,2),
                    tuple(np.multiply([c_hiptext_x,c_hiptext_y],frame_dims).astype(int))
                     )
            
            drawbox(image,
                    ["AI powered Squat Form Analyzer","  ",f"Reps: {rep_count}", f"Stage: {stage}", f"Valgus L: {valgus_left}", f"Valgus R: {valgus_right}"],
                    (0,0)
                    )
          #   Put_text(
          #           image,
          #           f"Reps: {rep_count}| Stage: {stage}| Valgus:[left : {valgus_left}, right: {valgus_right}] ",
          #           [100,100]
          #   )

          #   Put_text(
          #         image,
          #         f"Serverity[left:{round(l_severity,2)}, right:{round(r_severity,2)}]",
          #         [100,150]
          #   )

            # cv.putText(
            #       image,
            #       str(round(theta2, 2)),
            #       tuple(np.multiply([leftHip.x,leftHip.y],[640,480]).astype(int)),
            #       cv.FONT_HERSHEY_SCRIPT_COMPLEX,
            #       1,
            #       (225,225,225),
            #       2,
            #       cv.LINE_AA
            # )

            # cv.putText(
            #      image,
            #             f"Reps: {rep_count}| Stage: {stage}",
            #            tuple(np.multiply([leftKnee.x,leftKnee.y],[640 + 150 ,480 + 150]).astype(int)),
            #            cv.FONT_HERSHEY_SCRIPT_COMPLEX,
            #            1,
            #            (225,225,225),
            #            2,
            #            cv.LINE_AA
            # )


        except: 
             pass


        #displaying output
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        # new_width = 960
        # new_height = 720

        # resized_frame = cv.resize(image, (new_width, new_height))
        cv.imshow("Webcam",image)

        
        #Turn off condition
        if cv.waitKey(10) & 0xFF==ord('d'):
            break

    capture.release()

    cv.destroyAllWindows()




