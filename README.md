# SquatCoach AI

This is an ai squat analyzer!!

SquatCoach AI is a computer-vision tool that analyzes squat form in real time using OpenCV + MediaPipe Pose in Python.
It detects depth, counts reps, identifies bad reps, and flags knee valgus(when the knees cave and collapse in) using only a webcam.

This is the Fall 2025 Ignition Challenge submission for my AI-powered fitness project.


**Why Build this?**

Many beginners in the gym do not know proper form.
As they add weight without proper form, their risk of injury skyrockets, mainly due to knee valgus

Hiring a trainer is expensive

Thats where Squatcoach AI comes in
It analyzes your form live and tells you what your body is doing automatically



**Key Features**
 1. Side Mode – Rep Counting + Depth Tracking

Tracks hip–knee–ankle angles in real time

Recognizes squat stages (s1 → s2 → s3 → s1)

Counts good reps

Detects bad reps (shallow depth or incorrect sequence)

 2. Front Mode – Knee Valgus Detection

Detects inward knee collapse using hip width + knee position

Shows red knee indicators when valgus occurs at the bottom of the squat

Helps prevent injuries for beginners

 3. Automatic Mode Switching

Uses shoulder width to detect whether user is facing the camera

Switches between Side Mode and Front Mode instantly

No buttons, no manual input

4. Real-Time On-Screen Feedback

Angle text

Rep counter

Bad rep counter

Valgus status

Visual indicators (green means safe, red means warning)

Python: Holds all the core logic
OpenCV: allows us to utilize Webcam and on screen drawing
Mediapipe Pose: Landmark detections
Numpy: angle math as well as calulations


**How it Works**

1. Mediapipe tracks 33 pose landmarks

2. The program selects the knee closest to the camera

3. Calculates angle between hip–knee–ankle of the closest knee

4. Classifies squat into 3 states:

  s1 = standing

  s2 = Descending and Acending

  s3 = depth( bottom)

5. When user returns to s1, system determines:

  Good rep if the movement passed s3 properly

  Bad rep if shallow or incorrect sequence

6. In front mode, knee x-position is compared to hip midline

7.If knee crosses inward beyond a threshold, then valgus is detected


**Target Audience**

Beginners learning to squat,

Athletes practicing technique,

People without access to a coach,

Gym-goers wanting instant feedback,

or Developers interested in AI movement analysis,

In the future, I will add more excercises, use on screen and voice feedback, and train my very own model as well as making an app

**Demo Video**
[https://youtu.be/SsT3RmO5uZ4](url)

