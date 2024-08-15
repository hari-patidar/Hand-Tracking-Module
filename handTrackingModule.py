import cv2
import mediapipe as mp
import time

class handTracker:

    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            print(self.results.multi_hand_landmarks)
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, handNo = 0, draw = True):
        lmList = []

        if self.results.multi_hand_landmarks:
            mainHand = self.results.multi_hand_landmarks[handNo]

            for id, landmarks in enumerate(mainHand.landmark):
                height, width, channels = img.shape
                posX, posY = int(landmarks.x * width), int(landmarks.y * height)

                lmList.append([id, posX, posY])

                if draw:
                    cv2.circle(img, (posX, posY), 8, (255, 0, 0))
        print("landmark:", lmList)
        return lmList