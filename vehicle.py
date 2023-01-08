import cv2
import numpy as np
from PIL import ImageGrab
import time
#import pickle
import torch
# import pyautogui
model=torch.hub.load('ultralytics/yolov5','yolov5s')

last_time = time.time()
while (True):
    screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(250, 150, 1050, 750))), cv2.COLOR_BGR2RGB)
    #print(screen)
    #screen = cv2.resize(screen, (200, 100), fx=4, fy=4)
    #cap=cv2.VideoCapture(0)
    #print(cap.read()[1])
    #ret,frame=screen[0].read()
    results = model(screen)
    x=np.squeeze(results.render())
    cv2.imshow('window', cv2.resize(x, (750, 350), fx=4, fy=4))
    #print(1/(time.time()-last_time))
    #last_time = time.time()
    #print(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows
        break