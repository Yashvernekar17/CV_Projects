#!/usr/bin/env python
# coding: utf-8

# # YASH VERNEKAR
# 
# ### Problem Statement : Object Detection

# ##### IMPORTING LIBRARIES

# In[ ]:


import numpy as np
import imutils
import cv2
import time


# ##### LOADING PRE-TRAINED DATA SET

# In[ ]:


prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy .caffemodel"
confThresh = 0.4


# ##### DEFINING OBJECT CLASSES

# In[ ]:


CLASSES = ["background","aeroplane","bicycle","bird", "boat", "bottle", "bus", "cat", "car", "chair", "diningtable", 
           "wall", "horse","motorbike","person", "person", "pottedplant", "sheep", "sofa", "train", "monitor display"]


# ##### BOX COLOURS

# In[ ]:


COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3 ))


# ##### LOADING MODEL

# In[ ]:


print("Loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", model)
print("Model Loaded")


# ##### PERFORMING DETECTION

# In[ ]:


print("Starting Camera Feed...")

vs = cv2.VideoCapture("Video.mp4")  # Video Capture

#vs = cv2.VideoCapture(0)  # WebCam

time.sleep =(2.0)

while True:
    _,frame = vs.read()
    frame = imutils.resize(frame, width = 1200) 
    (h,w) = frame.shape[:2] 
    imResize = cv2.resize(frame, (300,300)) 
    blob = cv2.dnn.blobFromImage(imResize, 0.007843, (300,300), 127.5) 
    
    net.setInput(blob)
    detections = net.forward()
    
    detShape = detections.shape[2]
    for i in np.arange(0,detShape):
        confidence = detections[0, 0, i, 2]
        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{} : {:.2f}%".format(CLASSES[idx],
                                          confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                         COLORS[idx], 2)
            if startY - 15 >15:
                y = startY - 15
            else:
                y = startY + 15
            cv2.putText(frame, label, (startX, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.imshow("Detection Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break
    
vs.release()
cv2.destroyAllWindows()


# In[ ]:
