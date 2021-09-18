#!/usr/bin/env python
# coding: utf-8

# # YASH VERNEKAR
# 
# ### Problem statement : Colour Identification in Images

# ##### IMPORTING REQUIRED LIBRARIES

# In[1]:


import cv2
import pandas as pd
import numpy as np
import imutils


# ##### LOADING AN IMAGE

# In[2]:


img = cv2.imread("color_img.jpeg")
img = imutils.resize(img, width = 500)


# ##### IMPORTING COLOR.csv

# In[3]:


index=["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# ##### DEFINING GLOBAL VARIABLES

# In[4]:


clicked = False
r = g = b = xpos = ypos = 0


# ##### CREATING RECOGNIZE FUNCTION

# In[5]:


def recognize_color(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname


# ##### CREATING MOUSE CLICK FUCTION

# In[6]:


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)


# ##### MAIN PROGRAM

# In[8]:


cv2.namedWindow('Color Recognition')
cv2.setMouseCallback('Color Recognition', mouse_click)

while(1):
    cv2.imshow("Color Recognition",img)
    if (clicked):
   
        
        cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

        text = recognize_color(r,g,b) + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)
        
       
        cv2.putText(img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)

        if(r+g+b>=600):
            cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
            
        clicked=False
        
    
    if cv2.waitKey(20) & 0xFF ==27:
        break
        
cv2.destroyAllWindows()


# # THANK YOU