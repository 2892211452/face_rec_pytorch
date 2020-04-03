# -*- coding: utf-8 -*-


import json

import cv2
import torch as torch
from train_model import CNN
from train_model import pre

import threading

class Face_recognition(threading.Thread):

    def run(self):
        print("start")
        self.recongition()
        print('end')



    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

        self.flag =False#用于判断摄像机的使用权在不在后台

       
        # 框住人脸的矩形边框颜色
        self.color = (0, 255, 0)

        # 捕获指定摄像头的实时视频流
        #self.cap = cv2.VideoCapture(0)

        # 人脸识别分类器本地存储路径
        self.cascade_path = "/home/lwl/code/python/opencv/face_rec/haarcascade_frontalface_default.xml"




    def Send_message(self, name):
        print("someone come here:" +name )



    def recongition(self):
        # 捕获指定摄像头的实时视频流
        model =CNN(2)
        model = torch.load('./model.h5')
        self.cap = cv2.VideoCapture(0)

        while True:
            ret, frame = self.cap.read()  # 读取一帧视频

            if ret is True:

                # 图像灰化，降低计算复杂度q
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue
            # 使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(self.cascade_path)

            # 利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(42, 42))

            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect
                    # 截取脸部图像提交给模型识别这是谁
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    #然后处理图像大小

                    #在此处进行预测
                    index, probability = pre(model, image)
                    print(index, probability)

                    #name = self.contrast_table[str(name_number)]
                    if (index ==0):
                        if probability>0.5:
                            name = "lwl"
                            self.Send_message(name)
                      
                    else:  
                        name = "yl"
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), self.color, thickness=2)
                    # 文字提示是谁
                    if probability > 0.5:
                        cv2.putText(frame, name, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                   
            cv2.imshow("face_recognition", frame)

            # 等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            # 如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break

            if self.flag:
                break

        # 释放摄像头并销毁所有窗口
        self.cap.release()
        cv2.destroyAllWindows()


    def ShoutDown(self):
        if self.flag:
            self.flag = False
        else:
            self.flag = True    
    


if __name__ == '__main__':
   
    try:
        fr = Face_recognition(1,1)
        fr1 = Face_recognition(1,1)



        fr.start()

    except:
     print ("Error: unable to start thread")