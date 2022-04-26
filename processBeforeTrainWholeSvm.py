# 68-points
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# 2017-12-28
# By TimeStamp
# #cnblogs: http://www.cnblogs.com/AdaminXie/
import dlib  # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy

#添加标签
def writeTrainText(d,p):
    #对某个已经分好类，命名好的文件夹内的图片进行标注
     dirName = d
     picLable = p
     writeToText=""
     for filename in os.listdir(r"E:\\the_face_recongnise\\face_database\\" + dirName):
        # filename.sort(key=lambda x: int(x.split('.')[0]))
        writeToText=writeToText+filename+" "+picLable+"\n"
        if dirName=="red":
            # print(writeToText)
            print(filename)
          # print(writeToText+" yzj ")
     with open('E:\\the_face_recongnise\\face_database\\'+dirName+'.txt', 'w') as f:
          f.writelines(writeToText)
     #print(writeToText)

def tabkeLable():
     #依此生成所有文件夹中图片对应的label
     dirList = ["cyan", "red", "yellow", "white", "black", "normal"]
     number = ["1", "2", "3", "4", "5", "6"]
     j = 0
     for d in dirList:
          # print(d)
          writeTrainText(d, number[j])
          j = j + 1

# tabkeLable()

#图片处理
def changpic(path,x,y,w,h,savePath):
    img = Image.open(path)
    print(img.size)
    img = img.crop((x, y, x + w, y + h))  # 剪裁图片

    img = img.resize((128, 128))  # 将图像变成128*128大小的
    img.save(savePath)
    plt.figure()  # 用plt打开图片
    plt.imshow(img)
    plt.show()
def detecface(path,savePath):
    img = cv2.imread(path)
    color = (0, 255, 0)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    classfier = cv2.CascadeClassifier("D:\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")
    faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects) > 0:  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            changpic(path,x,y,w,h,savePath)
            #cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)  # 5控制绿色框的粗细

    else:print(path)

def getSuqareFace():
    for fileName in os.listdir(r"E:\\the_face_recongnise\\gross picture\\"):
        savePath = "E:\\the_face_recongnise\\gross picture\\" + fileName
        detecface("E:\\the_face_recongnise\\gross picture\\" + fileName,savePath)
        #对每一张图片都进行一个切割形成方形图片

def turnTo128(path,savePath):
    img = Image.open(path)
    print(img.size)
    img = img.resize((128, 128))  # 将图像变成128*128大小的
    print(img.size)
    img.save(savePath)

def get128Face():
    for fileName in os.listdir(r"E:\\the_face_recongnise\\gross picture\\" ):
        # print("C:\\Users\\余张建\\Desktop\\PycharmProjects\\face_suqare_database\\" + dirName + "\\" + fileName)
        savePath = "E:\\the_face_recongnise\\gross picture\\"+fileName
        turnTo128("E:\\the_face_recongnise\\gross picture\\" + fileName,savePath)
getSuqareFace()
# get128Face()
