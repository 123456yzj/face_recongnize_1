# 68-points
# 2017-12-28
# By TimeStamp
# #cnblogs: http://www.cnblogs.com/AdaminXie/
import dlib  # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import matplotlib.pyplot as plt
import cv2  # 图像处理的库OpenCv
from PIL import Image
#读取app转来的一张人脸，识别人脸区域并剪裁，改变像素为128*128

def changpic(path,x,y,w,h):
    img = Image.open(path)
    print(img.size)

    img = img.crop((x, y, x + w, y + h))  # 剪裁图 片

    img = img.resize((128, 128))  # 将图像变成128*128大小的
    img.save("black_10.jpg")  # 保存图片
    plt.figure()  # 用plt打开图片
    plt.imshow(img)
    plt.show()
def detecface(path):
    img = cv2.imread(path)
    color = (0, 255, 0)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img.resize((128, 128))  # 将图像变成128*128大小的
    # img.save("./hsv/new.jpg")  # 保存图片
    classfier = cv2.CascadeClassifier("D:\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")

    faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(faceRects) > 0:  # 大于0则检测到人脸
        # print("进入")
        # print(len(faceRects))
        for faceRect in faceRects:  # 单独框出每一张人脸

            x, y, w, h = faceRect
            changpic(path,x,y,w,h)
            cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)  # 5控制绿色框的粗细
    else:print("error")
detecface("E:\\the_face_recongnise\\face_database\\red\\red_8.jpg")





