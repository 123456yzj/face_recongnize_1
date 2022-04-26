# -*- coding=utf-8 -*-
import glob
import time
import cv2
from PIL import Image
from matplotlib import pyplot as plt

from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
import shutil
# from matplotlib import pyplot

#第一个是你的类别   第二个是类别对应的名称   输出结果的时候方便查看
label_map = {1:'白',
             2:'黑',
             3:'红',
             4:'黄',
             5: '青',
             6: '正常'}
#训练集图片的位置
train_image_path = 'wp_train\\'
#测试集图片的位置
test_image_path = 'wp_test\\'

#训练集标签的位置
train_label_path = 'w_train.txt'
#测试集标签的位置
test_label_path = 'w_test.txt'

size = 128

train_feat_path = 'w_train\\'
test_feat_path = 'w_test\\'
model_path = 'w_model\\'#修改为绝对路径


#获得图片列表
def get_image_list(filePath,nameList):
    img_list = []
    for name in nameList:
        temp = Image.open(filePath+name)
        img_list.append(temp.copy())
        temp.close()
    return img_list

#提取特征并保存
def get_feat(filePath,name,savePath):
    print("调用提取hog特征函数")
#注意，必须要将图片变为128大小的
    try:
        # 如果是灰度图片  把3改为-1
        image = Image.open(name)
        image = np.reshape(image, (size, size, 3))


    except:
        print( "出错")

    gray = rgb2gray(image) / 255.0
    plt.figure()  # 用plt打开图片
    plt.imshow(gray)
    plt.show()
    print("将彩色图像转化为灰度图像，灰度化处理功能")
    # 提取HOg特征梯度方向的个数选12，每个cell的像素点个数为8*8，每个block的个数为4*4

    fd = hog(gray, orientations=12, pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualize=False,
             transform_sqrt=True)
    fd = np.concatenate((fd, ["whatever"]))
    fd_name = name + '.feat'
    fd_path = os.path.join(savePath, fd_name)
    joblib.dump(fd, fd_path)
    print("调用提取hog特征函数over")

#变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

#获得图片名称与对应的类别
def get_name_label(file_path):
    name_list = []
    label_list = []
    with open(file_path) as f:
        for line in f.readlines():#按行读取readlines方法是读取整个文件所有行，保存在一个列表变量中，每行作为一个元素，比较占内存
            name_list.append(line.split(' ')[0])
            label_list.append(line.split(' ')[1])
    return name_list, label_list


#提取特征
def extra_feat():
    train_name, train_label = get_name_label(train_label_path)#获取train.txt中的数据
    test_name, test_label = get_name_label(test_label_path)
    print("打印train.txt中的图片名列表（train_name）")
    print(train_name)
    print("打印train.txt中的图片标签（train_label）")
    print(train_label)
    print("打印test.txt中的图片名列表（test_name）")
    print(test_name)
    print("打印test.txt中的图片标签（test_label）")
    print(test_label)
    train_image = get_image_list(train_image_path, train_name)
    print("打印获取训练集中的图片信息")
    print(train_image)
    test_image = get_image_list(test_image_path, test_name)
    # print("打印获取测试集中的图片信息")
    print(test_image)
    get_feat(train_image, train_name, train_label, train_feat_path, size)
    get_feat(test_image, test_name,  test_feat_path, size)


# 创建存放特征的文件夹
def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)

#训练和测试
def train_and_test():
    t0 = time.time()
    clf_type = 'LIN_SVM'
    fds = []
    labels = []
    num = 0
    total = 0

    if clf_type == 'LIN_SVM':
        # for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat'))://训练
        #     data = joblib.load(feat_path)
        #     fds.append(data[:-1])
        #     labels.append(data[-1])
        # print("Training a Linear LinearSVM Classifier.")
        # clf = LinearSVC()
        # clf.fit(fds, labels)
        # 下面的代码是保存模型的
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        # joblib.dump(clf, model_path+'model')
        #下面的代码是加载模型  可以注释上面的代码   直接进行加载模型  不进行训练#训练结束
        clf = joblib.load(model_path+'model')
        print(clf)
        # print("训练之后的模型存放在model文件夹中")
        # exit()
        # train_name, train_label = get_name_label(train_label_path)
        # test_name, test_label = get_name_label(test_label_path)
        result_list = []
        # resull_
        # print("利用训练好的模型进行预测")
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            total += 1
            print("原路径" + feat_path)
            print(feat_path.split('\\')[0])
            image_name = feat_path.split('\\')[1].split('.feat')[0]#此处因测试集路径不同，将第一个下标改为了0
            print("路径"+image_name)
            data_test = joblib.load(feat_path)
            print(data_test)
            data_test_feat = data_test[:-1].reshape((1,-1)).astype(np.float64)
            result = clf.predict(data_test_feat)
            # result_list.append(image_name+' '+label_map[int(result[0])]+'\n')
            print("hahah"+result[0])
            result_list.append(label_map[int(result[0])] + '\n')
        #     if int(result[0]) == int(data_test[-1]):
        #         num += 1
        # rate = float(num)/total
        # t1 = time.time()
        # print('准确率是： %f'%rate)
        # print('耗时是 : %f'%(t1-t0))
        print(result_list[-1])



if __name__ == '__main__':
    # shutil.rmtree(train_feat_path)
    # shutil.rmtree(test_feat_path)
    # shutil.rmtree(model_path)
    mkdir()  # 不存在文件夹就创建在当前项目目录下创建test train文件夹
    # print("1、获取特征并保存在文件夹")
    # extra_feat()  # 获取特征并保存在文件夹
    # print("2、训练并预测")
    get_feat(test_image_path, "black_10.jpg", test_feat_path)
    train_and_test() #训练并预测


