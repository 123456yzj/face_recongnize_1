# -*- coding=utf-8 -*-
import glob
import time
import cv2
# print(cv2.path)
from PIL import Image
#from Tools.scripts.treesync import raw_input
from matplotlib import pyplot as plt
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
from skimage.feature import local_binary_pattern
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
train_image_path = 'w_train/'
#测试集图片的位置
test_image_path = 'w_test/'

#训练集标签的位置
train_label_path = 'w_train.txt'
#测试集标签的位置
test_label_path = 'w_test.txt'

size = 128

train_feat_path = 'r_train\\'
test_feat_path = 'r_test\\'
model_path = 'w_model\\'#修改为绝对路径
#获取HSV颜色特征
hlist = [20, 40, 75, 155, 190, 270, 290, 316, 360]
svlist = [21, 178, 255]
def quantilize(h, s, v):
    '''hsv直方图量化'''
    # value : [21, 144, 23] h, s, v
    h = h * 2
    for i in range(len(hlist)):
        if h <= hlist[i]:
            h = i % 8
            break
    for i in range(len(svlist)):
        if s <= svlist[i]:
            s = i
            break
    for i in range(len(svlist)):
        if v <= svlist[i]:
            v = i
            break
    return 9 * h + 3 * s + v
quantilize_ufunc = np.frompyfunc(quantilize, 3, 1) # 自定义ufunc函数，即将quantilize函数转化为ufunc函数，其输入参数为３个，输出参数为１个。
def colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #cv2.imwrite(os.path.join("eee.jpg"), hsv)  # 写入图片文件
    nhsv = quantilize_ufunc(hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]).astype(np.uint8) # 由于frompyfunc函数返回结果为对象，所以需要转换类型
    hist = cv2.calcHist([nhsv], [0], None, [72], [0,71]) # 40x faster than np.histogram
    hist = hist.reshape(1, hist.shape[0]).astype(np.int32).tolist()[0]
    return hist

def getLBP(img):
    # settings for LBP
    radius = 3
    n_points = 8 * radius
    # 转换为灰度图显示
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 处理
    lbp = local_binary_pattern(image, n_points, radius)
    return lbp

#获得图片列表
def get_image_list(filePath,nameList):
    img_list = []
    for name in nameList:
        temp = Image.open(filePath+name)
        img_list.append(temp.copy())
        temp.close()
    return img_list

#提取特征并保存
def get_feat(image_list,name_list,label_list,savePath,size):
    print("调用提取hog特征函数")
    i = 0
    for image in image_list:
        try:
            #如果是灰度图片  把3改为-1
            image = np.reshape(image, (size, size, 3))
            # print(image)
            print("图片数组重塑" + name_list[i])
        except:
            print(name_list[i]+"出错")

            continue
        # gray = rgb2gray(image)/255.0
        # print("将彩色图像转化为灰度图像，灰度化处理功能")
        #提取HOg特征梯度方向的个数选12，每个cell的像素点个数为8*8，每个block的个数为4*4
        #fd = hog(gray, orientations=12, pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualise=False, transform_sqrt=True)

        fd=[]
        hsv = colors(image)   #重点查看
        lbp=getLBP(image)
        # for w in lbp:
        #     for v in w:
        #         s = np.str_(v)
        #         fd = np.concatenate((fd, [s]))

        for v in hsv:
            s = np.str_(v)
            # concatenate((a1, a2, ...), axis=0, out=None)，
            # 将具有相同结构的array序列结合成一个array，axis是拼接方向，
            # 0为横轴，1为纵轴。axis=0，拼接方向为横轴，需要纵轴结构相同
            # ，拼接方向可以理解为拼接完成后数量发生变化的方向。
            fd=np.concatenate((fd,[s]))
        fd = np.concatenate((fd, [label_list[i]]))
        fd_name = name_list[i]+'.feat'
        fd_path = os.path.join(savePath, fd_name) #连接两个或更多的路径名组件
        joblib.dump(fd, fd_path)
        i += 1
    print("Test features are extracted and saved.")

#变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

#获得图片名称与对应的类别
def get_name_label(file_path):
    name_list = []
    label_list = []
    with open(file_path) as f:
        i = 0
        for line in f.readlines():  # 按行读取readlines方法是读取整个文件所有行，保存在一个列表变量中，每行作为一个元素，比较占内存
            if line != "\n":
                i = i + 1
                # print(line.split(' ')[1].split("\n")[0])
                name_list.append(line.split(' ')[0])
                label_list.append(line.split(' ')[1].split("\n")[0])
        # print(i)
    return name_list, label_list

#提取特征
def extra_feat():
    train_name, train_label = get_name_label(train_label_path)#获取train.txt中的数据
    test_name, test_label = get_name_label(test_label_path)
    # print("打印train.txt中的图片名列表（train_name）")
    # print(train_name)
    # print("打印train.txt中的图片标签（train_label）")
    # print(train_label)
    # print("打印test.txt中的图片名列表（test_name）")
    # print(test_name)
    # print("打印test.txt中的图片标签（test_label）")
    # print(test_label)
    train_image = get_image_list(train_image_path, train_name)
    # print("打印获取训练集中的图片信息")
    # print(train_image)
    test_image = get_image_list(test_image_path, test_name)
    #print("打印获取测试集中的图片信息")
    #print(test_image)
    get_feat(train_image, train_name, train_label, train_feat_path, size)
    get_feat(test_image, test_name, test_label, test_feat_path, size)


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
        for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):#查找符合特定规则的文件路径名
            data = joblib.load(feat_path) # 调入本地文件
            fds.append(data[:-1])
            labels.append(data[-1])
        clf = LinearSVC(max_iter=10000)
        clf.fit(fds, labels)
        # 下面的代码是保存模型的
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(clf, model_path+'model')
        #下面的代码是加载模型  可以注释上面的代码   直接进行加载模型  不进行训练
        clf = joblib.load(model_path+'model')
        #print("训练之后的模型存放在model文件夹中")
        # exit()
        train_name, train_label = get_name_label(train_label_path)
        test_name, test_label = get_name_label(test_label_path)
        result_list = []
        #print("利用训练好的模型进行预测")
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            total += 1
            image_name = feat_path.split('\\')[1].split('.feat')[0] # 此处因测试集路径不同，将第一个下标改为了0
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1,-1)).astype(np.float64)
            result = clf.predict(data_test_feat)
            result_list.append(image_name+' '+label_map[int(result[0])]+'\n')
            if int(result[0]) == int(data_test[-1]):
                num += 1
        print("result_partcolor: ")
        for r in result_list:
            print(r)
        write_to_txt(result_list)
        rate = float(num)/total
        t1 = time.time()
        print('准确率是： %f'%rate)
        print('耗时是 : %f'%(t1-t0))

def write_to_txt(list):
    with open('r_result.txt', 'w') as f:
        f.writelines(list)
    print('每张图片的识别结果存放在r_result.txt里面')


if __name__ == '__main__':
    # shutil.rmtree(train_feat_path)  # 递归删除文件夹下所有文件子文件,包括此文件train
    # shutil.rmtree(test_feat_path)
    # shutil.rmtree(model_path)
    mkdir()  # 不存在文件夹就创建在当前项目目录下创建test train文件夹
    print("1、获取特征并保存在文件夹")
    # extra_feat()  # 获取特征并保存在文件夹
    print("2、训练并预测")
    train_and_test() #训练并预测

    #
    # need_input = raw_input('是否手动输入各个信息？y/n\n')
    #
    # if need_input == 'y':
    #     train_image_path = raw_input('请输入训练图片文件夹的位置,如 /home/icelee/image/ 结尾一定要带有分隔符\n')
    #     test_image_path = raw_input('请输入测试图片文件夹的位置,如 /home/icelee/image/ 结尾一定要带有分隔符\n')
    #     train_label_path = raw_input('请输入训练集合标签的位置,如 /home/icelee/train.txt\n')
    #     test_label_path = raw_input('请输入测试集合标签的位置,如 /home/icelee/test.txt\n')
    #     size = int(raw_input('请输入您图片的大小：如64x64，则输入64\n'))
    #
    # need_extra_feat = raw_input('是否需要重新获取特征？y/n\n')
    #
    # if need_extra_feat == 'y':
    #     shutil.rmtree(train_feat_path)
    #     shutil.rmtree(test_feat_path)
    #     mkdir()
    #     extra_feat()#获取特征并保存在文件夹
    #
    # train_and_test() #训练并预测

