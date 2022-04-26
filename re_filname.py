import os
import re
##将文件的文字更改为指定名字

if __name__ == "__main__":
    colors=["normal","red","yellow","cyan","white","black"]
    filepaths=input("请输入图片夹名称")
    for color in colors:
        filepath = filepaths+"\\"+ color  # 文件夹路径
        filenames = os.listdir(filepath)
        # print("文件数目为%i" % len(filenames))
        count = 0
        i = 1
        for name in filenames:
            name1 = name.split('.jpg')[0].split(".JPG")[0]
            newname = re.sub(name1, '', name)
            newname = color + str(i) + ".jpg"
            os.rename(filepath + '\\' + name, filepath + '\\' + newname)
            i = i + 1
        m = 1
        filenames1 = os.listdir(filepath)
        for name in filenames1:
            newname = color + "_" + str(m) + ".jpg"
            os.rename(filepath + '\\' + name, filepath + '\\' + newname)
            m = m + 1