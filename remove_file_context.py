import os.path
import shutil

filepath = "E:\\the_face_recongnise\\face_database\\"   # 移出目录
aimpath="E:\\the_face_recongnise\\w_train.txt"

if __name__ == '__main__':
    filenames = os.listdir(filepath)
    writeToText1 = ""
    writeToText2 = ""
    dirList = ["cyan", "red", "yellow", "white", "black", "normal"]
    for file in dirList:

        with open("E:\\the_face_recongnise\\face_database\\" +file+".txt" , 'r') as f:
            text=f.readlines()
            length=len(text)
            n=int(length/3)
            for num in range(n,length):
                writeToText1=writeToText1+text[num]+"\n"
            for num in range(n):
                writeToText2 = writeToText2 + text[num] + "\n"
    with open("E:\\the_face_recongnise\\w_train.txt", 'w') as f:
        f.writelines(writeToText1)
    with open("E:\\the_face_recongnise\\w_test.txt", 'w') as f:
        f.writelines(writeToText2)
