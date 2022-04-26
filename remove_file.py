import os.path
import shutil

# filepath = input("请输入移出目录名称")  # 移出目录
# aimpath = input("请输入目标目录名称")
#   # 将总目录中的文件移动到固定目录
# if __name__ == '__main__':
#
#     # filepath1="E:\\the_face_recongnise\\"+filepath
#     aimpath="E:\\the_face_recongnise\\"+aimpath
#     filenames = os.listdir(filepath+"\\")
#     print(filenames)
#     dirList = ["cyan", "red", "yellow", "white", "black", "normal"]
#     with open(aimpath+".txt", 'r') as f:  #读出要移动的文件
#         n=f.read().split("\n")
#         # print(n)
#         for files in n:  #检查是否存在文件
#             if files!="":
#                 name=files.split(" ")[0]
#                 filename=filepath+"\\"+name
#                 print(filename)
#                 shutil.copy(filename, aimpath+"\\")
#



  # 将五个文件夹中的文件集合到一个文件夹中
if __name__ == '__main__':

    dirList = ["cyan", "red", "yellow", "white", "black", "normal"]
        # print((n))
    for dir in dirList:
        filenames = os.listdir("E:\\the_face_recongnise\\face_database\\"+dir)
        for context in filenames:
            # print(context)
            context = "E:\\the_face_recongnise\\face_database\\" + dir+"\\"+context
            name = context.split(" ")[0]
            print(context)
            # filename=filepath+name
            # print(filename)
            shutil.copy(context, "E:\\the_face_recongnise\\gross picture\\")