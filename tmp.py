# _*_ coding:utf-8 _*_
import os


def change_one_file(cpp_path, old_ex, new_ex):
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']  # 按照可能的编码方式顺序进行尝试
    for encode in encodings:
        try:
            file_name = os.path.basename(cpp_path)
            if file_name.split(".")[-1] == old_ex:
                print(cpp_path)
                all_content = []

                with open(cpp_path, 'r', encoding=encode) as f:  # 逐行读取
                    while True:
                        line = f.readline()
                        if line:
                            all_content.append(line)  # 后缀是“ing”的词保存到列表中
                        else:
                            break

                # 写之前，先检验文件是否存在，存在就删掉
                save_path = cpp_path.replace(".{}".format(old_ex), ".{}".format(new_ex))
                if os.path.exists(save_path):
                    os.remove(save_path)
                # 以写的方式打开文件，如果文件不存在，就会自动创建
                file_write_obj = open(save_path, 'w', encoding=encode)  # 新文件
                for cur_line in all_content:
                    file_write_obj.write(cur_line)  # 逐行写入
                file_write_obj.close()
                print("保存文件成功")
        except UnicodeDecodeError:
            continue


def change(file_path, old_ex="h", new_ex="pts"):
    for root, dirs, files in os.walk(file_path):
        for file in files:
            cpp_path = os.path.join(root, file)
            change_one_file(cpp_path, old_ex, new_ex)


def yolo(file_path, old_ex="h"):
    new_ex = "new_" + old_ex
    change(file_path, old_ex=old_ex, new_ex=new_ex)
    change(file_path, old_ex=new_ex, new_ex=old_ex)
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(new_ex):
                cpp_path = os.path.join(root, file)
                os.remove(cpp_path)


if __name__ == '__main__':
    src_path = r"D:\BaiduNetdiskDownload\ImageView\ImageView"
    change(src_path, old_ex="newcss", new_ex="css")

    # file_full_path = r"D:\BaiduNetdiskDownload\ImageView\ImageView"
    # change_one_file(file_full_path, "newh", "h")

    # src_path = r"F:\zxq\code\E\ImageView\ImageView"
    # yolo(src_path, old_ex="cpp")
