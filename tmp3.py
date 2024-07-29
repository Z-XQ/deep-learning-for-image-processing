import os
import shutil

if __name__ == '__main__':
    res_data_path = r"D:\zxq\data\original\6QA\2024-04-20\通光孔\res\fail\delete\delete"
    extract_path = res_data_path + "_ext"
    os.makedirs(extract_path, exist_ok=True)

    orig_path = r"D:\zxq\data\original\6QA\2024-04-20\通光孔\orig"

    for root, dirs, files in os.walk(res_data_path):
        for fileName in files:

            orig_image_name = fileName
            # orig_image_name = fileName.replace("有无检测异常", "")
            orig_image_path = os.path.join(orig_path, orig_image_name)

            extract_image_path = os.path.join(extract_path, orig_image_name)
            shutil.copy(orig_image_path, extract_image_path)

