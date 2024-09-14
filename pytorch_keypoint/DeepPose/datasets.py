import os
from typing import List, Tuple

import cv2
import torch
import torch.utils.data as data
import numpy as np


class WFLWDataset(data.Dataset):
    """
    https://wywu.github.io/projects/LAB/WFLW.html

    dataset structure:

    ├── WFLW_annotations
    │   ├── list_98pt_rect_attr_train_test
    │   └── list_98pt_test
    └── WFLW_images
        ├── 0--Parade
        ├── 1--Handshaking
        ├── 10--People_Marching
        ├── 11--Meeting
        ├── 12--Group
        └── ......
    """
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transforms=None):
        super().__init__()

        # 原图路径
        self.img_root = os.path.join(root, "WFLW_images")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)

        # 标注文件完整路径
        ana_txt_name = "list_98pt_rect_attr_train.txt" if train else "list_98pt_rect_attr_test.txt"
        self.anno_path = os.path.join(root, "WFLW_annotations", "list_98pt_rect_attr_train_test", ana_txt_name)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.transforms = transforms
        self.keypoints: List[np.ndarray] = []
        self.face_rects: List[List[int]] = []
        self.img_paths: List[str] = []
        with open(self.anno_path, "rt") as f:  # 打开标注文件txt
            for line in f.readlines():  # 每一行对应一个人脸关键点集标注
                if not line.strip():
                    continue

                split_list = line.strip().split(" ")  # 将当前行使用空格符分割成一个一个字符串.
                keypoint_ = self.get_98_points(split_list)  # 前98个是关键点坐标
                keypoint = np.array(keypoint_, dtype=np.float32).reshape((-1, 2))  # list(196) to array(98,2)
                face_rect = list(map(int, split_list[196: 196 + 4]))  # 人脸外接矩形：xmin, ymin, xmax, ymax
                img_name = split_list[-1]  # 文件名称：**.jpg

                # 记录当前标注信息
                self.keypoints.append(keypoint)
                self.face_rects.append(face_rect)
                self.img_paths.append(os.path.join(self.img_root, img_name))

    @staticmethod
    def get_5_points(keypoints: List[str]) -> List[float]:
        five_num = [76, 82, 54, 96, 97]
        five_keypoint = []
        for i in five_num:
            five_keypoint.append(keypoints[i * 2])
            five_keypoint.append(keypoints[i * 2 + 1])
        return list(map(float, five_keypoint))

    @staticmethod
    def get_98_points(keypoints: List[str]) -> List[float]:
        return list(map(float, keypoints[:196]))

    @staticmethod
    def collate_fn(batch_infos: List[Tuple[torch.Tensor, dict]]):
        imgs, ori_keypoints, keypoints, m_invs = [], [], [], []
        for info in batch_infos:
            imgs.append(info[0])
            ori_keypoints.append(info[1]["ori_keypoint"])
            keypoints.append(info[1]["keypoint"])
            m_invs.append(info[1]["m_inv"])

        imgs_tensor = torch.stack(imgs)  # imgs组成一个batch, (b,3,256,256)
        keypoints_tensor = torch.stack(keypoints)  # (b,98,2)
        ori_keypoints_tensor = torch.stack(ori_keypoints)  # (b,98,2)
        m_invs_tensor = torch.stack(m_invs)  # (b,2,3)

        targets = {"ori_keypoints": ori_keypoints_tensor,  # (b,98,2)
                   "keypoints": keypoints_tensor,   # (b,98,2)
                   "m_invs": m_invs_tensor}   # (b,2,3)
        return imgs_tensor, targets

    def __getitem__(self, idx: int):
        img_bgr = cv2.imread(self.img_paths[idx], flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        target = {
            "box": self.face_rects[idx],
            "ori_keypoint": self.keypoints[idx],
            "keypoint": self.keypoints[idx]
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.keypoints)


if __name__ == '__main__':
    train_dataset = WFLWDataset(r"D:\data\WFLW", train=True)
    print(len(train_dataset))

    eval_dataset = WFLWDataset(r"D:\data\WFLW", train=False)
    print(len(eval_dataset))

    from utils import draw_keypoints
    img, target = train_dataset[0]
    keypoint = target["keypoint"]
    h, w, c = img.shape
    # 转成相对坐标。
    keypoint[:, 0] /= w
    keypoint[:, 1] /= h
    draw_keypoints(img, keypoint, "test_plot.jpg", is_rel=True)
