import json

import cv2
import numpy as np


if __name__ == '__main__':
    bg_image_path = r"D:\zxq\data\df\lingthole\ng\17-31-03-772_{97e1ad45-3cb2-4a45-9b1a-8c7299e843fa}_pass_N0_.png"
    json_path = bg_image_path.replace(".png", ".json")

    bg_image = cv2.imread(bg_image_path)

    json_file = open(json_path, mode="r")
    ct = json.load(json_file)
    shapes = ct["shapes"]

    for shape in shapes:
        points = shape["points"]
        pts_numpy = np.array(points)
        pts_numpy = pts_numpy.astype(np.int32)
        cv2.drawContours(bg_image, pts_numpy.reshape((1, -1, 2)), -1, (0, 0, 255), 1)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL), cv2.imshow("image", bg_image), cv2.waitKey()
