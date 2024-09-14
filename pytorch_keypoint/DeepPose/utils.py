import cv2
import numpy as np


def draw_keypoints(img: np.ndarray, coordinate: np.ndarray, save_path: str, radius: int = 3, is_rel: bool = False):
    """

    Args:
        img:
        coordinate: (n,2)
        save_path:
        radius:
        is_rel: 是否是相对坐标，是的话就需要乘以图片尺度得到绝对坐标。

    Returns:

    """
    coordinate_ = coordinate.copy()
    if is_rel:
        h, w, c = img.shape
        coordinate_[:, 0] *= w
        coordinate_[:, 1] *= h
    coordinate_ = coordinate_.astype(np.int64).tolist()

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for x, y in coordinate_:
        cv2.circle(img_bgr, center=(x, y), radius=radius, color=(255, 0, 0), thickness=-1)

    cv2.imwrite(save_path, img_bgr)
