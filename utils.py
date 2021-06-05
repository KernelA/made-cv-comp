import json
from typing import Optional, Sequence

import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from matplotlib import pyplot as plt
from matplotlib import patches

from transforms import TORCHVISION_RGB_STD, TORCHVISION_RGB_MEAN


@torch.jit.script
def denormalize_tensor_to_image(tensor_image, mean=TORCHVISION_RGB_MEAN, std=TORCHVISION_RGB_STD):
    """tensor_image is [C x H x W]
    """
    return tensor_image * std[:, None, None] + mean[:, None, None]


def maybe_resize_large_side(img, large_size: int):
    height, width = img.shape[:2]

    aspect_artio = height / width

    if width > large_size or height > large_size:
        if width > height:
            new_width = large_size
            new_height = round(new_width * aspect_artio)
        else:
            new_height = large_size
            new_width = round(new_height / aspect_artio)

        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return img


def simplify_contour(contour, n_corners=4):
    """Бинарный поиск для приближения предсказанной маски 4-хугольником
    """
    n_iter, max_iter = 0, 1000
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            print('simplify_contour didnt coverege')
            return None

        k = (lb + ub) / 2.
        eps = k * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx


def four_point_transform(image, pts):
    """
    Отображаем 4-хугольник в прямоугольник
    Спасибо ulebok за идею
    И вот этим ребятам за реализацию: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """

    rect = order_points(pts)

    tl, tr, br, bl = pts

    width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_1), int(width_2))

    height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_1), int(height_2))

    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# @torch.no_grad()
# def visualize_prediction_plate(file, model, device='cuda', verbose=True, thresh=0.0,
#                                n_colors=None, id_to_name=None):
#     """Визуализируем детекцию (4 точки, bounding box и приближенный по маске контур)
#     """
#     img = Image.open(file)
#     img_tensor = my_transforms(img)
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         predictions = model([img_tensor.to(device)])
#     prediction = predictions[0]

#     if n_colors is None:
#         n_colors = model.roi_heads.box_predictor.cls_score.out_features

#     palette = sns.color_palette(None, n_colors)

#     img = cv2.imread(file, cv2.COLOR_BGR2RGB)
#     h, w = img.shape[:2]
#     image = img

#     blackImg = np.zeros(image.shape, image.dtype)
#     blackImg[:, :] = (0, 0, 0)
#     for i in range(len(prediction['boxes'])):
#         x_min, y_min, x_max, y_max = map(int, prediction['boxes'][i].tolist())
#         label = int(prediction['labels'][i].cpu())
#         score = float(prediction['scores'][i].cpu())
#         mask = prediction['masks'][i][0, :, :].cpu().numpy()
#         name = id_to_name[label]
#         color = palette[label]

#         if verbose:
#             if score > thresh:
#                 print('Class: {}, Confidence: {}'.format(name, score))
#         if score > thresh:
#             crop_img = image[y_min:y_max, x_min:x_max]
#             print('Bounding box:')
#             show_image(crop_img, figsize=(10, 2))

#             # В разных версиях opencv этот метод возвращает разное число параметров
#             # contours,_ = cv2.findContours((mask > TRESHOLD_MASK).astype(np.uint8), 1, 1)
#             _, contours, _ = cv2.findContours((mask > 0.05).astype(np.uint8), 1, 1)
#             approx = simplify_contour(contours[0], n_corners=4)

#             if approx is None:
#                 x0, y0 = x_min, y_min
#                 x1, y1 = x_max, y_min
#                 x2, y2 = x_min, y_max
#                 x3, y3 = x_max, y_max
# #                 points = [[x_min, y_min], [x_min, y_max], [x_max, y_min],[x_max, y_max]]
#             else:
#                 x0, y0 = approx[0][0][0], approx[0][0][1]
#                 x1, y1 = approx[1][0][0], approx[1][0][1]
#                 x2, y2 = approx[2][0][0], approx[2][0][1]
#                 x3, y3 = approx[3][0][0], approx[3][0][1]

#             points = [[x0, y0], [x2, y2], [x1, y1], [x3, y3]]

#             points = np.array(points)
#             crop_mask_img = four_point_transform(img, points)
#             print('Rotated img:')
#             crop_mask_img = cv2.resize(crop_mask_img, (320, 64), interpolation=cv2.INTER_AREA)
#             show_image(crop_mask_img, figsize=(10, 2))
#             if approx is not None:
#                 cv2.drawContours(image, [approx], 0, (255, 0, 255), 3)
#             image = cv2.circle(image, (x0, y0), radius=5, color=(0, 0, 255), thickness=-1)
#             image = cv2.circle(image, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)
#             image = cv2.circle(image, (x2, y2), radius=5, color=(0, 0, 255), thickness=-1)
#             image = cv2.circle(image, (x3, y3), radius=5, color=(0, 0, 255), thickness=-1)

#             image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), np.array(color) * 255, 2)

#     show_image(image)
#     return prediction


def show_image(image, figsize=(16, 9), reverse=True):
    """# Просто показать картинку. С семинара
    """

    plt.figure(figsize=figsize)
    if reverse:
        plt.imshow(image[..., ::-1])
    else:
        plt.imshow(image)

    plt.axis('off')
    plt.show()


# Переводит предсказания модели в текст. С семинара
def decode(pred, alphabet):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], alphabet))
    return outputs


def pred_to_string(pred, alphabet):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([alphabet[c] for c in out])
    return out


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def draw_bbox(image, bboxes_xyxy: Sequence[Sequence[int]], bboxes_pred_xyxy: Optional[Sequence[Sequence[int]]] = None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(image)

    for bbox_xyxy in bboxes_xyxy:
        x1, y1, x2, y2 = bbox_xyxy
        true_rectangle = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="green")
        ax.add_patch(true_rectangle)

    if bboxes_pred_xyxy is not None:
        for pred_xyxy in bboxes_pred_xyxy:
            x1, y1, x2, y2 = pred_xyxy
            pred_rectangle = patches.Rectangle(
                (x1, y2), x2 - x1, y2 - y1, fill=False, edgecolor="red")
            ax.add_patch(pred_rectangle)

    return fig


class npEncoder(json.JSONEncoder):
    """# Чтобы без проблем сериализовывать json. Без него есть нюансы
    """

    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
