# import layoutparser as lp
import cv2
import numpy as np
from keras.models import Sequential, model_from_json

from utility.config import *

# IOU
def calculate_intersection_area(box, other_box):
    x, y, height, width = box
    x_other, y_other, height_other, width_other = other_box
    intersection_x = max(x, x_other)
    intersection_y = max(y, y_other)
    intersection_width = min(x + width, x_other + width_other) - intersection_x
    intersection_height = min(y + height, y_other + height_other) - intersection_y
    intersection_area = max(0, intersection_width) * max(0, intersection_height)
    return intersection_area

# Set overlap_threshold to 0.1 for more strict overlap detection
def has_overlap(block1, block2, overlap_threshold=0.5):
    x1l, y1l, x1h, y1h = block1[0:4]
    x2l, y2l, x2h, y2h = block2[0:4]
    height1 = y1h - y1l
    height2 = y2h - y2l
    width1 = x1h - x1l
    width2 = x2h - x2l
    intersection_area = calculate_intersection_area(
        [x1l, y1l, height1, width1], [x2l, y2l, height2, width2]
    )
    return intersection_area > overlap_threshold * (
        height1 * width1
    ) or intersection_area > overlap_threshold * (height2 * width2)


def has_overlap_lp(block1, block2, overlap_threshold=0.5):
    x1l, y1l, x1h, y1h = [
        int(block1.block.x_1),
        int(block1.block.y_1),
        int(block1.block.x_2),
        int(block1.block.y_2),
    ]
    x2l, y2l, x2h, y2h = block2[0:4]
    height1 = y1h - y1l
    height2 = y2h - y2l
    width1 = x1h - x1l
    width2 = x2h - x2l
    intersection_area = calculate_intersection_area(
        [x1l, y1l, height1, width1], [x2l, y2l, height2, width2]
    )
    return intersection_area > overlap_threshold * (
        height1 * width1
    ) or intersection_area > overlap_threshold * (height2 * width2)

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iout = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iout > 0

# LAYOUT PARSER DETECTED TABLES
# def get_lp_objs(file):
#     image = cv2.imread(file)
#     image = image[..., ::-1]
#     result = []

#     # Model 2. Tables - TableBank
#     tb_model = lp.Detectron2LayoutModel(
#         config_path="lp://TableBank/faster_rcnn_R_101_FPN_3x/config",
#         # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
#         label_map={0: "Table"},
#     )
#     layout = tb_model.detect(image)
#     # lp.draw_box(image,layout,show_element_type=True)
#     # tb_blocks = lp.Layout([b for b in layout if b.type == 'Table'])
#     print("table_blocks:", end=" ")
#     for blocks in layout:
#         print(blocks, end=" , ")
#         if blocks.type == "Table":
#             result.append(
#                 [
#                     int(blocks.block.x_1),
#                     int(blocks.block.y_1),
#                     int(blocks.block.x_2),
#                     int(blocks.block.y_2),
#                     blocks.type,
#                     int(blocks.block.y_2) - int(blocks.block.y_1),
#                 ]
#             )
#             print(blocks)
#     print()
#     return result

with open(LPMODEL_JSON, "r") as f:
    json_model = f.read()
lpmodel = model_from_json(json_model)
lpmodel.load_weights(LPMODEL_WEIGHTS)
lpmodel.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)


def get_lpmodel_objs(file, result):
    img = cv2.imread(file)
    # result=[]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
    for i, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        if h > 40 or w > 100:
            img_crp = img[y : y + h, x : x + w]
            img_crp = cv2.resize(img_crp, (500, 500))
            img_crp = img_crp / 255
            img_crp = img_crp.reshape([1, 500, 500, 3])
            r = np.argmax(lpmodel.predict(img_crp))
            if r == 0 and not any(has_overlap([x, y, x + w, y + h], b) for b in result):
                result.append([x, y, x + w, y + h, "Image", h])
            # elif r==2 and not any(has_overlap([x,y,x+w,y+h],b) for b in result):
            #   result.append([x,y,x+w,y+h,'Table',h])
    print("Lpmodel Objects: ", end=" ")
    for i in result:
        print(i)
    return result