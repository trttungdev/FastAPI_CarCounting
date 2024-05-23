import os
import numpy as np
import cv2
from collections import deque

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

object_counter = {}

object_counter1 = {}

line = [(100, 500), (1050, 500)]


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (128, 0, 0)
    elif label == 3:  # Motobike
        color = (0, 255, 255)
    elif label == 5:  # Bus
        color = (255, 0, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top left
    cv2.rectangle(img, (x1 + r, y1), (x1 + d, y1 + thickness), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x1 + thickness, y1 + d), color, -1)
    # Top right
    cv2.rectangle(img, (x2 - r, y1), (x2 - d, y1 + thickness), color, -1)
    cv2.rectangle(img, (x2 - thickness, y1 + r), (x2, y1 + d), color, -1)
    # Bottom left
    cv2.rectangle(img, (x1 + r, y2 - thickness), (x1 + d, y2), color, -1)
    cv2.rectangle(img, (x1, y2 - d), (x1 + thickness, y2 - r), color, -1)
    # Bottom right
    cv2.rectangle(img, (x2 - r, y2 - thickness), (x2 - d, y2), color, -1)
    cv2.rectangle(img, (x2 - thickness, y2 - d), (x2, y2 - r), color, -1)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)
    
    return img



def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str
def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
          direction = get_direction(data_deque[id][0], data_deque[id][1])
          if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
              cv2.line(img, line[0], line[1], (255, 255, 255), 3)
              if "South" in direction:
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
              if "North" in direction:
                if obj_name not in object_counter1:
                    object_counter1[obj_name] = 1
                else:
                    object_counter1[obj_name] += 1
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    
        # Define some constants for layout
        text_offset_x = 20
        text_offset_y = 25
        line_spacing = 50
        box_width = 400
        box_height = 150

        # Create gradient background
        gradient = np.zeros((box_height, box_width, 3), dtype=np.uint8)
        gradient[:] = np.linspace((80, 40, 200), (150, 150, 200), box_width)[np.newaxis, :]
        cv2.rectangle(img, (width - box_width, 0), (width, box_height), (255, 255, 255), -1)
        cv2.addWeighted(gradient, 0.6, img[0:box_height, width - box_width:width], 0.4, 0, img[0:box_height, width - box_width:width])

        cv2.rectangle(img, (0, 0), (box_width, box_height), (255, 255, 255), -1)
        cv2.addWeighted(gradient, 0.6, img[0:box_height, width - box_width:width], 0.4, 0, img[0:box_height, width - box_width:width])

        # Display counts for Vehicles Enter
        cv2.rectangle(img, (width - box_width, 0), (width, box_height), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Vehicles Enter', (width - box_width + text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = f'{key}: {value}'
            cv2.rectangle(img, (width - box_width + text_offset_x, 80 + idx * line_spacing), (width - text_offset_x, 80 + (idx + 1) * line_spacing), (200, 200, 200), -1)
            cv2.putText(img, cnt_str, (width - box_width + text_offset_x + 10, 110 + idx * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), thickness=2)

        # Display counts for Vehicles Leave
        cv2.rectangle(img, (0, 0), (box_width, box_height), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Vehicles Leave', (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str = f'{key}: {value}'
            cv2.rectangle(img, (text_offset_x, 80 + idx * line_spacing), (box_width - text_offset_x, 80 + (idx + 1) * line_spacing), (200, 200, 200), -1)
            cv2.putText(img, cnt_str, (text_offset_x + 10, 110 + idx * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), thickness=2)
    
    
    return img