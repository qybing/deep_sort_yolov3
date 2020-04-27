#! python3
# _*_ coding: utf-8 _*_
# @Time : 2020/4/26 15:30 
# @Author : Jovan
# @File : box_handler.py
# @desc :
import json

import numpy as np
from kafka import KafkaConsumer

from setting.config import DOOR_HIGH, FX, FY, DOOR_TOPIC, DOOR_IP_PORT
from util.log import logger


def iou(box1, box2):
    '''
    比较两个矩形之间的关系
    Args:
        box1: list [x1, y1, x2, y2]
        box2: list [x1, y1, x2, y2]
    Returns:
        inter: overlapping area
        iou: iou
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return inter, iou


def is_near_door(center_mass, door):
    '''
    Args:
        center_mass: (list) 人体矩形坐标
        door: (list) 门的坐标
    Returns:
        False or True 是否靠近门
    '''
    door_h_half = int(int((door[1] + door[3]) / 2) * DOOR_HIGH)
    near_door = True
    for key, value in center_mass.items():
        # tlwh1 = np.asarray(value[-1], dtype=np.float)
        tlwh1 = np.asarray(value, dtype=np.float)
        x2, y2 = tlwh1[0:2]
        x3, y3 = tlwh1[2:]
        if door[1] < y3 < door[3] and door[0] < x3 < door[2] or y2 <= door_h_half:
            near_door = True
            break
        else:
            near_door = False
    if near_door:
        logger.debug('person near door')
    else:
        logger.debug('person away door')
    return near_door
