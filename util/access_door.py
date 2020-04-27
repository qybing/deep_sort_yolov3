#! python3
# _*_ coding: utf-8 _*_
# @Time : 2020/4/26 16:12 
# @Author : Jovan
# @File : access_door.py
# @desc :


import numpy as np

from setting.config import DOOR_HIGH
from util.box_door import iou
from util.log import logger


def out_or_in(center_mass, door, in_house, disappear_box, in_out_door):
    '''
    判断人员进出门
    Args:
        center_mass: 人员矩形坐标点集合
        door: 门的坐标
        in_house:
        disappear_box:
        in_out_door:
    '''
    door_h_half = int(int((door[1] + door[3]) / 2) * DOOR_HIGH)

    def is_out_door(key, way):
        if in_out_door['into_door_per'] > in_out_door['out_door_per']:
            if key in in_house.keys():
                del in_house[key]
            in_out_door['out_door_per'] += 1
            logger.info('{} id:{} after out of door: {}'.format(way, key, in_out_door['out_door_per']))
        else:
            in_out_door['into_door_per'] = in_out_door['out_door_per'] = 0

    for key, value in disappear_box.items():
        logger.debug('id:{} box length:{}'.format(key, len(value)))
        tlwh0 = np.asarray(value[0], dtype=np.float)
        tlwh1 = np.asarray(value[-1], dtype=np.float)

        dis_x0, dis_y0 = tlwh0[:2]
        dis_x1, dis_y1 = tlwh0[2:]

        dis_x2, dis_y2 = tlwh1[:2]
        dis_x3, dis_y3 = tlwh1[2:]
        near_door = door[1] < dis_y2 < dis_y3 <= door[3] and door_h_half > dis_y2
        real_near_door = door[1] < dis_y2 < dis_y3 <= door[3] and door[0] < dis_x2 < dis_x3 < door[2]
        box0 = [dis_x0, dis_y0, dis_x1, dis_y1]
        box1 = [dis_x2, dis_y2, dis_x3, dis_y3]

        dis_inter0, dis_iou0 = iou(box0, door)
        dis_inter1, dis_iou1 = iou(box1, door)
        if dis_inter1 == 0 and len(value) < 3:
            continue
        # is_has = key not in disappear_id.keys()
        is_has = True
        x_in_door = door[0] < dis_x2 < door[2]

        if dis_inter1 >= dis_inter0 and door_h_half >= dis_y2 \
                and door_h_half >= dis_y0 and door[3] >= dis_y3 \
                and is_has:
            is_out_door(key, 'one')
        elif dis_x2 >= door[0] and dis_y2 >= door[1] and \
                door[3] >= dis_y3 and door_h_half >= dis_y2 and \
                is_has:
            is_out_door(key, 'two')
        elif door_h_half >= dis_y2 and door_h_half >= dis_y0 and \
                door[3] >= dis_y3 and is_has:
            is_out_door(key, 'three')
        elif near_door and is_has:
            is_out_door(key, 'foure')
        elif real_near_door and is_has:
            is_out_door(key, 'five')
        else:
            pass

    disappear_box.clear()
    logger.debug('center_mass:{}'.format(center_mass.keys()))
    for key, value in center_mass.items():
        tlwh0 = np.asarray(value[0], dtype=np.float)
        tlwh1 = np.asarray(value[-1], dtype=np.float)

        x0, y0 = tlwh0[:2]
        x1, y1 = tlwh0[2:]

        x2, y2 = tlwh1[:2]
        x3, y3 = tlwh1[2:]

        box0 = [x0, y0, x1, y1]
        box1 = [x2, y2, x3, y3]
        inter0, iou0 = iou(box0, door)
        inter1, iou1 = iou(box1, door)
        door_wide = door[2] - door[0]
        if inter1 == 0 and inter0 > 0 and door_h_half > y0 and \
                y1 < door[3] and key not in in_house.keys():
            in_house[key] = box0
            in_out_door['into_door_per'] += 1
            logger.info('3333333 id: {} after into of door: {}'.format(key, in_out_door['into_door_per']))

        if inter1 == 0 and inter0 > 0 and door_h_half > y2 and \
                door[1] < y3 < door[3] and door[0] - door_wide < x0 < door[2] + door_wide and \
                key not in in_house.keys():
            in_house[key] = box0
            in_out_door['into_door_per'] += 1
            logger.info('4444444 id: {} after into of door: {}'.format(key, in_out_door['into_door_per']))
