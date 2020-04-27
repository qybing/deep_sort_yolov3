#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import uuid
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from timeit import time
import warnings

import cv2
import numpy as np
from PIL import Image
from multiprocessing import Queue, Process
from multiprocessing.managers import BaseManager

from kafka import KafkaProducer

from tools.generate_detections import CreateBoxEncoder
from setting.config import VIDEO_NAME, RTMP, ENVIRO, DOCKER_ID, PROCESS_NUM, TIMES, KAFKA_ON, KAFKA_IP, KAFKA_PORT, \
    APISOURCE, VIDEO_CONDE, DOOR_HIGH, FX, FY, TOPIC_SHOW, TOPIC_NVR, \
    MAX_AGE
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

warnings.filterwarnings('ignore')
logger = logging.getLogger('person-num')
logger.setLevel(logging.DEBUG)
# file_log = logging.FileHandler("TfPoseEstimator.log")
# file_log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] [%(process)d]'
                              ' %(message)s')
# file_log.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(ch)


# logger.addHandler(file_log)
# if ES_ON:
#     handler = Elastic()
#     logger.addHandler(handler)

def iou(box1, box2):
    '''
    box:[top, left, bottom, right]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return inter, iou


def is_near_door(center_mass, door):
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


def out_or_in(center_mass, door, in_house, disappear_box, in_out_door):
    """
    :param center_mass:
    :param door:
    :param in_house:
    :param disappear_box:
    :param in_out_door:
    :return:
    """
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


def get_door(url):
    # logger.info('{} get door box'.format(url))
    # consumer = KafkaConsumer(DOOR_TOPIC, bootstrap_servers=DOOR_IP_PORT, auto_offset_reset='earliest',
    #                          consumer_timeout_ms=2000)
    # doors = []
    # for msg in consumer:
    #     dic = json.loads((msg.value).decode('utf-8'))
    #     if dic.get('videoUrl')[-4:] == url[-4:]:
    #         door = dic.get('door')
    #         if door and len(door) == 2:
    #             print('Two door')
    #             # real_door = door[1] if door[0][0] > door[1][0] else door[0]
    #             real_door = [int((door[0][0] + door[1][0]) / 2),
    #                          int((door[0][1] + door[1][1]) / 2),
    #                          int((door[0][2] + door[1][2]) / 2),
    #                          int((door[0][3] + door[1][3]) / 2), 0]
    #             doors.append(real_door)
    #         if door and len(door) == 1:
    #             doors.append(door[0])
    #         if len(doors) > 50:
    #             doors.pop(0)
    # if doors:
    #     res_door = doors[-1][:-1]
    # else:
    #     res_door = []
    # doors.clear()
    res_door = [650, 261, 706, 379]
    if res_door:
        logger.debug('door box:{}'.format(res_door))
        w = (res_door[2] - res_door[0])
        res_door[0] = int((res_door[0] - 1.60 * w) * FX)
        res_door[1] = int((res_door[1] - w * 0.4) * FY)
        res_door[2] = int((res_door[2] + 1.60 * w) * FX)
        res_door[3] = int((res_door[3] + w * 0.35) * FY)
        logger.info('{} get door box Success'.format(url))
    else:
        res_door = []
        logger.error('{} get door box fail'.format(url))
    return res_door


def main(yolo, url, CreateBoxEncoder, q):
    producer = None
    if KAFKA_ON:
        ip_port = '{}:{}'.format(KAFKA_IP, KAFKA_PORT)
        producer = KafkaProducer(bootstrap_servers=ip_port)
        logger.debug('open kafka')
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    door = get_door(url)
    #    init   var
    center_mass = {}
    miss_ids = []
    disappear_box = {}
    person_list = []
    in_house = {}
    in_out_door = {"out_door_per": 0, "into_door_per": 0}
    only_id = str(uuid.uuid4())
    logger.debug('rtmp: {} load finish'.format(url))
    last_person_num = 0
    last_monitor_people = 0
    while True:
        t1 = time.time()
        if q.empty():
            continue
        frame = q.get()
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, scores_ = yolo.detect_image(image)
        t2 = time.time()
        # print('5====={}======{}'.format(os.getpid(), round(t2 - t1, 4)))
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        logger.debug("box_num: {}".format(len(boxs)))
        features = CreateBoxEncoder.encoder(frame, boxs)
        # score to 1.0 here).
        # detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        detections = [Detection(bbox, scores_, feature) for bbox, scores_, feature in zip(boxs, scores_, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        # 实时人员ID保存
        track_id_list = []

        cv2.rectangle(frame, (door[0], door[1]), (door[2], door[3]), (0, 0, 255), 2)
        door_half_h = int(int((door[1] + door[3]) / 2) * DOOR_HIGH)
        cv2.line(frame, (0, door_half_h), (111111, door_half_h), (0, 255, 0), 1, 1)
        high_score_ids = {}
        for track in tracker.tracks:
            # 当跟踪的目标在未来的20帧未出现,则判断丢失,保存至消失的id中间区
            if track.time_since_update == MAX_AGE:
                miss_id = str(track.track_id)
                miss_ids.append(miss_id)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # 如果人id存在,就把人id的矩形框坐标放进center_mass 否则 创建一个key(人id),value(矩形框坐标)放进center_mass
            track_id = str(track.track_id)
            bbox = track.to_tlbr()
            near_door = is_near_door({track_id: bbox}, door)
            if track.score >= 0.92 and not near_door:
                high_score_ids[track_id] = [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]]

            track_id_list.append(track_id)

            if track_id in center_mass:
                center_ = center_mass.get(track_id)
                if len(center_) > 49:
                    center_.pop(0)
                center_.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            else:
                center_mass[track_id] = [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]]

            # # --------------------------------------------
            # # logger.debug('box1:{}'.format([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]))
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
            x0, y0 = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            cv2.putText(frame, str(round(track.score, 3)), (x0, y0), 0, 0.6, (0, 255, 0), 2)
            # cv2.circle(frame, (x0, y0), 2, (0, 255, 255), thickness=2, lineType=1, shift=0)
            # # --------------------------------------------

            # x0, y0 = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            # w = abs(int(bbox[3]) - int(bbox[1]))
            # h = abs(int(bbox[2]) - int(bbox[0]))
            logger.info('id:{}, score:{}'.format(track_id, track.score))

        for id in miss_ids:
            if id in center_mass.keys():
                disappear_box[id] = center_mass[id]
                del center_mass[id]
        miss_ids.clear()

        # # 进出门判断
        out_or_in(center_mass, door, in_house, disappear_box, in_out_door)
        # near_door = is_near_door(center_mass, door, disappear_id)

        # 相对精准识别人 用来实时传递当前人数
        box_score_person = [scores for scores in scores_ if scores > 0.72]
        person_sum = in_out_door['into_door_per'] - in_out_door['out_door_per']
        # if person_sum <= len(high_score_ids) and not near_door:
        if person_sum <= len(high_score_ids):
            # 当时精准人数大于进出门之差时 来纠正进门人数 并把出门人数置为0
            if person_sum == len(high_score_ids) == 1:
                pass
                # print('person_sum == len(high_score_ids) == 1')
            else:
                logger.warning('reset in_out_door person')
                in_out_door['out_door_per'] = 0
                in_out_door['into_door_per'] = len(high_score_ids)
                in_house.update(high_score_ids)
                # print('high score:{}'.format(high_score_ids))
                logger.warning(
                    '22222222-id: {} after into of door: {}'.format(in_house.keys(), in_out_door['into_door_per']))
                person_sum = len(high_score_ids)
        if in_out_door['into_door_per'] == in_out_door['out_door_per'] > 0:
            in_out_door['into_door_per'] = in_out_door['out_door_per'] = 0
        if len(person_list) > 100:
            person_list.pop(0)
        person_list.append(person_sum)
        # 从url提取摄像头编号
        pattern = str(url)[7:].split(r"/")
        logger.debug('pattern {}'.format(pattern[VIDEO_CONDE]))
        video_id = pattern[VIDEO_CONDE]
        logger.info('object tracking cost {}'.format(time.time() - t1))
        # 当列表中都是0的时候 重置进出门人数和所有字典参数变量
        if person_list.count(0) == len(person_list) == 101:
            logger.debug('long time person is 0')
            in_out_door['into_door_per'] = 0
            in_out_door['out_door_per'] = 0
            in_house.clear()
            logger.warning('All Clear')
        cv2.putText(frame, "person: " + str(person_sum), (40, 40), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "now_per: " + str(len(box_score_person)), (280, 40), 0, 5e-3 * 200, (0, 255, 0), 2)

        # 当满足条件时候 往前端模块发送人员的信息
        if (last_person_num != person_sum or last_monitor_people != len(box_score_person)) and producer:
            monitor_people_num = len(box_score_person)
            logger.debug("person-sum:{} monitor-people_num:{}".format(person_sum, monitor_people_num))
            # if int(time.time()) - last_time >= 1:
            cv2.imwrite("/opt/code/deep_sort_yolov3/image/{}.jpg".format(str(uuid.uuid4())), frame)
            # print('save img success')
            save_to_kafka(TOPIC_SHOW, now, person_sum, url, producer, video_id, monitor_people_num, only_id)
            if last_person_num > 0 and person_sum == 0:
                only_id = str(uuid.uuid4())

            if last_person_num == 0 and person_sum > 0:
                save_to_kafka(TOPIC_NVR, now, person_sum, url, producer, video_id, len(box_score_person), only_id)

            # last_time = int(time.time())
            last_person_num = person_sum
            last_monitor_people = len(box_score_person)
        # 当满足条件时候 往NVR模块发送信息

        logger.info('url:{} into_door_per: {}'.format(url, in_out_door['into_door_per']))
        logger.info('url:{} out_door_per: {}'.format(url, in_out_door['out_door_per']))
        logger.info('url:{} in_house: {}'.format(url, in_house))
        logger.info('url:{} monitor_people_num: {}'.format(url, len(box_score_person)))
        logger.info('url:{} person_sum: {}'.format(url, person_sum))
        logger.info('GPU image load cost {}'.format(time.time() - t1))
        t3 = time.time()
        fps = round(1 / (round(t3 - t1, 4)), 3)
        # print('pid:{}===fps:{}===time:{}'.format(os.getpid(), fps, round(t3 - t1, 4)))
        # print('*' * 30)
        fps = ((1 / (time.time() - t1)))
        logger.debug("fps= %f" % (fps))
        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def save_to_kafka(topic, now, person_sum, url, producer, video_id, monitor_people_num, messageId):
    logger.debug('Upload message save to kafka')
    # if monitor_people_num == 1 and person_sum == 0:
    #     person_sum = monitor_people_num
    msg = {
        "MessgeId": messageId,
        "equipCode": video_id,
        # "warningTime": now,
        "staffChangeTime": now,
        "peopleNumber": person_sum,
        "monitorPeopleNumber": monitor_people_num,
        "imgUrl": "",
        "videoUrl": "",
        'apISource': APISOURCE,
        "comment": "",
    }
    logger.debug(msg)
    msg = json.dumps(msg).encode('utf-8')
    try:
        # future = producer.send(TOPIC, key=KEY.encode('utf-8'), value=msg, partition=PARTITION)
        future = producer.send(topic, value=msg)
        result = future.get(timeout=20)
        logger.debug(result)
        logger.info('save to kafka is Success')
    except Exception as e:
        logger.error(e)
        logger.error('save to kafka is failed')


def get_cameras():
    rtmp = RTMP
    cameras = []
    file = open("camera.conf")
    line = file.readline()
    while line:
        id, rtsp = line.split()
        rtmp_url = rtmp.format(id)
        cameras.append(rtmp_url)
        line = file.readline()
    file.close()
    return cameras


class Mymanager(BaseManager):
    pass


Mymanager.register('YOLO', YOLO)
Mymanager.register('CreateBoxEncoder', CreateBoxEncoder)


def producer(url, q):
    while True:
        i = 0
        logger.debug('rtmp: {} read+'.format(url))
        video_capture = cv2.VideoCapture(url)
        ret_val, image = video_capture.read()
        if False is video_capture.isOpened() or False is ret_val:
            logger.warning('{} url is: {} {}'.format(url, video_capture.isOpened(), ret_val))
            continue
        logger.debug('rtmp: {} load finish'.format(url))
        while True:
            i += 1
            ret, frame = video_capture.read()
            if not ret:
                break
            if i % TIMES != 0 or image is None:
                continue
            if not FX == FY == 1:
                try:
                    logger.debug('{}: {} fps image resize'.format(url, i))
                    frame = cv2.resize(frame, (0, 0), fx=FX, fy=FY)
                except Exception as e:
                    logger.error(e)
                    logger.error('image is bad')
                    break
            if q.full():
                q.get()
            q.put(frame)
            logger.info('{} image save to  queue {}'.format(i, q.qsize()))


def rec_start(url, yolo, CreateBoxEncoder):
    q = Queue(300)
    pro = Process(target=producer, args=(url, q,))
    con = Process(target=main, args=(yolo, url, CreateBoxEncoder, q))
    pro.start()
    con.start()
    pro.join()
    con.join()


def start():
    manager = Mymanager()
    manager.start()
    model_filename = 'model_data/mars-small128.pb'
    CreateBoxEncoder = manager.CreateBoxEncoder(model_filename, batch_size=1)
    yolo = manager.YOLO()
    video_mes = VIDEO_NAME
    if ENVIRO and os.environ[DOCKER_ID]:
        video_mes = video_mes[
                    int(os.environ[DOCKER_ID]) * PROCESS_NUM - PROCESS_NUM:int(os.environ[DOCKER_ID]) * PROCESS_NUM]
    logger.debug('video_url size: {} Total is {}'.format(len(video_mes), video_mes))
    gpu_proccess = PROCESS_NUM
    if 0 < len(video_mes) < gpu_proccess:
        gpu_proccess = len(video_mes)
    logger.debug('proccess num {}'.format(gpu_proccess))
    if len(video_mes) > 0:
        # urls = [video_mes[i:i + step] for i in range(0, len(video_mes), step)]
        logger.debug('proccess loading')
        urls = video_mes
        with ProcessPoolExecutor(max_workers=gpu_proccess) as pool:
            for url in urls:
                pool.submit(rec_start, url, yolo, CreateBoxEncoder)
    else:
        logger.error('No stream was read')
    print('game over')


if __name__ == '__main__':
    start()
