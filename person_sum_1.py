#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import uuid
import os
from concurrent.futures import ProcessPoolExecutor
from timeit import time
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Queue, Process
from multiprocessing.managers import BaseManager

from kafka import KafkaProducer, KafkaConsumer

from tools.generate_detections import CreateBoxEncoder
from setting.config import VIDEO_NAME, RTMP, ENVIRO, DOCKER_ID, PROCESS_NUM, TIMES, KAFKA_ON, KAFKA_IP, KAFKA_PORT, \
    APISOURCE, VIDEO_CONDE, DOOR_TOPIC, DOOR_IP_PORT, DOOR_HIGH, FX, FY, TOPIC_SHOW, TOPIC_NVR, \
    MAX_AGE
from util.access_door import out_or_in
from util.box_door import is_near_door
from util.log import logger
from util.message import save_to_kafka, get_door
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


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
        high_score_ids = {}
        # if len(tracker.tracks) == 0:
        #     logger.info('sleep {}'.format(SLEEP_TIME))
        #     sleep(SLEEP_TIME)
        #     continue
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
            logger.info('id:{}, score:{}'.format(track_id, track.score))

        for id in miss_ids:
            if id in center_mass.keys():
                disappear_box[id] = center_mass[id]
                del center_mass[id]
        miss_ids.clear()

        # # 进出门判断
        out_or_in(center_mass, door, in_house, disappear_box, in_out_door)
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
        # 当满足条件时候 往前端模块发送人员的信息
        if (last_person_num != person_sum or last_monitor_people != len(box_score_person)) and producer:
            monitor_people_num = len(box_score_person)
            logger.debug("person-sum:{} monitor-people_num:{}".format(person_sum, monitor_people_num))
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
        fps = ((1 / (round(t3 - t1, 4))))
        print('6====={}======{}'.format(os.getpid(), round(round(t3 - t1, 4) * 1000, 2)))
        logger.debug("fps= %f" % (fps))


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
    # q = multiprocessing.Manager().Queue(100)
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
    logger.error('game over')


if __name__ == '__main__':
    start()
