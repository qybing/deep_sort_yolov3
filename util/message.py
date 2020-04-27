#! python3
# _*_ coding: utf-8 _*_
# @Time : 2020/4/27 9:53 
# @Author : Jovan
# @File : message.py
# @desc :
import json

from kafka import KafkaConsumer

from setting.config import FX, FY, DOOR_TOPIC, DOOR_IP_PORT, APISOURCE, RTMP
from util.log import logger


def get_door(url):
    '''
    从kafka获取门的坐标值
    Args:
        url: str ()
    Returns:
        最近的一条门的坐标 list
    '''
    logger.info('{} get door box'.format(url))
    consumer = KafkaConsumer(DOOR_TOPIC, bootstrap_servers=DOOR_IP_PORT, auto_offset_reset='earliest',
                             consumer_timeout_ms=2000)
    doors = []
    for msg in consumer:
        dic = json.loads((msg.value).decode('utf-8'))
        if dic.get('videoUrl')[-4:] == url[-4:]:
            door = dic.get('door')
            if door and len(door) == 2:
                # real_door = door[1] if door[0][0] > door[1][0] else door[0]
                real_door = [int((door[0][0] + door[1][0]) / 2),
                             int((door[0][1] + door[1][1]) / 2),
                             int((door[0][2] + door[1][2]) / 2),
                             int((door[0][3] + door[1][3]) / 2), 0]
                doors.append(real_door)
            if door and len(door) == 1:
                doors.append(door[0])
            if len(doors) > 50:
                doors.pop(0)
    if doors:
        res_door = doors[-1][:-1]
    else:
        res_door = []
    doors.clear()
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


def save_to_kafka(topic, now, person_sum, url, producer, video_id, monitor_people_num, messageId):
    '''
    上传信息到kafka
    Args:
        topic:
        now:
        person_sum:
        url:
        producer:
        video_id:
        monitor_people_num:
        messageId:
    '''
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
        "url": url
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
