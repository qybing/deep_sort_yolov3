#! python3
# _*_ coding: utf-8 _*_
# @Time : 2020/4/26 16:10 
# @Author : Jovan
# @File : log.py
# @desc :
import logging
import warnings

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
