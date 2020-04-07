"""Miscellaneous utility functions."""
import os
import time
from functools import reduce
import numpy as np
import cv2 as cv2
from PIL import Image


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    t1 = time.time()
    image_w, image_h = image.size
    w, h = size
    t2 = time.time()
    # print('1==={}'.format(round(t2 - t1, 4) * 1000))
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    t3 = time.time()
    # print('2==={}'.format(round(t3 - t2, 4) * 1000))

    resized_image = image.resize((new_w,new_h), Image.BICUBIC)
    # result = cv2.cvtColor(np. asarray(image), cv2.COLOR_RGB2BGR)
    # image = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # resized_image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    t4 = time.time()
    # print('3====={}======{}'.format(os.getpid(), round(round(t4 - t3, 4) * 1000, 2)))
    boxed_image = Image.new('RGB', size, (128,128,128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    t5 = time.time()
    # print('4==={}'.format(round(t5 - t4, 4) * 1000))
    return boxed_image
