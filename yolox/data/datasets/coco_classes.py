#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
###########-----------------原本COCO数据集-----------------###########
COCO_CLASSES_COCOori = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

"""
#################-----------------GTSRBCOCO数据集-----------------#################
# COCO_CLASSES = (
#     "speed limit 20 (prohibitory) 限速20(限制)",
#     "speed limit 30 (prohibitory) 限速30(限制)",
#     "speed limit 50 (prohibitory) 限速50(限制)",
#     "speed limit 60 (prohibitory) 限速60(限制)",
#     "speed limit 70 (prohibitory) 限速70(限制)",
#     "speed limit 80 (prohibitory) 限速80(限制)",
#     "restriction ends 80 (other) 限速80解除(其他)",
#     "speed limit 100 (prohibitory) 限速100(限制)",
#     "speed limit 120 (prohibitory) 限速120(限制)",
#     "no overtaking (prohibitory) 禁止超车(限制)",
#     "no overtaking (trucks) (prohibitory) 禁止卡车超车(限制)",
#     "priority at next intersection (danger) 下一个路口优先(危险)",
#     "priority road (other) 优先道路(其他)",
#     "give way (other) 让行(其他)",
#     "stop (other) 停车(其他)",
#     "no traffic both ways (prohibitory) 双向禁止通行(限制)",
#     "no trucks (prohibitory) 禁止卡车通行(限制)",
#     "no entry (other) 禁止进入(其他)",
#     "danger (danger) 危险(危险)",
#     "bend left (danger) 向左急弯(危险)",
#     "bend right (danger) 向右急弯(危险)",
#     "bend (danger) 急弯(危险)",
#     "uneven road (danger) 颠簸路面(危险)",
#     "slippery road (danger) 湿滑路面(危险)",
#     "road narrows (danger) 道路变窄(危险)",
#     "construction (danger) 施工(危险)",
#     "traffic signal (danger) 交通信号(危险)",
#     "pedestrian crossing (danger) 行人横道(危险)",
#     "school crossing (danger) 学校横道(危险)",
#     "cycles crossing (danger) 自行车横道(危险)",
#     "snow (danger) 雪天(危险)",
#     "animals (danger) 动物穿行(危险)",
#     "restriction ends (other) 限制解除(其他)",
#     "go right (mandatory) 向右行驶(强制)",
#     "go left (mandatory) 向左行驶(强制)",
#     "go straight (mandatory) 直行(强制)",
#     "go right or straight (mandatory) 右转或直行(强制)",
#     "go left or straight (mandatory) 左转或直行(强制)",
#     "keep right (mandatory) 靠右行驶(强制)",
#     "keep left (mandatory) 靠左行驶(强制)",
#     "roundabout (mandatory) 环形路(强制)",
#     "restriction ends (overtaking) (other) 超车限制解除(其他)",
#     "restriction ends (overtaking (trucks)) (other) 卡车超车限制解除(其他)"
# )
#
#  """

COCO_CLASSES_GTSRB = (
    "speed limit 20 (prohibitory)",
    "speed limit 30 (prohibitory)",
    "speed limit 50 (prohibitory)",
    "speed limit 60 (prohibitory)",
    "speed limit 70 (prohibitory)",
    "speed limit 80 (prohibitory)",
    "restriction ends 80 (other)",
    "speed limit 100 (prohibitory)",
    "speed limit 120 (prohibitory)",
    "no overtaking (prohibitory)",
    "no overtaking (trucks) (prohibitory)",
    "priority at next intersection (danger)",
    "priority road (other)",
    "give way (other)",
    "stop (other)",
    "no traffic both ways (prohibitory)",
    "no trucks (prohibitory)",
    "no entry (other)",
    "danger (danger)",
    "bend left (danger)",
    "bend right (danger)",
    "bend (danger)",
    "uneven road (danger)",
    "slippery road (danger)",
    "road narrows (danger)",
    "construction (danger)",
    "traffic signal (danger)",
    "pedestrian crossing (danger)",
    "school crossing (danger)",
    "cycles crossing (danger)",
    "snow (danger)",
    "animals (danger)",
    "restriction ends (other)",
    "go right (mandatory)",
    "go left (mandatory)",
    "go straight (mandatory)",
    "go right or straight (mandatory)",
    "go left or straight (mandatory)",
    "keep right (mandatory)",
    "keep left (mandatory)",
    "roundabout (mandatory)",
    "restriction ends (overtaking) (other)",
    "restriction ends (overtaking (trucks)) (other)"
)




COCO_CLASSES_Visdrone = (
# COCO_CLASSES = (
    'pedestrian',
    'people',
    'bicycle',
    'car',
    'van',
    'truck',
    'tricycle',
    'awning-tricycle',
    'bus',
    'motor'
)

##----------------SkyFusion数据集----------------##
# COCO_CLASSES_SkyFusion = (
COCO_CLASSES = (
        "Aircraft",
        "ship",
        "vehicle")


