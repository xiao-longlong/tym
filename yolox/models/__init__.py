#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from .build import *
from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
# tym10min 需要在添加很多个yolo_head.py文件并明确其名字后，回来修改这里
from .yolo_head import YOLOXHead
from .yolo_headori import YOLOXHeadori
from .yolo_head100 import YOLOXHead100
from .yolo_head2100 import YOLOXHead2100
from .yolo_head2101 import YOLOXHead2101
from .yolo_head2200 import YOLOXHead2200
from .yolo_head2201 import YOLOXHead2201
from .yolo_head1002100 import YOLOXHead1002100
from .yolo_head1002200 import YOLOXHead1002200
from .yolo_head1002300 import YOLOXHead1002300
from .yolo_head22002300 import YOLOXHead22002300
from .yolo_head210022002300 import YOLOXHead210022002300
from .yolo_head100210022002300 import YOLOXHead100210022002300



from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
