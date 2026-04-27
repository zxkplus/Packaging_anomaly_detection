"""
酒瓶标签检测算法包
"""

from .preprocessor import ImagePreprocessor
from .circle_detector import CircleDetector
from .polar_transform import PolarTransform
from .defect_detector import DefectDetector
from .detector import LabelDetector, get_detector

__version__ = '1.0.0'
__all__ = [
    'ImagePreprocessor',
    'CircleDetector',
    'PolarTransform',
    'DefectDetector',
    'LabelDetector',
    'get_detector'
]
