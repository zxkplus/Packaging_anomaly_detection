"""
主检测器模块
整合所有模块，完成完整的标签检测流程
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
import glob

from .preprocessor import ImagePreprocessor
from .circle_detector import CircleDetector
from .polar_transform import PolarTransform
from .defect_detector import DefectDetector


class LabelDetector:
    """酒瓶标签检测器主类"""

    def __init__(self, config_path: str = None):
        """
        初始化标签检测器

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path is None:
            config_path = "config/detector_config.json"

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 初始化各模块
        self.preprocessor = ImagePreprocessor(config.get('preprocessor', {}))
        self.circle_detector = CircleDetector(config.get('circle_detector', {}))
        self.polar_transform = PolarTransform(config.get('polar_transform', {}))
        self.defect_detector = DefectDetector(config.get('defect_detector', {}))

        # 配置
        self.config = config
        self.visualization = config.get('visualization', True)
        self.save_intermediate = config.get('save_intermediate', False)
        self.output_dir = config.get('output_dir', 'assets/output')

        # 确保输出目录存在
        if self.save_intermediate:
            os.makedirs(self.output_dir, exist_ok=True)

        # 检测结果缓存
        self.last_results = None

    def detect(self, image_path: str, train_mode: bool = False) -> Dict:
        """
        检测单张图像

        Args:
            image_path: 图像路径
            train_mode: 是否为训练模式（用于学习标准样本）

        Returns:
            检测结果字典
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        return self.detect_image(image, train_mode=train_mode, image_name=os.path.basename(image_path))

    def detect_image(self, image: np.ndarray, train_mode: bool = False, image_name: str = None) -> Dict:
        """
        检测图像

        Args:
            image: 输入图像（numpy数组）
            train_mode: 是否为训练模式
            image_name: 图像名称（用于保存中间结果）

        Returns:
            检测结果字典
        """
        result = {
            'status': 'success',
            'image_name': image_name,
            'steps': {},
            'final_result': None
        }

        try:
            # 步骤1: 图像预处理
            preprocessed = self.preprocessor.preprocess(image)
            result['steps']['preprocessing'] = 'success'

            if self.save_intermediate:
                cv2.imwrite(f"{self.output_dir}/{image_name}_1_preprocessed.jpg", preprocessed)

            # 步骤2: 圆形检测
            circles = self.circle_detector.detect_circles(image, preprocessed)
            best_circle = self.circle_detector.select_best_circle(circles, image.shape[:2])

            if best_circle is None:
                result['status'] = 'failed'
                result['error'] = '未检测到圆形标签'
                return result

            result['steps']['circle_detection'] = 'success'
            result['circle'] = best_circle

            if self.visualization:
                vis_circles = self.circle_detector.visualize_circles(image, circles, best_circle)
                if self.save_intermediate:
                    cv2.imwrite(f"{self.output_dir}/{image_name}_2_circles.jpg", vis_circles)

            # 步骤3: 极坐标变换和归一化
            normalized_circle = self.polar_transform.normalize_circle(preprocessed, best_circle)
            polar_image = self.polar_transform.polar_to_rectangular(preprocessed, best_circle)

            result['steps']['polar_transform'] = 'success'

            if self.save_intermediate:
                cv2.imwrite(f"{self.output_dir}/{image_name}_3_normalized.jpg", normalized_circle)
                cv2.imwrite(f"{self.output_dir}/{image_name}_4_polar.jpg", polar_image)

            # 步骤4: 训练或检测
            if train_mode:
                # 训练模式：学习标准样本
                self.defect_detector.train_reference(normalized_circle, polar_image)
                result['steps']['training'] = 'success'
                result['final_result'] = {
                    'status': 'trained',
                    'message': '标准样本学习完成'
                }
            else:
                # 检测模式：检测缺陷
                if not self.defect_detector.is_trained:
                    result['status'] = 'warning'
                    result['message'] = '未训练参考样本，使用统计方法检测'

                # 检测所有缺陷
                defect_results = self.defect_detector.detect_all(normalized_circle, polar_image)
                result['steps']['defect_detection'] = 'success'
                result['defect_results'] = defect_results

                # 获取摘要
                summary = self.defect_detector.get_summary(defect_results)
                result['final_result'] = summary

                # 可视化结果
                if self.visualization:
                    vis_defects = self.defect_detector.visualize_defects(
                        normalized_circle, polar_image, defect_results
                    )
                    if self.save_intermediate:
                        cv2.imwrite(f"{self.output_dir}/{image_name}_5_result.jpg", vis_defects)

            # 缓存结果
            self.last_results = result

            return result

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            return result

    def batch_detect(self, image_dir: str, train_image: str = None) -> List[Dict]:
        """
        批量检测图像

        Args:
            image_dir: 图像目录
            train_image: 训练图像文件名（用于学习标准样本）

        Returns:
            所有检测结果列表
        """
        results = []

        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        # 如果指定了训练图像，先进行训练
        if train_image and train_image in image_files:
            print(f"使用训练图像: {train_image}")
            train_result = self.detect(os.path.join(image_dir, train_image), train_mode=True)
            results.append(train_result)
            image_files.remove(train_image)

        # 批量检测其他图像
        for image_file in image_files:
            print(f"检测图像: {os.path.basename(image_file)}")
            result = self.detect(image_file, train_mode=False)
            results.append(result)

        return results

    def visualize_full_process(self, image_path: str, save_path: str = None) -> np.ndarray:
        """
        可视化完整处理流程

        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）

        Returns:
            可视化图像
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 预处理
        preprocessed = self.preprocessor.preprocess(image)

        # 圆形检测
        circles = self.circle_detector.detect_circles(image, preprocessed)
        best_circle = self.circle_detector.select_best_circle(circles, image.shape[:2])

        if best_circle is None:
            return image

        # 可视化圆形检测结果
        vis_circles = self.circle_detector.visualize_circles(image, circles, best_circle)

        # 极坐标变换
        normalized = self.polar_transform.normalize_circle(preprocessed, best_circle)
        polar = self.polar_transform.polar_to_rectangular(preprocessed, best_circle)

        # 调整大小以匹配原图高度
        h, w = image.shape[:2]
        normalized_resized = cv2.resize(normalized, (w, h))
        polar_resized = cv2.resize(polar, (w, h))

        # 合并所有图像
        vis = np.hstack([vis_circles, normalized_resized, polar_resized])

        # 添加文字标注
        cv2.putText(vis, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, "Normalized", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, "Polar", (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 保存
        if save_path:
            cv2.imwrite(save_path, vis)

        return vis

    def get_config(self) -> Dict:
        """
        获取当前配置

        Returns:
            配置字典
        """
        return self.config

    def update_config(self, new_config: Dict) -> None:
        """
        更新配置

        Args:
            new_config: 新配置字典
        """
        self.config.update(new_config)

        # 重新初始化相关模块
        if 'preprocessor' in new_config:
            self.preprocessor = ImagePreprocessor(new_config['preprocessor'])
        if 'circle_detector' in new_config:
            self.circle_detector = CircleDetector(new_config['circle_detector'])
        if 'polar_transform' in new_config:
            self.polar_transform = PolarTransform(new_config['polar_transform'])
        if 'defect_detector' in new_config:
            self.defect_detector = DefectDetector(new_config['defect_detector'])


# 全局变量（用于单例模式）
_detector_instance = None


def get_detector(config_path: str = None) -> LabelDetector:
    """
    获取检测器单例

    Args:
        config_path: 配置文件路径

    Returns:
        LabelDetector实例
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LabelDetector(config_path)
    return _detector_instance
