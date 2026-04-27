"""
缺陷检测模块
检测标签的贴重、贴正等缺陷
"""
import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
import json


class DefectDetector:
    """缺陷检测器"""

    def __init__(self, config: dict):
        """
        初始化缺陷检测器

        Args:
            config: 配置字典，包含检测参数
        """
        # 贴重检测参数
        self.edge_density_threshold = config.get('edge_density_threshold', 1.5)  # 边缘密度阈值
        self.brightness_var_threshold = config.get('brightness_var_threshold', 30.0)  # 亮度方差阈值
        self.detection_radius_ratio = config.get('detection_radius_ratio', 0.8)  # 检测半径比例（相对于标签半径）

        # 贴正检测参数
        self.angle_tolerance = config.get('angle_tolerance', 15.0)  # 角度容差（度）
        self.reference_angle = config.get('reference_angle', 0.0)  # 参考角度（度）
        self.use_auto_reference = config.get('use_auto_reference', True)  # 是否自动学习参考角度

        # 模板匹配参数
        self.template_match_threshold = config.get('template_match_threshold', 0.7)  # 模板匹配阈值
        self.template_path = config.get('template_path', None)  # 标准模板路径

        # 自学习参考模板
        self.reference_template = None
        self.reference_polar = None
        self.is_trained = False

    def train_reference(self, image: np.ndarray, polar_image: np.ndarray) -> None:
        """
        使用标准样本训练参考模板

        Args:
            image: 标准样本的归一化圆形图像
            polar_image: 标准样本的极坐标展开图像
        """
        # 保存参考图像
        self.reference_template = image
        self.reference_polar = polar_image

        # 如果需要自动学习参考角度
        if self.use_auto_reference:
            # 从极坐标图中检测特征角度（如文字、图案的起始位置）
            self.reference_angle = self._detect_feature_angle(polar_image)

        self.is_trained = True
        print(f"参考训练完成，参考角度: {self.reference_angle:.2f}°")

    def _detect_feature_angle(self, polar_image: np.ndarray) -> float:
        """
        从极坐标图中检测特征角度

        Args:
            polar_image: 极坐标展开图像

        Returns:
            检测到的角度（度）
        """
        # 转换为灰度图
        if len(polar_image.shape) == 3:
            gray = cv2.cvtColor(polar_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = polar_image

        # 计算每列的亮度变化（用于检测垂直特征，如文字边缘）
        col_intensity = np.mean(gray, axis=0)

        # 计算亮度梯度
        gradient = np.abs(np.gradient(col_intensity))

        # 找到梯度最大的位置（通常对应特征的边缘）
        peak_idx = np.argmax(gradient)

        # 转换为角度
        angle = (peak_idx / len(gradient)) * 360.0

        return angle

    def detect_duplicate(self, image: np.ndarray, polar_image: Optional[np.ndarray] = None) -> Dict:
        """
        检测贴重缺陷（重叠标签）

        Args:
            image: 归一化的圆形图像
            polar_image: 极坐标展开图像（可选）

        Returns:
            检测结果字典，包含 is_defect, confidence, details
        """
        result = {
            'is_defect': False,
            'confidence': 0.0,
            'defect_type': 'duplicate',
            'details': {}
        }

        if not self.is_trained:
            # 如果没有参考模板，使用统计方法检测
            return self._detect_duplicate_statistical(image, result)
        else:
            # 使用模板匹配方法检测
            return self._detect_duplicate_template_matching(image, result)

    def _detect_duplicate_statistical(self, image: np.ndarray, result: Dict) -> Dict:
        """
        使用统计方法检测贴重（边缘密度分析）

        Args:
            image: 输入图像
            result: 结果字典

        Returns:
            检测结果
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 计算边缘
        edges = cv2.Canny(gray, 50, 150)

        # 计算边缘密度（边缘像素占总像素的比例）
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # 正常标签的边缘密度通常在0.1-0.3之间，贴重时会更高
        result['details']['edge_density'] = edge_density

        if edge_density > self.edge_density_threshold:
            result['is_defect'] = True
            result['confidence'] = min(1.0, (edge_density - self.edge_density_threshold) * 2)
            result['details']['reason'] = f"边缘密度过高 ({edge_density:.3f} > {self.edge_density_threshold})"

        return result

    def _detect_duplicate_template_matching(self, image: np.ndarray, result: Dict) -> Dict:
        """
        使用模板匹配方法检测贴重

        Args:
            image: 输入图像
            result: 结果字典

        Returns:
            检测结果
        """
        if self.reference_template is None:
            return self._detect_duplicate_statistical(image, result)

        # 转换为灰度图
        if len(image.shape) == 3:
            test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(self.reference_template, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = image
            ref_gray = self.reference_template

        # 计算图像差异
        diff = cv2.absdiff(test_gray, ref_gray)

        # 统计差异
        diff_mean = np.mean(diff)
        diff_std = np.std(diff)

        result['details']['diff_mean'] = diff_mean
        result['details']['diff_std'] = diff_std

        # 如果差异过大，可能有缺陷
        if diff_mean > self.brightness_var_threshold:
            result['is_defect'] = True
            result['confidence'] = min(1.0, (diff_mean - self.brightness_var_threshold) / 50.0)
            result['details']['reason'] = f"亮度差异过大 ({diff_mean:.2f} > {self.brightness_var_threshold})"

        # 检测边缘密度作为补充
        edges = cv2.Canny(test_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        result['details']['edge_density'] = edge_density

        if edge_density > self.edge_density_threshold:
            result['is_defect'] = True
            result['confidence'] = max(result['confidence'], min(1.0, (edge_density - self.edge_density_threshold) * 2))

        return result

    def detect_rotation(self, polar_image: np.ndarray) -> Dict:
        """
        检测贴正缺陷（旋转偏差）

        Args:
            polar_image: 极坐标展开图像

        Returns:
            检测结果字典，包含 is_defect, confidence, angle_offset, details
        """
        result = {
            'is_defect': False,
            'confidence': 0.0,
            'defect_type': 'rotation',
            'angle_offset': 0.0,
            'details': {}
        }

        # 检测当前图像的特征角度
        current_angle = self._detect_feature_angle(polar_image)
        angle_offset = abs(current_angle - self.reference_angle)

        # 处理360度循环
        if angle_offset > 180:
            angle_offset = 360 - angle_offset

        result['angle_offset'] = angle_offset
        result['details']['current_angle'] = current_angle
        result['details']['reference_angle'] = self.reference_angle

        # 判断是否超出容差
        if angle_offset > self.angle_tolerance:
            result['is_defect'] = True
            result['confidence'] = min(1.0, (angle_offset - self.angle_tolerance) / (45 - self.angle_tolerance))
            result['details']['reason'] = f"角度偏移过大 ({angle_offset:.2f}° > {self.angle_tolerance}°)"

        return result

    def detect_all(self, image: np.ndarray, polar_image: np.ndarray) -> List[Dict]:
        """
        检测所有缺陷类型

        Args:
            image: 归一化的圆形图像
            polar_image: 极坐标展开图像

        Returns:
            所有检测结果列表
        """
        results = []

        # 检测贴重
        duplicate_result = self.detect_duplicate(image, polar_image)
        results.append(duplicate_result)

        # 检测贴正
        rotation_result = self.detect_rotation(polar_image)
        results.append(rotation_result)

        return results

    def visualize_defects(self, image: np.ndarray, polar_image: np.ndarray,
                         results: List[Dict]) -> np.ndarray:
        """
        可视化缺陷检测结果

        Args:
            image: 归一化的圆形图像
            polar_image: 极坐标展开图像
            results: 检测结果列表

        Returns:
            可视化图像
        """
        # 调整大小
        h1, w1 = image.shape[:2]
        h2, w2 = polar_image.shape[:2]

        # 调整polar图像宽度以匹配image
        polar_resized = cv2.resize(polar_image, (w1, int(h2 * w1 / w2)))

        # 创建可视化图像
        vis = np.vstack([image, polar_resized])

        # 添加检测结果文本
        y_offset = 30
        for result in results:
            if result['is_defect']:
                color = (0, 0, 255)  # 红色表示缺陷
                status = "NG"
            else:
                color = (0, 255, 0)  # 绿色表示正常
                status = "OK"

            defect_type = result['defect_type']
            confidence = result['confidence']

            text = f"{defect_type.upper()}: {status} (confidence: {confidence:.2f})"
            cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

            # 显示详细信息
            if result['is_defect'] and 'reason' in result['details']:
                reason = result['details']['reason']
                cv2.putText(vis, f"  Reason: {reason}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_offset += 25

        return vis

    def get_summary(self, results: List[Dict]) -> Dict:
        """
        获取检测结果摘要

        Args:
            results: 检测结果列表

        Returns:
            摘要字典
        """
        summary = {
            'overall_status': 'OK',
            'total_defects': 0,
            'defects': []
        }

        for result in results:
            if result['is_defect']:
                summary['total_defects'] += 1
                summary['overall_status'] = 'NG'
                summary['defects'].append({
                    'type': result['defect_type'],
                    'confidence': result['confidence'],
                    'details': result['details']
                })

        return summary
