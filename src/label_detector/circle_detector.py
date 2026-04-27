"""
圆形检测模块
使用霍夫圆变换检测图像中的圆形标签
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional


class CircleDetector:
    """圆形标签检测器"""

    def __init__(self, config: dict):
        """
        初始化圆形检测器

        Args:
            config: 配置字典，包含检测参数
        """
        # 霍夫圆变换参数
        self.dp = config.get('hough_dp', 1.5)  # 累加器分辨率与图像分辨率的比率
        self.min_dist = config.get('hough_min_dist', 50)  # 检测圆心之间的最小距离
        self.canny_param1 = config.get('hough_canny_param1', 100)  # Canny边缘检测高阈值
        self.accum_threshold = config.get('hough_accum_threshold', 80)  # 累加器阈值
        self.min_radius = config.get('hough_min_radius', 30)  # 最小半径
        self.max_radius = config.get('hough_max_radius', 200)  # 最大半径

        # 边缘检测参数
        self.canny_low = config.get('canny_low', 50)
        self.canny_high = config.get('canny_high', 150)

        # ROI参数
        self.use_fixed_roi = config.get('use_fixed_roi', True)
        self.roi = config.get('roi', None)  # (x, y, width, height)

    def detect_circles(self, image: np.ndarray, preprocessed: Optional[np.ndarray] = None) -> List[Tuple[int, int, int]]:
        """
        检测图像中的圆形

        Args:
            image: 原始图像
            preprocessed: 预处理后的图像（可选）

        Returns:
            检测到的圆形列表，每个圆形为 (center_x, center_y, radius)
        """
        # 如果没有提供预处理图像，使用原始图像
        if preprocessed is None:
            work_image = image.copy()
        else:
            work_image = preprocessed.copy()

        # 如果使用固定ROI，先提取ROI区域
        if self.use_fixed_roi and self.roi is not None:
            x, y, w, h = self.roi
            roi_image = work_image[y:y+h, x:x+w]
        else:
            roi_image = work_image
            x, y = 0, 0  # ROI偏移量

        # 转换为灰度图
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image

        # 应用Canny边缘检测
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # 使用霍夫圆变换检测圆形
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.canny_param1,
            param2=self.accum_threshold,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        detected_circles = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # 转换坐标（如果使用了ROI，需要加上ROI偏移量）
            for (cx, cy, r) in circles:
                detected_circles.append((cx + x, cy + y, r))

        return detected_circles

    def select_best_circle(self, circles: List[Tuple[int, int, int]], image_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
        """
        从多个检测到的圆形中选择最佳的一个

        Args:
            circles: 检测到的圆形列表
            image_shape: 图像尺寸 (height, width)

        Returns:
            最佳圆形 (center_x, center_y, radius)
        """
        if not circles:
            return None

        # 如果只有一个圆，直接返回
        if len(circles) == 1:
            return circles[0]

        # 多个圆的情况，选择最靠近图像中心且半径最大的圆
        img_center_x, img_center_y = image_shape[1] // 2, image_shape[0] // 2

        best_circle = None
        best_score = -1

        for (cx, cy, r) in circles:
            # 计算到图像中心的距离（距离越小越好）
            dist_to_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
            max_dist = np.sqrt(img_center_x**2 + img_center_y**2)
            center_score = 1 - (dist_to_center / max_dist)

            # 半径越大越好（归一化）
            radius_score = r / self.max_radius

            # 综合得分：中心位置权重0.6，半径权重0.4
            total_score = center_score * 0.6 + radius_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_circle = (cx, cy, r)

        return best_circle

    def visualize_circles(self, image: np.ndarray, circles: List[Tuple[int, int, int]],
                         best_circle: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 原始图像
            circles: 检测到的所有圆形
            best_circle: 最佳圆形（会以不同颜色标记）

        Returns:
            可视化图像
        """
        vis_image = image.copy()

        # 绘制所有检测到的圆形（绿色）
        for (cx, cy, r) in circles:
            cv2.circle(vis_image, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(vis_image, (cx, cy), 2, (0, 255, 0), 3)

        # 绘制最佳圆形（红色）
        if best_circle is not None:
            cx, cy, r = best_circle
            cv2.circle(vis_image, (cx, cy), r, (0, 0, 255), 3)
            cv2.circle(vis_image, (cx, cy), 2, (0, 0, 255), 5)

        return vis_image

    def extract_circle_region(self, image: np.ndarray, circle: Tuple[int, int, int],
                            padding: int = 10) -> np.ndarray:
        """
        提取圆形区域的图像（带边距）

        Args:
            image: 原始图像
            circle: 圆形参数 (center_x, center_y, radius)
            padding: 边距

        Returns:
            圆形区域的图像
        """
        cx, cy, r = circle

        # 计算带边距的矩形区域
        x1 = max(0, cx - r - padding)
        y1 = max(0, cy - r - padding)
        x2 = min(image.shape[1], cx + r + padding)
        y2 = min(image.shape[0], cy + r + padding)

        # 提取区域
        region = image[y1:y2, x1:x2]

        # 调整圆心坐标到新图像的相对坐标
        relative_circle = (cx - x1, cy - y1, r)

        return region, relative_circle
