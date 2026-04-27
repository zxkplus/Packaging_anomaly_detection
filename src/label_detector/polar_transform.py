"""
极坐标变换和图像归一化模块
将圆形区域展开为矩形或归一化为标准圆
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class PolarTransform:
    """极坐标变换器"""

    def __init__(self, config: dict):
        """
        初始化极坐标变换器

        Args:
            config: 配置字典，包含变换参数
        """
        self.output_height = config.get('polar_output_height', 200)  # 输出高度（对应半径）
        self.output_width = config.get('polar_output_width', 360)    # 输出宽度（对应角度）
        self.normalize_circle_size = config.get('normalize_circle_size', 300)  # 归一化后的圆形大小
        self.interpolation = config.get('interpolation', cv2.INTER_LINEAR)  # 插值方法

    def polar_to_rectangular(self, image: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
        """
        将圆形区域展开为矩形图像（极坐标变换）

        Args:
            image: 输入图像（包含圆形）
            circle: 圆形参数 (center_x, center_y, radius)

        Returns:
            展开后的矩形图像
        """
        cx, cy, r = circle
        # 转换为Python int类型
        cx = int(cx)
        cy = int(cy)
        r = int(r)

        # 计算极坐标变换
        # OpenCV的linearPolar可以将极坐标转换为笛卡尔坐标
        # 参数：输入图像，输出图像，圆心，最大半径，插值方法
        polar_image = cv2.linearPolar(
            image,
            (cx, cy),
            r,
            self.interpolation
        )

        # 调整大小到统一尺寸
        polar_image = cv2.resize(
            polar_image,
            (self.output_width, self.output_height),
            interpolation=self.interpolation
        )

        return polar_image

    def polar_to_rectangular_expanded(self, image: np.ndarray, circle: Tuple[int, int, int],
                                     min_radius: float = 0.0) -> np.ndarray:
        """
        将环形区域展开为矩形图像（从min_radius到radius）

        Args:
            image: 输入图像
            circle: 圆形参数 (center_x, center_y, radius)
            min_radius: 内圆半径（用于环形区域）

        Returns:
            展开后的矩形图像
        """
        cx, cy, radius = circle
        # 转换为Python int类型
        cx = int(cx)
        cy = int(cy)
        radius = int(radius)
        min_radius = int(min_radius)

        # 创建极坐标映射
        h, w = image.shape[:2]
        polar_h, polar_w = self.output_height, self.output_width

        # 生成极坐标网格
        theta = np.linspace(0, 2 * np.pi, polar_w, endpoint=False)
        rho = np.linspace(min_radius, radius, polar_h)

        theta_grid, rho_grid = np.meshgrid(theta, rho)

        # 转换为笛卡尔坐标
        x = cx + rho_grid * np.cos(theta_grid)
        y = cy + rho_grid * np.sin(theta_grid)

        # 确保坐标在图像范围内
        x = np.clip(x, 0, w - 1).astype(np.float32)
        y = np.clip(y, 0, h - 1).astype(np.float32)

        # 使用remap进行极坐标变换
        if len(image.shape) == 3:
            polar_image = np.zeros((polar_h, polar_w, 3), dtype=np.uint8)
            for i in range(3):
                polar_image[:, :, i] = cv2.remap(
                    image[:, :, i],
                    x, y,
                    self.interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
        else:
            polar_image = cv2.remap(
                image,
                x, y,
                self.interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        return polar_image

    def normalize_circle(self, image: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
        """
        将圆形区域归一化为固定大小的正圆

        Args:
            image: 输入图像
            circle: 圆形参数 (center_x, center_y, radius)

        Returns:
            归一化后的圆形图像（正方形，中心为圆形）
        """
        cx, cy, r = circle

        # 提取圆形区域的矩形边界
        x1 = max(0, int(cx - r))
        y1 = max(0, int(cy - r))
        x2 = min(image.shape[1], int(cx + r))
        y2 = min(image.shape[0], int(cy + r))

        # 提取区域
        region = image[y1:y2, x1:x2]

        # 创建圆形mask
        region_h, region_w = region.shape[:2]
        center = (region_w // 2, region_h // 2)
        radius = min(region_w, region_h) // 2

        # 创建标准大小的圆形图像
        size = self.normalize_circle_size
        normalized = cv2.resize(region, (size, size), interpolation=self.interpolation)

        # 创建圆形mask
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (size // 2, size // 2), size // 2, 255, -1)

        # 应用mask，将圆外区域设为黑色
        if len(normalized.shape) == 3:
            mask = cv2.merge([mask, mask, mask])
        normalized = cv2.bitwise_and(normalized, mask)

        return normalized

    def create_circle_mask(self, image: np.ndarray, circle: Tuple[int, int, int],
                          fill: bool = False) -> np.ndarray:
        """
        创建圆形区域的mask

        Args:
            image: 参考图像（用于获取尺寸）
            circle: 圆形参数 (center_x, center_y, radius)
            fill: 是否填充整个圆

        Returns:
            二值mask图像
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        cx, cy, r = circle
        thickness = -1 if fill else 2
        cv2.circle(mask, (cx, cy), r, 255, thickness)

        return mask

    def unwrap_circle_360(self, image: np.ndarray, circle: Tuple[int, int, int],
                         min_radius_ratio: float = 0.0) -> np.ndarray:
        """
        将圆形区域展开为360度全景图（从中心到边缘）

        Args:
            image: 输入图像
            circle: 圆形参数 (center_x, center_y, radius)
            min_radius_ratio: 内圆半径比例（0表示从中心开始）

        Returns:
            展开后的全景图
        """
        cx, cy, radius = circle
        min_radius = int(radius * min_radius_ratio)

        return self.polar_to_rectangular_expanded(image, circle, min_radius)

    def visualize_polar_transform(self, image: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
        """
        可视化极坐标变换结果

        Args:
            image: 原始图像
            circle: 圆形参数

        Returns:
            可视化图像（并排显示原图和展开图）
        """
        # 展开图像
        polar = self.polar_to_rectangular(image, circle)

        # 调整大小以便可视化
        h, w = image.shape[:2]
        if polar.shape[1] > w:
            polar = cv2.resize(polar, (w, polar.shape[0] * w // polar.shape[1]))

        # 合并图像
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if len(polar.shape) == 2:
            polar = cv2.cvtColor(polar, cv2.COLOR_GRAY2BGR)

        vis = np.vstack([image, polar])

        # 添加文字标注
        cv2.putText(vis, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, "Polar Transformed", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return vis
