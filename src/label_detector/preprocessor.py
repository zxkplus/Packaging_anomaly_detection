"""
图像预处理模块
用于对输入图像进行去噪、增强、对比度调整等预处理操作
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """图像预处理器"""

    def __init__(self, config: dict):
        """
        初始化预处理器

        Args:
            config: 配置字典，包含预处理参数
        """
        self.gaussian_blur_kernel = config.get('gaussian_blur_kernel', 5)
        self.median_blur_kernel = config.get('median_blur_kernel', 3)
        self.contrast_alpha = config.get('contrast_alpha', 1.2)
        self.brightness_beta = config.get('brightness_beta', 0)
        self.use_clahe = config.get('use_clahe', True)
        self.clahe_clip_limit = config.get('clahe_clip_limit', 2.0)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪

        Args:
            image: 输入图像

        Returns:
            去噪后的图像
        """
        # 高斯模糊去噪
        if self.gaussian_blur_kernel > 0:
            image = cv2.GaussianBlur(image, (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 0)

        # 中值滤波去噪（去除椒盐噪声）
        if self.median_blur_kernel > 0:
            image = cv2.medianBlur(image, self.median_blur_kernel)

        return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        增强对比度和亮度

        Args:
            image: 输入图像

        Returns:
            增强后的图像
        """
        # 简单的线性对比度和亮度调整
        enhanced = cv2.convertScaleAbs(image, alpha=self.contrast_alpha, beta=self.brightness_beta)

        # 使用CLAHE自适应直方图均衡化
        if self.use_clahe:
            # 转换为Lab颜色空间
            if len(image.shape) == 3:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # 对L通道应用CLAHE
                clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
                l = clahe.apply(l)

                # 合并通道并转回BGR
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                # 灰度图直接应用CLAHE
                clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
                enhanced = clahe.apply(enhanced)

        return enhanced

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        图像锐化

        Args:
            image: 输入图像

        Returns:
            锐化后的图像
        """
        # 使用拉普拉斯算子进行锐化
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        完整的预处理流程

        Args:
            image: 输入图像

        Returns:
            预处理后的图像
        """
        # 去噪
        processed = self.denoise(image)

        # 增强对比度
        processed = self.enhance_contrast(processed)

        # 锐化
        processed = self.sharpen(processed)

        return processed

    def extract_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        提取ROI区域

        Args:
            image: 输入图像
            roi: ROI区域 (x, y, width, height)

        Returns:
            ROI图像
        """
        x, y, w, h = roi
        return image[y:y+h, x:x+w]
