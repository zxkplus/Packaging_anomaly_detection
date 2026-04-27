"""
交互式圆形检测参数调试工具
通过滑动条实时调整 preprocessor 和 circle_detector 的参数，并可视化检测效果
支持参数保存为 JSON 文件和从 JSON 文件加载
"""
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Any

# 导入检测器模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from label_detector.preprocessor import ImagePreprocessor
from label_detector.circle_detector import CircleDetector


class InteractiveDebugger:
    """交互式参数调试器"""

    def __init__(self, image_path: str, config_path: str = None):
        """
        初始化调试器

        Args:
            image_path: 测试图像路径
            config_path: 初始配置文件路径（可选）
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        # 加载初始配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
            self.preprocessor_config = full_config.get('preprocessor', {})
            self.circle_detector_config = full_config.get('circle_detector', {})
        else:
            # 使用默认配置
            self.preprocessor_config = {
                'gaussian_blur_kernel': 5,
                'median_blur_kernel': 3,
                'contrast_alpha': 1.2,
                'brightness_beta': 0,
                'use_clahe': True,
                'clahe_clip_limit': 2.0
            }
            self.circle_detector_config = {
                'hough_dp': 1.5,
                'hough_min_dist': 50,
                'hough_canny_param1': 100,
                'hough_accum_threshold': 80,
                'hough_min_radius': 200,
                'hough_max_radius': 500,
                'canny_low': 50,
                'canny_high': 150,
                'use_fixed_roi': False,
                'roi': None
            }

        # 创建窗口
        cv2.namedWindow('Interactive Debugger', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Interactive Debugger', 1600, 900)

        # 创建滑动条
        self._create_trackbars()

        # 当前显示模式: 0=原始+检测结果, 1=预处理后, 2=边缘检测
        self.display_mode = 0

    def _create_trackbars(self):
        """创建所有滑动条"""
        # === Preprocessor 参数 ===
        cv2.createTrackbar('Gaussian Kernel', 'Interactive Debugger', 
                          int(self.preprocessor_config['gaussian_blur_kernel']), 20, self._on_change)
        cv2.createTrackbar('Median Kernel', 'Interactive Debugger', 
                          int(self.preprocessor_config['median_blur_kernel']), 20, self._on_change)
        cv2.createTrackbar('Contrast Alpha', 'Interactive Debugger', 
                          int(self.preprocessor_config['contrast_alpha'] * 10), 30, self._on_change)
        cv2.createTrackbar('Brightness Beta', 'Interactive Debugger', 
                          int(self.preprocessor_config['brightness_beta']) + 100, 200, self._on_change)
        cv2.createTrackbar('CLAHE Clip Limit', 'Interactive Debugger', 
                          int(self.preprocessor_config['clahe_clip_limit'] * 10), 50, self._on_change)
        
        # CLAHE 开关 (用滑动条模拟: 0=关闭, 1=开启)
        clahe_value = 1 if self.preprocessor_config.get('use_clahe', True) else 0
        cv2.createTrackbar('Use CLAHE', 'Interactive Debugger', clahe_value, 1, self._on_change)

        # === Circle Detector 参数 ===
        cv2.createTrackbar('Hough DP', 'Interactive Debugger', 
                          int(self.circle_detector_config['hough_dp'] * 10), 30, self._on_change)
        cv2.createTrackbar('Min Dist', 'Interactive Debugger', 
                          int(self.circle_detector_config['hough_min_dist']), 200, self._on_change)
        cv2.createTrackbar('Canny Param1', 'Interactive Debugger', 
                          int(self.circle_detector_config['hough_canny_param1']), 300, self._on_change)
        cv2.createTrackbar('Accum Threshold', 'Interactive Debugger', 
                          int(self.circle_detector_config['hough_accum_threshold']), 200, self._on_change)
        cv2.createTrackbar('Min Radius', 'Interactive Debugger', 
                          int(self.circle_detector_config['hough_min_radius']), 500, self._on_change)
        cv2.createTrackbar('Max Radius', 'Interactive Debugger', 
                          int(self.circle_detector_config['hough_max_radius']), 1000, self._on_change)
        cv2.createTrackbar('Canny Low', 'Interactive Debugger', 
                          int(self.circle_detector_config['canny_low']), 255, self._on_change)
        cv2.createTrackbar('Canny High', 'Interactive Debugger', 
                          int(self.circle_detector_config['canny_high']), 255, self._on_change)

    def _on_change(self, pos):
        """滑动条变化回调（空函数，仅用于触发更新）"""
        pass

    def _get_current_config(self) -> Dict[str, Any]:
        """获取当前滑动条的配置值"""
        # Preprocessor 配置
        gaussian_kernel = cv2.getTrackbarPos('Gaussian Kernel', 'Interactive Debugger')
        # 确保高斯核为奇数
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        
        median_kernel = cv2.getTrackbarPos('Median Kernel', 'Interactive Debugger')
        if median_kernel % 2 == 0:
            median_kernel += 1

        preprocessor_config = {
            'gaussian_blur_kernel': gaussian_kernel,
            'median_blur_kernel': median_kernel,
            'contrast_alpha': cv2.getTrackbarPos('Contrast Alpha', 'Interactive Debugger') / 10.0,
            'brightness_beta': cv2.getTrackbarPos('Brightness Beta', 'Interactive Debugger') - 100,
            'clahe_clip_limit': cv2.getTrackbarPos('CLAHE Clip Limit', 'Interactive Debugger') / 10.0,
            'use_clahe': bool(cv2.getTrackbarPos('Use CLAHE', 'Interactive Debugger'))
        }

        # Circle Detector 配置
        circle_detector_config = {
            'hough_dp': cv2.getTrackbarPos('Hough DP', 'Interactive Debugger') / 10.0,
            'hough_min_dist': cv2.getTrackbarPos('Min Dist', 'Interactive Debugger'),
            'hough_canny_param1': cv2.getTrackbarPos('Canny Param1', 'Interactive Debugger'),
            'hough_accum_threshold': cv2.getTrackbarPos('Accum Threshold', 'Interactive Debugger'),
            'hough_min_radius': cv2.getTrackbarPos('Min Radius', 'Interactive Debugger'),
            'hough_max_radius': cv2.getTrackbarPos('Max Radius', 'Interactive Debugger'),
            'canny_low': cv2.getTrackbarPos('Canny Low', 'Interactive Debugger'),
            'canny_high': cv2.getTrackbarPos('Canny High', 'Interactive Debugger'),
            'use_fixed_roi': False,
            'roi': None
        }

        return preprocessor_config, circle_detector_config

    def _process_and_detect(self) -> tuple:
        """执行预处理和圆形检测"""
        preprocessor_config, circle_detector_config = self._get_current_config()

        # 创建预处理器和检测器
        preprocessor = ImagePreprocessor(preprocessor_config)
        circle_detector = CircleDetector(circle_detector_config)

        # 预处理
        preprocessed = preprocessor.preprocess(self.image)

        # 检测圆形
        circles = circle_detector.detect_circles(self.image, preprocessed)
        
        # 选择最佳圆形
        best_circle = circle_detector.select_best_circle(circles, self.image.shape[:2])

        # 可视化
        vis_image = circle_detector.visualize_circles(self.image, circles, best_circle)

        return preprocessed, circles, best_circle, vis_image

    def run(self):
        """运行交互式调试器"""
        print("=" * 60)
        print("交互式圆形检测参数调试工具")
        print("=" * 60)
        print("操作说明:")
        print("  - 使用滑动条调整参数")
        print("  - 按 'm' 键切换显示模式 (原始/预处理/边缘)")
        print("  - 按 's' 键保存当前参数到 JSON 文件")
        print("  - 按 'l' 键从 JSON 文件加载参数")
        print("  - 按 'q' 键退出")
        print("=" * 60)

        while True:
            # 执行检测和可视化
            preprocessed, circles, best_circle, vis_image = self._process_and_detect()

            # 根据显示模式选择显示的图像
            if self.display_mode == 0:
                display_img = vis_image
                mode_text = "Mode: Original + Detection"
            elif self.display_mode == 1:
                display_img = preprocessed
                mode_text = "Mode: Preprocessed"
            else:  # mode 2
                # 显示边缘检测结果
                _, circle_detector_config = self._get_current_config()
                gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY) if len(preprocessed.shape) == 3 else preprocessed
                edges = cv2.Canny(gray, circle_detector_config['canny_low'], circle_detector_config['canny_high'])
                display_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                mode_text = "Mode: Edge Detection"

            # 添加信息文本
            info_text = [
                mode_text,
                f"Circles detected: {len(circles)}",
                f"Best circle: {best_circle if best_circle else 'None'}",
                "",
                "Keys: m=mode, s=save, l=load, q=quit"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(display_img, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

            # 显示图像
            cv2.imshow('Interactive Debugger', display_img)

            # 按键处理
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.display_mode = (self.display_mode + 1) % 3
            elif key == ord('s'):
                self._save_config()
            elif key == ord('l'):
                self._load_config()

        cv2.destroyAllWindows()

    def _save_config(self):
        """保存当前配置到 JSON 文件"""
        preprocessor_config, circle_detector_config = self._get_current_config()
        
        full_config = {
            'preprocessor': preprocessor_config,
            'circle_detector': circle_detector_config,
            'polar_transform': {
                'polar_output_height': 200,
                'polar_output_width': 360,
                'normalize_circle_size': 300,
                'interpolation': 1
            },
            'defect_detector': {
                'edge_density_threshold': 1.5,
                'brightness_var_threshold': 30.0,
                'detection_radius_ratio': 0.8,
                'angle_tolerance': 15.0,
                'reference_angle': 0.0,
                'use_auto_reference': True,
                'template_match_threshold': 0.7,
                'template_path': None
            },
            'visualization': True,
            'save_intermediate': True,
            'output_dir': 'assets/output'
        }

        save_path = 'config/detector_config_debug.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 配置已保存到: {save_path}")

    def _load_config(self):
        """从 JSON 文件加载配置"""
        load_path = input("\n请输入配置文件路径 (默认: config/detector_config_debug.json): ").strip()
        if not load_path:
            load_path = 'config/detector_config_debug.json'
        
        if not os.path.exists(load_path):
            print(f"✗ 文件不存在: {load_path}")
            return

        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)

            # 更新配置
            if 'preprocessor' in full_config:
                self.preprocessor_config = full_config['preprocessor']
            if 'circle_detector' in full_config:
                self.circle_detector_config = full_config['circle_detector']

            # 更新滑动条位置
            self._update_trackbars()
            
            print(f"✓ 配置已从 {load_path} 加载")
        except Exception as e:
            print(f"✗ 加载配置失败: {e}")

    def _update_trackbars(self):
        """根据当前配置更新滑动条位置"""
        # Preprocessor
        cv2.setTrackbarPos('Gaussian Kernel', 'Interactive Debugger', 
                          int(self.preprocessor_config.get('gaussian_blur_kernel', 5)))
        cv2.setTrackbarPos('Median Kernel', 'Interactive Debugger', 
                          int(self.preprocessor_config.get('median_blur_kernel', 3)))
        cv2.setTrackbarPos('Contrast Alpha', 'Interactive Debugger', 
                          int(self.preprocessor_config.get('contrast_alpha', 1.2) * 10))
        cv2.setTrackbarPos('Brightness Beta', 'Interactive Debugger', 
                          int(self.preprocessor_config.get('brightness_beta', 0)) + 100)
        cv2.setTrackbarPos('CLAHE Clip Limit', 'Interactive Debugger', 
                          int(self.preprocessor_config.get('clahe_clip_limit', 2.0) * 10))
        cv2.setTrackbarPos('Use CLAHE', 'Interactive Debugger', 
                          1 if self.preprocessor_config.get('use_clahe', True) else 0)

        # Circle Detector
        cv2.setTrackbarPos('Hough DP', 'Interactive Debugger', 
                          int(self.circle_detector_config.get('hough_dp', 1.5) * 10))
        cv2.setTrackbarPos('Min Dist', 'Interactive Debugger', 
                          int(self.circle_detector_config.get('hough_min_dist', 50)))
        cv2.setTrackbarPos('Canny Param1', 'Interactive Debugger', 
                          int(self.circle_detector_config.get('hough_canny_param1', 100)))
        cv2.setTrackbarPos('Accum Threshold', 'Interactive Debugger', 
                          int(self.circle_detector_config.get('hough_accum_threshold', 80)))
        cv2.setTrackbarPos('Min Radius', 'Interactive Debugger', 
                          int(self.circle_detector_config.get('hough_min_radius', 200)))
        cv2.setTrackbarPos('Max Radius', 'Interactive Debugger', 
                          int(self.circle_detector_config.get('hough_max_radius', 500)))
        cv2.setTrackbarPos('Canny Low', 'Interactive Debugger', 
                          int(self.circle_detector_config.get('canny_low', 50)))
        cv2.setTrackbarPos('Canny High', 'Interactive Debugger', 
                          int(self.circle_detector_config.get('canny_high', 150)))


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='交互式圆形检测参数调试工具')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--config', type=str, default='config/detector_config.json', 
                       help='初始配置文件路径 (可选)')
    
    args = parser.parse_args()
    
    debugger = InteractiveDebugger(args.image, args.config)
    debugger.run()


if __name__ == '__main__':
    main()
