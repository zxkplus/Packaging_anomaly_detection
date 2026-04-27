"""
交互式圆形检测参数调试工具（优化版）
通过滑动条和键盘输入实时调整 preprocessor 和 circle_detector 的参数
支持参数保存为 JSON 文件和从 JSON 文件加载
优化了性能和用户体验
"""
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# 导入检测器模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from label_detector.preprocessor import ImagePreprocessor
from label_detector.circle_detector import CircleDetector


class ParameterControl:
    """参数控制器 - 管理单个参数的值和输入"""
    
    def __init__(self, name: str, short_name: str, value: float, min_val: float, 
                 max_val: float, step: float = 1.0, is_int: bool = False):
        self.name = name
        self.short_name = short_name
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.is_int = is_int
        self.input_buffer = ""
        self.is_inputting = False
    
    def clamp(self, val: float) -> float:
        """限制值在范围内"""
        return max(self.min_val, min(self.max_val, val))
    
    def set_value(self, val: float):
        """设置值"""
        self.value = self.clamp(val)
        if self.is_int:
            self.value = int(round(self.value))
    
    def add_input_char(self, char: str):
        """添加输入字符"""
        if char.isdigit() or char == '.':
            self.input_buffer += char
    
    def confirm_input(self):
        """确认输入"""
        try:
            if self.input_buffer:
                val = float(self.input_buffer)
                self.set_value(val)
        except ValueError:
            pass
        finally:
            self.input_buffer = ""
            self.is_inputting = False
    
    def cancel_input(self):
        """取消输入"""
        self.input_buffer = ""
        self.is_inputting = False
    
    def get_display_text(self) -> str:
        """获取显示文本"""
        if self.is_inputting:
            return f"{self.short_name}: {self.input_buffer}_"
        if self.is_int:
            return f"{self.short_name}: {int(self.value)}"
        return f"{self.short_name}: {self.value:.1f}"


class InteractiveDebugger:
    """交互式参数调试器（优化版）"""

    def __init__(self, image_path: str, config_path: str = None):
        """
        初始化调试器

        Args:
            image_path: 测试图像路径
            config_path: 初始配置文件路径（可选）
        """
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        
        if self.original_image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        # 保存原始图像尺寸
        self.orig_height, self.orig_width = self.original_image.shape[:2]
        
        # 加载初始配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
            preprocessor_config = full_config.get('preprocessor', {})
            circle_detector_config = full_config.get('circle_detector', {})
        else:
            # 使用默认配置
            preprocessor_config = {
                'gaussian_blur_kernel': 5,
                'median_blur_kernel': 3,
                'contrast_alpha': 1.2,
                'brightness_beta': 0,
                'use_clahe': True,
                'clahe_clip_limit': 2.0
            }
            circle_detector_config = {
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

        # 初始化参数控制器
        self._init_parameters(preprocessor_config, circle_detector_config)

        # 当前显示模式: 0=原始+检测结果, 1=预处理后, 2=边缘检测
        self.display_mode = 0
        
        # 性能优化：缓存上一次的结果
        self.last_preprocessed = None
        self.last_circles = []
        self.last_best_circle = None
        self.last_vis_image = None
        self.last_params_hash = None
        
        # 当前正在编辑的参数索引
        self.editing_param_idx = -1
        
        # 创建窗口和滑动条（在初始化参数之后）
        self._create_windows_and_trackbars()

    def _init_parameters(self, preprocessor_config: dict, circle_detector_config: dict):
        """初始化所有参数控制器"""
        self.params = [
            # Preprocessor 参数
            ParameterControl("Gaussian Kernel", "GaussK", 
                           preprocessor_config.get('gaussian_blur_kernel', 5), 
                           1, 19, 2, True),
            ParameterControl("Median Kernel", "MedK", 
                           preprocessor_config.get('median_blur_kernel', 3), 
                           1, 19, 2, True),
            ParameterControl("Contrast Alpha", "Contrast", 
                           preprocessor_config.get('contrast_alpha', 1.2), 
                           0.1, 3.0, 0.1, False),
            ParameterControl("Brightness Beta", "Bright", 
                           preprocessor_config.get('brightness_beta', 0), 
                           -100, 100, 1, True),
            ParameterControl("CLAHE Clip Limit", "CLAHE", 
                           preprocessor_config.get('clahe_clip_limit', 2.0), 
                           0.1, 5.0, 0.1, False),
            ParameterControl("Use CLAHE", "UseCLAHE", 
                           1 if preprocessor_config.get('use_clahe', True) else 0, 
                           0, 1, 1, True),
            
            # Circle Detector 参数
            ParameterControl("Hough DP", "DP", 
                           circle_detector_config.get('hough_dp', 1.5), 
                           0.5, 3.0, 0.1, False),
            ParameterControl("Min Distance", "MinDist", 
                           circle_detector_config.get('hough_min_dist', 50), 
                           10, 200, 5, True),
            ParameterControl("Canny Param1", "CannyP1", 
                           circle_detector_config.get('hough_canny_param1', 100), 
                           10, 300, 5, True),
            ParameterControl("Accum Threshold", "AccumT", 
                           circle_detector_config.get('hough_accum_threshold', 80), 
                           10, 200, 5, True),
            ParameterControl("Min Radius", "MinR", 
                           circle_detector_config.get('hough_min_radius', 200), 
                           10, 500, 10, True),
            ParameterControl("Max Radius", "MaxR", 
                           circle_detector_config.get('hough_max_radius', 500), 
                           50, 1000, 10, True),
            ParameterControl("Canny Low", "CannyL", 
                           circle_detector_config.get('canny_low', 50), 
                           0, 255, 5, True),
            ParameterControl("Canny High", "CannyH", 
                           circle_detector_config.get('canny_high', 150), 
                           0, 255, 5, True),
        ]

    def _create_windows_and_trackbars(self):
        """创建窗口和滑动条（双列布局）"""
        # 创建主窗口用于显示图像
        cv2.namedWindow('Image Display', cv2.WINDOW_NORMAL)
        
        # 创建参数控制窗口
        cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)
        
        # 设置窗口位置：图像窗口在左，参数窗口在右
        # 计算合适的窗口尺寸
        max_display_height = 900
        max_display_width = 1200
        
        # 保持图像原始比例
        aspect_ratio = self.orig_width / self.orig_height
        
        if self.orig_height > max_display_height:
            display_height = max_display_height
            display_width = int(display_height * aspect_ratio)
        else:
            display_height = self.orig_height
            display_width = self.orig_width
            
        if display_width > max_display_width:
            display_width = max_display_width
            display_height = int(display_width / aspect_ratio)
        
        # 设置图像窗口大小
        cv2.resizeWindow('Image Display', display_width, display_height)
        
        # 设置参数窗口大小（固定宽度，高度与图像窗口一致）
        param_window_width = 450
        cv2.resizeWindow('Parameters', param_window_width, display_height)
        
        # 移动窗口位置（并排显示）
        cv2.moveWindow('Image Display', 100, 50)
        cv2.moveWindow('Parameters', 100 + display_width + 20, 50)
        
        # 在参数窗口中创建滑动条
        self._create_trackbars_in_panel()

    def _create_trackbars_in_panel(self):
        """在参数面板中创建滑动条（分为两列）"""
        window_name = 'Parameters'
        
        # 添加标题
        print("\n" + "="*60)
        print("参数控制面板已打开")
        print("="*60)
        
        # Preprocessor 参数组
        print("\n【图像预处理参数】")
        for i in range(6):  # 前6个是preprocessor参数
            param = self.params[i]
            if param.is_int:
                trackbar_val = int(param.value)
            else:
                trackbar_val = int(param.value * 10)
            
            max_val = int(param.max_val) if param.is_int else int(param.max_val * 10)
            
            cv2.createTrackbar(f"{param.short_name}", window_name, 
                             trackbar_val, max_val, 
                             lambda pos, idx=i: self._on_trackbar_change(idx, pos))
            print(f"  {i}: {param.name} (当前值: {param.value})")
        
        # 分隔线
        print("\n" + "-"*60)
        print("\n【圆形检测参数】")
        
        # Circle Detector 参数组
        for i in range(6, len(self.params)):  # 后面的都是circle_detector参数
            param = self.params[i]
            if param.is_int:
                trackbar_val = int(param.value)
            else:
                trackbar_val = int(param.value * 10)
            
            max_val = int(param.max_val) if param.is_int else int(param.max_val * 10)
            
            cv2.createTrackbar(f"{param.short_name}", window_name, 
                             trackbar_val, max_val, 
                             lambda pos, idx=i: self._on_trackbar_change(idx, pos))
            print(f"  {i}: {param.name} (当前值: {param.value})")
        
        print("\n" + "="*60)
        print("提示: 使用数字键 0-13 选择参数，直接输入数值后按 Enter 确认")
        print("="*60 + "\n")

    def _on_trackbar_change(self, param_idx: int, pos: int):
        """滑动条变化回调"""
        param = self.params[param_idx]
        if param.is_int:
            param.set_value(float(pos))
        else:
            param.set_value(pos / 10.0)
        
        # 清除缓存，强制重新计算
        self.last_params_hash = None

    def _get_current_config(self) -> tuple:
        """获取当前配置"""
        # Preprocessor 配置
        gaussian_kernel = int(self.params[0].value)
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        
        median_kernel = int(self.params[1].value)
        if median_kernel % 2 == 0:
            median_kernel += 1

        preprocessor_config = {
            'gaussian_blur_kernel': gaussian_kernel,
            'median_blur_kernel': median_kernel,
            'contrast_alpha': self.params[2].value,
            'brightness_beta': int(self.params[3].value),
            'clahe_clip_limit': self.params[4].value,
            'use_clahe': bool(int(self.params[5].value))
        }

        # Circle Detector 配置
        circle_detector_config = {
            'hough_dp': self.params[6].value,
            'hough_min_dist': int(self.params[7].value),
            'hough_canny_param1': int(self.params[8].value),
            'hough_accum_threshold': int(self.params[9].value),
            'hough_min_radius': int(self.params[10].value),
            'hough_max_radius': int(self.params[11].value),
            'canny_low': int(self.params[12].value),
            'canny_high': int(self.params[13].value),
            'use_fixed_roi': False,
            'roi': None
        }

        return preprocessor_config, circle_detector_config

    def _params_changed(self) -> bool:
        """检查参数是否发生变化"""
        current_hash = hash(tuple(p.value for p in self.params))
        if current_hash != self.last_params_hash:
            self.last_params_hash = current_hash
            return True
        return False

    def _process_and_detect(self) -> tuple:
        """执行预处理和圆形检测（带缓存优化）"""
        # 如果参数未改变，返回缓存结果
        if not self._params_changed() and self.last_vis_image is not None:
            return (self.last_preprocessed, self.last_circles, 
                   self.last_best_circle, self.last_vis_image)

        preprocessor_config, circle_detector_config = self._get_current_config()

        # 创建预处理器和检测器
        preprocessor = ImagePreprocessor(preprocessor_config)
        circle_detector = CircleDetector(circle_detector_config)

        # 预处理
        preprocessed = preprocessor.preprocess(self.original_image)

        # 检测圆形
        circles = circle_detector.detect_circles(self.original_image, preprocessed)
        
        # 选择最佳圆形
        best_circle = circle_detector.select_best_circle(circles, self.original_image.shape[:2])

        # 可视化
        vis_image = circle_detector.visualize_circles(self.original_image, circles, best_circle)

        # 缓存结果
        self.last_preprocessed = preprocessed
        self.last_circles = circles
        self.last_best_circle = best_circle
        self.last_vis_image = vis_image

        return preprocessed, circles, best_circle, vis_image

    def _draw_info_overlay(self, display_img: np.ndarray, circles: list, 
                          best_circle: Optional[tuple]) -> np.ndarray:
        """在图像上绘制简要信息（不占用太多空间）"""
        img_copy = display_img.copy()
        
        # 只在左上角显示少量关键信息
        info_lines = [
            f"Mode: {['Original', 'Preprocessed', 'Edge'][self.display_mode]}",
            f"Circles: {len(circles)}"
        ]
        
        if best_circle:
            info_lines.append(f"Best: ({best_circle[0]}, {best_circle[1]}, {best_circle[2]})")
        
        y_offset = 25
        for line in info_lines:
            cv2.putText(img_copy, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
        
        return img_copy

    def _draw_info_panel(self, display_img: np.ndarray, circles: list, 
                        best_circle: Optional[tuple]) -> np.ndarray:
        """绘制信息面板"""
        h, w = display_img.shape[:2]
        
        # 创建半透明背景
        overlay = display_img.copy()
        panel_height = 180
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)

        # 显示模式信息
        modes = ["Original + Detection", "Preprocessed", "Edge Detection"]
        mode_text = f"Mode: {modes[self.display_mode]}"
        cv2.putText(display_img, mode_text, (10, h - panel_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示检测结果
        result_text = f"Circles: {len(circles)} | Best: {best_circle if best_circle else 'None'}"
        cv2.putText(display_img, result_text, (10, h - panel_height + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示参数列表
        y_offset = h - panel_height + 80
        for i, param in enumerate(self.params):
            color = (0, 255, 255) if i == self.editing_param_idx else (200, 200, 200)
            text = param.get_display_text()
            cv2.putText(display_img, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 18

        # 显示操作提示
        help_y = h - 20
        help_text = "Keys: 0-9=edit, Enter=confirm, Esc=cancel, +/-=adjust, m=mode, s=save, l=load, q=quit"
        cv2.putText(display_img, help_text, (10, help_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return display_img

    def run(self):
        """运行交互式调试器"""
        print("=" * 60)
        print("交互式圆形检测参数调试工具（优化版 v2.1）")
        print("=" * 60)
        print("操作说明:")
        print("  - 数字键 0-13: 选择要编辑的参数")
        print("  - 数字键输入: 直接输入参数值")
        print("  - Enter: 确认输入")
        print("  - Esc: 取消输入")
        print("  - +/-: 微调当前选中的参数")
        print("  - m: 切换显示模式")
        print("  - s: 保存配置")
        print("  - l: 加载配置")
        print("  - q: 退出")
        print("=" * 60)
        
        # 显示初始参数状态
        self._print_param_status()

        while True:
            # 执行检测和可视化
            preprocessed, circles, best_circle, vis_image = self._process_and_detect()

            # 根据显示模式选择显示的图像
            if self.display_mode == 0:
                display_img = vis_image.copy()
            elif self.display_mode == 1:
                display_img = preprocessed.copy()
            else:  # mode 2
                _, circle_detector_config = self._get_current_config()
                gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY) if len(preprocessed.shape) == 3 else preprocessed
                edges = cv2.Canny(gray, circle_detector_config['canny_low'], circle_detector_config['canny_high'])
                display_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # 添加简要信息覆盖层
            display_img = self._draw_info_overlay(display_img, circles, best_circle)

            # 在图像窗口显示
            cv2.imshow('Image Display', display_img)

            # 按键处理
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.display_mode = (self.display_mode + 1) % 3
                self.last_params_hash = None  # 强制刷新
                self._print_current_mode()
            elif key == ord('s'):
                self._save_config()
            elif key == ord('l'):
                self._load_config()
            elif key == 27:  # Esc
                if self.editing_param_idx >= 0:
                    self.params[self.editing_param_idx].cancel_input()
                    self.editing_param_idx = -1
                    self._print_param_status()
            elif key == 13 or key == 10:  # Enter
                if self.editing_param_idx >= 0:
                    param = self.params[self.editing_param_idx]
                    param.confirm_input()
                    print(f"✓ 参数 '{param.name}' 已更新为: {param.value}")
                    self.editing_param_idx = -1
                    self.last_params_hash = None
            elif key == ord('+') or key == ord('='):
                if self.editing_param_idx >= 0:
                    param = self.params[self.editing_param_idx]
                    old_value = param.value
                    param.set_value(param.value + param.step)
                    print(f"  {param.short_name}: {old_value} → {param.value}")
                    self._update_trackbar(self.editing_param_idx)
                    self.last_params_hash = None
            elif key == ord('-') or key == ord('_'):
                if self.editing_param_idx >= 0:
                    param = self.params[self.editing_param_idx]
                    old_value = param.value
                    param.set_value(param.value - param.step)
                    print(f"  {param.short_name}: {old_value} → {param.value}")
                    self._update_trackbar(self.editing_param_idx)
                    self.last_params_hash = None
            elif 48 <= key <= 57:  # 数字键 0-9
                digit = key - 48
                if digit < len(self.params):
                    if self.editing_param_idx >= 0:
                        # 正在输入，添加数字
                        self.params[self.editing_param_idx].add_input_char(str(digit))
                    else:
                        # 选择参数进行编辑
                        self.editing_param_idx = digit
                        self.params[digit].is_inputting = True
                        print(f"\n→ 正在编辑: {self.params[digit].name} (当前值: {self.params[digit].value})")
                        print("  输入新数值后按 Enter 确认，或按 Esc 取消")
            else:
                # 其他按键，如果在编辑状态则取消
                if self.editing_param_idx >= 0:
                    self.params[self.editing_param_idx].cancel_input()
                    self.editing_param_idx = -1
                    self._print_param_status()

        cv2.destroyAllWindows()

    def _print_param_status(self):
        """打印当前参数状态"""
        print("\n" + "-"*60)
        print("当前参数状态:")
        print("-"*60)
        
        # Preprocessor 参数
        print("【图像预处理】")
        for i in range(6):
            param = self.params[i]
            marker = " ← EDITING" if i == self.editing_param_idx else ""
            print(f"  {i}: {param.short_name:10s} = {param.value:>8}{marker}")
        
        # Circle Detector 参数
        print("\n【圆形检测】")
        for i in range(6, len(self.params)):
            param = self.params[i]
            marker = " ← EDITING" if i == self.editing_param_idx else ""
            print(f"  {i}: {param.short_name:10s} = {param.value:>8}{marker}")
        
        print("-"*60)

    def _print_current_mode(self):
        """打印当前显示模式"""
        modes = ["原始图像 + 检测结果", "预处理后图像", "边缘检测结果"]
        print(f"\n→ 显示模式切换为: {modes[self.display_mode]}")

    def _update_trackbar(self, param_idx: int):
        """更新滑动条位置"""
        param = self.params[param_idx]
        if param.is_int:
            trackbar_val = int(param.value)
        else:
            trackbar_val = int(param.value * 10)
        cv2.setTrackbarPos(param.short_name, 'Parameters', trackbar_val)

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
                cfg = full_config['preprocessor']
                self.params[0].set_value(cfg.get('gaussian_blur_kernel', 5))
                self.params[1].set_value(cfg.get('median_blur_kernel', 3))
                self.params[2].set_value(cfg.get('contrast_alpha', 1.2))
                self.params[3].set_value(cfg.get('brightness_beta', 0))
                self.params[4].set_value(cfg.get('clahe_clip_limit', 2.0))
                self.params[5].set_value(1 if cfg.get('use_clahe', True) else 0)
                
            if 'circle_detector' in full_config:
                cfg = full_config['circle_detector']
                self.params[6].set_value(cfg.get('hough_dp', 1.5))
                self.params[7].set_value(cfg.get('hough_min_dist', 50))
                self.params[8].set_value(cfg.get('hough_canny_param1', 100))
                self.params[9].set_value(cfg.get('hough_accum_threshold', 80))
                self.params[10].set_value(cfg.get('hough_min_radius', 200))
                self.params[11].set_value(cfg.get('hough_max_radius', 500))
                self.params[12].set_value(cfg.get('canny_low', 50))
                self.params[13].set_value(cfg.get('canny_high', 150))

            # 更新滑动条位置
            for i in range(len(self.params)):
                self._update_trackbar(i)
            
            # 清除缓存
            self.last_params_hash = None
            
            print(f"✓ 配置已从 {load_path} 加载")
            self._print_param_status()
        except Exception as e:
            print(f"✗ 加载配置失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='交互式圆形检测参数调试工具（优化版）')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--config', type=str, default='config/detector_config.json', 
                       help='初始配置文件路径 (可选)')
    
    args = parser.parse_args()
    
    debugger = InteractiveDebugger(args.image, args.config)
    debugger.run()


if __name__ == '__main__':
    main()
