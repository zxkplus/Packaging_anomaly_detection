# 酒瓶标签检测算法使用文档

## 项目概述

本项目实现了一个完整的酒瓶丝印标签检测系统，主要用于检测标签的以下缺陷：

1. **贴重检测**：检测标签是否重复粘贴（重叠标签）
2. **贴正检测**：检测标签是否贴正（旋转偏差）

## 算法流程

```
原始图像
    ↓
[1] 图像预处理（去噪、增强对比度）
    ↓
[2] ROI区域提取
    ↓
[3] 圆形检测（霍夫圆变换）
    ↓
[4] 极坐标变换/图像归一化
    ↓
[5] 缺陷检测（贴重、贴正）
    ↓
检测结果
```

## 目录结构

```
.
├── config/
│   └── detector_config.json          # 检测算法配置文件
├── src/
│   └── label_detector/
│       ├── __init__.py               # 包初始化文件
│       ├── preprocessor.py           # 图像预处理模块
│       ├── circle_detector.py        # 圆形检测模块
│       ├── polar_transform.py        # 极坐标变换模块
│       ├── defect_detector.py        # 缺陷检测模块
│       └── detector.py               # 主检测器
├── assets/
│   ├── test_*.jpg                    # 测试图像
│   └── output/                       # 检测结果输出目录
├── example_detection.py              # 示例代码
└── README_LABEL_DETECTOR.md          # 本文档
```

## 快速开始

### 1. 安装依赖

项目使用 `uv` 管理依赖，主要依赖包括：
- opencv-python >= 4.12
- numpy
- 其他依赖已在 `pyproject.toml` 中声明

### 2. 运行示例

```bash
# 运行完整示例
python example_detection.py
```

示例会自动生成测试图像，并演示以下功能：
- 单张图像检测
- 批量检测
- 可视化处理流程
- 自定义配置

### 3. 基本使用

```python
from src.label_detector import LabelDetector, get_detector
import cv2

# 创建检测器（使用默认配置）
detector = get_detector("config/detector_config.json")

# 方法1：从文件检测
result = detector.detect("path/to/image.jpg", train_mode=False)

# 方法2：从numpy数组检测
image = cv2.imread("path/to/image.jpg")
result = detector.detect_image(image, train_mode=False)

# 查看结果
print(result['final_result'])
# {'overall_status': 'OK' or 'NG', 'total_defects': 0 or >0, 'defects': [...]}
```

## 核心功能

### 1. 训练标准样本

在检测之前，需要使用正常标签图像训练检测器：

```python
# 训练模式
result = detector.detect("normal_sample.jpg", train_mode=True)
print(result['final_result'])
# {'status': 'trained', 'message': '标准样本学习完成'}
```

训练后，检测器会学习：
- 标准标签的外观特征
- 参考角度（用于贴正检测）
- 边缘密度和亮度分布（用于贴重检测）

### 2. 检测图像

```python
# 检测模式
result = detector.detect("test_image.jpg", train_mode=False)

# 查看总体结果
print(result['final_result'])
# {
#     'overall_status': 'OK' or 'NG',
#     'total_defects': 0 or >0,
#     'defects': [
#         {'type': 'duplicate', 'confidence': 0.85, 'details': {...}},
#         {'type': 'rotation', 'confidence': 0.92, 'details': {...}}
#     ]
# }

# 查看详细缺陷信息
for defect in result['defect_results']:
    print(f"缺陷类型: {defect['defect_type']}")
    print(f"是否缺陷: {defect['is_defect']}")
    print(f"置信度: {defect['confidence']:.2f}")
    print(f"详细信息: {defect['details']}")
```

### 3. 批量检测

```python
# 批量检测目录中的所有图像
results = detector.batch_detect(
    image_dir="path/to/images",
    train_image="normal_sample.jpg"  # 可选：指定训练图像
)

# 统计结果
total = len(results)
ok_count = sum(1 for r in results if r['final_result']['overall_status'] == 'OK')
ng_count = sum(1 for r in results if r['final_result']['overall_status'] == 'NG')

print(f"总数: {total}, OK: {ok_count}, NG: {ng_count}")
```

### 4. 可视化

```python
# 可视化处理流程
vis = detector.visualize_full_process(
    "image.jpg",
    save_path="output/visualize.jpg"
)

# 检测结果会自动保存到 assets/output/ 目录（如果配置开启）
```

## 配置说明

配置文件位于 `config/detector_config.json`，包含以下部分：

### 1. 预处理配置 (preprocessor)

```json
{
  "preprocessor": {
    "gaussian_blur_kernel": 5,      // 高斯模糊核大小
    "median_blur_kernel": 3,        // 中值滤波核大小
    "contrast_alpha": 1.2,          // 对比度增强系数
    "brightness_beta": 0,           // 亮度调整值
    "use_clahe": true,              // 是否使用CLAHE
    "clahe_clip_limit": 2.0         // CLAHE对比度限制
  }
}
```

### 2. 圆形检测配置 (circle_detector)

```json
{
  "circle_detector": {
    "hough_dp": 1.5,                // 霍夫变换分辨率比率
    "hough_min_dist": 50,           // 圆心最小间距
    "hough_canny_param1": 100,      // Canny高阈值
    "hough_accum_threshold": 80,    // 累加器阈值（越小越敏感）
    "hough_min_radius": 30,         // 最小半径
    "hough_max_radius": 200,        // 最大半径
    "canny_low": 50,                // Canny低阈值
    "canny_high": 150,              // Canny高阈值
    "use_fixed_roi": true,          // 是否使用固定ROI
    "roi": null                     // ROI区域 [x, y, width, height]
  }
}
```

**调优建议**：
- 如果检测不到圆形：降低 `hough_accum_threshold`
- 如果检测到太多误检：提高 `hough_accum_threshold` 和 `hough_min_dist`
- 如果标签大小固定：设置准确的 `hough_min_radius` 和 `hough_max_radius`

### 3. 极坐标变换配置 (polar_transform)

```json
{
  "polar_transform": {
    "polar_output_height": 200,     // 极坐标图高度（对应半径）
    "polar_output_width": 360,      // 极坐标图宽度（对应角度）
    "normalize_circle_size": 300,   // 归一化圆形大小
    "interpolation": 1              // 插值方法
  }
}
```

### 4. 缺陷检测配置 (defect_detector)

```json
{
  "defect_detector": {
    "edge_density_threshold": 1.5,   // 边缘密度阈值（贴重检测）
    "brightness_var_threshold": 30.0,// 亮度方差阈值（贴重检测）
    "detection_radius_ratio": 0.8,   // 检测半径比例
    "angle_tolerance": 15.0,         // 角度容差（度，贴正检测）
    "reference_angle": 0.0,          // 参考角度（度）
    "use_auto_reference": true,      // 是否自动学习参考角度
    "template_match_threshold": 0.7, // 模板匹配阈值
    "template_path": null            // 标准模板路径
  }
}
```

**调优建议**：
- **贴重检测**：
  - 降低 `edge_density_threshold` 和 `brightness_var_threshold`：更敏感（可能增加误报）
  - 提高这两个阈值：更严格（可能漏检轻微贴重）

- **贴正检测**：
  - 降低 `angle_tolerance`：更严格的角度要求
  - 提高 `angle_tolerance`：更宽松的角度要求

### 5. 其他配置

```json
{
  "visualization": true,             // 是否生成可视化结果
  "save_intermediate": true,         // 是否保存中间结果
  "output_dir": "assets/output"      // 输出目录
}
```

## 动态更新配置

```python
# 获取当前配置
config = detector.get_config()

# 更新配置
new_config = {
    'defect_detector': {
        'angle_tolerance': 10.0,      // 更严格的角度要求
        'edge_density_threshold': 1.2  // 更敏感的贴重检测
    }
}
detector.update_config(new_config)
```

## 检测结果说明

### 结果结构

```python
{
    'status': 'success' | 'failed' | 'error',
    'image_name': 'image.jpg',
    'steps': {
        'preprocessing': 'success',
        'circle_detection': 'success',
        'polar_transform': 'success',
        'defect_detection': 'success'
    },
    'circle': (center_x, center_y, radius),
    'defect_results': [
        {
            'is_defect': True | False,
            'confidence': 0.0 ~ 1.0,
            'defect_type': 'duplicate' | 'rotation',
            'angle_offset': 0.0,      # 仅旋转检测
            'details': {
                # 贴重检测细节
                'diff_mean': 42.31,
                'diff_std': 67.59,
                'edge_density': 0.225,
                'reason': '亮度差异过大 (42.31 > 30.0)',

                # 或旋转检测细节
                'current_angle': 116.0,
                'reference_angle': 0.0,
                'reason': '角度偏移过大 (116.00° > 15.0°)'
            }
        }
    ],
    'final_result': {
        'overall_status': 'OK' | 'NG',
        'total_defects': 0,
        'defects': [...]
    }
}
```

## 性能优化建议

### 1. ROI 优化

如果标签位置相对固定，设置固定ROI可以显著提高速度：

```python
config = {
    'circle_detector': {
        'use_fixed_roi': True,
        'roi': [x, y, width, height]  # 手动设置ROI
    }
}
detector.update_config(config)
```

### 2. 圆形半径范围优化

设置准确的半径范围可以减少误检和提高速度：

```python
config = {
    'circle_detector': {
        'hough_min_radius': 80,   # 根据实际标签大小调整
        'hough_max_radius': 120
    }
}
detector.update_config(config)
```

### 3. 关闭可视化

生产环境可以关闭可视化以提高速度：

```python
config = {
    'visualization': False,
    'save_intermediate': False
}
detector.update_config(config)
```

## 常见问题

### Q1: 检测不到圆形标签

**可能原因**：
- 霍夫变换阈值太高
- 圆形半径范围设置不正确
- 光照不足或图像质量差

**解决方案**：
- 降低 `hough_accum_threshold`
- 检查并调整 `hough_min_radius` 和 `hough_max_radius`
- 增加图像预处理强度

### Q2: 贴重检测误报率高

**可能原因**：
- 阈值设置太敏感
- 训练样本不够代表性

**解决方案**：
- 提高 `edge_density_threshold` 和 `brightness_var_threshold`
- 使用更多正常样本进行训练
- 检查光照条件是否一致

### Q3: 贴正检测不准确

**可能原因**：
- 标签上的特征不够明显
- 角度容差设置不合理

**解决方案**：
- 确保标签上有清晰的特征（文字、图案）
- 调整 `angle_tolerance`
- 使用高质量的训练样本

### Q4: 检测速度太慢

**可能原因**：
- 图像分辨率太高
- 没有使用ROI
- 可视化开启

**解决方案**：
- 降低输入图像分辨率
- 设置固定ROI
- 关闭可视化和中间结果保存

## 输出文件说明

如果开启了 `save_intermediate`，以下文件会保存到 `assets/output/` 目录：

- `{image_name}_1_preprocessed.jpg` - 预处理后的图像
- `{image_name}_2_circles.jpg` - 圆形检测结果
- `{image_name}_3_normalized.jpg` - 归一化的圆形图像
- `{image_name}_4_polar.jpg` - 极坐标展开图像
- `{image_name}_5_result.jpg` - 缺陷检测结果可视化

## 技术支持

如需进一步优化或有技术问题，请检查：
1. 图像质量和光照条件
2. 配置参数是否合理
3. 训练样本是否具有代表性

## 许可证

本项目仅用于学习和研究目的。
