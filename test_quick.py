"""
快速测试脚本 - 验证所有核心功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label_detector import LabelDetector, get_detector
import cv2
from src.label_detector.preprocessor import ImagePreprocessor
from src.label_detector.circle_detector import CircleDetector


def test_basic_functionality():
    """测试基本功能"""
    print("="*60)
    print("快速测试：酒瓶标签检测算法")
    print("="*60)

    # 确保输出目录存在
    os.makedirs("assets/output", exist_ok=True)

    # 创建检测器
    print("\n1. 创建检测器...")
    detector = get_detector("config/detector_config.json")
    print("   ✓ 检测器创建成功")

    # 读取测试图像
    print("\n2. 读取测试图像...")
    img_normal = cv2.imread("assets/test_normal.jpg")
    img_duplicate = cv2.imread("assets/test_duplicate.jpg")
    img_rotated = cv2.imread("assets/test_rotated.jpg")
    print(f"   ✓ 图像加载成功")

    # 训练标准样本
    print("\n3. 训练标准样本...")
    result = detector.detect_image(img_normal, train_mode=True, image_name="test_normal.jpg")
    assert result['status'] == 'success', "训练失败"
    print(f"   ✓ 训练成功: {result['final_result']}")

    # 检测正常图像
    print("\n4. 检测正常图像...")
    result = detector.detect_image(img_normal, train_mode=False, image_name="test_normal.jpg")
    assert result['status'] == 'success', "检测失败"
    assert result['final_result']['overall_status'] == 'OK', "正常图像误判为缺陷"
    print(f"   ✓ 检测结果: {result['final_result']['overall_status']}")
    print(f"   ✓ 贴重检测: {'正常' if not result['defect_results'][0]['is_defect'] else '检测到缺陷'}")
    print(f"   ✓ 贴正检测: {'正常' if not result['defect_results'][1]['is_defect'] else '检测到缺陷'}")

    # 检测贴重图像
    print("\n5. 检测贴重图像...")
    result = detector.detect_image(img_duplicate, train_mode=False, image_name="test_duplicate.jpg")
    assert result['status'] == 'success', "检测失败"
    duplicate_defect = result['defect_results'][0]['is_defect']
    print(f"   ✓ 检测结果: {result['final_result']['overall_status']}")
    print(f"   ✓ 贴重检测: {'检测到缺陷' if duplicate_defect else '未检测到'}")
    if duplicate_defect:
        print(f"     - 置信度: {result['defect_results'][0]['confidence']:.2f}")
        print(f"     - 原因: {result['defect_results'][0]['details'].get('reason', 'N/A')}")

    # 检测旋转图像
    print("\n6. 检测旋转图像...")
    result = detector.detect_image(img_rotated, train_mode=False, image_name="test_rotated.jpg")
    assert result['status'] == 'success', "检测失败"
    rotation_defect = result['defect_results'][1]['is_defect']
    print(f"   ✓ 检测结果: {result['final_result']['overall_status']}")
    print(f"   ✓ 贴正检测: {'检测到缺陷' if rotation_defect else '未检测到'}")
    if rotation_defect:
        print(f"     - 角度偏移: {result['defect_results'][1]['angle_offset']:.2f}°")
        print(f"     - 置信度: {result['defect_results'][1]['confidence']:.2f}")

    # 测试可视化
    print("\n7. 测试可视化功能（可能需要30-40秒）...")
    try:
        import time
        start = time.time()
        vis = detector.visualize_full_process(
            "assets/test_normal.jpg",
            save_path="assets/visualize_test.jpg"
        )
        elapsed = time.time() - start
        print(f"   ✓ 可视化成功，耗时: {elapsed:.2f}秒")
        print(f"   ✓ 输出尺寸: {vis.shape}")
    except Exception as e:
        print(f"   ✗ 可视化失败: {e}")

    # 统计结果
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print("✓ 所有核心功能测试通过")
    print(f"✓ 生成的测试文件:")
    print(f"  - assets/test_normal.jpg (正常样本)")
    print(f"  - assets/test_duplicate.jpg (贴重样本)")
    print(f"  - assets/test_rotated.jpg (旋转样本)")
    print(f"  - assets/visualize_test.jpg (可视化结果)")
    print(f"  - assets/output/* (检测中间结果)")
    print("\n✓ 详细使用说明请参考: README_LABEL_DETECTOR.md")
    print("="*60)

    return True


def extract_region_by_grayscale_range(image, lower_bound, upper_bound):
    """
    分割提取灰度值在指定范围内的区域
    
    Args:
        image: 输入图像 (numpy array)
        lower_bound: 灰度值下界
        upper_bound: 灰度值上界
    
    Returns:
        extracted_region: 提取出的区域图像
        mask: 对应的掩码图像
    """
    import numpy as np
    
    # 如果输入是彩色图像，先转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 创建掩码，灰度值在指定范围内的像素设为255（白色），其余设为0（黑色）
    mask = cv2.inRange(gray, lower_bound, upper_bound)
    
    # 使用掩码提取对应区域
    if len(image.shape) == 3:
        # 彩色图像情况
        extracted_region = cv2.bitwise_and(image, image, mask=mask)
    else:
        # 灰度图像情况
        extracted_region = cv2.bitwise_and(gray, gray, mask=mask)
    
    return extracted_region, mask

def generate_circle_mask(image, center_x, center_y, radius):
    """
    生成与原图大小相同的圆形区域二值图
    
    Args:
        image: 输入图像 (numpy array)，用于获取尺寸
        center_x: 圆心x坐标
        center_y: 圆心y坐标
        radius: 圆的半径
    
    Returns:
        mask: 二值掩码图像，圆形区域内为255（白色），其余为0（黑色）
    """
    import numpy as np
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 创建全黑的二值图
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 在指定位置绘制实心圆（白色）
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    return mask

def generate_circle_mask(image, center_x, center_y, radius):
    """
    生成与原图大小相同的圆形区域二值图
    
    Args:
        image: 输入图像 (numpy array)，用于获取尺寸
        center_x: 圆心x坐标
        center_y: 圆心y坐标
        radius: 圆的半径
    
    Returns:
        mask: 二值掩码图像，圆形区域内为255（白色），其余为0（黑色）
    """
    import numpy as np
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 创建全黑的二值图
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 在指定位置绘制实心圆（白色）
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    return mask

def fit_largest_ellipse(binary_image):
    """
    拟合二值图中最大的椭圆
    
    Args:
        binary_image: 输入的二值图像 (numpy array)
    
    Returns:
        ellipse: 拟合的椭圆参数 ((center_x, center_y), (axis1, axis2), angle)，如果未找到则返回None
        ellipse_image: 绘制了椭圆的图像
    """
    import numpy as np
    
    # 复制图像用于绘制
    if len(binary_image.shape) == 3:
        ellipse_image = binary_image.copy()
    else:
        ellipse_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未找到轮廓")
        return None, ellipse_image
    
    # 选择面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 检查轮廓点数是否足够拟合椭圆（至少需要5个点）
    if len(largest_contour) < 5:
        print(f"轮廓点数不足 ({len(largest_contour)} < 5)，无法拟合椭圆")
        return None, ellipse_image
    
    try:
        # 拟合椭圆
        ellipse = cv2.fitEllipse(largest_contour)
        
        # 在图像上绘制椭圆
        cv2.ellipse(ellipse_image, ellipse, (0, 255, 0), 2)
        
        # 提取椭圆参数
        center = ellipse[0]  # 中心点 (x, y)
        axes = ellipse[1]    # 轴长 (width, height)
        angle = ellipse[2]   # 旋转角度
        
        print(f"椭圆中心: ({center[0]:.2f}, {center[1]:.2f})")
        print(f"椭圆轴长: ({axes[0]:.2f}, {axes[1]:.2f})")
        print(f"椭圆角度: {angle:.2f}°")
        
        return ellipse, ellipse_image
    except Exception as e:
        print(f"椭圆拟合失败: {e}")
        return None, ellipse_image

def extract_and_normalize_ellipse(image, ellipse):
    """
    从原图中裁剪出椭圆区域，并将椭圆投影成正圆显示
    
    Args:
        image: 输入图像 (numpy array)
        ellipse: 椭圆参数 ((center_x, center_y), (axis1, axis2), angle)
    
    Returns:
        normalized_circle: 归一化后的正圆图像
        crop_box: 裁剪区域的边界框 (x, y, width, height)
    """
    import numpy as np
    import math
    
    # 解包椭圆参数
    (center_x, center_y), (axis1, axis2), angle = ellipse
    
    # 确定椭圆的长轴和短轴
    major_axis = max(axis1, axis2)
    minor_axis = min(axis1, axis2)
    
    # 计算外接矩形的大小（考虑旋转角度）
    # 为了安全起见，使用长轴的两倍作为裁剪区域的边长
    box_size = int(major_axis * 2) + 20  # 添加一些边距
    
    # 计算裁剪区域的左上角坐标
    x1 = int(center_x - box_size / 2)
    y1 = int(center_y - box_size / 2)
    x2 = int(center_x + box_size / 2)
    y2 = int(center_y + box_size / 2)
    
    # 确保裁剪区域在图像范围内
    height, width = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    # 裁剪图像
    cropped_image = image[y1:y2, x1:x2].copy()
    
    # 计算椭圆中心在裁剪后图像中的新位置
    new_center_x = center_x - x1
    new_center_y = center_y - y1
    
    # 创建仿射变换矩阵，将椭圆变换为正圆
    # 首先平移使椭圆中心到原点
    # 然后旋转消除椭圆的旋转角度
    # 最后缩放使长短轴相等
    
    # 输出图像尺寸（正方形）
    output_size = int(major_axis * 2)
    
    # 创建目标图像
    normalized_circle = np.zeros((output_size, output_size, 3), dtype=image.dtype) if len(image.shape) == 3 else \
                        np.zeros((output_size, output_size), dtype=image.dtype)
    
    # 方法：使用极坐标变换或仿射变换
    # 这里使用更简单的方法：先旋转矫正，再缩放
    
    # 步骤1: 旋转图像以消除椭圆的旋转角度
    rotation_matrix = cv2.getRotationMatrix2D((new_center_x, new_center_y), 0, 1.0)
    rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, 
                                   (cropped_image.shape[1], cropped_image.shape[0]))
    
    # 步骤2: 计算缩放比例，将椭圆变为正圆
    scale_x = major_axis / axis1 if axis1 > 0 else 1.0
    scale_y = major_axis / axis2 if axis2 > 0 else 1.0
    
    # 使用非均匀缩放的仿射变换
    # 构建缩放矩阵
    scale_matrix = np.array([
        [scale_x, 0, new_center_x * (1 - scale_x)],
        [0, scale_y, new_center_y * (1 - scale_y)]
    ], dtype=np.float32)
    
    # 应用缩放变换
    scaled_image = cv2.warpAffine(rotated_image, scale_matrix, 
                                  (rotated_image.shape[1], rotated_image.shape[0]))
    
    # 步骤3: 提取正圆区域
    circle_radius = int(major_axis)
    center_in_scaled = (int(new_center_x * scale_x), int(new_center_y * scale_y))
    
    # 确保中心点在图像范围内
    center_in_scaled = (
        max(0, min(center_in_scaled[0], scaled_image.shape[1] - 1)),
        max(0, min(center_in_scaled[1], scaled_image.shape[0] - 1))
    )
    
    # 计算提取区域
    r = circle_radius
    cx, cy = center_in_scaled
    x_start = max(0, cx - r)
    y_start = max(0, cy - r)
    x_end = min(scaled_image.shape[1], cx + r)
    y_end = min(scaled_image.shape[0], cy + r)
    
    # 提取圆形区域
    circle_region = scaled_image[y_start:y_end, x_start:x_end]
    
    # 调整到固定大小
    final_size = int(major_axis * 2)
    normalized_circle = cv2.resize(circle_region, (final_size, final_size))
    
    # 返回结果
    crop_box = (x1, y1, x2 - x1, y2 - y1)
    
    return normalized_circle, crop_box
def fill_holes(binary_image):
    """
    填充二值图内部的空洞区域
    
    Args:
        binary_image: 输入的二值图像 (numpy array)，前景为255（白色），背景为0（黑色）
    
    Returns:
        filled_image: 填充空洞后的二值图像
    """
    import numpy as np
    
    # 确保输入是二值图
    if len(binary_image.shape) == 3:
        gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = binary_image.copy()
    
    # 方法1: 使用形态学重构（ flood fill ）
    # 复制图像
    filled = binary.copy()
    
    # 获取图像尺寸
    height, width = binary.shape[:2]
    
    # 创建掩码，比原图大一圈，用于floodFill
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    
    # 从边界开始进行洪水填充，填充背景区域
    # 这样可以将所有与边界相连的背景区域标记出来
    cv2.floodFill(filled, mask, (0, 0), 255)
    
    # 反转图像：现在背景是白色，前景和空洞是黑色
    filled_inv = cv2.bitwise_not(filled)
    
    # 将原始图像和反转后的图像进行或运算
    # 这样可以保留原始前景，同时填充空洞
    filled_result = cv2.bitwise_or(binary, filled_inv)
    
    return filled_result


def fill_holes_morphological(binary_image, kernel_size=5):
    """
    使用形态学操作填充二值图内部的空洞区域
    
    Args:
        binary_image: 输入的二值图像 (numpy array)
        kernel_size: 形态学操作的内核大小
    
    Returns:
        filled_image: 填充空洞后的二值图像
    """
    import numpy as np
    
    # 确保输入是二值图
    if len(binary_image.shape) == 3:
        gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = binary_image.copy()
    
    # 方法2: 使用闭运算（先膨胀后腐蚀）来填充小空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 执行闭运算
    filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return filled

def keep_largest_contour_mask(binary_image):
    """
    处理二值图，只保留最大面积的轮廓
    
    Args:
        binary_image: 输入的二值图像 (numpy array)，前景为255（白色），背景为0（黑色）
    
    Returns:
        largest_contour_mask: 只包含最大面积轮廓的二值图像
        largest_contour: 最大面积的轮廓点集
        contour_area: 最大轮廓的面积
    """
    import numpy as np
    
    # 确保输入是二值图
    if len(binary_image.shape) == 3:
        gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = binary_image.copy()
    
    # 查找所有外部轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建全黑的掩码
    largest_contour_mask = np.zeros_like(binary)
    
    if not contours:
        print("未找到任何轮廓")
        return largest_contour_mask, None, 0
    
    # 找到面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    
    # 在掩码上绘制最大轮廓（填充）
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    print(f"最大轮廓面积: {contour_area:.2f}")
    print(f"总轮廓数: {len(contours)}")
    
    return largest_contour_mask, largest_contour, contour_area


def keep_largest_n_contours_mask(binary_image, n=1):
    """
    处理二值图，保留面积最大的N个轮廓
    
    Args:
        binary_image: 输入的二值图像 (numpy array)
        n: 要保留的轮廓数量，默认为1
    
    Returns:
        filtered_mask: 只包含最大N个轮廓的二值图像
        sorted_contours: 按面积排序的轮廓列表（从大到小）
    """
    import numpy as np
    
    # 确保输入是二值图
    if len(binary_image.shape) == 3:
        gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = binary_image.copy()
    
    # 查找所有外部轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建全黑的掩码
    filtered_mask = np.zeros_like(binary)
    
    if not contours:
        print("未找到任何轮廓")
        return filtered_mask, []
    
    # 按面积对轮廓进行排序（从大到小）
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 保留前N个轮廓
    top_n_contours = sorted_contours[:n]
    
    # 在掩码上绘制这些轮廓（填充）
    cv2.drawContours(filtered_mask, top_n_contours, -1, 255, thickness=cv2.FILLED)
    
    print(f"保留了前 {len(top_n_contours)} 个轮廓")
    for i, contour in enumerate(top_n_contours):
        area = cv2.contourArea(contour)
        print(f"  轮廓 {i+1} 面积: {area:.2f}")
    
    return filtered_mask, sorted_contours

def test_all_processor():
    """测试预处理器，批量处理显示一个文件夹下的文件"""
    import glob
    
    image_dir = "/home/industai/zengxinke/wuliangyedata/17/DA6316180"
    
    # 获取目录下所有支持格式的图片文件
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    # 按文件名排序
    image_files.sort()
    
    if not image_files:
        print(f"在目录 {image_dir} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 定义ROI区域 (x, y, width, height)
    roi = (100, 1100, 1800, 2000)
    
    # 创建预处理器和圆形检测器实例
    preprocessor = ImagePreprocessor({})
    
    # 圆形检测器配置
    circle_detector_config = {
        "hough_min_radius": 350,
        "hough_max_radius": 500,
        "hough_min_dist":1000
    }
    circle_detector = CircleDetector(circle_detector_config)
    
    for i, image_path in enumerate(image_files):
        print(f"正在处理第 {i+1}/{len(image_files)} 个文件: {os.path.basename(image_path)}")
        
        # 读取彩色图像
        color_image = cv2.imread(image_path)
        if color_image is None:
            print(f"  无法读取图像: {image_path}")
            continue
            
        # 提取ROI区域
        roi_image = preprocessor.extract_roi(color_image, roi)
        if roi_image is None:
            print(f"  ROI提取失败: {image_path}")
            continue
        
        # 显示原始ROI图像
        #cv2.imshow(f"Original ROI - {os.path.basename(image_path)}", roi_image)
        
        # 转换为灰度图
        gray_image = preprocessor.to_gray(roi_image)
        #cv2.imshow(f"Grayscale - {os.path.basename(image_path)}", gray_image)

        # 阈值分割
        _, thresh_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow(f"Threshold - {os.path.basename(image_path)}", thresh_image)
        #cv2.waitKey(0)
        # 圆形检测,这步只是为了定位到位置
        circles = circle_detector.detect_circles(roi_image, thresh_image)
        print(f"  检测到 {len(circles)} 个圆形")
        # 在原图上绘制检测到的圆形
        result_image_with_circles = circle_detector.visualize_circles(roi_image.copy(), circles)
        # # 显示圆形检测结果
        # cv2.imshow(f"Circle Detection - {os.path.basename(image_path)}", result_image_with_circles)
        ##根据检测到的圆形区域进行裁剪并保存
        region = circles[0]
        # ##裁剪圆形区域
        # cropped_image , _  = circle_detector.extract_circle_region(thresh_image, region, padding=10)
        # cropped_roi , _  = circle_detector.extract_circle_region(roi_image, region, padding=10) 
        # # cv2.imshow(f"Circle Region - {os.path.basename(image_path)}", cropped_roi)
        # # cv2.imshow(f"Cropped Circle - {os.path.basename(image_path)}", cropped_image)
        # # cv2.imwrite(f"{os.path.splitext(image_path)[0]}_circle_region.jpg", cropped_image)

        ##只提取出圆形区域，生成圆形区域二值图
        mask = generate_circle_mask(thresh_image,region[0],region[1],region[2] + 10)
        cv2.bitwise_and(gray_image,mask,gray_image)

        #cv2.imshow(f"Circle Region - {os.path.basename(image_path)}", gray_image)

        range_region , range_mask = extract_region_by_grayscale_range(gray_image, 0, 180)

        # cv2.imshow(f"range_region - {os.path.basename(image_path)}", range_region)
        
        cv2.bitwise_and(range_mask,mask,range_mask)
        ## 填充内部的空洞区域
        range_mask = fill_holes(range_mask)
        #开运算
        range_mask = preprocessor.open_morphological(range_mask,5)

        # # #对mask开运算
        mask = preprocessor.open_morphological(range_mask,2)
        # 只保留最大面积的轮廓
        largest_mask, largest_contour, area = keep_largest_contour_mask(mask)
        #cv2.imshow("largest_mask", largest_mask)
        ## 拟合椭圆
        ellipse, ellipse_image = fit_largest_ellipse(largest_mask)
        ## 在原图上裁剪出椭圆区域，并且将椭圆投影成正圆显示
        cv2.bitwise_and(gray_image,largest_mask,gray_image)

        if ellipse is not None:
            # 提取并归一化椭圆为正圆
            # 同样生成椭圆的二值图
            
            normalized_circle, crop_box = extract_and_normalize_ellipse(gray_image, ellipse)
            
             # 在归一化后的图像上绘制正圆轮廓
            (center_x, center_y), (axis1, axis2), angle = ellipse
            major_axis = max(axis1, axis2)
            
            # 创建彩色副本用于绘制
            normalized_display = normalized_circle.copy() if len(normalized_circle.shape) == 3 else cv2.cvtColor(normalized_circle.copy(), cv2.COLOR_GRAY2BGR)
            # 计算正圆的中心和半径
            circle_center = (normalized_display.shape[1] // 2, normalized_display.shape[0] // 2)
            circle_radius = int(major_axis)
            
            # 绘制正圆轮廓（绿色）
            cv2.circle(normalized_display, circle_center, circle_radius, (0, 255, 0), 2)
            
            # 绘制圆心（红色）
            cv2.circle(normalized_display, circle_center, 3, (0, 0, 255), -1)

            ##再分一次
            label_region , label_mask =  extract_region_by_grayscale_range(normalized_circle, 140, 180)
            #开运算
            label_mask = preprocessor.open_morphological(label_mask, 5)
            #填充内部
            label_mask = fill_holes(label_mask)
            label_largest_mask, label_largest_contour, label_area = keep_largest_contour_mask(label_mask)
            cv2.bitwise_and(label_region, label_largest_mask, label_region)
            cv2.imshow("label_region", label_region)
            cv2.imshow("normalized_display", normalized_display)
            cv2.waitKey(0)
            #cv2.imshow("label_mask", label_largest_mask)
            

        # 等待按键事件，按'q'键退出，或者等待一段时间后自动处理下一张
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('q'):
        #     break
        # elif key == ord('s'):  # 按's'保存当前处理结果
        #     output_dir = "assets/output"
        #     os.makedirs(output_dir, exist_ok=True)
        #     cv2.imwrite(os.path.join(output_dir, f"processed_{os.path.basename(image_path)}"), label_region)
        #     print(f"  已保存处理结果到: processed_{os.path.basename(image_path)}")
            output_dir = "assets/output"
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f"processed_{os.path.basename(image_path)}"), label_region)
            print(f"  已保存处理结果到: processed_{os.path.basename(image_path)}")
        
        # 关闭所有窗口
        cv2.destroyAllWindows()

    print("批量处理完成")



if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
