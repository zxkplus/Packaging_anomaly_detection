"""
标签检测算法示例代码
演示如何使用标签检测器进行检测
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label_detector import LabelDetector, get_detector
import cv2
import numpy as np


def create_test_image():
    """
    创建测试用的模拟图像（带圆形标签）
    """
    # 创建黑色背景
    img = np.zeros((600, 800, 3), dtype=np.uint8)

    # 添加一些噪声
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    # 添加一些背景纹理
    cv2.rectangle(img, (100, 100), (700, 500), (100, 100, 100), 2)

    # 创建圆形标签（正常）
    center = (400, 300)
    radius = 100
    cv2.circle(img, center, radius, (255, 255, 255), -1)

    # 添加标签内容（模拟文字）
    cv2.putText(img, "WINE", (center[0] - 50, center[1] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "LABEL", (center[0] - 60, center[1] + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 保存正常图像
    cv2.imwrite("assets/test_normal.jpg", img)

    # 创建贴重的图像
    img_duplicate = img.copy()
    # 添加第二个偏移的圆形
    center2 = (420, 290)
    cv2.circle(img_duplicate, center2, radius, (200, 200, 200), -1)
    cv2.putText(img_duplicate, "WINE", (center2[0] - 50, center2[1] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
    cv2.imwrite("assets/test_duplicate.jpg", img_duplicate)

    # 创建旋转的图像
    img_rotated = np.zeros((600, 800, 3), dtype=np.uint8)
    noise = np.random.randint(0, 50, img_rotated.shape, dtype=np.uint8)
    img_rotated = cv2.add(img_rotated, noise)

    # 创建旋转的标签
    rotated_center = (400, 300)
    cv2.circle(img_rotated, rotated_center, radius, (255, 255, 255), -1)

    # 添加旋转的文字（通过创建旋转的图像并合并）
    text_img = np.zeros((200, 200, 3), dtype=np.uint8)
    text_img[:] = (255, 255, 255)
    cv2.putText(text_img, "WINE", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(text_img, "LABEL", (30, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 旋转文本图像
    rotation_matrix = cv2.getRotationMatrix2D((100, 100), 45, 1.0)
    rotated_text = cv2.warpAffine(text_img, rotation_matrix, (200, 200))

    # 创建圆形mask
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 100, 255, -1)
    mask = cv2.merge([mask, mask, mask])

    # 应用mask
    rotated_text = cv2.bitwise_and(rotated_text, mask)

    # 将旋转的标签放置到图像中
    y1, y2 = rotated_center[1] - 100, rotated_center[1] + 100
    x1, x2 = rotated_center[0] - 100, rotated_center[0] + 100
    img_rotated[y1:y2, x1:x2] = rotated_text

    cv2.imwrite("assets/test_rotated.jpg", img_rotated)

    print("测试图像已生成：")
    print("  - assets/test_normal.jpg (正常标签)")
    print("  - assets/test_duplicate.jpg (贴重标签)")
    print("  - assets/test_rotated.jpg (旋转标签)")


def example_1_single_detection():
    """
    示例1：单张图像检测
    """
    print("\n" + "="*60)
    print("示例1：单张图像检测")
    print("="*60)

    # 创建检测器
    detector = get_detector("config/detector_config.json")

    # 先训练标准样本
    print("\n步骤1：使用正常图像训练标准样本...")
    result = detector.detect("assets/test_normal.jpg", train_mode=True)
    print(f"训练结果: {result['final_result']}")

    # 检测正常图像
    print("\n步骤2：检测正常图像...")
    result = detector.detect("assets/test_normal.jpg")
    print(f"检测结果: {result['final_result']}")

    # 检测贴重图像
    print("\n步骤3：检测贴重图像...")
    result = detector.detect("assets/test_duplicate.jpg")
    print(f"检测结果: {result['final_result']}")

    # 检测旋转图像
    print("\n步骤4：检测旋转图像...")
    result = detector.detect("assets/test_rotated.jpg")
    print(f"检测结果: {result['final_result']}")


def example_2_batch_detection():
    """
    示例2：批量检测
    """
    print("\n" + "="*60)
    print("示例2：批量检测")
    print("="*60)

    # 创建检测器
    detector = get_detector("config/detector_config.json")

    # 批量检测
    print("\n开始批量检测...")
    results = detector.batch_detect("assets", train_image="test_normal.jpg")

    # 统计结果
    total = len(results)
    ok_count = sum(1 for r in results if r.get('final_result', {}).get('overall_status') == 'OK')
    ng_count = sum(1 for r in results if r.get('final_result', {}).get('overall_status') == 'NG')

    print(f"\n批量检测完成:")
    print(f"  总数: {total}")
    print(f"  OK: {ok_count}")
    print(f"  NG: {ng_count}")

    # 显示详细结果
    for result in results:
        image_name = result.get('image_name', 'unknown')
        final_result = result.get('final_result', {})
        print(f"\n  {image_name}: {final_result.get('overall_status', 'unknown')}")
        if final_result.get('total_defects', 0) > 0:
            for defect in final_result.get('defects', []):
                print(f"    - {defect['type']}: confidence={defect['confidence']:.2f}")


def example_3_visualization():
    """
    示例3：可视化处理流程
    """
    print("\n" + "="*60)
    print("示例3：可视化处理流程")
    print("="*60)

    # 创建检测器
    detector = get_detector("config/detector_config.json")

    # 可视化处理流程
    print("\n生成可视化流程图...")
    vis = detector.visualize_full_process(
        "assets/test_normal.jpg",
        save_path="assets/visualize_process.jpg"
    )
    print("可视化图像已保存到: assets/visualize_process.jpg")


def example_4_custom_config():
    """
    示例4：自定义配置
    """
    print("\n" + "="*60)
    print("示例4：自定义配置")
    print("="*60)

    # 创建检测器
    detector = get_detector("config/detector_config.json")

    # 获取当前配置
    config = detector.get_config()
    print("\n当前配置:")
    print(f"  角度容差: {config['defect_detector']['angle_tolerance']}°")
    print(f"  边缘密度阈值: {config['defect_detector']['edge_density_threshold']}")

    # 更新配置
    print("\n更新配置...")
    new_config = {
        'defect_detector': {
            'angle_tolerance': 10.0,  # 更严格的角度要求
            'edge_density_threshold': 1.2  # 更敏感的贴重检测
        }
    }
    detector.update_config(new_config)

    # 验证配置已更新
    config = detector.get_config()
    print(f"更新后的角度容差: {config['defect_detector']['angle_tolerance']}°")
    print(f"更新后的边缘密度阈值: {config['defect_detector']['edge_density_threshold']}")


def main():
    """
    主函数
    """
    print("="*60)
    print("酒瓶标签检测算法示例")
    print("="*60)

    # 确保输出目录存在
    os.makedirs("assets/output", exist_ok=True)

    # 生成测试图像
    print("\n生成测试图像...")
    create_test_image()

    # 运行示例
    try:
        example_1_single_detection()
        example_2_batch_detection()
        example_3_visualization()
        example_4_custom_config()

        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        print("\n生成的文件:")
        print("  - assets/test_*.jpg (测试图像)")
        print("  - assets/output/* (检测结果)")
        print("  - assets/visualize_process.jpg (可视化流程)")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
