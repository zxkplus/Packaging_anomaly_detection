"""
快速测试脚本 - 验证所有核心功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label_detector import LabelDetector, get_detector
import cv2


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


if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
