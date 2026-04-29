import os
import cv2
import numpy as np
from skimage.feature import hog
import joblib


def extract_features(image_path):
    """
    从图像中提取多种特征用于分类
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return None

    features = []

    # 1. HOG 特征 (Histogram of Oriented Gradients)
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    features.extend(hog_features)

    # 2. 边缘密度特征
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    features.append(edge_density)

    # 3. 纹理特征
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    texture_var = cv2.meanStdDev(img - blurred)[1].flatten()[0]
    features.append(texture_var)

    # 4. 区域特征 (灰度直方图)
    hist, _ = np.histogram(img.flatten(), bins=32, range=[0, 256])
    features.extend(hist / (img.shape[0] * img.shape[1]))

    # 5. 形状特征
    _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if area > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            features.append(circularity)
        else:
            features.append(0)
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox_area = w * h
        ratio_area_bbox = area / bbox_area if bbox_area > 0 else 0
        features.append(ratio_area_bbox)
    else:
        features.extend([0, 0])

    return np.array(features)


def predict_single_image(image_path, model, scaler):
    """
    对单张图像进行预测
    """
    features = extract_features(image_path)
    if features is None:
        return None, None
    
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return prediction, probabilities


def predict_batch_images(folder_path, model, scaler):
    """
    批量预测文件夹中的图像
    """
    results = []
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        prediction, probabilities = predict_single_image(img_path, model, scaler)
        
        if prediction is not None:
            result = {
                'image': img_file,
                'prediction': 'OK' if prediction == 0 else 'ERROR',
                'confidence_ok': probabilities[0],
                'confidence_error': probabilities[1]
            }
            results.append(result)
    
    return results


def main():
    # 加载训练好的模型和标准化器
    model_path = "label_detection_svm_model.pkl"
    scaler_path = "label_detection_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model files not found. Please run train_label_detector.py first.")
        return
    
    print("Loading trained model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print("Model loaded successfully!")
    
    # 预测测试
    test_folder = input("Enter the path to the test folder (or press Enter to skip): ").strip()
    
    if test_folder and os.path.exists(test_folder):
        print(f"\nProcessing images in {test_folder}...")
        results = predict_batch_images(test_folder, model, scaler)
        
        print("\nPrediction Results:")
        print("-" * 80)
        for result in results:
            print(f"{result['image']:<40} | "
                  f"Prediction: {result['prediction']:<5} | "
                  f"OK: {result['confidence_ok']:.3f} | "
                  f"ERROR: {result['confidence_error']:.3f}")
        
        # 统计结果
        ok_count = sum(1 for r in results if r['prediction'] == 'OK')
        error_count = len(results) - ok_count
        print(f"\nSummary: {ok_count} OK, {error_count} ERROR out of {len(results)} images")
    
    else:
        # 测试单张图片
        print("\nTesting single image prediction...")
        sample_ok_path = "assets/output_ok/" + os.listdir("assets/output_ok")[0]
        prediction, probabilities = predict_single_image(sample_ok_path, model, scaler)
        
        if prediction is not None:
            print(f"Sample: {os.path.basename(sample_ok_path)}")
            print(f"Prediction: {'OK' if prediction == 0 else 'ERROR'}")
            print(f"Probabilities - OK: {probabilities[0]:.3f}, ERROR: {probabilities[1]:.3f}")


if __name__ == "__main__":
    main()