import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from skimage.feature import hog
from skimage import exposure


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
    # HOG 特征非常适合检测形状和纹理模式
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    features.extend(hog_features)

    # 2. 边缘密度特征
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    features.append(edge_density)

    # 3. 纹理特征 (使用LBP的简化版本)
    # 计算局部方差作为纹理特征
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    texture_var = cv2.meanStdDev(img - blurred)[1].flatten()[0]  # 标准差作为纹理特征
    features.append(texture_var)

    # 4. 区域特征 (灰度直方图)
    hist, _ = np.histogram(img.flatten(), bins=32, range=[0, 256])
    features.extend(hist / (img.shape[0] * img.shape[1]))  # 归一化

    # 5. 形状特征 (轮廓复杂度)
    _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 计算最大轮廓的周长和面积比
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # 形状复杂度指标
        if area > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            features.append(circularity)
        else:
            features.append(0)
        
        # 轮廓面积与边界框面积比
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox_area = w * h
        ratio_area_bbox = area / bbox_area if bbox_area > 0 else 0
        features.append(ratio_area_bbox)
    else:
        features.extend([0, 0])

    return np.array(features)


def load_dataset(ok_folder, error_folder):
    """
    加载数据集并提取特征
    """
    X = []  # 特征
    y = []  # 标签 (0 for OK, 1 for ERROR)

    print(f"Loading OK samples from {ok_folder}...")
    ok_files = [os.path.join(ok_folder, f) for f in os.listdir(ok_folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for file_path in ok_files:
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(0)  # OK class

    print(f"Loaded {len(ok_files)} OK samples")

    print(f"Loading ERROR samples from {error_folder}...")
    error_files = [os.path.join(error_folder, f) for f in os.listdir(error_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for file_path in error_files:
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(1)  # ERROR class

    print(f"Loaded {len(error_files)} ERROR samples")

    return np.array(X), np.array(y)


def train_svm_classifier(X, y):
    """
    训练SVM分类器
    """
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练SVM分类器
    print("Training SVM classifier...")
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    svm_classifier.fit(X_train_scaled, y_train)

    # 预测和评估
    y_pred = svm_classifier.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['OK', 'ERROR']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 计算准确率
    accuracy = svm_classifier.score(X_test_scaled, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    return svm_classifier, scaler


def main():
    """
    主函数
    """
    # 定义数据路径
    ok_folder = "assets/output_ok"
    error_folder = "assets/output_error"

    # 检查文件夹是否存在
    if not os.path.exists(ok_folder) or not os.path.exists(error_folder):
        print(f"Error: One of the folders does not exist.")
        print(f"Looking for: {ok_folder} and {error_folder}")
        return

    # 加载数据集
    X, y = load_dataset(ok_folder, error_folder)

    if len(X) == 0:
        print("No valid images found in the specified folders.")
        return

    print(f"Total samples: {len(X)}, Feature dimension: {X.shape[1]}")

    # 训练SVM分类器
    classifier, scaler = train_svm_classifier(X, y)

    # 保存模型和预处理器
    model_path = "label_detection_svm_model.pkl"
    scaler_path = "label_detection_scaler.pkl"
    
    joblib.dump(classifier, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    # 测试单张图片预测功能
    print("\nTesting prediction on a sample from OK folder...")
    sample_ok_path = os.path.join(ok_folder, os.listdir(ok_folder)[0])
    sample_features = extract_features(sample_ok_path)
    
    if sample_features is not None:
        sample_features_scaled = scaler.transform([sample_features])
        prediction = classifier.predict(sample_features_scaled)[0]
        probability = classifier.predict_proba(sample_features_scaled)[0]
        
        print(f"Sample: {os.path.basename(sample_ok_path)}")
        print(f"Prediction: {'OK' if prediction == 0 else 'ERROR'}")
        print(f"Probabilities - OK: {probability[0]:.3f}, ERROR: {probability[1]:.3f}")


if __name__ == "__main__":
    main()