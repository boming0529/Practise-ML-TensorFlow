import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# 設置隨機種子以確保可重現性
np.random.seed(42)
tf.random.set_seed(42)

# 1. 數據準備
def prepare_data(labeled_data, unlabeled_data, batch_size=150):
    """將數據分為標記和未標記的小批量，每批包含相等數量的標記和未標記數據"""
    labeled_images, labeled_values = labeled_data
    unlabeled_images = unlabeled_data
    
    # 隨機選擇標記和未標記數據
    n_labeled = len(labeled_images)
    n_unlabeled = len(unlabeled_images)
    
    indices_l = np.random.permutation(n_labeled)[:batch_size // 2]
    indices_u = np.random.permutation(n_unlabeled)[:batch_size // 2]
    
    batch_labeled_images = labeled_images[indices_l]
    batch_labeled_values = labeled_values[indices_l]
    batch_unlabeled_images = unlabeled_images[indices_u]
    
    return batch_labeled_images, batch_labeled_values, batch_unlabeled_images

# 2. 特徵提取
def extract_features(images, model):
    """使用預訓練的 ResNet-18 提取 512 維特徵"""
    # 注意：TensorFlow 沒有直接的 ResNet-18，改用 ResNet50 並調整輸出
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation=None)(x)  # 調整為 512 維
    feature_model = Model(inputs=base_model.input, outputs=x)
    
    # 預處理圖像並提取特徵
    images = tf.image.resize(images, (224, 224))  # 假設輸入圖像需要調整大小
    features = feature_model.predict(images, batch_size=32)
    return features

# 3. t-SNE 降維與偽標籤生成
def generate_pseudo_labels(labeled_features, labeled_values, unlabeled_features, n_trials=5):
    """使用 t-SNE 降維並生成偽標籤"""
    all_features = np.vstack([labeled_features, unlabeled_features])
    n_labeled = len(labeled_features)
    
    best_pseudo_labels = None
    best_kl_div = float('inf')
    best_rmse = float('inf')
    
    for _ in range(n_trials):  # 多次 t-SNE 試驗以應對隨機性
        # t-SNE 降維到 2D
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(all_features)
        
        # 分離標記和未標記的嵌入
        embedded_labeled = embedded[:n_labeled]
        embedded_unlabeled = embedded[n_labeled:]
        
        # 計算平均最近鄰距離
        distances = cdist(embedded, embedded, metric='euclidean')
        nn_distances = np.min(distances + np.eye(len(distances)) * 1e10, axis=1)
        mean_nn_distance = np.mean(nn_distances)
        
        # 計算標記點對的 KL 散度和 RMSE
        labeled_pairs = cdist(embedded_labeled, embedded_labeled, metric='euclidean')
        valid_pairs = labeled_pairs < mean_nn_distance
        if np.sum(valid_pairs) > 0:
            kl_div = tsne.kl_divergence_
            rmse = mean_squared_error(
                labeled_values[valid_pairs[:, 0]],
                labeled_values[valid_pairs[:, 1]],
                squared=False
            )
        else:
            kl_div = float('inf')
            rmse = float('inf')
        
        # 更新最佳試驗
        if kl_div + rmse < best_kl_div + best_rmse:
            best_kl_div = kl_div
            best_rmse = rmse
            
            # 為未標記數據生成偽標籤
            pseudo_labels = []
            for i in range(len(embedded_unlabeled)):
                distances_to_labeled = cdist(
                    embedded_unlabeled[i:i+1], embedded_labeled, metric='euclidean'
                )[0]
                valid_indices = distances_to_labeled < mean_nn_distance
                if np.sum(valid_indices) > 0:
                    weights = 1 / (distances_to_labeled[valid_indices] + 1e-10)
                    pseudo_label = np.average(
                        labeled_values[valid_indices], weights=weights
                    )
                else:
                    pseudo_label = np.nan  # 如果沒有鄰居，標記為無效
                pseudo_labels.append(pseudo_label)
            
            best_pseudo_labels = np.array(pseudo_labels)
    
    return best_pseudo_labels

# 4. 數據平衡
def balance_dataset(images, values, max_samples=5000):
    """使用隨機過採樣平衡數據集，限制最大樣本數"""
    # 將值分為分位數
    quantiles = pd.qcut(values, q=10, duplicates='drop')
    ros = RandomOverSampler(random_state=42)
    
    # 過採樣
    indices = np.arange(len(values)).reshape(-1, 1)
    indices_resampled, _ = ros.fit_resample(indices, quantiles)
    indices_resampled = indices_resampled.flatten()
    
    # 限制樣本數
    if len(indices_resampled) > max_samples:
        indices_resampled = np.random.choice(indices_resampled, max_samples, replace=False)
    
    return images[indices_resampled], values[indices_resampled]

# 5. 回歸模型
def build_regression_model():
    """構建 EfficientNet-b0 回歸模型"""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation=None)(x)  # 回歸輸出
    model = Model(inputs=base_model.input, outputs=output)
    
    # 凍結基模型層
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 6. 主流程
def semi_supervised_learning(labeled_images, labeled_values, unlabeled_images, epochs=10):
    """半監督學習主流程"""
    # 初始化回歸模型
    regression_model = build_regression_model()
    
    # 迭代處理批次
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # 準備批次數據
        batch_labeled_images, batch_labeled_values, batch_unlabeled_images = prepare_data(
            (labeled_images, labeled_values), unlabeled_images
        )
        
        # 提取特徵
        labeled_features = extract_features(batch_labeled_images, ResNet50)
        unlabeled_features = extract_features(batch_unlabeled_images, ResNet50)
        
        # 生成偽標籤
        pseudo_labels = generate_pseudo_labels(
            labeled_features, batch_labeled_values, unlabeled_features
        )
        
        # 過濾有效偽標籤
        valid_mask = ~np.isnan(pseudo_labels)
        pseudo_labeled_images = batch_unlabeled_images[valid_mask]
        pseudo_labels = pseudo_labels[valid_mask]
        
        # 合併標記和偽標籤數據
        all_images = np.vstack([batch_labeled_images, pseudo_labeled_images])
        all_values = np.hstack([batch_labeled_values, pseudo_labels])
        
        # 平衡數據集
        balanced_images, balanced_values = balance_dataset(all_images, all_values)
        
        # 訓練回歸模型
        regression_model.fit(
            balanced_images, balanced_values,
            batch_size=32, epochs=1, verbose=1
        )
    
    return regression_model

# 示例用法
if __name__ == "__main__":
    # 模擬數據（需要替換為實際數據）
    labeled_images = np.random.rand(100, 224, 224, 3)  # 100 張標記圖像
    labeled_values = np.random.rand(100)  # 100 個標記值
    unlabeled_images = np.random.rand(1000, 224, 224, 3)  # 1000 張未標記圖像
    
    # 運行半監督學習
    model = semi_supervised_learning(labeled_images, labeled_values, unlabeled_images)
    model.save("semi_supervised_model.h5")