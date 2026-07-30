import cv2
import numpy as np
from skimage.feature import daisy

class DaisyFeature:
    def __init__(self, step=32, radius=58, rings=2, histograms=8, orientations=8, target_size=(256, 256)):
        """
        初始化Daisy局部特征提取器
        
        参数:
            step: 采样步长，控制特征的密度
            radius: 半径，控制描述符的大小
            rings: 环的数量
            histograms: 每个环上的直方图数量
            orientations: 方向区间数量
            target_size: 处理前将图像调整到的目标尺寸
        """
        self.step = step
        self.radius = radius
        self.rings = rings
        self.histograms = histograms
        self.orientations = orientations
        self.target_size = target_size
    
    def extract(self, image_path):
        """
        从图像中提取Daisy局部特征
        
        参数:
            image_path: 图像文件路径
            
        返回:
            numpy数组，Daisy特征向量
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 调整图像大小
            resized = cv2.resize(gray, self.target_size)
            
            # 计算Daisy特征
            descs, descs_img = daisy(
                resized,
                step=self.step,
                radius=self.radius,
                rings=self.rings,
                histograms=self.histograms,
                orientations=self.orientations,
                visualize=True
            )
            
            # 将描述符展平
            num_features = descs.shape[0] * descs.shape[1]
            feature_dim = descs.shape[2]
            
            if num_features > 0:
                # 只取部分特征点，避免维度过大
                max_points = 64  # 最大特征点数量
                if num_features > max_points:
                    # 均匀采样
                    indices = np.linspace(0, num_features - 1, max_points, dtype=int)
                    features = descs.reshape(num_features, feature_dim)[indices]
                else:
                    features = descs.reshape(num_features, feature_dim)
                
                # 计算所有特征点的均值作为全局特征
                global_feature = np.mean(features, axis=0)
                
                # 确保特征维度固定
                feature_vector = global_feature.astype(np.float32)
                
                return feature_vector
            else:
                print(f"未检测到特征点: {image_path}")
                return np.zeros(self.histograms * self.rings * self.orientations + 1, dtype=np.float32)
            
        except Exception as e:
            print(f"Daisy特征提取错误: {e}")
            return None 