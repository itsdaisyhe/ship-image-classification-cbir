import cv2
import numpy as np

class GaborFeature:
    def __init__(self, scales=5, orientations=8, target_size=(128, 128)):
        """
        初始化Gabor纹理特征提取器
        
        参数:
            scales: 尺度数量
            orientations: 方向数量
            target_size: 处理前将图像调整到的目标尺寸
        """
        self.scales = scales
        self.orientations = orientations
        self.target_size = target_size
        self.feature_size = self.scales * self.orientations * 2  # 每个滤波器提取均值和方差两个特征
    
    def extract(self, image_path):
        """
        从图像中提取Gabor纹理特征
        
        参数:
            image_path: 图像文件路径
            
        返回:
            numpy数组，Gabor纹理特征向量
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
            
            # 归一化到[0,1]
            resized = resized.astype(np.float32) / 255.0
            
            # 创建Gabor滤波器组
            kernels = self._build_gabor_filters()
            
            # 提取特征
            gabor_features = []
            for kernel in kernels:
                # 应用Gabor滤波器
                filtered = cv2.filter2D(resized, cv2.CV_32F, kernel)
                
                # 计算均值和方差
                mean = np.mean(filtered)
                std = np.std(filtered)
                
                # 添加到特征向量
                gabor_features.extend([mean, std])
            
            # 转换为numpy数组
            feature_vector = np.array(gabor_features, dtype=np.float32)
            
            return feature_vector
            
        except Exception as e:
            print(f"Gabor特征提取错误: {e}")
            return None
    
    def _build_gabor_filters(self):
        """
        构建Gabor滤波器组
        
        返回:
            Gabor滤波器列表
        """
        filters = []
        ksize = (31, 31)  # 滤波器大小
        sigma = 4.0       # 标准差
        
        for theta in np.arange(0, np.pi, np.pi / self.orientations):
            for lambd in np.arange(0, np.pi, np.pi / self.scales):
                if lambd == 0:  # 避免除以零
                    continue
                    
                gamma = 0.5     # 空间宽高比
                psi = 0         # 相位偏移
                
                kernel = cv2.getGaborKernel(
                    ksize, sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
                )
                
                # 归一化
                kernel /= 1.5 * kernel.sum()
                filters.append(kernel)
        
        return filters 