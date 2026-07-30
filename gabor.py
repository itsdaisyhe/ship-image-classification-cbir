import cv2
import numpy as np

class EdgeFeature:
    def __init__(self, low_threshold=30, high_threshold=100, bin_num=36):
        """
        初始化边缘特征提取器
        
        参数:
            low_threshold: Canny边缘检测的低阈值
            high_threshold: Canny边缘检测的高阈值
            bin_num: 方向直方图的区间数量
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.bin_num = bin_num
    
    def extract(self, image_path):
        """
        从图像中提取边缘特征
        
        参数:
            image_path: 图像文件路径
            
        返回:
            numpy数组，边缘方向直方图特征向量
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            
            # 边缘检测
            edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
            
            # 计算梯度
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            
            # 计算梯度方向
            magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
            
            # 创建方向直方图
            hist = np.zeros(self.bin_num, dtype=np.float32)
            bin_angle = 360.0 / self.bin_num
            
            # 只考虑边缘点的方向
            for i in range(edges.shape[0]):
                for j in range(edges.shape[1]):
                    if edges[i, j] > 0:  # 如果是边缘点
                        bin_idx = int(angle[i, j] / bin_angle) % self.bin_num
                        hist[bin_idx] += magnitude[i, j]
            
            # 归一化直方图
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            
            return hist
            
        except Exception as e:
            print(f"边缘特征提取错误: {e}")
            return None 