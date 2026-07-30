import cv2
import numpy as np

class RGBHistogram:
    def __init__(self, bins=(8, 8, 8)):
        """
        初始化RGB颜色直方图特征提取器
        
        参数:
            bins: 三元组，分别表示R、G、B通道的直方图区间数量
        """
        self.bins = bins
        self.hist_size = np.prod(bins)  # 直方图总长度
    
    def extract(self, image_path):
        """
        从图像中提取RGB颜色直方图特征
        
        参数:
            image_path: 图像文件路径
            
        返回:
            numpy数组，归一化后的RGB颜色直方图特征向量
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None
            
            # 转换为RGB (OpenCV默认读取为BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 计算RGB三通道的直方图
            hist = cv2.calcHist(
                [image], 
                [0, 1, 2], 
                None, 
                self.bins, 
                [0, 256, 0, 256, 0, 256]
            )
            
            # 归一化直方图
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            
            return hist
            
        except Exception as e:
            print(f"RGB直方图提取错误: {e}")
            return None 