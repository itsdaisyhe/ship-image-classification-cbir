import cv2
import numpy as np
from skimage.feature import hog

class HOGFeature:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), target_size=(128, 128)):
        """
        初始化HOG特征提取器
        
        参数:
            orientations: 方向区间数量
            pixels_per_cell: 每个单元格的像素尺寸
            cells_per_block: 每个块的单元格数量
            target_size: 处理前将图像调整到的目标尺寸
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.target_size = target_size
    
    def extract(self, image_path):
        """
        从图像中提取HOG特征
        
        参数:
            image_path: 图像文件路径
            
        返回:
            numpy数组，HOG特征向量
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
            
            # 计算HOG特征
            hog_features = hog(
                resized, 
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True
            )
            
            # 转换为float32类型
            hog_features = hog_features.astype(np.float32)
            
            return hog_features
            
        except Exception as e:
            print(f"HOG特征提取错误: {e}")
            return None 