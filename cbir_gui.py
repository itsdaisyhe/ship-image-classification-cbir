import sys
import os
import numpy as np
import sqlite3
import time
import platform
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg')
# 设置matplotlib中文字体支持
system = platform.system()
if system == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
elif system == 'Darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial']
else:  # Linux
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QComboBox, 
                            QFileDialog, QProgressBar, QGroupBox, QGridLayout, 
                            QSpinBox, QScrollArea, QSplitter, QFrame, 
                            QMessageBox, QLineEdit, QTabWidget, QStyle)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QRect
import cv2
from PIL import Image, ImageQt

# 导入特征提取模块
try:
    from feature_extractors.color import RGBHistogram
    from feature_extractors.edge import EdgeFeature
    from feature_extractors.hog import HOGFeature
    from feature_extractors.gabor import GaborFeature
    from feature_extractors.daisy import DaisyFeature
    from feature_extractors.vggnet import VGG19Feature
    from feature_extractors.resnet import ResNet152Feature
except ImportError:
    print("特征提取模块导入失败，请确保这些模块存在")

# 特征可视化类
class FeatureVisualization:
    @staticmethod
    def visualize_rgb_histogram(image_path, figure):
        """可视化RGB颜色直方图"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 清除图形
            figure.clear()
            
            # 创建三个子图
            axs = figure.subplots(3, 1, sharex=True)
            
            # 颜色通道名称和对应颜色
            channels = ['red (R)', 'green (G)', 'blue (B)']
            colors = ['r', 'g', 'b']
            
            # 绘制每个通道的直方图
            for i, (channel, color) in enumerate(zip(channels, colors)):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                axs[i].plot(hist, color=color)
                axs[i].set_ylabel(channel)
                axs[i].grid(True, alpha=0.3)
            
            axs[2].set_xlabel('像素值', fontsize=18)
            figure.suptitle('RGB 颜色直方图', fontsize=18)
            figure.tight_layout()
            
            return True
        except Exception as e:
            print(f"RGB直方图可视化错误: {e}")
            return False
    
    @staticmethod
    def visualize_edge_feature(image_path, figure):
        """可视化边缘特征"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 边缘检测
            edges = cv2.Canny(blurred, 30, 100)
            
            # 计算梯度
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            
            # 计算梯度方向和幅值
            magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
            
            # 创建伪彩色图像来表示方向
            hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            # 将角度映射到色调（0-179）
            hsv[..., 0] = angle * 179.0 / 360.0
            # 将幅值归一化为饱和度
            hsv[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # 值设为255
            hsv[..., 2] = 255
            
            # HSV转BGR
            direction_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # 将BGR转为RGB用于matplotlib
            direction_map = cv2.cvtColor(direction_map, cv2.COLOR_BGR2RGB)
            
            # 清除图形
            figure.clear()
            
            # 创建子图
            axs = figure.subplots(1, 3)
            
            # 显示原图
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0].set_title('原图', fontsize=18)
            axs[0].axis('off')
            
            # 显示边缘图
            axs[1].imshow(edges, cmap='gray')
            axs[1].set_title('edge', fontsize=18)
            axs[1].axis('off')
            
            # 显示方向图
            axs[2].imshow(direction_map)
            axs[2].set_title('梯度方向', fontsize=18)
            axs[2].axis('off')
            
            figure.suptitle('边缘特征可视化', fontsize=18)
            figure.tight_layout()
            
            return True
        except Exception as e:
            print(f"边缘特征可视化错误: {e}")
            return False
    
    @staticmethod
    def visualize_hog_feature(image_path, figure):
        """可视化HOG特征"""
        try:
            from skimage import feature
            from skimage import exposure
            
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 调整大小
            resized = cv2.resize(gray, (128, 128))
            
            # 计算HOG特征
            fd, hog_image = feature.hog(
                resized, 
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=True
            )
            
            # 增强对比度
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            # 清除图形
            figure.clear()
            
            # 创建子图
            axs = figure.subplots(1, 2)
            
            # 显示原图
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0].set_title('原图 ', fontsize=18)
            axs[0].axis('off')
            
            # 显示HOG特征
            axs[1].imshow(hog_image_rescaled, cmap='gray')
            axs[1].set_title('HOG 特征')
            axs[1].axis('off')
            
            figure.suptitle('HOG特征可视化', fontsize=18)
            figure.tight_layout()
            
            return True
        except Exception as e:
            print(f"HOG特征可视化错误: {e}")
            return False
    
    @staticmethod
    def visualize_gabor_feature(image_path, figure):
        """可视化Gabor纹理特征"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 调整大小
            resized = cv2.resize(gray, (128, 128))
            
            # 创建Gabor滤波器
            filters = []
            ksize = 31
            sigma = 4.0
            
            # 生成不同方向和尺度的滤波器
            for theta in np.arange(0, np.pi, np.pi/4):  # 4个方向
                for lambd in np.arange(5, 15, 5):  # 2个尺度
                    gamma = 0.5
                    kernel = cv2.getGaborKernel(
                        (ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F
                    )
                    kernel /= 1.5 * kernel.sum()
                    filters.append(kernel)
            
            # 应用滤波器
            filtered_images = []
            for kernel in filters[:8]:  # 只显示前8个滤波结果
                filtered = cv2.filter2D(resized, cv2.CV_8UC1, kernel)
                filtered_images.append(filtered)
            
            # 清除图形
            figure.clear()

            # 增加图形的大小
            figure.set_dpi(100)
            figure.set_size_inches(12, 8)  # 将图形大小设置为16x10英寸

            
            
            # 创建网格子图
            fig, axs = plt.subplots(3, 3, figsize=(12, 8), subplot_kw={'xticks': [], 'yticks': []})
            # 调整子图间距，确保标题和图像之间有足够的空间
            
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            
            
            # 显示原图
            axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title('原图', fontsize=12, pad=10)
            axs[0, 0].axis('off')
            
            # 显示滤波结果
            for i, filtered in enumerate(filtered_images):
                row = (i + 1) // 3
                col = (i + 1) % 3
                axs[row, col].imshow(filtered, cmap='gray')
                axs[row, col].set_title(f'filter {i+1}', fontsize=10, pad=5)
                axs[row, col].axis('off')
            
           
            
            figure.suptitle('Gabor特征可视化', fontsize=16, y=0.95)
            figure.tight_layout()
            # 显示图形
            plt.show()
            
            return True
        except Exception as e:
            print(f"Gabor特征可视化错误: {e}")
            return False
    
    @staticmethod
    def visualize_daisy_feature(image_path, figure):
        """可视化Daisy局部特征"""
        try:
            from skimage.feature import daisy
            from skimage import color, img_as_float
            
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 调整大小
            resized = cv2.resize(gray, (128, 128))
            img_resized = cv2.resize(img, (128, 128))
            
            # 计算Daisy特征
            descs, descs_img = daisy(
                img_as_float(resized),
                step=16,
                radius=15,
                rings=3,
                histograms=8,
                orientations=8,
                visualize=True
            )
            
            # 清除图形
            figure.clear()
            
            # 创建子图
            axs = figure.subplots(1, 2)
            
            # 显示原图
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0].set_title('原图 ', fontsize=18)
            axs[0].axis('off')
            
            # 显示Daisy特征
            axs[1].imshow(descs_img, cmap='hot')
            axs[1].set_title('Daisy特征', fontsize=18)
            axs[1].axis('off')
            
            figure.suptitle('Daisy特征可视化', fontsize=18)
            figure.tight_layout()
            
            return True
        except Exception as e:
            print(f"Daisy特征可视化错误: {e}")
            return False
    
    @staticmethod
    def visualize_deep_feature(image_path, figure, model_name):
        """可视化深度特征 (VGG19或ResNet152)"""
        try:
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
            from torchvision.models import vgg19, resnet152, VGG19_Weights, ResNet152_Weights
            import torch.nn.functional as F
            import cv2
            
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # 载入模型
            if model_name == "VGG19_Feature":
                model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:20]  # 使用前20层特征
            else:  # ResNet152
                model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
                # 创建一个新模型，只使用到layer3
                layers = list(model.children())[:-2]
                model = torch.nn.Sequential(*layers)
                
            model.eval()
            
            # 准备图像
            pil_img = Image.open(image_path).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(pil_img)
            input_batch = input_tensor.unsqueeze(0)
            
            # 提取特征
            with torch.no_grad():
                features = model(input_batch)
            
            # 创建特征激活热力图
            features = features.squeeze().sum(dim=0).numpy()
            features_norm = (features - features.min()) / (features.max() - features.min())
            
            # 调整热力图大小与原图匹配
            heatmap = cv2.resize(features_norm, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # 合并热力图和原图
            superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            
            # 清除图形
            figure.clear()
            
            # 创建子图
            axs = figure.subplots(1, 3)
            
            # 显示原图
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0].set_title('原图', fontsize=18)
            axs[0].axis('off')
            
            # 显示热力图
            axs[1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
            axs[1].set_title('热力图')
            axs[1].axis('off')
            
            # 显示叠加图
            axs[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            axs[2].set_title('叠加图')
            axs[2].axis('off')
            
            figure.suptitle(f'{model_name.replace("_Feature", "")}深度特征可视化', fontsize=18)
            figure.tight_layout()
            
            return True
        except Exception as e:
            print(f"深度特征可视化错误: {e}")
            figure.clear()
            ax = figure.subplots(1, 1)
            ax.text(0.5, 0.5, f"无法可视化深度特征\n需要安装PyTorch\n错误: {e}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.axis('off')
            figure.tight_layout()
            return False
    
    @staticmethod
    def visualize_feature(image_path, feature_type, figure):
        """根据特征类型调用对应的可视化方法"""
        if feature_type == "RGB_Histogram":
            return FeatureVisualization.visualize_rgb_histogram(image_path, figure)
        elif feature_type == "Edge_Feature":
            return FeatureVisualization.visualize_edge_feature(image_path, figure)
        elif feature_type == "HOG_Feature":
            return FeatureVisualization.visualize_hog_feature(image_path, figure)
        elif feature_type == "Gabor_Feature":
            return FeatureVisualization.visualize_gabor_feature(image_path, figure)
        elif feature_type == "Daisy_Feature":
            return FeatureVisualization.visualize_daisy_feature(image_path, figure)
        elif feature_type == "VGG19_Feature" or feature_type == "ResNet152_Feature":
            return FeatureVisualization.visualize_deep_feature(image_path, figure, feature_type)
        else:
            figure.clear()
            ax = figure.subplots(1, 1)
            ax.text(0.5, 0.5, f"不支持的特征类型: {feature_type}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, color='#CC0000')  # 使用红色显示错误信息
            ax.axis('off')
            figure.tight_layout()
            return False

# 数据库管理类
class DBManager:
    def __init__(self, db_path='cbir_features.db'):
        self.db_path = db_path
        self.conn = None
        # 不在初始化时创建连接，而是在需要时创建
    
    def create_connection(self):
        """创建一个新的数据库连接"""
        try:
            conn = sqlite3.connect(self.db_path)
            print(f"成功连接到SQLite数据库: {self.db_path}")
            return conn
        except sqlite3.Error as e:
            print(f"数据库连接错误: {e}")
            return None
    
    def create_tables(self, conn=None):
        """创建必要的数据库表"""
        close_conn = False
        if conn is None:
            conn = self.create_connection()
            close_conn = True
            
        if conn is None:
            return False
            
        try:
            cursor = conn.cursor()
            
            # 创建images表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL,
                    category TEXT,
                    width INTEGER,
                    height INTEGER,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(path)
                )
            ''')
            
            # 创建features表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY,
                    image_id INTEGER NOT NULL,
                    feature_type TEXT NOT NULL,
                    feature_vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE,
                    UNIQUE(image_id, feature_type)
                )
            ''')
            
            conn.commit()
            print("数据库表创建成功")
            
            if close_conn:
                conn.close()
                
            return True
        except sqlite3.Error as e:
            print(f"创建表错误: {e}")
            if close_conn and conn:
                conn.close()
            return False
    
    def add_image(self, path, category=None, conn=None):
        """添加图像到数据库"""
        close_conn = False
        if conn is None:
            conn = self.create_connection()
            close_conn = True
            
        if conn is None:
            return None
            
        try:
            # 检查图像是否有效
            img = cv2.imread(path)
            if img is None:
                print(f"无效的图像文件: {path}")
                if close_conn:
                    conn.close()
                return None
            
            height, width = img.shape[:2]
            
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO images (path, category, width, height) VALUES (?, ?, ?, ?)",
                (path, category, width, height)
            )
            conn.commit()
            
            # 获取image_id
            cursor.execute("SELECT id FROM images WHERE path = ?", (path,))
            result = cursor.fetchone()
            
            if close_conn:
                conn.close()
                
            return result[0] if result else None
            
        except sqlite3.Error as e:
            print(f"添加图像错误: {e}")
            if close_conn and conn:
                conn.close()
            return None
    
    def add_feature(self, image_id, feature_type, feature_vector, conn=None):
        """添加特征向量到数据库"""
        close_conn = False
        if conn is None:
            conn = self.create_connection()
            close_conn = True
            
        if conn is None:
            return False
            
        try:
            # 将numpy数组转换为二进制
            feature_blob = feature_vector.tobytes()
            dimension = feature_vector.size
            
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO features (image_id, feature_type, feature_vector, dimension) VALUES (?, ?, ?, ?)",
                (image_id, feature_type, feature_blob, dimension)
            )
            conn.commit()
            
            if close_conn:
                conn.close()
                
            return True
        except sqlite3.Error as e:
            print(f"添加特征错误: {e}")
            if close_conn and conn:
                conn.close()
            return False
    
    def get_feature(self, image_id, feature_type, conn=None):
        """获取图像的特征向量"""
        close_conn = False
        if conn is None:
            conn = self.create_connection()
            close_conn = True
            
        if conn is None:
            return None
            
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT feature_vector, dimension FROM features WHERE image_id = ? AND feature_type = ?",
                (image_id, feature_type)
            )
            result = cursor.fetchone()
            
            if close_conn:
                conn.close()
                
            if result:
                feature_blob, dimension = result
                # 将二进制转换回numpy数组
                feature_vector = np.frombuffer(feature_blob, dtype=np.float32)
                return feature_vector
            return None
        except sqlite3.Error as e:
            print(f"获取特征错误: {e}")
            if close_conn and conn:
                conn.close()
            return None
    
    def get_all_features(self, feature_type, conn=None):
        """获取所有指定类型的特征向量"""
        close_conn = False
        if conn is None:
            conn = self.create_connection()
            close_conn = True
            
        if conn is None:
            return []
            
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT i.id, i.path, f.feature_vector, f.dimension FROM images i JOIN features f ON i.id = f.image_id WHERE f.feature_type = ?",
                (feature_type,)
            )
            results = cursor.fetchall()
            
            features_data = []
            for image_id, path, feature_blob, dimension in results:
                feature_vector = np.frombuffer(feature_blob, dtype=np.float32)
                features_data.append((image_id, path, feature_vector))
            
            if close_conn:
                conn.close()
                
            return features_data
        except sqlite3.Error as e:
            print(f"获取所有特征错误: {e}")
            if close_conn and conn:
                conn.close()
            return []
    
    def get_image_count(self, conn=None):
        """获取数据库中的图像数量"""
        close_conn = False
        if conn is None:
            conn = self.create_connection()
            close_conn = True
            
        if conn is None:
            return 0
            
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM images")
            result = cursor.fetchone()
            
            if close_conn:
                conn.close()
                
            return result[0] if result else 0
        except sqlite3.Error as e:
            print(f"获取图像数量错误: {e}")
            if close_conn and conn:
                conn.close()
            return 0
    
    def get_feature_types(self, conn=None):
        """获取数据库中所有的特征类型"""
        close_conn = False
        if conn is None:
            conn = self.create_connection()
            close_conn = True
            
        if conn is None:
            return []
            
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT feature_type FROM features")
            results = cursor.fetchall()
            
            if close_conn:
                conn.close()
                
            return [result[0] for result in results]
        except sqlite3.Error as e:
            print(f"获取特征类型错误: {e}")
            if close_conn and conn:
                conn.close()
            return []
    
    def get_image_category(self, image_id, conn=None):
        """获取图像的类别"""
        close_conn = False
        if conn is None:
            conn = self.create_connection()
            close_conn = True
            
        if conn is None:
            return "未知"
            
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT category FROM images WHERE id = ?", (image_id,))
            result = cursor.fetchone()
            
            if close_conn:
                conn.close()
                
            return result[0] if result else "未知"
        except sqlite3.Error as e:
            print(f"获取图像类别错误: {e}")
            if close_conn and conn:
                conn.close()
            return "未知"
    
    def close(self):
        """关闭数据库连接"""
        # 不再需要这个方法，因为我们在每次操作后都关闭连接
        pass

# 特征提取线程
class FeatureExtractionThread(QThread):
    feature_extracted = pyqtSignal(object, object)  # 发送特征向量和特征类型
    error_occurred = pyqtSignal(str)  # 发送错误消息
    progress_updated = pyqtSignal(int)  # 进度更新信号
    
    def __init__(self, image_path, feature_type, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.feature_type = feature_type
        self.stopped = False
        self.db_manager = DBManager()
        
    def run(self):
        try:
            # 检查路径是否存在
            if not os.path.exists(self.image_path):
                self.error_occurred.emit(f"路径不存在: {self.image_path}")
                return
                
            # 判断是目录还是单个文件
            if os.path.isdir(self.image_path):
                # 处理目录中的所有图像
                image_files = []
                valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                
                # 递归遍历目录
                for root, _, files in os.walk(self.image_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in valid_extensions):
                            image_files.append(os.path.join(root, file))
                
                if not image_files:
                    self.error_occurred.emit(f"在目录中找不到有效的图像文件: {self.image_path}")
                    return
                
                # 创建数据库连接
                conn = self.db_manager.create_connection()
                if not conn:
                    self.error_occurred.emit("无法连接到数据库")
                    return
                    
                # 确保表已创建
                self.db_manager.create_tables(conn)
                
                # 处理每个图像文件
                total_files = len(image_files)
                for i, img_path in enumerate(image_files):
                    if self.stopped:
                        break
                        
                    # 更新进度
                    progress = int((i / total_files) * 100)
                    self.progress_updated.emit(progress)
                    
                    try:
                        # 从文件路径推断类别
                        parent_dir = os.path.basename(os.path.dirname(img_path))
                        
                        # 添加图像到数据库
                        image_id = self.db_manager.add_image(img_path, parent_dir, conn)
                        if image_id is None:
                            print(f"跳过无效图像: {img_path}")
                            continue
                            
                        # 读取图像
                        image = cv2.imread(img_path)
                        if image is None:
                            print(f"无法读取图像: {img_path}")
                            continue
                            
                        # 提取特征
                        if self.feature_type == "RGB_Histogram":
                            extractor = RGBHistogram()
                        elif self.feature_type == "Edge_Feature":
                            extractor = EdgeFeature()
                        elif self.feature_type == "HOG_Feature":
                            extractor = HOGFeature()
                        elif self.feature_type == "Gabor_Feature":
                            extractor = GaborFeature()
                        elif self.feature_type == "Daisy_Feature":
                            extractor = DaisyFeature()
                        elif self.feature_type == "VGG19_Feature":
                            extractor = VGG19Feature()
                        elif self.feature_type == "ResNet152_Feature":
                            extractor = ResNet152Feature()
                        else:
                            raise ValueError(f"不支持的特征类型: {self.feature_type}")
                            
                        feature_vector = extractor.extract(img_path)
                        
                        # 保存特征到数据库
                        self.db_manager.add_feature(image_id, self.feature_type, feature_vector, conn)
                    except Exception as e:
                        print(f"处理图像 {img_path} 时出错: {str(e)}")
                
                # 关闭数据库连接
                conn.close()
                
                # 完成后更新进度条到100%
                self.progress_updated.emit(100)
                
                # 发送成功信号
                self.feature_extracted.emit(None, self.feature_type)
            else:
                # 处理单个图像文件
                # 读取图像
                image = cv2.imread(self.image_path)
                if image is None:
                    self.error_occurred.emit(f"无法读取图像: {self.image_path}")
                    return
                    
                # 根据特征类型提取相应的特征
                if self.feature_type == "RGB_Histogram":
                    extractor = RGBHistogram()
                elif self.feature_type == "Edge_Feature":
                    extractor = EdgeFeature()
                elif self.feature_type == "HOG_Feature":
                    extractor = HOGFeature()
                elif self.feature_type == "Gabor_Feature":
                    extractor = GaborFeature()
                elif self.feature_type == "Daisy_Feature":
                    extractor = DaisyFeature()
                elif self.feature_type == "VGG19_Feature":
                    extractor = VGG19Feature()
                elif self.feature_type == "ResNet152_Feature":
                    extractor = ResNet152Feature()
                else:
                    self.error_occurred.emit(f"不支持的特征类型: {self.feature_type}")
                    return
                    
                feature_vector = extractor.extract(self.image_path)
                
                # 发送提取的特征
                self.feature_extracted.emit(feature_vector, self.feature_type)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"特征提取错误: {str(e)}")
    
    def stop(self):
        self.stopped = True

# 检索线程
class ImageRetrievalThread(QThread):
    retrieval_complete = pyqtSignal(list)  # 检索结果列表
    error_occurred = pyqtSignal(str)
    
    def __init__(self, db_manager, query_image_path, feature_type, distance_metric, top_k):
        super().__init__()
        self.db_manager = db_manager
        self.query_image_path = query_image_path
        self.feature_type = feature_type
        self.distance_metric = distance_metric
        self.top_k = top_k

    def run(self):
        try:
            # 创建特征提取器
            extractor = self.create_feature_extractor()
            if not extractor:
                self.error_occurred.emit("不支持的特征类型")
                return
            
            print(f"开始处理查询图像: {self.query_image_path}")
            # 提取查询图像的特征
            try:
                query_feature = extractor.extract(self.query_image_path)
                if query_feature is None:
                    self.error_occurred.emit("无法提取查询图像的特征")
                    return
            except Exception as e:
                self.error_occurred.emit(f"特征提取错误: {str(e)}")
                return
                
            print(f"成功提取查询图像特征，维度: {query_feature.shape}")
            
            # 在线程中创建数据库连接
            conn = sqlite3.connect(self.db_manager.db_path)
            
            # 获取所有特征
            features_data = self.db_manager.get_all_features(self.feature_type, conn)
            
            if not features_data:
                conn.close()
                self.error_occurred.emit(f"数据库中未找到类型为 {self.feature_type} 的特征")
                return
                
            print(f"数据库中找到 {len(features_data)} 个特征")
            
            # 计算相似度
            results = []
            for image_id, path, feature_vector in features_data:
                # 检查特征向量是否有效
                if feature_vector is None or len(feature_vector) == 0:
                    print(f"跳过无效特征: image_id={image_id}, path={path}")
                    continue
                    
                # 检查文件是否存在
                if not os.path.exists(path):
                    print(f"跳过不存在的图像文件: {path}")
                    continue
                
                # 确保查询特征和数据库特征维度一致
                if len(query_feature) != len(feature_vector):
                    print(f"特征维度不匹配: 查询特征 {len(query_feature)} vs 数据库特征 {len(feature_vector)}")
                    continue
                
                # 计算相似度/距离
                if self.distance_metric == "余弦相似度":
                    # 对于余弦相似度，直接使用相似度值（值越大越相似）
                    similarity = self.calculate_similarity(query_feature, feature_vector)
                    # 转换为0-1范围的相似度分数
                    similarity_score = (similarity + 1) / 2  # 将[-1,1]映射到[0,1]
                else:
                    # 对于距离度量（欧氏距离、曼哈顿距离），计算距离然后转换为相似度
                    distance = self.calculate_distance(query_feature, feature_vector)
                    # 距离转换为相似度，值越小越相似，需要取倒数
                    # 使用指数衰减确保值在0-1范围内
                    similarity_score = np.exp(-distance)
                
                # 查询图像类别
                category = self.db_manager.get_image_category(image_id, conn)
                
                # 添加结果，使用相似度分数作为排序依据
                results.append((image_id, path, similarity_score, category))
            
            # 关闭数据库连接
            conn.close()
            
            # 根据相似度排序（降序，相似度越高越靠前）
            results.sort(key=lambda x: x[2], reverse=True)
            
            print(f"为查询 {os.path.basename(self.query_image_path)} 找到 {len(results)} 个结果")
            
            # 返回前top_k个结果
            top_results = results[:self.top_k]
            for i, (_, path, similarity, cat) in enumerate(top_results):
                print(f"结果 #{i+1}: {os.path.basename(path)}, 类别: {cat}, 相似度: {similarity:.4f}")
                
            self.retrieval_complete.emit(top_results)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
    
    def create_feature_extractor(self):
        if self.feature_type == "RGB_Histogram":
            return RGBHistogram()
        elif self.feature_type == "Edge_Feature":
            return EdgeFeature()
        elif self.feature_type == "HOG_Feature":
            return HOGFeature()
        elif self.feature_type == "Gabor_Feature":
            return GaborFeature()
        elif self.feature_type == "Daisy_Feature":
            return DaisyFeature()
        elif self.feature_type == "VGG19_Feature":
            return VGG19Feature()
        elif self.feature_type == "ResNet152_Feature":
            return ResNet152Feature()
        return None
    
    def calculate_similarity(self, query_feature, db_feature):
        """计算余弦相似度，范围[-1,1]，值越大越相似"""
        dot_product = np.dot(query_feature, db_feature)
        norm_query = np.linalg.norm(query_feature)
        norm_db = np.linalg.norm(db_feature)
        
        if norm_query == 0 or norm_db == 0:
            return 0  # 避免除以零
        
        return dot_product / (norm_query * norm_db)
    
    def calculate_distance(self, query_feature, db_feature):
        """计算距离，值越小越相似"""
        if self.distance_metric == "欧氏距离":
            return np.sqrt(np.sum((query_feature - db_feature) ** 2))
        elif self.distance_metric == "曼哈顿距离":
            return np.sum(np.abs(query_feature - db_feature))
        else:
            # 默认使用欧氏距离
            return np.sqrt(np.sum((query_feature - db_feature) ** 2))

# 图像缩略图部件
class ThumbnailWidget(QWidget):
    def __init__(self, image_path, similarity=None, category=None):
        super().__init__()
        self.image_path = image_path
        self.similarity = similarity
        self.category = category
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        
        # 创建一个框架来包含图像标签
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.Box)
        image_frame.setFrameShadow(QFrame.Sunken)
        image_frame.setLineWidth(1)
        image_frame.setStyleSheet("QFrame { background-color: white; border: 1px solid #CCCCCC; border-radius: 4px; }")
        
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(3, 3, 3, 3)
        
        # 图像标签
        self.image_label = QLabel()
        self.image_label.setFixedSize(150, 150)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #DDDDDD; background-color: white;")
        layout.addWidget(self.image_label)
        
        if self.category:
            category_label = QLabel(f"类别: {self.category}")
            category_label.setAlignment(Qt.AlignCenter)
            category_label.setStyleSheet("font-weight: bold; color: #2C3E50;")
            layout.addWidget(category_label)
            
        if self.similarity is not None:
            similarity_value = 1 - self.similarity
            similarity_text = f"相似度: {similarity_value:.4f}"
            similarity_label = QLabel(similarity_text)
            similarity_label.setAlignment(Qt.AlignCenter)
            
            # 设置不同级别的相似度颜色 - 修改为深色以适应浅色背景
            if similarity_value > 0.8:
                color = "#006600"  # 高相似度：深绿色
            elif similarity_value > 0.6:
                color = "#004499"  # 中等相似度：深蓝色
            elif similarity_value > 0.4:
                color = "#CC6600"  # 低相似度：深橙色
            else:
                color = "#990000"  # 很低相似度：深红色
                
            similarity_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; }}")
            layout.addWidget(similarity_label)
        
        # 整体样式
        self.setMinimumWidth(180)
        self.setMaximumWidth(200)
        self.setMinimumHeight(230)
        self.setStyleSheet("QWidget { background-color: #F8F8F8; border-radius: 6px; }")
        
        self.setLayout(layout)
    
    def load_image(self):
        try:
            # 检查文件是否存在
            if not os.path.exists(self.image_path):
                print(f"文件不存在: {self.image_path}")
                self.image_label.setText("图像不存在")
                return
                
            # 首先尝试使用OpenCV加载图像
            img_cv = cv2.imread(self.image_path)
            if img_cv is not None:
                # 调整大小
                img_cv = cv2.resize(img_cv, (150, 150))
                # 转换为RGB (OpenCV默认是BGR)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                # 创建QImage
                height, width, channel = img_cv.shape
                bytesPerLine = 3 * width
                qimg = QImage(img_cv.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(pixmap)
                return
                
            # 如果OpenCV失败，尝试使用PIL
            img_pil = Image.open(self.image_path)
            img_pil = img_pil.resize((150, 150), Image.LANCZOS)
            
            # 确保图像是RGB模式
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
                
            # 转换为QImage
            qimg = ImageQt.ImageQt(img_pil)
            pixmap = QPixmap.fromImage(qimg)
            
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"加载图像失败 ({self.image_path}): {e}")
            self.image_label.setText(f"图像加载失败")

# 主窗口
class CBIRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db_manager = DBManager()
        self.extraction_thread = None
        self.retrieval_thread = None
        self.query_image_path = None
        self.selected_result_path = None
        
        # 创建数据库表结构
        conn = self.db_manager.create_connection()
        if conn:
            self.db_manager.create_tables(conn)
            conn.close()
        
        # 设置应用程序样式
        self.apply_stylesheet()
            
        self.setup_ui()
    
    def apply_stylesheet(self):
        """应用全局样式表"""
        qss = """
        QMainWindow {
            background-color: #F0F2F5;
        }
        QGroupBox {
            font-size: 30px;
            font-weight: bold;
            border: 1px solid #D0D0D0;
            border-radius: 6px;
            margin-top: 30px;
            background-color: #FFFFFF;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #2C3E50;
        }
        QPushButton {
            background-color: #3498DB;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            min-height: 30px;
        }
        QPushButton:hover {
            background-color: #2980B9;
        }
        QPushButton:pressed {
            background-color: #1D6FA5;
        }
        QPushButton:disabled {
            background-color: #BDC3C7;
            color: #7F8C8D;
        }
        QLabel {
            color: #2C3E50;
        }
        QComboBox {
            border: 1px solid #BDC3C7;
            border-radius: 4px;
            padding: 2px 10px;
            background-color: white;
            min-height: 28px;
            color: #2C3E50;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: none;
        }
        QLineEdit {
            border: 1px solid #BDC3C7;
            border-radius: 4px;
            padding: 4px;
            background-color: white;
            min-height: 28px;
            color: #2C3E50;
        }
        QProgressBar {
            border: 1px solid #BDC3C7;
            border-radius: 4px;
            text-align: center;
            background-color: #ECF0F1;
            min-height: 14px;
            color: #2C3E50;
        }
        QProgressBar::chunk {
            background-color: #3498DB;
            border-radius: 3px;
        }
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QScrollBar:vertical {
            border: none;
            background-color: #F0F2F5;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #BDC3C7;
            border-radius: 5px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #95A5A6;
        }
        QSpinBox {
            border: 1px solid #BDC3C7;
            border-radius: 4px;
            padding: 2px;
            background-color: white;
            min-height: 28px;
            color: #2C3E50;
        }
        QSplitter::handle {
            background-color: #D0D0D0;
        }
        """
        self.setStyleSheet(qss)
    
    def setup_ui(self):
        self.setWindowTitle("基于CBIR的船舶图像分类系统")
        self.setGeometry(100, 100, 1300, 750)  # 减小窗口高度
        
        # 设置窗口图标
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        
        # 创建中央部件
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)  # 减小布局间距
        self.setCentralWidget(central_widget)
        
        # 创建左右分割器
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(2)
        
        # 左侧：操作区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)  # 减小布局间距
        
        # 上部分：数据库和查询操作
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)  # 设置合适的间距
        
        # 数据库操作区域
        db_group = QGroupBox("数据库操作")
        db_layout = QVBoxLayout()
        
        # 数据集目录选择
        dataset_layout = QHBoxLayout()
        self.dataset_path_edit = QLineEdit()
        # 设置默认数据集路径为用户提供的船舶数据集路径
        self.dataset_path_edit.setText("E:/CBIR/simple-CBIR-core/database")
        self.dataset_path_edit.setReadOnly(True)
        dataset_layout.addWidget(QLabel("数据集目录:"))
        dataset_layout.addWidget(self.dataset_path_edit)
        
        browse_button = QPushButton("浏览...")
        browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse_button.clicked.connect(self.browse_dataset)
        browse_button.setMaximumWidth(100)
        dataset_layout.addWidget(browse_button)
        db_layout.addLayout(dataset_layout)
        
        # 特征类型选择
        feature_layout = QHBoxLayout()
        feature_layout.addWidget(QLabel("特征类型:"))
        
        self.feature_combo = QComboBox()
        self.feature_combo.addItems([
            "RGB_Histogram", 
            "Edge_Feature", 
            "HOG_Feature", 
            "Gabor_Feature", 
            "Daisy_Feature", 
            "VGG19_Feature", 
            "ResNet152_Feature"
        ])
        feature_layout.addWidget(self.feature_combo)
        db_layout.addLayout(feature_layout)
        
        # 提取特征按钮
        self.extract_button = QPushButton("提取特征并存入数据库")
        self.extract_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.extract_button.clicked.connect(self.extract_features)
        db_layout.addWidget(self.extract_button)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedHeight(20)  # 减小进度条高度
        db_layout.addWidget(self.progress_bar)
        
        # 数据库信息
        self.db_info_label = QLabel("数据库状态: 0 张图像")
        self.db_info_label.setStyleSheet("font-weight: bold; color: #16A085;")
        db_layout.addWidget(self.db_info_label)
        
        db_group.setLayout(db_layout)
        top_layout.addWidget(db_group)
        
        # 查询操作区域
        query_group = QGroupBox("图像检索")
        query_layout = QVBoxLayout()
        
        # 查询图像选择
        query_image_layout = QHBoxLayout()
        
        # 创建一个框架来包含查询图像
        query_image_frame = QFrame()
        query_image_frame.setFrameShape(QFrame.Box)
        query_image_frame.setFrameShadow(QFrame.Sunken)
        query_image_frame.setLineWidth(1)
        query_image_frame.setStyleSheet("background-color: white; border: 1px solid #CCCCCC; border-radius: 4px;")
        
        query_image_frame_layout = QVBoxLayout(query_image_frame)
        query_image_frame_layout.setContentsMargins(3, 3, 3, 3)
        
        self.query_image_label = QLabel("未选择图像")
        self.query_image_label.setFixedSize(150, 150)  # 减小查询图像尺寸
        self.query_image_label.setAlignment(Qt.AlignCenter)
        self.query_image_label.setScaledContents(True)
        self.query_image_label.setStyleSheet("border: none;")
        
        query_image_frame_layout.addWidget(self.query_image_label)
        query_image_layout.addWidget(query_image_frame)
        
        query_buttons_layout = QVBoxLayout()
        upload_button = QPushButton("上传查询图像")
        upload_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        upload_button.clicked.connect(self.upload_query_image)
        query_buttons_layout.addWidget(upload_button)
        
        # 按钮：可视化查询图像特征
        visualize_query_button = QPushButton("可视化查询特征")
        visualize_query_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))
        visualize_query_button.clicked.connect(self.visualize_query_feature)
        query_buttons_layout.addWidget(visualize_query_button)
        
        query_image_layout.addLayout(query_buttons_layout)
        query_layout.addLayout(query_image_layout)
        
        # 特征选择
        retrieval_feature_layout = QHBoxLayout()
        retrieval_feature_layout.addWidget(QLabel("检索特征类型:"))
        
        self.retrieval_feature_combo = QComboBox()
        self.update_feature_types()
        self.retrieval_feature_combo.currentTextChanged.connect(self.on_feature_type_changed)
        retrieval_feature_layout.addWidget(self.retrieval_feature_combo)
        query_layout.addLayout(retrieval_feature_layout)
        
        # 相似度度量选择
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("相似度度量:"))
        
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["欧氏距离", "余弦相似度", "曼哈顿距离"])
        metric_layout.addWidget(self.metric_combo)
        query_layout.addLayout(metric_layout)
        
        # 结果数量
        results_layout = QHBoxLayout()
        results_layout.addWidget(QLabel("返回结果数量:"))
        
        self.results_spin = QSpinBox()
        self.results_spin.setRange(1, 100)
        self.results_spin.setValue(20)
        results_layout.addWidget(self.results_spin)
        query_layout.addLayout(results_layout)
        
        # 检索按钮
        self.retrieve_button = QPushButton("开始检索")
        self.retrieve_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
        self.retrieve_button.clicked.connect(self.retrieve_images)
        self.retrieve_button.setStyleSheet("""
            QPushButton {
                background-color: #16A085;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138D75;
            }
            QPushButton:pressed {
                background-color: #0E6655;
            }
        """)
        query_layout.addWidget(self.retrieve_button)
        
        query_group.setLayout(query_layout)
        top_layout.addWidget(query_group)
        
        top_widget.setLayout(top_layout)
        left_layout.addWidget(top_widget)
        
        # 添加特征可视化区域
        feature_vis_group = QGroupBox("特征可视化")
        feature_vis_layout = QVBoxLayout()
        feature_vis_layout.setContentsMargins(5, 10, 5, 5)  # 减小内边距
        feature_vis_layout.setSpacing(5)  # 减小间距
        
        # 创建图表区域
        self.figure = Figure(figsize=(5, 3), dpi=100)  # 减小图形高度
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250)  # 减小最小高度
        self.canvas.setMaximumHeight(250)  # 限制最大高度
        
        # 设置图表样式
        self.figure.patch.set_facecolor('#FFFFFF')
        feature_vis_layout.addWidget(self.canvas)
        
        # 特征可视化控制区域
        vis_control_layout = QHBoxLayout()
        
        # 使用一个标签来显示当前选择的图像路径
        self.selected_image_label = QLabel("选择图像进行特征可视化")
        self.selected_image_label.setStyleSheet("font-style: italic; color: #2C3E50;")
        vis_control_layout.addWidget(self.selected_image_label)
        
        feature_vis_layout.addLayout(vis_control_layout)
        
        feature_vis_group.setLayout(feature_vis_layout)
        left_layout.addWidget(feature_vis_group)
        
        left_widget.setLayout(left_layout)
        
        # 右侧：结果显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        results_group = QGroupBox("检索分类结果")
        results_layout = QVBoxLayout()
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #FFFFFF; QScrollArea { padding-right: 0px; margin-right: 0px;  }")
        
        # 创建结果网格
        self.results_widget = QWidget()
        self.results_grid = QGridLayout(self.results_widget)
        self.results_grid.setContentsMargins(20, 20, 20, 20)
        self.results_grid.setSpacing(20)  # 增加项目之间的间距
        self.results_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        scroll_area.setWidget(self.results_widget)
        results_layout.addWidget(scroll_area)
        
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)
        
        right_widget.setLayout(right_layout)
        
        # 将左右部分添加到主分割器
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        
        # 设置分割器初始大小
        main_splitter.setSizes([550, 750])
        
        main_layout.addWidget(main_splitter)
        
        # 初始化数据库信息
        self.update_db_info()
        
        # 设置全局字体
        app_font = QFont()
        app_font.setFamily("微软雅黑" if system == "Windows" else "PingFang SC")
        app_font.setPointSize(10)
        QApplication.setFont(app_font)
    
    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if folder:
            self.dataset_path_edit.setText(folder)
    
    def upload_query_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择查询图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            try:
                # 加载图像
                img = Image.open(file_path)
                img = img.resize((150, 150), Image.LANCZOS)
                
                # 转换为QImage
                qimg = ImageQt.ImageQt(img)
                pixmap = QPixmap.fromImage(qimg)
                
                self.query_image_label.setPixmap(pixmap)
                self.query_image_path = file_path
                
                # 更新选择的图像
                self.selected_image_label.setText(f"查询图像: {os.path.basename(file_path)}")
                
                # 如果已选择特征类型，自动可视化
                if self.retrieval_feature_combo.count() > 0:
                    self.visualize_feature(file_path, self.retrieval_feature_combo.currentText())
            except Exception as e:
                QMessageBox.warning(self, "图像加载失败", f"无法加载图像: {str(e)}")

    def extract_features(self):
        dataset_path = self.dataset_path_edit.text()
        if not dataset_path:
            QMessageBox.warning(self, "错误", "请选择数据集目录")
            return
            
        feature_type = self.feature_combo.currentText()
        
        # 创建并启动特征提取线程
        self.extract_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.extraction_thread = FeatureExtractionThread(dataset_path, feature_type)
        self.extraction_thread.feature_extracted.connect(self.on_feature_extracted)
        self.extraction_thread.error_occurred.connect(self.on_extraction_error)
        self.extraction_thread.progress_updated.connect(self.update_progress)
        self.extraction_thread.start()
    
    def update_progress(self, value):
        """更新进度条的值"""
        self.progress_bar.setValue(value)
        
    def on_feature_extracted(self, feature, feature_type):
        self.extract_button.setEnabled(True)
        self.update_db_info()
        self.update_feature_types()
        QMessageBox.information(self, "完成", f"特征提取完成，特征类型: {feature_type}")
    
    def on_extraction_error(self, error_message):
        self.extract_button.setEnabled(True)
        QMessageBox.critical(self, "错误", f"特征提取错误: {error_message}")
    
    def update_db_info(self):
        conn = self.db_manager.create_connection()
        if conn:
            image_count = self.db_manager.get_image_count(conn)
            conn.close()
            self.db_info_label.setText(f"数据库状态: {image_count} 张图像")
    
    def update_feature_types(self):
        conn = self.db_manager.create_connection()
        if conn:
            self.retrieval_feature_combo.clear()
            feature_types = self.db_manager.get_feature_types(conn)
            conn.close()
            if feature_types:
                self.retrieval_feature_combo.addItems(feature_types)
    
    def on_feature_type_changed(self, feature_type):
        """当特征类型改变时更新可视化"""
        if self.query_image_path:
            self.visualize_feature(self.query_image_path, feature_type)
        elif self.selected_result_path:
            self.visualize_feature(self.selected_result_path, feature_type)
    
    def visualize_query_feature(self):
        """可视化查询图像的特征"""
        if not self.query_image_path:
            QMessageBox.warning(self, "错误", "请先上传查询图像")
            return
            
        if self.retrieval_feature_combo.count() == 0:
            QMessageBox.warning(self, "错误", "没有可用的特征类型，请先提取特征")
            return
            
        feature_type = self.retrieval_feature_combo.currentText()
        self.visualize_feature(self.query_image_path, feature_type)
    
    def visualize_result_feature(self, image_path):
        """可视化结果图像的特征"""
        if self.retrieval_feature_combo.count() == 0:
            QMessageBox.warning(self, "错误", "没有可用的特征类型，请先提取特征")
            return
            
        feature_type = self.retrieval_feature_combo.currentText()
        self.selected_result_path = image_path
        self.selected_image_label.setText(f"结果图像: {os.path.basename(image_path)}")
        self.visualize_feature(image_path, feature_type)
    
    def visualize_feature(self, image_path, feature_type):
        """可视化图像的特征"""
        if FeatureVisualization.visualize_feature(image_path, feature_type, self.figure):
            self.canvas.draw()
        else:
            # 可视化失败，显示错误信息
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"无法可视化 {feature_type} 特征", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, color='#CC0000')  # 使用红色显示错误信息
            ax.axis('off')
            self.canvas.draw()

    def retrieve_images(self):
        if not self.query_image_path:
            QMessageBox.warning(self, "错误", "请上传查询图像")
            return
            
        if self.retrieval_feature_combo.count() == 0:
            QMessageBox.warning(self, "错误", "没有可用的特征类型，请先提取特征")
            return
        
        feature_type = self.retrieval_feature_combo.currentText()
        distance_metric = self.metric_combo.currentText()
        top_k = self.results_spin.value()
        
        # 创建并启动检索线程
        self.retrieve_button.setEnabled(False)
        
        self.retrieval_thread = ImageRetrievalThread(
            self.db_manager, 
            self.query_image_path,
            feature_type,
            distance_metric,
            top_k
        )
        self.retrieval_thread.retrieval_complete.connect(self.display_results)
        self.retrieval_thread.error_occurred.connect(self.on_retrieval_error)
        self.retrieval_thread.start()

    def display_results(self, results):
        # 清除旧结果
        for i in reversed(range(self.results_grid.count())):
            widget = self.results_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # 显示新结果
        row, col = 0, 0
        max_cols = 4  # 每行显示的缩略图数量，减少为4个以适应新的更大的缩略图
        
        for image_id, path, similarity, category in results:
            # 创建可点击的缩略图部件
            thumbnail = ClickableThumbnailWidget(path, similarity, category)
            thumbnail.image_clicked.connect(lambda img_path=path: self.visualize_result_feature(img_path))
            self.results_grid.addWidget(thumbnail, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        self.retrieve_button.setEnabled(True)
    
    def on_retrieval_error(self, error_message):
        self.retrieve_button.setEnabled(True)
        QMessageBox.critical(self, "错误", f"图像检索错误: {error_message}")
    
    def closeEvent(self, event):
        # 停止线程
        if self.extraction_thread and self.extraction_thread.isRunning():
            self.extraction_thread.stop()
            self.extraction_thread.wait()
        
        # 关闭数据库连接
        self.db_manager.close()
        
        event.accept()

# 可点击的缩略图部件
class ClickableThumbnailWidget(QWidget):
    image_clicked = pyqtSignal(str)  # 点击图像时发出信号
    
    def __init__(self, image_path, similarity, category):
        super().__init__()
        self.image_path = image_path
        self.similarity = similarity
        self.category = category
        
        # 设置布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 减小边距
        layout.setSpacing(2)  # 减小组件之间的间距
        
        # 图像标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(200, 200)
        self.image_label.setScaledContents(True)
        
        # 加载图像
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.setText("无法加载图像")
                self.image_label.setStyleSheet("color: #CC0000; background-color: #F5F5F5; border: 1px solid #DADADA;")
        except Exception as e:
            self.image_label.setText("图像加载错误")
            self.image_label.setStyleSheet("color: #CC0000; background-color: #F5F5F5; border: 1px solid #DADADA;")
        
        # 文件名标签
        filename = os.path.basename(image_path)
        filename_label = QLabel(filename if len(filename) < 20 else filename[:17] + "...")
        filename_label.setAlignment(Qt.AlignCenter)
        filename_label.setStyleSheet("color: #2C3E50; font-size: 11px; font-weight: bold;")
        filename_label.setToolTip(filename)
        
        # 相似度标签
        similarity_str = f"{similarity:.4f}" if isinstance(similarity, float) else str(similarity)
        similarity_value = float(similarity_str) if similarity_str else 0
        
        # 根据相似度值设置颜色（使用更深的颜色以在浅色背景上更加可见）
        if similarity_value >= 0.7:
            color = "#006600"  # 深绿色
            bg_color = "#E8F5E9"  # 浅绿背景
        elif similarity_value >= 0.5:
            color = "#2E7D32"  # 绿色
            bg_color = "#F1F8E9"  # 非常浅的绿背景
        elif similarity_value >= 0.3:
            color = "#F57C00"  # 橙色
            bg_color = "#FFF3E0"  # 浅橙背景
        else:
            color = "#C62828"  # 红色
            bg_color = "#FFEBEE"  # 浅红背景
            
        self.similarity_label = QLabel(f"相似度: {similarity_str}")
        self.similarity_label.setAlignment(Qt.AlignCenter)
        self.similarity_label.setStyleSheet(f"""
            color: {color}; 
            background-color: {bg_color}; 
            border-radius: 4px; 
            padding: 2px 4px; 
            font-size: 17px;
            font-weight: bold;
        """)
        
        # 类别标签
        category_label = QLabel(f"类别: {category}" if category else "")
        category_label.setAlignment(Qt.AlignCenter)
        category_label.setStyleSheet(f"""
            color: #2C3E50; 
            background-color: #E8F0F8; 
            border-radius: 4px; 
            padding: 2px 4px; 
            font-size: 16px;
        """)
        
        # 设置整体样式
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border: 1px solid #DADADA;
                border-radius: 6px;
            }
            QWidget:hover {
                border: 1px solid #3498DB;
                background-color: #EFF8FD;
            }
            QWidget:pressed {
                border: 1px solid #2980B9;
                background-color: #D6EAF8;
            }
        """)
        
        # 添加组件到布局
        layout.addWidget(self.image_label)
        layout.addWidget(filename_label)
        layout.addWidget(self.similarity_label)
        if category:
            layout.addWidget(category_label)
        
        # 固定大小
        self.setFixedSize(180, 230)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 发送信号
            self.image_clicked.emit(self.image_path)
            # 设置选中状态样式
            self.setStyleSheet("""
                QWidget {
                    background-color: #D6EAF8;
                    border: 2px solid #3498DB;
                    border-radius: 6px;
                }
            """)
        super().mousePressEvent(event)

# 主函数
def main():
    app = QApplication(sys.argv)
    window = CBIRApp()
    window.show()
    sys.exit(app.exec_()) 

if __name__ == "__main__":
    main() 


