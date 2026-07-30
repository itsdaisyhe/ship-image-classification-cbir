import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

class VGG19Feature:
    def __init__(self, use_gpu=True):
        """
        初始化VGG19深度特征提取器
        
        参数:
            use_gpu: 是否使用GPU加速
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = models.vgg19(pretrained=True)
        
        # 移除最后的全连接层，使用倒数第二个全连接层作为特征
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # 设置为评估模式，不计算梯度
        self.model.eval()
        self.model.to(self.device)
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, image_path):
        """
        从图像中提取VGG19深度特征
        
        参数:
            image_path: 图像文件路径
            
        返回:
            numpy数组，VGG19深度特征向量
        """
        try:
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            
            # 预处理图像
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model(input_batch)
            
            # 将特征转换为numpy数组
            features = features.squeeze().cpu().numpy()
            
            # 将特征向量转换为float32类型
            features = features.astype(np.float32)
            
            # 如果特征维度太高，可以使用PCA等降维方法降低维度
            
            return features
            
        except Exception as e:
            print(f"VGG19特征提取错误: {e}")
            return None 