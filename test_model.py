import torch
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.roi_heads import Res5ROIHeads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2 import model_zoo

# 加载模型
model = torch.jit.load('./output/model.ts', map_location="cpu")  # 或者"cuda" if torch.cuda.is_available()

# 创建配置对象，并加载一些基本配置
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# 设置测试数据集的元数据（这里假设已正确设置）
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置分数阈值以过滤预测结果
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 获取元数据，确保DATASETS.TEST已正确设置
# 注意：此处cfg.DATASETS.TEST需要根据实际情况提前设定
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

# 确保模型与配置兼容，这里简化处理，实际应用中可能需要更复杂的逻辑来确保兼容性
# 假设我们处理的是Mask R-CNN模型
model = model.to(cfg.MODEL.DEVICE)

# 读取自定义图片
image_path = '0_6_13_3330.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)
# 将OpenCV图像（HWC, BGR）转换为RGB并转为PyTorch tensor
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div(255)
image = image.unsqueeze(0).to(cfg.MODEL.DEVICE)  # 添加batch维度并转移到相应设备上

# 准备输入格式
inputs = {"image": image, "height": image.shape[2], "width": image.shape[3]}

# 直接调用模型进行预测
with torch.no_grad():
    outputs = model(inputs)

# 后处理
if isinstance(outputs, dict):
    # 对于某些模型，输出可能是一个字典，需要进一步处理
    instances = outputs['instances']
else:
    # 假设输出直接是Instances对象
    instances = outputs

# 使用Visualizer可视化结果
v = Visualizer(image.numpy(), metadata=metadata, scale=1.2)
v = v.draw_instance_predictions(instances.to("cpu"))
cv2.imshow('Prediction', v.get_image()[:, :, ::-1])  # 显示图像
cv2.waitKey(0)  # 等待按键后关闭窗口


