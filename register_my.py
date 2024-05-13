# -*- coding: utf-8 -*-

"""
该脚本用于训练一个实例分割模型，基于COCO数据集，并对自定义的数据集进行训练和验证。
"""

import detectron2
# 导入 detectron2.utils.logger 模块中的 setup_logger 函数
from detectron2.utils.logger import setup_logger

# 配置日志记录器
setup_logger()



# 导入detectron2库的相关模块和类
from detectron2 import model_zoo
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

"""
detectron2是Facebook AI的一个开源物体检测库。以下是从该库导入的一些关键模块和函数，用于配置和使用模型。

model_zoo: 提供预训练模型的访问接口。
DatasetCatalog: 用于注册和获取数据集的目录。
MetadataCatalog: 用于存储数据集的元数据（如类别名称、像素值范围等）的目录。
register_coco_instances: 用于注册COCO格式的数据集。

这些导入的模块和函数是构建和训练物体检测模型的基础。
"""


# 注册训练集和验证集
register_coco_instances('self_coco_train', {},
                        './my_coco_dataset/data_dataset_coco_train/annotations.json',
                        './my_coco_dataset/data_dataset_coco_train')
register_coco_instances('self_coco_val', {},
                        './my_coco_dataset/data_dataset_coco_val/annotations.json',
                        './my_coco_dataset/data_dataset_coco_val')

# 获取元数据和数据字典
coco_val_metadata = MetadataCatalog.get("self_coco_val")
dataset_dicts = DatasetCatalog.get("self_coco_val")
coco_train_metadata = MetadataCatalog.get("self_coco_train")
dataset_dicts1 = DatasetCatalog.get("self_coco_train")

# 可视化训练数据
my_dataset_train_metadata = MetadataCatalog.get("self_coco_train")
dataset_dicts = DatasetCatalog.get("self_coco_train")

# 自定义训练师类，以在训练期间进行COCO验证评估
# 导入Detectron2库的相关模块
from detectron2.engine import DefaultTrainer  # DefaultTrainer是Detectron2中用于训练模型的默认类
from detectron2.evaluation import COCOEvaluator  # COCOEvaluator用于评估COCO数据集上的模型性能




class CocoTrainer(DefaultTrainer):
    """
    自定义训练师类，用于在训练过程中进行COCO验证集的评估。
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        构建评估器。

        参数:
        cfg: 配置文件。
        dataset_name: 数据集名称。
        output_folder: 输出文件夹路径。

        返回:
        返回一个COCOEvaluator实例。
        """
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# 配置训练参数
import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("self_coco_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置模型的测试阈值
cfg.DATASETS.TEST = ("self_coco_val",)
cfg.MODEL.DEVICE = 0

# 创建输出目录并开始训练
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.train()

# 导入os模块
# 该模块提供了访问操作系统功能的方法，例如创建和删除文件、管理进程、设置权限、访问环境变量等。
import os


# 打印模型权重路径并导出模型
print(cfg.MODEL.WEIGHTS)

os.system("python ./tools/deploy/export_model.py "
          "--config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml "
          "--output ./model_cpu --export-method tracing --format torchscript "
          "MODEL.WEIGHTS ./output/model_final.pth  MODEL.DEVICE cpu")
