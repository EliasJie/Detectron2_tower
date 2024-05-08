# -*- coding: utf-8 -*-

import os
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# 设置日志记录
setup_logger()

def register_datasets():
    """
    注册训练和验证用的COCO数据集。
    """
    try:
        # 注册训练数据集
        register_coco_instances('self_coco_train', {},
                                './my_coco_dataset/data_dataset_coco_train/annotations.json',
                                './my_coco_dataset/data_dataset_coco_train')
        # 注册验证数据集
        register_coco_instances('self_coco_val', {},
                                './my_coco_dataset/data_dataset_coco_val/annotations.json',
                                './my_coco_dataset/data_dataset_coco_val')
    except Exception as e:
        print(f"Error registering datasets: {e}")

def setup_config(output_dir):
    """
    设置并返回训练配置。

    参数:
    - output_dir: 训练输出的目录路径。

    返回值:
    - 配置对象(cfg): 包含训练所需全部配置的对象。
    """
    cfg = get_cfg()
    # 从预定义配置加载基础设置
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # 设置训练和测试的数据集
    cfg.DATASETS.TRAIN = ("self_coco_train",)
    cfg.DATASETS.TEST = ()
    # 设置数据加载器的工作线程数
    cfg.DATALOADER.NUM_WORKERS = 2
    # 设置模型初始权重
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # 设置训练的批次大小和学习率
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    # 设置总迭代次数
    cfg.SOLVER.MAX_ITER = 40000 //4000
    # 设置类别数
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    # 设置模型权重保存路径
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # 设置测试时的阈值
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # 设置测试数据集
    cfg.DATASETS.TEST = ("self_coco_val", )
    # 设置模型训练的设备
    cfg.MODEL.DEVICE = 0
    # 创建输出目录
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def train_model(cfg):
    """
    训练模型。

    参数:
    - cfg: 包含训练配置的配置对象。
    """
    try:
        # 创建训练器并开始训练
        trainer = DefaultTrainer(cfg)
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")

def export_model(cfg):
    """
    导出训练好的模型。

    参数:
    - cfg: 包含训练配置的配置对象。
    """
    try:
        # 打印模型权重路径并执行模型导出脚本
        print(cfg.MODEL.WEIGHTS)
        os.system("python ./tools/deploy/export_model.py --config-file ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml "
                  "--output ./output --export-method tracing --format torchscript "
                  "MODEL.WEIGHTS ./output/model_final.pth  MODEL.DEVICE cpu")
    except Exception as e:
        print(f"Error during model export: {e}")

if __name__ == "__main__":
    # 注册数据集
    register_datasets()
    # 设置输出目录并配置模型训练参数
    cfg = setup_config('./output')
    # 训练模型
    train_model(cfg)
    # 导出训练好的模型
    export_model(cfg)
