# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI), 2020. All Rights Reserved.
import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from coco_evaluation import COCOEvaluator
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    # COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from centermask.evaluation import (
    # COCOEvaluator,
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from centermask.config import get_cfg
from detectron2.data.datasets import register_coco_instances

register_coco_instances('LIVECell_all_train', {}, '/path/to/your/ann/file/to/LIVECell_all_train.json','/path/to/your/data/file/to/train' )
register_coco_instances('LIVECell_all_val', {}, '/path/to/your/ann/file/to/LIVECell_all_val.json','/path/to/your/data/file/to/val' )
register_coco_instances('LIVECell_all_test', {}, '/path/to/your/ann/file/to/LIVECell_all_test.json','/path/to/your/data/file/to/test' )
register_coco_instances('LIVECell_all_test_on_A172', {}, '/path/to/your/ann/file/to/LIVECell_A172_test.json','/path/to/your/data/file/to/test' )
register_coco_instances('LIVECell_all_test_on_BT474', {}, '/path/to/your/ann/file/to/LIVECell_BT474_test.json','/path/to/your/data/file/to/test' )
register_coco_instances('LIVECell_all_test_on_BV2', {}, '/path/to/your/ann/file/to/LIVECell_BV2_test.json','/path/to/your/data/file/to/test' )
register_coco_instances('LIVECell_all_test_on_Huh7', {}, '/path/to/your/ann/file/to/LIVECell_Huh7_test.json','/path/to/your/data/file/to/test' )
register_coco_instances('LIVECell_all_test_on_MCF7', {}, '/path/to/your/ann/file/to/LIVECell_MCF7_test.json','/path/to/your/data/file/to/test' )
register_coco_instances('LIVECell_all_test_on_SHSY5Y', {}, '/path/to/your/ann/file/to/LIVECell_SHSY5Y_test.json','/path/to/your/data/file/to/test' )
register_coco_instances('LIVECell_all_test_on_SkBr3', {}, '/path/to/your/ann/file/to/LIVECell_SkBr3_test.json','/path/to/your/data/file/to/test' )
register_coco_instances('LIVECell_all_test_on_SKOV3', {}, '/path/to/your/ann/file/to/LIVECell_SKOV3_test.json','/path/to/your/data/file/to/test' )
register_coco_instances('tissuenet_nuclear_train', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_all_train.json','/path/to/your/data/file/to/nuclear/train' )
register_coco_instances('tissuenet_nuclear_val', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_all_val.json','/path/to/your/data/file/to/nuclear/train' )
register_coco_instances('tissuenet_nuclear_test', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_all_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_Breast', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_Breast_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_Colon', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_Colon_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_Epidermis', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_Epidermis_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_Esophagus', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_Esophagus_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_Lung', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_Lung_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_lymph_node_metastasis', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_lymph_node_metastasis_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_Lymph_Node', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_Lymph_Node_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_Pancreas', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_Pancreas_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_Spleen', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_Spleen_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_nuclear_test_on_Tonsil', {}, '/path/to/your/ann/file/to/tissuenet_nuclear_Tonsil_test.json','/path/to/your/data/file/to/nuclear/test' )
register_coco_instances('tissuenet_wholecell_train', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_all_train.json','/path/to/your/data/file/to/wholecell/train' )
register_coco_instances('tissuenet_wholecell_val', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_all_val.json','/path/to/your/data/file/to/wholecell/val' )
register_coco_instances('tissuenet_wholecell_test', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_all_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_Breast', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_Breast_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_Colon', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_Colon_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_Epidermis', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_Epidermis_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_Esophagus', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_Esophagus_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_Lung', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_Lung_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_lymph_node_metastasis', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_lymph_node_metastasis_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_Lymph_Node', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_Lymph_Node_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_Pancreas', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_Pancreas_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_Spleen', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_Spleen_test.json','/path/to/your/data/file/to/wholecell/test' )
register_coco_instances('tissuenet_wholecell_test_on_Tonsil', {}, '/path/to/your/ann/file/to/tissuenet_wholecell_Tonsil_test.json','/path/to/your/data/file/to/wholecell/test' )



class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader` method.
    """



    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # print(evaluator_type)
        # exit(0)
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
