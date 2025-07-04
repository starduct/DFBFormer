# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "1"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "2"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "3"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "4"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "5"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "6"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "7"},
]


def _get_coco_zebrafish_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in COCO_CATEGORIES]
    assert len(stuff_ids) == 7, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


# def register_all_coco_zebrafish(root):
#     root = os.path.join(root, "coco_zebrafish_segmentation")
#     meta = _get_coco_zebrafish_meta()
#     for name, image_dirname, sem_seg_dirname in [
#         ("train", "train2017", "annotations/train2017"),
#         ("test", "val2017", "annotations/val2017"),
#     ]:
#         image_dir = os.path.join(root, image_dirname)
#         gt_dir = os.path.join(root, sem_seg_dirname)
#         name = f"coco_zebrafish_{name}_sem_seg"
#         DatasetCatalog.register(
#             name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
#         )
#         MetadataCatalog.get(name).set(
#             image_root=image_dir,
#             sem_seg_root=gt_dir,
#             evaluator_type="sem_seg",
#             ignore_label=255,
#             **meta,
#         )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
# _root = os.path.join("/lustre/chaixiujuan/ChaiXin", "datasets")
root = os.path.join(_root, "coco_zebrafish_segmentation")

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json
def register_all_coco_zebrafish(root):
    meta = _get_coco_zebrafish_meta()
    # load_coco_json(os.path.join(root, "annotations/train2017.json"), os.path.join(root, "train2017"), dataset_name="coco_zebrafish_train", extra_annotation_keys=None)
    # load_coco_json(os.path.join(root, "annotations/val2017.json"), os.path.join(root, "val2017"), dataset_name="coco_zebrafish_val", extra_annotation_keys=None)
    register_coco_instances(f"coco_zebrafish_train", {}, os.path.join(root, "annotations/train2017.json"), os.path.join(root, "train2017"))
    register_coco_instances(f"coco_zebrafish_val", {}, os.path.join(root, "annotations/val2017.json"), os.path.join(root, "val2017"))
    MetadataCatalog.get("coco_zebrafish_train").set(
        # evaluator_type="sem_seg",
        ignore_label=255,
        **meta,
        )
    MetadataCatalog.get("coco_zebrafish_val").set(
        # evaluator_type="sem_seg",
        ignore_label=255,
        **meta,
    )

register_all_coco_zebrafish(root)

