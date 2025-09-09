"""
The overall cost will consist of three main parts:

*) C_classes - the classification loss, i.e. how well the predicted class matches the true one.

*) C_boxes - the L1 distance (Manhattan distance) between the predicted coordinates and the actual coordinates of the bounding boxes.

*) C_giou - the negative generalized intersection over union, which shows how well the shape of the predicted bounding boxes matches the actual/target bounding boxes.

Along with that, we have the "importance" of each loss, so we have weights.
"""

# model training setup
# hungarian algorithm
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from torchvision import ops
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from dataSetup import CLASS2IDX
import pandas as pd
import torch

# skip corrupted images
class DetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_csv, transform=None):
        self.image_dir  = Path(image_dir)
        self.transform  = transform

        df = pd.read_csv(annotation_csv)


        df = df[df["class"] != "background"]


        exists = df["filename"].map(
            lambda x: (self.image_dir / Path(x).name).exists())
        self.annotations = df[exists].reset_index(drop=True)

        self.grouped     = self.annotations.groupby("filename")
        self.image_files = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        start_idx = idx
        while True:
            filename = self.image_files[idx]
            img_path = self.image_dir / Path(filename).name

            if img_path.exists():
                try:
                    image = Image.open(img_path).convert("RGB")
                    break
                except (FileNotFoundError, OSError):
                    pass

            idx = (idx + 1) % len(self)
            if idx == start_idx:
                raise RuntimeError("No valid images left in dataset")

        if self.transform:
            image = self.transform(image)

        records = self.grouped.get_group(filename)
        boxes   = torch.as_tensor(records[["xmin","ymin","xmax","ymax"]].values,
                                  dtype=torch.float32)
        labels  = torch.as_tensor(records["class"]
                                  .map(CLASS2IDX).values, dtype=torch.long)
        return image, {"boxes": boxes, "labels": labels}



# general project constants
IMAGE_SIZE      = 640
NUM_CLASSES     = 1
# # index of "empty" class = 10
EMPTY_CLASS     = NUM_CLASSES
# must match detr.num_queries
NUM_QUERIES     = 20
# weights of the composite COST metric
W_CLASS, W_L1, W_GIOU = 1, 5, 2


# helper functions
def _abs_boxes_xyxy_to_norm_cxcywh(boxes: torch.Tensor, img_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Translates boxes from absolute coordinates (xmin, ymin, xmax, ymax)
    to normalized ones (cx, cy, w, h) ∈ [0,1].
    """
    # center coordinates
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    # width / height
    w  = boxes[:, 2] - boxes[:, 0]
    h  = boxes[:, 3] - boxes[:, 1]

    # normalize to image size
    out = torch.stack([cx, cy, w, h], dim=-1) / img_size
    return out

# total COST function
def _cost_matrix(pred_boxes_cxcywh: torch.Tensor,
                 tgt_boxes_cxcywh: torch.Tensor,
                 pred_logits: torch.Tensor,
                 tgt_labels: torch.Tensor) -> torch.Tensor:
    """
    Build a cost matrix (Q × T) for one picture:
    -C_classes  + 5·L1(box)  – 2·gIoU.  Minus sign because we want maximum similarity
    """
    # class cost  (shape: Q × T)
    # shape: Q , C+1
    prob = pred_logits.softmax(dim=-1)
    # indexing by label vector (Q, T)
    C_classes = -prob[:, tgt_labels]


    #  L1 component (box cost)
    # shape: Q × T
    C_boxes = torch.cdist(pred_boxes_cxcywh, tgt_boxes_cxcywh, p=1)

    # gIoU cost (shape: Q × T)
    #  gIoU requires a representation of XYXY in the range [0,1]
    pred_xyxy = ops.box_convert(pred_boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')
    tgt_xyxy  = ops.box_convert(tgt_boxes_cxcywh,  in_fmt='cxcywh', out_fmt='xyxy')
    C_giou = -ops.generalized_box_iou(pred_xyxy, tgt_xyxy)

    # total COST
    C_total = W_CLASS * C_classes + W_L1 * C_boxes + W_GIOU * C_giou
    return C_total


# hungarian algorithm
def _hungarian_match(cost: torch.Tensor):
    """
    Applies scipy.optimize.linear_sum_assignment to the torch cost matrix.
    Returns indices of (pred_idx, tgt_idx) pairs as Tenors.
    """
    cost_np = cost.cpu().detach().numpy()
    pred_idx, tgt_idx = linear_sum_assignment(cost_np)
    return torch.as_tensor(pred_idx, dtype=torch.int64), torch.as_tensor(tgt_idx, dtype=torch.int64)


# calculate image loss
def _loss_per_sample(pred_boxes_cxcywh: torch.Tensor,
                     tgt_boxes_cxcywh: torch.Tensor,
                     pred_logits: torch.Tensor,
                     tgt_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the loss for one image.
    Returns: (loss_class, loss_bbox, loss_giou)
    """
    # how many Query tokens the model returned for given image
    n_queries = pred_boxes_cxcywh.size(0)

    # corner-case processing: there are no boxes in the frame at all
    if tgt_boxes_cxcywh.numel() == 0:
        empty_target = torch.full((n_queries,), EMPTY_CLASS, device=pred_logits.device, dtype=torch.long)
        loss_class = F.cross_entropy(pred_logits, empty_target)
        loss_bbox  = torch.tensor(0.0, device=pred_logits.device)
        loss_giou  = torch.tensor(0.0, device=pred_logits.device)
        return loss_class, loss_bbox, loss_giou

    #  a normal case when there is at least one target
    # hungarian‑mapping
    C_total = _cost_matrix(pred_boxes_cxcywh, tgt_boxes_cxcywh, pred_logits, tgt_labels)
    pred_ixs, tgt_ixs = _hungarian_match(C_total)

    #  reorder the predictions in the same order as the targets
    pred_ixs = pred_ixs[tgt_ixs.argsort()]

    # L1‑loss (box loss) by selected pairs
    num_boxes   = tgt_boxes_cxcywh.size(0)
    loss_bbox   = F.l1_loss(pred_boxes_cxcywh[pred_ixs], tgt_boxes_cxcywh, reduction='sum') / num_boxes

    # gIoU‑loss by selected pairs (same pairs)
    giou = ops.generalized_box_iou(
        ops.box_convert(pred_boxes_cxcywh[pred_ixs], in_fmt='cxcywh', out_fmt='xyxy'),
        ops.box_convert(tgt_boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')
    )
    loss_giou = 1.0 - torch.diag(giou).mean()

    # cross‑entropy by classes
    # we assign an "empty" class to all unmatched Query
    labels_for_ce = torch.full((n_queries,), EMPTY_CLASS, device=pred_logits.device, dtype=torch.long)
    labels_for_ce[pred_ixs] = tgt_labels
    loss_class = F.cross_entropy(pred_logits, labels_for_ce)

    return loss_class, loss_bbox, loss_giou


# loss wrapper for batch
def detr_loss_batch(class_logits: torch.Tensor,
                    pred_boxes: torch.Tensor,
                    targets: list[dict]) -> torch.Tensor:
    """
    *) class_logits – (B, Q, C+1)
    *) pred_boxes – (B, Q, 4)
    *) targets – dict list {'boxes': Tensor(N_i,4), 'labels': Tensor(N_i)}
    """
    B, Q, _ = class_logits.shape
    total_loss = torch.tensor(0.0, device=class_logits.device)

    for b in range(B):
        # transform boxes into normalized cxcywh
        tgt_boxes_cxcywh = _abs_boxes_xyxy_to_norm_cxcywh(targets[b]['boxes'].to(class_logits.device))
        tgt_labels       = targets[b]['labels'].to(class_logits.device)   # (N_i,)

        # losses for one example
        l_cls, l_l1, l_giou = _loss_per_sample(
            pred_boxes[b], tgt_boxes_cxcywh, class_logits[b], tgt_labels
        )
        # total COST / LOSS
        total_loss += W_CLASS * l_cls + W_L1 * l_l1 + W_GIOU * l_giou

    # average by batch
    return total_loss / B