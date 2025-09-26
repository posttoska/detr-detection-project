import torch
import math
import torch.nn as nn
import torchvision.models
from torch.nn import functional as F
from torchvision.models import resnet34
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def get_spatial_position_embeddings(embed_size: int, conv_out_tensor: torch.Tensor):
    assert embed_size % 4 == 0, ('Position embedding dimension must be divisible by 4')

    grid_size_h, grid_size_w = conv_out_tensor.shape[-2], conv_out_tensor.shape[-1]
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=conv_out_tensor.device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=conv_out_tensor.device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    factor = torch.arange(
        start=0,
        end=embed_size // 4,
        dtype=torch.float32,
        device=conv_out_tensor.device
    )

    factor /= (embed_size // 4)
    factor = 10000 ** factor
    vert_h_pos = grid_h_positions[:, None]
    vert_h_pos_extruded = vert_h_pos.repeat(1, embed_size // 4)
    grid_h_emb = vert_h_pos_extruded / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)
    horz_w_pos = grid_w_positions[:, None]
    horz_w_pos_extruded = horz_w_pos.repeat(1, embed_size // 4)
    grid_w_emb = horz_w_pos_extruded / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)
    pos_embeded = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    return pos_embeded


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.in_proj_q = nn.Linear(d_model, d_model)
        self.in_proj_k = nn.Linear(d_model, d_model)
        self.in_proj_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_head = d_model // num_heads

    def forward(self, q, k, v):

        Qb, Ql, Qe = q.shape
        Kb, Kl, Ke = k.shape
        Vb, Vl, Ve = v.shape
        broadcast_shape_Q = (Qb, Ql, self.num_heads, self.d_head)
        broadcast_shape_K = (Kb, Kl, self.num_heads, self.d_head)
        broadcast_shape_V = (Vb, Vl, self.num_heads, self.d_head)
        q = self.in_proj_q(q)
        k = self.in_proj_k(k)
        v = self.in_proj_v(v)

        q = q.view(broadcast_shape_Q)
        k = k.view(broadcast_shape_K)
        v = v.view(broadcast_shape_V)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        qk = q @ k.transpose(2, 3)
        qk /= math.sqrt(self.d_head)
        
        # we don't need mask for image classification at all
        # mask = torch.ones_like(qk, dtype=torch.bool).triu(1)
        # qk.masked_fill_(mask, -torch.inf)

        qk = F.softmax(qk, dim=-1)
        out = qk @ v
        att_map = qk.mean(dim=1)
        out = out.transpose(1, 2)
        out = out.reshape(Qb, Ql, Qe)
        out = self.out_proj(out)

        return out, att_map


class TransformerEncoder(nn.Module):
    r"""
    Encoder for transformer of DETR.
    This has sequence of encoder layers.
    Each layer has the following modules.
        1. LayerNorm for Self Attention.
        2. Self Attention.
        3. LayerNorm for MLP.
        4. MLP.
    """
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim, dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.attns = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])

        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Linear(ff_inner_dim, d_model),
                ) 
                for _ in range(num_layers)
            ])

        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(num_layers)])
        self.ff_dropouts = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, spatial_pos_embed):
        """
            -----> X TENSOR: (B, seq_len=400, d_model=256)
        """
        out = x
        attn_weights = []

        for i in range(self.num_layers):
            in_attn = self.attn_norms[i](out)

            q = in_attn + spatial_pos_embed
            k = in_attn + spatial_pos_embed
            v = in_attn

            out_attn, attn_weight = self.attns[i](q=q, k=k, v=v)
            attn_weights.append(attn_weight)

            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff

        out = self.output_norm(out)
        return out, torch.stack(attn_weights)


class TransformerDecoder(nn.Module):
    r"""
    Decoder for transformer of DETR.
    This has sequence of decoder layers.
    Each layer has the following modules.
        1. LayerNorm for Self Attention.
        2. Self Attention.
        3. LayerNorm for Cross Attention on encoder outputs.
        4. Cross Attention.
        5. LayerNorm for MLP.
        6. MLP.
    """
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim, dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.attns = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])
        self.cross_attns = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])

        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Linear(ff_inner_dim, d_model),
                ) 
                for _ in range(num_layers)
            ])

        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.attn_dropouts = nn.ModuleList(nn.Dropout(self.dropout_prob) for _ in range(num_layers))
        self.cross_attn_dropouts = nn.ModuleList(nn.Dropout(self.dropout_prob) for _ in range(num_layers))
        self.ff_dropouts = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query_objects, encoder_output, query_embed, spatial_pos_embed):
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []

        for i in range(self.num_layers):
            in_attn = self.attn_norms[i](out)
            q = in_attn + query_embed
            k = in_attn + query_embed
            v = in_attn

            out_attn, _ = self.attns[i](q=q, k=k, v=v)
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn
            in_attn = self.cross_attn_norms[i](out)

            q = in_attn + query_embed
            k = encoder_output + spatial_pos_embed
            v = encoder_output

            out_attn, decoder_cross_attn = self.cross_attns[i](q=q, k=k, v=v)

            decoder_cross_attn_weights.append(decoder_cross_attn)
            out_attn = self.cross_attn_dropouts[i](out_attn)
            out = out + out_attn

            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff
            decoder_outputs.append(self.output_norm(out))

        decoder_outputs = torch.stack(decoder_outputs)
        decoder_cross_attn_weights = torch.stack(decoder_cross_attn_weights)
        return decoder_outputs, decoder_cross_attn_weights


class DETR(nn.Module):
    r"""
    DETR MODEL DIMENSIONS:
    INPUT TENSOR: (B, c=3, h=640, w=640)
    """
    def __init__(self, config, num_classes, bg_class_idx):
        super().__init__()
        self.img_h = config['image_h']
        self.img_w = config['image_w']
        self.backbone_channels = config['backbone_channels']
        self.d_model = config['d_model']
        self.num_queries = config['num_queries']
        self.num_classes = num_classes
        self.num_encoder_layers = config['encoder_layers']
        self.num_decoder_layers = config['decoder_layers']
        self.num_encoder_heads = config['encoder_attn_heads']
        self.num_decoder_heads = config['decoder_attn_heads']
        self.cls_cost_weight = config['cls_cost_weight']
        self.l1_cost_weight = config['l1_cost_weight']
        self.giou_cost_weight = config['giou_cost_weight']
        self.bg_cls_weight = config['bg_class_weight']
        self.nms_threshold = config['nms_threshold']
        self.dropout_prob = config['dropout_prob']
        self.bg_class_idx = bg_class_idx
        self.ff_inner_dim = config['ff_inner_dim']
        valid_bg_idx = (self.bg_class_idx == 0 or self.bg_class_idx == (self.num_classes - 1))
        assert valid_bg_idx, "Background can only be 0 or num_classes - 1"

        self.seq_len = ((self.img_h + 31) // 32) * ((self.img_w + 31) // 32)

        self.backbone = nn.Sequential(*list(resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
            norm_layer=torchvision.ops.FrozenBatchNorm2d
        ).children())[:-2])

        if config['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone_proj = nn.Conv2d(self.backbone_channels, self.d_model, kernel_size=1)

        self.encoder = TransformerEncoder(num_layers=self.num_encoder_layers, 
                                          num_heads=self.num_encoder_heads, 
                                          d_model=self.d_model, 
                                          ff_inner_dim=self.ff_inner_dim, 
                                          dropout_prob=self.dropout_prob)

        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.d_model))

        self.decoder = TransformerDecoder(num_layers=self.num_decoder_layers, 
                                          num_heads=self.num_decoder_heads, 
                                          d_model=self.d_model,
                                          ff_inner_dim=self.ff_inner_dim, 
                                          dropout_prob=self.dropout_prob)
        
        self.class_mlp = nn.Linear(self.d_model, self.num_classes)

        self.bbox_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, out_features=4),
        )
    
    def forward(self, x, targets=None, score_thresh=0, use_nms=False):
        # x -> (b, ch, h, w)
        resnet_out = self.backbone(x)
        conv_out = self.backbone_proj(resnet_out)
        batch_size, d_model, feat_h, feat_w = conv_out.shape
        spatial_pos_embed = get_spatial_position_embeddings(d_model, conv_out)
        conv_out = conv_out.reshape(batch_size, d_model, feat_h * feat_w)
        conv_out = conv_out.transpose(1, 2)

        enc_output, enc_att_weights = self.encoder(conv_out, spatial_pos_embed)

        query_reshaped = self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1))
        query_objects_zeros = torch.zeros_like(query_reshaped)
        query_objects, decoder_attn_weights = self.decoder(query_objects_zeros, enc_output, query_reshaped, spatial_pos_embed)

        cls_output = self.class_mlp(query_objects)
        bbox_output = self.bbox_mlp(query_objects).sigmoid()

        losses = defaultdict(list)
        detections = []
        detr_output = {}

        if self.training:
            num_decoder_layers = self.num_decoder_layers

            for decoder_idx in range(num_decoder_layers):
                cls_idx_output = cls_output[decoder_idx]
                bbox_idx_output = bbox_output[decoder_idx]

                with torch.no_grad():
                    class_prob_tns = cls_idx_output.reshape((-1, self.num_classes))
                    class_prob_tns = class_prob_tns.softmax(dim=-1)
                    pred_boxes_tns = bbox_idx_output.reshape((-1, 4))
                    target_labels = torch.cat([target["labels"] for target in targets])
                    target_boxes = torch.cat([target["boxes"] for target in targets])

                    COST_CLS_REDUCED_TNS = -class_prob_tns[:, target_labels]
                    pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(pred_boxes_tns, 'cxcywh', 'xyxy')

                    COST_L1_REDUCED_TNS = torch.cdist(pred_boxes_x1y1x2y2, target_boxes, p=1)
                    COST_GIOU_REDUCED_TNS = -torchvision.ops.generalized_box_iou(pred_boxes_x1y1x2y2, target_boxes)
                    COST_TNS = (self.cls_cost_weight * COST_CLS_REDUCED_TNS + self.l1_cost_weight * COST_L1_REDUCED_TNS + self.giou_cost_weight * COST_GIOU_REDUCED_TNS)
                    COST_TNS = COST_TNS.reshape(batch_size, self.num_queries, -1).cpu()

                    num_targets_per_image = [len(target["labels"]) for target in targets]
                    ITO_COST_TUPLE =  COST_TNS.split(num_targets_per_image, dim=-1)

                    match_indices = []
                    for batch_idx in range(batch_size):
                        DCT = ITO_COST_TUPLE[batch_idx][batch_idx]
                        batch_idx_assignments = linear_sum_assignment(DCT)
                        batch_idx_pred, batch_idx_target = batch_idx_assignments
                        match_indices.append((torch.as_tensor(batch_idx_pred, dtype=torch.int64), torch.as_tensor(batch_idx_target, dtype=torch.int64)))

                pred_batch_idxs = torch.cat([torch.ones_like(pred_idx) * i for i, (pred_idx, _) in enumerate(match_indices)])
                pred_query_idx = torch.cat([pred_idx for (pred_idx, _) in match_indices])

                valid_obj_target_cls = torch.cat([target["labels"][target_obj_idx] for target, (_, target_obj_idx) in zip(targets, match_indices)])
                target_classes = torch.full(cls_idx_output.shape[:2], fill_value=self.bg_class_idx, dtype=torch.int64,device=cls_idx_output.device)
                target_classes[(pred_batch_idxs, pred_query_idx)] = valid_obj_target_cls


                cls_weights = torch.ones(self.num_classes)
                cls_weights[self.bg_class_idx] = self.bg_cls_weight
                loss_cls = torch.nn.functional.cross_entropy(cls_idx_output.reshape(-1, self.num_classes), target_classes.reshape(-1), cls_weights.to(cls_idx_output.device))

                matched_pred_boxes = bbox_idx_output[pred_batch_idxs, pred_query_idx]
                target_boxes = torch.cat([
                    target['boxes'][target_obj_idx]
                    for target, (_, target_obj_idx) in zip(targets, match_indices)],
                    dim=0
                )
                matched_pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(matched_pred_boxes,'cxcywh','xyxy')
                loss_bbox = torch.nn.functional.l1_loss(matched_pred_boxes_x1y1x2y2, target_boxes, reduction='none')
                loss_bbox = loss_bbox.sum() / matched_pred_boxes.shape[0]

                loss_giou = torchvision.ops.generalized_box_iou_loss(matched_pred_boxes_x1y1x2y2, target_boxes)
                loss_giou = loss_giou.sum() / matched_pred_boxes.shape[0]

                # losses
                losses['classification'].append(loss_cls * self.cls_cost_weight)
                losses['bbox_regression'].append(loss_bbox * self.l1_cost_weight + loss_giou * self.giou_cost_weight)

            detr_output['loss'] = losses

        else:
            # for inference we are only interested in last layer outputs
            cls_output = cls_output[-1]
            bbox_output = bbox_output[-1]
            prob = torch.nn.functional.softmax(cls_output, -1)

            if self.bg_class_idx == 0:
                scores, labels = prob[..., 1:].max(-1)
                labels = labels + 1
            else:
                scores, labels = prob[..., :-1].max(-1)

            boxes = torchvision.ops.box_convert(bbox_output,'cxcywh','xyxy')

            for batch_idx in range(boxes.shape[0]):
                scores_idx = scores[batch_idx]
                labels_idx = labels[batch_idx]
                boxes_idx = boxes[batch_idx]

                keep_idxs = scores_idx >= score_thresh

                scores_idx = scores_idx[keep_idxs]
                boxes_idx = boxes_idx[keep_idxs]
                labels_idx = labels_idx[keep_idxs]

                if use_nms:
                    keep_idxs = torchvision.ops.batched_nms(boxes_idx, scores_idx, labels_idx, iou_threshold=self.nms_threshold)
                    scores_idx = scores_idx[keep_idxs]
                    boxes_idx = boxes_idx[keep_idxs]
                    labels_idx = labels_idx[keep_idxs]

                detections.append(
                    {
                        "boxes": boxes_idx,
                        "scores": scores_idx,
                        "labels": labels_idx
                        ,
                    }
                )

            detr_output['detections'] = detections
            detr_output['enc_attn'] = enc_att_weights
            detr_output['dec_attn'] = decoder_attn_weights

        return detr_output