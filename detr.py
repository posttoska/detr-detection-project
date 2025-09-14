import torch
import math
import torch.nn as nn
import torchvision.models
from torch.nn import functional as F
from torchvision.models import resnet34
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_head = d_model // num_heads
    
    # q parameter is for cross attn
    def forward(self, x, q_cross=nn.Linear(0.0, 0.0), isCrossAtt=False):
        # x: (batch, sequence, d_model)
        input_shape = x.shape
        batch, seq_len, d_model = input_shape
        broadcast_shape = (batch, seq_len, self.num_heads, d_model)

        # (b, seq_len, d_model)
        x = self.in_proj(x)
        
        # 3 * (b, seq_len, d_model)
        q, k, v = x.chunk(3, dim=-1)

        # cross attention use case
        if isCrossAtt:
            q = q_cross
            is_dim = (q_cross.shape == k.shape) and (q_cross.shape == v.shape)
            assert is_dim, "cross q must be the same dim as k and v"


        # d_model/num_heads = d_heads
        # (b, num_heads, seq_len, d_heads)
        q = q.view(broadcast_shape).transpose(1, 2)
        k = k.view(broadcast_shape).transpose(1, 2)
        v = v.view(broadcast_shape).transpose(1, 2)

        # (b, num_heads, seq_len, seq_len)
        qk = q @ k.transpose(1, 2)
        qk /= math.sqrt(self.d_head)
        
        # we don't need mask for image classification at all
        # mask = torch.ones_like(qk, dtype=torch.bool).triu(1)
        # qk.masked_fill_(mask, -torch.inf)

        # (b, num_heads, seq_len, seq_len)
        qk = F.softmax(qk, dim=-1)
        # (b, num_heads, seq_len, d_heads)
        out = qk @ v
        # (b, seq_len, num_heads, d_heads)
        out = out.transpose(1, 2)

        # (b, seq_len, d_model)
        out = self.out_proj(out)

        return out


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

        # self attention module for all encoder layers
        self.attns = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])

        # MLP module for all encoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    # (b, seq_len, d_model)
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    # (b, seq_len, d_model)
                    nn.Linear(ff_inner_dim, d_model),
                ) 
                for _ in range(num_layers)
            ])

        # norm for MHSA for all encoder layers
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # norm for MLP for all encoder layers
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # dropout for self attention for all encoder layers
        self.attn_dropouts = nn.Module(nn.Dropout(self.dropout_prob) for _ in range(num_layers))
        
        # dropout for feed forward for all encoder layers
        self.ff_dropouts = nn.Module(nn.Dropout(self.dropout_prob for _ in range(num_layers)))

        # norm for encoder output for all encoder outputs
        self.output_norm = nn.LayerNorm(d_model)


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

        # self attention module for all decoder layers
        self.attns = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])

        # cross attention module for all decoder layers
        self.cross_attns = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])

        # MLP module for all decoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    # (b, seq_len, d_model)
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    # (b, seq_len, d_model)
                    nn.Linear(ff_inner_dim, d_model),
                ) 
                for _ in range(num_layers)
            ])
        
        # norm for MHSA for all decoder layers
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # norm for MHCA for all decoder layers
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # norm for MLP for all decoder layers
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # dropout for self attention for all decoder layers
        self.attn_dropouts = nn.Module(nn.Dropout(self.dropout_prob) for _ in range(num_layers))
        
        # dropout for cross attention for all decoder layers
        self.cross_attn_dropouts = nn.Module(nn.Dropout(self.dropout_prob) for _ in range(num_layers))
        
        # dropout for feed forward for all decoder layers
        self.ff_dropouts = nn.Module(nn.Dropout(self.dropout_prob for _ in range(num_layers)))

        # norm for decoder output for all decoder outputs
        self.output_norm = nn.LayerNorm(d_model)


class DETR(nn.Module):
    r"""
    DETR model class which instantiates all layers of DETR.
    A forward pass goes through the following layers:
        1. Backbone Call (currently frozen resnet 34).
        2. Backbone Featuremap Projection to d_model (emb_size) of transformer.
        3. Encoder of Transformer.
        4. Decoder of Transformer.
        5. Class and BBox MLPs.
    """
    def __init__(self, config, num_classes, bg_class_idx):
        super().__init__()
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

        self.encoder = TransformerDecoder(num_layers=self.num_decoder_layers, 
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