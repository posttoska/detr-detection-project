import torch
import math
import torch.nn as nn
import torchvision.models
from torch.nn import functional as F
from torchvision.models import resnet34
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def get_spatial_position_embeddings(embed_size: int, conv_out_tensor: torch.Tensor):
    
    # input: shape=(b, d_model, feat_h, feat_w)
    assert embed_size % 4 == 0, ('Position embedding dimension must be divisible by 4')
    
    # get last 2 dims len (20 and 20)
    grid_size_h, grid_size_w = conv_out_tensor.shape[-2], conv_out_tensor.shape[-1]
    
    # create tensors shape=(20)
    # tensors have increasing numbers
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=conv_out_tensor.device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=conv_out_tensor.device)

    # 2 tensor tuple, each tensor size=(20, 20)
    # first tensor has 20 rows filled with same nums (from 0 to 19 rowwise)
    # second tensor each row reapiting nums (from 0 to 19 inside each row)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    
    # concat it to one tensor shape=(2, 20, 20)
    grid = torch.stack(grid, dim=0)

    # flattening: grid_h_positions shape=(number_of_grid_cell_tokens=400)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # create factor vector with increasing nums (from 0 to (d_model/4)-1=63)
    # shape=(d_model/4=64)
    factor = torch.arange(
        start=0,
        end=embed_size // 4,
        dtype=torch.float32,
        device=conv_out_tensor.device
    )
    
    # get fractions (normalize)
    factor /= (embed_size // 4)

    # pos emb formula (first part): factor = 10000^(2i/d_model)
    # get increasing number vector shape=(64)
    factor = 10000 ** factor

    # create vertical vector that has 0-19 ints 20 times
    # shape=(400, 1)
    vert_h_pos = grid_h_positions[:, None]

    # extrude columnwise by length d_model // 4 = 64
    # shape=(seq_len=400, d_model/4=64)
    vert_h_pos_extruded = vert_h_pos.repeat(1, embed_size // 4)

    # grid hight embedding shape=(seq_len=400, d_model/4=64)
    # along vertical axis we have same vectors reapiting 20 times (representing 0 to 20 feature_h embedding)
    # along horizontal axis we have deacrising values (len 64)
    grid_h_emb = vert_h_pos_extruded / factor

    # concat them from the side (so one side is sin, second cos)
    # shape shape=(seq_len=400, d_model/2=128)
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)

    # create vertical vector that has 0, 1, 2 ... 19 ints 20 times
    # shape=(400, 1)
    horz_w_pos = grid_w_positions[:, None]

    # extrude columnwise by length d_model // 4 = 64
    # shape=(seq_len=400, d_model/4=64)
    horz_w_pos_extruded = horz_w_pos.repeat(1, embed_size // 4)

    # grid width embedding shape=(seq_len=400, d_model/4=64)
    # along vertical axis we have same sets of vectors reapiting 20 times and inside this set we have 0...19 vectors where each vector contains same nums (before factoring)
    # along horizontal axis we have deacrising values (len 64)
    grid_w_emb = horz_w_pos_extruded / factor

    # concat them from the side (so one side is sin, second cos)
    # shape shape=(seq_len=400, d_model/2=128)
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)

    # final concat where we again concat matricies from the side
    # output shape=(seq_len=400, d_model=256)
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
    
    # q parameter is for cross attn
    def forward(self, q, k, v):
        """
            SELF ATTN CASE:
                Q, K, V TENSORS: (B, seq_len=400, d_model=256)
            CROSS ATTN CASE:
                Q TENSOR:        (B, qemb=25, d_model=256)
                K, V TENSORS:    (B, seq_len=400, d_model=256)
        """
        Qb, Ql, Qe = q.shape
        Kb, Kl, Ke = k.shape
        Vb, Vl, Ve = v.shape
        broadcast_shape_Q = (Qb, Ql, self.num_heads, self.d_head)
        broadcast_shape_K = (Kb, Kl, self.num_heads, self.d_head)
        broadcast_shape_V = (Vb, Vl, self.num_heads, self.d_head)
        q = self.in_proj_q(q)
        k = self.in_proj_k(k)
        v = self.in_proj_v(v)

        """
            SELF ATTN CASE:
                -----> INPUT Q, K, V TENSORS: (B, seq_len=400, d_model=256)
                -----> OUTPUT TENSOR: (B, seq_len=400, h_num=8, d_head=32)
            CROSS ATTN CASE:
                -----> INPUT Q     TENSOR:  (B, qemb=25, d_model=256)
                -----> INPUT K, V  TENSORS: (B, seq_len=400, d_model=256)
                -----> OUTPUT Q    TENSOR:  (B, qemb=25, h_num=8, d_head=32)
                -----> OUTPUT K, V TENSOR:  (B, seq_len=400, h_num=8, d_head=32)
        """
        q = q.view(broadcast_shape_Q)
        k = k.view(broadcast_shape_K)
        v = v.view(broadcast_shape_V)

        """
            SELF ATTN CASE:
                -----> INPUT Q, K, V TENSORS: (B, seq_len=400, h_num=8, d_head=32)
                -----> OUTPUT TENSOR: (B, h_num=8, seq_len=400, d_head=32)
            CROSS ATTN CASE:
                -----> INPUT Q     TENSOR:  (B, qemb=25, h_num=8, d_head=32)
                -----> INPUT K, V  TENSORS: (B, seq_len=400, h_num=8, d_head=32)
                -----> OUTPUT Q    TENSOR:  (B, h_num=8, qemb=25,  d_head=32)
                -----> OUTPUT K, V TENSOR:  (B, h_num=8, seq_len=400, d_head=32)
        """
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        """
            SELF ATTN CASE:
                -----> INPUT Q@K.T:        (B, h_num=8, seq_len=400, d_head=32) @ (B, h_num=8, d_head=32, seq_len=400)
                -----> OUTPUT ATTN TENSOR: (B, h_num=8, seq_len=400, seq_len=400)
            CROSS ATTN CASE:
                -----> INPUT Q@K.T:        (B, h_num=8, qemb=25, d_head=32) @ (B, h_num=8, d_head=32, seq_len=400)
                -----> OUTPUT ATTN TENSOR: (B, h_num=8, qemb=25, seq_len=400)
        """
        qk = q @ k.transpose(2, 3)
        qk /= math.sqrt(self.d_head)
        
        # we don't need mask for image classification at all
        # mask = torch.ones_like(qk, dtype=torch.bool).triu(1)
        # qk.masked_fill_(mask, -torch.inf)

        qk = F.softmax(qk, dim=-1)
        """
            SELF ATTN CASE:
                -----> INPUT ATTN@V:  (B, h_num=8, seq_len=400, seq_len=400) @ (B, h_num=8, seq_len=400, d_head=32)
                -----> ATTN TENSOR:   (B, h_num=8, seq_len=400, d_head=32)
            CROSS ATTN CASE:
                -----> INPUT ATTN@V:  (B, h_num=8, qemb=25, seq_len=400) @ (B, h_num=8, seq_len=400, d_head=32)
                -----> OUTPUT TENSOR: (B, h_num=8, qemb=25, d_head=32)
        """
        out = qk @ v

        """
            SELF ATTN CASE:
                -----> INPUT ATTN MAP TENSOR:   (B, h_num=8, seq_len=400, seq_len=400)
                -----> OUTPUT ATTN MAP TENSOR:  (B, seq_len=400, seq_len=400)
            CROSS ATTN CASE:
                -----> INPUT OUT TENSOR:  (B, h_num=8, qemb=25, seq_len=400)
                -----> OUTPUT TENSOR:     (B, qemb=25, seq_len=400)
        """
        att_map = qk.mean(dim=1)

        """
            SELF ATTN CASE:
                -----> INPUT OUT TENSOR:  (B, h_num=8, seq_len=400, d_head=32)
                -----> OUTPUT OUT TENSOR: (B, seq_len=400, h_num=8, d_head=32)
            CROSS ATTN CASE:
                -----> INPUT OUT TENSOR:  (B, h_num=8, qemb=25, d_head=32)
                -----> OUTPUT TENSOR:     (B, qemb=25, h_num=8, d_head=32)
        """
        out = out.transpose(1, 2)

        """
            SELF ATTN CASE:
                -----> INPUT OUT TENSOR:  (B, seq_len=400, h_num=8, d_head=32)
                -----> OUTPUT OUT TENSOR: (B, seq_len=400, d_model=256)
            CROSS ATTN CASE:
                -----> INPUT OUT TENSOR:  (B, qemb=25, h_num=8, d_head=32)
                -----> OUTPUT TENSOR:     (B, qemb=25, d_model=256)
        """
        out = out.reshape(Qb, Ql, Qe)

        # (b, seq_len, d_model)
        out = self.out_proj(out)

        """
            SELF ATTN CASE:
                -----> OUTPUT ATTN TENSOR:       (B, seq_len=400, d_model=256)
                -----> OUTPUT ATTN MAP TENSOR:   (B, seq_len=400, seq_len=400)
            CROSS ATTN CASE:
                -----> OUTPUT ATTN TENSOR:       (B, qemb=25, d_model=256)
                -----> OUTPUT ATTN MAP TENSOR:   (B, qemb=25, seq_len=400)
        """
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

        # self attention module for all encoder layers
        self.attns = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])

        # MLP module for all encoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
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
        self.attn_dropouts = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(num_layers)])
        
        # dropout for feed forward for all encoder layers
        self.ff_dropouts = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(num_layers)])

        # norm for encoder output for all encoder outputs
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, spatial_pos_embed):
        """
            -----> X TENSOR: (B, seq_len=400, d_model=256)
        """
        out = x
        attn_weights = []

        # go through all encoder layers
        for i in range(self.num_layers):
            # norm MHSA
            in_attn = self.attn_norms[i](out)

            """
                -----> Q, K, V TENSORS: (B, seq_len=400, d_model=256)
            """
            # add spacial position embeddings to q and k for MHSA
            q = in_attn + spatial_pos_embed
            k = in_attn + spatial_pos_embed
            v = in_attn

            """
                 -----> INPUT Q, K, V TENSORS: (B, seq_len=400, d_model=256) 
                 -----> OUTPUT TENSOR:         (B, seq_len=400, d_model=256)
            """
            out_attn, attn_weight = self.attns[i](q=q, k=k, v=v)
            attn_weights.append(attn_weight)

            # dropout MHSA
            out_attn = self.attn_dropouts[i](out_attn)

            # residual connection MHSA
            out += out_attn

            # norm MLP
            in_ff = self.ff_norms[i](out)

            """
                 -----> INPUT Q, K, V TENSORS: (B, seq_len=400, d_model=256) 
                 -----> OUTPUT TENSOR:         (B, seq_len=400, d_model=256)
            """
            out_ff = self.ffs[i](in_ff)

            # dropout MLP
            out_ff = self.ff_dropouts[i](out_ff)

            # residual connection MLP
            out += out_ff

        # output norn
        out = self.output_norm(out)
        """
            -----> OUTPUT ENC TENSOR: (B, seq_len=400, d_model=256)
            -----> OUTPUT ATTN MAP TENSOR: (layers=4, B, seq_len=400, seq_len=400)
        """
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
        self.attn_dropouts = nn.ModuleList(nn.Dropout(self.dropout_prob) for _ in range(num_layers))
        
        # dropout for cross attention for all decoder layers
        self.cross_attn_dropouts = nn.ModuleList(nn.Dropout(self.dropout_prob) for _ in range(num_layers))
        
        # dropout for feed forward for all decoder layers
        self.ff_dropouts = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(num_layers)])

        # norm for decoder output for all decoder outputs
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query_objects, encoder_output, query_embed, spatial_pos_embed):
        """
            -----> ZERO QEMB INPUT TENSOR: (B, qemb=25, d_model=256)
            -----> INPUT ENC TENSOR:       (B, seq_len=400, d_model=256)
            -----> INPUT QEMB TENSOR:      (B, qemb=25, d_model=256)
            -----> INPUT POS TENSOR:       (seq_len=400, d_model=256)
        """
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []

        # go through all encoder layers
        for i in range(self.num_layers):

            # norm MHSA
            in_attn = self.attn_norms[i](out)

            """
                -----> Q, K, V TENSORS: (B, qemb=25, d_model=256)
            """
            q = in_attn + query_embed
            k = in_attn + query_embed
            v = in_attn

            """
                 -----> INPUT Q, K, V TENSORS: (B, qemb=25, d_model=256)
                 -----> OUTPUT TENSOR:         (B, qemb=25, d_model=256)
            """
            out_attn, _ = self.attns[i](q=q, k=k, v=v)
            
            # dropout MHSA
            out_attn = self.attn_dropouts[i](out_attn)

            # residual connection MHSA
            out += out_attn

            # norm MHCA
            in_attn = self.cross_attn_norms[i](out)

            # add query embeddings to q and spatial pos embedding to k for MHCA
            # where v will cross with encoder output
            # q shape=(b, query_embed=25, d_model=256)
            # k shape=(b, seq_len=400, d_model=256)
            # v shape=(b, seq_len=400, d_model=256)
            """
                 -----> Q TENSOR:     (B, qemb=25, d_model=256)
                 -----> K, V TENSORS: (B, seq_len=400, d_model=256)
            """
            q = in_attn + query_embed
            k = encoder_output + spatial_pos_embed
            v = encoder_output

            """
                 -----> INPUT Q TENSOR:                    (B, qemb=25, d_model=256)
                 -----> INPUT K, V TENSORS:                (B, seq_len=400, d_model=256)
                 -----> OUTPUT DCR TENSOR:                 (B, qemb=25, d_model=256)
                 -----> OUTPUT CROSS ATTN MAP TENSOR:      (B, qemb=25, seq_len=400)
            """
            out_attn, decoder_cross_attn = self.cross_attns[i](q=q, k=k, v=v)

            decoder_cross_attn_weights.append(decoder_cross_attn)
            out_attn = self.cross_attn_dropouts[i](out_attn)

            out += out_attn

            # norm MLP
            in_ff = self.ff_norms[i](out)

            """
                 -----> INPUT Q, K, V TENSORS: (B, qemb=25, d_model=256)
                 -----> OUTPUT TENSOR:         (B, qemb=25, d_model=256)
            """
            out_ff = self.ffs[i](in_ff)

            # dropout MLP
            out_ff = self.ff_dropouts[i](out_ff)

            # residual connection MLP
            out += out_ff

            # append
            decoder_outputs.append(self.output_norm(out))

        """
             -----> INPUT DCR TENSOR:                  (B, qemb=25, d_model=256) 
             -----> OUTPUT STACKED DCR TENSOR:         (layers=4, B, qemb=25, d_model=256)
             -----> INPUT CROSS ATTN TENSOR:           (B, qemb=25, d_model=256) 
             -----> OUTPUT STACKED CROSS ATTN TENSOR:  (layers=4, B, qemb=25, seq_len=400)
        """
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

        # formula (default torchvision.models.resnet34)
        # for an input (1, 3, H, W), the spatial size after layer4
        # (i.e., before the global avg‑pool)
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
        # default d_model = 256
        # default c = 3
        # default h, w = 640, 640
        # default feat_h, feat_w = 20, 20
        # default c_back = 512
        # resnet_stride = 32

        """
            op_formula = [([width+2*pad]-(kernel-1))/stride]*[([height+2*pad]-(kernel-1))/stride]

            BACKBONE resnet 1 conv layer: [([640+2*3]-(7-1))/2]*[([640+2*3]-(7-1))/2] = 102400 op (320x320)x64c
            BACKBONE resnet 1 pool layer: [([320+2*1]-(3-1))/2]*[([320+2*1]-(3-1))/2] = 25600 op (160x160)x64c

            BACKBONE resnet 2 conv layer: [([160+2*1]-(3-1))/1]*[([160+2*1]-(3-1))/1] = 25600 op (160x160)x64c
            BACKBONE resnet 3 conv layer: [([160+2*1]-(3-1))/1]*[([160+2*1]-(3-1))/1] = 25600 op (160x160)x64c
            BACKBONE resnet 4 conv layer: [([160+2*1]-(3-1))/1]*[([160+2*1]-(3-1))/1] = 25600 op (160x160)x64c
            BACKBONE resnet 5 conv layer: [([160+2*1]-(3-1))/1]*[([160+2*1]-(3-1))/1] = 25600 op (160x160)x64c
            BACKBONE resnet 6 conv layer: [([160+2*1]-(3-1))/1]*[([160+2*1]-(3-1))/1] = 25600 op (160x160)x64c
            BACKBONE resnet 7 conv layer: [([160+2*1]-(3-1))/1]*[([160+2*1]-(3-1))/1] = 25600 op (160x160)x64c

            BACKBONE resnet 8 conv layer: [([160+2*1]-(3-1))/2]*[([160+2*1]-(3-1))/2] = 6400 op (80x80)x128c (downsampling /2)
            BACKBONE resnet 9 conv layer:  [([80+2*1]-(3-1))/1]*[([80+2*1]-(3-1))/1] = 6400 op (80x80)x128c
            BACKBONE resnet 10 conv layer: [([80+2*1]-(3-1))/1]*[([80+2*1]-(3-1))/1] = 6400 op (80x80)x128c
            BACKBONE resnet 11 conv layer: [([80+2*1]-(3-1))/1]*[([80+2*1]-(3-1))/1] = 6400 op (80x80)x128c
            BACKBONE resnet 12 conv layer: [([80+2*1]-(3-1))/1]*[([80+2*1]-(3-1))/1] = 6400 op (80x80)x128c
            BACKBONE resnet 13 conv layer: [([80+2*1]-(3-1))/1]*[([80+2*1]-(3-1))/1] = 6400 op (80x80)x128c
            BACKBONE resnet 14 conv layer: [([80+2*1]-(3-1))/1]*[([80+2*1]-(3-1))/1] = 6400 op (80x80)x128c
            BACKBONE resnet 15 conv layer: [([80+2*1]-(3-1))/1]*[([80+2*1]-(3-1))/1] = 6400 op (80x80)x128c

            BACKBONE resnet 16 conv layer: [([80+2*1]-(3-1))/2]*[([80+2*1]-(3-1))/2] = 1600 op (40x40)x256c (downsampling /2)
            BACKBONE resnet 17 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 18 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 19 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 20 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 21 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 22 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 23 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 24 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 25 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 26 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c
            BACKBONE resnet 27 conv layer: [([40+2*1]-(3-1))/1]*[([40+2*1]-(3-1))/1] = 1600 op (40x40)x256c

            BACKBONE resnet 28 conv layer: [([40+2*1]-(3-1))/2]*[([40+2*1]-(3-1))/2] = 400 op (20x20)x512c (downsampling /2)
            BACKBONE resnet 29 conv layer: [([20+2*1]-(3-1))/1]*[([20+2*1]-(3-1))/1] = 400 op (20x20)x512c
            BACKBONE resnet 30 conv layer: [([20+2*1]-(3-1))/1]*[([20+2*1]-(3-1))/1] = 400 op (20x20)x512c
            BACKBONE resnet 31 conv layer: [([20+2*1]-(3-1))/1]*[([20+2*1]-(3-1))/1] = 400 op (20x20)x512c
            BACKBONE resnet 32 conv layer: [([20+2*1]-(3-1))/1]*[([20+2*1]-(3-1))/1] = 400 op (20x20)x512c
            BACKBONE resnet 33 conv layer: [([20+2*1]-(3-1))/1]*[([20+2*1]-(3-1))/1] = 400 op (20x20)x512c (output layer in our case)





            -----------------------------------------------------------------------------------------------------------------
            -----> INPUT TENSOR: (B, c=3, h=640, w=640)
            BACKBONE resnet 1 conv layer: (B,  3, 640, 640)      <conv2d> (c=3, out_c=64, k=7x7, st=2, pd=3) -> (B, 64, 320, 320) [102400 op]
            BACKBONE resnet 1 pool layer: (B, 64, 320, 320)      <pool2d> (k=3x3, st=2, pd=1)                -> (B, 64, 160, 160) [25600 op]

            BACKBONE resnet 2 conv layer: (B, 64, 160, 160)      <conv2d> (c=64, out_c=64, k=3x3, st=1, pd=1) -> (B, 64, 320, 320) [25600 op]
            BACKBONE resnet 3 conv layer: (B, 64, 160, 160)      <conv2d> (c=64, out_c=64, k=3x3, st=1, pd=1) -> (B, 64, 320, 320) [25600 op]
            BACKBONE resnet 4 conv layer: (B, 64, 160, 160)      <conv2d> (c=64, out_c=64, k=3x3, st=1, pd=1) -> (B, 64, 320, 320) [25600 op]
            BACKBONE resnet 5 conv layer: (B, 64, 160, 160)      <conv2d> (c=64, out_c=64, k=3x3, st=1, pd=1) -> (B, 64, 320, 320) [25600 op]
            BACKBONE resnet 6 conv layer: (B, 64, 160, 160)      <conv2d> (c=64, out_c=64, k=3x3, st=1, pd=1) -> (B, 64, 320, 320) [25600 op]
            BACKBONE resnet 7 conv layer: (B, 64, 160, 160)      <conv2d> (c=64, out_c=64, k=3x3, st=1, pd=1) -> (B, 64, 320, 320) [25600 op]

            BACKBONE resnet 8 conv layer: (B, 64, 160, 160) <conv2d> (c=64, out_c=128, k=3x3, st=2, pd=1) -> (B, 128, 80, 80) [6400 op] (downsampling /2)
            BACKBONE resnet 9 conv layer: (B, 128, 80, 80) <conv2d> (c=128, out_c=128, k=3x3, st=1, pd=1) -> (B, 128, 80, 80) [6400 op]
            BACKBONE resnet 10 conv layer: (B, 128, 80, 80) <conv2d> (c=128, out_c=128, k=3x3, st=1, pd=1) -> (B, 128, 80, 80) [6400 op]
            BACKBONE resnet 11 conv layer: (B, 128, 80, 80) <conv2d> (c=128, out_c=128, k=3x3, st=1, pd=1) -> (B, 128, 80, 80) [6400 op]
            BACKBONE resnet 12 conv layer: (B, 128, 80, 80) <conv2d> (c=128, out_c=128, k=3x3, st=1, pd=1) -> (B, 128, 80, 80) [6400 op]
            BACKBONE resnet 13 conv layer: (B, 128, 80, 80) <conv2d> (c=128, out_c=128, k=3x3, st=1, pd=1) -> (B, 128, 80, 80) [6400 op]
            BACKBONE resnet 14 conv layer: (B, 128, 80, 80) <conv2d> (c=128, out_c=128, k=3x3, st=1, pd=1) -> (B, 128, 80, 80) [6400 op]
            BACKBONE resnet 15 conv layer: (B, 128, 80, 80) <conv2d> (c=128, out_c=128, k=3x3, st=1, pd=1) -> (B, 128, 80, 80) [6400 op]

            BACKBONE resnet 16 conv layer: (B, 128, 80, 80) <conv2d> (c=128, out_c=256, k=3x3, st=2, pd=1) -> (B, 256, 40, 40) [1600 op] (downsampling /2)
            BACKBONE resnet 17 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 18 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 19 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 20 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 21 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 22 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 23 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 24 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 25 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 26 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]
            BACKBONE resnet 27 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 128, 40, 40) [1600 op]

            BACKBONE resnet 28 conv layer: (B, 256, 40, 40) <conv2d> (c=256, out_c=512, k=3x3, st=2, pd=1) -> (B, 512, 20, 20) [400 op] (downsampling /2)
            BACKBONE resnet 29 conv layer: (B, 512, 20, 20) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 512, 20, 20) [400 op]
            BACKBONE resnet 30 conv layer: (B, 512, 20, 20) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 512, 20, 20) [400 op]
            BACKBONE resnet 31 conv layer: (B, 512, 20, 20) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 512, 20, 20) [400 op]
            BACKBONE resnet 32 conv layer: (B, 512, 20, 20) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 512, 20, 20) [400 op]
            BACKBONE resnet 33 conv layer: (B, 512, 20, 20) <conv2d> (c=256, out_c=256, k=3x3, st=1, pd=1) -> (B, 512, 20, 20) [400 op] (output layer in our case)

            -----> INPUT TENSOR: (B, c=3, h=640, w=640)
            -----> OUTPUT TENSOR: (B, feat_c=512, feat_h=20, feat_w=20)
        """
        resnet_out = self.backbone(x)

        """
            -----> INPUT TENSOR:  (B, feat_c=512,  feat_h=20, feat_w=20) 
            -----> OUTPUT TENSOR: (B, d_model=256, feat_h=20, feat_w=20)
        """
        conv_out = self.backbone_proj(resnet_out)

        """
            -----> INPUT TENSOR:  (B, d_model=256, feat_h=20, feat_w=20)
            -----> OUTPUT TENSOR: (B, seq_len=400, d_model=256)
        """
        batch_size, d_model, feat_h, feat_w = conv_out.shape
        spatial_pos_embed = get_spatial_position_embeddings(d_model, conv_out)

        """
            -----> INPUT TENSOR:  (B, d_model=256, feat_h=20, feat_w=20)
            -----> OUTPUT TENSOR: (B, d_model=256, seq_len=400)
        """
        conv_out = conv_out.reshape(batch_size, d_model, feat_h * feat_w)

        """
            -----> INPUT TENSOR:  (B, d_model=256, seq_len=400)
            -----> OUTPUT TENSOR: (B, seq_len=400, d_model=256)
        """
        conv_out = conv_out.transpose(1, 2)

        """
            -----> INPUT ENC TENSOR:   (B, seq_len=400, d_model=256)
            
            -----> OUTPUT ENC TENSOR:  (B, seq_len=400, d_model=256)
            -----> OUTPUT ATTN TENSOR: (layers=4, B, seq_len=400, seq_len=400)
        """
        enc_output, enc_att_weights = self.encoder(conv_out, spatial_pos_embed)

        """
            -----> INPUT TENSOR:  (qemb=25, d_model=256)
            -----> OUTPUT TENSOR: (B, qemb=25, d_model=256)
        """
        query_reshaped = self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1))
        # init new query objects all to zeros
        query_objects_zeros = torch.zeros_like(query_reshaped)

        """
            -----> ZERO QEMB INPUT TENSOR: (B, qemb=25, d_model=256)
            -----> INPUT ENC TENSOR:       (B, seq_len=400, d_model=256)
            -----> INPUT QEMB TENSOR:      (B, qemb=25, d_model=256)
            -----> INPUT POS TENSOR:       (seq_len=400, d_model=256)
            
            -----> OUTPUT QUERY TENSOR:       (layers=4, B, qemb=25, d_model=256)
            -----> OUTPUT CROSS ATTN TENSOR:  (layers=4, B, qemb=25, seq_len=400)
        """
        query_objects, decoder_attn_weights = self.decoder(query_objects_zeros, enc_output, query_reshaped, spatial_pos_embed)

        """
            -----> INPUT QUERY TENSOR:  (layers=4, B, qemb=25, d_model=256)
            -----> OUTPUT CLASS TENSOR: (layers=4, B, qemb=25, cls=21)
        """
        cls_output = self.class_mlp(query_objects)

        """
            -----> INPUT QUERY TENSOR:  (layers=4, B, qemb=25, d_model=256)
            -----> OUTPUT CLASS TENSOR: (layers=4, B, qemb=25, coord=4)
        """
        bbox_output = self.bbox_mlp(query_objects).sigmoid()

        losses = defaultdict(list)
        detections = []
        detr_output = {}

        # TRAINING
        if self.training:
            num_decoder_layers = self.num_decoder_layers

            # perform mathing for each decoder layer
            for decoder_idx in range(num_decoder_layers):
                """
                    -----> INPUT CLS IDX TENSOR:        (layers=4, B, qemb=25, cls=21)
                    -----> OUTPUT CLS IDX TENSOR:       (B, qemb=25, cls=21)
                """
                cls_idx_output = cls_output[decoder_idx]
                """
                    -----> INPUT BBOX INX CLASS TENSOR:     (layers=4, B, qemb=25, coord=4)
                    -----> OUTPUT BBOX INX CLASS TENSOR:    (B, qemb=25, coord=4)
                """
                bbox_idx_output = bbox_output[decoder_idx]

                with torch.no_grad():
                    """
                        -----> INPUT CLS PROB TENSOR:       (B, qemb=25, cls=21)
                        -----> OUTPUT CLS PROB TENSOR:      (B_qemb=25*B, cls=21)
                    """
                    class_prob_tns = cls_idx_output.reshape((-1, self.num_classes))
                    class_prob_tns = class_prob_tns.softmax(dim=-1)

                    """
                        TNS - tensor
                        -----> INPUT BBOX PROB TENSOR:      (B, qemb=25, coord=4)
                        -----> OUTPUT BBOX PROB TENSOR:     (B_qemb=25*B, coord=4)
                    """
                    pred_boxes_tns = bbox_idx_output.reshape((-1, 4))

                    """
                        BTO - batch target objects
                        -----> INPUT TARGETS               {B, (2, ITO)}
                        -----> OUTPUT BATCH TARGETS:       (BTO)
                        -----> OUTPUT BATCH BBOXES:        (BTO, 4)
                        where ITO can vary
                    """
                    target_labels = torch.cat([target["labels"] for target in targets])
                    target_boxes = torch.cat([target["boxes"] for target in targets])

                    """
                        
                        -----> INPUT CLS PROB TENSOR:               (B_qemb=25*B, cls=21)
                        -----> output CLS COST REDUCED TENSOR:      (B_qemb=25*B, BTO)
                    """
                    COST_CLS_REDUCED_TNS = -class_prob_tns[:, target_labels]

                    # DETR predicts cx,cy,w,h , we need to covert to x1y1x2y2 for giou
                    # don't need to convert targets as they are already in x1y1x2y2
                    pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(pred_boxes_tns, 'cxcywh', 'xyxy')

                    """
                        -----> INPUT BBOX TENSOR:       (B_qemb=25*B, coord=4)
                        -----> OUTPUT L1 COST TENSOR:   (B_qemb=25*B, BTO)
                    """
                    COST_L1_REDUCED_TNS = torch.cdist(pred_boxes_x1y1x2y2, target_boxes, p=1)

                    """
                        -----> INPUT BBOX  TENSOR:          (B_qemb=25*B, coord=4)
                        -----> OUTPUT GIOU COST TENSOR:     (B_qemb=25*B, BTO)
                    """
                    COST_GIOU_REDUCED_TNS = -torchvision.ops.generalized_box_iou(pred_boxes_x1y1x2y2, target_boxes)

                    """
                        -----> INPUT sum(CLS COST, L1 COST, GIOU COST) TENSORS:     (B_qemb=25*B, BTO)
                        -----> OUTPUT COST TENSOR:                                  (B_qemb=25*B, BTO)
                    """
                    COST_TNS = (self.cls_cost_weight * COST_CLS_REDUCED_TNS + self.l1_cost_weight * COST_L1_REDUCED_TNS + self.giou_cost_weight * COST_GIOU_REDUCED_TNS)

                    """
                       -----> INPUT COST TENSOR:    (B_qemb=25*B, BTO)
                       -----> OUTPUT COST TENSOR:   (B, qemb=25, BTO)
                    """
                    COST_TNS = COST_TNS.reshape(batch_size, self.num_queries, -1).cpu()

                    """
                        -----> INPUT TARGETS SET (take "2" to get 'boxes'):         {B=IMG (2='classes'&'boxes', GT=PO)}
                    """
                    num_targets_per_image = [len(target["labels"]) for target in targets]

                    """
                    ITO - image target objects;
                       -----> INPUT COST TENSOR:       (B, qemb=25, BTO)
                       -----> OUTPUT COST TENSOR SET:  {IMG=B, (B, qemb=25, ITO)}
                       where ITO can vary per IMG (batch)
                    """
                    ITO_COST_TUPLE =  COST_TNS.split(num_targets_per_image, dim=-1)

                    match_indices = []
                    for batch_idx in range(batch_size):
                        """
                        DCT - output diagonal cost tensor,
                        ITO - image target objects
                           -----> INPUT COST TENSOR SET:           {IMG=B, (B, qemb=25, ITO)}
                           -----> OUTPUT DIAGONAL COST TENSORS:    (qemb=25, ITO)
                           where ITO is local value and can vary per tensor
                        """
                        DCT = ITO_COST_TUPLE[batch_idx][batch_idx]

                        """
                        GT - ground truth (objects),
                        PO - predicted objects
                        2 is prediction and label
                            -----> INPUT DIAGONAL COST TENSOR:    (qemb=25, ITO)
                            -----> OUTPUT LIN ASM TUPLE:          (2, GT=PO)
                        """
                        batch_idx_assignments = linear_sum_assignment(DCT)
                        batch_idx_pred, batch_idx_target = batch_idx_assignments

                        """
                            GT - ground truth (objects),
                            PO - predicted objects
                            2 is prediction and label
                                -----> INPUT LIN ASM TUPLE:         (2, GT=PO)
                                -----> OUTPUT MATCH INDICES SET:    {B=IMG (2, GT=PO)}
                                where GT=PO can is local value and can vary per batch
                        """
                        match_indices.append((torch.as_tensor(batch_idx_pred, dtype=torch.int64), torch.as_tensor(batch_idx_target, dtype=torch.int64)))

                """
                    BTO - batch target objects
                    -----> INPUT MATCH INDICES SET:      {B=IMG (2, GT=PO)}
                    -----> OUTPUT BATCH INDICES TENSOR:  (BTO)
                    where GT=PO can and can vary per batch
                """
                pred_batch_idxs = torch.cat([torch.ones_like(pred_idx) * i for i, (pred_idx, _) in enumerate(match_indices)])

                """
                    -----> INPUT MATCH INDICES SET:      {B=IMG (2, GT=PO)}
                    -----> OUTPUT QUERY INDICES TENSOR:  (BTO)
                    where GT=PO can and can vary per batch
                """
                pred_query_idx = torch.cat([pred_idx for (pred_idx, _) in match_indices])

                """
                    -----> INPUT MATCH INDICES SET:      {B=IMG (2, GT=PO)}
                    -----> INPUT TARGETS SET:            {B=IMG (2, GT=PO)}
                    -----> OUTPUT VALID TARGETS TENSOR:  (BTO)
                    where GT can and can vary per batch
                """
                valid_obj_target_cls = torch.cat([target["labels"][target_obj_idx] for target, (_, target_obj_idx) in zip(targets, match_indices)
                ])

                """
                    -----> INPUT CLS IDX TENSOR:            (B, qemb=25, cls=21)
                    -----> OUTPUT TARGET CLASSES TENSOR:    (B, qemb=25)
                """
                target_classes = torch.full(cls_idx_output.shape[:2], fill_value=self.bg_class_idx, dtype=torch.int64,device=cls_idx_output.device)


                """
                    -----> INPUT TARGET CLASSES TENSOR:     (B, qemb=25)
                    -----> INPUT BATCH INDICES TENSOR:      (BTO)
                    -----> INPUT QUERY INDICES TENSOR:      (BTO)
                    -----> INPUT VALID TARGETS TENSOR:      (BTO)
                    
                    -----> OUTPUT TARGET CLASSES TENSOR:    (B, qemb=25)
                """
                target_classes[(pred_batch_idxs, pred_query_idx)] = valid_obj_target_cls


                cls_weights = torch.ones(self.num_classes)
                """
                    -----> INPUT CLASS WEIGHTS TENSOR:      (cls=21)
                    -----> OUTPUT CLASS WEIGHTS TENSOR:     (cls=21)
                """
                cls_weights[self.bg_class_idx] = self.bg_cls_weight

                """
                    -----> INPUT TARGET CLASSES TENSOR:     (B, qemb=25)            -----> RESHAPED TARGET CLASSES TENSOR:      (B_qemb=25*B)
                    -----> INPUT CLS IDX TENSOR:            (B, qemb=25, cls=21)    -----> RESHAPED CLS IDX TENSOR:             (B_qemb=25*B, cls=21)
                    -----> INPUT CLASS WEIGHTS TENSOR:      (cls=21)
                    -----> OUTPUT LOSS SCALAR FOR CLASS     ()
                """
                loss_cls = torch.nn.functional.cross_entropy(cls_idx_output.reshape(-1, self.num_classes), target_classes.reshape(-1), cls_weights.to(cls_idx_output.device))

                """
                    -----> INPUT BATCH INDICES TENSOR:                  (BTO)
                    -----> INPUT QUERY INDICES TENSOR:                  (BTO)
                    -----> INPUT BBOX INX CLASS TENSOR:                 (B, qemb=25, coord=4)
                    -----> OUTPUT MATCHED (REDUCED) PRED BBOXES         (BTO, coord=4)
                """
                matched_pred_boxes = bbox_idx_output[pred_batch_idxs, pred_query_idx]

                """
                    -----> INPUT TARGETS SET (take "2" to get 'boxes'):         {B=IMG (2='classes'&'boxes', GT=PO)}
                    -----> INPUT MATCH INDICES SET:                             {B=IMG (2='target'&'predicted', GT=PO)}
                    -----> OUTPUT TARGET BOXES TENSOR:                          (BTO, coord=4)
                    where GT=PO can and can vary per batch
                """
                target_boxes = torch.cat([
                    target['boxes'][target_obj_idx]
                    for target, (_, target_obj_idx) in zip(targets, match_indices)],
                    dim=0
                )

                # Convert matched pred boxes to x1y1x2y2 format
                """
                    -----> INPUT MATCHED (REDUCED) PRED BBOXES          (BTO, coord=4)
                    -----> OUTPUT MATCHED (REDUCED) PRED BBOXES         (BTO, coord=4)
                """
                matched_pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(matched_pred_boxes,'cxcywh','xyxy')

                # Don't need to convert target boxes as they are in x1y1x2y2 format
                # Compute L1 Localization loss

                """
                    -----> INPUT MATCHED (REDUCED) PRED BBOXES          (BTO, coord=4)
                    -----> OUTPUT TARGET BOXES TENSOR:                  (BTO, coord=4)
                    -----> OUTPUT BBOX LOSS SCALAR:                     ()
                """
                loss_bbox = torch.nn.functional.l1_loss(matched_pred_boxes_x1y1x2y2, target_boxes, reduction='none')
                # norm
                loss_bbox = loss_bbox.sum() / matched_pred_boxes.shape[0]

                """
                    -----> INPUT MATCHED (REDUCED) PRED BBOXES          (BTO, coord=4)
                    -----> OUTPUT TARGET BOXES TENSOR:                  (BTO, coord=4)
                    -----> OUTPUT BBOX LOSS SCALAR:                     ()
                """
                loss_giou = torchvision.ops.generalized_box_iou_loss(matched_pred_boxes_x1y1x2y2, target_boxes)
                # norm
                loss_giou = loss_giou.sum() / matched_pred_boxes.shape[0]

                # losses
                losses['classification'].append(loss_cls * self.cls_cost_weight)
                losses['bbox_regression'].append(loss_bbox * self.l1_cost_weight + loss_giou * self.giou_cost_weight)

            detr_output['loss'] = losses

        else:
            # for inference we are only interested in last layer outputs

            """
                -----> INPUT CLS IDX TENSOR:        (layers=4, B, qemb=25, cls=21)
                -----> OUTPUT CLS IDX TENSOR:       (B, qemb=25, cls=21)
            """
            cls_output = cls_output[-1]

            """
                -----> INPUT BBOX INX CLASS TENSOR:     (layers=4, B, qemb=25, coord=4)
                -----> OUTPUT BBOX INX CLASS TENSOR:    (B, qemb=25, coord=4)
            """
            bbox_output = bbox_output[-1]

            """
                -----> INPUT CLS IDX TENSOR:        (B, qemb=25, cls=21)
                -----> OUTPUT CLS IDX TENSOR:       (B, qemb=25, cls=21)
            """
            prob = torch.nn.functional.softmax(cls_output, -1)

            # get all query boxes and their best fg class as label

            """
                -----> INPUT CLS IDX TENSOR:                (B, qemb=25, cls=21)
                -----> OUTPUT SCORES TENSOR:                (B, qemb)
                -----> OUTPUT LABELS TENSOR:                (B, qemb)
            """
            if self.bg_class_idx == 0:
                scores, labels = prob[..., 1:].max(-1)
                labels = labels + 1
            else:
                scores, labels = prob[..., :-1].max(-1)

            # convert to x1y1x2y2 format
            """
                -----> INPUT BBOX INX CLASS TENSOR:     (B, qemb=25, coord=4)
                -----> OUTPUT BBOX INX CLASS TENSOR:    (B, qemb=25, coord=4)
            """
            boxes = torchvision.ops.box_convert(bbox_output,'cxcywh','xyxy')

            for batch_idx in range(boxes.shape[0]):
                """
                    -----> INPUT SCORES TENSOR:                     (B, qemb)
                    -----> INPUT SCORES INX TENSOR:                 (qemb)
                """
                scores_idx = scores[batch_idx]

                """
                    -----> INPUT LABELS TENSOR:                     (B, qemb)
                    -----> INPUT SCORES INDEX TENSOR:               (qemb)
                """
                labels_idx = labels[batch_idx]

                """
                    -----> INPUT BBOX INX CLASS TENSOR:     (B, qemb=25, coord=4)
                    -----> OUTPUT BBOX INX CLASS TENSOR:     (qemb=25, coord=4)
                """
                boxes_idx = boxes[batch_idx]

                """
                    -----> INPUT SCORES INX TENSOR:                 (qemb)
                    -----> OUTPUT KEEP INX TENSOR:                  (qemb)
                """
                keep_idxs = scores_idx >= score_thresh

                """
                    -----> INPUT SCORES TENSOR:                     (B, qemb)
                    -----> OUTPUT SCORES INX TENSOR:                (qkept)
                """
                scores_idx = scores_idx[keep_idxs]

                """
                    -----> INPUT KEEP INX TENSOR:                   (qemb)
                    -----> OUTPUT BBOX INX TENSOR:                  (qkept, coord=4)
                """
                boxes_idx = boxes_idx[keep_idxs]

                """
                    -----> INPUT KEEP INX TENSOR:                   (qemb)
                    -----> OUTPUT LABELS INX CLASS TENSOR:          (qkept)
                """
                labels_idx = labels_idx[keep_idxs]

                # NMS filtering
                if use_nms:
                    """
                        -----> INPUT BBOX INX CLASS TENSOR:             (qkept, coord=4)
                        -----> INPUT SCORES INX TENSOR:                 (qkept)
                        -----> INPUT LABELS INX CLASS TENSOR:           (qkept)
                        -----> OUTPUT KEEP INX TENSOR:                  (qnms)
                    """
                    keep_idxs = torchvision.ops.batched_nms(boxes_idx, scores_idx, labels_idx, iou_threshold=self.nms_threshold)

                    """
                        -----> INPUT SCORES INX TENSOR:                 (qkept)
                        -----> INPUT KEEP INX TENSOR:                   (qnms)
                        -----> OUTPUT KEEP INX TENSOR:                  (qnms)
                    """
                    scores_idx = scores_idx[keep_idxs]

                    """
                        -----> INPUT BBOX INX CLASS TENSOR:             (qkept, coord=4)
                        -----> INPUT KEEP INX TENSOR:                   (qnms)
                        -----> OUTPUT BBOX INX TENSOR:                  (qnms, coord=4)
                    """
                    boxes_idx = boxes_idx[keep_idxs]

                    """
                        -----> INPUT LABELS INX CLASS TENSOR:           (qkept)
                        -----> INPUT KEEP INX TENSOR:                   (qnms)
                        -----> OUTPUT LABELS INX TENSOR:                (qnms)
                    """
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