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

    return pos_embeded, None


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_head = d_model // num_heads
    
    # q parameter is for cross attn
    def forward(self, q, k, v):
        # q, k, v: (b, seq_len, d_model)
        input_shape = v.shape
        batch, seq_len, d_model = input_shape
        broadcast_shape = (batch, seq_len, self.num_heads, d_model // self.num_heads)

        # d_model/num_heads = d_heads
        # (b, num_heads, seq_len, d_heads)
        q = q.view(broadcast_shape).transpose(1, 2)
        k = k.view(broadcast_shape).transpose(1, 2)
        v = v.view(broadcast_shape).transpose(1, 2)

        # (b, num_heads, seq_len, seq_len)
        qk = q @ k.transpose(-1, -2)
        qk /= math.sqrt(self.d_head)
        
        # we don't need mask for image classification at all
        # mask = torch.ones_like(qk, dtype=torch.bool).triu(1)
        # qk.masked_fill_(mask, -torch.inf)


        # (b, num_heads, seq_len, seq_len)
        qk = F.softmax(qk, dim=-1)
        # (b, num_heads, seq_len, d_heads)
        out = qk @ v
        # get attention map shape=(b, seq_len, seq_len)
        att_map = qk.mean(dim=1)
        # (b, seq_len, num_heads, d_heads)
        out = out.transpose(1, 2)
        # (b, seq_len, d_model)
        out = out.reshape(input_shape)

        # (b, seq_len, d_model)
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

    def forward(self, x, spatial_pos_embed):
        """
        *) x input shape=(b, seq_len, d_model)
        *) spatial_pos_embed shape=(seq_len=400, d_model=256)
        """
        out = x
        attn_weights = []

        # go through all encoder layers
        for i in range(self.num_layers):
            # norm MHSA
            in_attn = self.attn_norms[i](out)
            # add spacial position embeddings to q and k for MHSA
            q = in_attn + spatial_pos_embed
            k = in_attn + spatial_pos_embed
            v = in_attn

            # MHSA
            out_attn, attn_weight = self.attns[i](q=q, k=k, v=v)
            attn_weights.append(attn_weight)

            # dropout MHSA
            out_attn = self.attn_dropouts[i](out_attn)

            # residual connection MHSA
            out += out_attn

            # norm MLP
            in_ff = self.ff_norms[i](out) 

            # MLP
            out_ff = self.ffs[i](in_ff)

            # dropout MLP
            out_ff = self.ff_dropouts[i](out_ff)

            # residual connection MLP
            out += out_ff

            # output norn
            out = self.output_norm(out)
            return out, torch.stack(attn_weight)


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

    def forward(self, query_objects, encoder_output, query_embed, spatial_pos_embed):
        """
        *) query object input shape=(b, query_embed=25, d_model=256)
        *) encoder output input shape=(b, seq_len=400, d_model=256)
        *) query_embed shape=(b, query_embed=25, d_model=256)
        *) spatial_pos_embed shape=(seq_len=400, d_model=256)
        """
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []

        # go through all encoder layers
        for i in range(self.num_layers):

            # norm MHSA
            in_attn = self.attn_norms[i](out)

            # add query embeddings to q and k for MHSA
            q = in_attn + query_embed
            k = in_attn + query_embed
            v = in_attn

            # MHSA
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
            q = in_attn + query_embed
            k = encoder_output + spatial_pos_embed
            v = encoder_output
            # croos_att trouble

            .....







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

        -----> INPUT TENSOR: (B, c=3, h=640, w=640) -----> OUTPUT TENSOR: (B, c=512, h=20, w=20)
        """
        resnet_out = self.backbone(x)

        # (b, d_model=256, feat_h=20, feat_w=20)
        conv_out = self.backbone_proj(resnet_out)

        batch_size, d_model, feat_h, feat_w = conv_out.shape
        # shape=(seq_len=400, d_model=256)
        spatial_pos_embed = get_spatial_position_embeddings(self.d_model, conv_out)
        
        # reshape and transpose new shape=(b, seq_len=400, d_model=256)
        # feat_h=20 * feat_w=20 => seq_len=400
        conv_out = conv_out.reshape(batch_size, d_model, feat_h * feat_w).transpose(1, 2)

        # encoder call
        # enc_output out shape=(b, seq_len, d_model)
        # enc_att_weights out shape=(num_encoder_layers, b, seq_len, d_model)
        enc_output, enc_att_weights = self.encoder(conv_out, spatial_pos_embed)
        
        # old query reshaped to shape=(b, query_embed, d_model)
        query_reshaped = self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1))
        # init new query objects all to zeros
        query_objects_zeros = torch.zeros_like(query_reshaped)
        
        # query_objects out shape=(num_decoder_layers, b, num_queries, num_classes)
        # decoder_attn_weights out shape=(num_decoder_layers, b, num_queries, seq_len)
        query_objects, decoder_attn_weights = self.decoder(query_objects_zeros, enc_output, query_reshaped, spatial_pos_embed)

        # shape=(num_decoder_layers, b, num_queries, num_classes)
        cls_output = self.class_mlp(query_objects)

        # shape=(num_decoder_layers, b, num_queries, coord=4)
        bbox_output = self.bbox_mlp(query_objects).sigmoid()
        