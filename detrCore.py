import torch
import torch.nn as nn
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # linear projections for Q, K, V (shared for all heads)
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # final linear layer after concat
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x: torch.Tensor):
        B, L, E = x.shape  # [batch, seq_len, embed_size]

        # project into Q, K, V
        Q = self.query(x)  # [B, L, E]
        K = self.key(x)
        V = self.value(x)

        # split into heads: [B, L, num_heads, head_dim]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)  # [B, num_heads, L, head_dim]

        # concat heads back
        out = out.transpose(1, 2).contiguous().view(B, L, E)  # [B, L, E]

        # final linear projection
        out = self.fc_out(out)

        return out, attn_weights

class FeedForward(nn.Module):

    # def function
    def __init__(self, embed_size: int):
        super().__init__()

        # define 2-layer FCL
        self.layer1 = nn.Linear(embed_size, embed_size)
        self.layer2 = nn.Linear(embed_size, embed_size)

    # forward propagation function
    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        # SA block
        self.attention_layer = MultiHeadAttention(embed_size, num_heads)
        # FFN block
        self.feed_forward = FeedForward(embed_size)

        # norm and dropout
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor):
        # context is a matrix with attention embeddings
        # attention_scores are just attention matrix
        context, attention_scores = self.attention_layer(x)

        # process context matrix
        context = self.layer_norm1(context)
        context = self.dropout(context)
        context = self.feed_forward(context)
        context = F.relu(context)

        # return original context
        output = context + x
        return output, attention_scores

class Transformer(nn.Module):

    def __init__(self, embed_size: int, num_layers: int, num_heads: int):
        super().__init__()
        # build a blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size, num_heads) for _ in range(num_layers)])

    # forward pass algorithm
    def forward(self, x: torch.Tensor):
        attention_scores = []

        # iterate over each layer (block) and aggregate attention
        for transformer_block in self.transformer_blocks:
            x, attention_score = transformer_block(x)
            attention_scores.append(attention_score)

        return x, attention_scores

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, embed_size: int, max_seq_length: int):
        super().__init__()
        # create positional vector
        position = torch.arange(max_seq_length).unsqueeze(1)
        # create vector of frequencies for pos embedding
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        # matrix for pos embbeding
        pe = torch.zeros(max_seq_length, embed_size)
        # fill matrix with freq
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # save pos matrix to the buffer
        self.register_buffer('positional_embedding', pe)

    # embbed positional embeddings
    def forward(self, x: torch.Tensor):
        return x + self.positional_embedding[:x.size(1), :]

class DETR(nn.Module):

    def __init__(
        self,
        image_size:   int  = 640,    # length and width of the entrance, we assume a square
        patch_size:   int  = 32,     # patch side
        num_channels: int  = 3,      # RGB
        embed_size:   int  = 768,    # token length (= d_model)
        num_heads:    int  = 8,      # num of block heads in MHA
        num_layers:   int  = 1,      # number of layers in transformer
        num_queries:  int  = 20,     # how many objects maximum we search for
        num_classes:  int  = 2,      # unique defects; +1 will be "background"
        dropout:      float= 0.1,    # p(drop) before heads
    ):
        super().__init__()

        # class attributes here
        # count how many patches will be obtained
        # (2048//32)^2 = 64^2 = 4096 patch tokens
        self.num_patches = (image_size // patch_size) ** 2

        # patch encoder
        # Conv2d with stride makes an operation on the picture by numbers (2) and (2)*
        self.patch_embed = nn.Conv2d(
            in_channels=num_channels,    # RGB = 3 channels
            out_channels=embed_size,     # 768 - number of embedding channels (d_model)
            kernel_size=patch_size,      # 32×32 (patch)
            stride=patch_size            # step = patch size so patches don't overlap
        )

        # positional encoding for patch tokens
        self.pos_encoding = SinusoidalPositionEncoding(
            embed_size, max_seq_length=self.num_patches
        )

        # learnable object‑queries
        self.num_queries  = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_size))

        # transformer initialization
        self.transformer = Transformer(embed_size, num_layers, num_heads)
        self.dropout = nn.Dropout(dropout)

        # output heads
        # (+1) class is «no‑object» — required for DETR‑loss
        self.class_head = nn.Linear(embed_size, num_classes+1)

        # boxes: cx, cy, w, h ∈ (0,1) in normalized coordinates
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 4),
            nn.Sigmoid()                       # limit the range
        )

        # initialize weights
        self._init_weights()

    # define weights initialization
    # protected method
    def _init_weights(self):
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.constant_(self.class_head.bias, 0)
        # go through all Linear inside bbox_head and initialize their weights
        for m in self.bbox_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    # DETR forward pass
    def forward(self, images, return_attention: bool = False):
        """
        args:
            images: Tensor  (B, 3, H, W)  — expected  H=W=640
        return:
            class_logits: (B, Q, num_classes + 1)
            boxes:        (B, Q, 4)   — cx,cy,w,h  ∈ (0,1)
            [+] attn      (optionally) — raw attention maps of all layers
        """

        # current batch size (it is 1)
        B = images.size(0)

        # patches
        """
        Conv → (B, D, H', W')
        B - batch size;
        D - embedding size;
        H' - height (number of patches vertically)
        W' - width (number of patches horizontally)
        """
        x = self.patch_embed(images)

        # flatten = glue H' and W' into one axis - we get the length 4096
        x = x.flatten(2).transpose(1, 2)    # (B, N, D) where  N = 4096

        # add positional encoding
        x = self.pos_encoding(x)

        # expand queries for batch
        # query_tokens.shape = (1, Q, D) ---> (B, Q, D) "virtual" repeat without copy
        queries = self.query_tokens.expand(B, -1, -1)

        # merge: first patches, then queries
        # total length of the sequence = N + Q = 420
        src = torch.cat([x, queries], dim=1)  # (B, N+Q, D)

        # feed src into transformer
        #  output  (B, N+Q, D)
        #  attention_maps (list of [layer] with (B, head, L, L))
        src, attn = self.transformer(src)

        # separate only object‑queries
        # we know that queries are at the end
        hs = src[:, -self.num_queries:, :]      # (B, Q, D)

        # two heads output
        class_logits = self.class_head(self.dropout(hs))  # (B, Q, C+1)
        boxes        = self.bbox_head(self.dropout(hs))   # (B, Q, 4)

        if return_attention:
            return class_logits, boxes, attn
        return class_logits, boxes
