from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """
    EEG-style patch embedding block.

    Input:  x of shape (batch, channels, samples)  == (B, C, T)
    Output: y of shape (batch, time_steps, emb_size) == (B, L, E)

    Time length:
        T1 = T - temporal_kernel + 1                         # after temporal conv
        L  = floor((T1 - pool_kernel) / pool_stride + 1)     # after pooling

    Args:
        emb_size:        Output embedding dimension E.
        num_channels:    Number of input channels C.
        temporal_kernel: Temporal conv kernel size (along samples).
        num_filters:     Intermediate feature channels F.
        pool_kernel:     AvgPool kernel size (along samples).
        pool_stride:     AvgPool stride (along samples).
        dropout:         Dropout probability.
    """

    def __init__(
        self,
        emb_size: int = 40,
        num_channels: int = 22,
        temporal_kernel: int = 25,
        num_filters: int = 40,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.temporal_kernel = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(1, temporal_kernel),
            stride=(1, 1),
        )

        self.spatial_kernel = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(num_channels, 1),
            stride=(1, 1),
        )
        self.batch_norm = nn.BatchNorm2d(num_features=num_filters)
        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Conv2d(
            in_channels=num_filters,
            out_channels=emb_size,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T)
            B: Batch size
            C: Number of channels (num_channels)
            T: Number of temporal samples

        Returns:
            Tensor of shape (B, L, E)
            where:
                L = floor((T - temporal_kernel + 1 - pool_kernel) / pool_stride + 1)
                E = emb_size
        """
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.temporal_kernel(x)  # (B, num_filters, C, T1)
        x = self.spatial_kernel(x)  # (B, num_filters, 1, T1)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x)  # (B, num_filters, 1, L)
        x = self.dropout(x)
        x = self.projection(x)  # (B, E, 1, L)
        x = x.squeeze(2).transpose(1, 2)  # (B, L, E)
        return x


class ClassificationHead(nn.Sequential):
    """
    Lightweight MLP head for sequence classification.

    Input (to forward):
        x of shape (B, L, E)
          B: batch size
          L: number of time steps (from PatchEmbedding/encoder)
          E: embedding size (emb_size)

    Architecture:
        - Flatten over (L, E) -> (L*E)
        - Linear -> 256 -> ELU -> Dropout
        - Linear -> 32  -> ELU -> Dropout
        - Linear -> num_classes (logits)

    Args:
        dropout:     Dropout probability for hidden layers.
        num_classes: Number of target classes (size of the logits vector).
    """

    def __init__(self, dropout: float, num_classes: int) -> None:
        super().__init__(
            nn.Flatten(),
            nn.LazyLinear(out_features=256),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.LazyLinear(out_features=num_classes),
        )


class EEGConformer(nn.Module):
    """
    End-to-end EEG classifier combining PatchEmbedding, a Transformer encoder,
    and a lightweight classification head.

    Pipeline:
        1) PatchEmbedding:     (B, C, T) -> (B, L, E)
        2) TransformerEncoder: (B, L, E) -> (B, L, E)
        3) ClassificationHead: (B, L, E) -> (B, K)

    Shapes:
        - Input:
            x: (B, C, T)
              B: batch size
              C: number of EEG channels (num_channels)
              T: temporal samples
        - After PatchEmbedding:
            (B, L, E) where
                L = floor((T - temporal_kernel + 1 - pool_kernel) / pool_stride + 1)
                E = emb_size
        - After Encoder:
            (B, L, E)
        - Output:
            logits: (B, K) where K = num_classes

    Args:
        emb_size:        Output embedding dimension E produced by PatchEmbedding.
        num_channels:    Number of input channels C.
        temporal_kernel: Temporal conv kernel size in PatchEmbedding.
        num_filters:     Intermediate feature channels in PatchEmbedding.
        pool_kernel:     AvgPool kernel size in PatchEmbedding.
        pool_stride:     AvgPool stride in PatchEmbedding.
        num_heads:       Number of attention heads in the Transformer encoder.
        dim_feedfoward:  Hidden size of the Transformer feed-forward network.
                         (name kept as provided for compatibility)
        num_layers:      Number of Transformer encoder layers.
        num_classes:     Number of output classes K.
        dropout:         Dropout probability used in encoder and head.
    """

    def __init__(
        self,
        emb_size: int = 40,
        num_channels: int = 22,
        temporal_kernel: int = 25,
        num_filters: int = 40,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        num_heads: int = 10,
        dim_feedfoward: int = 160,
        num_layers: int = 6,
        num_classes: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            emb_size=emb_size,
            num_channels=num_channels,
            temporal_kernel=temporal_kernel,
            num_filters=num_filters,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=dim_feedfoward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.head = ClassificationHead(dropout=dropout, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T)
               B: batch size
               C: number of EEG channels (num_channerls)
               T: number of temporal samples

        Returns:
            Tensor of shape (B, K) with class logits, where:
                K = num_classes

        Shape flow:
            x                    : (B, C, T)
            patch_embedding(x)   : (B, L, E)
            encoder(...)         : (B, L, E)
            head(...)            : (B, K)
            where L = floor((T - temporal_kernel + 1 - pool_kernel) / pool_stride + 1),
                  E = emb_size.
        """
        x = self.patch_embedding(x)  # (B, L, E)
        x = self.encoder(x)  # (B, L, E)
        x = self.head(x)  # (B, K)
        return x
