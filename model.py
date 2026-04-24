"""
SE-Unet for GHI estimation from FY-4 satellite data
Inference-only module
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ImprovedUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 20,
            out_channels: int = 1,
            ch: int = 64,
            shortcut_method: str = 'cat',
            use_attention: bool = True,
            attention_reduction: int = 16
    ):
        super().__init__()

        if shortcut_method == 'cat':
            k = 2
            self.shortcut_ops = lambda a, b: torch.cat([a, b], dim=1)
        elif shortcut_method == 'res':
            k = 1
            self.shortcut_ops = lambda a, b: a + b
        else:
            raise ValueError("shortcut_method must be 'cat' or 'res'")

        # Encoder
        self.linear_head = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)

        self.encode_list = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(ch, 2 * ch, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                nn.BatchNorm2d(2 * ch),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(2 * ch, 4 * ch, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                nn.BatchNorm2d(4 * ch),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(4 * ch, 8 * ch, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                nn.BatchNorm2d(8 * ch),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(8 * ch, 16 * ch, kernel_size=3, padding=1),
            ),
        ])

        # Middle
        self.middle_block = nn.Sequential(
            nn.BatchNorm2d(16 * ch),
            nn.ReLU(),
            nn.Conv2d(16 * ch, 16 * ch, kernel_size=3, padding=1),
        )

        self.middle_attention = SEBlock(16 * ch, reduction=attention_reduction) if use_attention else nn.Identity()

        # Decoder
        self.decode_list = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(16 * ch * k),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16 * ch * k, 8 * ch, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                nn.BatchNorm2d(8 * ch * k),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(8 * ch * k, 4 * ch, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                nn.BatchNorm2d(4 * ch * k),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(4 * ch * k, 2 * ch, kernel_size=3, padding=1),
            ),
            nn.Sequential(
                nn.BatchNorm2d(2 * ch * k),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(2 * ch * k, 1 * ch, kernel_size=3, padding=1),
            ),
        ])

        # Decoder attention
        if use_attention:
            self.decoder_attentions = nn.ModuleList([
                SEBlock(8 * ch, reduction=attention_reduction),
                SEBlock(4 * ch, reduction=attention_reduction),
                SEBlock(2 * ch, reduction=attention_reduction),
                SEBlock(1 * ch, reduction=attention_reduction),
            ])
        else:
            self.decoder_attentions = nn.ModuleList([nn.Identity() for _ in range(4)])

        self.linear_tail = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, out_channels, kernel_size=1),
            nn.Sigmoid(),  # trained with Sigmoid; output ∈ (0,1), multiply by ghi_scale for W/m²
        )

    def _center_crop(self, tensor, target_h, target_w):
        h, w = tensor.size()[-2:]
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return tensor[..., top:top + target_h, left:left + target_w]

    def forward(self, x):
        shortcuts = []
        h = self.linear_head(x)

        # Encoder
        for m in self.encode_list:
            h = m(h)
            h = h * 0.7071  # rescale
            shortcuts.append(h)

        # Middle
        h = self.middle_block(h)
        h = self.middle_attention(h)

        # Decoder
        for idx, m in enumerate(self.decode_list):
            shortcut = shortcuts.pop()
            shortcut = self._center_crop(shortcut, h.size(-2), h.size(-1))
            h = self.shortcut_ops(shortcut, h)
            h = m(h)
            h = self.decoder_attentions[idx](h)

        return self.linear_tail(h)