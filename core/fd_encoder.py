import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import numpy as np

    
class twins_svt_large(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_large', pretrained=pretrained)

        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]
    
    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        x_4 = None
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            patch_size = embed.patch_size
            if i == layer - 1:
                embed.patch_size = (1, 1)
                embed.proj.stride = embed.patch_size
                x_4 = torch.nn.functional.pad(x, [1, 0, 1, 0], mode='constant', value=0)
                x_4, size_4 = embed(x_4)
                size_4 = (size_4[0] - 1, size_4[1] - 1)
                x_4 = drop(x_4)
                for j, blk in enumerate(blocks):
                    x_4 = blk(x_4, size_4)
                    if j == 0:
                        x_4 = pos_blk(x_4, size_4)

                if i < len(self.svt.depths) - 1:
                    x_4 = x_4.reshape(B, *size_4, -1).permute(0, 3, 1, 2).contiguous()

            embed.patch_size = patch_size
            embed.proj.stride = patch_size
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            
            if i == layer-1:
                break
        
        return x, x_4

    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            for param in embed.parameters():
                num +=  np.prod(param.size())

            for param in drop.parameters():
                num +=  np.prod(param.size())

            for param in blocks.parameters():
                num +=  np.prod(param.size())

            for param in pos_blk.parameters():
                num +=  np.prod(param.size())

            if i == layer-1:
                break

        for param in self.svt.head.parameters():
            num +=  np.prod(param.size())
        
        return num


class twins_svt_small_context(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model('twins_svt_small', pretrained=pretrained)

        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]

    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        x_4 = None
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            patch_size = embed.patch_size
            if i == layer - 1:
                embed.patch_size = (1, 1)
                embed.proj.stride = embed.patch_size
                x_4 = torch.nn.functional.pad(x, [1, 0, 1, 0], mode='constant', value=0)
                x_4, size_4 = embed(x_4)
                size_4 = (size_4[0] - 1, size_4[1] - 1)
                x_4 = drop(x_4)
                for j, blk in enumerate(blocks):
                    x_4 = blk(x_4, size_4)
                    if j == 0:
                        x_4 = pos_blk(x_4, size_4)

                if i < len(self.svt.depths) - 1:
                    x_4 = x_4.reshape(B, *size_4, -1).permute(0, 3, 1, 2).contiguous()

            embed.patch_size = patch_size
            embed.proj.stride = patch_size
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer - 1:
                break

        return x, x_4

    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            for param in embed.parameters():
                num += np.prod(param.size())

            for param in drop.parameters():
                num += np.prod(param.size())

            for param in blocks.parameters():
                num += np.prod(param.size())

            for param in pos_blk.parameters():
                num += np.prod(param.size())

            if i == layer - 1:
                break

        for param in self.svt.head.parameters():
            num += np.prod(param.size())

        return num

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(
                    num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim,
                               self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Cross attention layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
        
        # Spatial reconstruction
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, rgb_feats: torch.Tensor, event_feats: torch.Tensor):
        """
        Args:
            rgb_feats:   [B, C, H, W]
            event_feats: [B, C, H, W]
        Returns:
            fused_feats: [B, C, H, W]
        """
        # Reshape to sequence format
        B, C, H, W = rgb_feats.shape
        rgb_seq = rgb_feats.flatten(2).permute(2, 0, 1)   # [H*W, B, C]
        event_seq = event_feats.flatten(2).permute(2, 0, 1)
        
        # Cross attention: RGB as query, Event as key/value
        attended = self.cross_attn(
            query=self.norm1(rgb_seq),
            key=self.norm1(event_seq),
            value=self.norm1(event_seq),
        )[0]
        attended = attended + rgb_seq  # Residual connection
        
        # FFN
        feats = self.ffn(self.norm2(attended)) + attended
        
        # Reshape back to spatial
        feats = feats.permute(1, 2, 0).view(B, C, H, W)
        
        # Spatial refinement
        return self.conv(feats)

class FeatureFusionTransformer(nn.Module):
    def __init__(self, in_dim=128, depth=3, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionFusion(in_dim, num_heads)
            for _ in range(depth)
        ])
        
        # Initial projection (optional)
        self.rgb_proj = nn.Conv2d(in_dim, in_dim, 1)
        self.event_proj = nn.Conv2d(in_dim, in_dim, 1)
        
        # Final fusion
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.GroupNorm(32, in_dim),
            nn.GELU()
        )

    def forward(self, net: torch.Tensor, emap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            net:  RGB features [B, C, H, W]
            emap: Event features [B, C, H, W]
        Returns:
            fused features [B, C, H, W]
        """
        # Optional feature projection
        x = self.rgb_proj(net)
        e = self.event_proj(emap)
        
        # Multi-level fusion
        for layer in self.layers:
            x = layer(x, e)
        
        # Final refinement
        return self.out_conv(x) + net  # Residual connection
