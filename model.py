from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Reg(nn.Module):
    def __init__(self, img_size=128, num_classes=6, conv_dims=(32, 64, 128, 256), conv_depths=(2, 2, 2, 2),
                 stem_channel=16, embed_dims=(32, 64, 128, 256),
                 num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 depths=(2, 2, 2, 2)):
        super(Reg, self).__init__()

        self.num_classes = num_classes
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.conv_layers = nn.ModuleList()
        conv_s1 = nn.Sequential(nn.Conv2d(stem_channel, conv_dims[0], kernel_size=3, stride=2, padding=1),
                                nn.BatchNorm2d(conv_dims[0], eps=1e-6))
        self.conv_layers.append(conv_s1)

        for i in range(3):
            conv_layer = nn.Sequential(nn.Conv2d(conv_dims[i], conv_dims[i + 1], kernel_size=2, stride=2),
                                       nn.BatchNorm2d(conv_dims[i + 1], eps=1e-6))
            self.conv_layers.append(conv_layer)

        self.stages = nn.ModuleList()

        for i in range(4):
            stage = nn.Sequential(
                *[ConvBlock(dim=conv_dims[i])
                  for j in range(conv_depths[i])]
            )
            self.stages.append(stage)

        self.reg_head1 = RegHead(conv_dims[0])
        self.reg_head2 = RegHead(conv_dims[1])
        self.reg_head3 = RegHead(conv_dims[2])

        self.stem = StemBlock()

        self.patch_embed_a = PatchEmbed(
            img_size=img_size, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 2, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks_a = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer)
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer)
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer)
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer)
            for i in range(depths[3])])

        self.fea_fu1 = FeaFuseBlock(conv_dims[0])
        self.fea_fu2 = FeaFuseBlock(conv_dims[1])
        self.fea_fu3 = FeaFuseBlock(conv_dims[2])
        self.fea_fu4 = FeaFuseBlock(conv_dims[3])

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(2048, 100)
        self.fc2 = nn.Linear(100, 32)
        self.fc3 = nn.Linear(32, 6)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        # stem block
        x = self.stem(x)

        # local branch
        x_1 = self.conv_layers[0](x)
        x_1 = self.stages[0](x_1)
        x_2 = self.conv_layers[1](x_1)
        x_2 = self.stages[1](x_2)
        x_3 = self.conv_layers[2](x_2)
        x_3 = self.stages[2](x_3)
        x_4 = self.conv_layers[3](x_3)
        x_4 = self.stages[3](x_4)

        B = x.shape[0]

        # global branch
        x_s1, (H1, W1) = self.patch_embed_a(x)
        for i, blk in enumerate(self.blocks_a):
            x_s1 = blk(x_s1)
        x_s_r1 = x_s1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x_s2, (H2, W2) = self.patch_embed_b(x_s_r1)
        for i, blk in enumerate(self.blocks_b):
            x_s2 = blk(x_s2)
        x_s_r2 = x_s2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        x_s3, (H3, W3) = self.patch_embed_c(x_s_r2)
        for i, blk in enumerate(self.blocks_c):
            x_s3 = blk(x_s3)
        x_s_r3 = x_s3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        x_s4, (H4, W4) = self.patch_embed_d(x_s_r3)
        for i, blk in enumerate(self.blocks_d):
            x_s4 = blk(x_s4)

        x_fuse1 = self.fea_fu1(x_1, x_s1, None, H1, W1)
        aux_head1 = self.reg_head1(x_fuse1)
        x_fuse2 = self.fea_fu2(x_2, x_s2, x_fuse1, H2, W2)
        aux_head2 = self.reg_head2(x_fuse2)
        x_fuse3 = self.fea_fu3(x_3, x_s3, x_fuse2, H3, W3)
        aux_head3 = self.reg_head3(x_fuse3)
        x_fuse4 = self.fea_fu4(x_4, x_s4, x_fuse3, H4, W4)

        output = self.relu(self.bn1(self.conv1(x_fuse4)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.relu(self.bn3(self.conv3(output)))
        output = output.view(output.size()[0], -1)
        output = self.relu(self.fc1(output))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return aux_head1, aux_head2, aux_head3, output


class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depth wise conv
        self.bn1 = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1)  # point wise conv
        self.bn2 = nn.BatchNorm2d(dim, eps=1e-6)
        self.relu = nn.ReLU()

    def forward(self, x):
        shorcut = x
        x = self.relu(self.bn1(self.dwconv(x)))
        x = self.relu(self.bn2(self.pwconv(x)))

        x = x + shorcut

        return x


class RegHead(nn.Module):
    def __init__(self, channels, features=100):
        super().__init__()
        self.reg = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels * 2, features),
            nn.ReLU(),
            nn.Linear(features, 6)
        )

    def forward(self, x):
        return self.reg(x)

class StemBlock(nn.Module):
    def __init__(self, stem_ch=16):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, stem_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(stem_ch, eps=1e-6)

        self.conv2 = nn.Conv2d(stem_ch, stem_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(stem_ch, eps=1e-6)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=128, patch_size=4, in_chans=1, embed_dim=256):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class FeaFuseBlock(nn.Module):
    def __init__(self, in_planes, down_sample=2):
        super(FeaFuseBlock, self).__init__()

        self.down_sample = down_sample
        self.proj = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(in_planes, in_planes * 2, kernel_size=1)
        self.attn = Att(in_planes * 2)

    def forward(self, x_c, x_t, x_f, H, W):
        B, N, C = x_t.shape
        x_t = x_t.transpose(1, 2).reshape(B, C, H, W)
        x_t = self.relu(self.bn(self.proj(x_t)))

        x = torch.cat((x_c, x_t), dim=1)

        x = self.attn(x)

        if x_f is not None:
            x_f = F.interpolate(x_f, size=(H, W))
            x_f = self.conv(x_f)
            x = x + x_f

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)


class Att(nn.Module):
    def __init__(self, in_planes):
        super(Att, self).__init__()
        self.channel_attn = ChannelAttention(in_planes)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        c_attn = self.channel_attn(x)
        s_attn = self.spatial_attn(x)
        out = c_attn * x + s_attn * x

        return out


