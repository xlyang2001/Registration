import torch
import torch.nn as nn
import torch.nn.functional as F


class RegModel(nn.Module):
    def __init__(self, num_channels):
        super(RegModel, self).__init__()
        self.cor_fea = Feature_Extraction()
        self.sag_fea = Feature_Extraction()
        self.global_branch = Global_branch(num_channels)
        self.cor_attn = NLBlockND(in_channels=num_channels)
        self.sag_attn = NLBlockND(in_channels=num_channels)
        self.fuse = TransFuse(num_channels, num_channels)

        self.conv1 = nn.Conv2d(num_channels * 2, 16, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 6)

    def forward(self, x_cor, x_sag):
        # conv
        f_cor = self.cor_fea(x_cor)
        f_sag = self.sag_fea(x_sag)
        f_corr = self.global_branch(f_cor, f_sag)
        h_cor = self.cor_attn(f_cor)
        h_sag = self.sag_attn(f_sag)
        x_fuse = self.fuse(h_cor, h_sag)
        x_out = torch.cat([f_corr, x_fuse], dim=1)
        x_out = self.relu(self.bn1(self.conv1(x_out)))
        x_out = self.relu(self.bn2(self.conv2(x_out)))
        x_out = self.relu(self.bn3(self.conv3(x_out)))
        x_out = self.flatten(x_out)
        x_out = self.fc3(self.fc2(self.fc1(x_out)))

        return f_cor, f_sag, h_cor, h_sag, x_out


class NLBlockND(nn.Module):
    # Our implementation of the attention block referenced https://github.com/tea1528/Non-Local-NN-Pytorch

    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=2, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        # elif self.mode == "concatenate":
        else:  # default as concatenate
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class Feature_Extraction(nn.Module):
    def __init__(self):
        super(Feature_Extraction, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Correlation(nn.Module):
    def __init__(self, max_disp=1, kernel_size=1, stride=1):
        assert kernel_size == 1
        assert stride == 1
        super().__init__()

        self.max_disp = max_disp
        self.padlayer = nn.ConstantPad2d(max_disp, 0)

    def forward_run(self, x_1, x_2):
        x_2 = self.padlayer(x_2)
        offsetx, offsety = torch.meshgrid([torch.arange(0, 2 * self.max_disp + 1),
                                           torch.arange(0, 2 * self.max_disp + 1)], indexing='ij')

        w, h = x_1.shape[2], x_1.shape[3]
        x_out = torch.cat([torch.mean(x_1 * x_2[:, :, dx:dx + w, dy:dy + h], 1, keepdim=True)
                           for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))], 1)
        return x_out

    def forward(self, x_1, x_2):
        x = self.forward_run(x_1, x_2)
        return x


class Global_branch(nn.Module):
    def __init__(self, num_channels):
        super(Global_branch, self).__init__()
        self.corr = Correlation(max_disp=1)
        self.conv = nn.Conv2d(in_channels=73, out_channels=num_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        x_corr = self.corr(x1, x2)
        x = torch.cat([x1, x_corr, x2], dim=1)
        x = self.conv(x)

        return x


class TransFuse(nn.Module):
    def __init__(self, in_channels, mlp_dim):
        super(TransFuse, self).__init__()
        self.layer1 = TransLayer(in_channels, mlp_dim)
        self.layer2 = TransLayer(in_channels, mlp_dim)
        self.layer3 = TransLayer(in_channels, mlp_dim)
        self.layer4 = TransLayer(in_channels, mlp_dim)

    def forward(self, x, y):
        fuse_xy = self.layer1(x, y)
        fuse_yz = self.layer2(y, x)
        fuse_xx = self.layer3(fuse_xy, fuse_xy)
        fuse_yy = self.layer4(fuse_yz, fuse_yz)
        output = fuse_xx + fuse_yy

        return output


class TransLayer(nn.Module):
    def __init__(self, in_channels, mlp_dim):
        super().__init__()
        self.mlp_q = nn.Sequential(
            nn.Conv2d(in_channels, mlp_dim, kernel_size=1),
            nn.ReLU()
        )
        self.mlp_k = nn.Sequential(
            nn.Conv2d(in_channels, mlp_dim, kernel_size=1),
            nn.ReLU()
        )
        self.mlp_v = nn.Sequential(
            nn.Conv2d(in_channels, mlp_dim, kernel_size=1),
            nn.ReLU()
        )
        self.mlp_out = nn.Sequential(
            nn.Conv2d(in_channels, mlp_dim, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x, y):
        Q = self.mlp_q(x)
        K = self.mlp_k(y)
        V = self.mlp_v(y)
        B, C, H, W = Q.shape
        Q = Q.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, mlp_dim]
        K = K.view(B, C, -1)  # [B, mlp_dim, H*W]
        V = V.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, mlp_dim]

        attention = torch.matmul(Q, K)
        attention = F.softmax(attention, dim=-1)
        weighted_V = torch.matmul(attention, V)
        weighted_V = weighted_V.permute(0, 2, 1).view(B, C, H, W)

        output = self.mlp_out(weighted_V)
        output = output + x

        return output

