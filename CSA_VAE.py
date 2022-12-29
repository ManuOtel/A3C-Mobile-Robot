import torch
import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :x.shape[2]-1, :x.shape[3]-1]  # e.g. input 維度 [:,:,129,129], 129=>:128 (0~127共128)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    expansion = 1

    def __init__(self, planes):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()


    def forward(self, out):
        out = self.ca(out) * out # 广播机制
        out = self.sa(out) * out # 广播机制

        return out


class CA(nn.Module):
    expansion = 1

    def __init__(self, planes):
        super(CA, self).__init__()

        self.ca = ChannelAttention(planes)


    def forward(self, out):
        out = self.ca(out) * out # 广播机制

        return out


class SA(nn.Module):
    expansion = 1

    def __init__(self, planes):
        super(SA, self).__init__()

        self.sa = SpatialAttention()


    def forward(self, out):
        out = self.sa(out) * out # 广播机制

        return out


class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            # input img N, C ,H ,W
            nn.Conv2d(3, 32, stride=2, kernel_size=3, bias=False, padding=1),  # output N, 32, H/2, W/2
            # add spatial attention
            SA(32),
            # CBAM(32),
            nn.BatchNorm2d(32),                 # output shape same as input
            nn.PReLU(),                         # output shape same as input
            nn.Dropout2d(0.25),                 # output shape same as input ,隨機將整個通道歸零 通道是2D特徵圖 使用伯努利分佈的取樣，每個通道將在每次呼叫forward中以概率p獨立清零。
            #
            nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False, padding=1),  # output N, 64, H/4, W/4
            # add attention
            # CBAM(64),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),  # output N, 64, H/8, W/8
            # add attention
            # CBAM(64),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),  # output N, 64, H/16, W/16
            # add attention
            # CBAM(64),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),  # output N,64, H/32, W/32
            # add channel attention
            CA(64),
            # CBAM(64),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.Dropout2d(0.25),
            #
            nn.Flatten(),           # N, 64, H/32, W/32 (N,8,8)展開成只有一層 => N, 8192 (64*8*8)
            nn.Linear(4096, 512),    # fc
            nn.BatchNorm1d(512),
            nn.PReLU()
        )

        self.z_mean = nn.Sequential(
            nn.Linear(512, self.z_dim),
            #nn.BatchNorm1d(self.z_dim),
            #nn.ReLU6()
        ) # Mean: 200 node
        self.z_log_var = nn.Sequential(
            nn.Linear(512, self.z_dim),
            #nn.BatchNorm1d(self.z_dim),
            #nn.ReLU6()
        )
        self.z_fmap = nn.Sequential(
            nn.Linear(self.z_dim, 512),    # code to feature map
            nn.BatchNorm1d(512),
            nn.PReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.PReLU(),                   # source code就有的一層,需要轉成cnn input shape
            Reshape(-1, 64, 8, 8),                  # 4096 = 8x8 x 64channel
            # Reshape(-1, 64, 4, 4),  # 1024= 4 x 4 x 64
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(32, 3, stride=2, kernel_size=3, padding=1),
            #
            Trim(),  ### 3x129x129 -> 3x128x128   #3x63x63 ->3x64x64
            nn.ReLU(inplace=True)
        )

    def encoding_fn(self, cur_obs):    # train Navigation 用
        x = self.encoder(cur_obs)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)   # both node [1, 64]
        z = self.reparameterize(z_mean, z_log_var)              # z code: encoded node [1, 64]

        return z

    def Navigation_forward(self, cur_obs, target_obs):
        cur_z = self.encoding_fn(cur_obs)        # output pre feature map [1, 64]
        target_z = self.encoding_fn(target_obs)                      # output targe feature map [1, 64]

        return cur_z, target_z

    def reparameterize(self, z_mu, z_log_var):      # 重新random 取參數作為noise ,在乘上 exp 加總 ->code with noise
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())   # gpu
        # eps = torch.randn(z_mu.size(0), z_mu.size(1))                       # cpu
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        #z = z_mu * torch.exp(z_log_var / 2.)  # z_log_var表示 log(variance) 同log(sigma^2),所以 exp(log(var) / 2) 等於std
        return z

    def forward(self, x):   # train VAE用
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)  # both node [1, 200]
        z = self.reparameterize(z_mean, z_log_var)  # z code: encoded node [1, 200]

        z_fmap = self.z_fmap(z)                  # za 的feature map node [1, 512]
        z_decoded = self.decoder(z_fmap)
        return z, z_mean, z_log_var, z_decoded

