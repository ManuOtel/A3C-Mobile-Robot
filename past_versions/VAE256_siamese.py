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
        return x[:, :, :x.shape[2]-21, :x.shape[3]-21]  # e.g. input 維度 [:,:,129,129], 129=>:128 (0~127共128)


class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            # input img N, C ,H ,W
            nn.Conv2d(3, 32, stride=2, kernel_size=3, padding=1),  # output N, 32, H/2, W/2
            nn.BatchNorm2d(32),                 # output shape same as input
            nn.LeakyReLU(inplace=True),    # output shape same as input
            nn.Dropout2d(0.25),                 # output shape same as input ,隨機將整個通道歸零 通道是2D特徵圖 使用伯努利分佈的取樣，每個通道將在每次呼叫forward中以概率p獨立清零。
            #
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1),  # output N, 64, H/4, W/4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1),  # output N, 64, H/8, W/8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1),  # output N, 64, H/16, W/16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1),  # output N,64, H/32, W/32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.Flatten(),           # N, 64, H/32, W/32 (N,8,8)展開成只有一層 => N, 8192 (64*8*8)
            nn.Linear(6400, 512),   # fc
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.z_mean = nn.Sequential(
            nn.Linear(512, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
        )   # Mean: 200 node
        self.z_log_var = nn.Sequential(
            nn.Linear(512, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
        )   # var:  200 node
        self.z_fmap = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True)
        )   # code to feature map

        self.decoder = nn.Sequential(
            nn.Linear(512, 6400),                   # source code就有的一層,需要轉成cnn input shape
            nn.BatchNorm1d(6400),
            nn.LeakyReLU(inplace=True),
            Reshape(-1, 64, 10, 10),                  # 4096 = 8x8 x 64channel
            # Reshape(-1, 64, 4, 4),  # 1024= 4 x 4 x 64
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25),
            #
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(32, 3, stride=2, kernel_size=3, padding=1),
            #
            Trim(),
        )

    def encoding_fn(self, cur_obs):    # train Navigation 用
        x = self.encoder(cur_obs)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)   # both node [1, 64]
        z = self.reparameterize(z_mean, z_log_var)              # z code: encoded node [1, 64]

        # z_fmap = self.z_fmap(z)          # z 的feature map node [1, 512]
        # z_decoded = self.decoder(z_fmap)  # output feature map [1, 512]
        # return z_fmap, z_decoded
        return z

    def Navigation_forward(self, cur_obs, target_obs):
        # nxet_fmap, recon_pre_obs = self.encoding_fn(cur_obs)        # output pre feature map [1, 512]
        nxet_fmap = self.encoding_fn(cur_obs)        # output pre feature map [1, 64]
        # target_fmap = self.encoder(target_obs)                      # output targe feature map [1, 512]
        target_fmap = self.encoding_fn(target_obs)                      # output targe feature map [1, 64]
        
        return nxet_fmap, target_fmap

    def reparameterize(self, z_mu, z_log_var):      # 重新random 取參數作為noise ,在乘上 exp 加總 ->code with noise
        # eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())   # gpu
        # eps = torch.randn(z_mu.size(0), z_mu.size(1))                       # cpu
        z = z_mu * torch.exp(z_log_var / 2.)  # z_log_var表示 log(variance) 同log(sigma^2),所以 exp(log(var) / 2) 等於std
        return z

    def forward(self, x):   # train VAE用
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)  # both node [1, 200]
        z = self.reparameterize(z_mean, z_log_var)  # z code: encoded node [1, 200]
        # pre_a = F.relu(self.fpre_a(pre_a))          # pre_act node [1, 200]
        # za = torch.cat((z, pre_a), 1)               # 合併z, pre_act -> node [1, 400]
        z_fmap = self.z_fmap(z)                  # za 的feature map node [1, 512]
        z_decoded = self.decoder(z_fmap)
        return z, z_mean, z_log_var, z_decoded

