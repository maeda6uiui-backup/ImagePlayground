import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,latent_dim=100):
        super().__init__()

        self.main=nn.Sequential(
            #(1,1)->(4,4)
            nn.ConvTranspose2d(latent_dim,256,kernel_size=4,stride=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            #(4,4)->(8,8)
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            #(8,8)->(16,16)
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            #(16,16)->(32,32)
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            #(32,32)->(64,64)
            nn.ConvTranspose2d(32,3,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main=nn.Sequential(
            #(64,64)->(32,32)
            nn.Conv2d(3,32,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),

            #(32,32)->(16,16)
            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),

            #(16,16)->(8,8)
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            #(8,8)->(4,4)
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            #(4,4)->(1,1)
            nn.Conv2d(256,1,kernel_size=4,stride=1,padding=0)
        )

    def forward(self,x):
        return self.main(x).squeeze()
