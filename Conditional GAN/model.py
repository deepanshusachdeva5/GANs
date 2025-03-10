import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super().__init__()
        self.disc = nn.Sequential(nn.Conv2d(channels_img +1, features_d, kernel_size=4, stride=2, padding=1),
                                  self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
                                  self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
                                  self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
                                  nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
                                  )
        self.embed = nn.Embedding(num_classes, img_size*img_size)
        self.img_size = img_size


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.LeakyReLU(0.2),
                             )
    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size) # make it same size as x
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size):
        super().__init__()
        self.gen = nn.Sequential(self._block(z_dim+embed_size, features_g*16, kernel_size = 4, stride=2, padding =0),
                                self._block(features_g*16, features_g*8, kernel_size = 4, stride=2, padding =1),
                                self._block(features_g*8, features_g*4, kernel_size = 4, stride=2, padding =1),
                                self._block(features_g*4, features_g*2, kernel_size = 4, stride=2, padding =1),
                                nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
                                nn.Tanh()
                                )
        self.embed = nn.Embedding(num_classes, embed_size)
        self.img_size = img_size
                                 
                                 
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU()
                             )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    labels = torch.randint(0, 10, (N,))
    disc = Discriminator(in_channels, 10, 10, 64)
    
    assert disc(x, labels).shape == (N,1, 1, 1)
    gen = Generator(z_dim, in_channels, 8, 10, 64, 100)

    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z, labels).shape == (N, in_channels, H, W)
    print("Success")


test()