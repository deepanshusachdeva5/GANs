import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import Discriminator, Generator,initialize_weights
from torch.utils.tensorboard import SummaryWriter
import torchvision
from utils import gradient_penalty

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
lr = 1e-4
batch_size = 64
image_dim = 64
channels_img=1
num_classes = 10
gen_embedding = 100
z_dim = 100
num_epochs = 5
features_disc = 64
features_gen = 64
Critic_iter = 5
lambda_gp = 10

transforms = transforms.Compose([transforms.Resize(image_dim), transforms.ToTensor(), transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])])
datasets = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
gen = Generator(z_dim, channels_img, features_gen, num_classes, image_dim, gen_embedding).to(device)
disc = Discriminator(channels_img, features_disc, num_classes, image_dim).to(device)
initialize_weights(gen)
initialize_weights(disc)



opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, z_dim, 1,1, device=device)
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
step = 0
gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real,labels) in enumerate(dataloader):
        real = real.to(device)
        curr_batch_size = real.shape[0]
        labels = labels.to(device)
        for _ in range(Critic_iter):
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise, labels)
            disc_real = disc(real, labels).reshape(-1)
            disc_fake = disc(fake, labels).reshape(-1)
            gp = gradient_penalty(disc, labels, real, fake, device=device)
            loss_disc = (-(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp * gp)
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

        output = disc(fake, labels).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
            with torch.no_grad():
                fake = gen(noise, labels)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            step += 1