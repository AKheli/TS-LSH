import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')

class D_Net(nn.Module):
    def __init__(self, bais=False):
        super(D_Net, self).__init__()
        self.dnet1 = nn.Sequential(
            nn.Conv2d(3, 128, 5, 3, 3, bias=bais),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 3, bias=bais),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 3, bias=bais),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        self.dnet2 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, 2, bias=bais),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, 4, 1, bias=bais),
        )

    def forward(self, x):
        y = self.dnet1(x)
        out = self.dnet2(y)
        return y, out

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(128, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 5, 3, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

class Encoder(nn.Module):
    def __init__(self, bais=False):
        super(Encoder, self).__init__()
        self.en = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, bias=bais),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=bais),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, 1, bias=bais),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, 3, 2, 1, bias=bais),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, 3, 1, bias=bais),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 512, 3, 1, bias=bais),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 128, 3, 1, bias=bais),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        return self.en(x)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_net = D_Net().to(device)
    g_net = G_Net().to(device)
    encoder_ = Encoder().to(device)
    encoder_.eval()
    d_net.eval()
    g_net.eval()

    batch_size = 1
    date = np.loadtxt('./original_segmented.txt', delimiter=',')
    lis = [date[i].reshape((3, 32, 32)) / 10 for i in range(3072)]

    dataloader = DataLoader(lis, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    d_net.load_state_dict(torch.load(r"./gand_path"))
    g_net.load_state_dict(torch.load(r"./gang_path"))
    encoder_.load_state_dict(torch.load(r'./ganen_path'))
    print('success')

    seq = []
    for i, img in enumerate(dataloader):
        real_img = img.float().to(device)
        sap = encoder_(real_img)
        fake_img = g_net(sap)
        fake_mg = fake_img.view(-1)
        real_mg = real_img.view(-1)

        pbbox = [real_mg.cpu().detach().numpy()]
        bbox = [fake_mg.cpu().detach().numpy()]
        seq.append(bbox[0])

        plt.plot(pbbox[0], color='red')
        plt.savefig(f'./real_raw_f/{i}.pdf')
        plt.clf()

        plt.plot(pbbox[0], color='red')
        plt.plot(bbox[0], color='blue')
        plt.savefig(f'./fake_raw_f/{i}.pdf')
        plt.clf()
        print(i)

    seq = np.asarray(seq).reshape(len(seq), 3072)
    np.savetxt("synthetic_segments.txt", seq, fmt='%f', delimiter=',')

