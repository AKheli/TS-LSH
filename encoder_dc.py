import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

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
        y = self.en(x)
        return y

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_net = D_Net().to(device)
    g_net = G_Net().to(device)
    encoder_ = Encoder().to(device)
    d_net.eval()
    g_net.eval()
    encoder_.train()
    loss_fu = nn.MSELoss()
    optimizer = torch.optim.Adam(encoder_.parameters(), lr=0.0001)
    batch_size = 128

    date = np.loadtxt('./original_segmented.txt', delimiter=',')
    lis = [date[i].reshape((3, 32, 32)) / 10 for i in range(3072)]
    print(len(lis))

    dataloader = DataLoader(lis, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    d_net.load_state_dict(torch.load(r"./gand_path"))
    g_net.load_state_dict(torch.load(r"./gang_path"))
    try:
        encoder_.load_state_dict(torch.load(r'./ganen_path'))
        print('Model loaded successfully')
    except Exception as e:
        print(f'Failed to load model: {e}')

    for epoch in range(6000):
        for i, img in enumerate(dataloader):
            for p in d_net.parameters():
                p.data.clamp_(-0.01, 0.01)

            real_img = img.float().to(device)
            z = encoder_(real_img)
            real_out = g_net(z)
            out1, _ = d_net(real_img)
            out2, _ = d_net(real_out)
            loss1 = loss_fu(out2, out1)
            loss2 = loss_fu(real_out, real_img)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # pbbox.append(loss.cpu().detach().numpy())
            # sub_axix = filter(lambda x: x % 200 == 0, pbbox)
            # plt.plot(pbbox, color='green')
            # # plt.legend()
            # plt.title('en_loss')
            # plt.ylabel('en_loss')
            # plt.pause(0.001)

            if i % 10 == 0:
                print(loss.item())
                torch.save(encoder_.state_dict(), r"./ganen_path")


