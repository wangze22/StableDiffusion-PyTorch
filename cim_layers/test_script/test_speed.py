import time

from cim_qn_train.progressive_qn_train import *
from cim_runtime.cim_utils import *

batch = 16
img_size = 32
epoch = 100
output_dim = 10

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        # 定义一层卷积层
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 5)
        # 定义一层全连接层
        self.fc1 = nn.Linear(10 * 12 * 12, 50)  # 假设输入图像大小为28x28

    def forward(self, x):
        # 卷积层 + ReLU激活 + 最大池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # 展平
        x = x.view(-1, 10 * 12 * 12)
        # 全连接层
        x = self.fc1(x)
        return x


class ComplexConvNet(nn.Module):
    def __init__(self):
        super(ComplexConvNet, self).__init__()
        # 定义10层卷积层
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv9 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv10 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)

        # 定义5层全连接层
        self.img_s = int((img_size // 16)**2)
        self.fc1 = nn.Linear(256 * self.img_s, 512)  # 假设经过卷积层后输出特征图大小为1x1
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        # 10层卷积层 + ReLU激活 + 最大池化
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = F.relu(self.conv6(x))

        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv8(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv9(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv10(x))
        x = F.max_pool2d(x, 2)

        # 展平
        x = x.view(-1, 256 * self.img_s)

        # 5层全连接层 + ReLU激活
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # 最后一层不加激活函数

        return x


torch.manual_seed(0)
# 初始化网络
net = ComplexConvNet().to('cuda')

net_nn = ProgressiveTrain(net)
opt = optim.Adam(params = net_nn.model.parameters(), lr = 1e-3)

print(f'========================')
print(f'Start Training')
print(f'========================')
# print(f'conv1.adc_scale = {t_adda.model.conv1.adc_scale}')


t = time.time()
for i in range(epoch):
    x = torch.randn(batch, 1, img_size, img_size, device = 'cuda')
    out1 = net_nn.model(x)
    opt.zero_grad()
    y = torch.randn(batch, output_dim, device = 'cuda')
    loss = ((out1 - y) ** 2).sum()
    loss.backward()
    opt.step()

t_nn = time.time() - t
print(f'Time nn layers = {t_nn:.2f}s')

net_nn.convert_to_qn_layer()
net_nn.update_qn_parameter(weight_bit = 4,
                           feature_bit = 8,
                           )

t = time.time()
for i in range(epoch):
    x = torch.randn(batch, 1, img_size, img_size, device = 'cuda')
    out1 = net_nn.model(x)
    opt.zero_grad()
    y = torch.randn(batch, output_dim, device = 'cuda')
    loss = ((out1 - y) ** 2).sum()
    loss.backward()
    opt.step()
t_qn = time.time() - t
print(f'Time qn layers = {t_qn:.2f}s')

net_nn.convert_to_adda_layer()
net_nn.update_adda_parameter(weight_bit = 4,
                             feature_bit = 8,
                             adc_bit = 8,
                             dac_bit = 8
                             )

t = time.time()
for i in range(epoch):
    x = torch.randn(batch, 1, img_size, img_size, device = 'cuda')
    out1 = net_nn.model(x)
    opt.zero_grad()
    y = torch.randn(batch, output_dim, device = 'cuda')
    loss = ((out1 - y) ** 2).sum()
    loss.backward()
    opt.step()

t_adda = time.time() - t
print(f'Time adda layers = {t_adda:.2f}s')

delay_nn = 1
delay_qn = delay_nn * (t_qn / t_nn)
delay_adda = delay_nn * (t_adda / t_nn)

print(f'========================')
print(f'delay_nn = {delay_nn:.3f}')
print(f'delay_qn = {delay_qn:.3f}')
print(f'delay_adda = {delay_adda:.3f}')