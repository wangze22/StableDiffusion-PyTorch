from cim_qn_train.progressive_qn_train import *
from cim_runtime.cim_utils import *

batch = 16
img_size = 32
epoch = 5000
output_dim = 256


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        # 定义一层卷积层
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3,
                               padding = 1)
        # 定义一层全连接层
        self.fc1 = nn.Linear(16 * img_size * img_size // 4, output_dim)  # 假设输入图像大小为28x28

    def forward(self, x):
        # 卷积层 + ReLU激活 + 最大池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # 展平
        x = x.view(-1, 16 * img_size * img_size // 4)
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
        self.img_s = int((img_size // 16) ** 2)
        self.fc1 = nn.Linear(256 * self.img_s, 512)  # 假设经过卷积层后输出特征图大小为1x1
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)

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
net = SimpleConvNet().to('cuda')
net2 = SimpleConvNet().to('cuda')
# net2.load_state_dict(net.state_dict())

t_nn = ProgressiveTrain(net)
t_adda = ProgressiveTrain(net2)
t_adda.convert_to_qn_layer()
t_adda.update_qn_parameter(weight_bit = 4,
                           feature_bit = 8,
                           )

# t_adda.convert_to_adda_layer()
# t_adda.update_adda_parameter(weight_bit = 4,
#                              feature_bit = 8,
#                              adc_bit = 4,
#                              dac_bit = 2)

t_nn.compare_model_weights(t_adda.model, t_nn.model)
opt_weight = optim.Adam(params = t_adda.model.parameters(), lr = 1e-4)
# opt_adc_scale = optim.Adam(params = [t_adda.model.conv1.adc_scale,
#                                      t_adda.model.fc1.adc_scale
#                                      ], lr = 1e-3)

scheduler_w = lr_scheduler.ReduceLROnPlateau(opt_weight,
                                             min_lr = 1e-6,
                                             cooldown = 20,
                                             factor = 0.5,
                                             patience = 500)
# scheduler_adc = lr_scheduler.ReduceLROnPlateau(opt_adc_scale,
#                                                min_lr = 1e-6,
#                                                cooldown = 20,
#                                                factor = 0.5,
#                                                patience = 20)

loss_list = []
adc_scale_list = []
weight_diff_list = []
lr_w_lst = []
lr_adc_lst = []
print(f'========================')
print(f'Start Training')
print(f'========================')
# print(f'conv1.adc_scale = {t_adda.model.conv1.adc_scale}')
for i in range(epoch):
    x = torch.randn(batch, 1, img_size, img_size, device = 'cuda')
    out1 = t_nn.model(x)
    out2 = t_adda.model(x)
    if i == 0:
        scatter_plt(out2.detach().cpu().flatten(),
                    out1.detach().cpu().flatten())
    loss = ((out1 - out2) ** 2).sum()
    opt_weight.zero_grad()
    # opt_adc_scale.zero_grad()
    loss.backward()
    opt_weight.step()
    # opt_adc_scale.step()
    print(f'loss = {loss}')
    print(f'Epoch = {i + 1}/{epoch}')
    # adc_scale_list.append(t_adda.model.conv1.adc_scale.item())
    weight_diff = ((t_adda.model.conv1.weight.data - t_nn.model.conv1.weight.data) ** 2).sum()
    weight_diff_list.append(weight_diff.item())
    scheduler_w.step(loss)
    # scheduler_adc.step(loss)
    # lr_adc = opt_adc_scale.param_groups[0]['lr']
    lr_w = opt_weight.param_groups[0]['lr']
    lr_w_lst.append(lr_w)
    # lr_adc_lst.append(lr_adc)
    loss_list.append(loss.item())

scatter_plt(out2.detach().cpu().flatten(),
            out1.detach().cpu().flatten())
# print(f'conv1.adc_scale = {t_adda.model.conv1.adc_scale}')
plt.plot(loss_list)
plt.title(f'Loss')
plt.yscale('log')
plt.show()

plt.plot(adc_scale_list)
plt.title(f'ADC Scale')
plt.show()

plt.plot(weight_diff_list)
plt.title(f'Weight Difference')
plt.show()

#
# plt.plot(lr_w_lst)
# plt.title(f'lr_w')
# plt.show()
#
# plt.plot(lr_adc_lst)
# plt.title(f'lr_adc')
# plt.show()
