from cim_qn_train.layers_adda import *
import matplotlib.pyplot as plt

torch.manual_seed(0)
conv_cim = Conv2d_ADDA_aware(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 0,
                      groups = 1, bias = False,
                      weight_bit = 4, feature_bit = 8, noise_scale = 0.0,
                      adc_bit = 4, dac_bit = 2, clamp_std = 0).to('cuda')
conv = nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 0,
                 groups = 1, bias = False).to('cuda')
conv.weight.data = conv_cim.weight.data + 0

opt = optim.Adam(conv_cim.parameters(), lr = 0.001)
opt2 = optim.Adam([conv_cim.adc_scale], lr = 0.1)
print(conv_cim.adc_scale)
loss_list = []
adc_scale_list = []
weight_diff_list = []
for i in range(500):
    x = torch.randn(32, 32, 64, 64, device = 'cuda')
    y_cim = conv_cim(x)
    y = conv(x)
    loss = torch.sum((y_cim - y) ** 2)
    opt.zero_grad()
    opt2.zero_grad()
    loss.backward()
    adc_scale_list.append(conv_cim.adc_scale.data.cpu())
    weight_diff = ((conv_cim.weight.data - conv.weight.data)**2).sum()
    weight_diff_list.append(weight_diff.item())
    # print(f'y = {y}')
    # print(f'y_cim = {y_cim}')
    # print(f'conv.grad = {conv.weight.grad}')
    # print(f'conv_cim.grad = {conv_cim.weight.grad}')
    # print(f'conv_cim.adc_scale.grad = {conv_cim.adc_scale.grad}')
    # opt.step()
    opt2.step()
    loss_list.append(loss.item())
print(conv_cim.adc_scale)
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