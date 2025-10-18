from cim_qn_train.layers_adda import *
import matplotlib.pyplot as plt

torch.manual_seed(0)
linear_cim = Linear_ADDA_aware(in_features = 100, out_features = 100, bias = False,
                               weight_bit = 4, feature_bit = 8, noise_scale = 0.0,
                               adc_bit = 4, dac_bit = 2, clamp_std = 0).to('cuda')
linear = nn.Linear(in_features = 100, out_features = 100, bias = False).to('cuda')
linear.weight.data = linear_cim.weight.data + 0

opt = optim.Adam(linear_cim.parameters(), lr = 0.001)
opt2 = optim.Adam([linear_cim.adc_scale], lr = 0.1)
print(linear_cim.adc_scale)
loss_list = []
adc_scale_list = []
weight_diff_list = []
for i in range(500):
    x = torch.randn(32, 100, device = 'cuda')
    y_cim = linear_cim(x)
    y = linear(x)
    loss = torch.sum((y_cim - y) ** 2)
    opt.zero_grad()
    opt2.zero_grad()
    loss.backward()
    if i % 99 == 0:
        print(f'loss = {loss}')
    adc_scale_list.append(linear_cim.adc_scale.data.cpu())
    weight_diff = ((linear_cim.weight.data - linear.weight.data) ** 2).sum()
    weight_diff_list.append(weight_diff.item())
    # print(f'y = {y}')
    # print(f'y_cim = {y_cim}')
    # print(f'conv.grad = {conv.weight.grad}')
    # print(f'conv_cim.grad = {conv_cim.weight.grad}')
    # print(f'conv_cim.adc_scale.grad = {conv_cim.adc_scale.grad}')
    opt.step()
    opt2.step()
    loss_list.append(loss.item())
print(linear_cim.adc_scale)
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