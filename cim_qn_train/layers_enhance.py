import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GroupedLinear(nn.Module):
    def __init__(self, in_channels, group_size = 5):
        super().__init__()
        self.in_channels = in_channels
        self.group_size = group_size
        self.linears = nn.ModuleList()

        self.groups_indices = []
        for i in range(in_channels):
            start_idx = max(0, i - group_size // 2)
            end_idx = min(in_channels, i + group_size // 2 + 1)
            if group_size % 2 == 0 and start_idx > 0:
                start_idx -= 1
            self.groups_indices.append((start_idx, end_idx))

            actual_group_size = end_idx - start_idx
            linear_layer = nn.Linear(actual_group_size, 1, bias = True)
            self.linears.append(linear_layer)

            # 初始化权重和偏置
            with torch.no_grad():
                linear_layer.weight.fill_(0)  # 先将所有权重设置为0
                # 对于每个线性层，仅将中间的权重设置为1（考虑到实际的组索引）
                middle_index = i - start_idx  # 找到中间索引在当前组中的位置
                if 0 <= middle_index < actual_group_size:
                    linear_layer.weight[0, middle_index] = 1.0
                linear_layer.bias.fill_(0)  # 将偏置设置为0

    def forward(self, x):
        outputs = []

        for i, (start_idx, end_idx) in enumerate(self.groups_indices):
            group_inputs = x[:, start_idx:end_idx]
            output = self.linears[i](group_inputs).squeeze(-1)
            outputs.append(output)

        return torch.stack(outputs, dim = -1)


class EnhanceLayerConv2d(nn.Module):
    def __init__(self, original_conv, groups):
        super().__init__()
        self.original_conv = original_conv
        out_channels = self.original_conv.out_channels

        if out_channels % groups != 0:
            groups = max(1, min(out_channels, round(out_channels / groups)))
            while out_channels % groups != 0:
                groups -= 1

        self.groups = groups
        self.conv_enhance = nn.Conv2d(out_channels, out_channels,
                                      kernel_size = 1, stride = 1,
                                      padding = 0, groups = groups)
        # self.layer_flag = 'enhance_layer'
        self.conv_enhance.layer_flag = 'enhance_layer'
        self.conv_enhance.distance = 1
        self._initialize_weights()

    def _initialize_weights(self):
        eye_matrix = torch.eye(self.conv_enhance.out_channels // self.groups,
                               self.conv_enhance.out_channels // self.groups).repeat(self.groups, 1, 1)
        with torch.no_grad():
            self.conv_enhance.weight.copy_(eye_matrix.view(self.conv_enhance.out_channels,
                                                           self.conv_enhance.out_channels // self.groups, 1, 1))
            self.conv_enhance.bias.fill_(0)

    def forward(self, x):
        x = self.original_conv(x)
        x = self.conv_enhance(x)
        return x


class EnhanceBranchConv2d(nn.Module):
    def __init__(self, original_conv, groups = 1):
        super().__init__()
        self.original_conv = original_conv
        in_channels = self.original_conv.in_channels
        out_channels = self.original_conv.out_channels
        self.stride = self.original_conv.stride
        self.padding = self.original_conv.padding

        # 定义 1x1 的卷积层作为增强层
        self.enhance_layer = nn.Conv2d(in_channels, out_channels,
                                       kernel_size = 1, stride = 1, padding = self.padding,
                                       groups = groups)
        # self.layer_flag = 'enhance_branch'
        self.enhance_layer.layer_flag = 'enhance_branch'
        # 如果 stride 不是 1，定义池化层来调整输入尺寸
        if any(s != 1 for s in self.stride):
            self.pool = nn.AvgPool2d(kernel_size = self.stride, stride = self.stride, padding = 0)
        else:
            self.pool = None

        self.blend_factor = nn.Parameter(torch.tensor(0.0))

    def merge_feature_map(self, original_output, enhance_output):
        combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
                          + enhance_output * torch.sigmoid(self.blend_factor)
        return combined_output

    def forward(self, x):
        original_output = self.original_conv(x)
        enhance_output = self.enhance_layer(x)
        # 使用池化层调整输入尺寸
        if self.pool is not None:
            enhance_output = self.pool(enhance_output)

        # 调整 enhance_output 以匹配 original_output
        diff_y = original_output.shape[2] - enhance_output.shape[2]
        diff_x = original_output.shape[3] - enhance_output.shape[3]

        if diff_y < 0 or diff_x < 0:
            enhance_output = enhance_output[:, :, :original_output.shape[2], :original_output.shape[3]]
        elif diff_y > 0 or diff_x > 0:
            pad = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
            enhance_output = nn.functional.pad(enhance_output, pad)

        combined_output = self.merge_feature_map(original_output, enhance_output)
        return combined_output


class EnhanceBranchConv2d_LoR(nn.Module):
    def __init__(self, original_conv, groups = 1, rank_factor = 0.5, relu = False, sigmoid = True):
        super().__init__()
        self.original_conv = original_conv
        in_channels = self.original_conv.in_channels
        out_channels = self.original_conv.out_channels
        self.stride = self.original_conv.stride
        self.padding = self.original_conv.padding
        self.relu = relu
        self.sigmoid = sigmoid
        hidden_channels = int(max(round(max(in_channels, out_channels) * rank_factor), 1))
        # 定义 1x1 的卷积层作为增强层
        self.enhance_branch_1 = nn.Conv2d(in_channels, hidden_channels,
                                          kernel_size = 1, stride = 1, padding = self.padding,
                                          groups = groups)
        self.enhance_branch_2 = nn.Conv2d(hidden_channels, out_channels,
                                          kernel_size = 1, stride = 1, padding = self.padding,
                                          groups = groups)
        # self.layer_flag = 'enhance_branch'
        self.enhance_branch_1.layer_flag = 'enhance_branch'
        self.enhance_branch_2.layer_flag = 'enhance_branch'
        # 如果 stride 不是 1，定义池化层来调整输入尺寸
        if any(s != 1 for s in self.stride):
            self.pool = nn.AvgPool2d(kernel_size = self.stride, stride = self.stride, padding = 0)
        else:
            self.pool = None
        if self.sigmoid:
            self.blend_factor = nn.Parameter(torch.tensor(-6.9068))
        else:
            self.blend_factor = nn.Parameter(torch.tensor(0.0))
        self.std_ori = None
        self.mean_ori = None
        self.std_enhance = None
        self.mean_enhance = None

    def merge_feature_map(self, original_output, enhance_output):
        if self.sigmoid:
            combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
                              + enhance_output * torch.sigmoid(self.blend_factor)
        else:
            combined_output = original_output * (1 - self.blend_factor) \
                              + enhance_output * self.blend_factor
        self.std_ori = original_output.std()
        self.mean_ori = original_output.mean()
        self.std_enhance = enhance_output.std()
        self.mean_enhance = enhance_output.mean()
        return combined_output

    def forward(self, x):
        original_output = self.original_conv(x)
        enhance_output = self.enhance_branch_1(x)
        if self.relu:
            enhance_output = F.relu(enhance_output)
        enhance_output = self.enhance_branch_2(enhance_output)
        # 使用池化层调整输入尺寸
        if self.pool is not None:
            enhance_output = self.pool(enhance_output)

        # 调整 enhance_output 以匹配 original_output
        diff_y = original_output.shape[2] - enhance_output.shape[2]
        diff_x = original_output.shape[3] - enhance_output.shape[3]

        # 如果尺寸不同，执行填充或裁剪
        if diff_y != 0 or diff_x != 0:
            if diff_y < 0 or diff_x < 0:
                enhance_output = enhance_output[:, :, :original_output.shape[2], :original_output.shape[3]]

            elif diff_y > 0 or diff_x > 0:
                pad = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
                enhance_output = nn.functional.pad(enhance_output, pad)

        combined_output = self.merge_feature_map(original_output, enhance_output)
        return combined_output


class EnhanceLayerLinear(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        self.original_linear = original_linear
        out_features = original_linear.out_features
        self.fc1 = nn.Linear(out_features, out_features)
        self.fc1.layer_flag = 'enhance_layer'
        self.fc1.distance = 1
        # self.layer_flag = 'enhance_layer'
        # 初始化 enhance_branch_1 的权重为单位矩阵，偏置为 0
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(out_features))
            self.fc1.bias.fill_(0)

    def forward(self, x):
        x = self.original_linear(x)
        x = self.fc1(x)
        return x


class EnhanceBranchLinear_LoR(nn.Module):
    def __init__(self, original_linear, rank_factor, relu = False, sigmoid = True):
        super().__init__()
        self.original_linear = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.relu = relu
        self.sigmoid = sigmoid
        hidden_size = max(int(min(in_features, out_features) * rank_factor), 1)

        self.enhance_branch_1 = nn.Linear(in_features, hidden_size)
        self.enhance_branch_2 = nn.Linear(hidden_size, out_features)

        if self.sigmoid:
            self.blend_factor = nn.Parameter(torch.tensor(-6.9068))
        else:
            self.blend_factor = nn.Parameter(torch.tensor(0.0))

        # self.layer_flag = 'enhance_branch'
        self.enhance_branch_1.layer_flag = 'enhance_branch'
        self.enhance_branch_1.distance = 1
        self.enhance_branch_2.layer_flag = 'enhance_branch'
        self.enhance_branch_2.distance = 1

        self.std_ori = None
        self.mean_ori = None
        self.std_enhance = None
        self.mean_enhance = None

    def merge_feature_map(self, original_output, enhance_output):
        if self.sigmoid:
            combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
                              + enhance_output * torch.sigmoid(self.blend_factor)
        else:
            combined_output = original_output * (1 - self.blend_factor) \
                              + enhance_output * self.blend_factor
        self.std_ori = original_output.std()
        self.mean_ori = original_output.mean()
        self.std_enhance = enhance_output.std()
        self.mean_enhance = enhance_output.mean()
        return combined_output

    def forward(self, x):
        original_output = self.original_linear(x)
        enhanced_output = self.enhance_branch_1(x)
        if self.relu:
            enhanced_output = F.relu(enhanced_output)
        enhanced_output = self.enhance_branch_2(enhanced_output)
        combined_output = self.merge_feature_map(original_output, enhanced_output)
        return combined_output


# =============================================== #
# 弃用函数
# =============================================== #
# 在全连接层后面加入LoR结构，有可能会让原来的高维信息丢失，因此不使用该函数
class EnhanceLayerLinear_LoR(nn.Module):
    def __init__(self, original_linear, rank_factor = 0.25):
        super().__init__()
        self.original_linear = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        hidden_size = max(int(max(in_features, out_features) * rank_factor), 1)
        self.fc1 = nn.Linear(out_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)
        self.layer_flag = 'enhance_layer'
        self.fc1.layer_flag = 'enhance_layer'
        self.fc2.layer_flag = 'enhance_layer'

    def forward(self, x):
        x = self.original_linear(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class InterpolateBranchLinear(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        self.original_linear = original_linear
        self.blend_factor = nn.Parameter(torch.tensor(0.0))

    def merge_feature_map(self, original_output, enhance_output):
        combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
                          + enhance_output * torch.sigmoid(self.blend_factor)
        return combined_output

    def forward(self, x):
        original_output = self.original_linear(x)

        # 调整 enhance_output 以匹配 original_output 的形状
        enhanced_output = torch.nn.functional.interpolate(x.unsqueeze(1),
                                                          size = original_output.shape[-1],
                                                          mode = 'linear',
                                                          align_corners = False).squeeze(1)

        combined_output = self.merge_feature_map(original_output, enhanced_output)
        return combined_output


# 该函数目前有问题，interpolate只能对宽高进行插值，而无法改变通道数
# GPT目前的解决方案就是通过1x1的卷积调整通道数
# 因此单纯依靠interpolate可能无法实现增强分支
class InterpolateBranchConv2d(nn.Module):
    def __init__(self, original_conv):
        super().__init__()
        self.original_conv = original_conv
        self.stride = self.original_conv.stride
        self.padding = self.original_conv.padding

        self.blend_factor = nn.Parameter(torch.tensor(0.0))

    def merge_feature_map(self, original_output, enhance_output):
        combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
                          + enhance_output * torch.sigmoid(self.blend_factor)
        return combined_output

    def forward(self, x):
        original_output = self.original_conv(x)

        # 调整 enhance_output 以匹配 original_output
        enhance_output = torch.nn.functional.interpolate(x,
                                                         size = original_output.shape[2:],
                                                         mode = 'bilinear', align_corners = False)

        combined_output = self.merge_feature_map(original_output, enhance_output)
        return combined_output
