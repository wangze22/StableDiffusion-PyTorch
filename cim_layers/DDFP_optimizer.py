import torch
from torch.optim import Optimizer
from cim_layers.quant_noise_utils import *
import time


class AdamOptimizerINT(Optimizer):
    def __init__(self, params, name_list, named_parameters, model,
                 lr_bit = 5, betas = (0.5, 0.5), eps = 1e-8):
        # 参数字典，包含学习率、beta、epsilon 等超参数
        defaults = dict(lr = lr_bit, betas = betas, eps = eps)
        super().__init__(params, defaults)
        self.name_list = name_list
        self.named_parameters = dict(named_parameters)
        self.model = model

    def step(self, closure = None):
        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr_bit = group['lr']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                # grad = floor_no_pass(grad)

                param_name = self.name_list[idx]

                # 初始化 state
                state = self.state[param]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param.data)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)

                exp_avg = state['exp_avg']
                # exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # 1. 更新一阶动量 (exp_avg)
                new_exp_avg = floor_no_pass(beta1 * exp_avg) + floor_no_pass((1 - beta1) * grad)
                exp_avg.copy_(new_exp_avg)  # 显式更新 exp_avg

                # 2. 更新二阶动量 (exp_avg_sq)
                # new_exp_avg_sq = floor_no_pass(beta2 * exp_avg_sq) + floor_no_pass((1 - beta2) * (grad * grad))
                # exp_avg_sq.copy_(new_exp_avg_sq)  # 显式更新 exp_avg_sq

                # ================== #
                # 中间省略系数修正
                # ================== #

                # 5. 计算参数更新量
                # denom = floor_no_pass((exp_avg_sq ** 0.5))
                # denom[denom == 0] = 1

                lr = self.adaptive_lr(exp_avg, lr_bit = max(lr_bit, 1))
                update = floor_no_pass(exp_avg * lr)

                # print(f'lr = {lr}')
                # print(f'update_max = {update.abs().max()}')

                # 6. 更新参数
                new_param_data = param.data - update
                param.data.copy_(new_param_data)  # 显式更新参数

                idx += 1

    def adaptive_lr(self, update, lr_bit = 5):
        scale_factor = update.abs().max() / 2 ** lr_bit
        shift_bits = torch.log2(scale_factor)
        shift_bits = torch.ceil(shift_bits)
        shift_bits = torch.clamp(shift_bits, -32, 32)
        lr = 1 / 2 ** shift_bits
        return lr


class PercentOptimizerFP(Optimizer):
    def __init__(self, params,
                 lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8):
        # 参数字典，包含学习率、beta、epsilon 等超参数
        defaults = dict(lr = lr, betas = betas, eps = eps)
        super().__init__(params, defaults)

    def step(self, closure = None):
        idx = 0
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                # 初始化 state
                state = self.state[param]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param.data)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # 1. 更新一阶动量 (exp_avg)
                new_exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.copy_(new_exp_avg)  # 显式更新 exp_avg

                # 2. 更新二阶动量 (exp_avg_sq)
                new_exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)
                exp_avg_sq.copy_(new_exp_avg_sq)  # 显式更新 exp_avg_sq

                # 3. 计算偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 4. 计算调整后的学习率
                adjusted_lr = (bias_correction2 ** 0.5) / bias_correction1

                # 5. 计算参数更新量
                denom = (exp_avg_sq ** 0.5) / (bias_correction2 ** 0.5) + eps
                update = (adjusted_lr * exp_avg) / denom

                # ==================== #
                # 动态调整学习率
                # ==================== #
                max_update = update.abs().max()
                max_weight = param.data.abs().max()

                scale_factor = lr * max_weight / (max_update + eps)
                update *= scale_factor
                # ==================== #

                # 6. 更新参数
                new_param_data = param.data - update
                param.data.copy_(new_param_data)  # 显式更新参数

                idx += 1


class DDFP_scheduler:
    def __init__(self, optimizer, mode = 'min', factor = 2, patience = 10, threshold = 1e-4, threshold_mode = 'rel',
                 cooldown = 0, min_lr = 1, eps = 1e-8, verbose = True):
        if factor < 1 or not isinstance(factor, int):
            raise ValueError("Factor must be an integer greater than 1.")

        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr if isinstance(round(min_lr), list) else [round(min_lr)] * len(optimizer.param_groups)
        self.eps = eps
        self.verbose = verbose

        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = -1

        if mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'.")

        self.mode_worse = float('inf') if self.mode == 'min' else -float('inf')
        self.is_better = (lambda a, best: a < best - self.threshold) if mode == 'min' else (lambda a, best: a > best + self.threshold)
        self._reset()

    def _reset(self):
        """Resets the scheduler's state."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch = None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs during cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = old_lr - self.factor

            if new_lr < self.min_lr[i]:
                new_lr = self.min_lr[i]

            param_group['lr'] = new_lr
            if self.verbose:
                print(f"Reducing learning rate of group {i} to {new_lr}.")

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0


class SGDWithMomentum(Optimizer):
    def __init__(self, params, name_list, named_parameters, model,
                 lr = 2 ** -7, momentum = 0.9, weight_decay = 0.0):
        # 参数字典，包含学习率、动量、权重衰减等超参数
        defaults = dict(lr = lr, momentum = momentum, weight_decay = weight_decay)
        super().__init__(params, defaults)
        self.name_list = name_list
        self.named_parameters = dict(named_parameters)
        self.model = model

    # def get_layer_name(self, param_name):
    #     # 找到最后一个 '.' 的位置
    #     last_dot_index = param_name.rfind('.')
    #
    #     # 提取最后一个 '.' 之前的部分作为 layer_name
    #     if last_dot_index != -1:
    #         layer_name = param_name[:last_dot_index]
    #     else:
    #         # 如果没有 '.'，则返回整个 param_name
    #         layer_name = param_name
    #
    #     return layer_name
    #
    # def get_layer_module(self,layer_name):
    #     for name, module in self.model.named_modules():
    #         if name == layer_name:
    #             return module
    #
    # def get_step_sizes(self, layer_name):
    #     step_size_weight = f'{layer_name}.lsq_quant_weight.step_size'
    #     step_size_bias = f'{layer_name}.lsq_quant_bias.step_size'
    #     step_size_shift = f'{layer_name}.mvm_result_shift.step_size'
    #
    #     self.step_size_weight = self.named_parameters[step_size_weight]
    #     self.step_size_bias = self.named_parameters[step_size_bias]
    #     step_size_shift = self.named_parameters[step_size_shift]
    #     shift_bits = torch.log2(step_size_shift)
    #     self.shift_bits = round_pass_exp(shift_bits)
    #     self.step_size_shift = 2 ** shift_bits

    def step(self, closure = None):
        idx = 0
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                grad = floor_no_pass(grad)
                # =================================== #
                # 用于观察梯度
                # =================================== #
                # param_name = self.name_list[idx]
                # param_ = self.named_parameters[param_name]
                #
                # is_same = torch.allclose(param, param_, rtol=1e-05, atol=1e-08, equal_nan=True)
                # print(f'{is_same}')
                #
                # # layer_name = self.get_layer_name(param_name)
                # # self.get_step_sizes(layer_name)
                # # module = self.get_layer_module(layer_name)
                #
                # if 'bias' in param_name:
                #     # scale = self.step_size_bias
                #     scale = self.model.lsq_quant_bias.step_size
                # elif 'weight' in param_name:
                #     # scale = self.step_size_weight
                #     scale = self.model.lsq_quant_weight.step_size
                # else:
                #     print(f'unknown grad type')
                #     exit(1)
                # grad_int = grad / scale
                # print(f'grad_int = {grad_int}')
                # =================================== #

                # 获取当前参数的学习率和动量
                lr = group['lr']
                momentum = group['momentum']

                # 2. 初始化动量缓冲区
                if 'momentum_buffer' not in self.state[param]:
                    self.state[param]['momentum_buffer'] = torch.zeros_like(param.data)

                momentum_buffer = self.state[param]['momentum_buffer']

                # 3. 更新动量：momentum_buffer = momentum * momentum_buffer + grad
                momentum_buffer = momentum * momentum_buffer + grad

                # 4. 使用动量来更新参数：param = param - lr * momentum_buffer
                param.data = param.data - lr * momentum_buffer
                idx += 1


def unique_value_percentage(tensor):
    # 获取 tensor 中唯一值的数量
    unique_values = torch.unique(tensor)
    num_unique = unique_values.numel()  # 唯一值的数量

    # 获取 tensor 中元素的总数
    total_elements = tensor.numel()

    # 计算百分比
    percentage = (num_unique / total_elements) * 100

    return num_unique, percentage
