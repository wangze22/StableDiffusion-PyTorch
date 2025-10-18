# StableDiffusion-PyTorch 中文说明

面向希望复现、调研或拓展 Stable Diffusion 训练流程的开发者。本说明总结仓库结构、环境依赖、数据准备、训练与采样脚本以及常见问题，帮助你快速上手并安全地扩展项目。

- **适用人群**：对扩散模型、潜空间生成、条件生成感兴趣的研究者与工程师  
- **主要目标**：复现 Stable Diffusion 的 VQ-VAE + DDPM 训练流程，理解并定制条件控制

---

## 1. 项目概览

StableDiffusion-PyTorch 将 Stable Diffusion 的两阶段流程拆分为易于复用的组件：

1. **潜空间建模**：使用 `models/vqvae.py` 训练 VQ-VAE，将高分辨率图像压缩为低维潜向量。
2. **扩散建模**：基于潜向量训练 DDPM（`models/unet_base.py` / `models/unet_cond_base.py`），可按需接入类别、文本、掩码等条件。

项目提供完整的训练、推理与采样脚本，配合 YAML 配置文件即可运行 CelebA-HQ、MNIST 等示例任务，并支持扩展到自定义数据集。

---

## 2. 功能亮点

- **模块化设计**：VQ-VAE、UNet、调度器、条件编码解耦，便于替换与升级。
- **多条件支持**：原生实现类别条件、文本条件、文本+掩码条件，兼容 CLIP/BERT。
- **工具链完善**：提供训练、推理、采样、潜向量缓存、掩码生成等脚本。
- **配置驱动**：核心超参数由 YAML 管理，易于复现与对比实验。
- **示例齐全**：README 展示 CelebA-HQ、MNIST 的训练结果，便于校验流程。

---

## 3. 仓库结构

| 目录/文件 | 说明 |
| --- | --- |
| `config/` | CelebA-HQ、MNIST 等示例配置 (`*.yaml`) |
| `dataset/` | 数据集加载器（`mnist_dataset.py`、`celeb_dataset.py`） |
| `data/` | 建议放置原始数据与掩码的目录（需手动准备） |
| `models/` | VQ-VAE、UNet、LPIPS、判别器等模型定义 |
| `scheduler/` | 线性噪声调度器实现 |
| `tools/` | 训练、采样、潜向量缓存等脚本 |
| `utils/` | 文本处理、配置读取、扩散辅助工具 |
| `README.md` | 原作者英文说明与示例结果 |

---

## 4. 环境准备

1. **Python 版本**：建议 3.8，推荐使用 Conda 或 venv 创建独立环境。  
2. **GPU 要求**：支持 CUDA 的显卡可显著缩短训练时间，脚本会自动检测 `cuda`。  
3. **依赖安装**：
   ```bash
   pip install -r requirements.txt
   ```
4. **LPIPS 权重**：从浏览器下载 `vgg.pth`，放置在 `models/weights/v0.1/vgg.pth`。  
5. **可选优化**：
   - 设置 `TORCH_HOME` 以缓存模型下载位置；
   - 如需多 GPU，请自行封装 `DistributedDataParallel`。

---

## 5. 快速上手

1. 克隆仓库并完成环境准备。  
2. 复制适合的 YAML 配置（如 `config/celebhq.yaml`），调整数据路径与超参。  
3. **训练 VQ-VAE**：
   ```bash
   python -m tools.train_vqvae --config config/celebhq.yaml
   ```
4. **生成重建样本 / 缓存潜向量**（如需在扩散阶段复用潜向量）：
   ```bash
   python -m tools.infer_vqvae --config config/celebhq.yaml
   ```
   在配置中启用 `train_params.save_latents` 可写出 `.pkl` 潜向量。
5. **训练扩散模型**：
   - 无条件：`python -m tools.train_ddpm_vqvae --config config/celebhq.yaml`
   - 条件（文本/掩码等）：`python -m tools.train_ddpm_cond --config config/celebhq_text_cond.yaml`
6. **采样生成图像**：
   ```bash
   python -m tools.sample_ddpm_vqvae --config config/celebhq.yaml
   ```
   针对条件任务可使用对应的 `sample_ddpm_*.py` 脚本。
7. 在 `train_params.task_name` 目录下查看权重、可视化及采样输出。

---

## 6. 数据准备

### 6.1 MNIST

```
StableDiffusion-PyTorch/
  data/
    mnist/
      train/images/*.png
      test/images/*.png
```
`dataset/mnist_dataset.py` 会自动完成读入、缩放与标准化。

### 6.2 CelebAMask-HQ（无条件）

```
StableDiffusion-PyTorch/
  data/
    CelebAMask-HQ/
      CelebA-HQ-img/*.jpg
```

### 6.3 CelebAMask-HQ（文本 / 掩码条件）

```
StableDiffusion-PyTorch/
  data/
    CelebAMask-HQ/
      CelebA-HQ-img/*.jpg
      CelebAMask-HQ-mask-anno/0-14/*.png
      celeba-caption/*.txt
```

- 执行 `python -m utils.create_celeb_mask` 可将官方标注转换为模型需要的多通道掩码。
- 若训练文本条件模型，需确保 `celeba-caption` 包含与图像名匹配的描述文本。

---

## 7. 配置文件要点

示例（节选自 `config/celebhq_text_cond.yaml`）：

```yaml
dataset_params:
  name: celebhq
  im_path: data/CelebAMask-HQ
  im_size: 256
  im_channels: 3

autoencoder_params:
  model_dim: 128
  z_channels: 4
  down_sample: [1, 1, 1, 1]

train_params:
  task_name: outputs/celebhq_text
  autoencoder_batch_size: 16
  ldm_batch_size: 4
  save_latents: true
```

关键字段说明：

- `dataset_params`：数据集名称、路径、图像尺寸与通道数。
- `autoencoder_params`：VQ-VAE 编码/解码结构与潜向量维度。
- `train_params`：批大小、轮次、学习率、输出目录、权重文件名、潜向量缓存等。
- `diffusion_params`：扩散步数与噪声调度范围。
- `ldm_params`：UNet 架构、残差通道、注意力层配置、条件类型等。
- `condition_config`：当启用条件时，详细定义文本/图像条件的嵌入方式与 dropout。

---

## 8. 训练流程说明

### 8.1 VQ-VAE 阶段

- 脚本：`tools/train_vqvae.py`
- 损失：重建损失 (`MSE`)、感知损失 (`LPIPS`)、判别器损失。
- 注意：
  - `train_params.disc_start` 控制何时引入判别器更新。
  - 训练完成后会保存 autoencoder/discriminator 权重与重建样本。

### 8.2 扩散阶段

- 脚本（无条件）：`tools.train_ddpm_vqvae.py`
- 脚本（条件）：`tools.train_ddpm_cond.py`
- 流程：
  1. 读取潜向量（若 `use_latents=True`）或在线编码图像。
  2. 使用 `scheduler/linear_noise_scheduler.py` 注入噪声。
  3. 通过 UNet 预测噪声残差，最小化 `MSE`。
- 条件建模：
  - 文本条件利用 `utils/text_utils.py` 加载 CLIP/BERT，并提供空文本向量以实现 classifier-free guidance。
  - 掩码条件将多通道语义掩码拼接到条件输入。

---

## 9. 推理与采样

- **VQ-VAE 推理**：`python -m tools.infer_vqvae --config ...`  
  - 导出输入/编码/重建网格（PNG）。  
  - 当 `save_latents=True` 时，会批量写出 `.pkl` 潜向量至 `train_params.vqvae_latent_dir_name`。

- **扩散采样（无条件）**：`python -m tools.sample_ddpm_vqvae --config ...`  
  - 仅在最后一步解码潜向量，输出 `samples/x0_0.png` 等过程帧。

- **扩散采样（条件）**：  
  - 类别条件：`python -m tools.sample_ddpm_class_cond --config ...`  
  - 文本条件：`python -m tools.sample_ddpm_text_cond --config ...`  
  - 文本 + 掩码：`python -m tools.sample_ddpm_text_image_cond --config ...`

输出目录遵循 `task_name` / `samples` / `cond_*_samples` 的层级结构，便于对比不同时间步的生成效果。

---

## 10. 工具脚本速览

| 脚本 | 作用 |
| --- | --- |
| `tools/train_vqvae.py` | 训练 VQ-VAE，自适应加载 MNIST / CelebA-HQ |
| `tools/infer_vqvae.py` | 生成重建样本，并按需缓存潜向量 |
| `tools/train_ddpm_vqvae.py` | 训练无条件潜空间扩散模型 |
| `tools/train_ddpm_cond.py` | 训练类别/文本/掩码等条件扩散模型 |
| `tools/sample_ddpm_vqvae.py` | 无条件采样，导出生成图像 |
| `tools/sample_ddpm_class_cond.py` | 类别条件采样 |
| `tools/sample_ddpm_text_cond.py` | 文本条件采样 |
| `tools/sample_ddpm_text_image_cond.py` | 文本 + 掩码条件采样 |
| `utils/create_celeb_mask.py` | 将官方语义分割标注转换为多通道掩码 |

---

## 11. 常见问题

- **潜向量未找到**：确认已运行 `tools/infer_vqvae.py` 且 `train_params.save_latents=True`，扩散配置中的 `latent_path` 与输出目录匹配。  
- **显存不足**：降低 `autoencoder_batch_size`、`ldm_batch_size` 或缩小 `im_size`；必要时启用混合精度训练。  
- **条件训练收敛慢**：检查条件文件是否齐全；合理设置 `cond_drop_prob`，并确保文本/掩码维度与配置一致。  
- **生成结果偏暗**：推理阶段确保图像反归一化 `(x + 1) / 2`；必要时调整后处理流程。  
- **重复采样覆盖**：清理 `task_name/samples` 中旧文件以避免混淆。  
- **Windows 控制台乱码**：使用 `type docs/project_overview_zh.md | more` 或在支持 UTF-8 的编辑器中查看。

---

## 12. 进阶建议

1. **自定义数据集**：继承 `torch.utils.data.Dataset` 编写新数据加载器，并在配置中切换 `dataset_params.name` 与对应路径。  
2. **扩展条件**：在 `models/unet_cond_base.py` 中增添新的条件分支（如姿态关键点、文本摘要等）。  
3. **更换调度器**：实现自定义噪声调度（余弦、指数等），替换 `LinearNoiseScheduler`。  
4. **混合精度 & 分布式**：结合 `torch.cuda.amp` 与 `DistributedDataParallel` 优化训练速度。  
5. **模型结构改进**：尝试双向注意力、变分自编码器、或引入 U-Net 变体以提升生成质量。  
6. **实验管理**：将 `task_name` 映射到灵活的实验命名规则，结合 TensorBoard 或 Weights & Biases 做可视化。

---

## 13. 参考资料

- 仓库根目录 `README.md`：提供原作者的英文说明、示例图与教程视频链接。  
- 官方 Stable Diffusion 论文与开源实现，用于深入理解扩散模型理论。  
- PyTorch 官方文档（https://pytorch.org/docs/）了解最新 API 与最佳实践。

--- 

如在使用过程中遇到新的问题，建议结合源码（尤其是 `tools/`、`models/`、`config/` 目录）与训练日志定位原因，并在此基础上迭代改进配置与模型结构。
