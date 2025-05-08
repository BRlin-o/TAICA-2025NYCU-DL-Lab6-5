# Lab6

## 配置文件参数说明

### 模型架构设定

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| PRETRAINED_MODEL | 预训练模型名称/路径 | "stabilityai/sd-turbo" (快速) 或 "runwayml/stable-diffusion-v1-5" (慢但更强) |
| USE_LORA | 是否使用LoRA微调 | true (几乎总是推荐) |
| LORA_RANK | LoRA的秩 | SD-Turbo:16-24, SD-V1.5:32-48 (更大=更强但更慢) |
| LORA_ALPHA | LoRA的缩放因子 | 通常为 LORA_RANK 的2倍 |
| VAE_MODEL | 预训练VAE模型 | "stabilityai/sd-vae-ft-mse" |
| CONDITION_DIM | 条件嵌入维度 | SD-Turbo:256, SD-V1.5:768 (匹配预训练模型) |
| NUM_CLASSES | 物件类别数 | 24 (i-CLEVR数据集) |

### 扩散过程参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| NUM_TRAIN_TIMESTEPS | 训练时扩散时间步数 | 1000 |
| NUM_INFERENCE_STEPS | DDIM采样步数 | 50-100 (更多=更高质量) |
| BETA_SCHEDULE | beta噪声排程方式 | SD-Turbo:"squaredcos_cap_v2", SD-V1.5:"linear" |
| PREDICTION_TYPE | 预测类型 | SD-Turbo:"v_prediction", SD-V1.5:"epsilon" |

### 采样参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| GUIDANCE_SCALE | 无条件引导强度(CFG) | 7.5-9.0 (i-CLEVR可使用更高值如8-12) |
| CLASSIFIER_SCALE | 分类器引导强度 | SD-Turbo:8-12, SD-V1.5:12-20 (关键参数!) |

### 训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| BATCH_SIZE | 批次大小 | SD-Turbo:64-128, SD-V1.5:32-48 (3090) |
| NUM_EPOCHS | 训练轮数 | 40-60 (SD-Turbo), 60-100 (SD-V1.5) |
| LEARNING_RATE | 学习率 | SD-Turbo:1e-4, SD-V1.5:5e-5 |
| WEIGHT_DECAY | 权重衰减系数 | 1e-5 |
| FP16 | 混合精度训练 | false (如遇内存不足可开启) |
| GRAD_CLIP | 梯度裁剪阈值 | 0.5-1.0 |
| SAVE_EVERY | 每N轮储存检查点 | 5-10 |
| EVAL_EVERY | 每N轮评估模型 | 5 |

### 建议配置搭配

1. **快速验证 (SD-Turbo)**: 使用 `configs/iclevr_sdturbo_v1.json`
   - 训练快速 (约4-6小时在RTX 3090上)
   - 准确率有望达到0.6-0.7

2. **高精度 (SD-V1.5)**: 使用 `configs/iclevr_sdv1_v1.json` 
   - 训练时间较长 (约8-12小时在RTX 3090上)
   - 准确率有望达到0.7-0.85

如准确率不达标，首先尝试调整 CLASSIFIER_SCALE (增大) 和 NUM_INFERENCE_STEPS (增多)。

## 環境設定

### 虛擬環境建立與啟用

使用 Python 虛擬環境可避免套件衝突。請根據作業平台依序執行以下步驟：

- **MacOS / Linux:**

    ```bash
    ## env create
    python -m venv .venv
    source ./.venv/bin/activate

    ## pip update
    pip install --upgrade pip

    ## Normal
    pip install -r requirements.txt
    ## Using CUDA
    pip install -r requirements_CUDA.txt
    ```

- **Windows**

    ```bash
    ## env create
    python -m venv .venv
    .\.venv\Scripts\activate.bat

    ## pip update
    python -m pip install --upgrade pip

    ## Normal
    pip install -r requirements.txt
    ## Using CUDA
    pip install -r requirements_CUDA.txt
    ```

## 訓練模型

```bash
python -m main train

# 
python -m main train --use_wandb

# 使用SD-Turbo配置进行训练
python -m main train --config configs/iclevr_sdturbo_v1.json

# 使用SD-V1.5配置进行训练
python -m main train --config configs/iclevr_sdv1_v1.json
```

## 評估

```bash
python -m main evaluate

# 评估模型
python -m main evaluate --config configs/iclevr_sdturbo_v1.json --checkpoint output/20250503_123456_abcdef12/checkpoints/epoch_40.pth
```