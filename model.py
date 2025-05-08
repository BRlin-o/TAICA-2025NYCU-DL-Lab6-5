import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from config import Config
from peft import LoraConfig, get_peft_model
import copy
# from downloader import get_local_model_path, check_model_exists

class ConditionalLDM(nn.Module):
    def __init__(
        self,
        num_labels=None,
        condition_dim=None,
        vae_model_path=None,
        pretrained_model_name=None,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        use_config=True
    ):
        super().__init__()
        
        # 从Config读取参数或使用提供的参数
        if use_config:
            self.num_labels = num_labels or Config.NUM_CLASSES
            self.condition_dim = condition_dim or Config.CONDITION_DIM
            self.vae_model_path = vae_model_path or Config.VAE_MODEL
            self.pretrained_model_name = pretrained_model_name or Config.PRETRAINED_MODEL
            self.use_lora = Config.USE_LORA if hasattr(Config, 'USE_LORA') else use_lora
            self.lora_rank = Config.LORA_RANK if hasattr(Config, 'LORA_RANK') else lora_rank
            self.lora_alpha = Config.LORA_ALPHA if hasattr(Config, 'LORA_ALPHA') else lora_alpha
            self.num_train_timesteps = Config.NUM_TRAIN_TIMESTEPS
            self.beta_schedule = Config.BETA_SCHEDULE
            self.prediction_type = Config.PREDICTION_TYPE
        else:
            self.num_labels = num_labels or 24
            self.condition_dim = condition_dim or 64
            self.vae_model_path = vae_model_path or "stabilityai/sd-vae-ft-mse"
            self.num_train_timesteps = 1000
            self.beta_schedule = "squaredcos_cap_v2"
            self.prediction_type = "v_prediction"
            self.pretrained_model_name = pretrained_model_name or "stabilityai/sd-turbo"
            self.use_lora = True
            self.lora_rank = 16
            self.lora_alpha = 32

        # 1. 加載預訓練VAE
        self.vae = AutoencoderKL.from_pretrained(self.vae_model_path)

        # 凍結VAE參數
        for param in self.vae.parameters():
            param.requires_grad = False

        # 獲取VAE潛空間維度
        latent_channels = self.vae.config.latent_channels  # 通常為4

        # 2. 條件嵌入層
        self.condition_embedding = nn.Sequential(
            nn.Linear(num_labels, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, self.condition_dim),
            nn.SiLU(),
        )

        # 3. UNet擴散模型
        if self.pretrained_model_name:
            print(f"使用預訓練模型: {self.pretrained_model_name}")
            # 加載預訓練的UNet模型
            self.unet = UNet2DConditionModel.from_pretrained(
                self.pretrained_model_name,
                subfolder="unet",
            )

            # 根据i-CLEVR任务调整交叉注意力维度
            if self.unet.config.cross_attention_dim != self.condition_dim:
                print(f"UNet交叉注意力维度: {self.unet.config.cross_attention_dim}")
                # 保存原始权重
                orig_dtype = self.unet.dtype
                orig_device = next(self.unet.parameters()).device
                original_cross_attn_weights = {}
                
                # 暂时修改交叉注意力维度
                for name, module in self.unet.named_modules():
                    if hasattr(module, 'processor') and hasattr(module.processor, 'attn'):
                        attn = module.processor.attn
                        if hasattr(attn, 'to_k') and attn.to_k.in_features != self.condition_dim:
                            # 保存原始权重
                            original_cross_attn_weights[f"{name}.processor.attn.to_k"] = copy.deepcopy(attn.to_k.weight)
                            original_cross_attn_weights[f"{name}.processor.attn.to_v"] = copy.deepcopy(attn.to_v.weight)
                            
                            # 重新创建投影层
                            attn.to_k = nn.Linear(self.condition_dim, attn.inner_dim, bias=False, device=orig_device, dtype=orig_dtype)
                            attn.to_v = nn.Linear(self.condition_dim, attn.inner_dim, bias=False, device=orig_device, dtype=orig_dtype)
                
                # 重置交叉注意力权重为随机初始化
                print(f"\t调整至 -> {self.condition_dim}")
            
            # 应用LoRA (如果启用)
            if self.use_lora:
                # 配置LoRA
                lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"] 
                
                # SD-Turbo和SD-V1.5模型结构略有不同，需要检测
                if hasattr(self.unet, "named_modules"):
                    for name, _ in self.unet.named_modules():
                        if 'to_q' in name:
                            # 找到了目标模块，使用当前命名
                            break
                        elif 'q_proj' in name:
                            # 不同命名约定，更新目标模块
                            lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
                            break
                
                lora_config = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=0.1,
                    bias="none"
                )
                print(f"使用LoRA微调 (rank={self.lora_rank}, alpha={self.lora_alpha})")
                self.unet = get_peft_model(self.unet, lora_config)
                
                # 确保只有LoRA参数可训练
                for name, param in self.unet.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False
        else:
            print(f"使用自定义UNet模型")
            # 3. UNet擴散模型
            self.unet = UNet2DConditionModel(
                sample_size=16,  # 64/4=16 (VAE降維)
                in_channels=latent_channels,
                out_channels=latent_channels,
                layers_per_block=3,
                block_out_channels=(128, 256, 384, 512),
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
                attention_head_dim=8,  # 定義注意力頭數
                cross_attention_dim=self.condition_dim,
            )

        # 4. 噪聲調度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type,
            clip_sample=False
        )
        
        # 5. 採樣調度器
        self.sampler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type,
            clip_sample=False,
        )
    
    def encode(self, pixel_values):
        """圖像編碼為潛變量"""
        with torch.no_grad():
            latent_dist = self.vae.encode(pixel_values)
            latents = latent_dist.latent_dist.sample() * 0.18215
        return latents
    
    def decode(self, latents):
        """潛變量解碼為圖像"""
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            images = self.vae.decode(latents).sample
        return images
    
    def prepare_condition(self, labels):
        """處理條件標籤"""
        return self.condition_embedding(labels).unsqueeze(1)
    
    def forward(self, pixel_values, labels):
        """
        模型前向傳播
        
        參數:
            pixel_values: 標準化的圖像張量 [B, 3, 64, 64]
            labels: 標籤one-hot向量 [B, 24]
        
        返回:
            loss: 擴散損失
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # 1. 將圖像編碼為潛變量
        latents = self.encode(pixel_values)
        
        # 2. 為潛變量添加噪聲
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 3. 準備條件嵌入
        condition_embedding = self.prepare_condition(labels)
        
        # 4. 預測添加的噪聲
        noise_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=condition_embedding
        ).sample
        
        # 5. 計算損失
        # loss = F.mse_loss(noise_pred, noise)
        latent_loss = 0.1 * torch.mean(latents.pow(2))
        loss = F.mse_loss(noise_pred, noise) + latent_loss
        
        return loss
    
    @torch.no_grad()
    def generate(
        self, 
        labels, 
        batch_size=1, 
        generator=None,
        guidance_scale=None,
        num_inference_steps=None
    ):
        """
        條件圖像生成
        
        參數:
            labels: 標籤one-hot向量 [B, 24]
            batch_size: 批次大小
            generator: 隨機數生成器
            guidance_scale: CFG引導強度
            num_inference_steps: 推理步數
            
        返回:
            images: 生成的圖像
        """
        device = self.unet.device

        guidance_scale = guidance_scale or Config.GUIDANCE_SCALE
        num_inference_steps = num_inference_steps or Config.NUM_INFERENCE_STEPS
        
        # 準備採樣器
        self.sampler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.sampler.timesteps

        # 準備潛變量
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, 16, 16),
            generator=generator,
            device=device
        )

        # 準備條件嵌入
        condition_embedding = self.prepare_condition(labels)

        # 準備無條件嵌入(用於CFG)
        uncond_labels = torch.zeros_like(labels)
        uncond_embedding = self.prepare_condition(uncond_labels)

        # 採樣循環
        for i, t in enumerate(timesteps):
            # 清理GPU内存
            torch.cuda.empty_cache()

            # 計算進度
            progress = i / len(timesteps)

            # 擴展潛變量
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.sampler.scale_model_input(latent_model_input, t)

            # 擴展條件
            encoder_hidden_states = torch.cat([condition_embedding, uncond_embedding])
            
            # 預測噪聲
            with torch.no_grad():  # 确保UNet不保留梯度
                noise_pred = self.unet(
                    latent_model_input, 
                    t,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

            # 執行CFG
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 分类器引导 - 仅在后期步骤使用
            if hasattr(Config, 'CLASSIFIER_SCALE') and Config.CLASSIFIER_SCALE > 0 and progress > 0.3:
                # 动态调整分类器引导强度
                cls_scale = Config.CLASSIFIER_SCALE
                
                # 后期增强引导强度
                if progress > 0.7:
                    cls_scale = cls_scale * 1.5
                
                # 使用分类器引导
                noise_pred = self._apply_classifier_guidance(noise_pred, latents, t, labels, cls_scale)
            
            # 更新采样
            latents = self.sampler.step(noise_pred, t, latents).prev_sample

            # 清理本步骤中间变量
            del noise_pred, latent_model_input
            torch.cuda.empty_cache()
            
        # 解碼生成的潛變量
        images = self.decode(latents)
        
        # 裁剪到[0,1]範圍
        images = torch.clamp(images, -1.0, 1.0)
        
        return images

    # 添加分类器引导方法
    def _apply_classifier_guidance(self, noise_pred, latents, t, labels, classifier_scale=1.0):
        """内存优化版分类器引导"""
        if classifier_scale == 0:
            return noise_pred
        
        # 选择性应用 - 只在每4个步骤应用一次分类器引导，减少内存消耗
        step_idx = int(t.item())
        if step_idx % 4 != 0:
            return noise_pred
        
        # 减少批次大小并分批处理
        batch_size = latents.shape[0]
        if batch_size > 4:  # 批次过大时分批处理
            results = []
            for i in range(0, batch_size, 4):
                end = min(i + 4, batch_size)
                batch_result = self._apply_classifier_guidance_single(
                    noise_pred[i:end], 
                    latents[i:end], 
                    t, 
                    labels[i:end], 
                    classifier_scale
                )
                results.append(batch_result)
            return torch.cat(results, dim=0)
        else:
            return self._apply_classifier_guidance_single(
                noise_pred, latents, t, labels, classifier_scale
            )

    def _apply_classifier_guidance_single(self, noise_pred, latents, t, labels, classifier_scale=1.0):
        """单批次分类器引导，内存优化版"""
        # 主动清理缓存
        torch.cuda.empty_cache()
        
        device = latents.device
        
        # 创建可微分的潜变量
        latents_in = latents.detach().clone().requires_grad_(True)
        
        # 延迟加载分类器 - 确保仅加载一次
        if not hasattr(self, "_classifier"):
            from evaluator import evaluation_model
            self._classifier = evaluation_model()
            self._classifier_device = next(self._classifier.resnet18.parameters()).device
        
        try:
            with torch.enable_grad():
                # 1. 解码但使用较低精度
                latents_scaled = 1 / 0.18215 * latents_in
                # with torch.no_grad():  # 避免VAE解码产生梯度
                    # with torch.cuda.amp.autocast():  # 使用混合精度降低内存使用
                        # images = self.vae.decode(latents_scaled).sample
                images = self.vae.decode(latents_scaled).sample

                # 确保图像在正确的值范围 (-1 到 1)
                images_norm = torch.clamp(images, -1.0, 1.0)
                
                # 3. 确保分类器和图像在同一设备上
                cls_device = self._classifier_device
                pred = self._classifier.resnet18(images_norm.to(cls_device))
                
                # 4. 使用二元交叉熵损失
                target = labels.to(cls_device)
                loss = F.binary_cross_entropy_with_logits(pred, target)
                
                # 5. 计算梯度并立即释放
                grad = torch.autograd.grad(loss, latents_in)[0]
                
                # 6. 显式删除不需要的张量
                del loss, pred, images, images_norm, latents_scaled
                
                # 7. 标准化梯度
                grad_norm = torch.norm(grad) + 1e-10
                grad = grad / grad_norm
                
                # 8. 应用梯度
                return noise_pred - classifier_scale * grad.to(device)
        
        except Exception as e:
            print(f"分类器引导出错 ({t.item()}): {e}")
            torch.cuda.empty_cache()  # 错误时清理
            return noise_pred  # 出错时返回原始预测
        finally:
            # 确保资源释放
            torch.cuda.empty_cache()