import os
import json
import logging
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def normalize_for_evaluator(images):
    # 1. [-1,1] → [0,1]
    images = (images + 1) / 2
    # 2. 依規定做 Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ⇒ [-1,1]
    normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))
    return normalize(images)

def setup_logging(log_dir, run_id):
    """設置日誌記錄"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_id}.log")
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(run_id)
    logger.info(f"開始運行: {run_id}")
    
    return logger

def save_checkpoint(path, model, optimizer=None, epoch=0):
    """保存模型檢查點"""
    state_dict = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    
    torch.save(state_dict, path)

def load_checkpoint(path, model, optimizer=None, map_location=None):
    """載入模型檢查點"""
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict['model'])
    if optimizer is not None and 'optimizer' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict.get('epoch', 0)

def save_images(images, captions, output_path, nrow=8):
    """保存圖像網格"""
    # 將[-1,1]範圍轉換為[0,1]
    images = (images + 1) / 2
    
    # 創建網格
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    
    # 轉換為PIL
    grid_image = torchvision.transforms.ToPILImage()(grid)
    
    # 添加標題(如果有)
    if captions is not None:
        # 創建新畫布
        height = grid_image.height + 30 * len(images) // nrow
        canvas = Image.new('RGB', (grid_image.width, height), color=(255, 255, 255))
        canvas.paste(grid_image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        
        # 添加標題
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
            
        for i, caption in enumerate(captions):
            row = i // nrow
            col = i % nrow
            
            x = col * (grid_image.width // nrow)
            y = grid_image.height + row * 30
            
            caption_text = ", ".join(caption) if isinstance(caption, list) else caption
            draw.text((x, y), caption_text, font=font, fill=(0, 0, 0))
    
        # 保存
        canvas.save(output_path)
    else:
        # 直接保存網格
        grid_image.save(output_path)

def log_images(logger, images, captions, step):
    """記錄圖像(用於tensorboard等)"""
    # 將[-1,1]範圍轉換為[0,1]
    images = (images + 1) / 2
    
    # 創建網格
    grid = torchvision.utils.make_grid(images)
    
    # 記錄圖像
    logger.add_image('generated_images', grid, step)