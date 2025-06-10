# run_dalle_mia_complete.py
import os
import sys
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk
import pandas as pd
import logging
# Import from your existing file
from run_deepseek_mia_fixed import (
    process_image,
    compute_metrics,
    evaluate_attack,
    calculate_attack_metrics,
    plot_roc_curves
)
from deepseek_vl_evaclip_complete import DeepSeekVLWithEVACLIP
from deepseek_vl.models import VLChatProcessor
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Updated Paths
DALLE_DATASET_PATH = "/fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/Data/img_dalle"  # Updated path
MODEL_PATH = "/fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/deepseek-vl-7b-base"  # Updated path
OUTPUT_DIR = "/fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/Results/dalle_evaluation"  # Updated path

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model - FIXED THIS PART
    logger.info("Loading DeepSeek-VL with EVA-CLIP...")
    model = DeepSeekVLWithEVACLIP(language_model_path=MODEL_PATH)
    model.eval()
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    # Load processor
    processor = VLChatProcessor.from_pretrained(MODEL_PATH)
    
    # Load DALL-E dataset
    logger.info("Loading DALL-E dataset...")
    dataset = load_from_disk(DALLE_DATASET_PATH)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Split into members and non-members
    members = [item for item in dataset if item['label'] == 1]
    non_members = [item for item in dataset if item['label'] == 0]
    
    logger.info(f"Members: {len(members)}, Non-members: {len(non_members)}")
    
    # Process members
    logger.info("Processing member images...")
    member_features = []
    for i, item in enumerate(members):
        if i % 50 == 0:
            logger.info(f"Processing member {i}/{len(members)}")
        try:
            features = process_image(model, item['image'], 0)  # GPU 0
            features = compute_metrics(features)
            member_features.append(features)
        except Exception as e:
            logger.error(f"Error processing member {i}: {e}")
    
    # Process non-members
    logger.info("Processing non-member images...")
    non_member_features = []
    for i, item in enumerate(non_members):
        if i % 50 == 0:
            logger.info(f"Processing non-member {i}/{len(non_members)}")
        try:
            features = process_image(model, item['image'], 0)
            features = compute_metrics(features)
            non_member_features.append(features)
        except Exception as e:
            logger.error(f"Error processing non-member {i}: {e}")
    
    # Calculate attack metrics
    logger.info("Calculating attack metrics...")
    attack_results = calculate_attack_metrics(member_features, non_member_features)
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, "attack_metrics.txt")
    with open(results_path, 'w') as f:
        for attack_name, result in attack_results.items():
            f.write(f"{attack_name}: AUC = {result['auc']:.4f}, "
                   f"Accuracy = {result['accuracy']:.4f}, "
                   f"TPR@FPR=0.1 = {result['tpr@fpr=0.1']:.4f}\n")
    
    # Plot ROC curves
    plot_roc_curves(attack_results, OUTPUT_DIR)
    
    # Save detailed results
    torch.save({
        'attack_results': attack_results,
        'member_features': member_features,
        'non_member_features': non_member_features
    }, os.path.join(OUTPUT_DIR, 'results.pt'))
    
    logger.info(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()