# run_augmented_dalle_mia.py
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

# Paths for augmented DALL-E dataset
AUGMENTED_DALLE_DATASET_PATH = "/fred/oz402/aho/VLLM-MIA/Data/augmented_dalle"
MODEL_PATH = "/fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/deepseek-vl-7b-base"
OUTPUT_DIR = "/fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/Results/augmented_dalle_evaluation"

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    logger.info("Loading DeepSeek-VL with EVA-CLIP...")
    model = DeepSeekVLWithEVACLIP(language_model_path=MODEL_PATH)
    model.eval()
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    # Load processor
    processor = VLChatProcessor.from_pretrained(MODEL_PATH)
    
    # Load augmented DALL-E dataset
    logger.info("Loading augmented DALL-E dataset...")
    dataset = load_from_disk(AUGMENTED_DALLE_DATASET_PATH)
    logger.info(f"Augmented DALL-E dataset size: {len(dataset)}")
    
    # Split into members and non-members (assuming the dataset has labels like original DALL-E)
    try:
        members = [item for item in dataset if item['label'] == 1]
        non_members = [item for item in dataset if item['label'] == 0]
    except:
        # If no labels, split dataset in half like Flickr approach
        logger.info("No labels found, splitting dataset in half...")
        split_idx = len(dataset) // 2
        members = dataset[:split_idx]
        non_members = dataset[split_idx:]
    
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
    
    # Save results in the same format as your existing results
    results_path = os.path.join(OUTPUT_DIR, "attack_metrics.txt")
    with open(results_path, 'w') as f:
        f.write("Attack Method\tAUC\tAccuracy\tTPR@FPR=0.1\n")
        f.write("-" * 60 + "\n")
        
        # Order the attack methods to match your existing format
        attack_methods = [
            "ppl", "ppl/zlib", "ppl/lowercase_ppl",
            "Min_0%_Prob", "Min_5%_Prob", "Min_10%_Prob", "Min_20%_Prob",
            "Min_30%_Prob", "Min_40%_Prob", "Min_50%_Prob", "Min_60%_Prob",
            "Min_70%_Prob", "Min_80%_Prob", "Min_90%_Prob",
            "Modified_entropy", "Modified_renyi_05", "Modified_renyi_2"
        ]
        
        for attack_name in attack_methods:
            if attack_name in attack_results:
                result = attack_results[attack_name]
                f.write(f"{attack_name}\t{result['auc']:.4f}\t{result['accuracy']:.4f}\t{result['tpr@fpr=0.1']:.4f}\n")
    
    # Plot ROC curves
    plot_roc_curves(attack_results, OUTPUT_DIR)
    
    # Save detailed results
    torch.save({
        'attack_results': attack_results,
        'member_features': member_features,
        'non_member_features': non_member_features
    }, os.path.join(OUTPUT_DIR, 'results.pt'))
    
    logger.info(f"Augmented DALL-E results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
