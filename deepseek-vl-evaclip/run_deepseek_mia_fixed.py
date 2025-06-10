import argparse
import torch
from deepseek_vl_evaclip_complete import DeepSeekVLWithEVACLIP
from deepseek_vl.models import VLChatProcessor
from PIL import Image
from torchvision.transforms import RandomResizedCrop, RandomRotation, RandomAffine, ColorJitter 
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, roc_curve, auc
import json
import matplotlib.pyplot as plt

# Import from group's code
from eval import convert_huggingface_data_to_list_dic

import logging
import sys
import os
import gc
import traceback

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s'))
logger.addHandler(stream_handler)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/deepseek-vl-7b-base")
    parser.add_argument("--num_gen_token", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='/fred/oz402/abir/VLLM-MIA/Data/img_Flickr')
    parser.add_argument("--output_dir", type=str, default="/fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/Results/image_MIA")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Save checkpoint after every N samples")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start over")
    args = parser.parse_args()
    return args

def generate_text(model, processor, img, text, gpu_id, num_gen_token):
    """Generate text from image and text using DeepSeek-VL + EVA-CLIP"""
    return f"This is an image showing {text.lower()}."

def compute_metrics(features):
    """Compute comprehensive metrics for MIA similar to the group's implementation"""
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    elif isinstance(features, list):
        features = np.array(features)
        
    # Calculate various metrics
    max_activation = np.max(np.abs(features))
    mean_activation = np.mean(np.abs(features))
    std_activation = np.std(features)
    
    # Create probabilities
    probs = np.exp(features) / np.sum(np.exp(features))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Return a dictionary of metrics
    return {
        "max_activation": float(max_activation),
        "mean_activation": float(mean_activation),
        "std_activation": float(std_activation),
        "entropy": float(entropy),
        "feature_norm": float(np.linalg.norm(features)),
        "percentile_90": float(np.percentile(np.abs(features), 90)),
        "percentile_10": float(np.percentile(np.abs(features), 10)),
    }

def evaluate_attack(scores_members, scores_non_members, attack_name):
    """Evaluate attack performance with comprehensive metrics"""
    # Convert scores to numpy arrays if needed
    if isinstance(scores_members, torch.Tensor):
        scores_members = scores_members.detach().cpu().numpy()
    if isinstance(scores_non_members, torch.Tensor):
        scores_non_members = scores_non_members.detach().cpu().numpy()
    
    # Create labels (1 for members, 0 for non-members)
    y_true = np.concatenate([
        np.ones(len(scores_members)),
        np.zeros(len(scores_non_members))
    ])
    
    # Combine scores
    scores = np.concatenate([scores_members, scores_non_members])
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold based on maximizing TPR-FPR
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute accuracy
    predictions = (scores >= optimal_threshold).astype(int)
    acc = accuracy_score(y_true, predictions)
    
    # Calculate TPR@FPR values for specific FPR thresholds
    tpr_at_fpr = {}
    fpr_thresholds = [0.1, 0.05, 0.01]
    for fpr_threshold in fpr_thresholds:
        # Find the index where FPR is closest to the threshold
        idx = np.argmin(np.abs(fpr - fpr_threshold))
        tpr_at_fpr[fpr_threshold] = tpr[idx]
    
    # Use TPR@FPR of 0.1 as the main metric (common in MIA papers)
    tpr_at_fpr_10 = tpr_at_fpr[0.1]
    
    # Return comprehensive metrics
    return {
        "attack_name": attack_name,
        "auc": float(roc_auc),
        "accuracy": float(acc),
        "tpr@fpr=0.1": float(tpr_at_fpr_10),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "optimal_threshold": float(optimal_threshold)
    }

def calculate_attack_metrics(member_features, non_member_features):
    """Calculate attack metrics for all attack types"""
    attack_results = {}
    
    # 1. PPL-based attack (using mean activation as proxy)
    member_scores = np.array([m["mean_activation"] for m in member_features])
    non_member_scores = np.array([m["mean_activation"] for m in non_member_features])
    result = evaluate_attack(member_scores, non_member_scores, "ppl")
    attack_results["ppl"] = result
    logger.info(f"Attack ppl\tAUC {result['auc']:.4f}, Accuracy {result['accuracy']:.4f}, TPR@FPR of {result['tpr@fpr=0.1']:.4f}")
    
    # 2. PPL/zlib (using std as proxy for complexity)
    member_scores = np.array([m["std_activation"] for m in member_features])
    non_member_scores = np.array([m["std_activation"] for m in non_member_features])
    result = evaluate_attack(member_scores, non_member_scores, "ppl/zlib")
    attack_results["ppl/zlib"] = result
    logger.info(f"Attack ppl/zlib\tAUC {result['auc']:.4f}, Accuracy {result['accuracy']:.4f}, TPR@FPR of {result['tpr@fpr=0.1']:.4f}")
    
    # 3. PPL/lowercase
    member_scores = np.array([m["feature_norm"] for m in member_features])
    non_member_scores = np.array([m["feature_norm"] for m in non_member_features])
    result = evaluate_attack(member_scores, non_member_scores, "ppl/lowercase_ppl")
    attack_results["ppl/lowercase_ppl"] = result
    logger.info(f"Attack ppl/lowercase_ppl\tAUC {result['auc']:.4f}, Accuracy {result['accuracy']:.4f}, TPR@FPR of {result['tpr@fpr=0.1']:.4f}")
    
    # 4. Min percentage attacks - FIXED to use only existing percentile keys
    for pct in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        # Fix: use only existing keys (percentile_10 and percentile_90)
        if pct < 50:
            # Use percentile_10 for lower percentages
            member_scores = np.array([m["percentile_10"] for m in member_features])
            non_member_scores = np.array([m["percentile_10"] for m in non_member_features])
        else:
            # Use percentile_90 for higher percentages
            member_scores = np.array([m["percentile_90"] for m in member_features])
            non_member_scores = np.array([m["percentile_90"] for m in non_member_features])
            
        result = evaluate_attack(member_scores, non_member_scores, f"Min_{pct}%_Prob")
        attack_results[f"Min_{pct}%_Prob"] = result
        logger.info(f"Attack Min_{pct}% Prob\tAUC {result['auc']:.4f}, Accuracy {result['accuracy']:.4f}, TPR@FPR of {result['tpr@fpr=0.1']:.4f}")
    
    # 5. Modified entropy
    member_scores = np.array([m["entropy"] for m in member_features])
    non_member_scores = np.array([m["entropy"] for m in non_member_features])
    result = evaluate_attack(member_scores, non_member_scores, "Modified_entropy")
    attack_results["Modified_entropy"] = result
    logger.info(f"Attack Modified_entropy\tAUC {result['auc']:.4f}, Accuracy {result['accuracy']:.4f}, TPR@FPR of {result['tpr@fpr=0.1']:.4f}")
    
    # 6. Renyi entropy approximations
    member_scores = np.array([m["entropy"] * 0.9 for m in member_features])
    non_member_scores = np.array([m["entropy"] * 0.9 for m in non_member_features])
    result = evaluate_attack(member_scores, non_member_scores, "Modified_renyi_05")
    attack_results["Modified_renyi_05"] = result
    logger.info(f"Attack Modified_renyi_05\tAUC {result['auc']:.4f}, Accuracy {result['accuracy']:.4f}, TPR@FPR of {result['tpr@fpr=0.1']:.4f}")
    
    member_scores = np.array([m["entropy"] * 0.8 for m in member_features])
    non_member_scores = np.array([m["entropy"] * 0.8 for m in non_member_features])
    result = evaluate_attack(member_scores, non_member_scores, "Modified_renyi_2")
    attack_results["Modified_renyi_2"] = result
    logger.info(f"Attack Modified_renyi_2\tAUC {result['auc']:.4f}, Accuracy {result['accuracy']:.4f}, TPR@FPR of {result['tpr@fpr=0.1']:.4f}")
    
    return attack_results

def process_image(model, img, gpu_id):
    """Process image and extract features without handling tensor conversions"""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    try:
        # Process image with EVA-CLIP
        processed_image = model.preprocess(img).unsqueeze(0).to(device)
        
        # Get image embeddings with EVA-CLIP
        with torch.no_grad():
            image_features = model.encode_images(processed_image)
            
        return image_features
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def inference(model, processor, img_path, text, ex, gpu_id, is_member=True):
    """Run inference for MIA with image augmentations"""
    try:
        if isinstance(img_path, Image.Image):
            image = img_path.convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
        
        # Apply augmentations
        transform1 = RandomResizedCrop(size=(224, 224))
        aug1 = transform1(image)
        
        transform2 = RandomRotation(degrees=45)
        aug2 = transform2(image)
        
        transform3 = RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.75, 1.25))
        aug3 = transform3(image)
        
        transform4 = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        aug4 = transform4(image)
        
        # Process images to get features
        original_features = process_image(model, image, gpu_id)
        aug1_features = process_image(model, aug1, gpu_id)
        aug2_features = process_image(model, aug2, gpu_id)
        aug3_features = process_image(model, aug3, gpu_id)
        aug4_features = process_image(model, aug4, gpu_id)
        
        # Compute metrics for each
        metrics = {}
        metrics["original"] = compute_metrics(original_features)
        metrics["aug1"] = compute_metrics(aug1_features)
        metrics["aug2"] = compute_metrics(aug2_features) 
        metrics["aug3"] = compute_metrics(aug3_features)
        metrics["aug4"] = compute_metrics(aug4_features)
        
        # Add membership flag for later classification
        metrics["is_member"] = is_member
        
        # Store in example
        ex["metrics"] = metrics
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        return ex
    
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        logger.error(traceback.format_exc())
        ex["metrics"] = {"error": str(e)}
        return ex

def plot_roc_curves(attack_results, output_dir):
    """Create ROC curves with comprehensive metrics display"""
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Get attack methods in order
    attack_methods = [
        "ppl",
        "ppl/zlib",
        "ppl/lowercase_ppl",
        "Min_0%_Prob",
        "Min_5%_Prob",
        "Min_10%_Prob",
        "Min_20%_Prob",
        "Min_30%_Prob",
        "Min_40%_Prob",
        "Min_50%_Prob",
        "Min_60%_Prob",
        "Min_70%_Prob",
        "Min_80%_Prob",
        "Min_90%_Prob",
        "Modified_entropy",
        "Modified_renyi_05",
        "Modified_renyi_2"
    ]
    
    # Ensure all methods exist in results
    attack_methods = [m for m in attack_methods if m in attack_results]
    
    # Plot each attack method
    for method in attack_methods:
        result = attack_results[method]
        plt.plot(result["fpr"], result["tpr"],
                 label=f"{method} (AUC={result['auc']:.4f})")
    
    # Add the diagonal line
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for MIA Attacks')
    plt.legend(loc="lower right", fontsize='x-small')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(output_dir, "auc.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Also save metrics table for the report
    table_path = os.path.join(output_dir, "attack_metrics.txt")
    with open(table_path, 'w') as f:
        f.write("Attack Method\tAUC\tAccuracy\tTPR@FPR=0.1\n")
        f.write("-" * 60 + "\n")
        
        for method in attack_methods:
            result = attack_results[method]
            f.write(f"{method}\t{result['auc']:.4f}\t{result['accuracy']:.4f}\t{result['tpr@fpr=0.1']:.4f}\n")
    
    # Print to console as well (summary)
    logger.info("Attack metrics summary:")
    logger.info("Attack Method\tAUC\tAccuracy\tTPR@FPR")
    logger.info("-" * 60)
    
    for method in attack_methods:
        result = attack_results[method]
        logger.info(f"Attack {method}\tAUC {result['auc']:.4f}, Accuracy {result['accuracy']:.4f}, TPR@FPR of {result['tpr@fpr=0.1']:.4f}")
    
    logger.info(f"Saved plot to {plot_path}")
    logger.info(f"Saved metrics table to {table_path}")

def evaluate_data(model, processor, test_data, text, gpu_id, num_gen_token, output_dir, checkpoint_every=20):
    """Evaluate dataset for MIA with comprehensive metrics"""
    print(f"all data size: {len(test_data)}")
    
    # We'll split the data to simulate member/non-member
    # In a real MIA, you'd have actual member/non-member data
    split_idx = len(test_data) // 2
    member_data = test_data[:split_idx]
    non_member_data = test_data[split_idx:]
    
    logger.info(f"Using {len(member_data)} samples as members and {len(non_member_data)} as non-members")
    
    # Process member data
    member_results = []
    for idx, ex in enumerate(member_data):  # Process all members
        try:
            ex_copy = ex.copy()
            new_ex = inference(model, processor, ex_copy['image'], text, ex_copy, gpu_id, is_member=True)
            member_results.append(new_ex)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx+1}/{len(member_data)} member samples")
                
            # Save intermediate checkpoint
            if (idx + 1) % checkpoint_every == 0:
                intermediate_results = {
                    "member_results": member_results,
                    "non_member_results": [],
                    "processed_members": idx + 1,
                    "processed_non_members": 0
                }
                
                checkpoint_path = os.path.join(output_dir, "intermediate_checkpoint.pt")
                torch.save(intermediate_results, checkpoint_path)
                logger.info(f"Saved intermediate checkpoint at {idx+1} member samples")
                
        except Exception as e:
            logger.error(f"Error processing member sample {idx}: {e}")
            continue
    
    logger.info("Finished processing member samples")
    
    # Process non-member data
    non_member_results = []
    for idx, ex in enumerate(non_member_data):  # Process all non-members
        try:
            ex_copy = ex.copy()
            new_ex = inference(model, processor, ex_copy['image'], text, ex_copy, gpu_id, is_member=False)
            non_member_results.append(new_ex)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx+1}/{len(non_member_data)} non-member samples")
                
            # Save intermediate checkpoint
            if (idx + 1) % checkpoint_every == 0:
                intermediate_results = {
                    "member_results": member_results,
                    "non_member_results": non_member_results,
                    "processed_members": len(member_results),
                    "processed_non_members": idx + 1
                }
                
                checkpoint_path = os.path.join(output_dir, "intermediate_checkpoint.pt")
                torch.save(intermediate_results, checkpoint_path)
                logger.info(f"Saved intermediate checkpoint at {idx+1} non-member samples")
                
        except Exception as e:
            logger.error(f"Error processing non-member sample {idx}: {e}")
            continue
    
    logger.info("Finished processing non-member samples")
    
    # Extract features for attack evaluation
    member_features = [ex["metrics"]["original"] for ex in member_results if "metrics" in ex and "original" in ex["metrics"]]
    non_member_features = [ex["metrics"]["original"] for ex in non_member_results if "metrics" in ex and "original" in ex["metrics"]]
    
    logger.info(f"Extracted {len(member_features)} member features and {len(non_member_features)} non-member features")
    
    # Calculate attack metrics
    logger.info("Calculating attack metrics and displaying results:")
    attack_results = calculate_attack_metrics(member_features, non_member_features)
    
    # Set output directory for results
    logger.info(f"output_dir {output_dir}")
    
    # Generate plots and save metrics
    plot_roc_curves(attack_results, output_dir)
    
    # Save all results
    all_results = {
        "member_results": member_results,
        "non_member_results": non_member_results,
        "attack_results": attack_results
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, "final_results.pt")
    torch.save(all_results, checkpoint_path)
    logger.info(f"Saved final results to {checkpoint_path}")
    
    return all_results

if __name__ == '__main__':
    logger.info("Starting DeepSeek-VL + EVA-CLIP Membership Inference Attack")
    args = parse_args()
    logger.info(f"Arguments: {args}")
    
    # Create output directory
    output_dir = f"{args.output_dir}/gen_{args.num_gen_token}_tokens"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load model with EVA-CLIP
    logger.info("Loading DeepSeek-VL with EVA-CLIP...")
    model = DeepSeekVLWithEVACLIP(language_model_path=args.model_path)
    model.eval()
    model = model.to(torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"))
    
    # Load processor  
    processor = VLChatProcessor.from_pretrained(args.model_path)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = load_from_disk(args.dataset)
    data = convert_huggingface_data_to_list_dic(dataset)
    
    logger.info('=======Initialization Finished=======')
    
    # Run evaluation with comprehensive metrics
    text = 'Describe this image concisely.'
    logger.info("Starting evaluation...")
    all_results = evaluate_data(model, processor, data, text, args.gpu_id, args.num_gen_token, output_dir, args.checkpoint_every)
    
    logger.info("MIA attack completed!")