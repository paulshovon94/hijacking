#!/usr/bin/env python3
"""
Diagnostic script to identify corrupted feature files in the multimodal dataset.
Scans all x1-x7 feature files for NaN, Inf, or non-finite values.
"""

import numpy as np
import glob
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scan(pattern):
    """Scan files matching pattern for corruption."""
    paths = glob.glob(pattern, recursive=True)
    bad = []
    corrupted_details = []
    
    logger.info(f"Scanning {len(paths)} files matching pattern: {pattern}")
    
    for p in paths:
        try:
            a = np.load(p)
            
            # Check for various types of corruption
            has_nan = np.isnan(a).any()
            has_inf = np.isinf(a).any()
            not_finite = not np.all(np.isfinite(a))
            
            if has_nan or has_inf or not_finite:
                bad.append(p)
                
                # Get detailed corruption info
                corruption_info = {
                    'path': p,
                    'shape': a.shape,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'not_finite': not_finite,
                    'nan_count': np.isnan(a).sum() if has_nan else 0,
                    'inf_count': np.isinf(a).sum() if has_inf else 0,
                    'min_val': np.nanmin(a) if has_nan else a.min(),
                    'max_val': np.nanmax(a) if has_nan else a.max(),
                    'mean_val': np.nanmean(a) if has_nan else a.mean(),
                    'std_val': np.nanstd(a) if has_nan else a.std()
                }
                corrupted_details.append(corruption_info)
                
                logger.warning(f"Corrupted file: {p}")
                logger.warning(f"  Shape: {a.shape}, NaN: {has_nan}, Inf: {has_inf}, Not finite: {not_finite}")
                
        except Exception as e:
            logger.error(f"Error loading {p}: {str(e)}")
            bad.append(p)
    
    logger.info(f"Found {len(bad)} corrupted files out of {len(paths)} total files")
    return bad, corrupted_details

def analyze_corruption_patterns(corrupted_details):
    """Analyze patterns in corrupted files to identify root causes."""
    if not corrupted_details:
        logger.info("No corrupted files found to analyze")
        return
    
    logger.info("\n" + "="*60)
    logger.info("CORRUPTION ANALYSIS")
    logger.info("="*60)
    
    # Group by model family
    model_family_stats = {}
    for detail in corrupted_details:
        path = detail['path']
        if 'bart' in path.lower():
            family = 'BART'
        elif 'pegasus' in path.lower():
            family = 'Pegasus'
        elif 'gpt-2' in path.lower() or 'gpt2' in path.lower():
            family = 'GPT-2'
        else:
            family = 'Unknown'
        
        if family not in model_family_stats:
            model_family_stats[family] = {'count': 0, 'nan_count': 0, 'inf_count': 0}
        
        model_family_stats[family]['count'] += 1
        model_family_stats[family]['nan_count'] += detail['nan_count']
        model_family_stats[family]['inf_count'] += detail['inf_count']
    
    logger.info("Corruption by Model Family:")
    for family, stats in model_family_stats.items():
        logger.info(f"  {family}: {stats['count']} corrupted files")
        logger.info(f"    Total NaN values: {stats['nan_count']}")
        logger.info(f"    Total Inf values: {stats['inf_count']}")
    
    # Analyze feature types (x1-x7)
    feature_stats = {}
    for detail in corrupted_details:
        path = detail['path']
        for i in range(1, 8):
            if f'x{i}_batch_' in path:
                feature = f'x{i}'
                break
        
        if feature not in feature_stats:
            feature_stats[feature] = {'count': 0, 'nan_count': 0, 'inf_count': 0}
        
        feature_stats[feature]['count'] += 1
        feature_stats[feature]['nan_count'] += detail['nan_count']
        feature_stats[feature]['inf_count'] += detail['inf_count']
    
    logger.info("\nCorruption by Feature Type:")
    for feature, stats in feature_stats.items():
        logger.info(f"  {feature}: {stats['count']} corrupted files")
        logger.info(f"    Total NaN values: {stats['nan_count']}")
        logger.info(f"    Total Inf values: {stats['inf_count']}")
    
    # Value range analysis
    all_nan_counts = [d['nan_count'] for d in corrupted_details if d['nan_count'] > 0]
    all_inf_counts = [d['inf_count'] for d in corrupted_details if d['inf_count'] > 0]
    
    if all_nan_counts:
        logger.info(f"\nNaN Statistics:")
        logger.info(f"  Min NaN count per file: {min(all_nan_counts)}")
        logger.info(f"  Max NaN count per file: {max(all_nan_counts)}")
        logger.info(f"  Mean NaN count per file: {np.mean(all_nan_counts):.2f}")
    
    if all_inf_counts:
        logger.info(f"\nInf Statistics:")
        logger.info(f"  Min Inf count per file: {min(all_inf_counts)}")
        logger.info(f"  Max Inf count per file: {max(all_inf_counts)}")
        logger.info(f"  Mean Inf count per file: {np.mean(all_inf_counts):.2f}")

def main():
    """Main diagnostic function."""
    logger.info("Starting corruption diagnosis...")
    
    # We're already in the imdb_code directory, no need to change
    # os.chdir("summarization/imdb_code")  # Commented out since we're already here
    
    bad_all = []
    all_corrupted_details = []
    
    # Scan each feature type
    for k in range(1, 8):
        pattern = f"multimodal_dataset/**/x{k}_batch_*.npy"
        logger.info(f"\nScanning x{k} features...")
        
        bad_files, corrupted_details = scan(pattern)
        bad_all.extend(bad_files)
        all_corrupted_details.extend(corrupted_details)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total corrupted files found: {len(bad_all)}")
    
    if bad_all:
        logger.info("\nFirst 10 corrupted files:")
        for i, bad_file in enumerate(bad_all[:10]):
            logger.info(f"  {i+1}. {bad_file}")
        
        if len(bad_all) > 10:
            logger.info(f"  ... and {len(bad_all) - 10} more")
        
        # Analyze patterns
        analyze_corruption_patterns(all_corrupted_details)
        
        # Save list of corrupted files
        with open("corrupted_files.txt", "w") as f:
            for bad_file in bad_all:
                f.write(f"{bad_file}\n")
        logger.info(f"\nList of corrupted files saved to: corrupted_files.txt")
        
    else:
        logger.info("✅ No corrupted files found! All feature files are clean.")
    
    return bad_all, all_corrupted_details

if __name__ == "__main__":
    main()
