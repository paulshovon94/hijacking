#!/usr/bin/env python3
"""
Feature sanitization script to fix corrupted feature files.
Replaces NaN and Inf values with safe alternatives.
"""

import numpy as np
import glob
import os
import shutil
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_array(arr, method='zero', nan_replacement=0.0, inf_replacement=1e6):
    """
    Sanitize a numpy array by replacing NaN and Inf values.
    
    Args:
        arr: Input numpy array
        method: Sanitization method ('zero', 'mean', 'median', 'clamp')
        nan_replacement: Value to replace NaN with
        inf_replacement: Value to replace Inf with
    
    Returns:
        Sanitized numpy array
    """
    if not np.any(np.isnan(arr)) and not np.any(np.isinf(arr)):
        return arr  # No corruption, return as-is
    
    # Create a copy to avoid modifying original
    arr_clean = arr.copy()
    
    # Replace NaN values
    if np.any(np.isnan(arr_clean)):
        if method == 'zero':
            arr_clean = np.nan_to_num(arr_clean, nan=nan_replacement)
        elif method == 'mean':
            mean_val = np.nanmean(arr_clean)
            arr_clean = np.nan_to_num(arr_clean, nan=mean_val)
        elif method == 'median':
            median_val = np.nanmedian(arr_clean)
            arr_clean = np.nan_to_num(arr_clean, nan=median_val)
        elif method == 'clamp':
            # Replace NaN with min/max of finite values
            finite_vals = arr_clean[np.isfinite(arr_clean)]
            if len(finite_vals) > 0:
                min_val = np.min(finite_vals)
                max_val = np.max(finite_vals)
                arr_clean = np.nan_to_num(arr_clean, nan=min_val)
            else:
                arr_clean = np.nan_to_num(arr_clean, nan=nan_replacement)
    
    # Replace Inf values
    if np.any(np.isinf(arr_clean)):
        arr_clean = np.clip(arr_clean, -inf_replacement, inf_replacement)
    
    return arr_clean

def sanitize_file(file_path, method='zero', backup=True, dry_run=False):
    """
    Sanitize a single feature file.
    
    Args:
        file_path: Path to the feature file
        method: Sanitization method
        backup: Whether to create a backup
        dry_run: If True, don't actually modify files
    
    Returns:
        Tuple of (was_corrupted, corruption_details)
    """
    try:
        # Load the file
        arr = np.load(file_path)
        
        # Check for corruption
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        not_finite = not np.all(np.isfinite(arr))
        
        if not (has_nan or has_inf or not_finite):
            return False, None  # No corruption
        
        corruption_details = {
            'path': file_path,
            'shape': arr.shape,
            'has_nan': has_nan,
            'nan_count': np.isnan(arr).sum() if has_nan else 0,
            'has_inf': has_inf,
            'inf_count': np.isinf(arr).sum() if has_inf else 0,
            'not_finite': not_finite,
            'original_min': np.nanmin(arr) if has_nan else arr.min(),
            'original_max': np.nanmax(arr) if has_nan else arr.max(),
            'original_mean': np.nanmean(arr) if has_nan else arr.mean(),
            'original_std': np.nanstd(arr) if has_nan else arr.std()
        }
        
        if dry_run:
            logger.info(f"[DRY RUN] Would sanitize: {file_path}")
            logger.info(f"  NaN: {has_nan} ({corruption_details['nan_count']} values)")
            logger.info(f"  Inf: {has_inf} ({corruption_details['inf_count']} values)")
            return True, corruption_details
        
        # Create backup if requested
        if backup:
            backup_path = str(file_path) + '.backup'
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Sanitize the array
        arr_clean = sanitize_array(arr, method=method)
        
        # Save the sanitized array
        np.save(file_path, arr_clean)
        
        # Log the changes
        logger.info(f"Sanitized: {file_path}")
        logger.info(f"  Method: {method}")
        logger.info(f"  NaN values replaced: {corruption_details['nan_count']}")
        logger.info(f"  Inf values replaced: {corruption_details['inf_count']}")
        
        return True, corruption_details
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False, None

def sanitize_all_features(method='zero', backup=True, dry_run=False, pattern="multimodal_dataset/**/x*_batch_*.npy"):
    """
    Sanitize all feature files matching the pattern.
    
    Args:
        method: Sanitization method
        backup: Whether to create backups
        dry_run: If True, don't actually modify files
        pattern: Glob pattern to match files
    
    Returns:
        Summary statistics
    """
    logger.info(f"Starting feature sanitization...")
    logger.info(f"Method: {method}")
    logger.info(f"Backup: {backup}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Pattern: {pattern}")
    
    # Find all matching files
    file_paths = glob.glob(pattern, recursive=True)
    logger.info(f"Found {len(file_paths)} files to process")
    
    if not file_paths:
        logger.warning("No files found matching the pattern!")
        return
    
    # Process files
    total_processed = 0
    total_corrupted = 0
    corruption_summary = {
        'nan_files': 0,
        'inf_files': 0,
        'total_nan_values': 0,
        'total_inf_values': 0
    }
    
    for file_path in file_paths:
        was_corrupted, details = sanitize_file(file_path, method, backup, dry_run)
        
        if was_corrupted and details:
            total_corrupted += 1
            corruption_summary['nan_files'] += int(details['has_nan'])
            corruption_summary['inf_files'] += int(details['has_inf'])
            corruption_summary['total_nan_values'] += details['nan_count']
            corruption_summary['total_inf_values'] += details['inf_count']
        
        total_processed += 1
        
        if total_processed % 100 == 0:
            logger.info(f"Processed {total_processed}/{len(file_paths)} files...")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SANITIZATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files processed: {total_processed}")
    logger.info(f"Total corrupted files: {total_corrupted}")
    logger.info(f"Files with NaN: {corruption_summary['nan_files']}")
    logger.info(f"Files with Inf: {corruption_summary['inf_files']}")
    logger.info(f"Total NaN values replaced: {corruption_summary['total_nan_values']}")
    logger.info(f"Total Inf values replaced: {corruption_summary['total_inf_values']}")
    
    if dry_run:
        logger.info("\n⚠️  This was a DRY RUN - no files were actually modified!")
        logger.info("Run without --dry-run to actually sanitize the files.")
    
    return corruption_summary

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Sanitize corrupted feature files")
    parser.add_argument('--method', choices=['zero', 'mean', 'median', 'clamp'], 
                       default='zero', help='Sanitization method')
    parser.add_argument('--no-backup', action='store_true', 
                       help='Skip creating backup files')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without modifying files')
    parser.add_argument('--pattern', default="multimodal_dataset/**/x*_batch_*.npy",
                       help='File pattern to match')
    
    args = parser.parse_args()
    
    # We're already in the imdb_code directory, no need to change
    # os.chdir("summarization/imdb_code")  # Commented out since we're already here
    
    # Run sanitization
    sanitize_all_features(
        method=args.method,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        pattern=args.pattern
    )

if __name__ == "__main__":
    main()
