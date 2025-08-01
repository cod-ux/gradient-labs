#!/usr/bin/env python3
"""
Migration script to move existing data files to the new organized structure.
"""

import sys
import shutil
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import FileManager


def migrate_data_files():
    """Migrate existing files to the new organized structure."""
    print("=" * 60)
    print("DATA FILE MIGRATION")
    print("=" * 60)
    
    file_manager = FileManager()
    file_manager.ensure_directories()
    
    # Get legacy file paths
    legacy_files = file_manager.get_legacy_file_paths()
    
    moved_count = 0
    
    # Migrate ontology file
    if legacy_files["ontology"].exists():
        dest = file_manager.get_ontology_path()
        print(f"Moving ontology: {legacy_files['ontology']} -> {dest}")
        shutil.move(str(legacy_files["ontology"]), str(dest))
        moved_count += 1
    
    # Migrate raw conversations
    if legacy_files["conversations"].exists():
        dest = file_manager.get_raw_conversations_path()
        print(f"Moving conversations: {legacy_files['conversations']} -> {dest}")
        shutil.move(str(legacy_files["conversations"]), str(dest))
        moved_count += 1
    
    # Migrate agglomerative cluster files
    for cluster_file in legacy_files["clusters_agg"]:
        # Extract threshold from filename
        filename = cluster_file.name
        if "agglomerative_" in filename:
            threshold_str = filename.split("agglomerative_")[1].replace(".json", "")
            try:
                threshold = float(threshold_str)
                dest = file_manager.get_cluster_path("agglomerative", threshold)
                print(f"Moving cluster file: {cluster_file} -> {dest}")
                shutil.move(str(cluster_file), str(dest))
                moved_count += 1
            except ValueError:
                print(f"Could not parse threshold from {filename}, skipping")
    
    # Migrate HDBSCAN cluster files
    for cluster_file in legacy_files["clusters_hdb"]:
        # Extract parameter from filename
        filename = cluster_file.name
        if "hdbscan_" in filename:
            param_str = filename.split("hdbscan_")[1].replace(".json", "")
            try:
                param = float(param_str)
                dest = file_manager.get_cluster_path("hdbscan", param)
                print(f"Moving cluster file: {cluster_file} -> {dest}")
                shutil.move(str(cluster_file), str(dest))
                moved_count += 1
            except ValueError:
                print(f"Could not parse parameter from {filename}, skipping")
    
    # Migrate agglomerative intent category files
    for intent_file in legacy_files["intents_agg"]:
        # Extract threshold from filename
        filename = intent_file.name
        if "clustered_customer_intents_" in filename and "hdbscan" not in filename:
            # Handle both regular and merged files
            is_merged = "_merged" in filename
            threshold_part = filename.replace("clustered_customer_intents_", "").replace(".json", "")
            if is_merged:
                threshold_part = threshold_part.replace("_merged", "")
            
            try:
                threshold = float(threshold_part)
                dest = file_manager.get_intent_categories_path("agglomerative", threshold, is_merged)
                print(f"Moving intent categories: {intent_file} -> {dest}")
                shutil.move(str(intent_file), str(dest))
                moved_count += 1
            except ValueError:
                print(f"Could not parse threshold from {filename}, skipping")
    
    # Migrate HDBSCAN intent category files
    for intent_file in legacy_files["intents_hdb"]:
        # Extract parameter from filename
        filename = intent_file.name
        if "clustered_customer_intents_hdbscan_" in filename:
            param_str = filename.split("hdbscan_")[1].replace(".json", "")
            try:
                param = float(param_str)
                dest = file_manager.get_intent_categories_path("hdbscan", param)
                print(f"Moving intent categories: {intent_file} -> {dest}")
                shutil.move(str(intent_file), str(dest))
                moved_count += 1
            except ValueError:
                print(f"Could not parse parameter from {filename}, skipping")
    
    # Migrate classified conversation files
    for classified_file in legacy_files["classified"]:
        # Try to determine method and parameter from filename or content
        filename = classified_file.name
        if "classified_customer_conversations" in filename:
            # For now, assume it's from agglomerative clustering with default threshold
            # In practice, you might need more sophisticated logic to determine the source
            dest = file_manager.get_classified_conversations_path("agglomerative", 0.6)
            print(f"Moving classified conversations: {classified_file} -> {dest}")
            shutil.move(str(classified_file), str(dest))
            moved_count += 1
    
    # Migrate comparison reports
    for report_file in legacy_files["reports"]:
        filename = report_file.stem
        dest = file_manager.get_comparison_report_path(filename)
        print(f"Moving report: {report_file} -> {dest}")
        shutil.move(str(report_file), str(dest))
        moved_count += 1
    
    # Migrate visualizations
    for viz_file in legacy_files["visualizations"]:
        filename = viz_file.name
        dest = file_manager.get_visualization_path(filename.replace(".html", ""))
        print(f"Moving visualization: {viz_file} -> {dest}")
        shutil.move(str(viz_file), str(dest))
        moved_count += 1
    
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Moved {moved_count} files to organized structure")
    print("✅ All files are now in their proper locations")
    
    # Show the new structure
    print("\nNew directory structure:")
    for directory in [
        file_manager.raw_dir,
        file_manager.ontologies_dir,
        file_manager.clusters_dir,
        file_manager.intent_categories_dir,
        file_manager.evaluations_dir,
        file_manager.visualizations_dir
    ]:
        if directory.exists():
            files = list(directory.rglob("*"))
            if files:
                print(f"  {directory.name}/: {len([f for f in files if f.is_file()])} files")


if __name__ == "__main__":
    migrate_data_files()