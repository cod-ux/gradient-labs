#!/usr/bin/env python3
"""
Migration script to rename files according to the new nomenclature.

OLD STRUCTURE:
- data/ontologies/ -> Contains initial intents (wrong name)
- data/intent_categories/ -> Contains final ontology (wrong name)

NEW STRUCTURE:  
- data/initial_intents/ -> Contains initial intents from conversations
- data/ontologies/ -> Contains final clustered categories (the actual ontology)
"""

import sys
import shutil
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import FileManager


def migrate_nomenclature():
    """Migrate files to correct nomenclature."""
    print("=" * 60)
    print("NOMENCLATURE MIGRATION")
    print("=" * 60)
    
    file_manager = FileManager()
    file_manager.ensure_directories()
    
    moved_count = 0
    data_dir = Path("data")
    
    # 1. Move initial intents (currently in ontologies/ folder)
    old_ontologies_dir = data_dir / "ontologies"
    if old_ontologies_dir.exists():
        for file_path in old_ontologies_dir.glob("*.json"):
            if "customer_intent_ontology" in file_path.name:
                new_name = file_path.name.replace("customer_intent_ontology", "initial_intents")
                dest = file_manager.initial_intents_dir / new_name
                print(f"Moving initial intents: {file_path} -> {dest}")
                shutil.move(str(file_path), str(dest))
                moved_count += 1
    
    # 2. Move final ontologies (currently in intent_categories/ folder)  
    old_intent_categories_dir = data_dir / "intent_categories"
    if old_intent_categories_dir.exists():
        # Move agglomerative files
        agg_dir = old_intent_categories_dir / "agglomerative"
        if agg_dir.exists():
            for file_path in agg_dir.glob("*.json"):
                # Parse threshold from filename
                filename = file_path.name
                if "clustered_customer_intents_" in filename:
                    # Extract threshold
                    threshold_part = filename.replace("clustered_customer_intents_", "").replace(".json", "")
                    is_merged = "_merged" in threshold_part
                    if is_merged:
                        threshold_part = threshold_part.replace("_merged", "")
                        new_name = f"ontology_agg_{threshold_part}_merged.json"
                    else:
                        new_name = f"ontology_agg_{threshold_part}.json"
                    
                    dest = file_manager.ontologies_dir / "agglomerative" / new_name
                    print(f"Moving ontology: {file_path} -> {dest}")
                    shutil.move(str(file_path), str(dest))
                    moved_count += 1
        
        # Move HDBSCAN files
        hdb_dir = old_intent_categories_dir / "hdbscan"
        if hdb_dir.exists():
            for file_path in hdb_dir.glob("*.json"):
                # Parse parameter from filename
                filename = file_path.name
                if "clustered_customer_intents_hdbscan_" in filename:
                    param_part = filename.replace("clustered_customer_intents_hdbscan_", "").replace(".json", "")
                    new_name = f"ontology_hdbscan_{param_part}.json"
                    
                    dest = file_manager.ontologies_dir / "hdbscan" / new_name
                    print(f"Moving ontology: {file_path} -> {dest}")
                    shutil.move(str(file_path), str(dest))
                    moved_count += 1
    
    # 3. Clean up empty directories
    if old_ontologies_dir.exists() and not any(old_ontologies_dir.iterdir()):
        old_ontologies_dir.rmdir()
        print(f"Removed empty directory: {old_ontologies_dir}")
    
    if old_intent_categories_dir.exists():
        for subdir in old_intent_categories_dir.iterdir():
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()
                print(f"Removed empty directory: {subdir}")
        if not any(old_intent_categories_dir.iterdir()):
            old_intent_categories_dir.rmdir()
            print(f"Removed empty directory: {old_intent_categories_dir}")
    
    print("\n" + "=" * 60)
    print("NOMENCLATURE MIGRATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Moved {moved_count} files to correct naming structure")
    
    # Show the new structure
    print("\nNew directory structure:")
    if file_manager.initial_intents_dir.exists():
        files = list(file_manager.initial_intents_dir.glob("*.json"))
        print(f"  initial_intents/: {len(files)} files")
    
    if file_manager.ontologies_dir.exists():
        agg_files = list((file_manager.ontologies_dir / "agglomerative").glob("*.json"))
        hdb_files = list((file_manager.ontologies_dir / "hdbscan").glob("*.json"))
        print(f"  ontologies/agglomerative/: {len(agg_files)} files")
        print(f"  ontologies/hdbscan/: {len(hdb_files)} files")
    
    print("\nNomenclature is now correct:")
    print("  📄 initial_intents/ - Raw intents from conversations (Stage 1)")
    print("  🏗️  clusters/ - Detailed clustering results (Stage 2)")  
    print("  🎯 ontologies/ - Final clustered categories (Stage 3)")


if __name__ == "__main__":
    migrate_nomenclature()