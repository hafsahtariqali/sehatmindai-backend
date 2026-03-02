"""
Diagnostic script to check model directory structure and identify issues.
Run this before using predict.py to ensure the model is set up correctly.
"""

import json
from pathlib import Path

def check_model_directory():
    """Check the model directory structure and identify potential issues."""
    
    model_dir = Path(__file__).parent / "model"
    
    print("=" * 70)
    print("MODEL DIRECTORY DIAGNOSTIC")
    print("=" * 70)
    print(f"\nChecking directory: {model_dir}")
    print()
    
    if not model_dir.exists():
        print("[ERROR] Model directory does not exist!")
        print(f"   Expected: {model_dir}")
        return
    
    # Check root directory files
    print("[ROOT DIRECTORY FILES]")
    root_files = list(model_dir.iterdir())
    root_files = [f for f in root_files if f.is_file()]
    
    root_model_file = None
    root_config = model_dir / "config.json"
    
    for f in root_files:
        size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
        if f.name.endswith(('.safetensors', '.bin')):
            print(f"   [OK] {f.name} ({size_mb:.1f} MB)")
            root_model_file = f
        elif f.name == "config.json":
            print(f"   [OK] {f.name}")
        else:
            print(f"   [-] {f.name}")
    
    print()
    
    # Check checkpoint directories
    print("[CHECKPOINT DIRECTORIES]")
    checkpoint_dirs = sorted(
        [d for d in model_dir.iterdir() 
         if d.is_dir() and d.name.startswith('checkpoint')],
        key=lambda x: int(x.name.split('-')[-1]) if x.name.split('-')[-1].isdigit() else 0
    )
    
    if not checkpoint_dirs:
        print("   [ERROR] No checkpoint directories found!")
        print("   [WARN]  If you trained in Colab, checkpoint folders should exist.")
        print("   [WARN]  The root model.safetensors might be from initialization (UNTRAINED).")
        return
    
    print(f"   Found {len(checkpoint_dirs)} checkpoint(s):")
    
    best_checkpoint = None
    best_checkpoint_info = None
    
    for checkpoint_dir in checkpoint_dirs:
        print(f"\n   [DIR] {checkpoint_dir.name}/")
        
        # Check for model files
        model_files = list(checkpoint_dir.glob("*.safetensors")) + list(checkpoint_dir.glob("*.bin"))
        config_file = checkpoint_dir / "config.json"
        trainer_state = checkpoint_dir / "trainer_state.json"
        
        if model_files:
            size_mb = model_files[0].stat().st_size / (1024 * 1024)
            print(f"      [OK] model.safetensors or .bin exists ({size_mb:.1f} MB)")
            
            if size_mb < 100:
                print(f"      [WARN] Model file is too small! Expected ~260MB")
                print(f"      [WARN] This might be corrupted or incomplete.")
            elif size_mb > 200:
                print(f"      [OK] Model file size looks correct")
        else:
            print(f"      [ERROR] No model files found! This checkpoint is empty/incomplete.")
            continue
        
        if config_file.exists():
            print(f"      [OK] config.json exists")
        
        if trainer_state.exists():
            print(f"      [OK] trainer_state.json exists")
            try:
                with open(trainer_state, 'r') as f:
                    state = json.load(f)
                    
                step = int(checkpoint_dir.name.split('-')[-1])
                best_step = state.get('best_global_step', None)
                best_metric = state.get('best_metric', None)
                
                print(f"      Step: {step}")
                if best_step:
                    print(f"      Best global step: {best_step}")
                    print(f"      Best metric: {best_metric}")
                    
                    if best_step == step:
                        print(f"      [BEST] THIS IS THE BEST MODEL!")
                        best_checkpoint = checkpoint_dir
                        best_checkpoint_info = {
                            'step': step,
                            'metric': best_metric,
                            'size_mb': size_mb
                        }
                else:
                    print(f"      (No best_global_step found)")
            except Exception as e:
                print(f"      [WARN] Could not read trainer_state.json: {e}")
        else:
            print(f"      [-] trainer_state.json not found")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)
    
    if best_checkpoint:
        print(f"\n[SUCCESS] RECOMMENDED MODEL FOUND:")
        print(f"   Use: {best_checkpoint}")
        print(f"   Step: {best_checkpoint_info['step']}")
        print(f"   Metric: {best_checkpoint_info['metric']}")
        print(f"   Size: {best_checkpoint_info['size_mb']:.1f} MB")
        print(f"\n   The predict.py script should automatically use this checkpoint.")
        
        # Check if root model file exists and warn
        if root_model_file:
            root_size = root_model_file.stat().st_size / (1024 * 1024)
            print(f"\n   [WARN] NOTE: Root directory also has model.safetensors ({root_size:.1f} MB)")
            print(f"      This might be from initialization. The script should use the checkpoint instead.")
    
    elif checkpoint_dirs:
        latest = checkpoint_dirs[-1]
        print(f"\n[WARN] USING LATEST CHECKPOINT (best not identified):")
        print(f"   {latest}")
        print(f"   Make sure this checkpoint has model files!")
    
    else:
        if root_model_file:
            root_size = root_model_file.stat().st_size / (1024 * 1024)
            print(f"\n[WARN] WARNING: Only root directory model found ({root_size:.1f} MB)")
            print(f"   This model might be UNTRAINED (from initialization).")
            print(f"   If you trained in Colab, you should have checkpoint-XXXXX folders.")
            print(f"   Please re-download the model folder, ensuring all subdirectories are included.")
        else:
            print(f"\n[ERROR] No valid model files found!")
            print(f"   Please check your model directory structure.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_model_directory()

