"""
GPU Acceleration Setup Script
Run this once to enable GPU acceleration for all your scripts
"""
import torch
import os

def setup_gpu_acceleration():
    """Configure GPU settings for maximum performance"""
    
    # Check GPU availability
    print("=== GPU Status Check ===")
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available - will use CPU")
        return False
    
    # Set environment variables for GPU optimization
    print("\n=== Setting GPU Environment Variables ===")
    
    # Enable GPU memory growth (prevents OOM errors)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Optimize CUDA performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Device-side assertions
    
    # For Transformers library GPU optimization
    os.environ['TRANSFORMERS_CACHE'] = './models/cache'
    os.environ['HF_DATASETS_CACHE'] = './data/cache'
    
    # For spaCy GPU acceleration
    os.environ['SPACY_GPU'] = '1'
    
    print("Environment variables set for GPU acceleration")
    
    # Test GPU tensor operations
    print("\n=== GPU Test ===")
    try:
        # Create test tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("GPU tensor operations working")
        
        # Clear GPU memory
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"GPU test failed: {e}")
        return False
    
    return True

def optimize_batch_sizes():
    """Suggest optimal batch sizes based on GPU memory"""
    if not torch.cuda.is_available():
        return {
            'embedding_batch': 32,
            'model_batch': 1,
            'preprocessing_batch': 64
        }
    
    # Get GPU memory in GB
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if gpu_memory_gb >= 12:  # High-end GPU (RTX 3080+, A100, etc.)
        return {
            'embedding_batch': 256,
            'model_batch': 8,
            'preprocessing_batch': 512
        }
    elif gpu_memory_gb >= 8:  # Mid-range GPU (RTX 3070, etc.)
        return {
            'embedding_batch': 128,
            'model_batch': 4,
            'preprocessing_batch': 256
        }
    else:  # Lower-end GPU
        return {
            'embedding_batch': 64,
            'model_batch': 2,
            'preprocessing_batch': 128
        }

def patch_existing_scripts():
    """Create GPU-optimized versions of your scripts"""
    
    print("\n=== Creating GPU-Optimized Imports ===")
    
    # GPU-optimized imports that your scripts can use
    gpu_imports = '''
# GPU Acceleration - Add this to the top of your scripts
import torch
import os

# Auto-detect and use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# Optimize memory usage
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
'''
    
    # Save GPU imports file
    with open('gpu_imports.py', 'w') as f:
        f.write(gpu_imports)
    
    print(" Created 'gpu_imports.py' - import this in your scripts")
    
    # GPU-optimized functions
    gpu_functions = '''
# GPU Helper Functions - Use these in your scripts

import torch

def move_to_gpu(model):
    """Move model to GPU if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

def gpu_tokenize(tokenizer, texts, batch_size=None):
    """GPU-optimized tokenization"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(texts, str):
        texts = [texts]
    
    # Use recommended batch size
    if batch_size is None:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 4
        batch_size = min(128, int(gpu_memory_gb * 16))
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        results.append(tokens)
    
    return results

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_optimal_batch_size():
    """Get optimal batch size for current GPU"""
    if not torch.cuda.is_available():
        return 32
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return min(256, int(gpu_memory_gb * 16))
'''
    
    with open('gpu_helpers.py', 'w') as f:
        f.write(gpu_functions)
    
    print("Created 'gpu_helpers.py' - use these helper functions")

def main():
    """Main setup function"""
    print("GPU Acceleration Setup Starting...")
    
    # Setup GPU
    gpu_available = setup_gpu_acceleration()
    
    # Get optimal batch sizes
    batch_sizes = optimize_batch_sizes()
    print(f"\n=== Recommended Batch Sizes ===")
    for key, value in batch_sizes.items():
        print(f"{key}: {value}")
    
    # Create helper files
    patch_existing_scripts()
    
    print(f"\n=== Setup Complete ===")
    if gpu_available:
        print("GPU acceleration is ready!")
        print("To use in your scripts:")
        print("1. Add: from gpu_imports import *")
        print("2. Add: from gpu_helpers import *")
        print("3. Use: model = move_to_gpu(model)")
        print("4. Use: clear_gpu_memory() after processing")
    else:
        print("Running on CPU - install CUDA for GPU acceleration")
    
    return gpu_available, batch_sizes

if __name__ == "__main__":
    main()
