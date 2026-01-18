import torch
import ray

def check_env():
    print("="*30)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")
        
        # 检查 BF16 支持 (H100 核心特性)
        if torch.cuda.is_bf16_supported():
            print("✅ BF16 Acceleration: Supported (Great for H100!)")
        else:
            print("❌ BF16 Acceleration: Not Supported (Check CUDA version)")
            
        # 简单的 Tensor 测试
        x = torch.tensor([1.0]).cuda()
        print("✅ Tensor operations working on GPU")
    else:
        print("❌ GPU NOT DETECTED!")

    print("-" * 30)
    # 检查 Ray
    try:
        ray.init(ignore_reinit_error=True)
        print("✅ Ray initialized successfully")
        print(f"Ray Resources: {ray.available_resources()}")
        ray.shutdown()
    except Exception as e:
        print(f"❌ Ray failed to initialize: {e}")
    print("="*30)

if __name__ == "__main__":
    check_env()