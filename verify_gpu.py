# verify_gpu.py
import tensorflow as tf

print("--- GPU Verification Script ---")
print("TensorFlow Version:", tf.__version__)

# list_physical_devices will find all available GPUs TensorFlow can see.
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\n✅ Success! Found {len(gpus)} GPU(s):")
    try:
        # Print details for each GPU found
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  - {gpu.name}, Type: {gpu.device_type}")
    except RuntimeError as e:
        print(f"RuntimeError during GPU setup: {e}")
else:
    print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! WARNING: TensorFlow did NOT find any GPUs.              !!!")
    print("!!! The classification scripts will fall back to using the CPU. !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("\nCommon reasons for this include:")
    print("  - A mismatch between NVIDIA Driver, CUDA, and cuDNN versions.")
    print("  - The CUDA/cuDNN paths not being in the system's PATH environment variable.")
    print("  - Using a TensorFlow version that is not compatible with your CUDA version.")
    print("\nPlease consult the official TensorFlow GPU installation guide to troubleshoot.")

print("\n--- Verification Complete ---")