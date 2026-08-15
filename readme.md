# AI-Based Image Restoration for Semiconductor Inspection (Track 1 - KLA)
## SEMICON India Hackathon 2026 | Team [Your Team Name]

This repository contains our team's submission for the **SEMICON India Hackathon 2026 (Track 1 - Sponsored by KLA)**. Our solution focuses on building a robust, high-performance, and generalizable AI pipeline to restore signal-degraded semiconductor microscopic inspection images by simultaneously addressing:
1. **Speckle Noise:** Multiplicative, pixel-level noise pushing values outside the normal [0, 1] range.
2. **Gaussian Noise:** Additive noise causing soft, hazy edges and structural detail loss.
3. **Spatial Resolution Reduction:** Up-sampling and reconstructing high-frequency details from low-resolution down-sampled images (e.g., 256×256 → 512×512 or 128×128 → 256×256).

---

## 👥 Team Details
*   **Team Name:** [Your Team Name]
*   **Institution:** [Your College/University Name]
*   **Members & Roles:**
    *   **Member 1 (Lead):** [Name] - [Role: e.g., Model Architecture & Training] - [Email/Contact]
    *   **Member 2:** [Name] - [Role: e.g., Data Augmentation & Loss Functions] - [Email/Contact]
    *   **Member 3:** [Name] - [Role: e.g., Inference Pipeline & H100 GPU Optimization] - [Email/Contact]
    *   **Member 4:** [Name] - [Role: e.g., Validation & Benchmark Analysis] - [Email/Contact]

---

## 📂 Repository Structure

```directory
├── evaluation_script.py      # Standalone, automated Python script for KLA benchmarking (MANDATORY)
├── training_script.py        # Python script or Jupyter Notebook reproducing training from scratch
├── requirements.txt          # Standard pip freeze environment specification (MANDATORY)
├── weights/                  # Directory containing final model weights (or download links)
│   └── final_model.pt        # Final trained model checkpoint (e.g., .pt, .onnx, .h5)
├── restored_test_outputs/    # Folder containing restored output images on the test split
├── README.md                 # Project documentation and setup guide (This file)
└── assets/                   # Before/after comparison images and architecture diagrams
```

---

## 🛠️ Environment Setup & Installation

To ensure perfect reproducibility, we have pinned all dependencies in the `requirements.txt` file (generated via `pip freeze` from our training environment). 

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[your-username]/[your-repo-name].git
   cd [your-repo-name]
   ```

2. **Set up a clean virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run Inference (Automated Benchmarking)

Our standalone **`evaluation_script.py`** is designed to run end-to-end without any manual code modifications. It accepts two mandatory command-line arguments to load input degraded images, execute our optimized restoration model, and write the denoised outputs back to disk.

### **Inference Command**
```bash
python evaluation_script.py --input_dir /path/to/test_images --output_dir /path/to/save_outputs
```

### **What the Evaluation Script Does:**
1. **Initializes & Loads Model:** Instantiates the network architecture and loads the weights from `weights/final_model.pt` directly onto the target device (CPU/CUDA).
2. **Efficient Batch Loading:** Utilizes an optimized PyTorch `DataLoader` with custom I/O handling to stream grayscale images from disk without causing memory bottlenecks.
3. **Dynamic Intensity Handling:** Safely processes pixel intensities that exceed the typical $[0, 1]$ range (a critical physical feature of KLA's speckle noise) rather than blindly clipping them during preprocessing.
4. **End-to-End Restoration:** Restores spatial resolution (from 256×256 or 128×128 back to 512×512 or 256×256) while removing speckle and Gaussian noise.
5. **High-Speed Output I/O:** Saves the recovered grayscale output images back to the specified `--output_dir` in standard format.

---

## 🏋️ How to Reproduce Training

Our training process can be fully reproduced using the `training_script.py` (or notebook). 

To execute the training pipeline:
```bash
python training_script.py --data_dir /path/to/kla_paired_training_data
```

### **Training Hygiene & Best Practices:**
*   **Validation Split:** We use a strict out-of-distribution (OOD) validation split to prevent overfitting and guarantee generalization across different semiconductor structures.
*   **Data Augmentation:** Real-world semiconductor structures have diverse textures. We applied robust data augmentation (random rotations, synthetic noise injection, and multi-scale resizing) to help our model adapt to unseen test distributions.
*   **Custom Loss Design:** Our model is trained on a hybrid loss function combining **L1/L2 pixel-level loss**, structural loss (**SSIM**), and deep perceptual loss (**LPIPS**). This combination prevents over-smoothing and preserves fine sub-micron features.

---

## 🧠 Model Architecture & Methodology

*Provide a brief summary of your chosen model and strategy.*

*   **Network Backbone:** Describe your choice (e.g., Modified NAFNet, SwinIR, Restormer, or custom U-Net/GAN variant). Why is this suitable for high-resolution grayscale image reconstruction?
*   **Optimization Strategies:** Highlight how you minimized latency for the **NVIDIA H100 GPU** benchmarking platform (e.g., FP16/mixed precision, model quantization, optimized CUDA kernel operations, or PyTorch Compilation `torch.compile()`).
*   **I/O Optimizations:** Explain how you reduced disk bottlenecking (e.g., using multi-threaded PyTorch workers, optimized image reading libraries like OpenCV or Pillow-SIMD).

---

## 📊 Results & Performance Summary

The model has been self-evaluated on our internal validation split across the following metrics:

| Metric | Target Value (Baseline) | Our Solution Score |
| :--- | :---: | :---: |
| **SSIM** | *Higher is better* | `0.XX` |
| **pSNR** | *Higher is better* | `XX.XX dB` |
| **LPIPS** | *Lower is better* | `0.XX` |
| **Inference Speed** | *Average time per image on H100* | `X.XX ms` |

### **Visual Before/After Showcase**
*(Add representative cropped sub-sections of your outputs here to prove the model's structural preservation!)*

*   **Degraded Input:** Grainy, blurry, lost sub-micron boundaries.
*   **Our Restored Output:** Crisp, noise-free, sharp edges, preserved spatial intensity histogram.
*   **Ground Truth:** For comparison.

---

## 📚 References
*   [List of papers, textbooks, or frameworks you leveraged, e.g., NAFNet, Restormer, LPIPS papers, PyTorch documentation.]
