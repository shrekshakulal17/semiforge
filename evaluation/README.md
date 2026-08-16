# Model Evaluation

## Model

RestorationModel

## Input

256 × 256 RGB image

## Output

512 × 512 RGB image

## Dataset

DIV2K HR dataset

Total images: 800

Training images: 720

Validation images: 80

## Metrics

### PSNR

To be measured using evaluation.py.

### SSIM

To be measured using evaluation.py.

### LPIPS

To be measured separately.

### Inference time

To be measured using benchmark.py.

## Visual Evaluation

Visual comparisons are generated using visual_test.py.

Each test contains:

- degraded input
- ground truth
- restored output

## Final Results

PSNR: TBD

SSIM: TBD

LPIPS: TBD

Average inference time: TBD

Minimum inference time: TBD

Maximum inference time: TBD