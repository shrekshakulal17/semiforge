// api.js
//
// API layer for the restoration backend.
//
// Right now `restoreImage` returns a mocked result so the frontend can be
// built and demoed before the Python/PyTorch inference service exists.
// Swap the body of `restoreImage` for a real fetch() call when the API is
// ready — the return shape below is the contract the rest of the app
// expects, so no other file needs to change.
//
// Expected real response shape:
// {
//   restoredImage: string        // URL or base64 data URI of the restored image
//   psnr: number | null          // dB
//   ssim: number | null          // 0–1
//   lpips: number | null         // 0–1 (lower is better)
//   inferenceTime: number | null // ms
//   inputResolution: string      // e.g. "256×256"
//   outputResolution: string     // e.g. "512×512"
//   detectedDegradation: string[]
// }

const API_BASE_URL = '/api';

const PIPELINE_STAGES = [
  { id: 'analyze', label: 'Analyzing degradation' },
  { id: 'speckle', label: 'Suppressing speckle noise' },
  { id: 'gaussian', label: 'Reducing Gaussian noise' },
  { id: 'reconstruct', label: 'Reconstructing spatial detail' },
  { id: 'upscale', label: 'Upscaling and refining' },
  { id: 'finalize', label: 'Finalizing restoration' },
];

/**
 * Restore a degraded inspection image.
 *
 * @param {File} file - the uploaded image file
 * @param {(stageIndex: number) => void} [onStageChange] - optional progress callback,
 *   called as the (simulated, for now) pipeline advances through PIPELINE_STAGES
 * @returns {Promise<{
 *   restoredImage: string,
 *   psnr: number|null,
 *   ssim: number|null,
 *   lpips: number|null,
 *   inferenceTime: number|null,
 *   inputResolution: string,
 *   outputResolution: string,
 *   detectedDegradation: string[],
 * }>}
 */
async function restoreImage(file, onStageChange) {
  // --- MOCK IMPLEMENTATION -------------------------------------------------
  // Replace everything in this block with a real request once the inference
  // API is live, e.g.:
  //
  // const formData = new FormData()
  // formData.append('image', file)
  // const res = await fetch(`${API_BASE_URL}/restore`, { method: 'POST', body: formData })
  // if (!res.ok) throw new Error('Restoration failed')
  // return res.json()

  const inputResolution = await readImageResolution(file);
  const objectUrl = URL.createObjectURL(file);

  for (let i = 0; i < PIPELINE_STAGES.length; i++) {
    if (onStageChange) onStageChange(i);
    // eslint-disable-next-line no-await-in-loop
    await wait(650 + Math.random() * 300);
  }

  return {
    restoredImage: objectUrl,
    // Metrics are intentionally left unset — the mock pipeline does not run
    // a real model, so no PSNR/SSIM/LPIPS/timing values are fabricated.
    // The UI renders "--" until a real backend response populates these.
    psnr: null,
    ssim: null,
    lpips: null,
    inferenceTime: null,
    inputResolution,
    outputResolution: null,
    detectedDegradation: ['Speckle noise', 'Gaussian noise', 'Reduced spatial resolution'],
  };
  // --- END MOCK IMPLEMENTATION ---------------------------------------------
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readImageResolution(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      resolve(`${img.naturalWidth} × ${img.naturalHeight}`);
      URL.revokeObjectURL(url);
    };
    img.onerror = reject;
    img.src = url;
  });
}
