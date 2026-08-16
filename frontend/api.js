// api.js
//
// Real API connection to the Python/PyTorch restoration backend.

const API_BASE_URL = 'http://127.0.0.1:5000';

const PIPELINE_STAGES = [
  { id: 'analyze', label: 'Analyzing degradation' },
  { id: 'speckle', label: 'Suppressing speckle noise' },
  { id: 'gaussian', label: 'Reducing Gaussian noise' },
  { id: 'reconstruct', label: 'Reconstructing spatial detail' },
  { id: 'upscale', label: 'Upscaling and refining' },
  { id: 'finalize', label: 'Finalizing restoration' },
];

/**
 * Send an image to the real PyTorch backend.
 */
async function restoreImage(file, onStageChange) {

  // Show the first stage immediately
  if (onStageChange) {
    onStageChange(0);
  }

  // Prepare image for upload
  const formData = new FormData();
  formData.append('image', file);

  // Move through UI stages while backend is processing
  let currentStage = 0;

  const stageTimer = setInterval(() => {

    if (currentStage < PIPELINE_STAGES.length - 1) {
      currentStage += 1;

      if (onStageChange) {
        onStageChange(currentStage);
      }
    }

  }, 700);


  try {

    // Start timing
    const startTime = performance.now();

    // Send image to Flask
    const response = await fetch(
      `${API_BASE_URL}/restore`,
      {
        method: 'POST',
        body: formData,
      }
    );

    // Stop stage animation
    clearInterval(stageTimer);

    if (!response.ok) {

      let message = 'Restoration failed.';

      try {
        const errorData = await response.json();

        if (errorData.error) {
          message = errorData.error;
        }

      } catch (_) {
        // Ignore JSON parsing failure
      }

      throw new Error(message);
    }

    // Backend returns PNG image
    const blob = await response.blob();

    const restoredImage = URL.createObjectURL(blob);

    const inferenceTime = performance.now() - startTime;


    // Tell UI that final stage is complete
    if (onStageChange) {
      onStageChange(PIPELINE_STAGES.length - 1);
    }


    // Read original input resolution
    const inputResolution = await readImageResolution(file);


    // Our current model always outputs 512x512
    const outputResolution = '512 × 512';


    return {
      restoredImage,

      // Real metrics require ground-truth comparison.
      // We will populate these later in the evaluation pipeline.
      psnr: null,
      ssim: null,
      lpips: null,

      inferenceTime: Number(inferenceTime.toFixed(2)),

      inputResolution,
      outputResolution,

      detectedDegradation: [
        'Reduced spatial resolution',
        'Speckle noise',
        'Gaussian noise',
      ],
    };

  } catch (error) {

    clearInterval(stageTimer);

    console.error('Restoration API error:', error);

    throw error;
  }
}


/**
 * Read original image resolution.
 */
function readImageResolution(file) {

  return new Promise((resolve, reject) => {

    const img = new Image();

    const url = URL.createObjectURL(file);

    img.onload = () => {

      resolve(
        `${img.naturalWidth} × ${img.naturalHeight}`
      );

      URL.revokeObjectURL(url);
    };

    img.onerror = () => {

      URL.revokeObjectURL(url);

      reject(
        new Error('Could not read image resolution.')
      );
    };

    img.src = url;
  });
}