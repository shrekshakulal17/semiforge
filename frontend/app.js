// app.js
//
// Vanilla JS port of the React useRestoration hook + App.jsx render tree.
// Drives the EMPTY → UPLOADED → PROCESSING → RESULT / ERROR workflow with
// no framework — plain DOM manipulation, event listeners, and a small
// render() that rebuilds #app-root whenever state changes.

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];

const RESTORATION_PIPELINE_STAGES = [
  'Input',
  'Degradation Analysis',
  'Speckle Denoising',
  'Gaussian Denoising',
  'Feature Reconstruction',
  'Super-Resolution',
  'Restored Image',
];

const ZOOM_STEPS = [1, 1.5, 2, 2.5];

/* ------------------------------------------------------------------ */
/* state                                                               */
/* ------------------------------------------------------------------ */

const state = {
  appState: 'empty', // empty | uploaded | processing | result | error
  file: null,
  previewUrl: null,
  fileMeta: null, // { name, size, type, resolution }
  stageIndex: 0,
  progress: 0,
  result: null,
  errorMessage: null,
  // comparison viewer sub-state (only relevant in 'result')
  splitPct: 50,
  zoomIdx: 0,
};

let progressTimer = null;

const root = document.getElementById('app-root');

/* ------------------------------------------------------------------ */
/* state machine actions                                               */
/* ------------------------------------------------------------------ */

function reset() {
  if (progressTimer) clearInterval(progressTimer);
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.appState = 'empty';
  state.file = null;
  state.previewUrl = null;
  state.fileMeta = null;
  state.stageIndex = 0;
  state.progress = 0;
  state.result = null;
  state.errorMessage = null;
  state.splitPct = 50;
  state.zoomIdx = 0;
  render();
}

function loadFile(selected) {
  if (!selected) return;

  if (!ACCEPTED_TYPES.includes(selected.type)) {
    state.errorMessage = 'Unsupported image format. Please use PNG, JPG or JPEG.';
    state.appState = 'error';
    render();
    return;
  }

  const url = URL.createObjectURL(selected);
  const img = new Image();
  img.onload = () => {
    state.file = selected;
    state.previewUrl = url;
    state.fileMeta = {
      name: selected.name,
      size: selected.size,
      type: selected.type,
      resolution: `${img.naturalWidth} × ${img.naturalHeight}`,
    };
    state.appState = 'uploaded';
    render();
  };
  img.onerror = () => {
    state.errorMessage = 'Unsupported image format. Please use PNG, JPG or JPEG.';
    state.appState = 'error';
    render();
  };
  img.src = url;
}

function removeImage() {
  reset();
}

async function startRestoration() {
  if (!state.file) return;
  state.appState = 'processing';
  state.stageIndex = 0;
  state.progress = 0;
  render();

  // Smoothly animate the progress bar independent of stage timing.
  progressTimer = setInterval(() => {
    if (state.progress < 92) {
      state.progress += 1.5;
      updateProgressUI();
    }
  }, 90);

  try {
    const res = await restoreImage(state.file, (idx) => {
      state.stageIndex = idx;
      renderProcessing();
    });
    clearInterval(progressTimer);
    state.progress = 100;
    state.result = res;
    updateProgressUI();
    setTimeout(() => {
      state.appState = 'result';
      render();
    }, 300);
  } catch (err) {
    clearInterval(progressTimer);
    state.errorMessage =
      err && err.message === 'Failed to fetch'
        ? 'AI restoration service is currently unavailable.'
        : 'Restoration failed. Please try again.';
    state.appState = 'error';
    render();
  }
}

function dismissError() {
  if (state.file) {
    state.appState = 'uploaded';
    state.errorMessage = null;
    render();
  } else {
    reset();
  }
}

/* ------------------------------------------------------------------ */
/* helpers                                                              */
/* ------------------------------------------------------------------ */

function formatSize(bytes) {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function fmtMetric(v, digits = 2) {
  return v === null || v === undefined ? '--' : v.toFixed(digits);
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

/* ------------------------------------------------------------------ */
/* render: RestorationPipeline (shared by 'uploaded' and 'processing') */
/* ------------------------------------------------------------------ */

function renderPipelineStages(activeIndex) {
  return RESTORATION_PIPELINE_STAGES.map((stage, i) => {
    const isDone = activeIndex > i;
    const isActive = activeIndex === i;
    const dotClass = isDone ? 'done' : isActive ? 'active' : '';
    const nameClass = isActive ? 'active' : isDone ? 'done' : '';
    const dotContent = isDone
      ? '<svg data-lucide="check" style="width:14px;height:14px"></svg>'
      : String(i);
    const isLast = i === RESTORATION_PIPELINE_STAGES.length - 1;

    return `
      <div class="pipeline-stage">
        <div class="pipeline-stage-content">
          <div class="stage-dot ${dotClass}">${dotContent}</div>
          <span class="stage-name ${nameClass}">${escapeHtml(stage)}</span>
        </div>
        ${!isLast ? '<div class="stage-connector-h"></div>' : ''}
        ${!isLast ? '<div class="stage-connector-v"></div>' : ''}
      </div>
    `;
  }).join('');
}

function pipelinePanelHtml(activeIndex, subLabel) {
  return `
    <div class="pipeline-panel">
      <div class="pipeline-head">
        <span class="panel-label">Restoration Pipeline</span>
        <span class="sub">${escapeHtml(subLabel)}</span>
      </div>
      <div class="pipeline-stages">
        ${renderPipelineStages(activeIndex)}
      </div>
    </div>
  `;
}

/* ------------------------------------------------------------------ */
/* render: EMPTY — upload zone                                         */
/* ------------------------------------------------------------------ */

function renderEmpty() {
  root.innerHTML = `
    <div class="upload-zone" id="upload-zone" role="button" tabindex="0">
      <div class="bg-fine-grid" style="position:absolute;inset:0;opacity:.3;pointer-events:none;"></div>
      <span class="reticle tl"></span>
      <span class="reticle tr"></span>
      <span class="reticle bl"></span>
      <span class="reticle br"></span>
      <div class="upload-zone-inner">
        <div class="upload-icon-badge"><svg data-lucide="image-plus"></svg></div>
        <h3>Upload a degraded inspection image</h3>
        <p>Drop an image here or browse from your device</p>
        <p class="upload-formats">PNG • JPG • JPEG</p>
        <div class="upload-tags">
          <span class="tag">SPECKLE</span>
          <span class="tag">GAUSSIAN</span>
          <span class="tag">LOW RESOLUTION</span>
        </div>
      </div>
      <input type="file" id="file-input" accept="image/png, image/jpeg" class="hidden" />
    </div>
  `;
  refreshIcons();

  const zone = document.getElementById('upload-zone');
  const input = document.getElementById('file-input');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') input.click();
  });
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragging');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragging');
    const dropped = e.dataTransfer.files && e.dataTransfer.files[0];
    if (dropped) loadFile(dropped);
  });
  input.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) loadFile(f);
  });
}

/* ------------------------------------------------------------------ */
/* render: UPLOADED — image preview + metadata + restore button        */
/* ------------------------------------------------------------------ */

function renderUploaded() {
  const meta = state.fileMeta || {};
  const rows = [
    ['Filename', meta.name],
    ['Resolution', meta.resolution],
    ['File size', formatSize(meta.size)],
    ['Image type', meta.type ? meta.type.replace('image/', '').toUpperCase() : null],
  ];

  root.innerHTML = `
    <div class="uploaded-state animate-fade-up">
      <div class="preview-grid">
        <div class="panel">
          <div class="panel-head">
            <span class="panel-label">Input Image</span>
            <button class="btn-remove" id="btn-remove">
              <svg data-lucide="trash-2"></svg>
              Remove Image
            </button>
          </div>
          <div class="preview-image-wrap">
            <img src="${state.previewUrl}" alt="Uploaded inspection input" />
          </div>
        </div>

        <div class="panel meta-panel">
          <div>
            <span class="panel-label">Metadata</span>
            <dl class="meta-list">
              ${rows
                .map(
                  ([label, value]) => `
                <div class="meta-row">
                  <dt>${escapeHtml(label)}</dt>
                  <dd title="${escapeHtml(value)}">${escapeHtml(value) || '—'}</dd>
                </div>
              `
                )
                .join('')}
            </dl>
          </div>
          <button class="btn btn-primary" id="btn-restore">
            <svg data-lucide="sparkles"></svg>
            Restore Image
          </button>
        </div>
      </div>

      ${pipelinePanelHtml(0, 'visualization of intended pipeline stages')}
    </div>
  `;
  refreshIcons();

  document.getElementById('btn-remove').addEventListener('click', removeImage);
  document.getElementById('btn-restore').addEventListener('click', startRestoration);
}

/* ------------------------------------------------------------------ */
/* render: PROCESSING                                                  */
/* ------------------------------------------------------------------ */

function renderProcessing() {
  const currentLabel =
    (PIPELINE_STAGES[state.stageIndex] && PIPELINE_STAGES[state.stageIndex].label) ||
    'Processing';

  root.innerHTML = `
    <div class="processing-wrap animate-fade-up">
      <div class="processing-card">
        <div class="scan-frame">
          ${state.previewUrl ? `<img src="${state.previewUrl}" alt="Processing" />` : ''}
          <div class="bg-fine-grid"></div>
          <div class="scan-sweep"></div>
        </div>

        <div class="processing-title-row">
          <svg class="spinner" data-lucide="loader-circle"></svg>
          <h3>Restoring inspection image&hellip;</h3>
        </div>

        <p class="processing-stage-label">${escapeHtml(currentLabel)}</p>

        <div class="progress-wrap">
          <div class="progress-track">
            <div class="progress-fill" id="progress-fill" style="width:${state.progress}%"></div>
          </div>
          <div class="progress-meta">
            <span>PROCESSING</span>
            <span id="progress-pct">${Math.round(state.progress)}%</span>
          </div>
        </div>
      </div>

      ${pipelinePanelHtml(state.stageIndex + 1, 'visualization of intended pipeline stages')}
    </div>
  `;
  refreshIcons();
}

function updateProgressUI() {
  const fill = document.getElementById('progress-fill');
  const pct = document.getElementById('progress-pct');
  if (fill) fill.style.width = `${state.progress}%`;
  if (pct) pct.textContent = `${Math.round(state.progress)}%`;
}

/* ------------------------------------------------------------------ */
/* render: RESULT — comparison viewer + metrics + actions               */
/* ------------------------------------------------------------------ */

function renderResult() {
  const result = state.result;

  root.innerHTML = `
    <div class="result-wrap animate-fade-up">
      <div>
        <h2 class="result-title">Restoration Result</h2>
        ${comparisonViewerHtml()}
      </div>

      ${metricsPanelHtml(result)}

      <div class="result-actions">
        <a href="${state.result.restoredImage}" download="restored-${escapeHtml(
    (state.fileMeta && state.fileMeta.name) || 'image.png'
  )}" class="btn btn-primary" style="width:auto;margin-top:0;">
          <svg data-lucide="download"></svg>
          Download Restored Image
        </a>
        <button class="btn btn-outline" id="btn-reset">
          <svg data-lucide="rotate-ccw"></svg>
          Restore Another Image
        </button>
        <a href="#metrics" class="btn btn-ghost">View Metrics</a>
      </div>
    </div>
  `;
  refreshIcons();

  document.getElementById('btn-reset').addEventListener('click', reset);
  wireComparisonViewer();
  wireZoomControls();
}

function comparisonViewerHtml() {
  const zoom = ZOOM_STEPS[state.zoomIdx];
  return `
    <div class="compare-panel">
      <div class="compare-head">
        <div class="compare-labels">
          <span class="before">Degraded Input</span>
          <span class="after">AI Restored</span>
        </div>
        <div class="compare-controls">
          <button class="icon-btn" id="zoom-out" ${state.zoomIdx === 0 ? 'disabled' : ''} aria-label="Zoom out">
            <svg data-lucide="zoom-out"></svg>
          </button>
          <button class="icon-btn" id="zoom-in" ${
            state.zoomIdx === ZOOM_STEPS.length - 1 ? 'disabled' : ''
          } aria-label="Zoom in">
            <svg data-lucide="zoom-in"></svg>
          </button>
          <button class="fit-btn" id="zoom-fit">
            <svg data-lucide="maximize-2"></svg>
            Fit
          </button>
          <span class="zoom-readout">${zoom.toFixed(1)}×</span>
        </div>
      </div>

      <div class="compare-stage" id="compare-stage">
        <div class="compare-layer" id="layer-after" style="transform:scale(${zoom})">
          <img src="${state.result.restoredImage}" alt="AI restored" draggable="false" />
        </div>
        <div class="compare-layer" id="layer-before" style="clip-path: inset(0 ${
          100 - state.splitPct
        }% 0 0); transform:scale(${zoom})">
          <img src="${state.previewUrl}" alt="Degraded input" draggable="false" />
        </div>

        <div class="compare-divider" id="compare-divider" style="left:${state.splitPct}%">
          <div class="compare-handle"><span></span><span></span></div>
        </div>

        <span class="compare-tag left">Degraded</span>
        <span class="compare-tag right">Restored</span>
      </div>

      <p class="compare-note">Preview only — connect the restoration API to populate real before / after output.</p>
    </div>
  `;
}

function wireComparisonViewer() {
  const stageEl = document.getElementById('compare-stage');
  const dividerEl = document.getElementById('compare-divider');
  const beforeLayer = document.getElementById('layer-before');
  let dragging = false;

  function updateFromClientX(clientX) {
    const rect = stageEl.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    state.splitPct = Math.min(98, Math.max(2, pct));
    dividerEl.style.left = `${state.splitPct}%`;
    beforeLayer.style.clipPath = `inset(0 ${100 - state.splitPct}% 0 0)`;
  }

  stageEl.addEventListener('mousedown', (e) => {
    dragging = true;
    updateFromClientX(e.clientX);
  });
  stageEl.addEventListener('mousemove', (e) => {
    if (dragging) updateFromClientX(e.clientX);
  });
  window.addEventListener('mouseup', () => {
    dragging = false;
  });
  stageEl.addEventListener('mouseleave', () => {
    dragging = false;
  });
  stageEl.addEventListener('touchstart', (e) => updateFromClientX(e.touches[0].clientX));
  stageEl.addEventListener('touchmove', (e) => updateFromClientX(e.touches[0].clientX));
}

function wireZoomControls() {
  const zoomOut = document.getElementById('zoom-out');
  const zoomIn = document.getElementById('zoom-in');
  const zoomFit = document.getElementById('zoom-fit');

  zoomOut.addEventListener('click', () => {
    state.zoomIdx = Math.max(0, state.zoomIdx - 1);
    renderResult();
  });
  zoomIn.addEventListener('click', () => {
    state.zoomIdx = Math.min(ZOOM_STEPS.length - 1, state.zoomIdx + 1);
    renderResult();
  });
  zoomFit.addEventListener('click', () => {
    state.zoomIdx = 0;
    renderResult();
  });
}

function metricsPanelHtml(result) {
  const degradation = (result && result.detectedDegradation) || [];
  return `
    <div id="metrics">
      <div class="metrics-head">
        <h3>Restoration Quality</h3>
      </div>

      <div class="metrics-grid">
        ${metricCardHtml('PSNR', fmtMetric(result && result.psnr, 1), 'dB', 'up', 'Pixel-level fidelity')}
        ${metricCardHtml('SSIM', fmtMetric(result && result.ssim, 3), '', 'up', 'Structural similarity')}
        ${metricCardHtml('LPIPS', fmtMetric(result && result.lpips, 3), '', 'down', 'Perceptual distance')}
      </div>

      <div class="summary-grid">
        ${metricCardHtml(
          'Inference Time',
          result && result.inferenceTime !== null && result.inferenceTime !== undefined
            ? result.inferenceTime
            : '--',
          'ms',
          null,
          'End-to-end restoration time'
        )}

        <div class="summary-card">
          <span class="panel-label">Restoration Summary</span>
          <div class="summary-sub-grid">
            <div>
              <p class="summary-field-label">Input Resolution</p>
              <p class="summary-field-value">${escapeHtml((result && result.inputResolution) || '—')}</p>
            </div>
            <div>
              <p class="summary-field-label">Output Resolution</p>
              <p class="summary-field-value">${escapeHtml((result && result.outputResolution) || '—')}</p>
            </div>
          </div>
          <div style="margin-top:16px;">
            <p class="summary-field-label">Detected / Simulated Degradation</p>
            <ul class="degradation-list">
              ${degradation.map((d) => `<li>${escapeHtml(d)}</li>`).join('')}
            </ul>
          </div>
        </div>
      </div>
    </div>
  `;
}

function metricCardHtml(label, value, unit, direction, description) {
  return `
    <div class="metric-card">
      <div class="metric-top">
        <span class="metric-label">${escapeHtml(label)}</span>
        ${
          direction
            ? `<span class="metric-direction">${direction === 'up' ? 'Higher is better' : 'Lower is better'}</span>`
            : ''
        }
      </div>
      <div class="metric-value-row">
        <span class="metric-value">${escapeHtml(value)}</span>
        ${unit ? `<span class="metric-unit">${escapeHtml(unit)}</span>` : ''}
      </div>
      <p class="metric-desc">${escapeHtml(description)}</p>
    </div>
  `;
}

/* ------------------------------------------------------------------ */
/* render: ERROR                                                       */
/* ------------------------------------------------------------------ */

function renderError() {
  root.innerHTML = `
    <div class="error-card animate-fade-up">
      <div class="error-icon-badge"><svg data-lucide="triangle-alert"></svg></div>
      <p class="error-message">${escapeHtml(state.errorMessage || 'Something went wrong.')}</p>
      <div class="error-actions">
        <button class="btn btn-outline" id="btn-try-again">Try Again</button>
        <button class="btn btn-ghost" id="btn-start-over">Start Over</button>
      </div>
    </div>
  `;
  refreshIcons();

  document.getElementById('btn-try-again').addEventListener('click', dismissError);
  document.getElementById('btn-start-over').addEventListener('click', reset);
}

/* ------------------------------------------------------------------ */
/* master render                                                       */
/* ------------------------------------------------------------------ */

function render() {
  switch (state.appState) {
    case 'empty':
      renderEmpty();
      break;
    case 'uploaded':
      renderUploaded();
      break;
    case 'processing':
      renderProcessing();
      break;
    case 'result':
      renderResult();
      break;
    case 'error':
      renderError();
      break;
    default:
      renderEmpty();
  }
}

/* ------------------------------------------------------------------ */
/* boot                                                                 */
/* ------------------------------------------------------------------ */

document.addEventListener('DOMContentLoaded', () => {
  refreshIcons(); // static header / section icons
  render();
});
