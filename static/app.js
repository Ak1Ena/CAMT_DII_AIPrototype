// ======================================================================
// The Classifier — front-end logic
// Handles file selection, drag/drop, preview, prediction, result rendering.
// ======================================================================

(function () {
  'use strict';

  const dropzone     = document.getElementById('dropzone');
  const fileInput    = document.getElementById('fileInput');
  const dzEmpty      = document.getElementById('dzEmpty');
  const dzPreview    = document.getElementById('dzPreview');
  const previewImg   = document.getElementById('previewImg');
  const clearBtn     = document.getElementById('clearBtn');
  const predictBtn   = document.getElementById('predictBtn');

  const stateIdle    = document.getElementById('stateIdle');
  const stateLoading = document.getElementById('stateLoading');
  const stateDone    = document.getElementById('stateDone');
  const stateError   = document.getElementById('stateError');

  const verdictClass      = document.getElementById('verdictClass');
  const verdictConfidence = document.getElementById('verdictConfidence');
  const ranking           = document.getElementById('ranking');
  const errorMsg          = document.getElementById('errorMsg');

  // If the model failed to load, the dropzone won't be on the page.
  if (!dropzone) return;

  let currentFile = null;

  // -----------------------------------------------------------------
  // File selection
  // -----------------------------------------------------------------
  function setFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      showError('Please select an image file.');
      return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dzEmpty.hidden = true;
      dzPreview.hidden = false;
    };
    reader.readAsDataURL(file);

    predictBtn.disabled = false;
    showState('idle');
  }

  function clearFile() {
    currentFile = null;
    fileInput.value = '';
    previewImg.src = '';
    dzEmpty.hidden = false;
    dzPreview.hidden = true;
    predictBtn.disabled = true;
    showState('idle');
  }

  // Click to pick
  dropzone.addEventListener('click', (e) => {
    if (e.target.closest('.dz-clear')) return;
    fileInput.click();
  });
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) setFile(f);
  });

  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
  });

  // -----------------------------------------------------------------
  // Drag & drop
  // -----------------------------------------------------------------
  ['dragenter', 'dragover'].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragging');
    });
  });
  ['dragleave', 'drop'].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragging');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  // Don't let dragging files anywhere else navigate away
  ['dragover', 'drop'].forEach((evt) => {
    window.addEventListener(evt, (e) => {
      if (e.target === dropzone || dropzone.contains(e.target)) return;
      e.preventDefault();
    });
  });

  // -----------------------------------------------------------------
  // Predict
  // -----------------------------------------------------------------
  predictBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    showState('loading');

    const fd = new FormData();
    fd.append('image', currentFile);

    try {
      const res = await fetch('/api/predict', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) {
        showError(data.error || `Server returned ${res.status}.`);
        return;
      }
      renderResult(data);
    } catch (err) {
      showError(err.message || 'Network error. Is the server running?');
    }
  });

  // -----------------------------------------------------------------
  // State management
  // -----------------------------------------------------------------
  function showState(name) {
    stateIdle.hidden    = name !== 'idle';
    stateLoading.hidden = name !== 'loading';
    stateDone.hidden    = name !== 'done';
    stateError.hidden   = name !== 'error';
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    showState('error');
  }

  function renderResult(data) {
    verdictClass.textContent = data.predicted_class;
    verdictConfidence.textContent =
      (data.confidence * 100).toFixed(1) + '%';

    // Build the ranking bars
    ranking.innerHTML = '';
    const top = data.top_predictions || [];
    top.forEach((p, i) => {
      const row = document.createElement('div');
      row.className = 'ranking-row' + (i === 0 ? ' top' : '');
      row.style.animationDelay = (0.05 + i * 0.06) + 's';

      const name = document.createElement('div');
      name.className = 'rk-name';
      name.textContent = p.class;

      const bar = document.createElement('div');
      bar.className = 'rk-bar';
      const fill = document.createElement('div');
      fill.className = 'rk-bar-fill';
      fill.style.right = '100%';
      bar.appendChild(fill);

      const pct = document.createElement('div');
      pct.className = 'rk-pct';
      pct.textContent = (p.probability * 100).toFixed(1) + '%';

      row.appendChild(name);
      row.appendChild(bar);
      row.appendChild(pct);
      ranking.appendChild(row);

      // Animate fill on next frame
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          fill.style.right = (100 - p.probability * 100).toFixed(2) + '%';
        });
      });
    });

    showState('done');
  }
})();
