/**
 * AdOptimizer — Main JavaScript
 * Minimal JS for animations and Chart.js configuration only
 */

// ─── Dashboard Upload Toggle ────────────────────────────────
const toggleUploadBtn = document.getElementById('toggleUploadBtn');
const uploadSection = document.getElementById('uploadSection');

if (toggleUploadBtn && uploadSection) {
  toggleUploadBtn.addEventListener('click', () => {
    const isHidden = uploadSection.style.display === 'none';
    uploadSection.style.display = isHidden ? 'block' : 'none';
    toggleUploadBtn.innerHTML = isHidden ? 
      '<i class="fa-solid fa-minus"></i> Close' : 
      '<i class="fa-solid fa-plus"></i> Add Dataset';
  });
}

// ─── File Upload Handling ───────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');
const fileNameEl = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const clearFile = document.getElementById('clearFile');

if (dropZone && fileInput) {
  // Clicking drop zone triggers file input
  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  // File picker change
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Only CSV files are allowed.');
      fileInput.value = '';
      return;
    }
    fileNameEl.textContent = file.name;
    filePreview.style.display = 'flex';
    dropZone.style.display = 'none';
    uploadBtn.disabled = false;
  }

  if (clearFile) {
    clearFile.addEventListener('click', () => {
      fileInput.value = '';
      filePreview.style.display = 'none';
      dropZone.style.display = 'flex';
      uploadBtn.disabled = true;
    });
  }
}

// ─── Auto-dismiss flash messages ─────────────────────────────
document.querySelectorAll('.flash-alert').forEach(el => {
  setTimeout(() => {
    const bsAlert = bootstrap.Alert.getOrCreateInstance(el);
    if (bsAlert) bsAlert.close();
  }, 5000);
});

// ─── Animate KPI cards on load ────────────────────────────────
document.querySelectorAll('.kpi-card, .rec-card, .result-kpi').forEach((card, i) => {
  card.style.opacity = '0';
  card.style.transform = 'translateY(16px)';
  setTimeout(() => {
    card.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    card.style.opacity = '1';
    card.style.transform = 'translateY(0)';
  }, 80 * i);
});

// ─── Performance score bar animation ─────────────────────────
window.addEventListener('load', () => {
  const fill = document.querySelector('.perf-score-fill');
  if (fill) {
    const target = fill.style.width;
    fill.style.width = '0';
    setTimeout(() => { fill.style.width = target; }, 300);
  }
});

// ─── Chart.js global defaults ────────────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.font.family = 'DM Sans';
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.plugins.legend.labels.borderRadius = 4;
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(17,24,39,0.95)';
  Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.07)';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = '#f1f5f9';
  Chart.defaults.plugins.tooltip.bodyColor = '#94a3b8';
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
}
