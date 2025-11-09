// Common utilities for F1 data visualization pages

// Cache management
const DataCache = {
  // Storage key for cache bust flag
  CACHE_BUST_KEY: 'f1_data_cache_bust',

  // Cache of bust parameter for this page load
  _bustParam: null,
  _bustParamInitialized: false,

  // Check if we should bust cache (only set temporarily after reload button click)
  shouldBustCache() {
    if (!this._bustParamInitialized) {
      this._bustParam = sessionStorage.getItem(this.CACHE_BUST_KEY);
      this._bustParamInitialized = true;

      if (this._bustParam) {
        // Clear the flag after reading it once
        sessionStorage.removeItem(this.CACHE_BUST_KEY);
      }
    }
    return this._bustParam;
  },

  // Get CSV URL - only add cache buster if reload was requested
  getCSVUrl(filename) {
    const bustParam = this.shouldBustCache();
    if (bustParam) {
      const url = `${filename}?v=${bustParam}`;
      return url;
    }
    // No query parameter - let browser use normal HTTP caching
    return filename;
  },

  // Reload data (invalidate cache)
  reloadData() {
    // Set a flag with timestamp that will be used on next page load
    const timestamp = Date.now().toString();
    sessionStorage.setItem(this.CACHE_BUST_KEY, timestamp);
    window.location.reload();
  },

  // Wrap PapaParse to add timing
  loadCSV(url, config) {
    const startTime = performance.now();

    const originalComplete = config.complete;
    config.complete = function(results) {
      const loadTime = (performance.now() - startTime).toFixed(0);
      if (originalComplete) {
        originalComplete(results);
      }
    };

    Papa.parse(url, config);
  }
};

// No initialization needed - we use sessionStorage which is cleared between browser sessions

// Add reload data button to navigation
function addReloadDataButton() {
  const nav = document.querySelector('nav');
  if (!nav) return;

  // Check if button already exists
  if (document.getElementById('reload-data-btn')) return;

  // Create button with dark styling
  const button = document.createElement('button');
  button.id = 'reload-data-btn';
  button.textContent = '↻ Reload Data';
  button.style.cssText = `
    background: #11141a;
    color: #8b8b8b;
    text-decoration: none;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid #2a2f3a;
    border-radius: 8px;
    cursor: pointer;
    margin-left: auto;
    margin-top: 8px;
    margin-bottom: 8px;
    margin-right: 12px;
    transition: all 0.15s ease;
  `;

  button.onmouseover = () => {
    button.style.background = '#1a1d24';
    button.style.color = '#e8e8e8';
    button.style.borderColor = '#00e0ff';
  };
  button.onmouseout = () => {
    button.style.background = '#11141a';
    button.style.color = '#8b8b8b';
    button.style.borderColor = '#2a2f3a';
  };
  button.onclick = () => {
    DataCache.reloadData();
  };

  // Add to nav (append at end)
  nav.appendChild(button);
}

// Initialize on DOM load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', addReloadDataButton);
} else {
  addReloadDataButton();
}

// Common UI utilities
const UIHelpers = {
  // Populate a select element with unique, non-empty values
  // Clears existing options and adds new ones
  populateSelect(selectElement, values, options = {}) {
    const {
      filterEmpty = true,
      defaultValue = null,
      autoSelectFirst = true,
      autoSelectValue = null,
      onChange = null
    } = options;

    // Clear existing options
    selectElement.innerHTML = '';

    // Filter and get unique values
    let filteredValues = values;
    if (filterEmpty) {
      filteredValues = values.filter(v => v && String(v).trim() !== '');
    }
    const uniqueValues = [...new Set(filteredValues)];

    // Add options
    uniqueValues.forEach(value => {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = value;
      selectElement.appendChild(opt);
    });

    // Enable the select if there are options
    selectElement.disabled = uniqueValues.length === 0;

    // Auto-select logic
    if (uniqueValues.length > 0) {
      if (autoSelectValue && uniqueValues.includes(autoSelectValue)) {
        selectElement.value = autoSelectValue;
        if (onChange) onChange();
      } else if (autoSelectFirst) {
        selectElement.value = uniqueValues[0];
        if (onChange) onChange();
      }
    }

    return uniqueValues;
  }
};

// Export for use in other scripts
window.DataCache = DataCache;
window.UIHelpers = UIHelpers;
