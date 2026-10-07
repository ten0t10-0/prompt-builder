import { app } from "../../scripts/app.js";

const id = 'b-prompt-builder-panel';
const cssId = 'bpb-css';
const cssHref = '/extensions/prompt-builder/prompt_builder.css';

function ensureCss() {
  if (document.getElementById(cssId)) return;
  const link = document.createElement('link');
  link.id = cssId;
  link.rel = 'stylesheet';
  link.type = 'text/css';
  link.href = cssHref;
  document.head.appendChild(link);
}

let state = {
  values: {},
  activated: {},
  expanded: {},
  active_tabs: {}  // {tabId: activeIndex}
};

async function loadLayout() {
  try {
    const resp = await fetch('/extensions/prompt-builder/layout.json', { cache: 'no-cache' });
    if (resp.ok) return await resp.json();
  } catch (e) {}
  return null;
}

function escapeHtml(s) {
  if (s == null) return '';
  const div = document.createElement('div');
  div.textContent = String(s);
  return div.innerHTML;
}

let evalTimer = null;
function scheduleEval() {
  if (evalTimer) clearTimeout(evalTimer);
  evalTimer = setTimeout(evalState, 150);
}

async function evalState() {
  try {
    // Push state to backend for node to pick up
    await fetch('/extensions/prompt-builder/state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: state })
    });

    const resp = await fetch('/extensions/prompt-builder/eval', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: JSON.stringify(state) })
    });
    if (resp.ok) {
      const j = await resp.json();
      const p = document.getElementById('bpb-pos');
      const n = document.getElementById('bpb-neg');
      if (p) p.textContent = j.positive || '';
      if (n) n.textContent = j.negative || '';
    }
  } catch (e) {}
}

function createPanel() {
  const panel = document.createElement('div');
  panel.id = id;
  panel.className = 'bpb-panel';
  
  // Header with title and controls
  const header = document.createElement('div');
  header.className = 'bpb-header';
  header.innerHTML = 
    '<span class="bpb-title">B Prompt Builder</span>' +
    '<div class="bpb-header-actions">' +
    '<button id="bpb-clear-all" title="Clear all selections">Clear All</button>' +
    '<button id="bpb-reset" title="Reset to initial layout state">Reset</button>' +
    '</div>';
  
  // Scrollable content area
  const contentWrapper = document.createElement('div');
  contentWrapper.className = 'bpb-content-wrap';
  contentWrapper.innerHTML =
    '<div id="bpb-tabs"></div>' +
    '<div id="bpb-content"></div>';
  
  // Sticky footer with prompt previews
  const footer = document.createElement('div');
  footer.className = 'bpb-footer';
  footer.innerHTML =
    '<div class="bpb-preview-label">Positive:</div>' +
    '<div id="bpb-pos" class="bpb-preview"></div>' +
    '<div class="bpb-preview-label">Negative:</div>' +
    '<div id="bpb-neg" class="bpb-preview"></div>';
  
  panel.appendChild(header);
  panel.appendChild(contentWrapper);
  panel.appendChild(footer);
  
  // Attach button handlers
  header.querySelector('#bpb-reset').onclick = () => {
    resetToInitial();
  };
  header.querySelector('#bpb-clear-all').onclick = () => {
    clearAllSelections();
  };
  
  return panel;
}

// Store initial layout state for reset
let initialLayoutState = null;

function resetToInitial() {
  if (initialLayoutState) {
    state.values = { ...initialLayoutState.values };
    state.activated = { ...initialLayoutState.activated };
    state.expanded = { ...initialLayoutState.expanded };
    state.active_tabs = { ...initialLayoutState.active_tabs };
    // Re-render everything
    loadLayout().then(layout => {
      if (layout && layout.tabs) {
        window.__prompt_builder_lists = layout.lists || {};
        renderTabs(layout.tabs);
      }
      evalState();
    });
  }
}

function clearAllSelections() {
  state.activated = {};
  loadLayout().then(layout => {
    if (layout && layout.tabs) {
      renderTabs(layout.tabs);
    }
    evalState();
  });
}

function getTabId(tab, parentId) {
  return parentId ? parentId + '::' + (tab.name || 'tab') : (tab.name || 'root');
}

function renderTabs(tabs, parentId = null, isSubTabs = false) {
  const tabId = parentId || 'root';
  // For sub-tabs, the tab bar goes in bpb-tabs-<id>, content in bpb-content-<id>
  // For root tabs, tab bar goes in bpb-tabs, content in bpb-content
  const isRoot = parentId === null;
  const tabBarContainerId = isRoot ? 'bpb-tabs' : 'bpb-tabs-' + parentId.replace(/::/g, '-');
  const contentContainerId = isRoot ? 'bpb-content' : 'bpb-content-' + parentId.replace(/::/g, '-');
  const el = document.getElementById(tabBarContainerId);
  if (!el) return;
  if (!tabs || !tabs.length) {
    el.innerHTML = '';
    return;
  }
  const activeIdx = state.active_tabs[tabId] || 0;
  let html = '<div class="bpb-tabbar">';
  for (let i = 0; i < tabs.length; i++) {
    const active = activeIdx === i ? ' bpb-active' : '';
    const label = escapeHtml(tabs[i].name || 'Tab ' + (i+1));
    html += '<button class="bpb-tab-btn' + active + '" data-tabid="' + escapeHtml(tabId) + '" data-i="' + i + '">' + label + '</button>';
  }
  html += '</div>';
  el.innerHTML = html;
  el.querySelectorAll('button').forEach(b => {
    b.onclick = () => {
      const tabId = b.getAttribute('data-tabid');
      const idx = parseInt(b.getAttribute('data-i'));
      state.active_tabs[tabId] = idx;
      renderTabs(tabs, tabId === 'root' ? null : tabId);
      renderNestedContent(tabs, tabId);
      scheduleEval();
    };
  });
  // Render initial content
  renderNestedContent(tabs, tabId);
}

function renderNestedContent(tabs, tabId) {
  const activeIdx = state.active_tabs[tabId] || 0;
  const tab = tabs[activeIdx];
  if (!tab) return;
  
  const containerId = tabId === 'root' ? 'bpb-content' : 'bpb-content-' + tabId.replace(/::/g, '-');
  const cont = document.getElementById(containerId);
  if (!cont) return;
  
  if (!tab.children) {
    cont.innerHTML = '';
    return;
  }
  
  // Check if this container has tab children - if so, render a tab bar for them
  const tabChildren = tab.children.filter(c => (c.type || '').toLowerCase() === 'tab');
  const nonTabChildren = tab.children.filter(c => (c.type || '').toLowerCase() !== 'tab');
  
  let html = '';
  
  // Render non-tab children first
  for (let c of nonTabChildren) {
    html += renderElement(c, tabs, tabId, tab.children);
  }
  
  // If there are tab children, render a tab bar for them
  if (tabChildren.length > 0) {
    const subTabId = tabId + '::subtabs';
    const safeId = subTabId.replace(/::/g, '-');
    html += '<div id="bpb-tabs-' + escapeHtml(safeId) + '" class="bpb-subtabs"></div>';
    html += '<div id="bpb-content-' + escapeHtml(safeId) + '" class="bpb-subcontent"></div>';
    // Schedule tab bar rendering
    setTimeout(() => renderTabs(tabChildren, subTabId), 0);
  }
  
  cont.innerHTML = html;
  
  // Attach handlers for this container
  cont.querySelectorAll('[data-action]').forEach(el => {
    el.onclick = (ev) => {
      ev.stopPropagation(); // Prevent bubbling to expand/collapse trigger
      const name = el.getAttribute('data-name');
      const act = el.getAttribute('data-action');
      if (act === 'toggle-activate') {
        state.activated[name] = !state.activated[name];
        scheduleEval();
        renderNestedContent(tabs, tabId);
      } else if (act === 'toggle-expand') {
        state.expanded[name] = !state.expanded[name];
        renderNestedContent(tabs, tabId);
      } else if (act === 'clear-select') {
        // Find the select element recursively in the current tab's children
        const activeIdx = state.active_tabs[tabId] || 0;
        const currentTab = tabs[activeIdx];
        
        function findSelect(elem, targetName) {
          if ((elem.name || elem.i) === targetName && (elem.type || '').toLowerCase() === 'select') {
            return elem;
          }
          if (elem.children) {
            for (const child of elem.children) {
              const found = findSelect(child, targetName);
              if (found) return found;
            }
          }
          return null;
        }
        
        if (currentTab && currentTab.children) {
          let selectElem = null;
          for (const child of currentTab.children) {
            selectElem = findSelect(child, name);
            if (selectElem) break;
          }
          
          if (selectElem && selectElem.children) {
            // Clear direct children
            for (const child of selectElem.children) {
              const childName = child.name || child.i || child.label;
              if (childName) {
                state.activated[childName] = false;
              }
              // Also clear from_list items using the correct naming pattern with postfix
              if ((child.type || '').toLowerCase() === 'from_list') {
                const listName = child.name || child.i;
                const postfix = child.postfix || '';
                const safePostfix = postfix ? '_' + postfix.replace(/\s+/g, '_') : '';
                const lists = window.__prompt_builder_lists || {};
                for (const [k, v] of Object.entries(lists)) {
                  if (k.toLowerCase() === listName.toLowerCase()) {
                    for (const item of v) {
                      const itemName = name + '_' + listName + safePostfix + '_' + item;
                      state.activated[itemName] = false;
                    }
                    break;
                  }
                }
              }
            }
          }
        }
        scheduleEval();
        renderNestedContent(tabs, tabId);
      }
    };
  });
  cont.querySelectorAll('[data-input]').forEach(el => {
    el.oninput = (ev) => {
      const name = el.getAttribute('data-name');
      const t = el.getAttribute('data-input');
      if (t === 'text') {
        state.values[name] = el.value;
        scheduleEval();
      } else if (t === 'range') {
        state.values[name] = parseFloat(el.value);
        const disp = document.getElementById('v-' + name);
        if (disp) disp.textContent = String(state.values[name]);
        scheduleEval();
      } else if (t === 'checkbox') {
        state.values[name] = el.checked;
        scheduleEval();
      }
    };
  });
}

function getListItems(listName) {
  const lists = window.__prompt_builder_lists || {};
  for (const [k, v] of Object.entries(lists)) {
    if (String(k).toLowerCase() === String(listName).toLowerCase()) return v;
  }
  return [];
}

// Expand a from_list element into single-like options.
// Always sorted alphabetically by display label so list options
// interleave deterministically with normal singles.
function expandFromListOptions(c, parentSelectName, parentTabId) {
  const listName = c.name || c.i;
  const postfix = c.postfix || '';
  const selectName = parentSelectName || (parentTabId || '').split('::').pop() || 'select';
  const items = getListItems(listName);
  const safePostfix = postfix ? '_' + String(postfix).replace(/\s+/g, '_') : '';
  const opts = (items || []).map(item => ({
    type: 'single',
    name: selectName + '_' + listName + safePostfix + '_' + item,
    display: String(item) + (postfix ? ' ' + postfix : ''),
    prompt: String(item).toLowerCase(),
    _fromList: true
  }));
  opts.sort((a, b) => String(a.display).localeCompare(String(b.display)));
  return opts;
}

function renderSingleRow(name, displayLabel, opts) {
  opts = opts || {};
  const act = !!state.activated[name];
  const emphasis = parseFloat(state.values[name + '_emphasis'] ?? opts.emphasis ?? 1);
  const isNegative = state.values[name + '_is_negative'] ?? opts.is_negative ?? false;
  const promptValue = escapeHtml(state.values[name + '_prompt'] ?? opts.prompt ?? '');
  const checkboxId = 'single-act-' + escapeHtml(name);
  return '<div class="bpb-single' + (act ? ' bpb-on' : '') + '">' +
    '<label class="bpb-single-label">' +
      '<input type="checkbox" id="' + checkboxId + '" ' + (act?'checked':'') + ' data-action="toggle-activate" data-name="' + escapeHtml(name) + '"/>' +
      '<span class="bpb-single-name">' + escapeHtml(displayLabel) + '</span>' +
    '</label>' +
    (act ? '<div class="bpb-single-controls">' +
      '<input type="text" value="' + promptValue + '" data-input="text" data-name="' + escapeHtml(name + '_prompt') + '" title="Prompt text"/>' +
      '<input type="number" min="0" step="0.1" value="' + emphasis + '" data-input="range" data-name="' + escapeHtml(name + '_emphasis') + '" title="Emphasis (0 to omit)"/>' +
      '<label class="bpb-neg-toggle" title="Toggle negative prompt">' +
        '<input type="checkbox" ' + (isNegative?'checked':'') + ' data-input="checkbox" data-name="' + escapeHtml(name + '_is_negative') + '"/>' +
        '<span>N</span>' +
      '</label>' +
    '</div>' : '') +
  '</div>';
}

function renderElement(c, tabs, parentTabId, siblingTabs, currentSelectName) {
  const type = (c.type || '').toLowerCase();
  if (type === 'tab') {
    // Tab elements are now handled by parent in renderNestedContent
    // Just render the tab's non-tab children directly (the parent will handle tab bar)
    const nonTabChildren = (c.children || []).filter(ch => (ch.type || '').toLowerCase() !== 'tab');
    let html = '';
    for (let ch of nonTabChildren) {
      html += renderElement(ch, tabs, parentTabId, c.children, currentSelectName);
    }
    // Nested tabs inside this tab will be handled recursively by renderNestedContent
    const tabChildren = (c.children || []).filter(ch => (ch.type || '').toLowerCase() === 'tab');
    if (tabChildren.length > 0) {
      const subTabId = (parentTabId || 'root') + '::' + (c.name || 'tab') + '::subtabs';
      const safeId = subTabId.replace(/::/g, '-');
      html += '<div id="bpb-tabs-' + escapeHtml(safeId) + '" class="bpb-subtabs"></div>';
      html += '<div id="bpb-content-' + escapeHtml(safeId) + '" class="bpb-subcontent"></div>';
      setTimeout(() => renderTabs(tabChildren, subTabId), 0);
    }
    return html;
  }
  if (type === 'row') {
    let html = '<div class="bpb-row">';
    for (let ch of (c.children||[])) html += '<div>' + renderElement(ch,tabs,parentTabId, c.children, currentSelectName) + '</div>';
    html += '</div>';
    return html;
  }
  if (type === 'column') {
    let html = '<div class="bpb-col">';
    for (let ch of (c.children||[])) html += renderElement(ch,tabs,parentTabId, c.children, currentSelectName);
    html += '</div>';
    return html;
  }
  if (type === 'separator') return '<hr class="bpb-sep"/>';
  if (type === 'accordion' || type === 'group') {
    const name = c.name || c.label || 'group';
    const exp = state.expanded[name] !== false;
    const childHtml = exp ? '<div class="bpb-block-body">' + ((c.children||[]).map(x=>renderElement(x,tabs,parentTabId, c.children, currentSelectName)).join('')) + '</div>' : '';
    return '<div class="bpb-block"><div class="bpb-block-head" data-action="toggle-expand" data-name="' + escapeHtml(name) + '">' + escapeHtml(name) + '</div>' + childHtml + '</div>';
  }
  if (type === 'select') {
    const name = c.name || c.i;
    if (!name) return '<div>' + escapeHtml(c.label||'') + '</div>';
    // Collapsed by default
    const exp = state.expanded[name] === true;
    // Sort children by name unless sort: false.
    // List-based options are expanded in place (always alphabetically
    // sorted) so with sort enabled they interleave with normal singles,
    // and with sort: false they stay slotted where the from_list appeared.
    const sortChildren = c.sort !== false;
    let children = [];
    for (const ch of (c.children || [])) {
      if ((ch.type || '').toLowerCase() === 'from_list') {
        for (const opt of expandFromListOptions(ch, name, parentTabId)) children.push(opt);
      } else {
        children.push(ch);
      }
    }
    if (sortChildren) {
      children = [...children].sort((a, b) => String(a.display || a.name || '').localeCompare(String(b.display || b.name || '')));
    }
    const childHtml = exp ? '<div class="bpb-block-body">' + (children.map(x => x._fromList ? renderSingleRow(x.name, x.display, { prompt: x.prompt }) : renderElement(x,tabs,parentTabId, c.children, name)).join('')) + '</div>' : '';
    return '<div class="bpb-block"><div class="bpb-block-head" data-action="toggle-expand" data-name="' + escapeHtml(name) + '"><b>' + escapeHtml(c.label||name) + '</b><span class="bpb-head-right"><span class="bpb-caret">' + (exp ? '▼' : '▶') + '</span><button class="bpb-clear-btn" data-action="clear-select" data-name="' + escapeHtml(name) + '" title="Clear all selections in this list">Clear</button></span></div>' + childHtml + '</div>';
  }
if (type === 'single') {
    const name = c.name || c.i || c.label;
    if (!name) return '';
    return renderSingleRow(name, name, { prompt: c.prompt, emphasis: c.emphasis, is_negative: c.is_negative });
  }
  if (type === 'dual') {
    const name = c.name || c.i || c.label;
    if (!name) return '';
    const act = !!state.activated[name];
    const exp = state.expanded[name] === true;
    const posPrompt = state.values[name + '_prompt_pos'] ?? c.prompt_pos ?? '';
    const negPrompt = state.values[name + '_prompt_neg'] ?? c.prompt_neg ?? '';
    const posEmphasis = parseFloat(state.values[name + '_emphasis_pos'] ?? c.emphasis_pos ?? c.emphasis ?? 1);
    const negEmphasis = parseFloat(state.values[name + '_emphasis_neg'] ?? c.emphasis_neg ?? c.emphasis ?? 1);
    const prefix = c.prefix ? escapeHtml(c.prefix) + ' ' : '';
    const postfix = c.postfix ? ' ' + escapeHtml(c.postfix) : '';
    const childHtml = exp ? '<div class="bpb-block-body">' +
      '<div class="bpb-dual-field">' +
        '<label class="bpb-mini-label">Positive' + (prefix || postfix ? ' (with affixes)' : '') + '</label>' +
        '<div class="bpb-inline-row">' +
          '<input type="text" value="' + escapeHtml(posPrompt) + '" data-input="text" data-name="' + escapeHtml(name + '_prompt_pos') + '"/>' +
          '<input type="number" min="0" step="0.1" value="' + posEmphasis + '" data-input="range" data-name="' + escapeHtml(name + '_emphasis_pos') + '" title="Emphasis (0 to omit)"/>' +
        '</div>' +
      '</div>' +
      '<div class="bpb-dual-field">' +
        '<label class="bpb-mini-label">Negative' + (prefix || postfix ? ' (with affixes)' : '') + '</label>' +
        '<div class="bpb-inline-row">' +
          '<input type="text" value="' + escapeHtml(negPrompt) + '" data-input="text" data-name="' + escapeHtml(name + '_prompt_neg') + '"/>' +
          '<input type="number" min="0" step="0.1" value="' + negEmphasis + '" data-input="range" data-name="' + escapeHtml(name + '_emphasis_neg') + '" title="Emphasis (0 to omit)"/>' +
        '</div>' +
      '</div>' +
    '</div>' : '';
    const affixInfo = (prefix || postfix) ? ' <span class="bpb-affix">[affix: ' + prefix.trim() + ' / ' + postfix.trim() + ']</span>' : '';
    return '<div class="bpb-block"><div class="bpb-block-head" data-action="toggle-expand" data-name="' + escapeHtml(name) + '"><input type="checkbox" ' + (act?'checked':'') + ' data-action="toggle-activate" data-name="' + escapeHtml(name) + '"/><b>' + escapeHtml(name) + '</b>' + affixInfo + ' <span class="bpb-caret bpb-push-right">' + (exp ? '▼' : '▶') + '</span></div>' + childHtml + '</div>';
  }
  if (type === 'edit' || type === 'edit_link') {
    const name = c.name || c.i || c.label;
    if (!name) return '';
    const act = !!state.activated[name];
    const isLinked = type === 'edit_link';
    // Use 'edit' from layout as default, not 'default'
    const val = state.values[name] != null ? state.values[name] : (c.edit != null ? c.edit : (c.default != null ? c.default : 0.5));
    const prefix = c.prefix ? escapeHtml(c.prefix) + ' ' : '';
    const postfix = c.postfix ? ' ' + escapeHtml(c.postfix) : '';
    const rangeText = prefix + '[range: ' + escapeHtml(c.prompt_a||'') + ' → ' + escapeHtml(c.prompt_b||'') + ']' + postfix;
    const linkText = isLinked ? 'Controlled by: ' + escapeHtml(c.link || 'unknown') : '';
    const tooltipText = isLinked 
      ? escapeHtml(rangeText + '\n' + linkText)
      : escapeHtml(rangeText);
    let html = '<div class="bpb-edit">';
    html += '<label class="bpb-edit-label">';
    html += '<input type="checkbox" ' + (act?'checked':'') + ' data-action="toggle-activate" data-name="' + escapeHtml(name) + '"/>';
    html += '<span>' + escapeHtml(name);
    html += ' <span class="bpb-info" title="' + tooltipText + '">ⓘ</span>';
    html += '</span>';
    html += '</label>';
    if (!isLinked) {
      html += '<div class="bpb-edit-body">';
      html += '<div class="bpb-slider-row">';
      html += '<input type="range" min="0" max="1" step="0.1" value="' + val + '" data-input="range" data-name="' + escapeHtml(name) + '"/>';
      html += '<span id="v-' + escapeHtml(name) + '" class="bpb-val">' + val + '</span>';
      html += '</div>';
      html += '</div>';
    }
    html += '</div>';
    return html;
  }
  if (type === 'from_list') {
    // Rendered exactly like normal singles (same row + inline controls).
    // No descriptor header — list options are indistinguishable from singles.
    const opts = expandFromListOptions(c, currentSelectName, parentTabId);
    if (!opts.length) {
      const name = c.name || c.i;
      return '<div class="bpb-list-empty">[from_list: ' + escapeHtml(name) + ' - list empty]</div>';
    }
    return opts.map(o => renderSingleRow(o.name, o.display, { prompt: o.prompt })).join('');
  }
  if (type === 'preset') {
    const name = c.name || c.label || c.i;
    if (!name) return '';
    return '<button class="bpb-preset" data-action="preset" data-name="' + escapeHtml(name) + '">' + escapeHtml(name) + '</button>';
  }
  return '<div class="bpb-unknown">' + escapeHtml(type) + '</div>';
}

function initPanel(panel) {
  loadLayout().then(layout => {
    if (layout && layout.tabs) {
      window.__prompt_builder_lists = layout.lists || {};
      
      function collectActivated(elements) {
        for (const el of elements) {
          if (el.activated) {
            const name = el.name || el.i || el.label;
            if (name) state.activated[name] = true;
          }
          if (el.children) collectActivated(el.children);
        }
      }
      for (const tab of layout.tabs) {
        if (tab.children) collectActivated(tab.children);
      }
      
      // Capture initial state for reset functionality
      initialLayoutState = {
        values: { ...state.values },
        activated: { ...state.activated },
        expanded: { ...state.expanded },
        active_tabs: { ...state.active_tabs }
      };
      
      renderTabs(layout.tabs);
    }
    evalState();
  }).catch(e => console.error('[B Prompt Builder] initPanel error:', e));
}

app.registerExtension({
  name: 'B.PromptBuilder',
  async setup() {
    ensureCss();
    if (!app.extensionManager || !app.extensionManager.registerSidebarTab) {
      console.warn('[B Prompt Builder] ComfyUI sidebar tab API not available');
      return;
    }

    const panel = createPanel();
    let initialized = false;

    app.extensionManager.registerSidebarTab({
      id: id,
      title: 'Prompt Builder',
      icon: 'pi pi-pen-to-square',
      tooltip: 'Build prompts with custom layouts',
      type: 'custom',
      render: (element) => {
        // Set up the outer container for proper flex layout
        element.style.display = 'flex';
        element.style.flexDirection = 'column';
        element.style.height = '100%';
        element.style.minHeight = '0';
        element.appendChild(panel);
        if (!initialized) {
          initPanel(panel);
          initialized = true;
        }
      }
    });
  }
});