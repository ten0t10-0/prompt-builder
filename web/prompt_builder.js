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

// Registry of rendered select names; opening one dropdown closes the rest.
const bpbSelectNames = new Set();

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
      if (p) p.value = j.positive || '';
      if (n) n.value = j.negative || '';
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
    '<button id="bpb-reset" title="Reset all to initial layout state">Reset All</button>' +
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
    '<textarea id="bpb-pos" class="bpb-preview" readonly rows="3" spellcheck="false"></textarea>' +
    '<div class="bpb-preview-label">Negative:</div>' +
    '<textarea id="bpb-neg" class="bpb-preview" readonly rows="3" spellcheck="false"></textarea>';
  
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
    // Layout chrome (active tab, expanded/collapsed) is intentionally left
    // alone — Reset All restores prompt state, not navigation.
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

// Collect base element names in a tab subtree (for tab-scoped reset/clear).
// Mirrors the render traversal: threads the enclosing select name so
// generated from_list item names match what the renderer produces.
function collectTabNames(tab, tabId) {
  const names = [];
  const seen = new Set();
  function add(n) {
    if (n && !seen.has(n)) { seen.add(n); names.push(n); }
  }
  function walk(elements, currentSelectName) {
    for (const el of (elements || [])) {
      const type = ((el.type || '') + '').toLowerCase();
      if (type === 'single' || type === 'dual' || type === 'edit' || type === 'edit_link') {
        add(el.name || el.i || el.label);
      } else if (type === 'select') {
        walk(el.children, el.name || el.i || currentSelectName);
      } else if (type === 'from_list') {
        for (const opt of expandFromListOptions(el, currentSelectName, tabId)) add(opt.name);
      } else if (el.children) {
        walk(el.children, currentSelectName);
      }
    }
  }
  walk(tab.children, undefined);
  return names;
}

// State value keys derived from a base element name.
const BPB_VALUE_SUFFIXES = ['', '_prompt', '_emphasis', '_is_negative', '_prompt_pos', '_prompt_neg', '_emphasis_pos', '_emphasis_neg'];

function resetTabScope(names) {
  if (!initialLayoutState) return;
  for (const base of names) {
    if (base in initialLayoutState.activated) state.activated[base] = initialLayoutState.activated[base];
    else delete state.activated[base];
    if (base in initialLayoutState.expanded) state.expanded[base] = initialLayoutState.expanded[base];
    else delete state.expanded[base];
    for (const s of BPB_VALUE_SUFFIXES) {
      const k = base + s;
      if (k in initialLayoutState.values) state.values[k] = initialLayoutState.values[k];
      else delete state.values[k];
    }
  }
}

function clearTabScope(names) {
  for (const base of names) state.activated[base] = false;
}

// True when anything selectable in a subtree is currently activated.
// Used to only show a select's clear button when there is something to clear.
function subtreeHasActive(elements, currentSelectName, tabId) {
  for (const el of (elements || [])) {
    const type = ((el.type || '') + '').toLowerCase();
    if (type === 'single' || type === 'dual' || type === 'edit' || type === 'edit_link') {
      if (state.activated[el.name || el.i || el.label]) return true;
    } else if (type === 'select') {
      if (subtreeHasActive(el.children, el.name || el.i || currentSelectName, tabId)) return true;
    } else if (type === 'from_list') {
      for (const opt of expandFromListOptions(el, currentSelectName, tabId)) {
        if (state.activated[opt.name]) return true;
      }
    } else if (el.children) {
      if (subtreeHasActive(el.children, currentSelectName, tabId)) return true;
    }
  }
  return false;
}

// Count of activated options in a select subtree (for the closed summary).
function countSelectActive(selectEl, selName, tabId) {
  let n = 0;
  function walk(elements, currentSelectName) {
    for (const el of (elements || [])) {
      const type = ((el.type || '') + '').toLowerCase();
      if (type === 'single' || type === 'dual' || type === 'edit' || type === 'edit_link') {
        if (state.activated[el.name || el.i || el.label]) n++;
      } else if (type === 'select') {
        walk(el.children, el.name || el.i || currentSelectName);
      } else if (type === 'from_list') {
        for (const opt of expandFromListOptions(el, currentSelectName, tabId)) {
          if (state.activated[opt.name]) n++;
        }
      } else if (el.children) {
        walk(el.children, currentSelectName);
      }
    }
  }
  walk(selectEl.children, selName);
  return n;
}

function getTabId(tab, parentId) {
  return parentId ? parentId + '::' + (tab.name || 'tab') : (tab.name || 'root');
}

// ---------- presets ----------
// A preset holds set children; each set mirrors the applicable params of one
// target entry (matched by name), plus an explicit activated flag (default
// true when absent). Select sets hold value children addressed by display
// label (case-insensitive), each carrying single-like params + activated.
// Modes: global (whole layout, unmentioned entries fully cleared), partial
// (only mentioned entries; provided params applied, rest cleared to type
// defaults), additive (only provided params written; explicit
// activated:false switches off without touching params).
// Value prompt text is excluded from presets entirely (it is derived from
// the option); global/partial restores the initial prompt instead of
// blanking it. Standalone singles/duals still clear to blank.
// Type defaults (NOT layout defaults) are used whenever params are cleared.

function scalarMirror(type) {
  if (type === 'single') return [['prompt', '_prompt', ''], ['emphasis', '_emphasis', 1], ['is_negative', '_is_negative', false]];
  if (type === 'dual') return [['prompt_pos', '_prompt_pos', ''], ['prompt_neg', '_prompt_neg', ''], ['emphasis_pos', '_emphasis_pos', 1], ['emphasis_neg', '_emphasis_neg', 1]];
  if (type === 'edit') return [['edit', '', 0.5], ['is_negative', '_is_negative', false]];
  return null;
}

function clearScalarState(base, mirror) {
  state.activated[base] = false;
  for (const [, suffix, def] of mirror) state.values[base + suffix] = def;
}

function applyScalarSet(base, setEl, mirror, clearRest) {
  const explicitOff = setEl.activated != null && !setEl.activated;
  for (const [k, suffix, def] of mirror) {
    if (setEl[k] != null) state.values[base + suffix] = setEl[k];
    else if (clearRest) state.values[base + suffix] = def;
  }
  state.activated[base] = !explicitOff;
}

function collectSelectOptions(selectEl, selName, tabId) {
  const opts = [];
  for (const ch of (selectEl.children || [])) {
    const t = ((ch.type || '') + '').toLowerCase();
    if (t === 'single') {
      const n = ch.name || ch.i || ch.label;
      if (n) opts.push({ stateName: n, display: n, initialPrompt: ch.prompt ?? '' });
    } else if (t === 'from_list') {
      for (const o of expandFromListOptions(ch, selName, tabId)) opts.push({ stateName: o.name, display: o.display, initialPrompt: o.prompt ?? '' });
    }
  }
  return opts;
}

function clearOptionState(opt) {
  state.activated[opt.stateName] = false;
  // Restores the initial prompt (never blank): presets must not corrupt
  // option text, which is derived from the option itself.
  state.values[opt.stateName + '_prompt'] = opt.initialPrompt ?? '';
  state.values[opt.stateName + '_emphasis'] = 1;
  state.values[opt.stateName + '_is_negative'] = false;
}

function applySelectEntry(selectEl, selName, setEl, tabId, clearRest) {
  const options = collectSelectOptions(selectEl, selName, tabId);
  // Select-level activated:false, or a value-less set, fully clears all options.
  const selectOff = !!(setEl && setEl.activated != null && !setEl.activated);
  // First value wins per display label (case-insensitive).
  const byDisplay = new Map();
  if (!selectOff) {
    for (const ch of ((setEl && setEl.children) || [])) {
      if ((((ch.type || '') + '').toLowerCase()) !== 'value') continue;
      const vn = ch.name || ch.i;
      if (!vn) continue;
      const lk = String(vn).toLowerCase();
      if (!byDisplay.has(lk)) byDisplay.set(lk, ch);
    }
  }
  if (selectOff || !byDisplay.size) {
    for (const opt of options) clearOptionState(opt);
    return;
  }
  for (const opt of options) {
    const v = byDisplay.get(String(opt.display).toLowerCase());
    if (!v) {
      if (clearRest) clearOptionState(opt);
      continue;
    }
    const optOff = v.activated != null && !v.activated;
    // NOTE: value prompt is excluded by design — presets never rewrite option
    // text. Global/partial restores the initial prompt; additive leaves it.
    if (clearRest) state.values[opt.stateName + '_prompt'] = opt.initialPrompt ?? '';
    if (v.emphasis != null) state.values[opt.stateName + '_emphasis'] = v.emphasis;
    else if (clearRest) state.values[opt.stateName + '_emphasis'] = 1;
    if (v.is_negative != null) state.values[opt.stateName + '_is_negative'] = v.is_negative;
    else if (clearRest) state.values[opt.stateName + '_is_negative'] = false;
    state.activated[opt.stateName] = !optOff;
  }
}

function findPresetByName(tabs, name) {
  for (const el of (tabs || [])) {
    if ((((el.type || '') + '').toLowerCase()) === 'preset' && (el.name === name || el.i === name)) return el;
    if (el.children) {
      const found = findPresetByName(el.children, name);
      if (found) return found;
    }
  }
  return null;
}

function applyPreset(presetEl, tabs) {
  const rawMode = String(presetEl.mode != null ? presetEl.mode : 'partial').toLowerCase();
  const mode = rawMode === 'global' ? 'global' : rawMode === 'additive' ? 'additive' : 'partial';
  const clearRest = mode !== 'additive';
  // First set wins per target entry.
  const setMap = {};
  for (const ch of (presetEl.children || [])) {
    if ((((ch.type || '') + '').toLowerCase()) !== 'set') continue;
    const k = ch.name || ch.i;
    if (k && !(k in setMap)) setMap[k] = ch;
  }
  function lookup(el) {
    const k = el.name || el.i || el.label;
    return (k && setMap[k]) || null;
  }
  // Walk tab pages (all of them, not just active); tabId threading mirrors
  // the renderer so standalone from_list fallbacks resolve identically.
  function walkTabs(tabsArr, tabId) {
    for (const tab of (tabsArr || [])) {
      walkContents(tab.children, undefined, tabId);
      const tabKids = (tab.children || []).filter(c => (((c.type || '') + '').toLowerCase()) === 'tab');
      if (tabKids.length) walkTabs(tabKids, tabId + '::subtabs');
    }
  }
  function walkContents(elements, currentSelectName, tabId) {
    for (const el of (elements || [])) {
      const type = ((el.type || '') + '').toLowerCase();
      if (type === 'preset' || type === 'set' || type === 'value') continue;
      if (type === 'tab') {
        const nonTabs = (el.children || []).filter(c => (((c.type || '') + '').toLowerCase()) !== 'tab');
        walkContents(nonTabs, currentSelectName, tabId);
        const nested = (el.children || []).filter(c => (((c.type || '') + '').toLowerCase()) === 'tab');
        if (nested.length) walkTabs(nested, tabId + '::' + (el.name || 'tab') + '::subtabs');
        continue;
      }
      if (type === 'single' || type === 'dual' || type === 'edit') {
        const key = el.name || el.i || el.label;
        if (!key) continue;
        const set = lookup(el);
        const mirror = scalarMirror(type);
        if (set) applyScalarSet(key, set, mirror, clearRest);
        else if (mode === 'global') clearScalarState(key, mirror);
      } else if (type === 'edit_link') {
        // Only activation control is applicable; other keys are ignored.
        const key = el.name || el.i || el.label;
        if (!key) continue;
        const set = lookup(el);
        if (set && set.activated != null) state.activated[key] = !!set.activated;
        else if (mode === 'global' && !set) state.activated[key] = false;
      } else if (type === 'select') {
        const selName = el.name || el.i;
        if (selName) {
          const set = setMap[selName] || null;
          if (set) applySelectEntry(el, selName, set, tabId, clearRest);
          else if (mode === 'global') applySelectEntry(el, selName, null, tabId, true);
        }
        // Recurse into nested containers only (direct options handled above).
        for (const ch of (el.children || [])) {
          const ct = ((ch.type || '') + '').toLowerCase();
          if (ct === 'single' || ct === 'from_list' || ct === 'value' || ct === 'set') continue;
          if (ch.children) walkContents([ch], selName, tabId);
        }
      } else if (type === 'from_list') {
        // Standalone list options: only global touches them (sets can't address them).
        if (mode === 'global') {
          for (const o of expandFromListOptions(el, currentSelectName, tabId)) {
            state.activated[o.name] = false;
            state.values[o.name + '_prompt'] = o.prompt ?? '';
            state.values[o.name + '_emphasis'] = 1;
            state.values[o.name + '_is_negative'] = false;
          }
        }
      } else if (el.children) {
        walkContents(el.children, currentSelectName, tabId);
      }
    }
  }
  walkTabs(tabs, 'root');
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

function panelSelectName(panel) {
  const block = panel.closest('.bpb-block');
  const head = block ? block.querySelector('[data-select-toggle]') : null;
  return head ? head.getAttribute('data-name') : null;
}

// Size open dropdown panels to the visible scroll area below their header,
// flipping upward when there is no room. Measuring against the content
// wrapper (not the viewport) keeps the panel above the sticky preview
// footer and avoids forcing a second scrollbar on the wrapper.
function fitSelectPanels(cont) {
  cont.querySelectorAll('.bpb-select-panel').forEach(panel => {
    const block = panel.closest('.bpb-block');
    const head = block ? block.querySelector('[data-select-toggle]') : null;
    const rect = (head || panel).getBoundingClientRect();
    const wrap = panel.closest('.bpb-content-wrap');
    const wrapRect = wrap ? wrap.getBoundingClientRect() : null;
    const viewBottom = wrapRect ? wrapRect.bottom : window.innerHeight;
    const viewTop = wrapRect ? wrapRect.top : 0;
    const margin = 8, minH = 96;
    const below = Math.floor(viewBottom - rect.bottom - 2 - margin);
    const above = Math.floor(rect.top - viewTop - margin);
    if (below < minH && above > below) {
      panel.style.top = 'auto';
      panel.style.bottom = 'calc(100% + 2px)';
      panel.style.maxHeight = Math.max(minH, above) + 'px';
    } else {
      panel.style.top = '';
      panel.style.bottom = '';
      panel.style.maxHeight = Math.max(minH, below) + 'px';
    }
  });
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

  // Tab-scoped actions at the bottom: Clear first, then Reset (matches
  // the global header order), right-aligned, labeled with the tab name.
  // Opt-out via no_reset (legacy is_reset_visible: false also hides).
  const tabActionsHidden = tab.no_reset === true || tab.no_reset === 'true' || tab.is_reset_visible === false;
  if (!tabActionsHidden) {
    const tabLabel = escapeHtml(tab.name || 'Tab');
    html += '<div class="bpb-tab-footer">' +
      '<button data-action="tab-clear" title="Clear selections in ' + tabLabel + '">Clear ' + tabLabel + '</button>' +
      '<button data-action="tab-reset" title="Reset ' + tabLabel + ' to initial state">Reset ' + tabLabel + '</button>' +
    '</div>';
  }
  
  // Preserve open dropdown scroll across the re-render below.
  const panelScroll = new Map();
  cont.querySelectorAll('.bpb-select-panel').forEach(panel => {
    const selName = panelSelectName(panel);
    if (selName) panelScroll.set(selName, panel.scrollTop);
  });

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
        const willOpen = !state.expanded[name];
        if (willOpen && el.hasAttribute('data-select-toggle')) {
          // Dropdown behavior: opening one select closes the others.
          for (const n of bpbSelectNames) if (n !== name) state.expanded[n] = false;
        }
        state.expanded[name] = willOpen;
        renderNestedContent(tabs, tabId);
      } else if (act === 'close-select') {
        state.expanded[name] = false;
        renderNestedContent(tabs, tabId);
      } else if (act === 'tab-reset') {
        const idx = state.active_tabs[tabId] || 0;
        const currentTab = tabs[idx];
        if (currentTab) resetTabScope(collectTabNames(currentTab, tabId));
        scheduleEval();
        renderNestedContent(tabs, tabId);
      } else if (act === 'tab-clear') {
        const idx = state.active_tabs[tabId] || 0;
        const currentTab = tabs[idx];
        if (currentTab) clearTabScope(collectTabNames(currentTab, tabId));
        scheduleEval();
        renderNestedContent(tabs, tabId);
      } else if (act === 'preset') {
        // Presets can target the whole layout, so reload it fresh and
        // re-render everything after patching state.
        loadLayout().then(layout => {
          if (layout && layout.tabs) {
            window.__prompt_builder_lists = layout.lists || {};
            window.__prompt_builder_layout = layout;
            const presetEl = findPresetByName(layout.tabs, name);
            if (presetEl) applyPreset(presetEl, layout.tabs);
            renderTabs(layout.tabs);
          }
          evalState();
        });
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
  // Fit open dropdowns to the available space, then restore their scroll.
  fitSelectPanels(cont);
  cont.querySelectorAll('.bpb-select-panel').forEach(panel => {
    const selName = panelSelectName(panel);
    if (selName && panelScroll.has(selName)) panel.scrollTop = panelScroll.get(selName);
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
// Code-point ordering (matches the legacy app): '(' sorts before '-',
// unlike localeCompare, whose ICU collation reorders punctuation.
function compareLabels(a, b) {
  const sa = String(a), sb = String(b);
  if (sa < sb) return -1;
  if (sa > sb) return 1;
  return 0;
}

function toTitleCase(s) {
  return String(s).toLowerCase().replace(/(?:^|\s)\S/g, c => c.toUpperCase());
}
function expandFromListOptions(c, parentSelectName, parentTabId) {
  const listName = c.name || c.i;
  const postfix = c.postfix || '';
  const selectName = parentSelectName || (parentTabId || '').split('::').pop() || 'select';
  const items = getListItems(listName);
  const safePostfix = postfix ? '_' + String(postfix).replace(/\s+/g, '_') : '';
  const opts = (items || []).map(item => ({
    type: 'single',
    name: selectName + '_' + listName + safePostfix + '_' + item,
    // Display-only: "{postfix} - {item}" in Title Case. Prompt untouched.
    display: toTitleCase(postfix ? postfix + ' - ' + item : item),
    prompt: String(item).toLowerCase(),
    _fromList: true
  }));
  opts.sort((a, b) => compareLabels(a.display, b.display));
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
    bpbSelectNames.add(name);
    // Dropdown: collapsed header with selection count; options in a floating
    // panel with a click-catching backdrop. Collapsed by default.
    const exp = state.expanded[name] === true;
    // Sort children by display label unless sort: false.
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
      children = [...children].sort((a, b) => compareLabels(a.display || a.name || '', b.display || b.name || ''));
    }
    const activeCount = countSelectActive(c, name, parentTabId);
    const countBadge = activeCount > 0 ? ' <span class="bpb-select-count">(' + activeCount + ')</span>' : '';
    // Icon clear button, only when something inside is actively selected.
    const selectClearBtn = subtreeHasActive(c.children, name, parentTabId)
      ? '<button class="bpb-clear-btn" data-action="clear-select" data-name="' + escapeHtml(name) + '" title="Clear selections in this list">×</button>'
      : '';
    const panelHtml = exp
      ? '<div class="bpb-select-backdrop" data-action="close-select" data-name="' + escapeHtml(name) + '"></div>' +
        '<div class="bpb-select-panel">' + (children.map(x => x._fromList ? renderSingleRow(x.name, x.display, { prompt: x.prompt }) : renderElement(x,tabs,parentTabId, c.children, name)).join('')) + '</div>'
      : '';
    return '<div class="bpb-block"><div class="bpb-block-head" data-action="toggle-expand" data-select-toggle data-name="' + escapeHtml(name) + '"><b>' + escapeHtml(c.label||name) + '</b>' + countBadge + '<span class="bpb-head-right"><span class="bpb-caret">' + (exp ? '▼' : '▶') + '</span>' + selectClearBtn + '</span></div>' + panelHtml + '</div>';
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
    const isNegative = state.values[name + '_is_negative'] ?? c.is_negative ?? false;
    const prefix = c.prefix ? escapeHtml(c.prefix) + ' ' : '';
    const postfix = c.postfix ? ' ' + escapeHtml(c.postfix) : '';
    const rangeText = prefix + '[range: ' + escapeHtml(c.prompt_a||'') + ' → ' + escapeHtml(c.prompt_b||'') + ']' + postfix;
    const linkText = isLinked ? 'Controlled by: ' + escapeHtml(c.link || 'unknown') + ' (follows value and negative)' : '';
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
      html += '<label class="bpb-neg-toggle" title="Toggle negative prompt">' +
        '<input type="checkbox" ' + (isNegative?'checked':'') + ' data-input="checkbox" data-name="' + escapeHtml(name + '_is_negative') + '"/>' +
        '<span>N</span>' +
      '</label>';
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
      window.__prompt_builder_layout = layout;
      
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

      // Seed initial expand/collapse from the open param (absent = type
      // default: group/accordion expanded, dual collapsed; selects are
      // dropdowns and always start closed).
      function collectOpen(elements) {
        for (const el of (elements || [])) {
          const t = ((el.type || '') + '').toLowerCase();
          const n = el.name || el.i || el.label;
          if (n && (t === 'group' || t === 'accordion' || t === 'dual') && el.open != null) {
            state.expanded[n] = !!el.open;
          }
          if (el.children) collectOpen(el.children);
        }
      }
      for (const tab of layout.tabs) {
        if (tab.children) collectOpen(tab.children);
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