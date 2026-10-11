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
let evalSeq = 0;
function scheduleEval() {
  if (evalTimer) clearTimeout(evalTimer);
  evalTimer = setTimeout(evalState, 150);
}

async function evalState(retry) {
  const mySeq = ++evalSeq;
  try { writeWidgetFromPanel(); } catch (e) {}
  try {
    // Push state to backend for node to pick up (also runs unbound so the
    // shared mirror never goes stale).
    await fetch('/extensions/prompt-builder/state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: state })
    });
  } catch (e) {}
  // Unbound (no node in this workflow): no output exists, so keep the
  // previews cleared instead of evaluating layout defaults.
  if (bpbBoundNodeId == null) {
    const p0 = document.getElementById('bpb-pos');
    const n0 = document.getElementById('bpb-neg');
    if (p0) p0.value = '';
    if (n0) n0.value = '';
    return;
  }
  try {
    const resp = await fetch('/extensions/prompt-builder/eval', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: JSON.stringify(state) })
    });
    if (!resp.ok) throw new Error('eval failed: ' + resp.status);
    const j = await resp.json();
    // Drop superseded responses: a newer eval started while this one was in
    // flight, so painting here would regress the previews.
    if (mySeq !== evalSeq) return;
    const p = document.getElementById('bpb-pos');
    const n = document.getElementById('bpb-neg');
    if (p) p.value = j.positive || '';
    if (n) n.value = j.negative || '';
  } catch (e) {
    // A failed eval leaves stale/empty previews with no recovery trigger,
    // so retry once (e.g. server busy during workflow load).
    if (!retry) setTimeout(() => { try { evalState(true); } catch (e2) {} }, 1000);
  }
}

function createPanel() {
  const panel = document.createElement('div');
  panel.id = id;
  panel.className = 'bpb-panel';
  
  // Header with title and controls
  const header = document.createElement('div');
  header.className = 'bpb-header';
  header.innerHTML = 
    '<span class="bpb-title">B Prompt Builder<span id="bpb-bound-label" class="bpb-bound-label"></span></span>' +
    '<div class="bpb-header-actions">' +
    '<button id="bpb-clear-all" title="Clear all selections">Clear All</button>' +
    '<button id="bpb-reset" title="Reset all to initial layout state">Reset All</button>' +
    '</div>';
  
  // Scrollable content area
  const contentWrapper = document.createElement('div');
  contentWrapper.className = 'bpb-content-wrap';
  contentWrapper.innerHTML =
    '<div id="bpb-tabs"></div>' +
    '<div id="bpb-content"></div>' +
    '<div id="bpb-nobind" class="bpb-empty-note" style="display:none">No Prompt Builder node found in this workflow.<br>Add one from the node menu to start building prompts.</div>';
  
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

// Activation stats for a select subtree in one walk: whether anything is
// active (clear-button visibility) and how many (closed summary).
function selectActiveStats(selectEl, selName, tabId) {
  const stats = { has: false, count: 0 };
  function walk(elements, currentSelectName) {
    for (const el of (elements || [])) {
      const type = ((el.type || '') + '').toLowerCase();
      if (type === 'single' || type === 'dual' || type === 'edit' || type === 'edit_link') {
        if (state.activated[el.name || el.i || el.label]) { stats.has = true; stats.count++; }
      } else if (type === 'select') {
        walk(el.children, el.name || el.i || currentSelectName);
      } else if (type === 'from_list') {
        for (const opt of expandFromListOptions(el, currentSelectName, tabId)) {
          if (state.activated[opt.name]) { stats.has = true; stats.count++; }
        }
      } else if (el.children) {
        walk(el.children, currentSelectName);
      }
    }
  }
  walk(selectEl.children, selName);
  return stats;
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
  // the renderer (including per-tab subtab ids) so standalone from_list
  // fallbacks resolve identically.
  function walkTabs(tabsArr, tabId) {
    const arr = tabsArr || [];
    for (let ti = 0; ti < arr.length; ti++) {
      const tab = arr[ti];
      walkContents(tab.children, undefined, tabId);
      const tabKids = (tab.children || []).filter(c => (((c.type || '') + '').toLowerCase()) === 'tab');
      if (tabKids.length) walkTabs(tabKids, tabId + '::' + (tab.name || ('tab' + ti)) + '::subtabs');
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

// ---------- per-workflow node binding ----------
// Panel state persists per PromptBuilder node via a hidden BPB_STATE widget
// (serialized into the workflow JSON). The panel binds to a single node per
// workflow tab: the graph-remembered node if still present, else the first
// node found (top level first, then subgraph interiors). No selection
// following: clicking nodes never rebinds. Panel edits stream into the bound
// widget continuously, and binding (re)loads the widget, so tab switches are
// lossless. Multi-node workflows show the first node; per-node panel tabs
// remain a possible future step (they'd need rename tracking).
let bpbBoundNodeId = null;
const bpbGraphBinding = new Map(); // LGraph -> node id
let bpbPanelReady = false;
let bpbDirtyTimer = null;
let bpbBindSeq = 0;
let bpbWidgetApplied = false;

function bpbGraphNodeList(graph) {
  // Version-proof, subgraph-aware enumeration from the graph root.
  // Subgraph interiors are NOT in the top-level list, so recurse via
  // node.subgraph (official walkGraph pattern). Depth-first, top level
  // first, cycle-guarded. NOTE: node ids are local per graph level, so a
  // first id match wins by walk order.
  const out = [];
  const seenGraphs = new Set();
  function walk(g) {
    if (!g || seenGraphs.has(g)) return;
    seenGraphs.add(g);
    let level = [];
    try {
      const src = g.nodes != null ? g.nodes : g._nodes;
      if (!src) return;
      if (Array.isArray(src)) level = src.filter(n => n);
      else if (typeof src.values === 'function') { for (const n of src.values()) if (n) level.push(n); }
      else { for (const k of Object.keys(src)) { const n = src[k]; if (n) level.push(n); } }
    } catch (e) { return; }
    for (const n of level) {
      out.push(n);
      try { if (n.subgraph) walk(n.subgraph); } catch (e) {}
    }
  }
  try { walk(graph || app.graph); } catch (e) {}
  return out;
}

function bpbFindNodeById(graph, id) {
  try {
    const g = graph || app.graph;
    if (g && typeof g.getNodeById === 'function') {
      const n = g.getNodeById(id);
      if (n) return n;
    }
  } catch (e) {}
  const list = bpbGraphNodeList(graph);
  for (const n of list) {
    try { if (n && n.id === id) return n; } catch (e) {}
  }
  return null;
}

function bpbFindNodes(graph) {
  return bpbGraphNodeList(graph).filter(n => n && n.type === 'PromptBuilderNode');
}

function bpbGetWidget(node) {
  if (!node || !node.widgets) return null;
  for (const w of node.widgets) {
    if (w && w.name === 'state') return w;
  }
  return null;
}

function chooseBindingForGraph(graph) {
  const nodes = bpbFindNodes(graph);
  if (!nodes.length) return null;
  try {
    const remembered = bpbGraphBinding.get(graph);
    if (remembered != null) {
      for (const n of nodes) if (n.id === remembered) return remembered;
    }
  } catch (e) {}
  return nodes[0].id;
}

// Rebind only when the target differs: keeps repeated graph-configured
// events and polls from repainting a settled panel.
function rebindIfChanged() {
  try {
    if (!bpbPanelReady || !app.graph) return;
    const id = chooseBindingForGraph(app.graph);
    if (id !== bpbBoundNodeId) bindPanelToNode(id);
  } catch (e) {}
}

// Self-healer: if bound but the tab bar never painted (e.g. a transient
// fetch failure or mid-load ordering on workflow open), re-render once.
// Skips entirely while the panel is detached (nothing could paint anyway).
function ensurePanelPainted() {
  try {
    if (!bpbPanelReady || bpbBoundNodeId == null) return;
    const tabsEl = document.getElementById('bpb-tabs');
    if (!tabsEl) return;
    const knownTabs = (window.__prompt_builder_layout && window.__prompt_builder_layout.tabs) || [];
    if (!knownTabs.length) return;
    if (tabsEl.querySelector('.bpb-tabbar')) return;
    loadLayout().then(layout => {
      if (layout && layout.tabs) {
        window.__prompt_builder_lists = layout.lists || {};
        window.__prompt_builder_layout = layout;
        renderTabs(layout.tabs);
      }
      evalState();
    });
  } catch (e) {}
}

// Runs on EVERY sidebar open (not just the first): the tab DOM is mounted
// lazily, so reconcile visibility + paint against current binding state,
// then refresh the previews so the footer can never show stale content.
function syncPanelToDom() {
  try {
    refreshPanelBinding();
    ensurePanelPainted();
    evalState();
  } catch (e) {}
}

function serializePanelState() {
  return JSON.stringify({ values: state.values, activated: state.activated, expanded: state.expanded });
}

function markBpbDirty() {
  // Same signal core widget edits produce (dirty indicator + auto-queue);
  // saving itself always reads live widget values regardless.
  try {
    if (app.configuringGraph) return;
  } catch (e) {}
  if (bpbDirtyTimer) return;
  bpbDirtyTimer = setTimeout(() => {
    bpbDirtyTimer = null;
    try {
      const payload = app.graph.serialize();
      if (app.api && typeof app.api.dispatchCustomEvent === 'function') {
        app.api.dispatchCustomEvent('graphChanged', payload);
      } else if (app.api && typeof app.api.dispatchEvent === 'function') {
        app.api.dispatchEvent(new CustomEvent('graphChanged', { detail: payload }));
      }
    } catch (e) {}
  }, 200);
}

function writeWidgetFromPanel() {
  if (bpbBoundNodeId == null || !app.graph) return;
  let node = null;
  try { node = bpbFindNodeById(app.graph, bpbBoundNodeId); } catch (e) { return; }
  if (!node || node.type !== 'PromptBuilderNode') return;
  const widget = bpbGetWidget(node);
  if (!widget) return;
  const json = serializePanelState();
  try {
    if (widget.value !== json) widget.value = json; // setter routes via setValue + callback(dirty)
  } catch (e) {}
  try { if (node.setDirtyCanvas) node.setDirtyCanvas(true, true); } catch (e) {}
}

function seedPanelLayoutDefaults(tabs) {
  state.values = {};
  state.activated = {};
  state.expanded = {};
  function collectActivated(elements) {
    for (const el of elements) {
      if (el.activated) {
        const name = el.name || el.i || el.label;
        if (name) state.activated[name] = true;
      }
      if (el.children) collectActivated(el.children);
    }
  }
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
  for (const tab of (tabs || [])) {
    if (tab.children) { collectActivated(tab.children); collectOpen(tab.children); }
  }
}

// Applies the bound widget's saved state over freshly seeded defaults.
// Returns true when a saved (non-virgin) state was applied.
function applyBoundWidgetState() {
  if (bpbBoundNodeId == null || !app.graph) return false;
  let node = null;
  try { node = bpbFindNodeById(app.graph, bpbBoundNodeId); } catch (e) { return false; }
  const widget = bpbGetWidget(node);
  if (!widget) return false;
  let d = null;
  try {
    const raw = widget.value;
    d = (typeof raw === 'string') ? JSON.parse(raw) : raw;
  } catch (e) { return false; }
  if (!d || typeof d !== 'object' || (!('values' in d) && !('activated' in d) && !('expanded' in d))) return false;
  if (d.values && typeof d.values === 'object') state.values = { ...d.values };
  if (d.activated && typeof d.activated === 'object') state.activated = { ...d.activated };
  if (d.expanded && typeof d.expanded === 'object') state.expanded = { ...d.expanded };
  return true;
}

function refreshPanelBinding() {
  const tabsEl = document.getElementById('bpb-tabs');
  const contentEl = document.getElementById('bpb-content');
  const noteEl = document.getElementById('bpb-nobind');
  const labelEl = document.getElementById('bpb-bound-label');
  let title = 'Not bound';
  if (bpbBoundNodeId == null) {
    if (tabsEl) { tabsEl.style.display = 'none'; tabsEl.innerHTML = ''; }
    if (contentEl) { contentEl.style.display = 'none'; contentEl.innerHTML = ''; }
    if (noteEl) noteEl.style.display = '';
    const p = document.getElementById('bpb-pos');
    const n = document.getElementById('bpb-neg');
    if (p) p.value = '';
    if (n) n.value = '';
  } else {
    if (tabsEl) tabsEl.style.display = '';
    if (contentEl) contentEl.style.display = '';
    if (noteEl) noteEl.style.display = 'none';
    try {
      const node = app.graph ? bpbFindNodeById(app.graph, bpbBoundNodeId) : null;
      if (node) title = node.title || ('Node #' + node.id);
    } catch (e) {}
  }
  if (labelEl) labelEl.textContent = title;
}

function bindPanelToNode(nodeId, preloaded) {
  const mySeq = ++bpbBindSeq;
  bpbBoundNodeId = nodeId;
  try {
    if (app.graph) {
      if (nodeId == null) bpbGraphBinding.delete(app.graph);
      else bpbGraphBinding.set(app.graph, nodeId);
    }
  } catch (e) {}
  refreshPanelBinding();
  const tabs = (window.__prompt_builder_layout && window.__prompt_builder_layout.tabs) || (preloaded && preloaded.tabs) || [];
  seedPanelLayoutDefaults(tabs);
  bpbWidgetApplied = applyBoundWidgetState();
  const finish = (layout) => {
    // Drop stale renders: only the latest bind may paint.
    if (mySeq !== bpbBindSeq) return;
    if (layout && layout.tabs) {
      window.__prompt_builder_lists = layout.lists || {};
      window.__prompt_builder_layout = layout;
      renderTabs(layout.tabs);
    }
    evalState();
  };
  if (preloaded && preloaded.tabs) finish(preloaded);
  else loadLayout().then(l => finish(l));
}

function renderTabs(tabs, parentId = null) {
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
  
  // If there are tab children, render a tab bar for them. The sub-tab id
  // includes this tab's name so sibling tabs don't share selection state.
  // ('::subtabs' stays last so from_list fallbacks resolve identically.)
  if (tabChildren.length > 0) {
    const subTabId = tabId + '::' + (tab.name || ('tab' + activeIdx)) + '::subtabs';
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
        scheduleEval(); // persist open state to the widget (drives save)
        renderNestedContent(tabs, tabId);
      } else if (act === 'close-select') {
        state.expanded[name] = false;
        scheduleEval(); // persist open state to the widget (drives save)
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
    const stats = selectActiveStats(c, name, parentTabId);
    const countBadge = stats.count > 0 ? ' <span class="bpb-select-count">(' + stats.count + ')</span>' : '';
    // Icon clear button, only when something inside is actively selected.
    const selectClearBtn = stats.has
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
      seedPanelLayoutDefaults(layout.tabs);

      // Capture initial state for reset functionality
      initialLayoutState = {
        values: { ...state.values },
        activated: { ...state.activated },
        expanded: { ...state.expanded },
        active_tabs: { ...state.active_tabs }
      };

      bpbPanelReady = true;
      bindPanelToNode(chooseBindingForGraph(app.graph), layout);
    } else {
      bpbPanelReady = true;
      evalState();
    }
  }).catch(e => console.error('[B Prompt Builder] initPanel error:', e));
}

app.registerExtension({
  name: 'B.PromptBuilder',
  // Hidden per-node state store: serialized into the workflow JSON like any
  // other widget value, and fed to the backend as the `state` input.
  getCustomWidgets(app) {
    return {
      BPB_STATE(node, inputName, inputData) {
        const el = document.createElement('div');
        el.style.display = 'none';
        let initial = '{}';
        try {
          const opts = Array.isArray(inputData) ? inputData[1] : (inputData && (inputData.options || inputData));
          if (opts && typeof opts.default === 'string') initial = opts.default;
        } catch (e) {}
        let stored = initial;
        const widget = node.addDOMWidget(inputName, 'hidden', el, {
          getValue: () => stored,
          setValue: (v) => { stored = (typeof v === 'string') ? v : JSON.stringify(v ?? {}); },
        });
        try { widget.hidden = true; } catch (e) {}
        try { widget.serialize = true; } catch (e) {}
        widget.callback = () => { markBpbDirty(); };
        return widget;
      }
    };
  },
  async nodeCreated(node) {
    // Freshly added node while unbound (e.g. first node in the workflow):
    // bind the panel to it.
    try {
      if (!node || node.type !== 'PromptBuilderNode') return;
      if (app.configuringGraph) return;
      if (bpbPanelReady && bpbBoundNodeId == null) bindPanelToNode(node.id);
    } catch (e) {}
  },
  async afterConfigureGraph() {
    // Fires on workflow load and workflow-tab switches: rebind only when the
    // resolved node differs, so repeated events can't repaint a settled
    // panel. A deferred settle pass covers mid-load orderings (transient
    // empty graph, widget values not yet applied) and repaint misses.
    try {
      if (!bpbPanelReady || !app.graph) return;
      rebindIfChanged();
      setTimeout(() => {
        try {
          if (!bpbPanelReady || !app.graph) return;
          rebindIfChanged();
          // Widget values may not have been applied at bind time; if the
          // bind showed defaults, retry once now (panel edits stream into
          // the widget, so re-applying can only restore saved content).
          if (!bpbWidgetApplied && applyBoundWidgetState()) {
            const layout = window.__prompt_builder_layout;
            if (layout && layout.tabs) renderTabs(layout.tabs);
            evalState();
          }
          ensurePanelPainted('settle');
        } catch (e) {}
      }, 400);
    } catch (e) {}
  },
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
        } else {
          // Sync now for the live-DOM case, and once more after ComfyUI
          // finishes any container rebuild/restore around this render call
          // (icon-open path). Both passes are idempotent.
          syncPanelToDom();
          setTimeout(() => { try { syncPanelToDom(); } catch (e) {} }, 60);
        }
      }
    });

    // Backstop poll (1s): rebind only when the bound node vanished or when
    // unbound and a node appeared. No selection following.
    setInterval(() => {
      try {
        if (document.hidden) return; // idle in background tabs
        if (!bpbPanelReady || !app.graph) return;
        if (bpbBoundNodeId != null) {
          const node = bpbFindNodeById(app.graph, bpbBoundNodeId);
          if (!node || node.type !== 'PromptBuilderNode') rebindIfChanged();
          else ensurePanelPainted();
        } else {
          rebindIfChanged();
        }
      } catch (e) {}
    }, 1000);
  }
});