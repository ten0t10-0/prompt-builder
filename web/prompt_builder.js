import { app } from "../../scripts/app.js";

const id = 'b-prompt-builder-panel';

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
  panel.style.display = 'flex';
  panel.style.flexDirection = 'column';
  panel.style.height = '100%';
  panel.style.minHeight = '0'; // Critical for flex child to shrink
  
  // Header with title and controls
  const header = document.createElement('div');
  header.style.padding = '8px';
  header.style.borderBottom = '1px solid var(--border-color,#555)';
  header.style.display = 'flex';
  header.style.justifyContent = 'space-between';
  header.style.alignItems = 'center';
  header.style.flexShrink = '0';
  header.innerHTML = 
    '<span style="font-weight:bold">B Prompt Builder</span>' +
    '<div style="display:flex;gap:4px">' +
    '<button id="bpb-clear-all" style="padding:2px 8px;font-size:0.75em;background:var(--comfy-input-bg,#333);color:var(--fg-color,#eee);border:1px solid var(--border-color,#555);border-radius:3px" title="Clear all selections">Clear All</button>' +
    '<button id="bpb-reset" style="padding:2px 8px;font-size:0.75em;background:var(--comfy-input-bg,#333);color:var(--fg-color,#eee);border:1px solid var(--border-color,#555);border-radius:3px" title="Reset to initial layout state">Reset</button>' +
    '</div>';
  
  // Scrollable content area
  const contentWrapper = document.createElement('div');
  contentWrapper.style.flex = '1';
  contentWrapper.style.overflow = 'auto';
  contentWrapper.style.padding = '8px';
  contentWrapper.style.minHeight = '0'; // Critical for flex child to shrink
  contentWrapper.innerHTML =
    '<div id="bpb-tabs"></div>' +
    '<div id="bpb-content" style="margin-top:8px;"></div>';
  
  // Sticky footer with prompt previews
  const footer = document.createElement('div');
  footer.style.padding = '8px';
  footer.style.borderTop = '1px solid var(--border-color,#555)';
  footer.style.flexShrink = '0';
  footer.style.background = 'var(--comfy-menu-bg,#1e1e1e)';
  footer.innerHTML =
    '<div style="font-size:0.75em;color:gray;margin-bottom:2px">Positive:</div>' +
    '<div id="bpb-pos" style="font-size:0.8em;word-break:break-word;white-space:pre-wrap;max-height:120px;overflow:auto;font-family:monospace;background:var(--comfy-input-bg,#333);padding:6px;border-radius:3px;border:1px solid var(--border-color,#555)"></div>' +
    '<div style="font-size:0.75em;color:gray;margin:6px 0 2px">Negative:</div>' +
    '<div id="bpb-neg" style="font-size:0.8em;word-break:break-word;white-space:pre-wrap;max-height:120px;overflow:auto;font-family:monospace;background:var(--comfy-input-bg,#333);padding:6px;border-radius:3px;border:1px solid var(--border-color,#555)"></div>';
  
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
  let html = '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px">';
  for (let i = 0; i < tabs.length; i++) {
    const active = activeIdx === i ? 'font-weight:bold;border-bottom:2px solid var(--accent-color,#00bcd4)' : '';
    html += '<button data-tabid="' + escapeHtml(tabId) + '" data-i="' + i + '" style="padding:4px 8px;background:var(--comfy-input-bg,#333);color:var(--fg-color,#eee);border:1px solid var(--border-color,#555);' + active + '">' + escapeHtml(tabs[i].name || 'Tab ' + (i+1)) + '</button>';
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
    html += '<div id="bpb-tabs-' + escapeHtml(safeId) + '" style="margin-top:8px"></div>';
    html += '<div id="bpb-content-' + escapeHtml(safeId) + '" style="margin-top:8px;padding-left:8px;border-left:2px solid var(--border-color,#555)"></div>';
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
      }
    };
  });
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
      html += '<div id="bpb-tabs-' + escapeHtml(safeId) + '" style="margin-top:8px"></div>';
      html += '<div id="bpb-content-' + escapeHtml(safeId) + '" style="margin-top:8px;padding-left:8px;border-left:2px solid var(--border-color,#555)"></div>';
      setTimeout(() => renderTabs(tabChildren, subTabId), 0);
    }
    return html;
  }
  if (type === 'row') {
    let html = '<div style="display:flex;gap:8px;flex-wrap:wrap">';
    for (let ch of (c.children||[])) html += '<div style="flex:1;min-width:120px">' + renderElement(ch,tabs,parentTabId, c.children, currentSelectName) + '</div>';
    html += '</div>';
    return html;
  }
  if (type === 'column') {
    let html = '<div style="display:flex;flex-direction:column;gap:6px">';
    for (let ch of (c.children||[])) html += renderElement(ch,tabs,parentTabId, c.children, currentSelectName);
    html += '</div>';
    return html;
  }
  if (type === 'separator') return '<hr style="border-color:var(--border-color,#555)"/>';
  if (type === 'accordion' || type === 'group') {
    const name = c.name || c.label || 'group';
    const exp = state.expanded[name] !== false;
    const childHtml = exp ? '<div style="padding:6px">' + ((c.children||[]).map(x=>renderElement(x,tabs,parentTabId, c.children, currentSelectName)).join('')) + '</div>' : '';
    return '<div style="border:1px solid var(--border-color,#555);margin-bottom:6px"><div style="padding:4px;background:var(--comfy-input-bg,#333);cursor:pointer" data-action="toggle-expand" data-name="' + escapeHtml(name) + '">' + escapeHtml(name) + '</div>' + childHtml + '</div>';
  }
  if (type === 'select') {
    const name = c.name || c.i;
    if (!name) return '<div>' + escapeHtml(c.label||'') + '</div>';
    // Collapsed by default
    const exp = state.expanded[name] === true;
    // Sort children by name unless sort: false
    const sortChildren = c.sort !== false;
    let children = c.children || [];
    if (sortChildren) {
      children = [...children].sort((a, b) => String(a.name || '').localeCompare(String(b.name || '')));
    }
    const childHtml = exp ? '<div style="padding-left:12px;margin-top:4px">' + (children.map(x=>renderElement(x,tabs,parentTabId, c.children, name)).join('')) + '</div>' : '';
    return '<div style="border:1px solid var(--border-color,#555);margin-bottom:6px;border-radius:4px"><div style="padding:4px 8px;background:var(--comfy-input-bg,#333);cursor:pointer;display:flex;align-items:center;justify-content:space-between" data-action="toggle-expand" data-name="' + escapeHtml(name) + '"><b>' + escapeHtml(c.label||name) + '</b><span style="display:flex;align-items:center;gap:6px"><span style="font-size:0.8em;color:gray">' + (exp ? '▼' : '▶') + '</span><button data-action="clear-select" data-name="' + escapeHtml(name) + '" style="padding:0 6px;font-size:0.7em;background:transparent;color:var(--accent-color,#00bcd4);border:1px solid var(--accent-color,#00bcd4);border-radius:3px;cursor:pointer" title="Clear all selections in this list">Clear</button></span></div>' + childHtml + '</div>';
  }
  if (type === 'single') {
    const name = c.name || c.i || c.label;
    if (!name) return '';
    const act = !!state.activated[name];
    return '<label style="display:flex;align-items:center;gap:4px;margin:2px 0"><input type="checkbox" ' + (act?'checked':'') + ' data-action="toggle-activate" data-name="' + escapeHtml(name) + '"/>' + escapeHtml(name) + '</label>';
  }
  if (type === 'dual') {
    const name = c.name || c.i || c.label;
    if (!name) return '';
    const act = !!state.activated[name];
    const exp = state.expanded[name] === true;
    const posPrompt = state.values[name + '_pos'] ?? c.prompt_pos ?? '';
    const negPrompt = state.values[name + '_neg'] ?? c.prompt_neg ?? '';
    const posEmphasis = parseFloat(state.values[name + '_pos_emphasis'] ?? c.emphasis_pos ?? c.emphasis ?? 1);
    const negEmphasis = parseFloat(state.values[name + '_neg_emphasis'] ?? c.emphasis_neg ?? c.emphasis ?? 1);
    const prefix = c.prefix ? escapeHtml(c.prefix) + ' ' : '';
    const postfix = c.postfix ? ' ' + escapeHtml(c.postfix) : '';
    const childHtml = exp ? '<div style="padding-left:12px;margin-top:4px;padding-bottom:4px">' +
      '<div style="margin-bottom:6px">' +
        '<label style="font-size:0.8em;color:gray;margin-bottom:2px;display:block">Positive' + (prefix || postfix ? ' (with affixes)' : '') + '</label>' +
        '<div style="display:flex;gap:4px;align-items:center">' +
          '<input type="text" value="' + escapeHtml(posPrompt) + '" data-input="text" data-name="' + escapeHtml(name + '_pos') + '" style="flex:1;padding:4px;font-size:0.85em"/>' +
          '<input type="number" min="0" step="0.1" value="' + posEmphasis + '" data-input="range" data-name="' + escapeHtml(name + '_pos_emphasis') + '" style="width:70px;padding:4px;font-size:0.85em" title="Emphasis (0 to omit)"/>' +
        '</div>' +
      '</div>' +
      '<div style="margin-bottom:6px">' +
        '<label style="font-size:0.8em;color:gray;margin-bottom:2px;display:block">Negative' + (prefix || postfix ? ' (with affixes)' : '') + '</label>' +
        '<div style="display:flex;gap:4px;align-items:center">' +
          '<input type="text" value="' + escapeHtml(negPrompt) + '" data-input="text" data-name="' + escapeHtml(name + '_neg') + '" style="flex:1;padding:4px;font-size:0.85em"/>' +
          '<input type="number" min="0" step="0.1" value="' + negEmphasis + '" data-input="range" data-name="' + escapeHtml(name + '_neg_emphasis') + '" style="width:70px;padding:4px;font-size:0.85em" title="Emphasis (0 to omit)"/>' +
        '</div>' +
      '</div>' +
    '</div>' : '';
    const affixInfo = (prefix || postfix) ? ' <span style="font-size:0.7em;color:gray">[affix: ' + prefix.trim() + ' / ' + postfix.trim() + ']</span>' : '';
    return '<div style="border:1px solid var(--border-color,#555);margin-bottom:6px;border-radius:4px"><div style="padding:4px 8px;background:var(--comfy-input-bg,#333);cursor:pointer" data-action="toggle-expand" data-name="' + escapeHtml(name) + '"><input type="checkbox" ' + (act?'checked':'') + ' data-action="toggle-activate" data-name="' + escapeHtml(name) + '" style="margin-right:8px"/><b>' + escapeHtml(name) + '</b>' + affixInfo + ' <span style="font-size:0.8em;color:gray;margin-left:auto">' + (exp ? '▼' : '▶') + '</span></div>' + childHtml + '</div>';
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
    let html = '<div style="margin:4px 0">';
    html += '<label style="display:flex;align-items:center;gap:4px;margin-bottom:2px">';
    html += '<input type="checkbox" ' + (act?'checked':'') + ' data-action="toggle-activate" data-name="' + escapeHtml(name) + '"/>';
    html += '<span style="font-size:0.9em">' + escapeHtml(name);
    if (isLinked) {
      html += ' <span style="cursor:help;color:var(--accent-color,#00bcd4);margin-left:4px" title="' + tooltipText + '">ⓘ</span>';
    }
    html += '</span>';
    html += '</label>';
    html += '<div style="margin-left:20px">';
    if (!isLinked) {
      html += '<div style="font-size:0.8em;color:gray;margin-bottom:2px">' + rangeText + '</div>';
      html += '<input type="range" min="0" max="1" step="0.1" value="' + val + '" data-input="range" data-name="' + escapeHtml(name) + '"/>';
      html += '<span id="v-' + escapeHtml(name) + '" style="font-size:0.8em;margin-left:8px">' + val + '</span>';
    } else {
      html += '<span style="color:gray;font-size:0.8em" title="' + tooltipText + '"></span>';
    }
    html += '</div>';
    html += '</div>';
    return html;
  }
  if (type === 'from_list') {
    const name = c.name || c.i;
    if (!name) return '<div>from_list</div>';
    const listName = c.name || c.i;
    const postfix = c.postfix || '';
    // Use currentSelectName passed from parent select, fallback to derived name
    const parentSelectName = currentSelectName || (parentTabId || '').split('::').pop() || 'select';
    // Case-insensitive list lookup
    let listItems = [];
    const lists = window.__prompt_builder_lists || {};
    for (const [k, v] of Object.entries(lists)) {
      if (k.toLowerCase() === listName.toLowerCase()) {
        listItems = v;
        break;
      }
    }
    if (!listItems.length) {
      return '<div style="font-size:0.9em;color:gray">[from_list: ' + escapeHtml(name) + ' - list empty]</div>';
    }
    let html = '<div style="font-size:0.9em;margin-bottom:4px">';
    html += '<b>' + escapeHtml(name) + '</b>';
    html += '<span style="font-weight:normal;font-style:italic;color:var(--description-color,#aaa);margin-left:8px;font-size:0.85em">← ' + escapeHtml(listName) + ' list</span>';
    html += '</div>';
    for (const item of listItems) {
      // Include parent select name AND postfix to avoid collisions across selects AND postfixes
      const safePostfix = postfix ? '_' + postfix.replace(/\s+/g, '_') : '';
      const itemName = parentSelectName + '_' + listName + safePostfix + '_' + item;
      const act = !!state.activated[itemName];
      html += '<label style="display:flex;align-items:center;gap:4px;margin:2px 0">';
      html += '<input type="checkbox" ' + (act?'checked':'') + ' data-action="toggle-activate" data-name="' + escapeHtml(itemName) + '"/>';
      html += escapeHtml(item + (postfix ? ' ' + postfix : ''));
      html += '</label>';
    }
    return html;
  }
  if (type === 'preset') {
    const name = c.name || c.label || c.i;
    if (!name) return '';
    return '<button style="padding:4px 6px;background:var(--comfy-input-bg,#333);color:var(--fg-color,#eee);border:1px solid var(--border-color,#555);margin:2px" data-action="preset" data-name="' + escapeHtml(name) + '">' + escapeHtml(name) + '</button>';
  }
  return '<div style="font-size:0.8em;color:gray;margin:2px 0">' + escapeHtml(type) + '</div>';
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