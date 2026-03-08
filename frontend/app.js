/* ================================================================
   FragilityGraph — Frontend Application
   Full interaction model:
     - File explorer (kept intact)
     - Folder picker
     - File-focused analysis with blue root node
     - Change impact analysis
     - AI explanations on node click
     - Risk profile gauge
     - Auto-analysis on startup
   ================================================================ */

const API = '';

// ── State ─────────────────────────────────────────────────
let cy = null;
let ws = null;
let focusedFile = '';
let focusedData = null;       // last analyze_focused response
let graphData = { nodes: [], edges: [] };
let selectedNode = null;
let allTreeItems = [];        // flat list for highlighting

// ── DOM refs ──────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const statusPill = $('#statusPill');
const statusLabel = statusPill.querySelector('.status-label');
const activeBadge = $('#activeBadge');
const filePathInput = $('#filePathInput');
const fileTreeEl = $('#fileTree');
const graphEmpty = $('#graphEmpty');
const aiContent = $('#aiContent');
const gaugeValue = $('#gaugeValue');
const gaugeFill = $('#gaugeFill');
const codeModal = $('#codeModal');
const modalTitle = $('#modalTitle');
const modalCode = $('#modalCode');
const modalExpl = $('#modalExplanation');
const modalClose = $('#modalClose');
const folderModal = $('#folderModal');

// ── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initCytoscape();
    loadFileTree().then(() => autoSelectFirstFile());
    connectWebSocket();

    $('#btnOpenFolder').addEventListener('click', showFolderModal);
    $('#btnAnalyze').addEventListener('click', () => analyzeCurrentFile());
    $('#btnImpact').addEventListener('click', runImpactAnalysis);
    $('#changeInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') runImpactAnalysis(); });

    // Robust modal closing
    modalClose.addEventListener('click', closeCodeModal);
    codeModal.addEventListener('click', (e) => { if (e.target === codeModal) closeCodeModal(); });
    folderModal.addEventListener('click', (e) => { if (e.target === folderModal) closeFolderModal(); });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeCodeModal();
            closeFolderModal();
        }
    });
});


/* ================================================================
   CYTOSCAPE
   ================================================================ */
function initCytoscape() {
    cy = cytoscape({
        container: $('#cyContainer'),
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'background-color': 'data(color)',
                    'color': '#e2e8f0',
                    'text-valign': 'bottom',
                    'text-halign': 'center',
                    'font-size': '10px',
                    'font-family': "'Inter', sans-serif",
                    'text-margin-y': 7,
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'border-width': 2,
                    'border-color': 'data(borderColor)',
                    'text-outline-width': 2,
                    'text-outline-color': '#0a0e17',
                    'transition-property': 'background-color, width, height, border-color',
                    'transition-duration': '0.3s',
                },
            },
            {
                selector: 'node.root',
                style: {
                    'background-color': '#38bdf8',
                    'border-color': 'rgba(56,189,248,0.6)',
                    'border-width': 3,
                    'font-weight': 'bold',
                    'text-outline-color': '#0a0e17',
                    'z-index': 10,
                },
            },
            {
                selector: 'node.impacted',
                style: {
                    'border-color': '#f87171',
                    'border-width': 4,
                    'border-style': 'double',
                },
            },
            {
                selector: 'edge',
                style: {
                    'width': 1.5,
                    'line-color': 'rgba(56,189,248,0.2)',
                    'target-arrow-color': 'rgba(56,189,248,0.35)',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'arrow-scale': 0.8,
                },
            },
            {
                selector: 'node:selected',
                style: {
                    'border-width': 3,
                    'border-color': '#22d3ee',
                    'overlay-padding': 5,
                    'overlay-color': 'rgba(34,211,238,0.12)',
                    'overlay-opacity': 1,
                },
            },
        ],
        layout: { name: 'cose', animate: true, animationDuration: 500, nodeRepulsion: () => 6000, idealEdgeLength: () => 100, padding: 30 },
        minZoom: 0.3, maxZoom: 3,
    });

    cy.on('tap', 'node', (e) => {
        selectedNode = e.target.data();
        showNodeDetails(selectedNode);
    });

    cy.on('tap', (e) => {
        if (e.target === cy) {
            selectedNode = null;
            showFileSummary();
        }
    });

    // Hover scores for edges
    cy.on('mouseover', 'edge', (e) => {
        const edge = e.target;
        const targetFrag = edge.target().data('fragility') || 0;
        if (targetFrag > 0) {
            edge.style('label', targetFrag.toFixed(0) + '%');
            edge.style('font-size', '8px');
            edge.style('color', fragilityColor(targetFrag));
            edge.style('text-outline-width', 1);
            edge.style('text-outline-color', '#0a0e17');
        }
    });
    cy.on('mouseout', 'edge', (e) => {
        e.target.style('label', '');
    });

    // Hover scores for nodes
    cy.on('mouseover', 'node', (e) => {
        const node = e.target;
        const frag = node.data('fragility') || 0;
        if (frag > 0) {
            node.style('text-outline-color', fragilityColor(frag));
            node.style('text-outline-width', 1);
        }
    });
    cy.on('mouseout', 'node', (e) => {
        const node = e.target;
        node.style('text-outline-color', '#0a0e17');
        node.style('text-outline-width', 2);
    });
}


function renderGraph(data) {
    graphData = data;
    cy.elements().remove();

    if (!data.nodes || !data.nodes.length) {
        graphEmpty.style.display = '';
        return;
    }

    // Filter out 0-fragility nodes unless they are the root file
    const filteredNodes = data.nodes.filter(n => (n.fragility && n.fragility > 0) || n.file_path === focusedFile);

    if (filteredNodes.length === 0) {
        graphEmpty.style.display = '';
        return;
    }
    graphEmpty.style.display = 'none';

    const elements = [];
    const filteredIds = new Set(filteredNodes.map(n => n.id));

    filteredNodes.forEach((n) => {
        const score = n.fragility || 0;
        const isRoot = n.file_path === focusedFile && n.type !== 'class';
        const color = isRoot ? '#38bdf8' : fragilityColor(score);
        const size = isRoot ? 40 : Math.max(20, Math.min(55, 22 + score * 0.35));

        elements.push({
            data: {
                id: n.id, label: n.label,
                color, borderColor: isRoot ? 'rgba(56,189,248,0.5)' : fragilityBorder(score),
                size, ...n,
            },
            classes: isRoot ? 'root' : '',
        });
    });

    (data.edges || []).forEach((e) => {
        if (filteredIds.has(e.source) && filteredIds.has(e.target)) {
            elements.push({ data: { source: e.source, target: e.target, relationship: e.relationship } });
        }
    });

    cy.add(elements);
    cy.layout({
        name: 'cose', animate: true, animationDuration: 500,
        nodeRepulsion: () => 6000, idealEdgeLength: () => 100, padding: 30,
    }).run();
}


/* ================================================================
   FILE TREE (kept intact per user request)
   ================================================================ */
async function loadFileTree(root) {
    try {
        const url = root
            ? `${API}/api/v1/file_tree?root=${encodeURIComponent(root)}`
            : `${API}/api/v1/file_tree`;
        const res = await fetch(url);
        const tree = await res.json();
        fileTreeEl.innerHTML = '';
        allTreeItems = [];
        renderTreeNode(tree, fileTreeEl, 0);
    } catch (e) {
        fileTreeEl.innerHTML = '<p class="subtle" style="padding:12px">Could not load file tree.</p>';
    }
}

function renderTreeNode(node, parent, depth) {
    const el = document.createElement('div');
    el.className = 'tree-item ' + (node.type === 'directory' ? 'tree-dir' : 'tree-file');
    el.style.setProperty('--indent', (10 + depth * 14) + 'px');

    const icon = node.type === 'directory' ? '📁' : fileIcon(node.name);
    el.innerHTML = `<span class="icon">${icon}</span><span>${node.name}</span>`;

    if (node.type === 'file' && node.name.endsWith('.py')) {
        el.dataset.path = node.path;
        allTreeItems.push(el);
        el.addEventListener('click', () => selectFile(node.path));
    }

    parent.appendChild(el);

    if (node.children) {
        const sorted = [...node.children].sort((a, b) => {
            if (a.type === b.type) return a.name.localeCompare(b.name);
            return a.type === 'directory' ? -1 : 1;
        });
        sorted.forEach((child) => renderTreeNode(child, parent, depth + 1));
    }
}

function fileIcon(name) {
    if (name.endsWith('.py')) return '🐍';
    if (name.endsWith('.js')) return '📜';
    if (name.endsWith('.html')) return '🌐';
    if (name.endsWith('.css')) return '🎨';
    if (name.endsWith('.json')) return '📋';
    if (name.endsWith('.md')) return '📝';
    return '📄';
}


/* ================================================================
   FILE SELECTION & ANALYSIS
   ================================================================ */
function selectFile(filePath) {
    focusedFile = filePath;
    filePathInput.value = filePath.split('/').pop();

    // Highlight in tree
    allTreeItems.forEach(el => el.classList.remove('focused'));
    const match = allTreeItems.find(el => el.dataset.path === filePath);
    if (match) match.classList.add('focused');

    // Auto-analyse
    analyzeCurrentFile();
}


async function autoSelectFirstFile() {
    // Try to find a substantial file (e.g. main.py, routes.py, or any non-__init__)
    if (allTreeItems.length === 0) return;

    let target = allTreeItems.find(el => {
        const name = el.dataset.path.split('/').pop();
        return name === 'main.py' || name === 'routes.py';
    });

    if (!target) {
        // Fallback to any file that isn't __init__.py
        target = allTreeItems.find(el => !el.dataset.path.endsWith('__init__.py'));
    }

    if (!target) {
        // Absolute fallback
        target = allTreeItems[0];
    }

    if (target) {
        selectFile(target.dataset.path);
    }
}


async function analyzeCurrentFile() {
    if (!focusedFile) return;

    activeBadge.textContent = 'ANALYSING';
    activeBadge.classList.add('analysing');
    setStatus('analysing', 'Analysing...');

    try {
        const res = await fetch(`${API}/api/v1/analyze_focused?file_path=${encodeURIComponent(focusedFile)}`, { method: 'POST' });
        const data = await res.json();
        focusedData = data;

        // Update graph
        renderGraph({ nodes: data.nodes || [], edges: data.edges || [] });

        // Update risk profile
        updateRiskProfile(data);

        // Show file summary in AI panel
        showFileSummary();

        activeBadge.textContent = 'ACTIVE ANALYSIS';
        activeBadge.classList.remove('analysing');
        setStatus('connected', 'Analysis complete');
    } catch (e) {
        console.error('Analyse error:', e);
        setStatus('error', 'Analysis failed');
        activeBadge.textContent = 'ERROR';
    }
}


/* ================================================================
   RISK PROFILE
   ================================================================ */
function updateRiskProfile(data) {
    const maxFrag = data.max_fragility || 0;
    const nodeCount = (data.nodes || []).length;
    const edgeCount = (data.edges || []).length;

    // Gauge
    gaugeValue.textContent = Math.round(maxFrag);
    const pct = maxFrag / 100;
    const offset = 314 * (1 - pct);
    gaugeFill.style.strokeDashoffset = offset;
    gaugeFill.style.stroke = fragilityColor(maxFrag);

    // Metrics
    const structuralRisk = nodeCount > 0 ? Math.min(100, Math.round((edgeCount / nodeCount) * 50)) : 0;
    const blastRadius = edgeCount > 0 ? Math.min(nodeCount, edgeCount + 1) : 0;
    const confidence = nodeCount > 0 ? Math.min(100, 60 + nodeCount * 2) : 0;

    $('#metricStructural').textContent = structuralRisk + '%';
    $('#metricBlast').textContent = blastRadius + ' nodes';
    $('#metricConfidence').textContent = confidence + '%';

    // Color the metric values
    $('#metricStructural').style.color = fragilityColor(structuralRisk);
    $('#metricBlast').style.color = blastRadius > 5 ? '#f87171' : blastRadius > 2 ? '#fbbf24' : '#34d399';
}


/* ================================================================
   AI PANEL
   ================================================================ */
function showFileSummary() {
    if (!focusedData) {
        aiContent.innerHTML = `
            <div class="ai-placeholder">
                <div class="ai-placeholder-icon">🤖</div>
                <p>No active trace.</p>
                <p class="subtle">Select a file or click a node to get AI-powered insights.</p>
            </div>`;
        return;
    }

    const fileName = focusedFile.split('/').pop();
    const summary = focusedData.summary || 'Generating summary...';
    const nodeCount = (focusedData.nodes || []).length;
    const riskCount = (focusedData.line_risks || []).length;

    aiContent.innerHTML = `
        <div class="ai-section">
            <div class="ai-section-title">File Summary</div>
            <div class="ai-summary-text">${escapeHtml(summary)}</div>
        </div>

        <div class="ai-section">
            <div class="ai-section-title">Statistics</div>
            <div class="metric-card" style="margin-bottom:4px">
                <span class="metric-name">Functions / Classes</span>
                <span class="metric-value">${nodeCount}</span>
            </div>
            <div class="metric-card">
                <span class="metric-name">Risky Lines</span>
                <span class="metric-value" style="color:${riskCount > 0 ? '#f87171' : '#34d399'}">${riskCount}</span>
            </div>
        </div>

        <div class="ai-section">
            <button class="btn btn-primary btn-view-code" onclick="openCodeModal('${focusedFile}')">
                📄 View Full Code
            </button>
        </div>

        <div class="ai-section">
            <p class="subtle" style="text-align:center;font-size:11px">Click a graph node for detailed AI explanation</p>
        </div>
    `;
}


async function showNodeDetails(data) {
    const score = data.fragility || 0;
    const risk = score >= 70 ? 'High' : score >= 40 ? 'Medium' : 'Low';
    const badgeClass = score >= 70 ? 'high' : score >= 40 ? 'medium' : 'low';

    // Show loading state immediately
    aiContent.innerHTML = `
        <div class="ai-section">
            <div class="ai-node-header">
                <span class="ai-node-name">${escapeHtml(data.label)}</span>
                <span class="ai-node-badge ${badgeClass}">${risk}</span>
            </div>
            <div class="metric-card">
                <span class="metric-name">Fragility Score</span>
                <span class="metric-value" style="color:${fragilityColor(score)}">${score.toFixed(0)}/100</span>
            </div>
        </div>

        <div class="ai-section">
            <div class="ai-section-title">AI Explanation</div>
            <div class="ai-loading">
                <div class="spinner"></div>
                <p>Generating AI insight...</p>
            </div>
        </div>
    `;

    // Fetch AI explanation
    try {
        const res = await fetch(`${API}/api/v1/explain_node`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: data.id,
                label: data.label,
                file_path: data.file_path || focusedFile,
                fragility: score,
            }),
        });
        const result = await res.json();

        const deps = result.dependencies || [];

        aiContent.innerHTML = `
            <div class="ai-section">
                <div class="ai-node-header">
                    <span class="ai-node-name">${escapeHtml(data.label)}</span>
                    <span class="ai-node-badge ${badgeClass}">${risk}</span>
                </div>
                <div class="metric-card" style="margin-bottom:4px">
                    <span class="metric-name">Fragility Score</span>
                    <span class="metric-value" style="color:${fragilityColor(score)}">${score.toFixed(0)}/100</span>
                </div>
                <div class="metric-card">
                    <span class="metric-name">Type</span>
                    <span class="metric-value">${data.type || 'function'}</span>
                </div>
            </div>

            <div class="ai-section">
                <div class="ai-section-title">AI Explanation</div>
                <div class="ai-explanation-text">${escapeHtml(result.explanation || 'No explanation available.')}</div>
            </div>

            ${deps.length ? `
            <div class="ai-section">
                <div class="ai-section-title">Dependencies (${deps.length})</div>
                <ul class="ai-deps-list">
                    ${deps.map(d => `<li>${escapeHtml(d)}</li>`).join('')}
                </ul>
            </div>` : ''}

            <div class="ai-section">
                <button class="btn btn-primary btn-view-code" onclick="openCodeModal('${data.file_path || focusedFile}')">
                    📄 View Full Code
                </button>
            </div>
        `;
    } catch (e) {
        console.error('Explain error:', e);
    }
}


/* ================================================================
   IMPACT ANALYSIS
   ================================================================ */
async function runImpactAnalysis() {
    const change = $('#changeInput').value.trim();
    if (!change || !focusedFile) return;

    $('#btnImpact').textContent = 'Analysing...';
    $('#btnImpact').disabled = true;

    try {
        const res = await fetch(`${API}/api/v1/impact_analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: focusedFile, change_description: change }),
        });
        const data = await res.json();
        const impactedIds = new Set(data.affected_node_ids || []);

        // Highlight impacted nodes in graph
        cy.nodes().removeClass('impacted');
        cy.nodes().forEach(n => {
            if (impactedIds.has(n.id())) {
                n.addClass('impacted');
            }
        });

        // Show impact in AI panel
        const affected = data.affected_functions || [];
        aiContent.innerHTML = `
            <div class="ai-section">
                <div class="ai-section-title">Impact Analysis</div>
                <div class="ai-explanation-text">
                    <strong>Change:</strong> "${escapeHtml(change)}"<br><br>
                    <strong>${affected.length}</strong> of ${data.total_functions || 0} functions affected.
                </div>
            </div>

            ${affected.length ? `
            <div class="ai-section">
                <div class="ai-section-title">Affected Functions</div>
                <ul class="ai-deps-list">
                    ${affected.map(f => `<li style="color:#f87171">${escapeHtml(f)}</li>`).join('')}
                </ul>
            </div>` : ''}

            <div class="ai-section">
                <button class="btn btn-primary btn-view-code" onclick="openCodeModal('${focusedFile}')">
                    📄 View Full Code
                </button>
            </div>
        `;
    } catch (e) {
        console.error('Impact error:', e);
    } finally {
        $('#btnImpact').textContent = 'Analyse Impact';
        $('#btnImpact').disabled = false;
    }
}


/* ================================================================
   FOLDER PICKER
   ================================================================ */
function showFolderModal() {
    folderModal.style.display = '';
    $('#folderPathInput').focus();
}
function closeFolderModal() {
    folderModal.style.display = 'none';
}
function openFolder() {
    const path = $('#folderPathInput').value.trim();
    if (!path) return;
    closeFolderModal();
    loadFileTree(path).then(() => autoSelectFirstFile());
}


/* ================================================================
   CODE MODAL
   ================================================================ */
async function openCodeModal(filePath) {
    if (!filePath) return;
    codeModal.style.display = '';
    modalTitle.textContent = filePath.split('/').pop();
    modalCode.innerHTML = '<div style="padding:20px;color:#475569">Loading...</div>';
    modalExpl.style.display = 'none';

    try {
        const res = await fetch(`${API}/api/v1/line_risks?file_path=${encodeURIComponent(filePath)}`);
        const data = await res.json();

        const riskMap = {};
        (data.lines || []).forEach(lr => { riskMap[lr.line_number] = lr; });

        const lines = (data.content || '').split('\n');
        let html = '';
        lines.forEach((line, i) => {
            const num = i + 1;
            const risk = riskMap[num];
            let cls = '', badge = '';
            if (risk) {
                if (risk.risk_score >= 70) { cls = 'risk-high'; badge = `<span class="line-risk-badge badge-red">${risk.risk_score.toFixed(0)}</span>`; }
                else if (risk.risk_score >= 40) { cls = 'risk-medium'; badge = `<span class="line-risk-badge badge-yellow">${risk.risk_score.toFixed(0)}</span>`; }
                else { cls = 'risk-low'; badge = `<span class="line-risk-badge badge-green">${risk.risk_score.toFixed(0)}</span>`; }
            }
            html += `<div class="code-line ${cls}" title="${risk ? escapeHtml(risk.reason) : ''}">` +
                `<span class="line-number">${num}</span>` +
                `<span class="line-content">${escapeHtml(line)}</span>` +
                badge + `</div>`;
        });
        modalCode.innerHTML = html;
    } catch (e) {
        modalCode.innerHTML = '<div style="padding:20px;color:#f87171">Failed to load file.</div>';
    }
}

function closeCodeModal() { codeModal.style.display = 'none'; }


/* ================================================================
   WEBSOCKET
   ================================================================ */
function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => setStatus('connected', 'Connected');
    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'fragility_update' && msg.file_path === focusedFile) {
                renderGraph({ nodes: msg.nodes || [], edges: msg.edges || [] });
            }
        } catch (e) { console.warn('WS parse error:', e); }
    };
    ws.onclose = () => { setStatus('error', 'Disconnected'); setTimeout(connectWebSocket, 3000); };
    ws.onerror = () => setStatus('error', 'Connection error');
}


/* ================================================================
   HELPERS
   ================================================================ */
function setStatus(state, text) {
    statusPill.className = 'status-pill ' + state;
    statusLabel.textContent = text;
}
function fragilityColor(s) {
    if (s > 90) return '#7f1d1d'; // Fatal (Dark Red)
    if (s > 30) return '#ef4444'; // High (Red)
    return '#eab308'; // Medium/Low (Yellow)
}
function fragilityBorder(s) {
    if (s > 90) return 'rgba(127,29,29,0.5)';
    if (s > 30) return 'rgba(239,68,68,0.4)';
    return 'rgba(234,179,8,0.3)';
}
function escapeHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
