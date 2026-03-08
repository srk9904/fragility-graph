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
                selector: 'core',
                style: { 'active-bg-opacity': 0 }
            },
            {
                selector: ':active',
                style: { 'overlay-opacity': 0 }
            },
            {
                selector: 'node',
                style: {
                    'shape': 'ellipse',
                    'background-opacity': 0.9,
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'background-color': 'data(color)',
                    'label': 'data(label)',
                    'color': '#cbd5e1',
                    'font-size': '11px',
                    'text-valign': 'bottom',
                    'text-halign': 'center',
                    'text-margin-y': '6px',
                    'text-background-color': '#0a0e17',
                    'text-background-opacity': 1,
                    'text-background-padding': '3px',
                    'text-background-shape': 'rectangle',
                    'z-index': 20,
                    'border-width': 2,
                    'border-color': 'data(borderColor)',
                    'text-outline-width': 2,
                    'text-outline-color': '#0a0e17',
                },
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': 'rgba(56,189,248,0.15)',
                    'target-arrow-color': 'rgba(56,189,248,0.25)',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'arrow-scale': 0.8,
                    'label': 'data(fragLabel)',
                    'font-size': '9px',
                    'color': '#ffffff',
                    'font-weight': '700',
                    'text-background-color': '#0a0e17',
                    'text-background-opacity': 1,
                    'text-background-padding': '3px',
                    'text-background-shape': 'rectangle',
                    'text-outline-width': 0,
                    'text-margin-y': -10,
                    'edge-text-rotation': 'autorotate',
                },
            },
            {
                selector: '.dimmed',
                style: {
                    'opacity': 0.15,
                    'text-opacity': 0,
                    'z-index': 1
                }
            },
            {
                selector: 'edge.highlighted',
                style: {
                    'opacity': 1,
                    'text-opacity': 1,
                    'z-index': 100,
                    'line-color': '#38bdf8',
                    'target-arrow-color': '#38bdf8',
                    'width': 3
                }
            },
            {
                selector: 'node.highlighted',
                style: {
                    'opacity': 1,
                    'text-opacity': 1,
                    'z-index': 100,
                }
            },
            {
                selector: 'node.hovered',
                style: {
                    'border-width': 4,
                    'border-color': '#ffffff',
                    'z-index': 999,
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'border-width': 3,
                    'border-color': '#38bdf8',
                    'overlay-opacity': 0,
                },
            },
            {
                selector: 'node.impacted',
                style: {
                    'border-color': '#f87171',
                    'background-color': '#7f1d1d',
                    'border-width': 4,
                    'overlay-color': '#f87171',
                    'overlay-opacity': 0.4,
                    'z-index': 20
                }
            },
            {
                selector: 'node.request-node',
                style: {
                    'shape': 'pentagon',
                    'background-color': '#a78bfa',
                    'border-color': '#fff',
                    'border-width': 3,
                    'text-valign': 'center',
                    'font-size': '12px',
                    'font-weight': 'bold',
                    'color': '#fff',
                    'text-outline-color': '#0a0e17',
                    'text-outline-width': 3,
                    'z-index': 100
                }
            },
            {
                selector: 'edge.impact-edge',
                style: {
                    'line-color': '#a78bfa',
                    'width': 3,
                    'line-style': 'dashed',
                    'target-arrow-color': '#a78bfa'
                }
            }
        ],
        layout: { name: 'cose', animate: false, nodeRepulsion: () => 100000, idealEdgeLength: () => 200, nodeOverlap: 30, padding: 80, gravity: 0.08, numIter: 1200, coolingFactor: 0.95 },

        minZoom: 0.3, maxZoom: 3,
    });

    /* NEW */
    let nodeDragged = false;
    cy.on('grab', 'node', () => { nodeDragged = false; });
    cy.on('drag', 'node', () => { nodeDragged = true; });
    cy.on('tap', 'node', (e) => {
        if (nodeDragged) return;
        selectedNode = e.target.data();
        showNodeDetails(selectedNode);
    });

    cy.on('tap', (e) => {
        if (e.target === cy) {
            selectedNode = null;
            showFileSummary();
        }
    });

    // --- Limelight & Hover Interaction ---
    cy.on('mouseover', 'node', (e) => {
        const node = e.target;
        const root = cy.nodes('.root');

        // Slight zoom on the hovered node
        node.style('width', node.data('size') * 1.15);
        node.style('height', node.data('size') * 1.15);

        // Dim everything else
        cy.elements().addClass('dimmed');

        // Highlight this node + its path to root
        node.removeClass('dimmed').addClass('highlighted').addClass('hovered');
        node.predecessors().removeClass('dimmed').addClass('highlighted');
        node.connectedEdges().removeClass('dimmed').addClass('highlighted');

        if (root.length) {
            root.removeClass('dimmed').addClass('highlighted');
        }
    });

    cy.on('mouseout', 'node', (e) => {
        const node = e.target;
        // Revert zoom
        node.style('width', node.data('size'));
        node.style('height', node.data('size'));
        cy.elements().removeClass('dimmed').removeClass('highlighted').removeClass('hovered');
    });

    // --- Edge tooltip ---
    const edgeTooltip = document.getElementById('edgeTooltip');
    const tooltipRel = document.getElementById('tooltipRel');
    const tooltipReason = document.getElementById('tooltipReason');

    cy.on('mouseover', 'edge', (e) => {
        const edge = e.target;
        const rel = edge.data('relationship') || 'dependency';
        const reason = edge.data('reason') ||
            `${edge.source().data('label')} calls into ${edge.target().data('label')}`;

        tooltipRel.textContent = rel;
        tooltipReason.textContent = reason;
        edgeTooltip.style.display = 'block';
        edge.addClass('highlighted').removeClass('dimmed');
    });

    cy.on('mousemove', 'edge', (e) => {
        edgeTooltip.style.left = (e.originalEvent.clientX + 14) + 'px';
        edgeTooltip.style.top = (e.originalEvent.clientY - 10) + 'px';
    });

    cy.on('mouseout', 'edge', (e) => {
        edgeTooltip.style.display = 'none';
        e.target.removeClass('highlighted');
    });
}


function renderGraph(data) {
    graphData = data;
    cy.elements().remove();

    if (!data.nodes || !data.nodes.length) {
        graphEmpty.style.display = '';
        return;
    }

    const edgesRaw = data.edges || [];
    const elements = [];

    // 1. Calculate weighted edges (sum of outgoing = 100%)
    const outgoingEdges = {};
    edgesRaw.forEach(e => {
        if (!outgoingEdges[e.source]) outgoingEdges[e.source] = [];
        outgoingEdges[e.source].push(e);
    });

    const normalizedEdges = [];
    Object.keys(outgoingEdges).forEach(source => {
        const sourceEdges = outgoingEdges[source];
        const count = sourceEdges.length;
        sourceEdges.forEach(e => {
            normalizedEdges.push({
                ...e,
                weight: 100 / count
            });
        });
    });

    // 2. Strict Connected Component Filtering: only show nodes reachable from (or reaching) the root
    // Root node is the file under focus. We match by file_path and ensure it's a file node.
    const searchPath = focusedFile.toLowerCase().replace(/\\/g, '/');
    const rootNodeData = data.nodes.find(n => (n.file_path || "").toLowerCase().replace(/\\/g, '/') === searchPath && n.type !== 'class');
    const connectedIds = new Set();

    if (rootNodeData) {
        connectedIds.add(rootNodeData.id);

        // Build adjacency list for reachability
        const adj = {};
        normalizedEdges.forEach(e => {
            if (!adj[e.source]) adj[e.source] = [];
            if (!adj[e.target]) adj[e.target] = [];
            adj[e.source].push(e.target);
            adj[e.target].push(e.source); // Treat as undirected for "component" membership
        });

        // BFS to find all nodes in the same component as root
        const queue = [rootNodeData.id];
        while (queue.length > 0) {
            const curr = queue.shift();
            (adj[curr] || []).forEach(neighbor => {
                if (!connectedIds.has(neighbor)) {
                    connectedIds.add(neighbor);
                    queue.push(neighbor);
                }
            });
        }
    } else {
        // No root found — show empty graph state
        graphEmpty.style.display = '';
        $('#emptyStateMsg').textContent = 'No Root Node Found';
        $('#emptyStateSubMsg').textContent = 'Could not locate the focused file in the graph data.';
        return;
    }

    // Also include same-file nodes that have at least one edge connection
    // (prevents hiding functions that aren't BFS-reachable from root but DO have edges)
    const edgeNodeIds = new Set();
    normalizedEdges.forEach(e => {
        edgeNodeIds.add(e.source);
        edgeNodeIds.add(e.target);
    });
    data.nodes.forEach(n => {
        const nodePath = (n.file_path || "").toLowerCase().replace(/\\/g, '/');
        if (nodePath === searchPath && edgeNodeIds.has(n.id)) {
            connectedIds.add(n.id);
        }
    });

    // 3. Filter nodes — only connected nodes (with at least one edge or is root)
    const filteredNodes = data.nodes.filter(n => connectedIds.has(n.id));

    if (filteredNodes.length === 0) {
        graphEmpty.style.display = '';
        $('#emptyStateMsg').textContent = 'No Dependents Found';
        $('#emptyStateSubMsg').textContent = 'This file appeared isolated in the trace.';
        return;
    }
    graphEmpty.style.display = 'none';

    // 4. Build elements
    // NEW
    filteredNodes.forEach((n) => {
        const score = Math.min(100, Math.max(0,
            (typeof n.fragility === 'number' && isFinite(n.fragility)) ? n.fragility : 0
        ));

        const isRoot = (n.id.toLowerCase() === searchPath);

        const color = isRoot ? '#38bdf8' : '#f97316';
        const borderColor = isRoot ? 'rgba(56,189,248,0.6)' : 'rgba(249,115,22,0.5)';
        const size = isRoot ? 28 : Math.max(14, Math.min(24, 14 + score * 0.12));

        const {
            width: _w, height: _h, size: _s,
            shape: _shape, color: _c,
            background_color: _bc,
            ...safeNode
        } = n;

        elements.push({
            data: {
                ...safeNode,
                id: n.id,
                label: n.label || n.id.split('/').pop(),
                color,
                borderColor,
                size,
            },
            classes: isRoot ? 'root' : '',
        });
    });

    normalizedEdges.forEach((e) => {
        if (connectedIds.has(e.source) && connectedIds.has(e.target)) {
            const weightValue = e.weight || 0;
            elements.push({
                data: {
                    source: e.source,
                    target: e.target,
                    relationship: e.relationship || '',
                    reason: e.reason || e.description || '',
                    fragLabel: weightValue > 0 ? weightValue.toFixed(0) + '%' : '',
                    fragColor: '#38bdf8'
                }
            });
        }
    });

    cy.add(elements);
    cy.layout({
        name: 'cose',
        animate: false,
        /* Adjusted repulsion to 100000 as requested */
        nodeRepulsion: () => 100000,
        idealEdgeLength: () => 100,
        nodeOverlap: 25,
        padding: 50,
        gravity: 0.1,
        numIter: 1000
    }).run();
    cy.fit(cy.elements(), 50);
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

    const icon = node.type === 'directory' ? '<i class="fa-solid fa-folder"></i>' : fileIcon(node.name);
    el.innerHTML = `<span class="icon">${icon}</span><span>${node.name}</span>`;

    if (node.type === 'file') {
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
    const n = name.toLowerCase();
    if (n.endsWith('.py')) return '<i class="fab fa-python"   style="color:#3776ab"></i>';
    if (n.endsWith('.js') || n.endsWith('.mjs')) return '<i class="fab fa-js"        style="color:#f7df1e"></i>';
    if (n.endsWith('.ts') || n.endsWith('.tsx')) return '<i class="fa-solid fa-code" style="color:#3178c6"></i>';
    if (n.endsWith('.html') || n.endsWith('.htm')) return '<i class="fab fa-html5"     style="color:#e34f26"></i>';
    if (n.endsWith('.css') || n.endsWith('.scss')) return '<i class="fab fa-css3-alt"  style="color:#1572b6"></i>';
    if (n.endsWith('.json')) return '<i class="fa-solid fa-brackets-curly" style="color:#22d3ee"></i>';
    if (n.endsWith('.md')) return '<i class="fa-solid fa-file-lines"      style="color:#94a3b8"></i>';
    if (n.endsWith('.yml') || n.endsWith('.yaml')) return '<i class="fa-solid fa-sliders"          style="color:#f59e0b"></i>';
    if (n.endsWith('.sh') || n.endsWith('.bash')) return '<i class="fa-solid fa-terminal"          style="color:#34d399"></i>';
    if (n.endsWith('.txt')) return '<i class="fa-solid fa-file-lines"        style="color:#64748b"></i>';
    if (n.endsWith('.env') || n.startsWith('.')) return '<i class="fa-solid fa-gear"              style="color:#94a3b8"></i>';
    return '<i class="fa-solid fa-file" style="color:#64748b"></i>';
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

    const btn = $('#btnAnalyze');
    const originalText = btn.innerHTML;

    // Show loading overlay on graph
    const graphCard = $('#graphCard');
    let overlay = $('#graphLoadingOverlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'graphLoadingOverlay';
        overlay.innerHTML = `
            <div class="graph-loader">
                <svg viewBox="0 0 80 80">
                    <line x1="40" y1="5"  x2="40" y2="40"/>
                    <line x1="75" y1="40" x2="40" y2="40"/>
                    <line x1="40" y1="75" x2="40" y2="40"/>
                    <line x1="5"  y1="40" x2="40" y2="40"/>
                </svg>
                <div class="node"></div>
                <div class="node"></div>
                <div class="node"></div>
                <div class="node"></div>
                <div class="node"></div>
            </div>
            <p>Tracing dependencies...</p>
        `;
        graphCard.appendChild(overlay);
    } else {
        overlay.style.display = 'flex';
    }

    graphEmpty.style.display = 'none';

    btn.classList.add('btn-analysing');
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analysing...`;
    activeBadge.textContent = 'ANALYSING';
    activeBadge.classList.add('analysing');
    setStatus('analysing', 'Analysing...');

    try {
        const res = await fetch(`${API}/api/v1/analyze_focused?file_path=${encodeURIComponent(focusedFile)}`, { method: 'POST' });
        const data = await res.json();
        focusedData = data;

        // Small delay so the transition feels intentional
        await new Promise(r => setTimeout(r, 300));

        // Fade out overlay
        overlay.style.transition = 'opacity 0.3s ease';
        overlay.style.opacity = '0';
        setTimeout(() => { overlay.style.display = 'none'; overlay.style.opacity = '1'; }, 300);

        renderGraph({ nodes: data.nodes || [], edges: data.edges || [] });
        updateRiskProfile(data);
        showFileSummary();

        activeBadge.textContent = 'ACTIVE ANALYSIS';
        activeBadge.classList.remove('analysing');
        setStatus('connected', 'Analysis complete');
        btn.classList.remove('btn-analysing');
        btn.innerHTML = originalText;
    } catch (e) {
        console.error('Analyse error:', e);
        overlay.style.display = 'none';
        setStatus('error', 'Analysis failed');
        activeBadge.textContent = 'ERROR';
        btn.classList.remove('btn-analysing');
        btn.innerHTML = originalText;
    }
}


/* ================================================================
   RISK PROFILE
   ================================================================ */
function updateRiskProfile(data) {
    const nodes = data.nodes || [];
    const nodeCount = nodes.length;
    const edgeCount = (data.edges || []).length;

    // Recalculate max fragility from actual node data, ignore nulls
    const scores = nodes
        .map(n => n.fragility)
        .filter(f => typeof f === 'number' && isFinite(f) && f > 0);
    const maxFrag = scores.length > 0 ? Math.max(...scores) : (data.max_fragility || 0);

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
                <div class="ai-placeholder-icon"><i class="fa-solid fa-brain" style="font-size:32px;color:#475569"></i></div>
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
                <i class="fa-solid fa-code" style="margin-right:4px"></i>View Full Code
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
                    <i class="fa-solid fa-code" style="margin-right:4px"></i>View Full Code
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

        const requestId = 'change-request-node';
        cy.remove(`#${requestId}`);

        cy.add({
            group: 'nodes',
            data: { id: requestId, label: 'PROPOSED CHANGE', color: '#a78bfa', borderColor: '#fff', size: 50 },
            classes: 'request-node',
            position: { x: cy.width() / 2, y: cy.height() / 2 }
        });

        // Highlight impacted nodes in graph
        cy.nodes().removeClass('impacted');
        cy.nodes().forEach(n => {
            const nid = n.id().toLowerCase().replace(/\\/g, '/');
            let isImpacted = false;
            impactedIds.forEach(targetId => {
                const normalizedAId = targetId.toLowerCase().replace(/\\/g, '/');
                if (nid === normalizedAId || nid.endsWith(normalizedAId.split('/').pop())) {
                    isImpacted = true;
                }
            });

            if (isImpacted) {
                n.addClass('impacted');
                cy.add({ group: 'edges', data: { source: requestId, target: n.id() }, classes: 'impact-edge' });
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
                    <i class="fa-solid fa-code" style="margin-right:4px"></i>View Full Code
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
        const res = await fetch(`${API}/api/v1/line_risks?file_path=${encodeURIComponent(filePath)}&focus_path=${encodeURIComponent(focusedFile)}`);
        const data = await res.json();

        const riskMap = {};
        (data.lines || []).forEach(lr => { riskMap[lr.line_number] = lr; });

        // Count risks by severity
        let criticalCount = 0, elevatedCount = 0, minorCount = 0;
        (data.lines || []).forEach(lr => {
            if (lr.risk_score >= 70) criticalCount++;
            else if (lr.risk_score >= 40) elevatedCount++;
            else minorCount++;
        });

        const lines = (data.content || '').split('\n');
        const lineClasses = new Array(lines.length + 1).fill('');
        const lineBadges = new Array(lines.length + 1).fill('');
        const lineTooltips = new Array(lines.length + 1).fill('');

        // Pre-process risks into per-line maps (handling multi-line blocks)
        (data.lines || []).forEach(lr => {
            const start = lr.line_number;
            const end = lr.line_end || lr.line_number;
            for (let n = start; n <= end; n++) {
                if (n > lines.length) continue;

                let cls = '', severityLabel = '';
                if (lr.risk_score >= 70) {
                    cls = 'risk-high';
                    severityLabel = 'CRITICAL';
                } else if (lr.risk_score >= 40) {
                    cls = 'risk-medium';
                    severityLabel = 'ELEVATED';
                } else {
                    cls = 'risk-low';
                    severityLabel = 'MINOR';
                }

                // If this line already has a higher risk highlight, skip
                const existingCls = lineClasses[n];
                const existingPriority = existingCls === 'risk-high' ? 3 : (existingCls === 'risk-medium' ? 2 : (existingCls === 'risk-low' ? 1 : 0));
                const currentPriority = cls === 'risk-high' ? 3 : (cls === 'risk-medium' ? 2 : (cls === 'risk-low' ? 1 : 0));

                if (currentPriority > existingPriority) {
                    lineClasses[n] = cls;
                    // Only show badge/tooltip on the start line of the range
                    if (n === start) {
                        let badge = '';
                        if (cls === 'risk-high') badge = `<span class="line-risk-badge badge-red">⚠ ${lr.risk_score.toFixed(0)}</span>`;
                        else if (cls === 'risk-medium') badge = `<span class="line-risk-badge badge-yellow">● ${lr.risk_score.toFixed(0)}</span>`;
                        else badge = `<span class="line-risk-badge badge-green">○ ${lr.risk_score.toFixed(0)}</span>`;

                        lineBadges[n] = badge;
                        lineTooltips[n] = `<span class="code-line-tooltip">
                            <span class="tooltip-severity ${cls}">${severityLabel} — Score ${lr.risk_score.toFixed(0)}/100</span>
                            <span class="tooltip-text">${escapeHtml(lr.reason)}</span>
                            <span class="tooltip-line-ref">Line ${n}</span>
                        </span>`;
                    }
                }
            }
        });

        // Build legend bar
        let legendHtml = `<div class="risk-legend">
            <span class="risk-legend-title">Risk Legend</span>
            <div class="risk-legend-items">
                <span class="risk-legend-item legend-critical"><span class="legend-dot dot-critical"></span>Critical Risk <span class="legend-count">${criticalCount}</span></span>
                <span class="risk-legend-item legend-elevated"><span class="legend-dot dot-elevated"></span>Elevated Risk <span class="legend-count">${elevatedCount}</span></span>
                <span class="risk-legend-item legend-minor"><span class="legend-dot dot-minor"></span>Minor Concern <span class="legend-count">${minorCount}</span></span>
            </div>
        </div>`;

        let codeHtml = '';
        lines.forEach((line, i) => {
            const num = i + 1;
            const cls = lineClasses[num] || '';
            const badge = lineBadges[num] || '';
            const tooltip = lineTooltips[num] || '';

            codeHtml += `<div class="code-line ${cls}">` +
                `<span class="line-number">${num}</span>` +
                `<span class="line-content">${escapeHtml(line)}</span>` +
                badge + tooltip + `</div>`;
        });
        modalCode.innerHTML = legendHtml + codeHtml;
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
