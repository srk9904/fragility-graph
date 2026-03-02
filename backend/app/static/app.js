/**
 * Fragility Graph Enterprise Controller v5
 * Professional Engineering Insights
 */

let cy = null;
let currentMockCode = [];
let analysisMode = 'file';

document.addEventListener('DOMContentLoaded', () => {
    initCytoscape();
    setupDynamicInputs();

    // Initial State
    document.getElementById('file-search').value = "services/audio_core.py";
    triggerSimulation();
});

function initCytoscape() {
    cy = cytoscape({
        container: document.getElementById('cy'),
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'color': '#fff',
                    'font-family': 'Space Grotesk',
                    'font-size': '11px',
                    'font-weight': '600',
                    'text-valign': 'bottom',
                    'text-margin-y': '10px', // Increased to avoid overlap
                    'background-color': '#333',
                    'width': 'mapData(score, 0, 1, 28, 55)', // Reduced default size for leaves
                    'height': 'mapData(score, 0, 1, 28, 55)',
                    'border-width': '2px',
                    'border-color': 'rgba(255,255,255,0.1)',
                    'transition-property': 'background-color, border-color, width, height',
                    'transition-duration': '0.3s',
                    'text-outline-color': '#08080a',
                    'text-outline-width': '2px',
                    'z-index': 10
                }
            },
            {
                selector: 'node[type="root"]',
                style: {
                    'background-color': '#00ff88', // NEON GREEN
                    'width': 'mapData(score, 0, 1, 40, 75)', // Boosted size for Parent
                    'height': 'mapData(score, 0, 1, 40, 75)',
                    'border-color': '#fff',
                    'border-width': '3px',
                    'box-shadow': '0 0 20px #00ff88'
                }
            },
            {
                selector: 'node[type="leaf"]',
                style: {
                    'background-color': 'data(color)',
                }
            },
            {
                selector: 'node.critical-node',
                style: {
                    'border-color': '#bd0000',
                    'border-width': '4px',
                    'text-outline-color': '#bd0000',
                    'text-outline-width': '3px'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 'mapData(val, 0, 1, 1.5, 6)',
                    'line-color': 'rgba(255,255,255,0.1)',
                    'target-arrow-color': 'rgba(255,255,255,0.1)',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'label': '',
                    'font-size': '11px',
                    'color': '#fff',
                    'font-weight': '800',
                    'text-background-opacity': 0.8,
                    'text-background-color': '#000',
                    'text-background-padding': '3px',
                    'text-background-shape': 'roundrectangle',
                    'text-margin-y': '-15px',
                    'edge-text-rotation': 'autorotate',
                    'target-distance-from-node': '8px'
                }
            },
            {
                selector: 'edge.highlighted',
                style: {
                    'line-color': 'data(hlColor)',
                    'target-arrow-color': 'data(hlColor)',
                    'label': 'data(weight)',
                    'opacity': 1,
                    'width': 5
                }
            }
        ],
        layout: { name: 'cose', padding: 100 },
        minZoom: 0.2,
        maxZoom: 2.0,
        wheelSensitivity: 0.05 // Dramatically reduced for smoother scroll
    });

    // Interaction Handlers
    cy.on('tap', 'node', (evt) => {
        openCodeModal(evt.target.data('label'), evt.target.data('score'));
    });

    cy.on('mouseover', 'node', (evt) => {
        evt.target.connectedEdges().addClass('highlighted');
    });

    cy.on('mouseout', 'node', (evt) => {
        evt.target.connectedEdges().removeClass('highlighted');
    });

    cy.on('mouseover', 'edge', (evt) => {
        evt.target.addClass('highlighted');
    });

    cy.on('mouseout', 'edge', (evt) => {
        evt.target.removeClass('highlighted');
    });
}

function setupDynamicInputs() {
    const fileIn = document.getElementById('file-search');
    const intentIn = document.getElementById('intent-search');
    const fileWrapper = document.getElementById('file-input-wrapper');
    const intentWrapper = document.getElementById('intent-input-wrapper');

    fileIn.onfocus = () => fileWrapper.classList.add('expanded');
    fileIn.onblur = () => fileWrapper.classList.remove('expanded');

    intentIn.onfocus = () => intentWrapper.classList.add('expanded');
    intentIn.onblur = () => intentWrapper.classList.remove('expanded');
}

function toggleAnalysisMode() {
    analysisMode = analysisMode === 'file' ? 'intent' : 'file';
    const toggle = document.getElementById('analysis-mode-toggle');
    const text = document.getElementById('mode-text');
    const fileWrapper = document.getElementById('file-input-wrapper');
    const intentWrapper = document.getElementById('intent-input-wrapper');

    if (analysisMode === 'intent') {
        toggle.classList.add('intent');
        text.innerText = 'CHANGE';
        fileWrapper.style.display = 'none';
        intentWrapper.style.display = 'flex';
    } else {
        toggle.classList.remove('intent');
        text.innerText = 'FILE';
        fileWrapper.style.display = 'flex';
        intentWrapper.style.display = 'none';
    }
}

function simulateFolderPicker() {
    const folder = prompt("Analyze Project Folder:", "C:/Users/Dev/NewProject");
    if (folder) {
        const selector = document.getElementById('project-selector');
        const option = document.createElement('option');
        const folderName = folder.split('/').pop() || folder.split('\\').pop();
        option.value = folderName;
        option.text = "Active: " + folderName;
        option.selected = true;
        selector.add(option, 0);

        // Update Brand badge
        document.querySelector('.badge-live').innerText = "Project: " + folderName;
        triggerSimulation();
    }
}

async function triggerSimulation() {
    const btn = document.getElementById('analyze-trigger');
    const fileIn = document.getElementById('file-search');
    const intentIn = document.getElementById('intent-search');
    const proj = document.getElementById('project-selector').value;

    const path = fileIn.value;
    const intent = intentIn.value;

    btn.classList.add('simulating');
    btn.innerHTML = 'ANALYZING...';

    try {
        const response = await fetch('/api/v1/analyze_mock', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: path,
                intent: intent,
                project: proj,
                mode: analysisMode
            })
        });

        const data = await response.json();
        const result = data[0];
        currentMockCode = result.mock_code;

        // UI Updates
        animateValue("main-score", 0, Math.round(result.score * 100), 800);
        document.getElementById('val-complexity').innerText = result.score > 0.7 ? "HIGH" : (result.score > 0.3 ? "MODERATE" : "STABLE");
        document.getElementById('val-impact').innerText = result.impact_count + " Nodes";
        document.getElementById('val-confidence').innerText = "0.98";

        // Score Glow
        const container = document.getElementById('score-container');
        container.className = 'score-circle';
        if (result.score > 0.7) container.classList.add('risk-glow-red');
        else if (result.score > 0.3) container.classList.add('risk-glow-orange');
        else container.classList.add('risk-glow-green');

        // Process Graph Elements
        const elements = result.graph_elements.map(el => {
            if (el.data.score !== undefined && el.data.type !== 'root') {
                const s = el.data.score;
                if (s > 0.85) {
                    el.data.color = '#bd0000'; // Dark Red
                    el.data.label = "⚠ CRITICAL\n" + el.data.label;
                    el.classes = 'critical-node';
                }
                else if (s > 0.6) el.data.color = '#ff3e3e'; // Light Red
                else if (s > 0.35) el.data.color = '#ffa94d'; // Orange
                else el.data.color = '#ffca28'; // Yellow (Defaulting to yellow as requested)
            }
            if (el.data.source) {
                // Determine edge highlight color
                const val = parseFloat(el.data.val);
                if (val > 0.7) el.data.hlColor = '#ff3e3e';
                else if (val > 0.4) el.data.hlColor = '#ffa94d';
                else el.data.hlColor = '#1ed2ff';
            }
            return el;
        });

        cy.elements().remove();
        cy.add(elements);
        cy.layout({ name: 'cose', animate: true, padding: 100 }).run();

        if (elements.length < 5) {
            cy.zoom(1.1);
            cy.center();
        }

        updateInsights(result);

    } catch (e) {
        console.error(e);
    } finally {
        btn.classList.remove('simulating');
        btn.innerHTML = '<span class="btn-content">ANALYZE</span>';
    }
}

function openCodeModal(filename, score) {
    const modal = document.getElementById('code-modal');
    document.getElementById('modal-filename').innerText = filename.replace("⚠ CRITICAL\n", "");

    let html = '';
    currentMockCode.forEach((item, index) => {
        const riskClass = item.risk !== 'none' ? `risk-${item.risk}` : '';
        html += `
            <div class="code-line ${riskClass}">
                <div class="line-num">${index + 1}</div>
                <div class="line-content">${escapeHtml(item.line)}</div>
            </div>
        `;
    });

    document.getElementById('modal-code').innerHTML = html;
    modal.classList.add('active');
}

function closeModal() { document.getElementById('code-modal').classList.remove('active'); }

function updateInsights(result) {
    const container = document.getElementById('insight-container');
    const color = result.score > 0.7 ? '#bd0000' : (result.score > 0.3 ? '#ffa94d' : '#00ff88');

    container.innerHTML = `
        <div class="insight-card" style="border-left-color: ${color}">
            <h3>
                <span>${result.score > 0.85 ? '🚨 CRITICAL FAILURE RISK' : 'STABILITY ANALYSIS'}</span>
                <span style="opacity:0.3; font-size:0.6rem">${result.name}</span>
            </h3>
            <div class="insight-body">
                <b>${result.name}</b> has ${result.impact_count} downstream dependencies. ${result.score > 0.7 ? 'This module is a structural bottleneck—modifying it risks a cascading system failure.' : 'Low fragility detected.'}
            </div>
            <div class="blast-radius-info">
                ${result.blast_radius.map(node => `<span class="affected-tag">${node}</span>`).join('')}
            </div>
        </div>
        <div style="font-size: 0.75rem; color: #1ed2ff; padding: 0.5rem 1rem; font-weight: 700;">
            PRO TIP: HOVER OVER NODES TO VIEW BLAST WEIGHTS
        </div>
    `;
}

function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) window.requestAnimationFrame(step);
    };
    window.requestAnimationFrame(step);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
