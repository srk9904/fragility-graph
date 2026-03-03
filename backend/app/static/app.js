/**
 * FragilityGraph Enterprise Controller v6
 * World-Class Predictive Blast Radius Analysis
 */

let cy = null;
let currentMockCode = [];
let analysisMode = 'file';
let historyChart = null;
let riskHistory = [];

// ======================== INITIALIZATION ========================
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initCytoscape();
    initHistoryChart();
    setupDynamicInputs();

    // Initial demo analysis
    document.getElementById('file-search').value = "services/auth_provider.py";
    triggerSimulation();
});

// ======================== PARTICLE BACKGROUND ========================
function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let particles = [];
    const PARTICLE_COUNT = 60;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    resize();
    window.addEventListener('resize', resize);

    class Particle {
        constructor() {
            this.reset();
        }

        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 1.5 + 0.5;
            this.speedX = (Math.random() - 0.5) * 0.3;
            this.speedY = (Math.random() - 0.5) * 0.3;
            this.opacity = Math.random() * 0.3 + 0.05;
            this.color = Math.random() > 0.5 ? '30, 210, 255' : '157, 80, 187';
        }

        update() {
            this.x += this.speedX;
            this.y += this.speedY;

            if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
            if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${this.color}, ${this.opacity})`;
            ctx.fill();
        }
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push(new Particle());
    }

    function drawConnections() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    const opacity = (1 - distance / 150) * 0.06;
                    ctx.strokeStyle = `rgba(30, 210, 255, ${opacity})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(p => {
            p.update();
            p.draw();
        });

        drawConnections();
        requestAnimationFrame(animate);
    }

    animate();
}

// ======================== CYTOSCAPE GRAPH ========================
function initCytoscape() {
    cy = cytoscape({
        container: document.getElementById('cy'),
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'color': '#fff',
                    'font-family': 'Inter, sans-serif',
                    'font-size': '10px',
                    'font-weight': '600',
                    'text-valign': 'bottom',
                    'text-margin-y': '10px',
                    'background-color': '#333',
                    'width': 'mapData(score, 0, 1, 24, 50)',
                    'height': 'mapData(score, 0, 1, 24, 50)',
                    'border-width': '2px',
                    'border-color': 'rgba(255,255,255,0.08)',
                    'shadow-blur': 0,
                    'shadow-color': 'data(color)',
                    'shadow-opacity': 0.8,
                    'transition-property': 'background-color, border-color, width, height, shadow-blur, opacity',
                    'transition-duration': '0.25s',
                    'text-outline-color': '#06060a',
                    'text-outline-width': '2px',
                    'z-index': 10
                }
            },
            {
                selector: 'node[type="root"]',
                style: {
                    'background-color': '#00ff88',
                    'width': 'mapData(score, 0, 1, 38, 70)',
                    'height': 'mapData(score, 0, 1, 38, 70)',
                    'border-color': '#fff',
                    'border-width': '3px',
                    'shadow-blur': 15,
                    'shadow-color': '#00ff88',
                    'font-size': '11px'
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
                    'shadow-blur': 20,
                    'shadow-color': '#bd0000'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 'mapData(val, 0, 1, 1, 5)',
                    'line-color': 'rgba(255,255,255,0.06)',
                    'target-arrow-color': 'rgba(255,255,255,0.06)',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'label': '',
                    'font-size': '10px',
                    'color': '#fff',
                    'font-weight': '700',
                    'text-background-opacity': 0.9,
                    'text-background-color': '#06060a',
                    'text-background-padding': '3px',
                    'text-background-shape': 'roundrectangle',
                    'text-margin-y': '-12px',
                    'edge-text-rotation': 'autorotate',
                    'target-distance-from-node': '6px',
                    'transition-property': 'line-color, target-arrow-color, width, opacity',
                    'transition-duration': '0.3s'
                }
            },
            {
                selector: 'node.hovered',
                style: {
                    'shadow-blur': 30,
                    'border-width': '4px',
                    'border-color': '#fff',
                    'z-index': 1000
                }
            },
            {
                selector: '.dimmed',
                style: {
                    'opacity': 0.1,
                    'label': ''
                }
            },
            {
                selector: 'edge.highlighted',
                style: {
                    'line-color': 'data(hlColor)',
                    'target-arrow-color': 'data(hlColor)',
                    'label': 'data(weight)',
                    'opacity': 1,
                    'width': 5,
                    'z-index': 999
                }
            }
        ],
        layout: { name: 'cose', padding: 80 },
        minZoom: 0.2,
        maxZoom: 2.5,
        wheelSensitivity: 0.04
    });

    // Interaction Handlers
    cy.on('tap', 'node', (evt) => {
        openCodeModal(evt.target.data('label'), evt.target.data('score'));
    });

    cy.on('mouseover', 'node', (evt) => {
        const node = evt.target;
        const neighborhood = node.closedNeighborhood();

        cy.elements().addClass('dimmed');
        neighborhood.removeClass('dimmed');
        neighborhood.edges().addClass('highlighted');
        node.addClass('hovered');
    });

    cy.on('mouseout', 'node', (evt) => {
        cy.elements().removeClass('dimmed');
        cy.elements().removeClass('highlighted');
        evt.target.removeClass('hovered');
    });

    cy.on('mouseover', 'edge', (evt) => {
        evt.target.addClass('highlighted');
    });

    cy.on('mouseout', 'edge', (evt) => {
        evt.target.removeClass('highlighted');
    });
}

function relayout() {
    if (cy) {
        cy.layout({ name: 'cose', animate: true, padding: 80, nodeRepulsion: 8000 }).run();
    }
}

// ======================== HISTORY CHART ========================
function initHistoryChart() {
    const ctx = document.getElementById('history-chart');
    if (!ctx) return;

    historyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Risk Score',
                data: [],
                borderColor: '#1ed2ff',
                backgroundColor: 'rgba(30, 210, 255, 0.08)',
                fill: true,
                tension: 0.4,
                pointRadius: 3,
                pointBackgroundColor: '#1ed2ff',
                pointBorderColor: '#06060a',
                pointBorderWidth: 2,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    display: true,
                    min: 0,
                    max: 100,
                    ticks: {
                        color: 'rgba(255,255,255,0.15)',
                        font: { size: 9, family: 'JetBrains Mono' },
                        maxTicksLimit: 3
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.03)'
                    }
                }
            }
        }
    });
}

function addToHistory(score) {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    if (historyChart) {
        if (historyChart.data.labels.length >= 10) {
            historyChart.data.labels.shift();
            historyChart.data.datasets[0].data.shift();
        }
        historyChart.data.labels.push(time);
        historyChart.data.datasets[0].data.push(Math.round(score * 100));

        // Change color based on score
        const color = score > 0.7 ? '#ff3e3e' : (score > 0.3 ? '#ffca28' : '#00ff88');
        historyChart.data.datasets[0].borderColor = color;
        historyChart.data.datasets[0].pointBackgroundColor = color;
        historyChart.data.datasets[0].backgroundColor = color.replace(')', ', 0.08)').replace('rgb', 'rgba');

        historyChart.update('none');
    }
}

// ======================== INPUT HANDLERS ========================
function setupDynamicInputs() {
    const fileIn = document.getElementById('file-search');
    const intentIn = document.getElementById('intent-search');
    const fileWrapper = document.getElementById('file-input-wrapper');
    const intentWrapper = document.getElementById('intent-input-wrapper');

    if (fileIn) {
        fileIn.onfocus = () => fileWrapper.classList.add('expanded');
        fileIn.onblur = () => fileWrapper.classList.remove('expanded');

        // Enter key triggers analysis
        fileIn.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') triggerSimulation();
        });
    }

    if (intentIn) {
        intentIn.onfocus = () => intentWrapper.classList.add('expanded');
        intentIn.onblur = () => intentWrapper.classList.remove('expanded');
    }
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
    const folder = prompt("Analyze Project Folder:", "C:/Users/Dev/MyProject");
    if (folder) {
        const selector = document.getElementById('project-selector');
        const option = document.createElement('option');
        const folderName = folder.split('/').pop() || folder.split('\\').pop();
        option.value = folderName;
        option.text = folderName;
        option.selected = true;
        selector.add(option, 0);

        const badge = document.querySelector('.badge-live');
        if (badge) {
            badge.innerHTML = `<span class="pulse-dot"></span> ${folderName}`;
        }
        triggerSimulation();
    }
}

// ======================== MAIN ANALYSIS ========================
async function triggerSimulation() {
    const btn = document.getElementById('analyze-trigger');
    const fileIn = document.getElementById('file-search');
    const intentIn = document.getElementById('intent-search');
    const proj = document.getElementById('project-selector').value;

    const path = fileIn ? fileIn.value : '';
    const intent = intentIn ? intentIn.value : '';

    // Update UI state
    btn.classList.add('simulating');
    btn.innerHTML = '<span class="btn-content">ANALYZING...</span>';

    const statusBadge = document.getElementById('status-badge');
    if (statusBadge) {
        statusBadge.innerHTML = '<span class="pulse-dot"></span> ANALYZING';
        statusBadge.style.borderColor = 'rgba(30, 210, 255, 0.3)';
        statusBadge.style.color = '#1ed2ff';
    }

    const startTime = performance.now();

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

        const latency = Math.round(performance.now() - startTime);

        // Update Ticker
        document.getElementById('ticker-latency').innerText = latency + 'ms';
        document.getElementById('ticker-nodes').innerText = (result.graph_elements || []).filter(e => !e.data.source).length;
        document.getElementById('ticker-time').innerText = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        // Animate Score
        const scorePercent = Math.round(result.score * 100);
        animateValue("main-score", 0, scorePercent, 800);

        // Update Score Ring
        updateScoreRing(result.score);

        // Add to history
        addToHistory(result.score);

        // Metrics
        const riskLevel = result.score > 0.7 ? "CRITICAL" : (result.score > 0.3 ? "ELEVATED" : "STABLE");
        document.getElementById('val-complexity').innerText = riskLevel;
        document.getElementById('val-impact').innerText = result.impact_count + " nodes";
        document.getElementById('val-confidence').innerText = "0." + (85 + Math.floor(Math.random() * 14));
        document.getElementById('val-coupling').innerText = (result.blast_radius || []).length + " deps";

        // Score Container Class
        const container = document.getElementById('score-container');
        container.className = 'score-circle';
        if (result.score > 0.7) container.classList.add('risk-glow-red');
        else if (result.score > 0.3) container.classList.add('risk-glow-orange');
        else container.classList.add('risk-glow-green');

        // Process Graph Elements
        const elements = (result.graph_elements || []).map(el => {
            if (el.data.score !== undefined && el.data.type !== 'root') {
                const s = el.data.score;
                if (s > 0.85) {
                    el.data.color = '#bd0000';
                    el.data.label = "⚠ " + el.data.label;
                    el.classes = 'critical-node';
                }
                else if (s > 0.6) el.data.color = '#ff3e3e';
                else if (s > 0.35) el.data.color = '#ffa94d';
                else el.data.color = '#ffca28';
            }
            if (el.data.source) {
                const val = parseFloat(el.data.val);
                if (val > 0.7) el.data.hlColor = '#ff3e3e';
                else if (val > 0.4) el.data.hlColor = '#ffa94d';
                else el.data.hlColor = '#1ed2ff';
            }
            return el;
        });

        cy.elements().remove();
        cy.add(elements);
        cy.layout({ name: 'cose', animate: true, padding: 80, nodeRepulsion: 6000, idealEdgeLength: 100 }).run();

        if (elements.length < 5) {
            cy.zoom(1.1);
            cy.center();
        }

        // Update Insights
        updateInsights(result);

        // Update Dependencies
        updateDependencies(result);

        // Update status
        if (statusBadge) {
            statusBadge.innerHTML = `<span class="pulse-dot"></span> ${riskLevel}`;
            if (result.score > 0.7) {
                statusBadge.style.borderColor = 'rgba(255, 62, 62, 0.3)';
                statusBadge.style.color = '#ff3e3e';
            } else if (result.score > 0.3) {
                statusBadge.style.borderColor = 'rgba(255, 202, 40, 0.3)';
                statusBadge.style.color = '#ffca28';
            } else {
                statusBadge.style.borderColor = 'rgba(0, 255, 136, 0.3)';
                statusBadge.style.color = '#00ff88';
            }
        }

    } catch (e) {
        console.error('Analysis failed:', e);
        if (statusBadge) {
            statusBadge.innerHTML = '<span class="pulse-dot"></span> ERROR';
            statusBadge.style.color = '#ff3e3e';
        }
    } finally {
        btn.classList.remove('simulating');
        btn.innerHTML = '<span class="btn-content"><svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="margin-right: 6px;"><path d="M8 1v6M8 15v-6M1 8h6M15 8h-6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>ANALYZE</span>';
    }
}

// ======================== SCORE RING ========================
function updateScoreRing(score) {
    const ring = document.getElementById('score-ring-progress');
    if (!ring) return;

    const circumference = 2 * Math.PI * 54; // r=54
    const offset = circumference * (1 - score);
    ring.style.strokeDashoffset = offset;
}

// ======================== INSIGHTS ========================
function updateInsights(result) {
    const container = document.getElementById('insight-container');
    if (!container) return;

    const color = result.score > 0.7 ? '#ff3e3e' : (result.score > 0.3 ? '#ffa94d' : '#00ff88');
    const severity = result.score > 0.85 ? '🚨 CRITICAL' : (result.score > 0.7 ? '⚠️ HIGH RISK' : (result.score > 0.3 ? '⚡ ELEVATED' : '✅ STABLE'));

    let html = `
        <div class="insight-card" style="border-left-color: ${color}">
            <h3>
                <span>${severity}</span>
                <span style="opacity:0.3; font-size:0.55rem; font-family: 'JetBrains Mono', monospace;">${result.name}</span>
            </h3>
            <div class="insight-body">
                <b>${result.name}</b> has <b>${result.impact_count}</b> downstream dependencies. 
                ${result.score > 0.7
            ? 'This module is a <b>structural bottleneck</b>—modifying it risks a cascading system failure across dependent services.'
            : result.score > 0.3
                ? 'Moderate coupling detected. Changes should be tested against dependent modules before deployment.'
                : 'Low fragility. This module is well-isolated with minimal blast radius.'}
            </div>
            <div class="blast-radius-info">
                ${(result.blast_radius || []).map(node => `<span class="affected-tag">${node}</span>`).join('')}
            </div>
        </div>
    `;

    // AI Explanation (from Bedrock)
    if (result.explanation) {
        html += `
            <div class="ai-explanation">
                ${result.explanation}
            </div>
        `;
    }

    // Propagation insight
    html += `
        <div class="insight-card" style="border-left-color: var(--accent-purple); animation-delay: 0.15s;">
            <h3>
                <span>🧠 GNN Propagation</span>
            </h3>
            <div class="insight-body">
                The GraphSAGE model propagated risk through <b>${(result.blast_radius || []).length + 2}</b> hops of the dependency graph, 
                computing aggregated neighborhood embeddings to predict a fragility index of <b>${(result.score * 100).toFixed(1)}%</b>.
            </div>
        </div>
        
        <div style="font-size: 0.65rem; color: var(--accent-blue); padding: 0.3rem 0.8rem; font-weight: 700; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.08em;">
            💡 Hover over graph nodes to inspect blast weights
        </div>
    `;

    container.innerHTML = html;
}

// ======================== DEPENDENCIES ========================
function updateDependencies(result) {
    const section = document.getElementById('deps-section');
    const list = document.getElementById('deps-list');
    if (!section || !list) return;

    const blastRadius = result.blast_radius || [];

    if (blastRadius.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';

    list.innerHTML = blastRadius.map((dep, i) => {
        const strength = (1 - (i * 0.12)).toFixed(2);
        const strengthClass = strength > 0.6 ? 'high' : (strength > 0.3 ? 'medium' : 'low');
        return `
            <div class="dep-item">
                <span class="dep-name">${dep}</span>
                <span class="dep-strength ${strengthClass}">${(strength * 100).toFixed(0)}%</span>
            </div>
        `;
    }).join('');
}

// ======================== CODE MODAL ========================
function openCodeModal(filename, score) {
    const modal = document.getElementById('code-modal');
    const cleanName = filename.replace("⚠ ", "");
    document.getElementById('modal-filename').innerText = cleanName;

    const scoreBadge = document.getElementById('modal-score-badge');
    if (scoreBadge) {
        const pct = Math.round(score * 100);
        scoreBadge.innerText = pct + '%';
        if (score > 0.7) {
            scoreBadge.style.background = 'rgba(255, 62, 62, 0.15)';
            scoreBadge.style.color = '#ff3e3e';
            scoreBadge.style.borderColor = 'rgba(255, 62, 62, 0.3)';
        } else if (score > 0.3) {
            scoreBadge.style.background = 'rgba(255, 202, 40, 0.15)';
            scoreBadge.style.color = '#ffca28';
            scoreBadge.style.borderColor = 'rgba(255, 202, 40, 0.3)';
        } else {
            scoreBadge.style.background = 'rgba(0, 255, 136, 0.15)';
            scoreBadge.style.color = '#00ff88';
            scoreBadge.style.borderColor = 'rgba(0, 255, 136, 0.3)';
        }
    }

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

function closeModal() {
    document.getElementById('code-modal').classList.remove('active');
}

// Keyboard close
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

// ======================== UTILITIES ========================
function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    if (!obj) return;
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        obj.innerHTML = Math.floor(eased * (end - start) + start);
        if (progress < 1) window.requestAnimationFrame(step);
    };
    window.requestAnimationFrame(step);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
