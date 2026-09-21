// ==========================================================================
// ZIURGABETASUN LABORATEGIA - FRONTEND LOGIC & INTERACTIVITIES
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    // --- Canvas Elements & State ---
    const canvas = document.getElementById('paintCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const eraserBtn = document.getElementById('eraserBtn');
    const eraserText = document.getElementById('eraserText');
    const strokeBtns = document.querySelectorAll('.stroke-btn');

    // --- Analytics UI Elements ---
    const emptyState = document.getElementById('emptyState');
    const resultBox = document.getElementById('resultBox');
    const predictionText = document.getElementById('predictionText');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const entropyBadge = document.getElementById('entropyBadge');
    const entropyGaugeFill = document.getElementById('entropyGaugeFill');
    const entropyThresholdMarker = document.getElementById('entropyThresholdMarker');
    const entropyStatusLabel = document.getElementById('entropyStatusLabel');
    const thresholdValText = document.getElementById('thresholdValText');
    const probsBarChart = document.getElementById('probsBarChart');
    const warningBox = document.getElementById('warningBox');

    // --- Modal Phase 2 Elements (Continual Learning) ---
    const teachSystemBtn = document.getElementById('teachSystemBtn');
    const learningOverlay = document.getElementById('learningOverlay');
    const labelInputGroup = document.getElementById('labelInputGroup');
    const charLabelInput = document.getElementById('charLabel');
    const startLearningBtn = document.getElementById('startLearningBtn');
    const cancelLearningBtn = document.getElementById('cancelLearningBtn');

    const drawingGroup = document.getElementById('drawingGroup');
    const learnCanvas = document.getElementById('learnCanvas');
    const learnCtx = learnCanvas.getContext('2d');
    const clearLearnBtn = document.getElementById('clearLearnBtn');
    const saveDrawingBtn = document.getElementById('saveDrawingBtn');
    const drawingProgress = document.getElementById('drawingProgress');
    const drawingPrompt = document.getElementById('drawingPrompt');
    const drawnThumbnails = document.getElementById('drawnThumbnails');
    const learningStatus = document.getElementById('learningStatus');

    const summaryOverlay = document.getElementById('summaryOverlay');
    const finalEntropyBadge = document.getElementById('finalEntropyBadge');
    const finalStatusBadge = document.getElementById('finalStatusBadge');
    const finishLearningBtn = document.getElementById('finishLearningBtn');

    // --- Modal Phase 3 Elements (Tree & History) ---
    const toggleTreeBtn = document.getElementById('toggleTreeBtn');
    const treeOverlay = document.getElementById('treeOverlay');
    const closeTreeBtn = document.getElementById('closeTreeBtn');
    const treeContainer = document.getElementById('treeContainer');
    const historyContainer = document.getElementById('historyContainer');
    const activeModelText = document.getElementById('activeModelText');

    // State Variables
    let currentStrokeWidth = 14;
    let isEraserMode = false;
    let isDrawingMain = false;
    let isDrawingLearn = false;

    let learningLabel = "";
    let collectedImages = [];
    let targetImages = 10;
    let totalGiven = 0;

    // --- Canvas Initialization ---
    function initCanvas(context, cvs) {
        context.fillStyle = 'white';
        context.fillRect(0, 0, cvs.width, cvs.height);
        context.lineWidth = currentStrokeWidth;
        context.lineCap = 'round';
        context.lineJoin = 'round';
        context.strokeStyle = isEraserMode ? 'white' : 'black';
    }

    initCanvas(ctx, canvas);
    initCanvas(learnCtx, learnCanvas);

    // Stroke width buttons
    strokeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            strokeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentStrokeWidth = parseInt(btn.getAttribute('data-size'));
            ctx.lineWidth = currentStrokeWidth;
            learnCtx.lineWidth = currentStrokeWidth;
        });
    });

    // Eraser toggle
    if (eraserBtn) {
        eraserBtn.addEventListener('click', () => {
            isEraserMode = !isEraserMode;
            if (isEraserMode) {
                eraserBtn.classList.add('active');
                eraserText.textContent = 'Goma';
            } else {
                eraserBtn.classList.remove('active');
                eraserText.textContent = 'Luma';
            }
            ctx.strokeStyle = isEraserMode ? 'white' : 'black';
        });
    }

    // Canvas Mouse / Touch Events Setup
    function setupCanvasDrawing(cvs, context, getDrawingFlag, setDrawingFlag) {
        function getPos(evt) {
            const rect = cvs.getBoundingClientRect();
            return {
                x: (evt.clientX || evt.touches[0].clientX) - rect.left,
                y: (evt.clientY || evt.touches[0].clientY) - rect.top
            };
        }

        function start(e) {
            setDrawingFlag(true);
            const pos = getPos(e);
            context.beginPath();
            context.moveTo(pos.x, pos.y);
        }

        function stop() {
            setDrawingFlag(false);
            context.beginPath();
        }

        function draw(e) {
            if (!getDrawingFlag()) return;
            const pos = getPos(e);
            context.lineTo(pos.x, pos.y);
            context.stroke();
            context.beginPath();
            context.moveTo(pos.x, pos.y);
        }

        cvs.addEventListener('mousedown', start);
        cvs.addEventListener('mouseup', stop);
        cvs.addEventListener('mouseleave', stop);
        cvs.addEventListener('mousemove', draw);

        cvs.addEventListener('touchstart', (e) => { e.preventDefault(); start(e); }, { passive: false });
        cvs.addEventListener('touchend', stop);
        cvs.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); }, { passive: false });
    }

    setupCanvasDrawing(canvas, ctx, () => isDrawingMain, (v) => isDrawingMain = v);
    setupCanvasDrawing(learnCanvas, learnCtx, () => isDrawingLearn, (v) => isDrawingLearn = v);

    // Clear Canvas Actions
    clearBtn.addEventListener('click', () => {
        initCanvas(ctx, canvas);
        emptyState.classList.remove('hidden');
        resultBox.classList.add('hidden');
    });

    clearLearnBtn.addEventListener('click', () => {
        initCanvas(learnCtx, learnCanvas);
    });

    // --- PREDICT ACTION & ANALYTICS RENDER ---
    predictBtn.addEventListener('click', async () => {
        const dataURL = canvas.toDataURL('image/png');
        predictBtn.disabled = true;
        predictBtn.querySelector('span').textContent = 'Iragartzen...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            });

            const data = await response.json();

            if (data.error) {
                alert("Errorea: " + data.error);
            } else {
                emptyState.classList.add('hidden');
                resultBox.classList.remove('hidden');

                // 1. Prediction Symbol & Confidence
                predictionText.textContent = data.prediction_label !== undefined ? data.prediction_label : data.prediction;
                confidenceBadge.textContent = `${(data.confidence * 100).toFixed(1)}%`;
                entropyBadge.textContent = data.entropy.toFixed(3);

                // 2. Entropy Gauge Calculation & Rendering
                const threshold = data.threshold || 1.25;
                const maxEntropy = 2.30; // Max theoretical entropy for K=10
                const entropyPct = Math.min(100, Math.max(0, (data.entropy / maxEntropy) * 100));
                const thresholdPct = Math.min(100, Math.max(0, (threshold / maxEntropy) * 100));

                entropyGaugeFill.style.width = `${entropyPct}%`;
                entropyThresholdMarker.style.left = `${thresholdPct}%`;
                thresholdValText.textContent = `Muga: ${threshold.toFixed(2)}`;

                if (data.is_unknown) {
                    entropyStatusLabel.className = 'status-badge red';
                    entropyStatusLabel.textContent = 'Informazio-Itzala (OOD)';
                    warningBox.classList.remove('hidden');
                } else if (data.entropy > 0.5) {
                    entropyStatusLabel.className = 'status-badge yellow';
                    entropyStatusLabel.textContent = 'Zalantza Ertaina';
                    warningBox.classList.add('hidden');
                } else {
                    entropyStatusLabel.className = 'status-badge green';
                    entropyStatusLabel.textContent = 'Ikusgaiko Datu Zerbitzatua';
                    warningBox.classList.add('hidden');
                }

                // 3. Class Probabilities Bar Chart Rendering
                renderProbabilitiesChart(data.probabilities, data.labels, data.prediction);
            }
        } catch (err) {
            alert("Konexio errorea gertatu da zerbitzariarekin.");
            console.error(err);
        } finally {
            predictBtn.disabled = false;
            predictBtn.querySelector('span').textContent = 'Iragarri eta Neurtu';
        }
    });

    // Render 10-Class Probability Bars
    function renderProbabilitiesChart(probs, labels, topIndex) {
        probsBarChart.innerHTML = '';
        if (!probs || probs.length === 0) return;

        probs.forEach((prob, idx) => {
            const labelStr = labels && labels[idx] ? labels[idx] : idx;
            const pct = (prob * 100).toFixed(1);
            const isTop = idx === topIndex;

            const card = document.createElement('div');
            card.className = `prob-bar-card ${isTop ? 'top-class' : ''}`;
            card.innerHTML = `
                <div class="prob-class-label">
                    <span>${labelStr}</span>
                    <span>${pct}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: ${pct}%"></div>
                </div>
            `;
            probsBarChart.appendChild(card);
        });
    }

    // --- CONTINUAL LEARNING WORKFLOW ---
    teachSystemBtn.addEventListener('click', () => {
        learningOverlay.classList.remove('hidden');
        labelInputGroup.classList.remove('hidden');
        drawingGroup.classList.add('hidden');
        learningStatus.classList.add('hidden');
        charLabelInput.value = '';
        collectedImages = [];
        totalGiven = 0;
        targetImages = 10;
        drawnThumbnails.innerHTML = '';
    });

    cancelLearningBtn.addEventListener('click', () => {
        learningOverlay.classList.add('hidden');
    });

    startLearningBtn.addEventListener('click', () => {
        const label = charLabelInput.value.trim();
        if (!label) {
            alert("Sartu karaktere berriaren izena mesedez.");
            return;
        }
        learningLabel = label;

        labelInputGroup.classList.add('hidden');
        drawingGroup.classList.remove('hidden');
        updateProgressUI();
        initCanvas(learnCtx, learnCanvas);
    });

    function updateProgressUI() {
        drawingProgress.textContent = `${totalGiven} / ${targetImages}`;
        drawingPrompt.textContent = `'${learningLabel}' karakterearen lagin berritua marraztu.`;
    }

    saveDrawingBtn.addEventListener('click', async () => {
        const dataURL = learnCanvas.toDataURL('image/png');
        collectedImages.push(dataURL);
        totalGiven++;

        // Add Thumbnail preview
        const thumb = document.createElement('img');
        thumb.src = dataURL;
        thumb.className = 'thumb-img';
        drawnThumbnails.appendChild(thumb);

        if (totalGiven < targetImages) {
            updateProgressUI();
            initCanvas(learnCtx, learnCanvas);
        } else {
            await sendDataForRetraining();
        }
    });

    async function sendDataForRetraining() {
        drawingGroup.classList.add('hidden');
        learningStatus.classList.remove('hidden');

        try {
            const response = await fetch('/train_new_class', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    label: learningLabel,
                    images: collectedImages
                })
            });

            const data = await response.json();

            if (data.error) {
                alert("Errorea entrenatzean: " + data.error);
                learningOverlay.classList.add('hidden');
                return;
            }

            if (data.status === 'needs_more_data') {
                learningStatus.classList.add('hidden');
                drawingGroup.classList.remove('hidden');

                targetImages = data.required_total;
                collectedImages = [];
                totalGiven = data.current_total;

                updateProgressUI();
                initCanvas(learnCtx, learnCanvas);
                alert(`Entropia altuegia da oraindik (${data.entropy.toFixed(3)}). Beste ${targetImages - totalGiven} lagin gehiago behar ditu!`);
            } else if (data.status === 'success') {
                learningOverlay.classList.add('hidden');
                summaryOverlay.classList.remove('hidden');

                finalEntropyBadge.textContent = data.metrics.entropy.toFixed(3);
                finalStatusBadge.textContent = `${data.metrics.new_version_id || 'v2'} (${totalGiven} lagin)`;
                fetchTreeData();
            }
        } catch (err) {
            alert("Konexio errorea backend-arekin.");
            console.error(err);
            learningOverlay.classList.add('hidden');
        }
    }

    finishLearningBtn.addEventListener('click', () => {
        summaryOverlay.classList.add('hidden');
        initCanvas(ctx, canvas);
        emptyState.classList.remove('hidden');
        resultBox.classList.add('hidden');
    });

    // --- MODEL TREE & HISTORY WORKFLOW ---
    const resetTreeNavBtn = document.getElementById('resetTreeNavBtn');

    if (resetTreeNavBtn) {
        resetTreeNavBtn.addEventListener('click', () => {
            if (!confirm("Ziur zaude entrenatutako bertsio berri guztiak ezabatu eta oinarrizko ereduarekin zerotik hasi nahi duzula?")) return;
            fetch('/reset_tree', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            }).then(r => r.json()).then(data => {
                if (data.error) {
                    alert("Errorea berrezartzean: " + data.error);
                } else {
                    alert("Eredua arrakastaz berrezarri da Oinarrizko Bertsiora!");
                    fetchTreeData();
                    initCanvas(ctx, canvas);
                    emptyState.classList.remove('hidden');
                    resultBox.classList.add('hidden');
                }
            }).catch(e => console.error(e));
        });
    }

    if (toggleTreeBtn) {
        toggleTreeBtn.addEventListener('click', () => {
            treeOverlay.classList.remove('hidden');
            fetchTreeData();
        });
    }

    if (closeTreeBtn) {
        closeTreeBtn.addEventListener('click', () => {
            treeOverlay.classList.add('hidden');
        });
    }

    function fetchTreeData() {
        fetch('/get_tree')
            .then(r => r.json())
            .then(data => {
                renderTree(data.tree, data.active_id);
                renderHistory(data.history);
                if (activeModelText) activeModelText.textContent = `Eredu Aktiboa: ${data.active_id}`;
            })
            .catch(e => console.error("Zuhaitza lortzean errorea:", e));
    }

    function renderTree(nodes, activeId) {
        if (!treeContainer) return;
        treeContainer.innerHTML = '';

        const rootNode = nodes.find(n => n.parent_id === null);
        if (!rootNode) return;

        const buildHTML = (node, depth) => {
            const children = nodes.filter(n => n.parent_id === node.id);
            const isActive = node.id === activeId;

            let html = `
                <div class="tree-node-card ${isActive ? 'active-node' : ''}" style="margin-left: ${depth * 15}px;">
                    <div class="tree-node-title">
                        <span>${node.name} (${node.id})</span>
                        <span class="tree-tag ${isActive ? 'active' : ''}">${isActive ? 'AKTIBOA' : 'Gordeta'}</span>
                    </div>
            `;

            if (node.capabilities && node.capabilities.length > 0) {
                html += `<div class="tree-node-caps">Ikutsitako Karaktere Berriak: <strong>${node.capabilities.join(', ')}</strong></div>`;
            }

            if (node.metrics && node.metrics.entropy !== undefined) {
                html += `<div class="tree-node-caps">Azken Entropia: <strong>${node.metrics.entropy.toFixed(3)}</strong></div>`;
            }

            html += `<div class="tree-node-actions">`;
            if (!isActive) {
                html += `<button class="tree-btn primary" onclick="setActiveModel('${node.id}')">Hau Aktibatu</button>`;
            }
            if (node.parent_id && nodes.filter(n => n.parent_id === node.parent_id).length === 1) {
                html += `<button class="tree-btn warning" onclick="mergeModel('${node.id}')">Gurasoarekin Fusionatu</button>`;
            }
            html += `</div></div>`;

            children.forEach(c => { html += buildHTML(c, depth + 1); });
            return html;
        };

        treeContainer.innerHTML = buildHTML(rootNode, 0);
    }

    function renderHistory(historyList) {
        if (!historyContainer) return;
        historyContainer.innerHTML = '';
        if (!historyList || historyList.length === 0) {
            historyContainer.innerHTML = '<p style="color: var(--text-muted); font-size: 0.85rem;">Ez dago inferentziarik eredu aktibo honekin.</p>';
            return;
        }

        historyList.slice().reverse().forEach(log => {
            const d = new Date(log.timestamp);
            const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

            const card = document.createElement('div');
            card.className = 'history-item-card';
            card.innerHTML = `
                <div>
                    <strong>${log.prediction_label}</strong> 
                    <span style="color: var(--text-muted);">(${(log.confidence * 100).toFixed(0)}%)</span>
                </div>
                <div>
                    <span style="font-family: 'JetBrains Mono'; margin-right: 0.5rem;">H=${log.entropy.toFixed(2)}</span>
                    ${log.is_unknown ? '<span class="status-badge red">OOD</span>' : '<span class="status-badge green">Ikusgai</span>'}
                </div>
            `;
            historyContainer.appendChild(card);
        });
    }

    // Global Window Action Handlers
    window.setActiveModel = function(modelId) {
        fetch('/set_active', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        }).then(r => r.json()).then(() => fetchTreeData()).catch(e => console.error(e));
    };

    window.mergeModel = function(modelId) {
        if (!confirm("Ziur zaude bertsio hau gurasoarekin fusionatu nahi duzula?")) return;
        fetch('/merge_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        }).then(r => r.json()).then(() => fetchTreeData()).catch(e => console.error(e));
    };

    // Initial Tree Fetch
    fetchTreeData();
});
