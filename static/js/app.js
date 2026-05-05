/* ═══════════════════════════════════════════════════════════════
   CV_05 — Face Detection & Recognition — Frontend Logic
   ═══════════════════════════════════════════════════════════════ */

// ── State ──
const state = {
    facePath: null,
    recogPath: null,
};

// ── Helpers ──
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

function setStatus(id, msg, type = "") {
    const el = $(id);
    if (!el) return;
    el.className = "status " + type;
    el.innerHTML = type === "loading" ? `<span class="spinner"></span>${msg}` : msg;
}

function badge(text, cls = "cyan") {
    return `<span class="badge badge-${cls}">${text}</span>`;
}

function showImage(containerId, url) {
    const el = $(containerId);
    if (!el) return;
    el.innerHTML = `<img src="${url}" onclick="zoomImage(this.src)">`;
}

function zoomImage(src) {
    const overlay = document.createElement("div");
    overlay.className = "zoom-overlay";
    overlay.innerHTML = `<img src="${src}">`;
    overlay.onclick = () => overlay.remove();
    document.body.appendChild(overlay);
}

async function uploadFile(file) {
    const fd = new FormData();
    fd.append("image", file);
    const resp = await fetch("/api/upload/", { method: "POST", body: fd });
    return resp.json();
}

function setupUpload(zoneId, fileId, previewId, stateKey) {
    const zone = $(zoneId);
    const input = $(fileId);
    if (!zone || !input) return;

    zone.onclick = () => input.click();
    zone.ondragover = (e) => { e.preventDefault(); zone.style.borderColor = "var(--accent-cyan)"; };
    zone.ondragleave = () => { zone.style.borderColor = ""; };
    zone.ondrop = (e) => {
        e.preventDefault();
        zone.style.borderColor = "";
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event("change"));
        }
    };

    input.onchange = async () => {
        if (!input.files.length) return;
        const file = input.files[0];
        zone.classList.add("has-file");
        zone.innerHTML = `<span class="upload-icon">✅</span><span>${file.name}</span>`;

        // Preview
        if (previewId) {
            const reader = new FileReader();
            reader.onload = (e) => {
                $(previewId).innerHTML = `<img src="${e.target.result}">`;
            };
            reader.readAsDataURL(file);
        }

        // Upload
        try {
            const data = await uploadFile(file);
            if (data.path) state[stateKey] = data.path;
        } catch (err) {
            console.error("Upload error:", err);
        }
    };
}

// ── Tabs ──
function initTabs() {
    $$(".tab").forEach((tab) => {
        tab.onclick = () => {
            $$(".tab").forEach((t) => t.classList.remove("active"));
            $$(".panel").forEach((p) => p.classList.add("hidden"));
            tab.classList.add("active");
            $(`#panel-${tab.dataset.tab}`).classList.remove("hidden");
        };
    });
}

// ── Sliders ──
function initSliders() {
    const sliders = {
        sliderMinSize: { display: "valMinSize" },
        sliderMaxSize: { display: "valMaxSize" },
        sliderSplit: { display: "valSplit" },
        sliderComp: { display: "valComp" },
        sliderThresh: { display: "valThresh" },
        sliderSplitEval: { display: "valSplitEval" },
        sliderCompEval: { display: "valCompEval" },
    };

    for (const [id, cfg] of Object.entries(sliders)) {
        const slider = $(`#${id}`);
        const display = $(`#${cfg.display}`);
        if (!slider || !display) continue;
        const update = () => {
            display.textContent = cfg.transform ? cfg.transform(slider.value) : slider.value;
        };
        slider.oninput = update;
        update();
    }
}

// ── Auto-load Dataset Info ──
async function loadDatasetInfo() {
    try {
        const resp = await fetch("/api/dataset-info/");
        const data = await resp.json();

        const infoHtml = data.loaded
            ? `✅ <strong>Dataset loaded:</strong> ${data.subjects} subjects, ${data.images} images`
            : `⚠️ <strong>No dataset found.</strong> Add subject folders (s1, s2, ...) with face images to the <code>dataset/</code> folder.`;

        for (const id of ["#datasetInfo", "#datasetInfoEval"]) {
            const el = $(id);
            if (el) el.innerHTML = infoHtml;
        }
    } catch (err) {
        console.error("Error loading dataset info:", err);
        for (const id of ["#datasetInfo", "#datasetInfoEval"]) {
            const el = $(id);
            if (el) el.innerHTML = "❌ Error loading dataset info";
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  API CALLS
// ═══════════════════════════════════════════════════════════

// ── Face Detection ──
async function detectFaces() {
    if (!state.facePath) { setStatus("#statusFace", "Upload an image first", "error"); return; }
    setStatus("#statusFace", "Detecting faces...", "loading");
    $("#btnDetectFaces").disabled = true;

    try {
        const fd = new FormData();
        fd.append("image_path", state.facePath);
        fd.append("min_size", $("#sliderMinSize").value);
        fd.append("max_size", $("#sliderMaxSize").value);

        const resp = await fetch("/api/detect-faces/", { method: "POST", body: fd });
        const data = await resp.json();

        if (data.error) throw new Error(data.error);

        showImage("#resultFace", data.output);
        $("#badgesFace").innerHTML = [
            badge(`${data.face_count} face(s)`, "green"),
            badge(`${data.time_ms}ms`, "cyan"),
        ].join("");
        setStatus("#statusFace", "Done!", "success");
    } catch (err) {
        setStatus("#statusFace", err.message, "error");
    }
    $("#btnDetectFaces").disabled = false;
}

// ── Train Model ──
async function trainModel() {
    setStatus("#statusTrain", "Training eigenfaces model... This may take a moment.", "loading");
    $("#btnTrain").disabled = true;

    try {
        const fd = new FormData();
        fd.append("train_ratio", $("#sliderSplit").value / 100);
        fd.append("num_components", $("#sliderComp").value);

        const resp = await fetch("/api/train-model/", { method: "POST", body: fd });
        const data = await resp.json();

        if (data.error) throw new Error(data.error);

        $("#badgesTrain").innerHTML = [
            badge(`${data.num_components} components`, "cyan"),
            badge(`${data.num_training} training images`, "green"),
            badge(`${data.num_subjects} subjects`, "purple"),
            badge(`${data.time_ms}ms`, "orange"),
        ].join("");

        // Show eigenfaces
        let efHtml = "";
        if (data.mean_face) {
            efHtml += `<div class="eigenface-item"><img src="${data.mean_face}"><div class="ef-label">Mean Face</div></div>`;
        }
        if (data.eigenfaces) {
            data.eigenfaces.forEach((url, i) => {
                efHtml += `<div class="eigenface-item"><img src="${url}"><div class="ef-label">EF ${i + 1}</div></div>`;
            });
        }
        $("#eigenfacesGrid").innerHTML = efHtml;
        setStatus("#statusTrain", `Model trained with ${data.num_components} eigenfaces`, "success");
    } catch (err) {
        setStatus("#statusTrain", err.message, "error");
    }
    $("#btnTrain").disabled = false;
}

// ── Recognize Face ──
async function recognizeFace() {
    if (!state.recogPath) { setStatus("#statusRecog", "Upload a face image first", "error"); return; }
    setStatus("#statusRecog", "Recognizing...", "loading");
    $("#btnRecognize").disabled = true;

    try {
        const fd = new FormData();
        fd.append("image_path", state.recogPath);
        fd.append("threshold", $("#sliderThresh").value);

        const resp = await fetch("/api/recognize/", { method: "POST", body: fd });
        const data = await resp.json();

        if (data.error) throw new Error(data.error);

        const isKnown = data.predicted_label >= 0;
        $("#resultRecog").innerHTML = `
            <div class="recog-label ${isKnown ? "" : "unknown"}">
                ${isKnown ? `Subject ${data.predicted_label}` : "Unknown"}
            </div>
            <div class="recog-info">
                Distance: ${data.distance} · Confidence: ${(data.confidence * 100).toFixed(2)}% · Time: ${data.time_ms}ms
            </div>
        `;
        setStatus("#statusRecog", isKnown ? "Match found!" : "No match (unknown face)", isKnown ? "success" : "error");
    } catch (err) {
        setStatus("#statusRecog", err.message, "error");
    }
    $("#btnRecognize").disabled = false;
}

// ── Full Evaluation ──
async function runEvaluation() {
    setStatus("#statusEval", "Running full evaluation... This may take several minutes.", "loading");
    $("#btnEvaluate").disabled = true;

    try {
        const fd = new FormData();
        fd.append("train_ratio", $("#sliderSplitEval").value / 100);
        fd.append("num_components", $("#sliderCompEval").value);

        const resp = await fetch("/api/evaluate/", { method: "POST", body: fd });
        const data = await resp.json();

        if (data.error) throw new Error(data.error);

        // Metrics
        $("#metricsGrid").innerHTML = `
            <div class="metric-card"><div class="metric-value">${data.accuracy}%</div><div class="metric-label">Accuracy</div></div>
            <div class="metric-card"><div class="metric-value">${data.correct}/${data.total}</div><div class="metric-label">Correct</div></div>
            <div class="metric-card"><div class="metric-value">${data.num_components}</div><div class="metric-label">Components</div></div>
            <div class="metric-card"><div class="metric-value">${data.num_subjects}</div><div class="metric-label">Subjects</div></div>
            <div class="metric-card"><div class="metric-value">${data.auc}</div><div class="metric-label">AUC</div></div>
            <div class="metric-card"><div class="metric-value">${data.eer}</div><div class="metric-label">EER</div></div>
            <div class="metric-card"><div class="metric-value">${data.train_time_ms}ms</div><div class="metric-label">Train Time</div></div>
            <div class="metric-card"><div class="metric-value">${data.eval_time_ms}ms</div><div class="metric-label">Eval Time</div></div>
        `;

        // ROC Curve
        if (data.roc_curve) {
            $("#rocContainer").innerHTML = `<h3>ROC Curve</h3><img src="${data.roc_curve}" onclick="zoomImage(this.src)">`;
        }

        // Confusion Matrix
        if (data.confusion_matrix) {
            $("#cmContainer").innerHTML = `<h3>Confusion Matrix</h3><img src="${data.confusion_matrix}" onclick="zoomImage(this.src)">`;
        }

        // Eigenfaces
        let efHtml = "";
        if (data.mean_face) {
            efHtml += `<div class="eigenface-item"><img src="${data.mean_face}"><div class="ef-label">Mean</div></div>`;
        }
        if (data.eigenfaces) {
            data.eigenfaces.forEach((url, i) => {
                efHtml += `<div class="eigenface-item"><img src="${url}"><div class="ef-label">EF${i + 1}</div></div>`;
            });
        }
        $("#eigenfacesGridEval").innerHTML = efHtml;

        setStatus("#statusEval", `Evaluation complete — Accuracy: ${data.accuracy}%`, "success");
    } catch (err) {
        setStatus("#statusEval", err.message, "error");
    }
    $("#btnEvaluate").disabled = false;
}

// ═══════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initSliders();
    loadDatasetInfo();

    // Upload zones
    setupUpload("#uploadFace", "#fileFace", "#previewFace", "facePath");
    setupUpload("#uploadRecog", "#fileRecog", "#previewRecog", "recogPath");

    // Buttons
    $("#btnDetectFaces").onclick = detectFaces;
    $("#btnTrain").onclick = trainModel;
    $("#btnRecognize").onclick = recognizeFace;
    $("#btnEvaluate").onclick = runEvaluation;
});
