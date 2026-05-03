/**
 * CV_03 — Frontend Application
 * Feature Detection, SIFT Descriptors & Feature Matching
 */

(function () {
    'use strict';

    /* ═══════════════════════════════════════════════════════════════
     *  STATE
     * ═══════════════════════════════════════════════════════════════ */

    const state = {
        features: { path: null, url: null },
        sift:     { path: null, url: null },
        match1:   { path: null, url: null },
        match2:   { path: null, url: null },
    };

    /* ═══════════════════════════════════════════════════════════════
     *  DOM REFS
     * ═══════════════════════════════════════════════════════════════ */

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const loader = $('#loader');

    /* ═══════════════════════════════════════════════════════════════
     *  TAB NAVIGATION
     * ═══════════════════════════════════════════════════════════════ */

    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            $$('.tab-panel').forEach(p => p.classList.remove('active'));
            const panel = $(`#panel-${btn.dataset.tab}`);
            if (panel) panel.classList.add('active');
        });
    });

    /* ═══════════════════════════════════════════════════════════════
     *  SLIDER VALUE SYNC
     * ═══════════════════════════════════════════════════════════════ */

    $$('.ctrl input[type="range"]').forEach(input => {
        const valSpan = input.closest('.ctrl').querySelector('.val');
        const update = () => {
            valSpan.textContent = input.value;
        };
        input.addEventListener('input', update);
        update();
    });

    /* ═══════════════════════════════════════════════════════════════
     *  FILE UPLOAD HELPER
     * ═══════════════════════════════════════════════════════════════ */

    function setupUpload(inputId, zoneId, thumbId, textId, stateKey, enableBtns) {
        const input = $(`#${inputId}`);
        const zone  = $(`#${zoneId}`);
        const thumb = $(`#${thumbId}`);
        const text  = $(`#${textId}`);

        // Drag & drop
        zone.addEventListener('dragover', e => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', e => {
            e.preventDefault();
            zone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                handleFile(input.files[0]);
            }
        });

        input.addEventListener('change', () => {
            if (input.files[0]) handleFile(input.files[0]);
        });

        function handleFile(file) {
            text.textContent = file.name;

            // Preview
            const reader = new FileReader();
            reader.onload = (e) => {
                thumb.src = e.target.result;
                thumb.hidden = false;
            };
            reader.readAsDataURL(file);

            // Upload to server
            const fd = new FormData();
            fd.append('image', file);

            showLoader('Uploading…');
            fetch('/api/upload/', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(data => {
                    state[stateKey].path = data.path;
                    state[stateKey].url = data.url;
                    hideLoader();

                    // Enable buttons
                    enableBtns.forEach(id => {
                        const btn = $(`#${id}`);
                        if (btn) checkEnableBtn(btn);
                    });
                })
                .catch(err => {
                    hideLoader();
                    alert('Upload failed: ' + err.message);
                });
        }
    }

    function checkEnableBtn(btn) {
        const id = btn.id;
        if (id === 'btn-detect')     btn.disabled = !state.features.path;
        if (id === 'btn-sift')       btn.disabled = !state.sift.path;
        if (id === 'btn-match-ssd')  btn.disabled = !(state.match1.path && state.match2.path);
        if (id === 'btn-match-ncc')  btn.disabled = !(state.match1.path && state.match2.path);
    }

    // Wire up all upload zones
    setupUpload('input-features', 'upload-zone-features', 'thumb-features',
                'upload-text-features', 'features', ['btn-detect']);

    setupUpload('input-sift', 'upload-zone-sift', 'thumb-sift',
                'upload-text-sift', 'sift', ['btn-sift']);

    setupUpload('input-match1', 'upload-zone-match1', 'thumb-match1',
                'upload-text-match1', 'match1', ['btn-match-ssd', 'btn-match-ncc']);

    setupUpload('input-match2', 'upload-zone-match2', 'thumb-match2',
                'upload-text-match2', 'match2', ['btn-match-ssd', 'btn-match-ncc']);

    /* ═══════════════════════════════════════════════════════════════
     *  LOADER
     * ═══════════════════════════════════════════════════════════════ */

    function showLoader(msg) {
        loader.querySelector('.loader-text').textContent = msg || 'Processing…';
        loader.hidden = false;
    }

    function hideLoader() {
        loader.hidden = true;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  GET CONTROL VALUES
     * ═══════════════════════════════════════════════════════════════ */

    function getCtrlVal(key) {
        const ctrl = document.querySelector(`.ctrl[data-key="${key}"] input[type="range"]`);
        return ctrl ? ctrl.value : null;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  1. FEATURE DETECTION (Harris + λ⁻)
     * ═══════════════════════════════════════════════════════════════ */

    $('#btn-detect').addEventListener('click', async () => {
        if (!state.features.path) return;

        const params = {
            image_path: state.features.path,
            k: getCtrlVal('k'),
            threshold: getCtrlVal('threshold'),
            nms_radius: getCtrlVal('nms_radius'),
        };

        const lambdaParams = {
            image_path: state.features.path,
            threshold: getCtrlVal('lambda_threshold'),
            nms_radius: getCtrlVal('lambda_nms_radius'),
        };

        showLoader('Detecting Harris corners…');

        try {
            // Run Harris
            const harrisFd = new FormData();
            for (const [k, v] of Object.entries(params)) harrisFd.append(k, v);
            const harrisRes = await fetch('/api/harris/', { method: 'POST', body: harrisFd });
            const harris = await harrisRes.json();

            // Run Lambda-minus
            showLoader('Detecting λ⁻ corners…');
            const lambdaFd = new FormData();
            for (const [k, v] of Object.entries(lambdaParams)) lambdaFd.append(k, v);
            const lambdaRes = await fetch('/api/lambda/', { method: 'POST', body: lambdaFd });
            const lambda = await lambdaRes.json();

            hideLoader();

            // Update UI
            $('#results-features').hidden = false;

            $('#img-feat-original').src = state.features.url;
            $('#img-harris').src = harris.output + '?t=' + Date.now();
            $('#img-lambda').src = lambda.output + '?t=' + Date.now();

            $('#harris-count').textContent = harris.corner_count + ' corners';
            $('#harris-time').textContent = harris.time_ms + ' ms';
            $('#lambda-count').textContent = lambda.corner_count + ' corners';
            $('#lambda-time').textContent = lambda.time_ms + ' ms';

            // Scroll results into view
            $('#results-features').scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (err) {
            hideLoader();
            alert('Detection failed: ' + err.message);
        }
    });

    /* ═══════════════════════════════════════════════════════════════
     *  2. SIFT DESCRIPTORS
     * ═══════════════════════════════════════════════════════════════ */

    $('#btn-sift').addEventListener('click', async () => {
        if (!state.sift.path) return;

        const fd = new FormData();
        fd.append('image_path', state.sift.path);
        fd.append('num_octaves', getCtrlVal('num_octaves'));
        fd.append('scales_per_octave', getCtrlVal('scales_per_octave'));
        fd.append('contrast_threshold', getCtrlVal('contrast_threshold'));
        fd.append('edge_threshold', getCtrlVal('edge_threshold'));

        showLoader('Generating SIFT descriptors…');

        try {
            const res = await fetch('/api/sift/', { method: 'POST', body: fd });
            const data = await res.json();
            hideLoader();

            $('#results-sift').hidden = false;

            $('#img-sift-original').src = state.sift.url;
            $('#img-sift').src = data.output + '?t=' + Date.now();

            $('#sift-count').textContent = data.keypoint_count + ' keypoints';
            $('#sift-dim').textContent = data.descriptor_dim + '-D';
            $('#sift-time').textContent = data.time_ms + ' ms';

            // Scroll results into view
            $('#results-sift').scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (err) {
            hideLoader();
            alert('SIFT failed: ' + err.message);
        }
    });

    /* ═══════════════════════════════════════════════════════════════
     *  3. FEATURE MATCHING
     * ═══════════════════════════════════════════════════════════════ */

    async function runMatching(method) {
        if (!state.match1.path || !state.match2.path) return;

        const fd = new FormData();
        fd.append('image_path_1', state.match1.path);
        fd.append('image_path_2', state.match2.path);

        const endpoint = method === 'ssd' ? '/api/match-ssd/' : '/api/match-ncc/';
        const methodName = method === 'ssd' ? 'SSD' : 'NCC';

        if (method === 'ssd') {
            fd.append('ratio_threshold', getCtrlVal('ratio_threshold'));
        } else {
            fd.append('ncc_threshold', getCtrlVal('ncc_threshold'));
        }

        showLoader(`Matching features (${methodName})…`);

        try {
            const res = await fetch(endpoint, { method: 'POST', body: fd });
            const data = await res.json();
            hideLoader();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            $('#results-matching').hidden = false;
            $('#match-stats-bar').hidden = false;

            $('#match-method-title').textContent = `Feature Matches (${methodName})`;
            $('#img-matches').src = data.output + '?t=' + Date.now();

            $('#match-kp1').textContent = data.kp_count_1;
            $('#match-kp2').textContent = data.kp_count_2;
            $('#match-count').textContent = data.match_count;
            $('#match-sift1-time').textContent = data.sift1_time_ms + ' ms';
            $('#match-sift2-time').textContent = data.sift2_time_ms + ' ms';
            $('#match-match-time').textContent = data.match_time_ms + ' ms';
            $('#match-total-time').textContent = data.total_time_ms + ' ms';

            // Scroll results into view
            $('#results-matching').scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (err) {
            hideLoader();
            alert('Matching failed: ' + err.message);
        }
    }

    $('#btn-match-ssd').addEventListener('click', () => runMatching('ssd'));
    $('#btn-match-ncc').addEventListener('click', () => runMatching('ncc'));

    /* ═══════════════════════════════════════════════════════════════
     *  IMAGE ZOOM
     * ═══════════════════════════════════════════════════════════════ */

    document.addEventListener('click', (e) => {
        const img = e.target;
        if (img.tagName === 'IMG' && img.closest('.result-card') && img.src) {
            const overlay = document.createElement('div');
            overlay.className = 'zoom-overlay';
            const zoomed = document.createElement('img');
            zoomed.src = img.src;
            overlay.appendChild(zoomed);
            document.body.appendChild(overlay);

            overlay.addEventListener('click', () => overlay.remove());
            document.addEventListener('keydown', function handler(e) {
                if (e.key === 'Escape') {
                    overlay.remove();
                    document.removeEventListener('keydown', handler);
                }
            });
        }
    });

})();
