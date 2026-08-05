document.addEventListener('DOMContentLoaded', () => {
    const addStreamForm = document.getElementById('add-stream-form');
    const activeStreamsContainer = document.getElementById('active-streams-container');
    const noStreamsMessage = document.getElementById('no-streams-message');
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');

    let streamPollingIntervals = {}; // 用於保存每個串流的狀態輪詢定時器
    let streamAnalysisCache = {}; // session_id -> 最新分析狀態（供下方面板彙總顯示）
    let streamCache = {}; // session_id -> 完整串流對象（避免 HTML 屬性 JSON 解析錯誤）
    let lastFetchId = 0; // 用於避免並行/非同步呼叫導致的卡片重複渲染問題
    let streamStartTimes = {}; // 用於記錄每個串流在本次瀏覽器會話中的最新啟動時間戳記，避免重啟時顯示舊的分析結果
    const MIN_CAPTURE_INTERVAL_MINUTES = 1;
    const MAX_CAPTURE_INTERVAL_MINUTES = 60;
    const DEFAULT_CAPTURE_INTERVAL_MINUTES = 1;
    const captureIntervalInput = document.getElementById('capture-interval-minutes');
    const intervalHint = document.getElementById('interval-hint');
    const globalIntervalSettingsBtn = document.getElementById('global-interval-settings-btn');
    const globalIntervalModal = document.getElementById('global-interval-modal');
    const saveGlobalIntervalBtn = document.getElementById('save-global-interval-btn');
    const closeGlobalIntervalModalBtn = document.getElementById('close-global-interval-modal-btn');
    let globalCaptureIntervalMinutes = DEFAULT_CAPTURE_INTERVAL_MINUTES;

    // 定義各廠商的模型清單 (與 upload.html 保持一致)
    const modelsByProvider = {
        'openrouter': [
            { value: 'google/gemini-3.1-flash-lite', label: 'Gemini 3.1 Flash Lite' },
            { value: 'openai/gpt-5.4', label: 'GPT-5.4' },
            { value: 'openai/gpt-4o', label: 'GPT-4o' },
            { value: 'google/gemini-3.1-flash-lite-preview', label: 'Gemini 3.1 Flash Lite Preview' },
            { value: 'qwen/qwen3.5-397b-a17b', label: 'Qwen 3.5 397B' },
            { value: 'google/gemini-2.0-flash-001', label: 'Gemini 2.0 Flash' },
            { value: 'anthropic/claude-3.5-sonnet', label: 'Claude 3.5 Sonnet' },
            { value: 'meta-llama/llama-3.2-90b-vision-instruct', label: 'Llama 3.2 90B Vision' }
        ],
        'local': [
            { value: 'qwen3-vl-235b-a22b-instruct-1m_moe', label: 'Qwen3-VL 23.5B' },
            { value: 'glm-4.6v', label: 'GLM-4.6V (Local)' }
        ]
    };

    // 全域函數，供 HTML 中的 onchange 呼叫
    function updateApiKeyField() {
        const provider = document.getElementById('provider-select').value;
        const apiKeyInput = document.getElementById('api-key');
        if (provider === 'local') {
            apiKeyInput.removeAttribute('required');
            apiKeyInput.placeholder = '地端模型不需 API Key';
        } else {
            apiKeyInput.setAttribute('required', 'required');
            apiKeyInput.placeholder = 'sk-or-v1-...（必填）';
        }
    }

    function parseCaptureIntervalMinutes(rawValue) {
        const parsed = Number(rawValue);
        if (!Number.isInteger(parsed)) return null;
        if (parsed < MIN_CAPTURE_INTERVAL_MINUTES || parsed > MAX_CAPTURE_INTERVAL_MINUTES) return null;
        return parsed;
    }

    function getIntervalMinutesFromStream(stream) {
        const fromMinutes = parseCaptureIntervalMinutes(stream?.capture_interval_minutes);
        if (fromMinutes !== null) return fromMinutes;

        const seconds = Number(stream?.capture_interval_seconds);
        if (Number.isFinite(seconds) && seconds > 0 && seconds % 60 === 0) {
            const minutes = seconds / 60;
            const parsedMinutes = parseCaptureIntervalMinutes(minutes);
            if (parsedMinutes !== null) return parsedMinutes;
        }

        return DEFAULT_CAPTURE_INTERVAL_MINUTES;
    }

    function updateIntervalGuidance() {
        if (!captureIntervalInput) return;
        const selectedMinutes = parseCaptureIntervalMinutes(captureIntervalInput.value);
        const intervalText = selectedMinutes ?? 'N';

        if (intervalHint) {
            intervalHint.textContent = `全域分析間隔：首次分析會在下一個整分鐘開始，之後每 ${intervalText} 分鐘執行一次。`;
        }
    }

    function openGlobalIntervalModal() {
        if (!globalIntervalModal) return;
        captureIntervalInput.value = String(globalCaptureIntervalMinutes);
        captureIntervalInput.classList.remove('is-invalid');
        updateIntervalGuidance();
        globalIntervalModal.classList.remove('d-none');
        globalIntervalModal.setAttribute('aria-hidden', 'false');
        captureIntervalInput.focus();
    }

    function closeGlobalIntervalModal() {
        if (!globalIntervalModal) return;
        globalIntervalModal.classList.add('d-none');
        globalIntervalModal.setAttribute('aria-hidden', 'true');
    }

    function setCaptureIntervalInput(minutes) {
        if (!captureIntervalInput) return;
        const parsed = parseCaptureIntervalMinutes(minutes);
        if (parsed !== null) {
            globalCaptureIntervalMinutes = parsed;
            captureIntervalInput.value = String(parsed);
            captureIntervalInput.classList.remove('is-invalid');
        }
        updateIntervalGuidance();
    }

    async function fetchGlobalCaptureInterval() {
        if (!captureIntervalInput) return;
        try {
            const response = await fetch('/api/stream/global_interval');
            const result = await response.json();
            if (result.success) {
                setCaptureIntervalInput(result.capture_interval_minutes);
            } else {
                updateIntervalGuidance();
            }
        } catch (error) {
            console.error('讀取全域分析間隔失敗:', error);
            updateIntervalGuidance();
        }
    }

    async function saveGlobalCaptureInterval(minutes) {
        const parsed = parseCaptureIntervalMinutes(minutes);
        if (parsed === null) {
            return false;
        }
        try {
            const response = await fetch('/api/stream/global_interval', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    capture_interval_minutes: parsed
                }),
            });
            const result = await response.json();
            if (!result.success) {
                showAlert(result.error || '更新全域分析間隔失敗', 'danger');
                return false;
            }
            setCaptureIntervalInput(parsed);
            return true;
        } catch (error) {
            showAlert('更新全域分析間隔失敗: ' + error.message, 'danger');
            return false;
        }
    }

    window.updateModelOptions = function() {
        const provider = document.getElementById('provider-select').value;
        const modelSelect = document.getElementById('model-select');
        modelSelect.innerHTML = ''; // 清空選項

        if (modelsByProvider[provider]) {
            modelsByProvider[provider].forEach(model => {
                const option = document.createElement('option');
                option.value = model.value;
                option.textContent = model.label;
                modelSelect.appendChild(option);
            });
        }
        updateApiKeyField();
    };

    updateModelOptions();
    updateIntervalGuidance();
    fetchGlobalCaptureInterval();

    if (captureIntervalInput) {
        captureIntervalInput.addEventListener('input', updateIntervalGuidance);
    }

    if (globalIntervalSettingsBtn) {
        globalIntervalSettingsBtn.addEventListener('click', openGlobalIntervalModal);
    }

    if (closeGlobalIntervalModalBtn) {
        closeGlobalIntervalModalBtn.addEventListener('click', closeGlobalIntervalModal);
    }

    if (globalIntervalModal) {
        globalIntervalModal.addEventListener('click', (event) => {
            if (event.target === globalIntervalModal) {
                closeGlobalIntervalModal();
            }
        });
    }

    if (saveGlobalIntervalBtn) {
        saveGlobalIntervalBtn.addEventListener('click', async () => {
            const selectedMinutes = parseCaptureIntervalMinutes(captureIntervalInput.value);
            if (selectedMinutes === null) {
                captureIntervalInput.classList.add('is-invalid');
                return;
            }
            captureIntervalInput.classList.remove('is-invalid');
            const success = await saveGlobalCaptureInterval(selectedMinutes);
            if (success) {
                closeGlobalIntervalModal();
            } else {
                setCaptureIntervalInput(globalCaptureIntervalMinutes);
            }
        });
    }

    // 載入記住的 API Key
    const initApiKeyInput = document.getElementById('api-key');
    const savedApiKey = localStorage.getItem('openrouter_api_key');

    if (savedApiKey) {
        if (initApiKeyInput) initApiKeyInput.value = savedApiKey;
    }

    // === 相機範本管理邏輯 ===
    let selectedTemplateId = null;
    let templatesCache = {}; // template_id -> template_data

    const templatesListContainer = document.getElementById('templates-list');
    const saveTemplateBtn = document.getElementById('save-template-btn');
    const deleteTemplateBtn = document.getElementById('delete-template-btn');

    // 獲取並渲染範本列表
    async function fetchAndRenderTemplates() {
        if (!templatesListContainer) return;
        try {
            const response = await fetch('/api/stream/templates');
            const result = await response.json();
            if (result.success) {
                renderTemplates(result.templates);
            } else {
                console.error('載入範本失敗:', result.error);
                templatesListContainer.innerHTML = `<span class="text-danger" style="font-size: 0.85rem;">載入範本失敗</span>`;
            }
        } catch (error) {
            console.error('載入範本出錯:', error);
            templatesListContainer.innerHTML = `<span class="text-danger" style="font-size: 0.85rem;">載入範本出錯</span>`;
        }
    }

    // 渲染範本按鈕
    function renderTemplates(templates) {
        templatesCache = {};
        if (!templates || templates.length === 0) {
            templatesListContainer.innerHTML = `<span class="text-muted" style="font-size: 0.85rem; padding-left: 5px;">尚無儲存的範本</span>`;
            if (deleteTemplateBtn) deleteTemplateBtn.style.display = 'none';
            return;
        }

        templatesListContainer.innerHTML = '';
        templates.forEach(t => {
            templatesCache[t.template_id] = t;
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = `btn btn-sm ${selectedTemplateId === t.template_id ? 'btn-primary' : 'btn-outline-primary'}`;
            btn.style.borderRadius = '20px';
            btn.style.fontWeight = '600';
            btn.style.padding = '4px 12px';
            btn.style.fontSize = '0.8rem';
            btn.style.margin = '2px';
            btn.textContent = t.template_id;
            
            btn.addEventListener('click', () => selectTemplate(t.template_id));
            templatesListContainer.appendChild(btn);
        });
    }

    // 選擇特定範本
    function selectTemplate(templateId) {
        // 如果點擊已選中的範本，則取消選取
        if (selectedTemplateId === templateId) {
            selectedTemplateId = null;
            resetFormFields();
            renderTemplates(Object.values(templatesCache));
            return;
        }

        selectedTemplateId = templateId;
        const t = templatesCache[templateId];
        if (!t) return;

        // 填入表單欄位
        document.getElementById('camera-name').value = t.camera_name || '';
        document.getElementById('rtsp-url').value = t.rtsp_url || '';
        
        const providerSelect = document.getElementById('provider-select');
        providerSelect.value = t.provider || 'openrouter';
        
        // 更新模型選項
        window.updateModelOptions();
        
        const modelSelect = document.getElementById('model-select');
        modelSelect.value = t.model || '';
        
        document.getElementById('api-key').value = t.api_key || '';
        document.getElementById('capture-interval-minutes').value = String(globalCaptureIntervalMinutes);
        updateIntervalGuidance();

        // 顯示刪除按鈕
        if (deleteTemplateBtn) deleteTemplateBtn.style.display = 'block';

        // 重新渲染範本按鈕以更新選中狀態
        renderTemplates(Object.values(templatesCache));
    }

    // 重置表單欄位
    function resetFormFields() {
        document.getElementById('camera-name').value = '';
        document.getElementById('rtsp-url').value = '';
        document.getElementById('capture-interval-minutes').value = String(globalCaptureIntervalMinutes);
        document.getElementById('provider-select').value = 'openrouter';
        window.updateModelOptions();
        document.getElementById('api-key').value = localStorage.getItem('openrouter_api_key') || '';
        updateIntervalGuidance();
        if (deleteTemplateBtn) deleteTemplateBtn.style.display = 'none';
        selectedTemplateId = null;
    }

    // 儲存範本按鈕事件
    if (saveTemplateBtn) {
        saveTemplateBtn.addEventListener('click', async () => {
            const cameraName = document.getElementById('camera-name').value.trim();
            const rtspUrl = document.getElementById('rtsp-url').value.trim();
            const provider = document.getElementById('provider-select').value;
            const model = document.getElementById('model-select').value;
            const apiKey = document.getElementById('api-key').value.trim();
            const intervalMinutesInput = document.getElementById('capture-interval-minutes');
            const captureIntervalMinutes = parseCaptureIntervalMinutes(intervalMinutesInput.value);

            if (!cameraName || !rtspUrl) {
                showAlert('請先填寫攝影機名稱與 RTSP URL 才能儲存為範本。', 'warning');
                return;
            }
            if (captureIntervalMinutes === null) {
                intervalMinutesInput.classList.add('is-invalid');
                showAlert('分析間隔必須為 1~60 的整數分鐘。', 'warning');
                return;
            }
            intervalMinutesInput.classList.remove('is-invalid');

            // 提示輸入範本名稱
            const defaultName = selectedTemplateId || cameraName || 'camera-1';
            const templateIdInput = prompt('請輸入範本名稱（例如：camera-1）：', defaultName);
            if (templateIdInput === null) return; // 取消

            const templateId = templateIdInput.trim();
            if (!templateId) {
                showAlert('範本名稱不能為空。', 'danger');
                return;
            }

            try {
                const response = await fetch('/api/stream/templates', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        template_id: templateId,
                        camera_name: cameraName,
                        rtsp_url: rtspUrl,
                        provider: provider,
                        model: model,
                        api_key: apiKey,
                        capture_interval_minutes: captureIntervalMinutes
                    })
                });
                const result = await response.json();
                if (result.success) {
                    showAlert(result.message, 'success');
                    
                    // 儲存 API Key 到 localStorage (如果是 OpenRouter 且有輸入)
                    if (provider === 'openrouter' && apiKey) {
                        localStorage.setItem('openrouter_api_key', apiKey);
                    }

                    selectedTemplateId = templateId;
                    await fetchAndRenderTemplates();
                } else {
                    showAlert(result.error, 'danger');
                }
            } catch (error) {
                showAlert('儲存範本失敗: ' + error.message, 'danger');
            }
        });
    }

    // 刪除範本按鈕事件
    if (deleteTemplateBtn) {
        deleteTemplateBtn.addEventListener('click', async () => {
            if (!selectedTemplateId) return;
            if (!confirm(`確定要刪除範本 [${selectedTemplateId}] 嗎？`)) return;

            try {
                const response = await fetch(`/api/stream/templates/${encodeURIComponent(selectedTemplateId)}`, {
                    method: 'DELETE'
                });
                const result = await response.json();
                if (result.success) {
                    showAlert(result.message, 'success');
                    resetFormFields();
                    await fetchAndRenderTemplates();
                } else {
                    showAlert(result.error, 'danger');
                }
            } catch (error) {
                showAlert('刪除範本失敗: ' + error.message, 'danger');
            }
        });
    }

    // 初始化範本列表
    fetchAndRenderTemplates();

    function showAlert(message, type = 'danger') {
        errorMessage.textContent = message;
        errorAlert.className = `alert alert-${type} mt-4`;
        errorAlert.classList.remove('d-none');
        
        // 滾動到警告框，確保使用者能看見
        errorAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });

        const hideMs = type === 'danger' ? 15000 : 8000;
        setTimeout(() => {
            errorAlert.classList.add('d-none');
        }, hideMs);
    }

    // 渲染單個串流卡片
    function renderStreamCard(stream) {
        const isRunning = stream.current_status;
        const statusBadgeClass = isRunning ? 'bg-success' : 'bg-danger';
        const statusText = isRunning ? '運行中' : '已停止';
        const cardClass = isRunning ? 'active' : 'inactive';
        const captureIntervalMinutes = getIntervalMinutesFromStream(stream);
        const captureIntervalSeconds = captureIntervalMinutes * 60;

        const cardHtml = `
            <div class="col-md-6 stream-card-wrapper" 
                 style="padding: 10px;"
                 data-session-id="${stream.session_id}" 
                 data-camera-name="${stream.camera_name}"
                 data-rtsp-url="${escapeAttr(stream.rtsp_url || '')}"
                 data-llm-model="${escapeAttr(stream.llm_model || stream.model || '')}"
                 data-capture-interval-minutes="${captureIntervalMinutes}"
                 data-capture-interval-seconds="${captureIntervalSeconds}"
                 data-task-name="${escapeAttr(stream.task_name || '')}">
                <div class="stream-card ${cardClass}" style="margin-bottom: 0; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h5 class="card-title mb-0" style="font-weight: 700; font-size: 1rem; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                            ${stream.camera_name}
                            ${stream.task_name ? `<span class="badge bg-secondary" style="font-size: 0.7rem; font-weight: 500; padding: 3px 6px;">📋 ${stream.task_name}</span>` : ''}
                            <a href="/details?session_id=${stream.session_id}&tab=stream" class="btn btn-xs btn-outline-primary" title="查看此鏡頭的歷史資料" style="padding: 1px 6px; font-size: 0.7rem; border-radius: 4px; text-decoration: none; display: inline-flex; align-items: center; gap: 3px; line-height: 1.2;">
                                📊 歷史資料
                            </a>
                        </h5>
                        <span class="badge ${statusBadgeClass} badge-custom" style="font-size: 0.7rem; padding: 4px 8px;">${statusText}</span>
                    </div>
                    
                    <div class="stream-info-item" style="margin-bottom: 5px;">
                        <span class="stream-info-label" style="width: 60px; font-size: 0.8rem;">RTSP:</span>
                        <span class="stream-info-value" style="font-size: 0.7rem; max-width: 120px; overflow: hidden; text-overflow: ellipsis; display: inline-block; vertical-align: bottom;">${stream.rtsp_url}</span>
                    </div>
                    <div class="stream-info-item" style="margin-bottom: 5px;">
                        <span class="stream-info-label" style="width: 60px; font-size: 0.8rem;">間隔:</span>
                        <span class="stream-info-value" style="font-size: 0.7rem;">${captureIntervalMinutes} 分鐘</span>
                    </div>
                    <div class="stream-info-item" style="margin-bottom: 5px;">
                        <span class="stream-info-label" style="width: 60px; font-size: 0.8rem;">模式:</span>
                        <span class="stream-info-value" style="font-size: 0.7rem;">整分鐘起始</span>
                    </div>

                    <div class="mt-3 d-flex justify-content-end gap-2">
                        ${isRunning ? `
                            <button class="btn btn-outline-danger btn-sm stop-stream-btn" 
                                    style="border-radius: 6px; font-weight: 600; padding: 3px 10px; font-size: 0.75rem;"
                                    data-session-id="${stream.session_id}">
                                停止
                            </button>
                        ` : `
                            <button class="btn btn-outline-success btn-sm start-stream-btn" 
                                    style="border-radius: 6px; font-weight: 600; padding: 3px 10px; font-size: 0.75rem;"
                                    data-session-id="${stream.session_id}">
                                啟動
                            </button>
                            <button class="btn btn-outline-secondary btn-sm delete-stream-btn" 
                                    style="border-radius: 6px; font-weight: 600; padding: 3px 10px; font-size: 0.75rem;"
                                    data-session-id="${stream.session_id}">
                                移除卡面
                            </button>
                        `}
                    </div>
                </div>
            </div>
        `;
        return cardHtml;
    }

    function getStreamAnalyses(stream) {
        const analyses = stream.latest_analyses || (stream.latest_analysis ? [stream.latest_analysis] : []);
        
        // 只有在串流「運行中」時，才需要過濾掉啟動前的舊資料
        if (stream.current_status) {
            const lastKnownTimestamp = streamStartTimes[stream.session_id];
            if (lastKnownTimestamp && lastKnownTimestamp !== 'none') {
                // 過濾掉與啟動前最後已知時間相同的舊分析
                return analyses.filter(analysis => analysis.analyzed_at !== lastKnownTimestamp);
            }
        }
        return analyses;
    }

    function upsertStreamInCache(stream) {
        if (!stream || !stream.session_id) return;
        streamAnalysisCache[stream.session_id] = {
            session_id: stream.session_id,
            camera_name: stream.camera_name,
            rtsp_url: stream.rtsp_url,
            task_name: stream.task_name,
            current_status: stream.current_status,
            capture_interval_seconds: stream.capture_interval_seconds,
            capture_interval_minutes: stream.capture_interval_minutes,
            latest_analysis: stream.latest_analysis,
            latest_analyses: stream.latest_analyses
        };
    }

    function removeStreamFromCache(sessionId) {
        delete streamAnalysisCache[sessionId];
    }

    const analysisPanelPlaceholderHtml = `
        <div class="no-data-placeholder">
            <div class="no-data-icon">📡</div>
            <p>目前尚無分析數據</p>
            <p style="font-size: 0.85rem;">啟動監測後，系統將自動擷取畫面並進行分析</p>
        </div>
    `;

    function escapeAttr(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderAnalysisThumb(analysis) {
        if (!analysis.screen_image_url) {
            return '<div class="analysis-no-image-compact">無影像</div>';
        }
        const imgUrl = escapeAttr(analysis.screen_image_url);
        return `
            <a href="${imgUrl}" target="_blank" rel="noopener" class="d-block" title="點擊查看原圖">
                <img src="${imgUrl}" alt="分析截圖" class="latest-analysis-image">
            </a>
            <a href="${imgUrl}" target="_blank" rel="noopener" class="analysis-thumb-link">查看原圖</a>
        `;
    }

    function renderStreamAnalysisSection(stream) {
        const analyses = getStreamAnalyses(stream);
        const captureIntervalMinutes = getIntervalMinutesFromStream(stream);
        if (analyses.length === 0) {
            if (stream.current_status) {
                const loadingHtml = `
                    <div class="analysis-entry-compact loading-placeholder-entry" style="border: 1px dashed var(--primary-color); background: rgba(0, 123, 255, 0.02); padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 10px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                        <div class="spinner"></div>
                        <p class="text-primary mb-1" style="font-weight: 600; font-size: 0.9rem; letter-spacing: 0.5px; color: var(--primary-color) !important;">📡 正在進行首次畫面擷取與 AI 分析...</p>
                        <p class="text-muted mb-0" style="font-size: 0.75rem; color: #718096;">首次分析會在下一個整分鐘開始，之後每隔 ${captureIntervalMinutes} 分鐘擷取並辨識，請稍候 (約需 10-15 秒)</p>
                    </div>
                `;
                return `
                    <div class="camera-analysis-group">
                        <h6 class="camera-analysis-title" style="display: flex; justify-content: space-between; align-items: center;">
                            <span>📷 ${stream.camera_name}</span>
                            <a href="/details?session_id=${stream.session_id}&tab=stream" class="btn btn-xs btn-outline-primary" title="查看此鏡頭的歷史資料" style="padding: 1px 6px; font-size: 0.7rem; border-radius: 4px; text-decoration: none; display: inline-flex; align-items: center; gap: 3px; line-height: 1.2;">
                                📊 歷史資料
                            </a>
                        </h6>
                        ${loadingHtml}
                    </div>
                `;
            }
            return '';
        }

        let entriesHtml = '';
        analyses.forEach((analysis, index) => {
            const medicalValuesHtml = Object.entries(analysis.medical_values || {})
                .map(([key, value]) => `
                    <div class="analysis-item">
                        <span class="analysis-label">${key}</span>
                        <span class="analysis-value ${value === null ? 'null' : ''}">${value !== null ? value : '--'}</span>
                    </div>`)
                .join('');

            const screenLabel = analyses.length > 1 ? ` (螢幕 ${analysis.screen_number || index + 1})` : '';
            const analyzedAt = analysis.analyzed_at || '未知';
            const shortTime = analyzedAt.length > 19 ? analyzedAt.slice(0, 19).replace('T', ' ') : analyzedAt;

            entriesHtml += `
                <div class="analysis-entry-compact">
                    <div class="analysis-entry-header">
                        <span class="badge bg-primary badge-custom" style="background: var(--primary-color) !important;">來源: ${stream.camera_name}${screenLabel}</span>
                        <span class="analysis-entry-time">${shortTime}</span>
                    </div>
                    <div class="analysis-entry-body">
                        <div class="analysis-entry-metrics">
                            <div class="analysis-grid-compact">
                                ${medicalValuesHtml || '<div class="text-muted" style="font-size:0.75rem;">無數值</div>'}
                            </div>
                        </div>
                        <div class="analysis-entry-thumb">
                            ${renderAnalysisThumb(analysis)}
                        </div>
                    </div>
                </div>
            `;
        });

        return `
            <div class="camera-analysis-group">
                <h6 class="camera-analysis-title" style="display: flex; justify-content: space-between; align-items: center;">
                    <span>📷 ${stream.camera_name}</span>
                    <a href="/details?session_id=${stream.session_id}&tab=stream" class="btn btn-xs btn-outline-primary" title="查看此鏡頭的歷史資料" style="padding: 1px 6px; font-size: 0.7rem; border-radius: 4px; text-decoration: none; display: inline-flex; align-items: center; gap: 3px; line-height: 1.2;">
                        📊 歷史資料
                    </a>
                </h6>
                ${entriesHtml}
            </div>
        `;
    }

    // 彙總所有鏡頭的最新分析結果至下方面板
    function refreshGlobalAnalysisPanel() {
        const latestAnalysisContent = document.getElementById('latest-analysis-content');
        if (!latestAnalysisContent) return;

        const streams = Object.values(streamAnalysisCache)
            .filter((s) => s.current_status || getStreamAnalyses(s).length > 0)
            .sort((a, b) => String(a.camera_name).localeCompare(String(b.camera_name), 'zh-Hant'));

        if (streams.length === 0) {
            latestAnalysisContent.innerHTML = analysisPanelPlaceholderHtml;
            return;
        }

        let html = '';
        streams.forEach((stream) => {
            html += renderStreamAnalysisSection(stream);
        });
        latestAnalysisContent.innerHTML = html;
    }

    function updateStreamAnalysisState(stream) {
        upsertStreamInCache(stream);
        refreshGlobalAnalysisPanel();
    }

    // 獲取並顯示所有監測串流（包含活躍與已停止）
    async function fetchAndRenderStreams() {
        const currentFetchId = ++lastFetchId;
        try {
            const response = await fetch('/api/stream/list');
            const result = await response.json();

            if (currentFetchId !== lastFetchId) return; // 有更新的請求已發出，捨棄此舊請求的結果

            if (result.success) {
                const streamCountBadge = document.getElementById('stream-count-badge');
                
                if (result.streams.length === 0) {
                    activeStreamsContainer.innerHTML = '';
                    streamAnalysisCache = {};
                    streamCache = {};
                    noStreamsMessage.classList.remove('d-none');
                    if (streamCountBadge) streamCountBadge.classList.add('d-none');
                    updateIntervalGuidance();
                    refreshGlobalAnalysisPanel();
                } else {
                    noStreamsMessage.classList.add('d-none');
                    if (streamCountBadge) {
                        // 計算真正處於活躍（運行中）狀態的串流數量
                        const activeCount = result.streams.filter(s => s.current_status).length;
                        streamCountBadge.textContent = `${activeCount} 個活躍`;
                        streamCountBadge.classList.remove('d-none');
                    }
                    
                    const renderedCards = [];
                    const tempStreamCache = {};
                    const tempStreamAnalysisCache = {};
                    
                    for (const stream of result.streams) {
                        try {
                            const statusResponse = await fetch(`/api/stream/status/${stream.session_id}`);
                            const statusResult = await statusResponse.json();
                            if (currentFetchId !== lastFetchId) return; // 再次檢查，避免在 await 期間有新請求
                            
                            if (statusResult.success) {
                                stream.latest_analysis = statusResult.status.latest_analysis;
                                stream.latest_analyses = statusResult.status.latest_analyses;
                                stream.total_analysis_count = statusResult.status.total_analysis_count;
                                stream.current_status = statusResult.status.is_running_in_memory;
                            }
                        } catch (e) {
                            console.error(`獲取串流 ${stream.session_id} 狀態失敗:`, e);
                        }
                        tempStreamCache[stream.session_id] = stream;
                        tempStreamAnalysisCache[stream.session_id] = {
                            session_id: stream.session_id,
                            camera_name: stream.camera_name,
                            rtsp_url: stream.rtsp_url,
                            task_name: stream.task_name,
                            current_status: stream.current_status,
                            capture_interval_seconds: stream.capture_interval_seconds,
                            capture_interval_minutes: stream.capture_interval_minutes,
                            latest_analysis: stream.latest_analysis,
                            latest_analyses: stream.latest_analyses
                        };
                        renderedCards.push({
                            sessionId: stream.session_id,
                            currentStatus: stream.current_status,
                            html: renderStreamCard(stream)
                        });
                    }
                    
                    if (currentFetchId !== lastFetchId) return; // 最後一次檢查
                    
                    // 一次性寫入 DOM，完全避免重複嵌套與並行呼叫產生的重複卡片問題
                    activeStreamsContainer.innerHTML = renderedCards.map(c => c.html).join('');
                    
                    // 更新全域快取
                    streamCache = { ...streamCache, ...tempStreamCache };
                    streamAnalysisCache = tempStreamAnalysisCache;
                    updateIntervalGuidance();
                    
                    // 根據狀態綁定對應的按鈕事件
                    renderedCards.forEach(c => {
                        if (c.currentStatus) {
                            bindStopButtonEvent(c.sessionId);
                            startStreamPolling(c.sessionId);
                        } else {
                            bindStartButtonEvent(c.sessionId);
                            bindDeleteButtonEvent(c.sessionId);
                        }
                    });
                    
                    refreshGlobalAnalysisPanel();
                }
            } else {
                showAlert(result.error);
            }
        } catch (error) {
            if (currentFetchId === lastFetchId) {
                showAlert('獲取串流列表失敗: ' + error.message);
            }
        }
    }

    // 啟動單個串流的狀態輪詢
    function startStreamPolling(sessionId) {
        // 如果已經有定時器，先清除
        if (streamPollingIntervals[sessionId]) {
            clearInterval(streamPollingIntervals[sessionId]);
        }
        streamPollingIntervals[sessionId] = setInterval(async () => {
            try {
                const response = await fetch(`/api/stream/status/${sessionId}`);
                const result = await response.json();

                if (result.success) {
                    const streamCardWrapper = document.querySelector(`[data-session-id="${sessionId}"]`);
                    if (streamCardWrapper) {
                        // 更新卡片內容，例如狀態標籤
                        const stream = { 
                            ...(streamCache[sessionId] || {}), // 保留原有數據
                            session_id: sessionId, // 確保 session_id 存在
                            camera_name: streamCardWrapper.dataset.cameraName || (streamCache[sessionId] || {}).camera_name, // 確保 camera_name 存在
                            rtsp_url: streamCardWrapper.dataset.rtspUrl || (streamCache[sessionId] || {}).rtsp_url,
                            llm_model: streamCardWrapper.dataset.llmModel || (streamCache[sessionId] || {}).llm_model,
                            capture_interval_minutes: Number(streamCardWrapper.dataset.captureIntervalMinutes || (streamCache[sessionId] || {}).capture_interval_minutes),
                            capture_interval_seconds: Number(streamCardWrapper.dataset.captureIntervalSeconds || (streamCache[sessionId] || {}).capture_interval_seconds),
                            task_name: streamCardWrapper.dataset.taskName || (streamCache[sessionId] || {}).task_name,
                            current_status: result.status.is_running_in_memory,
                            latest_analysis: result.status.latest_analysis,
                            latest_analyses: result.status.latest_analyses,
                            total_analysis_count: result.status.total_analysis_count
                        };
                        streamCache[sessionId] = stream; // 更新全域快取
                        // 重新渲染整個卡片 (使用 outerHTML 避免 DOM 重複嵌套)
                        streamCardWrapper.outerHTML = renderStreamCard(stream);
                        
                        // 根據狀態重新綁定事件
                        if (stream.current_status) {
                            bindStopButtonEvent(sessionId);
                        } else {
                            bindStartButtonEvent(sessionId);
                            bindDeleteButtonEvent(sessionId);
                        }

                        // 顯示後端傳回的錯誤 (例如 API Key 錯誤)
                        if (result.status.last_error) {
                            showAlert(`串流 [${stream.camera_name}] 發生錯誤: ${result.status.last_error}`);
                        }

                        // 更新下方面板（彙總所有鏡頭）
                        updateStreamAnalysisState(stream);

                        // 如果串流停止了，清除輪詢
                        if (!result.status.is_running_in_memory) {
                            clearInterval(streamPollingIntervals[sessionId]);
                            delete streamPollingIntervals[sessionId];
                            // 停止時保留在快取中，這樣下方的最新分析結果不會消失
                            refreshGlobalAnalysisPanel();
                            // 刷新整個列表以確保狀態正確更新
                            fetchAndRenderStreams(); 
                        }
                    }
                } else {
                    showAlert('獲取串流狀態失敗: ' + result.error);
                    clearInterval(streamPollingIntervals[sessionId]);
                    delete streamPollingIntervals[sessionId];
                    fetchAndRenderStreams();
                }
            } catch (error) {
                showAlert('輪詢串流狀態失敗: ' + error.message);
                clearInterval(streamPollingIntervals[sessionId]);
                delete streamPollingIntervals[sessionId];
                fetchAndRenderStreams();
            }
        }, 5000); // 每 5 秒輪詢一次
    }

    // 停止監測按鈕事件處理
    function bindStopButtonEvent(sessionId) {
        const stopButton = document.querySelector(`.stop-stream-btn[data-session-id="${sessionId}"]`);
        if (stopButton) {
            stopButton.onclick = async () => {
                const confirmStop = confirm('確定要停止這個串流監測嗎？');
                if (!confirmStop) return;

                stopButton.disabled = true;
                stopButton.textContent = '停止中...';
                try {
                    const response = await fetch(`/api/stream/stop/${sessionId}`, {
                        method: 'POST',
                    });
                    const result = await response.json();

                    if (result.success) {
                        showAlert(result.message, 'success');
                        clearInterval(streamPollingIntervals[sessionId]);
                        delete streamPollingIntervals[sessionId];
                        // 停止時保留在快取中，以便在下方顯示最後的分析數據
                        fetchAndRenderStreams(); // 重新載入列表
                    } else {
                        showAlert(result.error);
                        stopButton.disabled = false;
                        stopButton.textContent = '停止';
                    }
                } catch (error) {
                    showAlert('停止監測失敗: ' + error.message);
                    stopButton.disabled = false;
                    stopButton.textContent = '停止';
                }
            };
        }
    }

    // 啟動監測按鈕事件處理（用於重啟已停止的卡片）
    function bindStartButtonEvent(sessionId) {
        const startButton = document.querySelector(`.start-stream-btn[data-session-id="${sessionId}"]`);
        if (startButton) {
            startButton.onclick = async () => {
                const streamCardWrapper = document.querySelector(`[data-session-id="${sessionId}"]`);
                if (!streamCardWrapper) return;
                
                // 優先從全域快取獲取，其次從 DOM dataset 獲取，確保數據不遺失
                const stream = {
                    ...(streamCache[sessionId] || {}),
                    camera_name: streamCardWrapper.dataset.cameraName || (streamCache[sessionId] || {}).camera_name,
                    rtsp_url: streamCardWrapper.dataset.rtspUrl || (streamCache[sessionId] || {}).rtsp_url,
                    llm_model: streamCardWrapper.dataset.llmModel || (streamCache[sessionId] || {}).llm_model,
                    capture_interval_minutes: Number(streamCardWrapper.dataset.captureIntervalMinutes || (streamCache[sessionId] || {}).capture_interval_minutes),
                    capture_interval_seconds: Number(streamCardWrapper.dataset.captureIntervalSeconds || (streamCache[sessionId] || {}).capture_interval_seconds),
                    task_name: streamCardWrapper.dataset.taskName || (streamCache[sessionId] || {}).task_name
                };
                const captureIntervalMinutes = parseCaptureIntervalMinutes(captureIntervalInput?.value) ?? globalCaptureIntervalMinutes;
                const captureIntervalSeconds = captureIntervalMinutes * 60;
                
                // 獲取 API Key（優先從輸入框獲取，其次從 localStorage 載入）
                const apiKeyInput = document.getElementById('api-key');
                let apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
                
                if (!apiKey) {
                    apiKey = localStorage.getItem('openrouter_api_key') || '';
                }
                
                // 判斷是否需要 API Key
                const provider = stream.provider || (stream.llm_model && stream.llm_model.includes('/') ? 'openrouter' : 'local');
                if (provider === 'openrouter' && !apiKey) {
                    showAlert('請先在左側表單中輸入 OpenRouter API Key，再點擊啟動！', 'danger');
                    if (apiKeyInput) {
                        apiKeyInput.focus();
                        apiKeyInput.classList.add('is-invalid');
                    }
                    return;
                }

                startButton.disabled = true;
                startButton.textContent = '啟動中...';
                
                try {
                    const response = await fetch('/api/stream/start', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            session_id: sessionId, // 傳遞現有的 session_id，重複使用
                            camera_name: stream.camera_name,
                            rtsp_url: stream.rtsp_url,
                            api_key: apiKey,
                            provider: provider,
                            model: stream.llm_model || stream.model,
                            capture_interval_minutes: captureIntervalMinutes,
                            capture_interval_seconds: captureIntervalSeconds
                        }),
                    });
                    const result = await response.json();

                    if (result.success) {
                        showAlert(result.message, 'success');
                        
                        // 記錄啟動時的最後已知分析時間戳記，避免顯示上一次運行的舊資料
                        const currentStream = streamCache[sessionId] || {};
                        const lastAnalysis = currentStream.latest_analysis;
                        streamStartTimes[sessionId] = lastAnalysis ? lastAnalysis.analyzed_at : 'none';
                        
                        fetchAndRenderStreams(); // 重新載入列表，卡片將變為「運行中」
                    } else {
                        showAlert(result.error);
                        startButton.disabled = false;
                        startButton.textContent = '啟動';
                    }
                } catch (error) {
                    showAlert('啟動監測失敗: ' + error.message);
                    startButton.disabled = false;
                    startButton.textContent = '啟動';
                }
            };
        }
    }

    // 刪除監測會話按鈕事件處理
    function bindDeleteButtonEvent(sessionId) {
        const deleteButton = document.querySelector(`.delete-stream-btn[data-session-id="${sessionId}"]`);
        if (deleteButton) {
            deleteButton.onclick = async () => {
                const confirmDelete = confirm('確定要從此面板移除此鏡頭卡面嗎？\n（此操作僅會隱藏卡面，該鏡頭的所有歷史分析數據仍會完整保留在資料庫中）');
                if (!confirmDelete) return;

                deleteButton.disabled = true;
                deleteButton.textContent = '移除中...';
                try {
                    const response = await fetch(`/api/stream/delete/${sessionId}`, {
                        method: 'POST',
                    });
                    const result = await response.json();

                    if (result.success) {
                        showAlert(result.message, 'success');
                        removeStreamFromCache(sessionId); // 從快取中移除，使下方分析面板也同步移除
                        fetchAndRenderStreams(); // 重新載入列表
                    } else {
                        showAlert(result.error);
                        deleteButton.disabled = false;
                        deleteButton.textContent = '移除卡面';
                    }
                } catch (error) {
                    showAlert('移除卡面失敗: ' + error.message);
                    deleteButton.disabled = false;
                    deleteButton.textContent = '移除卡面';
                }
            };
        }
    }

    // 表單提交處理
    addStreamForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const taskNameInput = document.getElementById('task-name');
        const cameraNameInput = document.getElementById('camera-name');
        const rtspUrlInput = document.getElementById('rtsp-url');
        const captureIntervalInput = document.getElementById('capture-interval-minutes');
        const apiKeyInput = document.getElementById('api-key');
        const providerSelect = document.getElementById('provider-select');
        const modelSelect = document.getElementById('model-select');

        const taskName = taskNameInput.value.trim();
        const cameraName = cameraNameInput.value.trim();
        const rtspUrl = rtspUrlInput.value.trim();
        const captureIntervalMinutes = parseCaptureIntervalMinutes(captureIntervalInput.value);
        const apiKey = apiKeyInput.value.trim();
        const provider = providerSelect.value;
        const modelName = modelSelect.value;

        // 重置驗證狀態
        [taskNameInput, cameraNameInput, rtspUrlInput, captureIntervalInput, apiKeyInput].forEach(el => el.classList.remove('is-invalid'));

        let hasError = false;

        // 0. 驗證任務名稱
        if (!taskName) {
            taskNameInput.classList.add('is-invalid');
            hasError = true;
        }

        // 1. 驗證攝影機名稱
        if (!cameraName) {
            cameraNameInput.classList.add('is-invalid');
            hasError = true;
        }

        // 2. 驗證 RTSP URL
        if (!rtspUrl || (!rtspUrl.startsWith('rtsp://') && !rtspUrl.startsWith('rtsps://'))) {
            rtspUrlInput.classList.add('is-invalid');
            hasError = true;
        }

        // 3. 驗證分析間隔（1~60 的整數分鐘）
        if (captureIntervalMinutes === null) {
            captureIntervalInput.classList.add('is-invalid');
            hasError = true;
        }

        // 4. 驗證 API Key（OpenRouter 必填）
        if (provider === 'openrouter' && !apiKey) {
            apiKeyInput.classList.add('is-invalid');
            hasError = true;
        }

        if (hasError) {
            return;
        }

        const submitButton = addStreamForm.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        submitButton.textContent = '啟動中...';

        try {
            await saveGlobalCaptureInterval(captureIntervalMinutes);
            const response = await fetch('/api/stream/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    task_name: taskName,
                    camera_name: cameraName,
                    rtsp_url: rtspUrl,
                    capture_interval_minutes: captureIntervalMinutes,
                    capture_interval_seconds: captureIntervalMinutes * 60,
                    api_key: apiKey,
                    provider: provider,
                    model: modelName
                }),
            });
            const result = await response.json();

            if (result.success) {
                showAlert(result.message, 'success');
                
                // 記錄新啟動的會話 ID，避免顯示舊資料
                if (result.session_id) {
                    streamStartTimes[result.session_id] = 'none';
                }

                // 儲存 API Key 到 localStorage (如果是 OpenRouter 且有輸入)
                if (provider === 'openrouter' && apiKey) {
                    localStorage.setItem('openrouter_api_key', apiKey);
                }

                addStreamForm.reset();
                setCaptureIntervalInput(captureIntervalMinutes);
                
                // 重置範本選取狀態
                selectedTemplateId = null;
                if (deleteTemplateBtn) deleteTemplateBtn.style.display = 'none';
                fetchAndRenderTemplates(); // 重新整理範本列表（確保選取狀態更新）

                // 重新載入已儲存的 API Key 到重置後的表單中
                const newApiKeyInput = document.getElementById('api-key');
                const currentSavedApiKey = localStorage.getItem('openrouter_api_key');
                if (currentSavedApiKey) {
                    if (newApiKeyInput) newApiKeyInput.value = currentSavedApiKey;
                }

                fetchAndRenderStreams(); // 重新載入列表
            } else {
                showAlert(result.error);
            }
        } catch (error) {
            showAlert('啟動監測失敗: ' + error.message);
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = '🚀 啟動監測任務';
        }
    });

    // 初始化載入串流列表
    fetchAndRenderStreams();
});