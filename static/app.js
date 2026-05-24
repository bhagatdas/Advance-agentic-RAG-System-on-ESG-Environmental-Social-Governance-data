/* ═══════════════════════════════════════════════════════════
   PdfAgent — Chat Logic
   ═══════════════════════════════════════════════════════════ */

const API_BASE = '';
let currentThreadId = 'session-' + Math.random().toString(36).substring(2, 10);
let detailsVisible = false;

const WELCOME_HTML = `
    <div class="welcome-screen" id="welcomeScreen">
        <div class="welcome-icon">
            <svg viewBox="0 0 48 48" fill="none">
                <rect width="48" height="48" rx="12" fill="url(#wG)"/>
                <g transform="translate(6,6)" fill="none" stroke="white" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 9V3H12"/>
                    <path d="m12 27-6 6V12a3 3 0 0 1 3-3h18a3 3 0 0 1 3 3v12a3 3 0 0 1-3 3Z"/>
                    <path d="M3 18h3"/>
                    <path d="M13.5 16.5v3"/>
                    <path d="M22.5 16.5v3"/>
                    <path d="M30 18h3"/>
                </g>
                <defs><linearGradient id="wG" x1="0" y1="0" x2="48" y2="48"><stop stop-color="#10b981"/><stop offset="1" stop-color="#3b82f6"/></linearGradient></defs>
            </svg>
        </div>
        <h2>What would you like to know from your PDFs?</h2>
        <p>Ask anything about your uploaded documents — themes, data, comparisons, or specific facts. Every answer comes with source citations.</p>
        <div class="welcome-cards">
            <button class="welcome-card" onclick="fillQuery('Summarize the key findings and main conclusions of this document')">
                <div class="welcome-card-title">Quick summary</div>
                <div class="welcome-card-desc">Get the main takeaways from the document</div>
            </button>
            <button class="welcome-card" onclick="fillQuery('Extract all numerical data, metrics, and figures from this document')">
                <div class="welcome-card-title">Pull the numbers</div>
                <div class="welcome-card-desc">Surface every metric, percentage, or figure</div>
            </button>
            <button class="welcome-card" onclick="fillQuery('Compare data across different sections or time periods and explain the changes')">
                <div class="welcome-card-title">Compare &amp; contrast</div>
                <div class="welcome-card-desc">Multi-hop comparison with explanation</div>
            </button>
            <button class="welcome-card" onclick="fillQuery('What are the main themes, risks, and opportunities discussed?')">
                <div class="welcome-card-title">Themes &amp; risks</div>
                <div class="welcome-card-desc">High-level analysis across the whole document</div>
            </button>
        </div>
    </div>`;

// ── Core Functions ───────────────────────────────────────────

function fillQuery(text) {
    const input = document.getElementById('queryInput');
    input.value = text;
    autoResize(input);
    input.focus();
}

function newChat() {
    currentThreadId = 'session-' + Math.random().toString(36).substring(2, 10);
    document.getElementById('messagesContainer').innerHTML = WELCOME_HTML;
    document.getElementById('queryInput').value = '';
    autoResize(document.getElementById('queryInput'));
    // Reset details panel state
    const sourcesList = document.getElementById('sourcesList');
    if (sourcesList) sourcesList.innerHTML = '<p class="empty-state">Submit a query to see sources</p>';
    const traceList = document.getElementById('traceList');
    if (traceList) traceList.innerHTML = '<p class="empty-state">No trace yet</p>';
    const metaGrid = document.getElementById('metaGrid');
    if (metaGrid) metaGrid.innerHTML = '';
    const sqlSection = document.getElementById('detailSQL');
    if (sqlSection) sqlSection.style.display = 'none';
}

function toggleDetails() {
    const panel = document.getElementById('detailsPanel');
    detailsVisible = !detailsVisible;
    panel.style.display = detailsVisible ? 'flex' : 'none';
}

async function submitQuery() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    if (!query) return;

    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    setStatus('Processing…', 'busy');

    // Hide welcome screen
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.remove();

    addMessage('user', query);
    input.value = '';
    autoResize(input);

    const stepsId = addStepList();

    try {
        const response = await fetch(`${API_BASE}/query/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
            body: JSON.stringify({
                query: query,
                thread_id: currentThreadId,
                user_id: 'default',
            }),
        });

        if (!response.ok) {
            removeMessage(stepsId);
            let detail = 'Query failed';
            try { detail = (await response.json()).detail || detail; } catch {}
            throw new Error(detail);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let finalData = null;
        let streamError = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // SSE frames are separated by a blank line (\n\n)
            const frames = buffer.split(/\n\n/);
            buffer = frames.pop() || '';

            for (const frame of frames) {
                if (!frame.trim()) continue;
                let eventName = 'message';
                let dataStr = '';
                for (const line of frame.split('\n')) {
                    if (line.startsWith('event:')) eventName = line.slice(6).trim();
                    else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
                }
                if (!dataStr) continue;
                let payload;
                try { payload = JSON.parse(dataStr); } catch { continue; }

                if (eventName === 'step') {
                    appendStep(stepsId, payload);
                } else if (eventName === 'complete') {
                    finalData = payload;
                } else if (eventName === 'error') {
                    streamError = payload.error || 'Unknown stream error';
                }
            }
        }

        if (streamError) throw new Error(streamError);

        removeMessage(stepsId);
        if (finalData) {
            addAIMessage(finalData);
            updateDetailsPanel(finalData);
        } else {
            addMessage('ai', 'Stream ended without a final response.', true);
        }

    } catch (error) {
        removeMessage(stepsId);
        addMessage('ai', `Error: ${error.message}`, true);
        showToast(error.message, 'error');
    } finally {
        btn.disabled = false;
        setStatus('System Ready', 'online');
        input.focus();
    }
}

function addStepList() {
    const container = document.getElementById('messagesContainer');
    const id = 'steps-' + Date.now();
    const html = `
        <div class="message-row ai" id="${id}">
            <div class="message-inner">
                <div class="message-avatar ai-avatar">P</div>
                <div class="message-content">
                    <div class="message-label">Answer</div>
                    <ul class="step-list"></ul>
                    <div class="step-progress">
                        <span class="typing-indicator"><span></span><span></span><span></span></span>
                        <span class="step-progress-label">Working…</span>
                    </div>
                </div>
            </div>
        </div>`;
    container.insertAdjacentHTML('beforeend', html);
    container.scrollTop = container.scrollHeight;
    return id;
}

function appendStep(rowId, step) {
    const row = document.getElementById(rowId);
    if (!row) return;
    const ul = row.querySelector('.step-list');
    if (!ul) return;

    const tokensChip = step.tokens && step.tokens.total_tokens
        ? `<span class="step-tokens" title="Prompt: ${step.tokens.prompt_tokens} · Completion: ${step.tokens.completion_tokens} · Calls: ${step.tokens.calls}">
              ${formatTokens(step.tokens.total_tokens)} tok
           </span>`
        : '';

    const chunksToggle = (step.node === 'retrieval' && step.chunks_preview && step.chunks_preview.length)
        ? `<button class="chunks-toggle" data-step-id="chunks-${rowId}-${Date.now()}" onclick="toggleChunks(this)">
              ${step.chunks_preview.length} chunk${step.chunks_preview.length === 1 ? '' : 's'} ▾
           </button>`
        : '';

    const li = document.createElement('li');
    li.className = 'step-item';
    li.innerHTML = `
        <div class="step-row">
            <span class="step-check">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
            </span>
            <span class="step-label">${esc(step.label || step.node || 'Step')}</span>
            ${step.summary ? `<span class="step-summary">${esc(step.summary)}</span>` : ''}
            ${tokensChip}
            ${chunksToggle}
        </div>`;

    // Attach the chunks payload to the toggle so it can render on demand
    if (step.node === 'retrieval' && step.chunks_preview) {
        const wrap = document.createElement('div');
        wrap.className = 'chunks-panel';
        wrap.style.display = 'none';
        wrap.innerHTML = renderChunks(step.chunks_preview, step.retrieval_queries || []);
        li.appendChild(wrap);
        // Wire the toggle to this specific panel
        const toggleBtn = li.querySelector('.chunks-toggle');
        if (toggleBtn) toggleBtn.dataset.target = ''; // sibling-based, see toggleChunks
    }

    ul.appendChild(li);

    // Update the bottom progress label to hint at next-likely step
    const progressLabel = row.querySelector('.step-progress-label');
    if (progressLabel) progressLabel.textContent = 'Working on next step…';

    const container = document.getElementById('messagesContainer');
    container.scrollTop = container.scrollHeight;
}

function toggleChunks(btn) {
    // Look for a sibling .chunks-panel inside the same .step-item
    const item = btn.closest('.step-item');
    if (!item) return;
    const panel = item.querySelector('.chunks-panel');
    if (!panel) return;
    const open = panel.style.display !== 'none';
    panel.style.display = open ? 'none' : 'block';
    btn.innerHTML = btn.innerHTML.replace(open ? '▴' : '▾', open ? '▾' : '▴');
}

function renderChunks(chunks, queries) {
    if (!chunks || !chunks.length) return '<p class="empty-state">No chunks retrieved.</p>';

    const queriesHtml = queries && queries.length
        ? `<div class="chunks-queries"><strong>Queries fanned out:</strong> ${queries.map(q => `<code>${esc(q)}</code>`).join(' · ')}</div>`
        : '';

    // Track running rank for children only; parents are rendered as
    // sub-items belonging to the preceding child (so they don't get a rank).
    let childRank = 0;

    const rows = chunks.map((c) => {
        const isParent = c.is_parent_context;
        const score = c.rerank_score ?? c.rrf_score ?? c.dense_score ?? c.bm25_score;
        const scoreLabel = c.rerank_score != null ? 'rerank' :
                           c.rrf_score    != null ? 'rrf'    :
                           c.dense_score  != null ? 'dense'  :
                           c.bm25_score   != null ? 'bm25'   : '';
        const scoreText = score != null ? `${scoreLabel} ${Number(score).toFixed(3)}` : '';
        const typeChip = c.chunk_type ? `<span class="chunk-type chunk-type-${esc(c.chunk_type)}">${esc(c.chunk_type)}</span>` : '';
        const raptorChip = c.raptor_level ? `<span class="chunk-raptor">L${c.raptor_level}</span>` : '';

        if (isParent) {
            const isNeighbor = c.neighbor_expansion;
            const badgeText = isNeighbor ? 'ADJACENT PAGE ←→' : 'PARENT CONTEXT ↑';
            const itemClass = isNeighbor ? 'chunk-item chunk-parent-item chunk-neighbor-item' : 'chunk-item chunk-parent-item';
            const badgeClass = isNeighbor ? 'chunk-parent-badge chunk-neighbor-badge' : 'chunk-parent-badge';
            const noteText = isNeighbor
                ? `${c.content_length} chars total — pulled in because hybrid search hit this adjacent page`
                : `${c.content_length} chars total — expanded for reasoning context`;
            return `
                <div class="${itemClass}">
                    <div class="chunk-head">
                        <span class="${badgeClass}">${badgeText}</span>
                        <span class="chunk-doc">${esc(c.document || 'Unknown')}</span>
                        <span class="chunk-page">p.${esc(String(c.page ?? '?'))}</span>
                        ${typeChip}
                        <span class="chunk-score">${esc(scoreText)}</span>
                    </div>
                    <div class="chunk-parent-id">parent id: <code>${esc(c.chunk_id || c.parent_id || '?')}</code></div>
                    <div class="chunk-body chunk-parent-body">${esc(c.content_preview || '')}</div>
                    <div class="chunk-meta">${noteText}</div>
                </div>`;
        }

        childRank += 1;

        // If this is a table_repr chunk and we have structured rows, render
        // them as a real table + offer a JSON toggle. The original text blob
        // (what the LLM actually receives) is still available under a toggle.
        const tableBlock = (c.chunk_type === 'table_repr' && c.table_data)
            ? renderTableChunk(c.table_data, c.content_preview)
            : `<div class="chunk-body">${esc(c.content_preview || '')}</div>`;

        return `
            <div class="chunk-item chunk-child-item">
                <div class="chunk-head">
                    <span class="chunk-rank">#${childRank}</span>
                    <span class="chunk-doc">${esc(c.document || 'Unknown')}</span>
                    <span class="chunk-page">p.${esc(String(c.page ?? '?'))}</span>
                    ${typeChip}
                    ${raptorChip}
                    <span class="chunk-score">${esc(scoreText)}</span>
                </div>
                ${c.parent_id ? `<div class="chunk-parent-id">parent: <code>${esc(c.parent_id)}</code></div>` : ''}
                ${tableBlock}
                <div class="chunk-meta">${c.content_length} chars total · chunk_id: <code>${esc(c.chunk_id || '?')}</code></div>
            </div>`;
    }).join('');

    return queriesHtml + rows;
}

function renderContradictions(items) {
    if (!items || !items.length) return '';
    const rows = items.map(ct => {
        const sev = (ct.severity || 'low').toLowerCase();
        return `
            <li class="contradiction-item contradiction-sev-${esc(sev)}">
                <div class="contradiction-head">
                    <span class="contradiction-sev">${esc(sev)}</span>
                    <span class="contradiction-metric">${esc(ct.metric || 'metric')}</span>
                </div>
                <div class="contradiction-grid">
                    <div class="contradiction-side">
                        <div class="contradiction-value">${esc(ct.value_a || '—')}</div>
                        <div class="contradiction-source">${esc(ct.source_a || '?')}</div>
                    </div>
                    <div class="contradiction-vs">vs</div>
                    <div class="contradiction-side">
                        <div class="contradiction-value">${esc(ct.value_b || '—')}</div>
                        <div class="contradiction-source">${esc(ct.source_b || '?')}</div>
                    </div>
                </div>
                ${ct.note ? `<div class="contradiction-note">${esc(ct.note)}</div>` : ''}
            </li>`;
    }).join('');

    return `
        <div class="contradictions-block">
            <div class="contradictions-head">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                <span>${items.length} contradiction${items.length === 1 ? '' : 's'} found across retrieved chunks</span>
            </div>
            <ul class="contradictions-list">${rows}</ul>
        </div>`;
}

function renderAttributionMismatches(items, score) {
    if (!items || !items.length) return '';
    const labels = {
        wrong_entity:       'Wrong entity',
        no_supporting_fact: 'Unsupported',
        unit_mismatch:      'Unit mismatch',
        value_mismatch:     'Value mismatch',
    };
    const rows = items.slice(0, 8).map(m => {
        const mode = (m.failure_mode || 'wrong_entity').toLowerCase();
        const label = labels[mode] || mode;
        const nf = m.nearest_fact || {};
        const expected = nf.entity
            ? `<div class="attr-expected"><span class="attr-label">in facts:</span> <strong>${esc(nf.entity)}</strong> · ${esc(nf.metric || '?')} = ${esc(nf.value_raw || '?')} ${esc(nf.unit || '')}</div>`
            : '';
        return `
            <li class="attr-item attr-mode-${esc(mode)}">
                <div class="attr-head">
                    <span class="attr-mode">${esc(label)}</span>
                </div>
                <div class="attr-claim">
                    <span class="attr-label">claim:</span>
                    <strong>${esc(m.claimed_entity || '(unknown)')}</strong>
                    paired with
                    <strong>${esc(m.claimed_value_raw || '?')}</strong>
                    ${m.claimed_unit ? esc(m.claimed_unit) : ''}
                </div>
                ${expected}
                ${m.claim_sentence ? `<div class="attr-sentence">"${esc(m.claim_sentence)}"</div>` : ''}
            </li>`;
    }).join('');
    const pct = typeof score === 'number' ? Math.round(score * 100) + '%' : '';
    return `
        <div class="attribution-block">
            <div class="attribution-head">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                <span>Attribution check — ${items.length} numeric claim${items.length === 1 ? '' : 's'} flagged${pct ? ` (score ${pct})` : ''}</span>
            </div>
            <ul class="attribution-list">${rows}</ul>
        </div>`;
}

function renderFactsPanel(facts) {
    if (!facts || !facts.length) return '';
    const rows = facts.slice(0, 50).map(f => {
        const period = f.period ? ` · ${esc(f.period)}` : '';
        const unit = f.unit ? ` ${esc(f.unit)}` : '';
        const cite = f.source_doc
            ? `[${esc(f.source_doc)}, p${esc(String(f.source_page || '?'))}]`
            : '';
        return `
            <tr>
                <td>${esc(f.entity || '?')}</td>
                <td>${esc(f.metric || '?')}${period}</td>
                <td class="facts-value"><strong>${esc(f.value_raw || '?')}</strong>${unit}</td>
                <td class="facts-cite">${cite}</td>
            </tr>`;
    }).join('');
    const more = facts.length > 50 ? `<div class="facts-more">…and ${facts.length - 50} more</div>` : '';
    return `
        <details class="facts-block">
            <summary class="facts-head">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
                <span>${facts.length} extracted fact${facts.length === 1 ? '' : 's'} used for grounding</span>
            </summary>
            <div class="facts-table-wrap">
                <table class="facts-table">
                    <thead><tr><th>Entity</th><th>Metric</th><th>Value</th><th>Source</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
                ${more}
            </div>
        </details>`;
}

function formatTokens(n) {
    if (n == null) return '0';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return String(n);
}

function renderTableChunk(td, originalText) {
    const tableHtml = renderSqlTable(td);
    const jsonText = JSON.stringify({
        name: td.name,
        columns: td.columns,
        column_types: td.column_types,
        rows: td.rows,
        row_count: td.row_count,
    }, null, 2);
    const safeText = esc(originalText || '');
    const safeJson = esc(jsonText);

    return `
        <div class="chunk-table-block">
            <div class="chunk-table-tabs">
                <button class="chunk-tab active" data-view="table" onclick="switchChunkView(this, 'table')">Table</button>
                <button class="chunk-tab" data-view="json" onclick="switchChunkView(this, 'json')">JSON</button>
                <button class="chunk-tab" data-view="text" onclick="switchChunkView(this, 'text')">Text (LLM-facing)</button>
                <span class="chunk-table-info">${td.displayed_rows} of ${td.row_count} rows · ${td.columns.length} cols</span>
            </div>
            <div class="chunk-view chunk-view-table">${tableHtml}</div>
            <div class="chunk-view chunk-view-json" style="display:none"><pre>${safeJson}</pre></div>
            <div class="chunk-view chunk-view-text" style="display:none"><pre>${safeText}</pre></div>
        </div>`;
}

function renderSqlTable(td) {
    if (!td.columns || !td.columns.length) return '<p class="empty-state">No columns</p>';
    const head = `<thead><tr>${td.columns.map(col => `<th>${esc(col)}</th>`).join('')}</tr></thead>`;
    const body = (td.rows || []).map(row => {
        return `<tr>${(row || []).map(cell =>
            `<td>${esc(formatSqlCell(cell))}</td>`
        ).join('')}</tr>`;
    }).join('');
    return `<div class="chunk-sql-table-wrap"><table class="chunk-sql-table"><colgroup>${
        td.columns.map(() => '<col>').join('')
    }</colgroup>${head}<tbody>${body}</tbody></table></div>`;
}

function formatSqlCell(v) {
    if (v === null || v === undefined) return '—';
    if (typeof v === 'number') {
        if (Number.isInteger(v)) return v.toLocaleString();
        // Trim trailing zeros on floats (1.000 → 1, 0.450 stays)
        return Number.isFinite(v) ? Number(v.toFixed(4)).toString() : String(v);
    }
    return String(v);
}

function switchChunkView(btn, view) {
    const block = btn.closest('.chunk-table-block');
    if (!block) return;
    block.querySelectorAll('.chunk-tab').forEach(t => t.classList.toggle('active', t === btn));
    block.querySelectorAll('.chunk-view').forEach(v => {
        v.style.display = v.classList.contains('chunk-view-' + view) ? '' : 'none';
    });
}

// ── Message Rendering ────────────────────────────────────────

function addMessage(role, text, isError = false) {
    const container = document.getElementById('messagesContainer');
    const id = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 4);
    const avatarClass = role === 'user' ? 'user-avatar' : 'ai-avatar';
    const avatarText  = role === 'user' ? 'U' : 'P';
    const label       = role === 'user' ? 'You' : 'Answer';
    const rowClass    = role === 'user' ? 'user' : 'ai';

    const html = `
        <div class="message-row ${rowClass}" id="${id}">
            <div class="message-inner">
                <div class="message-avatar ${avatarClass}">${avatarText}</div>
                <div class="message-content">
                    <div class="message-label">${label}</div>
                    <div class="message-text">${isError ? `<p style="color:var(--red)">${esc(text)}</p>` : formatText(text)}</div>
                </div>
            </div>
        </div>`;

    container.insertAdjacentHTML('beforeend', html);
    container.scrollTop = container.scrollHeight;
    return id;
}

function addAIMessage(data) {
    const container = document.getElementById('messagesContainer');
    const id = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 4);

    const conf     = typeof data.confidence_score   === 'number' ? data.confidence_score   : null;
    const faith    = typeof data.faithfulness_score === 'number' ? data.faithfulness_score : null;
    const coverage = typeof data.citation_coverage  === 'number' ? data.citation_coverage  : null;
    const attr     = typeof data.attribution_score  === 'number' ? data.attribution_score  : null;

    const pills = [];
    if (conf !== null)     pills.push(qualityPill('Confidence',  conf));
    if (faith !== null)    pills.push(qualityPill('Faithfulness', faith));
    if (coverage !== null) pills.push(qualityPill('Citation coverage', coverage));
    // Only show the attribution pill when there were numeric claims to check
    // (a score of exactly 1.0 with no mismatches is uninformative for narrative answers).
    if (attr !== null && (attr < 1.0 || (Array.isArray(data.attribution_mismatches) && data.attribution_mismatches.length))) {
        pills.push(qualityPill('Attribution', attr));
    }

    let gapsHtml = '';
    if (Array.isArray(data.information_gaps) && data.information_gaps.length > 0) {
        gapsHtml = `<div class="info-gaps-inline"><strong>Info gaps:</strong> ${data.information_gaps.map(g => esc(g)).join('; ')}</div>`;
    }

    let contradictionsHtml = '';
    if (Array.isArray(data.contradictions) && data.contradictions.length > 0) {
        contradictionsHtml = renderContradictions(data.contradictions);
    }

    let attributionHtml = '';
    if (Array.isArray(data.attribution_mismatches) && data.attribution_mismatches.length > 0) {
        attributionHtml = renderAttributionMismatches(data.attribution_mismatches, attr);
    }

    let factsHtml = '';
    if (Array.isArray(data.facts) && data.facts.length > 0) {
        factsHtml = renderFactsPanel(data.facts);
    }

    const answerText = data.answer || '';

    const html = `
        <div class="message-row ai" id="${id}">
            <div class="message-inner">
                <div class="message-avatar ai-avatar">P</div>
                <div class="message-content">
                    <div class="message-label">
                        Answer
                        <span class="msg-actions">
                            <button class="msg-action-btn" onclick="copyAnswer('${id}')" title="Copy answer">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
                                Copy
                            </button>
                        </span>
                    </div>
                    <div class="message-text" data-answer="${esc(answerText)}">${formatAnswer(answerText)}</div>
                    ${pills.length ? `<div class="quality-row">${pills.join('')}</div>` : ''}
                    ${attributionHtml}
                    ${contradictionsHtml}
                    ${factsHtml}
                    ${gapsHtml}
                </div>
            </div>
        </div>`;

    container.insertAdjacentHTML('beforeend', html);
    container.scrollTop = container.scrollHeight;
}

function qualityPill(label, score) {
    const pct = Math.round(score * 100);
    const cls = score >= 0.75 ? 'q-high' : score >= 0.45 ? 'q-med' : 'q-low';
    return `<span class="quality-pill ${cls}"><span class="pill-dot"></span>${label}: ${pct}%</span>`;
}

function addLoadingMessage() {
    const container = document.getElementById('messagesContainer');
    const id = 'loading-' + Date.now();
    const html = `
        <div class="message-row ai" id="${id}">
            <div class="message-inner">
                <div class="message-avatar ai-avatar">P</div>
                <div class="message-content">
                    <div class="message-label">Answer</div>
                    <div class="typing-indicator"><span></span><span></span><span></span></div>
                    <span class="typing-hint">Searching your PDFs &amp; synthesizing answer…</span>
                </div>
            </div>
        </div>`;
    container.insertAdjacentHTML('beforeend', html);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function copyAnswer(msgId) {
    const row = document.getElementById(msgId);
    if (!row) return;
    const textEl = row.querySelector('.message-text');
    const raw = textEl ? (textEl.getAttribute('data-answer') || textEl.innerText) : '';
    if (!raw) return;
    navigator.clipboard.writeText(raw).then(() => {
        showToast('Answer copied to clipboard', 'success');
    }).catch(() => {
        showToast('Copy failed — your browser blocked clipboard access', 'error');
    });
}

// ── Details Panel ────────────────────────────────────────────

function updateDetailsPanel(data) {
    // Sources
    const sourcesList = document.getElementById('sourcesList');
    const citations = data.citations || [];
    if (citations.length > 0) {
        sourcesList.innerHTML = citations.map(c =>
            `<div class="source-chip" title="${esc(c.document || '')} — Page ${esc(String(c.page || '?'))}">
                <span class="doc">${esc(c.document || 'Unknown')}</span>
                <span class="page">p.${esc(String(c.page || '?'))}</span>
            </div>`
        ).join('');
    } else {
        sourcesList.innerHTML = '<p class="empty-state">No citations in this response</p>';
    }

    // SQL
    const sqlSection = document.getElementById('detailSQL');
    if (data.generated_sql) {
        sqlSection.style.display = 'block';
        document.getElementById('sqlDisplay').textContent = data.generated_sql;
    } else {
        sqlSection.style.display = 'none';
    }

    // Trace
    const traceList = document.getElementById('traceList');
    const trace = data.execution_trace || [];
    if (trace.length > 0) {
        traceList.innerHTML = trace.map(s =>
            `<div class="trace-step-row">
                <span class="trace-agent-name">${esc(s.agent || '?')}</span>
                <span class="trace-summary">${esc(s.output_summary || s.action || '')}</span>
                <span class="trace-time">${formatMs(s.duration_ms)}</span>
            </div>`
        ).join('');
    } else {
        traceList.innerHTML = '<p class="empty-state">No trace yet</p>';
    }

    // Meta
    const metaGrid = document.getElementById('metaGrid');
    const tu = data.token_usage || {};
    const tokenSummary = tu.total_tokens
        ? `${formatTokens(tu.total_tokens)} (${tu.prompt_tokens}↓ / ${tu.completion_tokens}↑, ${tu.calls} calls)`
        : '—';
    metaGrid.innerHTML = [
        { label: 'Query Type', value: data.query_type },
        { label: 'Scope',      value: data.query_scope },
        { label: 'Strategy',   value: data.retrieval_strategy },
        { label: 'Duration',   value: formatMs(data.duration_ms) },
        { label: 'Total tokens', value: tokenSummary },
        { label: 'Sub-questions', value: (data.sub_questions || []).length || '—' },
    ].map(m =>
        `<div class="meta-item"><div class="label">${m.label}</div><div class="value">${esc(String(m.value || '—'))}</div></div>`
    ).join('');

    // Retrieved chunks (debug panel)
    const chunksContainer = document.getElementById('retrievedChunksList');
    if (chunksContainer) {
        const chunks = data.retrieved_chunks || [];
        if (chunks.length === 0) {
            chunksContainer.innerHTML = '<p class="empty-state">No chunks retrieved (table-only or end-early path).</p>';
        } else {
            chunksContainer.innerHTML = renderChunks(chunks, data.retrieval_queries || []);
        }
    }

    // Per-step token table in trace section
    updateTokenBreakdown(data.execution_trace || []);

    // Auto-show panel if not visible
    if (!detailsVisible) toggleDetails();
}

function updateTokenBreakdown(trace) {
    const tokenList = document.getElementById('tokenBreakdown');
    if (!tokenList) return;
    const withTokens = trace.filter(s => s.tokens && s.tokens.total_tokens);
    if (!withTokens.length) {
        tokenList.innerHTML = '<p class="empty-state">No LLM calls recorded.</p>';
        return;
    }
    tokenList.innerHTML = `
        <table class="token-table">
            <thead>
                <tr><th>Step</th><th>Calls</th><th>Prompt</th><th>Completion</th><th>Total</th></tr>
            </thead>
            <tbody>
                ${withTokens.map(s => `
                    <tr>
                        <td>${esc(s.agent || '?')}</td>
                        <td>${s.tokens.calls || 0}</td>
                        <td>${formatTokens(s.tokens.prompt_tokens)}</td>
                        <td>${formatTokens(s.tokens.completion_tokens)}</td>
                        <td><strong>${formatTokens(s.tokens.total_tokens)}</strong></td>
                    </tr>`).join('')}
            </tbody>
        </table>`;
}

// ── Formatting ───────────────────────────────────────────────

function formatAnswer(text) {
    if (!text) return '<p class="empty-state">No answer generated</p>';

    // Normalize first: the LLM emits headings + tables without blank lines
    // between them, and sometimes injects literal `<br>` HTML inside table
    // cells. Block-splitting on blank lines only works if those blank lines
    // actually exist around block boundaries, so insert them here.
    const normalized = normalizeMarkdown(String(text));

    // Split on blank lines → blocks. Each block is rendered as paragraph,
    // markdown table, list, heading, or code fence.
    const blocks = normalized.split(/\n\s*\n+/);
    return blocks.map(renderBlock).filter(Boolean).join('\n');
}

function normalizeMarkdown(text) {
    // Leave literal `<br>` alone here — the LLM uses it for in-cell line
    // breaks inside table rows. `renderInline` un-escapes it back to <br>
    // after the rest of the text is HTML-escaped.
    const lines = text.split('\n');
    const result = [];
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const prev = result.length ? result[result.length - 1] : '';
        const trimmed = line.trim();

        const isHeading = /^#{1,6}\s+/.test(trimmed);
        const isTableRow = /^\|/.test(trimmed);
        const prevTrim = prev.trim();
        const prevIsTableRow = /^\|/.test(prevTrim);
        const prevIsBlank = prevTrim === '';

        // Blank line before a heading (unless we're already at the top or
        // following a blank).
        if (isHeading && result.length && !prevIsBlank) {
            result.push('');
        }

        // Blank line before a table starts (transition non-table → table).
        if (isTableRow && result.length && !prevIsBlank && !prevIsTableRow) {
            result.push('');
        }

        // Blank line after a table ends (transition table → non-table).
        if (prevIsTableRow && trimmed !== '' && !isTableRow) {
            result.push('');
        }

        result.push(line);

        // Blank line after a heading line if the next non-empty line isn't blank.
        if (isHeading && i + 1 < lines.length && lines[i + 1].trim() !== '') {
            result.push('');
        }
    }
    return result.join('\n');
}

function renderBlock(block) {
    const rawLines = block.split('\n');
    const lines = rawLines.map(l => l.replace(/\s+$/, '')).filter(l => l.trim() !== '');
    if (!lines.length) return '';

    // ── Bottom-line callout ──
    // Matches the new `**Bottom line:**` prefix AND the legacy `**TL;DR:**`
    // form so saved conversations still render with the green callout box.
    const blMatch = lines[0].match(/^(?:\*\*\s*)?(?:Bottom\s*line|TL;?DR)\s*:?(?:\s*\*\*)?\s*(.*)$/i);
    if (blMatch) {
        const firstLine = blMatch[1];
        const rest = lines.slice(1);
        const body = [firstLine, ...rest].filter(Boolean).map(renderInline).join('<br>');
        return `<div class="tldr-callout"><span class="tldr-label">Bottom line</span><span class="tldr-body">${body}</span></div>`;
    }

    // ── Markdown table — header row + divider row (---|---) + body rows ──
    if (lines.length >= 2 && /\|/.test(lines[0]) && /^\s*\|?[\s:\-|]+\|?\s*$/.test(lines[1]) && /-/.test(lines[1])) {
        return renderMdTable(lines);
    }

    // ── Heading (#, ##, ###) ──
    const headMatch = lines.length === 1 && lines[0].match(/^(#{1,6})\s+(.+)$/);
    if (headMatch) {
        const level = Math.min(headMatch[1].length + 2, 6); // h3-h6 so it doesn't fight the chat header
        return `<h${level} class="md-h">${renderInline(headMatch[2])}</h${level}>`;
    }

    // ── Bulleted / ordered list ──
    const isUnordered = lines.every(l => /^\s*[-*•]\s+/.test(l));
    const isOrdered   = lines.every(l => /^\s*\d+\.\s+/.test(l));
    if (isUnordered || isOrdered) {
        const tag = isOrdered ? 'ol' : 'ul';
        const items = lines.map(l => l.replace(/^\s*(?:[-*•]|\d+\.)\s+/, ''));
        return `<${tag} class="md-list">${items.map(i => `<li>${renderInline(i)}</li>`).join('')}</${tag}>`;
    }

    // ── Default paragraph — preserve single newlines as <br> ──
    return `<p>${lines.map(renderInline).join('<br>')}</p>`;
}

function renderMdTable(lines) {
    const parseRow = (line) => {
        let s = line.trim();
        if (s.startsWith('|')) s = s.slice(1);
        if (s.endsWith('|'))   s = s.slice(0, -1);
        return s.split('|').map(c => c.trim());
    };

    const header = parseRow(lines[0]);
    const body   = lines.slice(2).map(parseRow);

    const thead = `<thead><tr>${header.map(h => `<th>${renderInline(h)}</th>`).join('')}</tr></thead>`;
    const tbody = `<tbody>${body.map(r =>
        `<tr>${r.map(c => `<td>${renderInline(c)}</td>`).join('')}</tr>`
    ).join('')}</tbody>`;

    return `<div class="md-table-wrap"><table class="md-table">${thead}${tbody}</table></div>`;
}

function renderInline(text) {
    // Escape first, then re-apply trusted markup transforms.
    return esc(text)
        .replace(/&lt;br\s*\/?&gt;/gi, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
        .replace(/`([^`]+?)`/g, '<code>$1</code>')
        .replace(/\[(.*?),\s*Page\s*(\d+)\]/gi, '<span class="citation" title="$1, Page $2">$1, p.$2</span>');
}

function formatText(text) {
    return `<p>${esc(text).replace(/\n/g, '<br>')}</p>`;
}

function esc(str) {
    if (str === null || str === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function formatMs(ms) {
    if (typeof ms !== 'number') return '—';
    if (ms < 1000) return ms.toFixed(0) + 'ms';
    return (ms / 1000).toFixed(1) + 's';
}

function setStatus(text, type) {
    const el = document.getElementById('systemStatus');
    if (!el) return;
    const dot = el.querySelector('.status-dot');
    const label = el.querySelector('span:last-child');
    label.textContent = text;
    dot.className = 'status-dot ' + type;
}

// ── Toast ────────────────────────────────────────────────────

function showToast(message, kind = 'error') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'toast' + (kind === 'success' ? ' success' : '');
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transition = 'opacity 0.25s ease';
        setTimeout(() => el.remove(), 280);
    }, 2400);
}

// ── Auto-resize textarea ─────────────────────────────────────

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

document.getElementById('queryInput').addEventListener('input', function () {
    autoResize(this);
});

document.getElementById('queryInput').addEventListener('keydown', function (e) {
    // Enter sends; Shift+Enter inserts a newline; Cmd/Ctrl+Enter also sends.
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitQuery();
    }
});

// ── Upload PDF (single-file, wipes-then-ingests) ─────────────────

let _selectedPdfFile = null;
let _uploadInFlight = false;

function openUploadModal() {
    _selectedPdfFile = null;
    document.getElementById('uploadBackdrop').style.display = 'flex';
    document.getElementById('uploadResult').style.display = 'none';
    document.getElementById('uploadResult').innerHTML = '';
    document.getElementById('uploadSteps').style.display = 'none';
    document.getElementById('uploadStepList').innerHTML = '';
    document.getElementById('ollamaHelp').style.display = 'none';
    document.getElementById('ollamaHelp').innerHTML = '';
    document.getElementById('fileDrop').style.display = '';
    document.getElementById('fileDropInner').innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <span class="file-drop-text">Click to choose a PDF</span>
        <span class="file-drop-hint">or drop it here</span>`;
    document.getElementById('pdfFileInput').value = '';
    document.getElementById('uploadStartBtn').disabled = true;
    document.getElementById('uploadStartBtn').textContent = 'Wipe & Ingest';
    document.getElementById('uploadCancelBtn').textContent = 'Cancel';

    wireFileDrop();

    // Pre-flight Ollama check so the user finds out before picking a file
    fetch(`${API_BASE}/ollama/health`).then(r => r.json()).then(info => {
        if (!info.ready) renderOllamaHelp(info);
    }).catch(() => {/* offline — handled at upload time */});
}

function closeUploadModal() {
    if (_uploadInFlight) {
        const ok = confirm('Ingestion is running — close anyway? The job will continue on the server.');
        if (!ok) return;
    }
    document.getElementById('uploadBackdrop').style.display = 'none';
}

function onBackdropClick(e, kind) {
    if (e.target === e.currentTarget) {
        if (kind === 'upload') closeUploadModal();
    }
}

function wireFileDrop() {
    const drop = document.getElementById('fileDrop');
    if (!drop || drop.dataset.wired === '1') return;
    drop.dataset.wired = '1';
    drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('dragover'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
    drop.addEventListener('drop', (e) => {
        e.preventDefault();
        drop.classList.remove('dragover');
        const f = (e.dataTransfer && e.dataTransfer.files) ? e.dataTransfer.files[0] : null;
        if (f) acceptFile(f);
    });
}

function onFileChosen(e) {
    const f = e.target.files && e.target.files[0];
    if (f) acceptFile(f);
}

function acceptFile(f) {
    if (!/\.pdf$/i.test(f.name)) {
        showToast('Only PDF files are accepted', 'error');
        return;
    }
    _selectedPdfFile = f;
    const sizeKb = Math.round(f.size / 1024);
    document.getElementById('fileDropInner').innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span class="file-drop-text">${esc(f.name)}</span>
        <span class="file-drop-hint">${sizeKb.toLocaleString()} KB · click to choose another</span>`;
    document.getElementById('uploadStartBtn').disabled = false;
}

async function startUpload() {
    if (!_selectedPdfFile || _uploadInFlight) return;
    _uploadInFlight = true;

    const startBtn = document.getElementById('uploadStartBtn');
    const cancelBtn = document.getElementById('uploadCancelBtn');
    startBtn.disabled = true;
    startBtn.textContent = 'Working…';
    cancelBtn.textContent = 'Close';

    document.getElementById('ollamaHelp').style.display = 'none';
    document.getElementById('uploadResult').style.display = 'none';
    document.getElementById('uploadSteps').style.display = '';
    document.getElementById('uploadStepList').innerHTML = '';
    document.getElementById('uploadStepsLabel').textContent = 'Preparing…';
    document.getElementById('fileDrop').style.display = 'none';

    setStatus('Ingesting…', 'busy');

    const form = new FormData();
    form.append('file', _selectedPdfFile);

    let finalData = null;
    let errorPayload = null;

    try {
        const resp = await fetch(`${API_BASE}/upload-ingest/stream`, {
            method: 'POST',
            headers: { 'Accept': 'text/event-stream' },
            body: form,
        });
        if (!resp.ok) {
            let detail = `HTTP ${resp.status}`;
            try { detail = (await resp.json()).detail || detail; } catch {}
            throw new Error(detail);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const frames = buffer.split(/\n\n/);
            buffer = frames.pop() || '';

            for (const frame of frames) {
                if (!frame.trim() || frame.trim().startsWith(':')) continue;
                let eventName = 'message';
                let dataStr = '';
                for (const line of frame.split('\n')) {
                    if (line.startsWith('event:')) eventName = line.slice(6).trim();
                    else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
                }
                if (!dataStr) continue;
                let payload;
                try { payload = JSON.parse(dataStr); } catch { continue; }

                if (eventName === 'step') {
                    appendUploadStep(payload);
                } else if (eventName === 'complete') {
                    finalData = payload;
                } else if (eventName === 'error') {
                    errorPayload = payload;
                }
            }
        }
    } catch (e) {
        errorPayload = { error: e.message };
    }

    _uploadInFlight = false;
    setStatus('System Ready', 'online');
    document.getElementById('uploadSteps').style.display = 'none';

    if (errorPayload) {
        const resBox = document.getElementById('uploadResult');
        resBox.style.display = '';
        resBox.className = 'upload-result error';
        resBox.innerHTML = `
            <div class="upload-result-title">Ingestion failed</div>
            <div class="upload-result-msg">${esc(errorPayload.error || 'Unknown error')}</div>`;
        if (errorPayload.ollama) renderOllamaHelp(errorPayload.ollama);
        startBtn.disabled = !_selectedPdfFile;
        startBtn.textContent = 'Retry';
        return;
    }

    if (finalData) {
        const s = finalData.stats || {};
        const resBox = document.getElementById('uploadResult');
        resBox.style.display = '';
        resBox.className = 'upload-result success';
        resBox.innerHTML = `
            <div class="upload-result-title">${esc(finalData.filename || 'PDF')} indexed</div>
            <div class="upload-result-msg">
                ${s.total_pages || 0} pages · ${s.text_chunks || 0} chunks ·
                ${s.tables_extracted || 0} tables · ${s.images_processed || 0} images ·
                ${s.raptor_nodes || 0} RAPTOR nodes · ${s.total_time_seconds || 0}s
            </div>
            <div class="upload-result-msg">Ready — ask anything about this document.</div>`;
        startBtn.textContent = 'Done';
        startBtn.disabled = true;
        cancelBtn.textContent = 'Close';
        showToast('PDF indexed — ready to query', 'success');
    }
}

function appendUploadStep(step) {
    const ul = document.getElementById('uploadStepList');
    if (!ul) return;
    const label = step.label || step.node || 'Step';
    const summary = step.summary ? `<span class="step-summary">${esc(step.summary)}</span>` : '';
    const running = step.status === 'running';
    const li = document.createElement('li');
    li.className = 'step-item';
    li.innerHTML = `
        <div class="step-row">
            <span class="step-check ${running ? 'running' : ''}">
                ${running
                    ? '<span class="spinner"></span>'
                    : '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>'}
            </span>
            <span class="step-label">${esc(label)}</span>
            ${summary}
        </div>`;
    ul.appendChild(li);
    document.getElementById('uploadStepsLabel').textContent = label;
}

function renderOllamaHelp(info) {
    const box = document.getElementById('ollamaHelp');
    if (!box) return;
    const osKey = info.os || 'windows';
    const steps = (info.install_instructions && info.install_instructions[osKey]) || [];
    const required = (info.models_required || []).map(m => `<code>${esc(m)}</code>`).join(', ') || '—';
    const missing  = (info.models_missing || []).map(m => `<code>${esc(m)}</code>`).join(', ');
    const cloud    = (info.cloud_models || []).map(m => `<code>${esc(m)}</code>`).join(', ');

    const reachable = info.reachable
        ? '<span class="ollama-pill ok">Ollama reachable</span>'
        : '<span class="ollama-pill bad">Ollama not reachable</span>';
    const missingPill = missing
        ? `<span class="ollama-pill bad">Missing: ${missing}</span>`
        : '';
    const cloudPill = cloud
        ? `<span class="ollama-pill info">Cloud (needs <code>ollama signin</code>): ${cloud}</span>`
        : '';

    box.style.display = '';
    box.innerHTML = `
        <div class="ollama-help-head">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>
            <span>Ollama setup required</span>
        </div>
        <div class="ollama-status">
            ${reachable}
            ${missingPill}
            ${cloudPill}
        </div>
        <div class="ollama-required">
            <strong>Required at <code>${esc(info.base_url || '')}</code>:</strong> ${required}
        </div>
        <ol class="ollama-steps">
            ${steps.map(s => `<li>${esc(s)}</li>`).join('')}
        </ol>
        <div class="ollama-actions">
            <a class="btn-secondary" href="${esc(info.install_instructions && info.install_instructions.download_url || 'https://ollama.com/download')}" target="_blank" rel="noopener">Open Ollama download</a>
            <button class="btn-primary" onclick="recheckOllama()">Retry check</button>
        </div>`;
}

async function recheckOllama() {
    try {
        const info = await fetch(`${API_BASE}/ollama/health`).then(r => r.json());
        if (info.ready) {
            document.getElementById('ollamaHelp').style.display = 'none';
            document.getElementById('ollamaHelp').innerHTML = '';
            showToast('Ollama ready — you can ingest now', 'success');
            const startBtn = document.getElementById('uploadStartBtn');
            startBtn.disabled = !_selectedPdfFile;
            startBtn.textContent = 'Wipe & Ingest';
        } else {
            renderOllamaHelp(info);
        }
    } catch (e) {
        showToast('Could not reach the app server', 'error');
    }
}
