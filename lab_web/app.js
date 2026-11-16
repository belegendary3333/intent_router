'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent = value => value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
const splitNames = {train:'样例集',validation:'验证集',test:'测试集',probe:'手动探针'};
const modeNames = {chat:'聊天',skill:'任务',singing:'唱歌',dancing:'跳舞',focus:'专注模式'};
const outcomeNames = {...modeNames,clarify:'需澄清'};
const schemeNames = {baseline:'原有样例',expanded:'扩充样例',anchors:'代表锚点',classifier:'轻量分类器'};
const policyName = (scheme,key,r) => scheme==='classifier' ? (key==='default'?'裸分类（不拒识）':'温度校准 + 按意图拒识') : (key==='default'?'现有 0.62 / 0.03':`候选 ${r.candidate_thresholds.min_score} / ${r.candidate_thresholds.min_margin}`);
let data = null, currentLayout = null, busy = false, layoutRequest = 0, buildPoll = null;
let comparisonData = null, layoutsByModel = {}, queriesByModel = {};
let feedbackData = null, correctionTarget = null;
const detailModel = () => $('detail-model').value || 'small';
const modelPath = (path, model=detailModel()) => model==='small' ? path : `${path}${path.includes('?')?'&':'?'}model=${model}`;

async function api(path, options={}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail));
  return body;
}
function error(message='') { $('error').hidden = !message; $('error').textContent = message; }
function setBusy(value) {
  busy = value;
  for (const id of ['send','rebuild','scope','perplexity','seed','anchor-k','detail-model','import-real','real-file']) $(id).disabled = value || !data;
  if (!data && !value) $('rebuild').disabled = false;
  updateFeedbackControls();
}
function updateFeedbackControls() {
  const existing=correctionTarget?.record, frozen=existing?.split==='test';
  $('save-feedback').disabled=busy||!data||!correctionTarget||frozen;
  $('train-feedback').disabled=busy||!data||!feedbackData?.count||!feedbackData.pending;
  for(const id of ['feedback-mode','feedback-source','feedback-family'])$(id).disabled=busy||frozen;
  $('feedback-split').disabled=busy||Boolean(existing);
}
function table(headers, rows) {
  return `<table><thead><tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
}
function textButton(point, text=point.text) {
  return `<button type="button" class="text-button" data-point="${esc(point.id)}">${esc(text)}</button>`;
}
function badge(mode) { return `<span class="badge ${esc(mode)}">${esc(outcomeNames[mode] || mode)}</span>`; }

function renderStatic() {
  const total = data.cases.length;
  $('model-meta').textContent = `${data.model_key || 'small'} · ${data.dimension} 维 · ${total} 条案例 · ${data.anchors.length} 个锚点`;
  $('anchor-method').textContent = `${detailModel()} 原始 ${data.dimension} 维向量聚类 → 每簇选平均余弦最高的真实句；只使用样例集。`;
  for (const [id,name] of [['download-anchors','anchors.candidate.yaml'],['download-report','report.json'],['download-cases','cases.generated.yaml']]) $(id).href=modelPath(`/api/download/${name}`);
  $('download-classifier').href=modelPath('/api/download/classifier.candidate.json');
  $('anchor-count').textContent = data.anchors.length;
  $('confusion-count').textContent = data.confusions.length;
  $('anchor-k').value = data.settings.anchors_per_intent;
  $('built-at').textContent = `最近分析 ${new Date(data.built_at).toLocaleString()}`;
  $('anchors-table').innerHTML = table(['意图 / 簇','代表原句','成员数','簇内凝聚度','最近异类句 / 相似度'], data.anchors.map(a=>
    `<tr><td>${badge(a.mode)} <span class="small">#${a.cluster+1}</span></td><td class="table-text">${textButton(a)}<div class="small">${esc(a.subtype)}</div></td><td class="numeric">${a.member_count}</td><td class="numeric">${a.cohesion.toFixed(4)}</td><td class="table-text">${badge(a.nearest_other.mode)} ${textButton(a.nearest_other)} <span class="small">${a.nearest_other.similarity.toFixed(4)}</span></td></tr>`));
  $('confusions-table').innerHTML = data.confusions.length ? table(['原句 / 真实意图','集合','同类近邻','异类近邻','分数差'],data.confusions.map(c=>
    `<tr><td class="table-text">${textButton(c)}<div>${badge(c.mode)}</div></td><td>${splitNames[c.split]}</td><td class="table-text">${esc(c.own_example)} <span class="small">${c.own_score.toFixed(4)}</span></td><td class="table-text">${badge(c.other_mode)} ${esc(c.other_example)} <span class="small">${c.other_score.toFixed(4)}</span></td><td class="numeric ${c.gap<0?'negative':''}">${c.gap.toFixed(4)}</td></tr>`)) : '<p class="empty">没有分数差小于 0.08 的案例。</p>';
  const counts = Object.values(data.counts).reduce((a,c)=>({train:a.train+c.train,validation:a.validation+c.validation,test:a.test+c.test}),{train:0,validation:0,test:0});
  $('dataset-summary').textContent = `${total} 条合成句：样例 ${counts.train}、验证 ${counts.validation}、测试 ${counts.test}。阈值只在验证集选择，测试集不参与选锚点。`;
  $('quality-note').textContent = `${data.quality.note} 精确重复 ${data.quality.exact_overlap} 条；语义近重复告警 ${data.quality.near_duplicates.length} 条（余弦 ≥ 0.97）。候选阈值及锚点不会自动部署。`;
  const rows = [];
  for (const [scheme,r] of Object.entries(data.reports)) for (const key of ['default','calibrated']) {
    const m=r[key];
    rows.push(`<tr><td>${esc(schemeNames[scheme])} <span class="small">${r.bank_count} 条</span></td><td>${policyName(scheme,key,r)}</td><td class="numeric">${percent(m.accuracy)}</td><td class="numeric">${m.macro_f1.toFixed(3)}</td><td class="numeric">${m.unsafe_chat_activation}</td><td class="numeric">${m.rejected} / ${percent(m.rejection_rate)}</td><td>${percent(m.action_recall)}</td><td class="numeric">${m.count}</td></tr>`);
  }
  $('report-summary').innerHTML=table(['方案 / 训练或样例数','决策策略','全量测试准确率','Macro F1','聊天误触发动作','拒识数 / 比例','动作召回率','测试数'],rows);
  $('leaks-table').innerHTML=data.quality.near_duplicates.length ? table(['验证/测试原句','集合','最相似样例','余弦相似度'],data.quality.near_duplicates.map(c=>`<tr><td class="table-text">${textButton(c)}</td><td>${splitNames[c.split]}</td><td class="table-text">${esc(c.nearest_text)}</td><td class="numeric">${c.similarity.toFixed(4)}</td></tr>`)) : '<p class="empty">未检出余弦相似度 ≥ 0.97 的近重复。</p>';
  const head=data.classifier;
  const validationNames={'real_asr':'真实 ASR',synthetic:'合成','real_asr+feedback':'真实 ASR + 人工纠错','synthetic+feedback':'合成 + 人工纠错'};
  $('classifier-note').textContent=`C=${head.C} · 温度=${head.temperature} · 训练 ${head.train_count} 条（真实 ASR ${head.real_train_count}，纠错 ${head.feedback_train_count||0}）· 验证 ${head.validation_count} 条（${validationNames[head.validation_source]||head.validation_source}）。${head.note}`;
  $('classifier-policy').innerHTML=table(['意图','概率下限','与第二类差值下限','验证接受数','验证接受精度','验证召回率'],Object.entries(head.thresholds).map(([mode,t])=>`<tr><td>${badge(mode)}${t.disabled?' · 已禁用接受':''}</td><td>${t.min_probability}</td><td>${t.min_margin}</td><td>${t.validation_accepted}</td><td>${percent(t.validation_precision)}</td><td>${percent(t.validation_recall)}</td></tr>`));
  renderReport(); renderHistory(); renderProbeReport(); renderComparison(); renderReal(); renderFeedback();
}

function renderComparison() {
  if (!comparisonData?.ready) {
    $('comparison-note').textContent=comparisonData?.message || 'large 未就绪';
    return;
  }
  $('comparison-note').textContent=comparisonData.note;
  const models=Object.entries(comparisonData.models);
  $('comparison-summary').innerHTML=table(['模型','维度','原始空间轮廓系数','扩充样例准确率','Macro F1','聊天误触发动作','单句 P50 / P95'],models.map(([key,m])=>{
    const r=m.reports.expanded.default;
    return `<tr><td>${esc(key)} · INT8</td><td>${m.dimension}</td><td class="numeric">${m.clustering.silhouette.toFixed(4)}</td><td>${percent(r.accuracy)}</td><td>${r.macro_f1.toFixed(3)}</td><td>${r.unsafe_chat_activation} / ${r.per_class.chat.support}</td><td class="numeric">${m.benchmark.p50_ms} / ${m.benchmark.p95_ms} ms</td></tr>`;
  }));
  $('classifier-comparison').innerHTML=table(['模型','决策方法','全量准确率','Macro F1','聊天误触发动作','拒识率','动作召回率','接受部分准确率'],models.flatMap(([key,m])=>['default','calibrated'].map(policy=>{
    const r=m.reports.classifier,v=r[policy];
    return `<tr><td>${key}</td><td>${policyName('classifier',policy,r)}</td><td>${percent(v.accuracy)}</td><td>${v.macro_f1.toFixed(3)}</td><td>${v.unsafe_chat_activation}</td><td>${percent(v.rejection_rate)}</td><td>${percent(v.action_recall)}</td><td>${percent(v.accepted_accuracy)}</td></tr>`;
  })));
  $('comparison-clusters').innerHTML=table(['意图','small 轮廓系数','large 轮廓系数','small / large 类内余弦','锚点相同数','聚类一致度 ARI'],Object.keys(modeNames).map(mode=>{
    const s=comparisonData.models.small.clustering.per_mode[mode],l=comparisonData.models.large.clustering.per_mode[mode],a=comparisonData.cluster_agreement[mode];
    return `<tr><td>${badge(mode)}</td><td>${s.silhouette.toFixed(4)}</td><td>${l.silhouette.toFixed(4)}</td><td>${s.pairwise_cosine.toFixed(4)} / ${l.pairwise_cosine.toFixed(4)}</td><td>${a.anchor_overlap} / ${a.anchors_each}</td><td>${a.adjusted_rand_index.toFixed(4)}</td></tr>`;
  }));
  const rows=[];
  for (const [key,m] of models) for (const [scheme,r] of Object.entries(m.reports)) for (const policy of ['default','calibrated']) {
    const v=r[policy];
    rows.push(`<tr><td>${key}</td><td>${esc(schemeNames[scheme])} · ${r.bank_count}</td><td>${policyName(scheme,policy,r)}</td><td>${percent(v.accuracy)}</td><td>${v.macro_f1.toFixed(3)}</td><td>${v.unsafe_chat_activation}</td><td>${percent(v.rejection_rate)}</td><td>${percent(v.action_recall)}</td><td>${v.count}</td></tr>`);
  }
  $('comparison-reports').innerHTML=table(['模型','方案 / 句数','决策策略','测试准确率','Macro F1','聊天误触发动作','拒识率','动作召回率','测试数'],rows);
  const differences=comparisonData.test_differences[$('report-scheme').value][$('report-threshold').value];
  const changes={large_corrected:'large 纠正',large_regressed:'large 新增误判',both_wrong:'两边都错'};
  $('comparison-differences').innerHTML=differences.length?table(['测试原句','真实意图','small','large','变化'],differences.map(c=>`<tr><td class="table-text">${textButton(c)}</td><td>${badge(c.expected_mode)}</td><td>${badge(c.small.mode)}</td><td>${badge(c.large.mode)}</td><td>${changes[c.change]}</td></tr>`)):'<p class="empty">此方案下两模型的测试预测一致。</p>';
  $('comparison-routes').innerHTML=table(['意图','共用设定词 / 原有样例'],Object.entries(comparisonData.routes).map(([mode,texts])=>`<tr><td>${badge(mode)} · ${texts.length}</td><td>${texts.map(esc).join('、')}</td></tr>`));
}

function renderReport() {
  if (!data) return;
  const scheme=$('report-scheme').value, key=$('report-threshold').value, r=data.reports[scheme], m=r[key];
  const matrixLabels=m.matrix_labels || Object.keys(modeNames);
  $('matrix').innerHTML=table(['真实 ↓ / 预测 →',...matrixLabels.map(l=>outcomeNames[l])], m.matrix.map((row,i)=>`<tr><th scope="row">${outcomeNames[matrixLabels[i]]}</th>${row.map((n,j)=>`<td class="numeric" style="background:rgba(99,220,232,${n?Math.min(.55,.08+n/Math.max(1,m.count)*2):0})">${n}</td>`).join('')}</tr>`));
  $('class-metrics').innerHTML=table(['意图','Precision','Recall','F1','测试数'],Object.entries(m.per_class).map(([mode,c])=>`<tr><td>${badge(mode)}</td><td class="numeric">${percent(c.precision)}</td><td class="numeric">${percent(c.recall)}</td><td class="numeric">${c['f1-score'].toFixed(3)}</td><td class="numeric">${c.support}</td></tr>`));
  const failures=r.cases.filter(c=>c[key].mode!==c.mode);
  $('report-errors').innerHTML=failures.length ? table(['测试原句','真实 → 预测','分数 / 差值'],failures.map(c=>`<tr><td class="table-text">${textButton(c)}</td><td>${badge(c.mode)} → ${badge(c[key].mode)}</td><td class="numeric">${c[key].score.toFixed(4)} / ${c[key].margin.toFixed(4)}</td></tr>`)) : '<p class="empty">此方案在当前合成测试集上无误判，不代表真实环境无误判。</p>';
}

function renderProbeReport() {
  if (!data) return;
  const r=data.probe_report;
  $('probe-note').textContent=`共 ${r.total} 条探针，其中 ${r.labeled} 条有人工标注。${r.note}`;
  $('probe-table').innerHTML=r.labeled ? table(['方案','标注数','全量准确率','Macro F1','聊天误触发动作','拒识率','动作召回率'],Object.entries(r.schemes).map(([s,m])=>`<tr><td>${esc(schemeNames[s])}${s==='classifier'?'（拒识策略）':''}</td><td class="numeric">${m.count}</td><td class="numeric">${percent(m.accuracy)}</td><td class="numeric">${m.macro_f1.toFixed(3)}</td><td class="numeric">${m.unsafe_chat_activation}</td><td>${percent(m.rejection_rate)}</td><td>${percent(m.action_recall)}</td></tr>`)) : '<p class="empty">发送句子前可选择期望意图。没有标注时不展示准确率。</p>';
}
function renderHistory() {
  $('history-count').textContent=data.queries.length;
  $('history-list').innerHTML=data.queries.slice().reverse().map(q=>`<button type="button" class="history-item" data-probe="${esc(q.id)}">${esc(q.text)}<div class="small">${esc(modeNames[q.mode])}${q.expected_mode?` · 期望 ${esc(modeNames[q.expected_mode])}`:' · 未标注'}</div></button>`).join('');
}
function renderProbe(q) {
  const paired=Object.entries(queriesByModel).map(([model,queries])=>({model,q:queries.find(p=>p.id===q.id)})).filter(p=>p.q);
  $('probe-result').innerHTML=`<h3 class="result-title">${esc(q.text)}<span>${q.expected_mode?`期望：${esc(modeNames[q.expected_mode])}`:'未提供期望意图'} · ${esc(detailModel())} 详情 · 向量用默认阈值，分类器用拒识策略</span></h3>`+
    (paired.length===2?'<h3>同一句模型对照</h3>'+table(['模型 / 方法','意图','余弦或概率 / 差值'],paired.flatMap(({model,q:p})=>['expanded','classifier'].map(s=>{const r=p.results[s];return `<tr><td>${model} / ${schemeNames[s]}</td><td>${badge(r.mode)}</td><td>${r.score.toFixed(4)} / ${r.margin.toFixed(4)}</td></tr>`;}))):'')+
    Object.entries(q.results).map(([s,r])=>`<div class="scheme-row"><strong>${esc(schemeNames[s])}</strong>${badge(r.mode)}<small>${r.accepted?'已接受候选':s==='classifier'?`需澄清，候选 ${esc(modeNames[r.matched_mode])}`:`低置信度回退聊天，候选 ${esc(modeNames[r.matched_mode])}`}</small><span class="score-text">${r.score.toFixed(4)}</span><small>${s==='classifier'?'校准概率；与第二类差值':'余弦；与第二意图差值'}</small><span class="score-text">${r.margin.toFixed(4)}</span>${s==='classifier'?`<small>裸分类：${esc(outcomeNames[r.raw.mode])}（${r.raw.score.toFixed(4)}）${r.clarification?'<br>'+esc(r.clarification):''}</small>`:''}</div>`).join('')+
    '<h3>相近锚点</h3><ul class="neighbor-list">'+q.nearest_anchors.map(a=>`<li>${textButton(a)}<span>${esc(modeNames[a.mode])} ${a.similarity.toFixed(4)}</span></li>`).join('')+'</ul>'+
    '<h3>相近样例</h3><ul class="neighbor-list">'+q.nearest_cases.map(a=>`<li>${textButton(a)}<span>${esc(modeNames[a.mode])} ${a.similarity.toFixed(4)}</span></li>`).join('')+'</ul>';
  setupFeedbackEditor(q);
}

function setupFeedbackEditor(q, record=null) {
  record=record||feedbackData?.cases.find(c=>c.text===q.text||c.normalized_text===q.normalized_text)||null;
  correctionTarget={id:q.id,text:q.text,record};
  $('feedback-editor').hidden=false;$('feedback-text').textContent=q.text;
  $('feedback-mode').value=record?.mode||q.expected_mode||'';
  $('feedback-split').value=record?.split||'train';
  $('feedback-source').value=record?.source||'manual';
  $('feedback-family').value=record?.family||'';
  $('feedback-editor-note').textContent=record?.split==='test'?'此条回归留出已冻结，只评测，不可改标签或移入训练。':record?'已归档；可修正人工标签或来源，但不可换集合。修改后需重新训练。':'归档不立即训练。正确句也可归档；同会话或相近改写请填相同家族 ID，固定测试句不可用于训练。';
  updateFeedbackControls();
}

function renderFeedback() {
  if(!feedbackData)return;
  const f=feedbackData;
  $('feedback-count').textContent=f.count;
  $('feedback-note').textContent=`归档 ${f.count} 条 · 当前生效 ${f.active_count} 条 · ${f.pending?'有未应用的纠错，请重新训练。':'当前归档已应用，或尚无数据。'} ${f.note}`;
  $('feedback-training-meta').textContent=f.last_training?`最近纠错训练 ${new Date(f.last_training.trained_at).toLocaleString()}`:'尚未进行纠错训练';
  $('feedback-counts').innerHTML=table(['集合',...Object.values(modeNames)],Object.entries(f.counts).map(([split,counts])=>`<tr><td>${split==='test'?'回归留出':splitNames[split]}</td>${Object.keys(modeNames).map(m=>`<td>${counts[m]||0}</td>`).join('')}</tr>`));
  $('feedback-table').innerHTML=f.count?table(['原句','正确意图','集合','来源','家族 / 操作'],f.cases.map(c=>`<tr><td class="table-text">${esc(c.text)}</td><td>${badge(c.mode)}</td><td>${c.split==='test'?'回归留出':splitNames[c.split]}</td><td>${c.source==='asr'?'真实 ASR（人工确认）':'手动输入'}</td><td>${esc(c.family)}<br><button type="button" class="text-button" data-feedback="${esc(c.id)}">${c.split==='test'?'查看冻结标注':'编辑标注'}</button></td></tr>`)):'<p class="empty">先发送新句，在左侧确认标签并归档，不必手写 JSON。</p>';
  const run=f.last_training,rows=[];
  $('feedback-run-note').textContent=run?.note||'归档后点击重新训练，才生成训练前后对照。';
  if(run)for(const [model,r]of Object.entries(run.models))for(const [prefix,label]of [['synthetic','固定合成测试'],['feedback','纠错回归留出'],['real_asr','真实 ASR 留出']])for(const phase of ['before','after']){
    const report=r[`${prefix}_${phase}`];if(!report)continue;
    const m=report.calibrated;
    rows.push(`<tr><td>${model}</td><td>${label}</td><td>${phase==='before'?'训练前':'训练后'}</td><td>${percent(m.accuracy)}</td><td>${m.macro_f1.toFixed(3)}</td><td>${m.unsafe_chat_activation}</td><td>${percent(m.rejection_rate)}</td><td>${percent(m.action_recall)}</td><td>${m.count}</td></tr>`);
  }
  $('feedback-run').innerHTML=rows.length?table(['模型','数据集','阶段（均用拒识策略）','全量准确率','Macro F1','聊天误触发','拒识率','动作召回','测试数'],rows):'<p class="empty">尚无训练记录；不会虚构提升。</p>';
  const evaluation=f.by_model?.[detailModel()]||data.feedback_report,evalRows=[],failures=[];
  if(evaluation?.test_count)for(const [s,r]of Object.entries(evaluation.reports))for(const policy of ['default','calibrated']){
    const m=r[policy];evalRows.push(`<tr><td>${schemeNames[s]}</td><td>${policyName(s,policy,data.reports[s])}</td><td>${percent(m.accuracy)}</td><td>${m.unsafe_chat_activation}</td><td>${percent(m.rejection_rate)}</td><td>${percent(m.action_recall)}</td><td>${m.count}</td></tr>`);
    for(const c of r.cases.filter(c=>c[policy].mode!==c.mode))failures.push(`<tr><td>${esc(c.text)}</td><td>${schemeNames[s]} / ${policyName(s,policy,data.reports[s])}</td><td>${badge(c.mode)} → ${badge(c[policy].mode)}</td></tr>`);
  }
  $('feedback-evaluation').innerHTML=evalRows.length?table(['方案','策略','全量准确率','聊天误触发','拒识率','动作召回','留出数'],evalRows):'<p class="empty">没有已生效的纠错回归留出。训练和验证句不在此报准确率。</p>';
  $('feedback-errors').innerHTML=failures.length?table(['回归原句','方案','真实 → 预测'],failures):'<p class="empty">暂无可展示的回归误判；缺少留出不代表准确。</p>';
  $('feedback-leaks').innerHTML=evaluation?.near_duplicates?.length?table(['留出句','最近训练句','余弦'],evaluation.near_duplicates.map(c=>`<tr><td>${esc(c.text)}</td><td>${esc(c.nearest_text)}</td><td>${c.similarity}</td></tr>`)):'<p class="empty">暂无语义近重复告警；不保证没有语义泄漏。</p>';
  updateFeedbackControls();
}

async function refreshAnalysis() {
  comparisonData=await api('/api/comparison');feedbackData=await api('/api/feedback');data=await api(modelPath('/api/state'));
  renderStatic();await renderLayout();
}

async function saveCorrection(input=null) {
  if(busy||!data||!correctionTarget)throw new Error('请先选择一个原句');
  const target={...correctionTarget};
  const request=input||{text:target.text,mode:$('feedback-mode').value,split:$('feedback-split').value,source:$('feedback-source').value,family:$('feedback-family').value};
  if(!Object.hasOwn(modeNames,request.mode))throw new Error('请人工选择正确意图');
  error();setBusy(true);
  try {
    const result=await api('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
    await refreshAnalysis();
    const q=data.queries.find(p=>p.id===target.id);
    if(q)renderProbe(q);else setupFeedbackEditor(result.record,result.record);
    $('status').textContent='人工标签已归档，尚未用于训练。打开“纠错训练”，点击重新训练使两个分类器候选同时更新。';
    return result;
  } finally {setBusy(false);}
}

async function trainCorrections() {
  if(busy||!data||!feedbackData?.count)throw new Error('请先归档人工标注句子');
  const target=correctionTarget?.id;
  error();setBusy(true);$('train-feedback').textContent='正在训练并对照评测…';
  try {
    await api('/api/feedback/train',{method:'POST'});await refreshAnalysis();
    const q=data.queries.find(p=>p.id===target)||data.queries.at(-1);if(q)renderProbe(q);
    $('status').textContent='small / large 分类器候选已同步更新；查看“纠错训练”的前后指标。原向量方案、固定案例和机器人路由不变。';
  } finally {setBusy(false);$('train-feedback').textContent='用归档数据重新训练 small / large';}
}

function renderReal() {
  const r=data.real_report;
  if(!r.ready){$('real-note').textContent='尚未通过本页导入独立 ASR 数据。纠错归档中人工确认的 ASR 记录只在纠错回归页展示，也不等于新盲测。合成/纠错回归成绩均不代表生产准确率。';for(const id of ['real-counts','real-summary','real-errors','real-leaks'])$(id).innerHTML='<p class="empty">等待独立人工标注数据</p>';return;}
  $('real-note').textContent=`${r.dataset_id} · ${detailModel()} · 真实留出 ${r.test_count} 条 · ${r.previously_evaluated?'此批含已评测测试句/表达家族，只作回归。':'此次首次登记这些测试句；查看后即不再是新盲测。'} ${r.note}`;
  $('real-counts').innerHTML=table(['集合',...Object.values(modeNames)],Object.entries(r.counts).map(([split,counts])=>`<tr><td>${splitNames[split]}</td>${Object.keys(modeNames).map(m=>`<td>${counts[m]||0}</td>`).join('')}</tr>`));
  const rows=[],errors=[];
  for(const [s,report]of Object.entries(r.reports))for(const policy of ['default','calibrated']){
    const m=report[policy];rows.push(`<tr><td>${schemeNames[s]}</td><td>${policyName(s,policy,data.reports[s])}</td><td>${percent(m.accuracy)}</td><td>${m.macro_f1.toFixed(3)}</td><td>${m.unsafe_chat_activation}</td><td>${percent(m.rejection_rate)}</td><td>${percent(m.action_recall)}</td><td>${m.count}</td></tr>`);
    for(const c of report.cases.filter(c=>c[policy].mode!==c.mode))errors.push(`<tr><td>${esc(c.text)}</td><td>${schemeNames[s]} / ${policyName(s,policy,data.reports[s])}</td><td>${badge(c.mode)} → ${badge(c[policy].mode)}</td><td>${c[policy].score} / ${c[policy].margin}</td></tr>`);
  }
  $('real-summary').innerHTML=rows.length?table(['方案','决策策略','全量准确率','Macro F1','聊天误触发动作','拒识率','动作召回率','测试数'],rows):'<p class="empty">此批没有测试句；只用于训练与验证，尚无真实测试准确率。</p>';
  $('real-errors').innerHTML=errors.length?table(['原句','方案','真实 → 预测','分数 / 差值'],errors):'<p class="empty">无可展示的测试误判；缺少测试数据时不表示准确。</p>';
  $('real-leaks').innerHTML=r.near_duplicates.length?table(['测试原句','最近训练句','余弦'],r.near_duplicates.map(c=>`<tr><td>${esc(c.text)}</td><td>${esc(c.nearest_text)}</td><td>${c.similarity}</td></tr>`)):'<p class="empty">未检出余弦 ≥ 0.97 的训练近重复；这不能证明不存在语义泄漏。</p>';
}

async function importRealDataset(text) {
  if(busy||!data)throw new Error('模型尚未就绪');
  error();setBusy(true);$('import-real').textContent='正在校验、训练并评测…';
  try {
    const result=await api('/api/real-dataset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset:text})});
    await refreshAnalysis();
    if(data.queries.length)renderProbe(data.queries.at(-1));
    $('status').textContent=`已导入 ${result.imported} 条人工标注 ASR 句。原有向量库与散点案例不变；分类器候选参数已更新，未上线。`;
    return result;
  } finally {setBusy(false);$('import-real').textContent='导入并对比两模型';}
}

async function renderLayout() {
  const request=++layoutRequest;
  const params=new URLSearchParams({scope:$('scope').value,perplexity:$('perplexity').value,seed:$('seed').value});
  $('plot-meta').textContent='正在计算 t-SNE…';
  const models=comparisonData?.ready?['small','large']:['small'];
  const results=await Promise.all(models.map(async model=>[model,await api(modelPath(`/api/layout?${params}`,model))]));
  if (request!==layoutRequest) return;
  layoutsByModel=Object.fromEntries(results);
  currentLayout=layoutsByModel[detailModel()] || layoutsByModel.small;
  for (const [model,layout] of results) {
    queriesByModel[model]=layout.points.filter(p=>p.kind==='probe');
    await drawLayout(layout,model);
  }
  $('plot-meta').textContent=results.map(([model,l])=>`${model}: ${l.points.length} 点 · 邻域保持度 ${l.trustworthiness} · ${l.elapsed_ms} ms`).join('　|　');
}

async function drawLayout(layout,model) {
  const styles=getComputedStyle(document.documentElement), traces=[];
  for (const mode of Object.keys(modeNames)) {
    const points=layout.points.filter(p=>p.kind==='case'&&p.mode===mode);
    if (!points.length) continue;
    traces.push({type:'scatter',mode:'markers',name:modeNames[mode],x:points.map(p=>p.x),y:points.map(p=>p.y),customdata:points.map(p=>[esc(p.text),esc(splitNames[p.split]),esc(p.subtype),p.id]),
      marker:{color:styles.getPropertyValue(`--${mode}`).trim(),size:points.map(p=>p.is_anchor?11:7),opacity:.78,symbol:points.map(p=>p.is_anchor?'diamond':p.split==='test'?'square-open':p.split==='validation'?'triangle-up-open':'circle'),line:{width:1}},
      hovertemplate:'%{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}<extra>'+modeNames[mode]+'</extra>'});
  }
  const probes=layout.points.filter(p=>p.kind==='probe');
  if (probes.length) traces.push({type:'scatter',mode:'markers',name:'新句探针',x:probes.map(p=>p.x),y:probes.map(p=>p.y),customdata:probes.map(p=>[esc(p.text),p.expected_mode?`期望 ${modeNames[p.expected_mode]}`:'未标注',modeNames[p.mode],p.id]),marker:{color:styles.getPropertyValue('--probe').trim(),size:15,symbol:'star',line:{color:'#0a1020',width:1}},hovertemplate:'%{customdata[0]}<br>%{customdata[1]} · 预测 %{customdata[2]}<extra>新句探针</extra>'});
  const target=model==='small'?'scatter':'scatter-large';
  await Plotly.react(target,traces,{paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{family:'-apple-system,PingFang SC,sans-serif',size:13,color:styles.getPropertyValue('--muted').trim()},margin:{l:45,r:10,t:15,b:45},showlegend:false,hovermode:'closest',dragmode:'pan',uirevision:`${model}:${layout.scope}:${layout.perplexity}:${layout.seed}:${layout.query_revision}`,xaxis:{title:{text:'t-SNE 1（无物理单位）'},gridcolor:'#26344b',zeroline:false},yaxis:{title:{text:'t-SNE 2'},gridcolor:'#26344b',zeroline:false}},{responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d','toImage']});
  $(target).removeAllListeners('plotly_click');
  $(target).on('plotly_click',async ev=>{try{await changeModel(model);showSelection(ev.points[0].customdata[3]);}catch(e){error(e.message);}});
}

async function changeModel(model) {
  if (busy || !data || model===data.model_key || (model==='small'&&!data.model_key)) return;
  const previous=data.model_key || 'small';
  setBusy(true);
  try {
    const snapshot=await api(modelPath('/api/state',model));
    $('detail-model').value=model;data=snapshot;
    currentLayout=layoutsByModel[model];renderStatic();
    if(data.queries.length)renderProbe(data.queries.at(-1));
  } catch(e) {$('detail-model').value=previous;throw e;} finally {setBusy(false);}
}

function showSelection(id) {
  const point=currentLayout?.points.find(p=>p.id===id)||data.cases.find(p=>p.id===id)||data.queries.find(p=>p.id===id);
  if (!point) return;
  $('selection').innerHTML=`<strong>${esc(point.text)}</strong><span class="selection-meta">${badge(point.mode)} · ${esc(splitNames[point.split])} · ${esc(point.family||'当前会话探针')} ${point.is_anchor?'· 代表锚点':''} ${point.expected_mode?`· 人工期望 ${esc(modeNames[point.expected_mode])}`:''}</span>`;
  if (point.split==='probe') renderProbe(point);
}

async function submitTexts(texts,expected=null) {
  if (busy || !data) throw new Error('模型未就绪或正在分析，请稍后再试');
  if (!Array.isArray(texts)||!texts.length||texts.length>20||texts.some(t=>typeof t!=='string'||!t.trim()||t.length>256)) throw new Error('每条句子需为 1–256 字符，一次最多 20 条');
  if (expected!==null&&!Object.hasOwn(modeNames,expected)) throw new Error('期望意图必须为五类之一');
  error();setBusy(true);$('send').textContent='正在分析并生成散点…';
  try {
    const result=await api('/api/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({texts,expected_mode:expected})});
    if(result.by_model)for(const [model,r]of Object.entries(result.by_model))queriesByModel[model]=r.queries;
    const selected=result.by_model?.[detailModel()] || result;
    data.queries=selected.queries;data.probe_report=selected.probe_report;renderHistory();renderProbeReport();renderProbe(selected.added.at(-1));
    await renderLayout();showSelection(selected.added.at(-1).id);
    $('status').textContent=`已分析 ${selected.added.length} 条新句${comparisonData?.ready?'，small / large 同时更新':''}。独立测试报告保持不变；新句没有自动入库。`;
    return {analyzed:selected.added.map(q=>({text:q.text,expected_mode:q.expected_mode,results:q.results})),by_model:result.by_model,probe_report:selected.probe_report};
  } finally {setBusy(false);$('send').textContent='发送并分析';}
}

async function watchBuild() {
  clearTimeout(buildPoll);
  try {
    const status=await api('/api/status');$('status').textContent=status.message;
    if (status.error) {error(status.error);setBusy(false);return;}
    if (!status.ready) {setBusy(true);buildPoll=setTimeout(watchBuild,800);return;}
    comparisonData=await api('/api/comparison');
    $('detail-model').querySelector('option[value="large"]').disabled=!comparisonData.ready;
    if(!comparisonData.ready)$('detail-model').value='small';
    data=await api(modelPath('/api/state'));feedbackData=await api('/api/feedback');renderStatic();await renderLayout();setBusy(false);
    if (data.queries.length) renderProbe(data.queries.at(-1));
  } catch(e) {error(e.message);setBusy(false);}
}

$('message-form').addEventListener('submit',async e=>{e.preventDefault();try{await submitTexts($('message').value.split(/\r?\n/).map(t=>t.trim()).filter(Boolean),$('expected').value||null);}catch(err){error(err.message);}});
$('real-form').addEventListener('submit',async e=>{e.preventDefault();try{await importRealDataset($('real-json').value);}catch(err){error(err.message);}});
$('feedback-form').addEventListener('submit',async e=>{e.preventDefault();try{await saveCorrection();}catch(err){error(err.message);}});
$('train-feedback').addEventListener('click',async()=>{try{await trainCorrections();}catch(err){error(err.message);}});
$('real-file').addEventListener('change',async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>500000)throw new Error('JSON 文件最大 500 KB');$('real-json').value=await file.text();error();}catch(err){error(err.message);}});
for (const id of ['scope','perplexity','seed']) $(id).addEventListener('change',async()=>{if(!data||busy)return;setBusy(true);try{await renderLayout();}catch(e){error(e.message);}finally{setBusy(false);}});
for (const id of ['report-scheme','report-threshold']) $(id).addEventListener('change',()=>{renderReport();renderComparison();});
$('detail-model').addEventListener('change',async()=>{try{await changeModel(detailModel());}catch(e){error(e.message);}});
$('rebuild').addEventListener('click',async()=>{error();setBusy(true);try{await api('/api/build',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({anchors_per_intent:Number($('anchor-k').value)})});watchBuild();}catch(e){error(e.message);setBusy(false);}});
document.addEventListener('click',e=>{const feedback=e.target.closest('[data-feedback]');if(feedback){if(busy)return;const c=feedbackData?.cases.find(c=>c.id===feedback.dataset.feedback);if(c){const q=data.queries.find(q=>q.normalized_text===c.normalized_text||q.text===c.text);if(q)renderProbe(q);else{$('probe-result').innerHTML='<p class="help">该归档句不在当前探针内存中；重新发送可看实时分类。下方只编辑人工标注。</p>';setupFeedbackEditor(c,c);}}return;}const probe=e.target.closest('[data-probe]');if(probe){const q=data?.queries.find(q=>q.id===probe.dataset.probe);if(q){renderProbe(q);showSelection(q.id);}return;}const point=e.target.closest('[data-point]');if(point&&data)showSelection(point.dataset.point);const tab=e.target.closest('[role=tab]');if(tab){for(const t of document.querySelectorAll('[role=tab]')){const selected=t===tab;t.setAttribute('aria-selected',String(selected));$(t.getAttribute('aria-controls')).hidden=!selected;}}});
document.querySelector('[role=tablist]').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;const tabs=[...document.querySelectorAll('[role=tab]')],i=tabs.indexOf(document.activeElement);if(i<0)return;e.preventDefault();const j=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[j].focus();tabs[j].click();});

// Optional browser-native agent surface. Uses the exact same visible workflow.
if (document.modelContext?.registerTool) {
  const lifecycle=new AbortController();
  const tool={name:'analyze_intent_utterances',title:'分析意图测试句',description:'分析 1–20 条原句，对比三个向量方案和轻量分类器（可拒识），更新交互 t-SNE 与人工标注探针统计；不修改线上样例或执行动作。',inputSchema:{type:'object',properties:{texts:{type:'array',minItems:1,maxItems:20,items:{type:'string',minLength:1,maxLength:256}},expected_mode:{type:'string',enum:Object.keys(modeNames)}},required:['texts'],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:true},execute:input=>submitTexts(input?.texts,input?.expected_mode??null)};
  try {Promise.resolve(document.modelContext.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});} catch (_) { /* unsupported implementations must not block the app */ }
  window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
}
watchBuild();
