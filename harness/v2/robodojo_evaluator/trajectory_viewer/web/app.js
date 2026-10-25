'use strict';
const $ = id => document.getElementById(id);
const node = (tag, text, cls) => {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
};
const pretty = value => JSON.stringify(value, null, 2);
const when = value => value ? new Date(typeof value === 'number' ? value * 1000 : value).toLocaleString('zh-CN', {timeZone: 'Asia/Shanghai', hour12: false}) : '—';
const statusLabel = {running: '运行中', loading: '加载中', completed: '已结束', incomplete: '未完成 / 中断'};
const kinds = {read: '读取文本 / 原生 Read', search: '搜索内容 / 文件', schema: '读取参数 schema', execute: '执行 Python 工具', process_image: '程序打开图片', program_read: '程序读取数据', import: '导入本地函数'};
const contextLabels = {hook_additional_context: 'Hook / Skill 上下文', skill_listing: '可用 Skills', total_tokens_reminder: 'Token 提醒', compact_file_reference: '压缩后的文件引用'};
kinds.view_image = '原生查看图片';
let catalog = [], data = null, transcript = null, id = '', selected = '', tab = 'calls';
let nativePage = 0, busy = false, epoch = 0, renderedNative = '', promptSession = '';
const renderedRows = new Map();

async function api(path, params = {}) {
  const r = await fetch(path + '?' + new URLSearchParams(params), {cache: 'no-store'});
  if (!r.ok) throw new Error('读取失败 HTTP ' + r.status);
  return r.json();
}

function stat(label, value, hint) {
  const el = node('div', undefined, 'stat');
  el.append(node('div', label, 'stat-label'), node('div', String(value), 'stat-value'));
  if (hint) el.append(node('small', hint));
  return el;
}

function jsonDetails(label, value) {
  const el = node('details');
  el.append(node('summary', label), node('pre', pretty(value)));
  return el;
}

function setLocation() {
  const u = new URL(location.href);
  u.searchParams.set('id', id);
  u.searchParams.set('tab', tab);
  if (selected) u.searchParams.set('call', selected); else u.searchParams.delete('call');
  history.replaceState(null, '', u);
}

function openImage(url, label) {
  $('full-image').src = url;
  $('image-caption').textContent = label;
  $('image-original').href = url;
  $('image-dialog').showModal();
}

function imageBlock(url, label = '模型收到的图片') {
  const figure = node('figure', undefined, 'context-image'), image = node('img');
  const caption = node('figcaption', label);
  image.src = url;
  image.alt = label;
  image.loading = 'lazy';
  image.tabIndex = 0;
  image.onload = () => { caption.textContent = `${label} · ${image.naturalWidth} × ${image.naturalHeight}`; };
  image.onclick = () => openImage(url, caption.textContent);
  image.onkeydown = e => { if (e.key === 'Enter') image.click(); };
  figure.append(image, caption);
  return figure;
}

// Original strings remain in the API. Present JSON tool output as fields,
// without a transport/event JSON envelope.
function fields(value) {
  if (value === null || typeof value !== 'object') {
    return node('span', value === null ? 'null' : value === '' ? '（空字符串）' : String(value), 'field-scalar');
  }
  if (Array.isArray(value) && value.every(x => x === null || typeof x !== 'object')) {
    return node('span', '[' + value.map(x => typeof x === 'string' ? JSON.stringify(x) : String(x)).join(', ') + ']', 'field-scalar');
  }
  const box = node('div', undefined, 'result-fields');
  if (!Object.keys(value).length) box.append(node('span', Array.isArray(value) ? '[]' : '{}', 'field-scalar'));
  for (const [key, item] of Object.entries(value)) {
    const row = node('div', undefined, 'field');
    row.append(node('div', key, 'field-key'), fields(item));
    box.append(row);
  }
  return box;
}

function scrollContent(content, long = false, cls = '') {
  const box = node('div', undefined, 'content-box ' + cls);
  const body = node('div', undefined, 'scroll-content' + (long ? ' constrained' : ''));
  body.append(content);
  box.append(body);
  if (long) {
    const button = node('button', '展开全文', 'expand-content');
    button.setAttribute('aria-expanded', 'false');
    button.onclick = () => {
      const expanded = body.classList.toggle('expanded');
      button.textContent = expanded ? '收起长文本' : '展开全文';
      button.setAttribute('aria-expanded', String(expanded));
      const row = body.closest('.trace-row');
      if (expanded && row && !row.classList.contains('expanded-row')) row.querySelector('.row-height-button')?.click();
    };
    box.append(button);
  }
  return box;
}

function textContent(text, cls = '', structured = false) {
  let content = node('pre', text, cls);
  if (structured) {
    try {
      const parsed = JSON.parse(text);
      if (parsed !== null && typeof parsed === 'object') content = fields(parsed);
    } catch { /* Plain stdout remains literal text. */ }
  }
  return scrollContent(content, text.length > 800 || text.split('\n').length > 16, cls);
}

function renderContent(block, container, structured = true, imageContainer = container) {
  if (typeof block === 'string') { container.append(textContent(block, '', structured)); return; }
  if (!block || typeof block !== 'object') return;
  if (block.type === 'thinking') {
    if (block.label) container.append(node('div', block.label, 'block-label'));
    container.append(textContent(block.thinking || '', 'thinking'));
  } else if (block.type === 'text') {
    if (block.exit_code !== undefined && block.exit_code !== null && block.exit_code !== 0) container.append(node('span', 'Exit code ' + block.exit_code, 'return-error'));
    container.append(textContent(block.text || '', 'message-text', structured));
  }
  else if (block.type === 'image') {
    const url = block.source?.image_url || block.source?.url;
    if (url) imageContainer.append(imageBlock(url));
    else imageContainer.append(node('p', '日志没有保存可读取的图像内容。', 'muted'));
  } else if (block.type === 'tool_result') {
    if (block.is_error) container.append(node('span', '工具返回错误', 'return-error'));
    const content = Array.isArray(block.content) ? block.content : [block.content];
    for (const item of content) renderContent(item, container, structured, imageContainer);
  } else if (block.type === 'redacted_thinking') {
    container.append(node('p', block.text || '此段 Thinking 未公开保存。', 'muted'));
  } else {
    container.append(node('div', block.type || '内容', 'block-label'), scrollContent(fields(block), true));
  }
}

function copyButton(value, label) {
  const button = node('button', label, 'copy-button');
  button.onclick = async () => {
    try {
      await navigator.clipboard.writeText(value);
      button.textContent = '已复制';
      setTimeout(() => { button.textContent = label; }, 1800);
    } catch { button.textContent = '请选中文本复制'; }
  };
  return button;
}

function contextChunks(chunks, label = '补充上下文') {
  const box = node('details', undefined, 'context-chunks');
  box.append(node('summary', label));
  const body = node('div', undefined, 'context-body');
  for (const chunk of chunks) {
    body.append(node('div', contextLabels[chunk.label] || chunk.label, 'block-label'));
    for (const block of chunk.blocks) renderContent(block, body);
  }
  box.append(body);
  return box;
}

function toolRow(entry, visible) {
  const row = node('article', undefined, 'trace-row');
  row.id = entry.id;
  row.dataset.kind = entry.kind;
  row.dataset.action = String(!!entry.action);
  row.dataset.line = entry.line;
  const thinking = node('section', undefined, 'thinking-cell');
  const call = node('section', undefined, 'call-cell');
  const result = node('section', undefined, 'result-cell');
  const images = node('section', undefined, 'images-cell');
  thinking.setAttribute('aria-label', 'Thinking');
  call.setAttribute('aria-label', 'Tool call');
  result.setAttribute('aria-label', 'Tool result · 文字返回');
  images.setAttribute('aria-label', 'Images · 工具返图');
  if (entry.kind === 'tool') {
    const title = node('div', undefined, 'tool-title');
    title.append(node('span', '#' + entry.number, 'step-number'), node('strong', entry.call.name));
    call.append(title);
    if (entry.robot_tools.length) call.append(node('div', entry.robot_tools.join(' · '), 'robot-tool-name'));
    const args = entry.call.input ?? {};
    if (typeof args.code === 'string') {
      for (const command of args.commands || []) {
        call.append(node('div', 'Bash · 调用中的原始命令', 'block-label'), node('pre', command, 'exact-command'), copyButton(command, '复制命令'));
      }
      const code = node('details');
      code.open = !(args.commands || []).length;
      code.append(node('summary', '完整原生工具调用 · ' + entry.call.name), node('pre', args.code, 'exact-command'), copyButton(args.code, '复制原始调用'));
      call.append(code);
    } else if (entry.call.name === 'Bash' && typeof args.command === 'string') {
      call.append(node('pre', args.command, 'exact-command'), copyButton(args.command, '复制命令'));
      const other = Object.fromEntries(Object.entries(args).filter(([key]) => key !== 'command'));
      if (Object.keys(other).length) call.append(fields(other));
    } else {
      if (args.__unparsedToolInput) call.append(node('div', '工具参数解析失败 · 原生日志中的输入', 'return-error'));
      call.append(scrollContent(fields(args), JSON.stringify(args).length > 1400), copyButton(JSON.stringify(args), '复制参数'));
    }
    if (entry.shared_before && visible.has(entry.shared_before)) {
      const shared = node('button', '同一轮模型回复 · 查看上方原文', 'shared-thinking');
      shared.onclick = () => selectCall(entry.shared_before);
      thinking.append(shared);
    } else {
      if (entry.shared_before) thinking.append(node('div', '同一轮回复中的共享原文', 'block-label'));
      for (const item of entry.before) {
        if (item.block.type === 'text') thinking.append(node('div', 'Assistant', 'block-label'));
        renderContent(item.block, thinking, false);
      }
      if (!entry.before.length) thinking.append(node('p', '这次调用前没有已保存的模型文字。', 'muted'));
    }
    for (const output of entry.results) renderContent(output.block, result, true, images);
    if (!entry.results.length) {
      const completed = catalog.find(s => s.id === id)?.status === 'completed';
      result.append(node('p', completed ? '评测已结束；日志未保存此调用的返回。' : '尚未记录工具返回…', 'pending-return'));
    }
    if (entry.after.length) result.append(contextChunks(entry.after));
  } else if (entry.kind === 'assistant') {
    call.append(node('div', 'Assistant', 'tool-title'));
    for (const item of entry.before) renderContent(item.block, item.block.type === 'thinking' ? thinking : call, false);
  } else {
    call.append(node('div', '工具返回 · 对应调用未保存在此日志', 'muted'));
    for (const output of entry.results) renderContent(output.block, result, true, images);
  }
  const heightButton = node('button', '展开本次调用', 'row-height-button');
  heightButton.setAttribute('aria-expanded', 'false');
  heightButton.onclick = () => {
    const expanded = row.classList.toggle('expanded-row');
    heightButton.textContent = expanded ? '恢复统一行高' : '展开本次调用';
    heightButton.setAttribute('aria-expanded', String(expanded));
  };
  (call.querySelector('.tool-title') || call).append(heightButton);
  row.append(thinking, call, result, images);
  return row;
}

function conversationRow(entry, visible) {
  if (['tool', 'assistant', 'orphan_result'].includes(entry.kind)) return toolRow(entry, visible);
  let row;
  if (entry.kind === 'boundary') {
    row = node('div', '上下文已压缩 · 后续从摘要及保留消息继续', 'compact-boundary');
  } else {
    const label = entry.kind === 'user' ? 'User · 输入消息' : entry.kind === 'summary' ? '压缩摘要 · 完整原文' : '上下文 · Skill / Hook / 提醒';
    row = contextChunks(entry.chunks, label);
    row.classList.add('context-row');
  }
  row.id = entry.id;
  row.dataset.kind = entry.kind;
  return row;
}

function visibleEntries() {
  const q = $('call-search').value.trim().toLowerCase();
  const actionOnly = $('call-filter').value === 'action';
  return (transcript?.entries || []).filter(entry => (!actionOnly || entry.action) && (!q || JSON.stringify(entry).toLowerCase().includes(q)));
}

function preserveRowState(old, next) {
  if (old.classList.contains('expanded-row')) next.querySelector('.row-height-button')?.click();
  const oldScroll = [...old.querySelectorAll('.scroll-content')];
  const nextScroll = [...next.querySelectorAll('.scroll-content')];
  oldScroll.forEach((el, i) => {
    if (!nextScroll[i]) return;
    if (el.classList.contains('expanded')) nextScroll[i].parentNode.querySelector('.expand-content')?.click();
    nextScroll[i].scrollTop = el.scrollTop;
  });
  const oldDetails = [...old.querySelectorAll('details')];
  [...next.querySelectorAll('details')].forEach((el, i) => { el.open = !!oldDetails[i]?.open; });
}

function renderConversation() {
  const list = $('conversation-list'), entries = visibleEntries();
  const visible = new Set(entries.map(entry => entry.id));
  if (!renderedRows.size) list.replaceChildren();
  for (const [key, old] of renderedRows) {
    if (!visible.has(key)) { old.element.remove(); renderedRows.delete(key); }
  }
  let previous = null;
  for (const entry of entries) {
    const revision = entry.revision + '|' + visible.has(entry.shared_before);
    let record = renderedRows.get(entry.id);
    if (!record || record.revision !== revision) {
      const element = conversationRow(entry, visible);
      if (record) {
        const old = record.element;
        old.replaceWith(element);
        preserveRowState(old, element);
      }
      record = {revision, element};
      renderedRows.set(entry.id, record);
    }
    const cursor = previous ? previous.nextSibling : list.firstChild;
    if (cursor !== record.element) list.insertBefore(record.element, cursor);
    record.element.classList.toggle('selected', selected === entry.id || entry.robot_records?.includes(selected));
    previous = record.element;
  }
  list.querySelector('.empty')?.remove();
  if (!entries.length) list.append(node('p', '没有匹配的内容。', 'empty'));
  const shown = entries.filter(entry => entry.kind === 'tool').length;
  $('call-count').textContent = `${shown} / ${transcript?.tool_count || 0} 次工具调用`;
}

function selectCall(value) {
  if (!value || !transcript) return;
  const entry = transcript.entries.find(e => e.id === value || e.robot_records?.includes(value));
  if (!entry) return;
  selected = entry.id;
  if (!visibleEntries().some(e => e.id === entry.id)) {
    $('call-search').value = '';
    $('call-filter').value = 'all';
  }
  renderConversation();
  setLocation();
  $(entry.id)?.scrollIntoView({block: 'start'});
}

function renderCatalog() {
  $('session').replaceChildren(...catalog.map(s => new Option(`${s.run} · ${s.instance} · ${statusLabel[s.status]}${s.q !== null ? ' · Q ' + s.q.toFixed(4) : ''}${s.diagnostic ? ' · 诊断' : ''}`, s.id)));
  $('session').value = id;
}

function renderStats() {
  const meta = catalog.find(s => s.id === id) || {};
  const commands = data.calls.reduce((n, c) => n + (c.commands || 0), 0);
  $('title').textContent = (meta.run || id.split('/')[0]) + ' / ' + (meta.instance || id.split('/')[1]);
  $('subtitle').textContent = `${statusLabel[meta.status] || '已归档'} · ${transcript.tool_count} 次工具调用 · ${transcript.action_count} 次 Robot 调用`;
  const cap = data.native_capabilities || {};
  $('trace-capabilities').hidden = cap.format !== 'codex';
  $('trace-capabilities').textContent = cap.format === 'codex'
    ? `原生 Codex + Astra · ${cap.returned_image_blocks} 个实际返图块 · ${cap.reasoning_items} 个 reasoning 项，其中 ${cap.readable_reasoning_items} 个有可读文本。左列的 Assistant 是公开回复；加密 reasoning 无法显示为 thinking。` : '';
  $('stats').replaceChildren(
    stat('正式成绩', meta.q == null ? '暂无' : meta.q.toFixed(4), '接口 smoke 不属于正式任务成绩'),
    stat('模型工具调用', transcript.tool_count, '原生 Bash、Read 等全部调用'),
    stat('已记录关节命令', commands.toLocaleString(), '物理步数以 evaluator 最终记录为准'),
    stat('原生消息', data.native_total, '含运行日志与上下文注入'),
    stat('主动读取记录', data.access_counts.read || 0, '不含执行、schema 和自动注入'));
}

function jumpLink(number, label) {
  const button = node('button', label || '消息 #' + number, 'evidence-button');
  button.onclick = () => jumpNative(number);
  return button;
}

function renderAccess() {
  const rows = data.accesses, q = $('access-search').value.toLowerCase(), kind = $('access-kind').value;
  $('access-summary').replaceChildren(stat('主动读取路径', rows.filter(x => x.kind === 'read').length, 'Read / cat / sed 等'), stat('读取参数 schema', data.access_counts.schema || 0, '只展示参数，不能证明读了算法'), stat('执行工具路径', rows.filter(x => x.kind === 'execute').length, '执行与阅读分别统计'), stat('本地函数导入', data.access_counts.import || 0, '导入函数不等于阅读源码'));
  const tbody = $('access-list');
  tbody.replaceChildren();
  for (const row of rows.filter(r => (!kind || r.kind === kind) && r.path.toLowerCase().includes(q))) {
    const tr = node('tr'), evidence = node('td');
    for (const event of row.events) {
      const button = jumpLink(event.line);
      button.title = event.evidence;
      evidence.append(button);
    }
    tr.append(node('td', row.path), node('td', kinds[row.kind] || row.kind), node('td', row.events.length), evidence);
    tbody.append(tr);
  }
  if (!tbody.children.length) {
    const tr = node('tr'), td = node('td', '没有这类访问证据');
    td.colSpan = 4;
    tr.append(td);
    tbody.append(tr);
  }
  const skills = $('skill-access');
  skills.replaceChildren(node('h3', '通过 activate_skill 返回的指令'));
  for (const skill of data.activated_skills) {
    const p = node('p', skill.name + ' '), button = node('button', '查看激活调用', 'evidence-button');
    button.onclick = () => { setTab('calls'); selectCall(skill.call); };
    p.append(button);
    skills.append(p);
  }
  skills.append(node('p', data.native_capabilities?.format === 'codex'
    ? '此实验通过原生文件读取和 activate_skill 获取指令。执行记录取自 CommandExecution；查看图片取自 ImageView，均可回到原始消息核对。'
    : 'baseline 还会由原生 SessionStart Hook 注入。激活返回与 Hook 注入不计为主动文件阅读。', 'footnote'));
}

function renderAbout() {
  const summary = data.run_summary, score = $('scores');
  const details = Object.values(summary.details || {});
  score.replaceChildren(node('p', details.length ? '正式 evaluator 已结算本例' : '本例尚未保存正式成绩'));
  const table = node('table'), heading = node('tr');
  for (const title of ['Layout', '正式得分', '成功']) heading.append(node('th', title));
  table.append(heading);
  for (const value of details) {
    const tr = node('tr');
    tr.append(node('td', value.layout_id), node('td', Number(value.score).toFixed(4)), node('td', value.success ? '是' : '否'));
    table.append(tr);
  }
  score.append(table);
  $('provenance').textContent = data.workspace;
  $('prompt-link').href = data.prompt_url;
  const audit = data.delivery_audit;
  $('audit-status').textContent = audit.total_image_results !== undefined ? `最近返图审计快照：${audit.verified_including_native_read_results}/${audit.total_image_results} 次图片结果与原生会话对应。实时新增调用可能尚未审计。` : '此轨迹尚无独立返图审计快照。';
  if (promptSession !== id) {
    promptSession = id;
    $('prompt-text').textContent = '展开后读取本次保存的完整 Prompt。';
    $('prompt-details').open = false;
  }
}

function nativeRow(row, open = false) {
  const event = row.event, details = node('details', undefined, 'record');
  details.id = 'native-' + row.number;
  const summary = node('summary'), heading = node('div', undefined, 'record-title');
  const content = (row.display || event).message?.content || (row.display || event).attachment?.content;
  const types = Array.isArray(content) ? content.map(x => x.name || x.type).filter(Boolean).join(' · ') : event.payload?.type || event.subtype || '';
  heading.append(node('span', '#' + row.number, 'pill'), node('span', event.type || '事件', 'label'), node('span', when(event.timestamp), 'time'), node('span', types, 'time'));
  summary.append(heading);
  details.append(summary);
  let filled = false;
  function fill() {
    if (filled) return;
    filled = true;
    const body = node('div', undefined, 'record-content');
    if (Array.isArray(content)) {
      for (const block of content) {
        if (block.type === 'tool_use') body.append(node('div', block.name, 'block-label'), node('pre', pretty(block.input)));
        else renderContent(block, body);
      }
    } else body.append(node('pre', typeof content === 'string' ? content : pretty(event)));
    body.append(node('p', '来源：' + row.source, 'footnote'), jsonDetails('完整记录 JSON（image base64 替换为原图链接）', event));
    details.append(body);
  }
  details.addEventListener('toggle', () => { if (details.open) fill(); });
  if (open) { fill(); details.open = true; }
  return details;
}

async function loadNative() {
  const token = epoch;
  try {
    const result = await api('/api/native', {id, page: nativePage, q: $('native-search').value, kind: $('native-kind').value});
    if (token !== epoch) return;
    const key = id + '|' + result.records.map(r => r.number).join(',');
    if (renderedNative !== key) {
      const opened = new Set([...$('native-list').querySelectorAll('details[open]')].map(e => e.id));
      $('native-list').replaceChildren(...result.records.map(r => nativeRow(r, opened.has('native-' + r.number))));
      renderedNative = key;
    }
    $('native-count').textContent = `${result.total ? nativePage * 30 + 1 : 0}–${Math.min((nativePage + 1) * 30, result.total)} / ${result.total} 条`;
    $('native-prev').disabled = nativePage === 0;
    $('native-next').disabled = (nativePage + 1) * 30 >= result.total;
  } catch (e) { $('native-list').replaceChildren(node('p', e.message, 'error-text')); }
}

async function jumpNative(number) {
  $('native-search').value = '';
  $('native-kind').value = '';
  nativePage = Math.floor((number - 1) / 30);
  await setTab('native');
  const element = $('native-' + number);
  if (element) { element.open = true; element.classList.add('highlight'); element.scrollIntoView({block: 'start'}); }
}

async function setTab(value) {
  tab = value;
  setLocation();
  document.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  for (const name of ['calls', 'native', 'access', 'about']) $(name + '-tab').hidden = name !== tab;
  if (tab === 'native') await loadNative();
  if (tab === 'access') renderAccess();
  if (tab === 'about') renderAbout();
}

async function loadSession(key, keep = false) {
  const generation = ++epoch;
  id = key;
  busy = true;
  $('connection').textContent = '读取原始记录…';
  try {
    const [session, conversation] = await Promise.all([api('/api/session', {id: key}), api('/api/conversation', {id: key})]);
    if (generation !== epoch) return;
    data = session;
    transcript = conversation;
    if (!keep) { renderedRows.clear(); selected = ''; }
    renderStats();
    renderConversation();
    renderAccess();
    renderAbout();
    renderCatalog();
    await setTab(tab);
    $('connection').textContent = '已同步';
  } catch (e) { $('connection').textContent = e.message; }
  finally { if (generation === epoch) busy = false; }
}

async function refresh() {
  if (busy) return;
  try {
    catalog = await api('/api/sessions');
    renderCatalog();
    if (id) await loadSession(id, true);
  } catch (e) { $('connection').textContent = e.message; }
}

$('session').onchange = () => { nativePage = 0; loadSession($('session').value); };
$('refresh').onclick = refresh;
document.querySelectorAll('.tabs button').forEach(button => { button.onclick = () => setTab(button.dataset.tab); });
$('call-search').oninput = renderConversation;
$('call-filter').onchange = renderConversation;
$('first').onclick = () => { $('calls-tab').scrollIntoView({block: 'start'}); };
$('latest').onclick = () => { const entries = visibleEntries(); selectCall(entries.findLast(e => e.kind === 'tool')?.id); };
$('native-go').onclick = () => { nativePage = 0; loadNative(); };
$('native-search').onkeydown = event => { if (event.key === 'Enter') $('native-go').click(); };
$('native-kind').onchange = () => { nativePage = 0; loadNative(); };
$('native-prev').onclick = () => { nativePage--; loadNative(); };
$('native-next').onclick = () => { nativePage++; loadNative(); };
for (const [value, label] of Object.entries(kinds)) $('access-kind').append(new Option(label, value));
$('access-search').oninput = renderAccess;
$('access-kind').onchange = renderAccess;
$('prompt-details').ontoggle = async () => {
  if ($('prompt-details').open) {
    const key = id, result = await fetch(data.prompt_url);
    if (key === id) $('prompt-text').textContent = await result.text();
  }
};
$('close-image').onclick = () => $('image-dialog').close();
$('image-dialog').onclick = event => { if (event.target === $('image-dialog')) $('image-dialog').close(); };

(async () => {
  try {
    catalog = await api('/api/sessions');
    const params = new URLSearchParams(location.search);
    id = params.get('id') || catalog.find(s => s.status === 'running' && !s.diagnostic)?.id || catalog[0]?.id || '';
    tab = ['calls', 'native', 'access', 'about'].includes(params.get('tab')) ? params.get('tab') : 'calls';
    selected = params.get('call') || '';
    renderCatalog();
    if (id) {
      await loadSession(id, true);
      if (tab === 'calls' && selected) selectCall(selected);
    } else $('connection').textContent = '尚无已初始化的工作区轨迹。';
  } catch (e) { $('connection').textContent = e.message; }
})();
setInterval(() => { if ($('live').checked && !document.hidden) refresh(); }, 20000);
