// CPU-only browser verification against saved native conversation content.
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert/strict');

(async () => {
  const output = path.resolve(process.argv[2] || 'behavior_evaluator/validation/trajectory-viewer');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_BIN,
    headless: true, args: ['--disable-gpu', '--disable-dev-shm-usage', '--no-sandbox']});
  const page = await browser.newPage({viewport: {width: 1560, height: 1120}}), errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    if (response.status() >= 400 && !response.url().endsWith('/favicon.ico')) errors.push(`${response.status()} ${response.url()}`);
  });
  const base = 'http://127.0.0.1:18792';
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'], {origin: base});
  await page.goto(base + '/?id=task01-standalone-v7%2Finstance_301');
  await page.waitForFunction(() => document.querySelector('#connection').textContent === '已同步');
  await page.locator('#live').uncheck();
  const snapshot = await page.evaluate(() => transcript);
  const calls = snapshot.entries.filter(entry => entry.kind === 'tool');
  assert.equal(await page.locator('#conversation-list>[data-kind=tool]').count(), snapshot.tool_count);
  assert(calls.some(entry => entry.call.name === 'Read'));
  assert(calls.some(entry => entry.call.input.command?.includes('--schema')));
  assert.equal(await page.locator('.conversation-head>span').count(), 4);
  const discrepancies = await page.evaluate(() => transcript.entries.filter(entry => entry.kind === 'tool').flatMap(entry => {
    const row = document.getElementById(entry.id), issues = [];
    function imageURLs(block) {
      if (!block || typeof block !== 'object') return [];
      if (block.type === 'image') return [block.source?.image_url || block.source?.url].filter(Boolean);
      if (block.type !== 'tool_result') return [];
      return (Array.isArray(block.content) ? block.content : [block.content]).flatMap(imageURLs);
    }
    if (entry.call.name === 'Bash' && typeof entry.call.input.command === 'string' && row.querySelector('.exact-command').textContent !== entry.call.input.command) issues.push(entry.id + ': changed command');
    if (entry.call.input.__unparsedToolInput && !row.querySelector('.call-cell').textContent.includes(entry.call.input.__unparsedToolInput.raw)) issues.push(entry.id + ': lost malformed call');
    if (!entry.shared_before) {
      for (const item of entry.before) {
        const text = item.block.thinking ?? item.block.text;
        if (text && !row.querySelector('.thinking-cell').textContent.includes(text)) issues.push(entry.id + ': lost preceding content');
      }
    }
    const expectedImages = entry.results.flatMap(result => imageURLs(result.block));
    const shownImages = [...row.querySelectorAll('.images-cell img')].map(image => image.getAttribute('src'));
    if (JSON.stringify(shownImages) !== JSON.stringify(expectedImages)) issues.push(entry.id + ': changed image count, source or order');
    if (row.querySelector('.result-cell>figure.context-image')) issues.push(entry.id + ': returned image left in text column');
    if (row.querySelectorAll(':scope>section').length !== 4) issues.push(entry.id + ': expected four cells');
    return issues;
  }));
  assert.deepEqual(discrepancies, []);
  const rowHeights = await page.locator('#conversation-list>[data-kind=tool]').evaluateAll(rows => rows.map(row => [...row.children].map(cell => cell.getBoundingClientRect().height)));
  assert(rowHeights.every(heights => heights.length === 4 && heights.every(height => height === 360)), 'all four cells use a consistent default height');
  assert.equal(await page.locator('#' + calls[0].id).evaluate(row => row.nextElementSibling.id), calls[1].id, 'two calls from the same response are consecutive rows');
  assert.equal(await page.locator('#calls-tab .time').count(), 0);
  assert.equal(await page.locator('#calls-tab .record').count(), 0);
  await page.screenshot({path: path.join(output, 'all-tools.png'), fullPage: false});

  const selected = calls.find(entry => entry.robot_records.includes('000007_spin_to_facing_point'));
  assert(selected);
  await page.evaluate(call => selectCall(call), '000007_spin_to_facing_point');
  const row = page.locator('#' + selected.id);
  const boxes = await Promise.all(['.thinking-cell', '.call-cell', '.result-cell', '.images-cell'].map(selector => row.locator(selector).boundingBox()));
  assert(boxes.every((box, index) => index === 0 || boxes[index - 1].x < box.x));
  assert(boxes.every(box => box.y === boxes[0].y));
  assert.equal(await row.locator('.exact-command').textContent(), selected.call.input.command);
  await row.getByRole('button', {name: '复制命令', exact: true}).click();
  assert.equal(await page.evaluate(() => navigator.clipboard.readText()), selected.call.input.command);
  const thumbnail = row.locator('.images-cell img').first();
  await thumbnail.scrollIntoViewIfNeeded();
  await thumbnail.evaluate(image => image.decode());
  assert.equal(await thumbnail.evaluate(image => image.naturalWidth), 720);
  assert((await thumbnail.boundingBox()).width <= 160);
  const source = selected.results.flatMap(result => result.block.content).find(block => block.type === 'image').source;
  const imageResponse = await page.request.get(base + source.image_url);
  assert.equal(crypto.createHash('sha256').update(await imageResponse.body()).digest('hex'), source.sha256);
  await thumbnail.click();
  await page.waitForFunction(() => document.querySelector('#full-image').naturalWidth === 720);
  assert(await page.locator('#image-dialog').isVisible());
  await page.locator('#close-image').click();
  await page.evaluate(call => selectCall(call), selected.id);
  await page.screenshot({path: path.join(output, 'calls.png'), fullPage: false});

  await page.locator('#call-filter').selectOption('action');
  assert.equal(await page.locator('#conversation-list>[data-kind=tool]').count(), snapshot.action_count);
  assert.equal(await page.locator('#conversation-list>[data-action=false]').count(), 0);
  const firstAction = calls.find(entry => entry.action);
  assert(firstAction.shared_before);
  assert.match(await page.locator('#' + firstAction.id + ' .thinking-cell').textContent(), /Let's start by reading/);
  await page.evaluate(call => selectCall(call), firstAction.id);
  await page.screenshot({path: path.join(output, 'action-only.png'), fullPage: false});
  await page.locator('#call-filter').selectOption('all');
  await page.locator('#call-search').fill('impossible-unique-no-result-123');
  assert.equal(await page.locator('#conversation-list>[data-kind=tool]').count(), 0);
  await page.locator('#call-search').fill('');
  assert.equal(await page.locator('#conversation-list>[data-kind=tool]').count(), snapshot.tool_count);

  const long = calls.find(entry => !entry.shared_before && entry.before.some(item => (item.block.thinking || '').length > 1400));
  await page.evaluate(call => selectCall(call), long.id);
  await page.locator('#' + long.id + ' .thinking-cell .expand-content').first().click();
  const beforeRefresh = await page.evaluate(call => {
    const el = document.getElementById(call);
    el.dataset.preserved = 'yes';
    return {scroll: window.scrollY, text: el.querySelector('.thinking-cell').textContent};
  }, long.id);
  await Promise.all([
    page.waitForResponse(response => response.url().includes('/api/conversation?') && response.status() === 200),
    page.evaluate(() => document.getElementById('refresh').click()),
  ]);
  await page.waitForFunction(() => document.querySelector('#connection').textContent === '已同步');
  assert.equal(await page.locator('#' + long.id).getAttribute('data-preserved'), 'yes');
  assert.equal(await page.locator('#' + long.id + ' .thinking-cell .expand-content').first().getAttribute('aria-expanded'), 'true');
  assert.equal(await page.locator('#' + long.id + ' .thinking-cell').textContent(), beforeRefresh.text);
  assert(Math.abs((await page.evaluate(() => window.scrollY)) - beforeRefresh.scroll) < 3);

  const user = snapshot.entries.find(entry => entry.kind === 'user');
  await page.locator('#' + user.id + '>summary').click();
  assert((await page.locator('#' + user.id).textContent()).includes(user.chunks[0].blocks[0].text));
  assert.equal(await page.locator('.compact-boundary').count(), snapshot.entries.filter(e => e.kind === 'boundary').length);
  await page.getByRole('button', {name: '工作区访问', exact: true}).click();
  await page.locator('#access-kind').selectOption('read');
  assert.match(await page.locator('#access-list').innerText(), /tools\/README.md/);
  await page.locator('#access-list button').first().click();
  await page.waitForSelector('#native-16[open]');
  assert.match(await page.locator('#native-16').innerText(), /tools\/README.md/);
  await page.evaluate(() => jumpNative(84));
  await page.waitForSelector('#native-84[open]');
  await page.locator('#native-84 img').first().evaluate(image => image.decode());
  assert.equal(await page.locator('#native-84 img').first().evaluate(image => image.naturalWidth), 720);
  await page.getByRole('button', {name: '说明与成绩', exact: true}).click();
  await page.locator('#prompt-details summary').click();
  await page.waitForFunction(() => document.querySelector('#prompt-text').textContent.includes('three soda cans'));

  await page.getByRole('button', {name: '动作轨迹', exact: true}).click();
  await page.setViewportSize({width: 390, height: 844});
  await page.evaluate(call => selectCall(call), selected.id);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: false});
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(output, 'browser.json'), JSON.stringify({passed: true,
    tool_count: snapshot.tool_count, action_count: snapshot.action_count,
    checks: ['every native tool rendered, including Read and schema',
      'every Bash command and preceding thinking checked against native transcript',
      'four columns in requested order', 'uniform 360px rows with consecutive tools kept adjacent', 'clipboard preserves exact shell command',
      'returned image sources and order match every native result in the fourth column', 'native image SHA256 and original 720px verified',
      'small thumbnails with original-resolution modal', 'Action only filter with shared thinking retained',
      'search/reset keeps all calls', 'refresh preserves expanded content, DOM and scroll position',
      'user prompt, summaries and context boundaries retained', 'workspace evidence links to native messages',
      'full task prompt available', 'mobile without horizontal overflow'], errors}, null, 2));
  await browser.close();
  console.log('Browser checks passed; ' + output);
})().catch(error => { console.error(error); process.exit(1); });
