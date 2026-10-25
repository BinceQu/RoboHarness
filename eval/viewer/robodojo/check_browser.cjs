// Verify actual native content, image hashes, layout and action filtering.
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert/strict');

(async () => {
  const base = process.env.VIEWER_URL || 'http://127.0.0.1:18793';
  const id = process.argv[2];
  assert(id, 'Pass the timestamped X5 session ID');
  const output = path.resolve(process.argv[3] || 'validation/trajectory-viewer');
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({executablePath:process.env.CHROMIUM_BIN,
    headless:true, args:['--disable-gpu','--disable-dev-shm-usage','--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1560,height:1120}}), errors=[];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base+'/?id='+encodeURIComponent(id));
    await page.waitForFunction(() => document.querySelector('#connection').textContent === '已同步');
    await page.locator('#live').uncheck();
    const snapshot = await page.evaluate(() => transcript);
    const calls = snapshot.entries.filter(entry => entry.kind === 'tool');
    assert(calls.length > 0, 'Native agent has not called a tool yet');
    assert.equal(await page.locator('#conversation-list>[data-kind=tool]').count(), snapshot.tool_count);
    assert.equal(await page.locator('.conversation-head>span').count(), 4);
    const discrepancies = await page.evaluate(() => transcript.entries.filter(entry => entry.kind === 'tool').flatMap(entry => {
      const row=document.getElementById(entry.id), issues=[];
      function imageURLs(block) {
        if (!block || typeof block !== 'object') return [];
        if (block.type === 'image') return [block.source?.image_url || block.source?.url].filter(Boolean);
        if (block.type !== 'tool_result') return [];
        return (Array.isArray(block.content) ? block.content : [block.content]).flatMap(imageURLs);
      }
      if (entry.call.name === 'Bash' && row.querySelector('.exact-command').textContent !== entry.call.input.command)
        issues.push(entry.id+': changed command');
      if (!entry.shared_before) for (const item of entry.before) {
        const text=item.block.thinking ?? item.block.text;
        if (text && !row.querySelector('.thinking-cell').textContent.includes(text)) issues.push(entry.id+': lost thinking');
      }
      const expectedImages=entry.results.flatMap(result => imageURLs(result.block));
      const shownImages=[...row.querySelectorAll('.images-cell img')].map(image => image.getAttribute('src'));
      if (JSON.stringify(shownImages) !== JSON.stringify(expectedImages)) issues.push(entry.id+': changed image count, source or order');
      if (row.querySelector('.result-cell>figure.context-image')) issues.push(entry.id+': returned image left in text column');
      if (row.querySelectorAll(':scope>section').length !== 4) issues.push(entry.id+': expected four cells');
      return issues;
    }));
    assert.deepEqual(discrepancies, []);
    const heights = await page.locator('#conversation-list>[data-kind=tool]').evaluateAll(rows => rows.map(row => [...row.children].map(cell => cell.getBoundingClientRect().height)));
    assert(heights.every(cells => cells.length === 4 && cells.every(height => height === 360)));
    await page.screenshot({path:path.join(output,'all-tools.png')});
    let images=0;
    for (const call of calls) for (const result of call.results) for (const block of result.block.content || []) {
      if (block.type !== 'image') continue;
      const response=await page.request.get(base+block.source.image_url);
      assert.equal(crypto.createHash('sha256').update(await response.body()).digest('hex'),block.source.sha256);
      images++;
    }
    assert(images > 0, 'No actual images in the native transcript');
    const illustrated=calls.find(call => call.results.some(result => Array.isArray(result.block.content) && result.block.content.some(block => block.type === 'image')));
    await page.evaluate(call => selectCall(call), illustrated.id);
    const row=page.locator('#'+illustrated.id);
    const boxes=await Promise.all(['.thinking-cell','.call-cell','.result-cell','.images-cell'].map(selector => row.locator(selector).boundingBox()));
    assert(boxes.every((box,index) => box.y === boxes[0].y && (index === 0 || boxes[index-1].x < box.x)));
    const thumbnail=row.locator('.images-cell img').first();
    await thumbnail.scrollIntoViewIfNeeded();
    await thumbnail.evaluate(image => image.decode());
    const originalWidth=await thumbnail.evaluate(image => image.naturalWidth);
    assert(originalWidth > 0);
    assert((await thumbnail.boundingBox()).width <= 160);
    await thumbnail.click();
    await page.waitForFunction(width => document.querySelector('#full-image').naturalWidth === width,originalWidth);
    assert(await page.locator('#image-dialog').isVisible());
    await page.locator('#close-image').click();
    await page.screenshot({path:path.join(output,'four-columns.png')});
    await page.locator('#call-filter').selectOption('action');
    assert.equal(await page.locator('#conversation-list>[data-kind=tool]').count(),snapshot.action_count);
    await page.screenshot({path:path.join(output,'action-only.png')});
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({ok:true,session:id,
      native_tools:snapshot.tool_count,robot_calls:snapshot.action_count,exact_images:images,
      columns:4,default_row_height:360,returned_image_sources_and_order:true,
      image_modal:true,action_filter:true},null,2)+'\n');
    console.log('Browser checks passed; '+output);
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exit(1);});
