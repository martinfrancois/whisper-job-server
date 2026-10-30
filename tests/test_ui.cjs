const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'app.py'), 'utf8');
const script = [...source.matchAll(/<script>([\s\S]*?)<\/script>/g)]
  .map(match => match[1]).find(script => script.includes('function renderJobs('));
assert.ok(script, 'The application UI script must be tested');

function loadUI(protocol = 'http:') {
  const cards = [];
  const sockets = [];
  const elements = {
    'job-list': { innerHTML: '', appendChild: card => cards.push(card) },
    totalTime: { textContent: '' },
  };
  const context = vm.createContext({
    document: {
      getElementById: id => elements[id],
      createElement: () => ({}),
      addEventListener: () => {},
    },
    location: { protocol, host: 'transcription.example:8443' },
    WebSocket: function (url) { sockets.push(url); },
  });
  vm.runInContext(script, context);
  return { context, cards, sockets };
}

for (const field of ['original_filename', 'mode', 'language', 'status', 'error']) {
  test(`renders ${field} as escaped text`, () => {
    const { context, cards } = loadUI();
    const payload = '<img src=x onerror="window.xss=1"> & \'quoted\'';
    const job = {
      original_filename: 'tone.wav', input_file: 'tone.wav', mode: 'one',
      language: 'en', status: 'failed', [field]: payload,
    };
    context.renderJobs({ synthetic: job });
    assert.equal(cards.length, 1);
    assert.ok(!cards[0].innerHTML.includes('<img'));
    assert.ok(cards[0].innerHTML.includes('&lt;img src=x onerror=&quot;window.xss=1&quot;&gt; &amp; &#39;quoted&#39;'));
  });
}

test('retains job labels and download controls', () => {
  const { context, cards } = loadUI();
  context.renderJobs({ synthetic: {
    original_filename: 'tone.wav', mode: 'one', language: 'en', status: 'done',
  } });
  assert.ok(cards[0].innerHTML.includes('tone.wav'));
  assert.ok(cards[0].innerHTML.includes('Language: English'));
  assert.ok(cards[0].innerHTML.includes('href="/download/synthetic/without"'));
  assert.ok(cards[0].innerHTML.includes('href="/download/synthetic/with"'));
  assert.ok(cards[0].innerHTML.includes("removeJob('synthetic')"));
});

for (const [protocol, expected] of [['http:', 'ws:'], ['https:', 'wss:']]) {
  test(`uses ${expected} for a page served over ${protocol}`, () => {
    const { context, sockets } = loadUI(protocol);
    context.connectWebSocket();
    assert.deepEqual(sockets, [`${expected}//transcription.example:8443/ws`]);
  });
}
