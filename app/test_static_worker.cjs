// Run the actual service worker in Node with isolated CacheStorage/network fixtures.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const origin = 'http://127.0.0.1:8767';
const shell = ['/', '/index.html', '/styles.css', '/app.js', '/app-icon.svg', '/manifest.webmanifest'];
const current = 'soundtrace-shell-v5-public-only';
const old = 'soundtrace-shell-v4-text1';
const entries = new Map([[old, new Map([['/voiceprints/profiles.json', new Response('PRIVATE-SENTINEL')]])], ['other-application', new Map()]]);
const handlers = {};
let network = async () => new Response('fresh asset');
let claimed = false;
const key = request => typeof request === 'string' ? request : request.url;
const cache = name => {
  if (!entries.has(name)) entries.set(name, new Map());
  const values = entries.get(name);
  return {
    addAll: async urls => {for (const url of urls) values.set(origin + url, new Response('cached ' + url));},
    put: async (request, response) => values.set(key(request), response),
    match: async request => values.get(key(request))?.clone(),
  };
};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'sw.js'), 'utf8'), {
  self: {location: {origin}, clients: {claim: async () => {claimed = true;}},
    skipWaiting() {}, addEventListener: (name, handler) => {handlers[name] = handler;}},
  caches: {open: async name => cache(name), keys: async () => [...entries.keys()],
    delete: async name => entries.delete(name)},
  fetch: request => network(request), URL, Response,
});
async function lifecycle(name) {
  let pending;
  handlers[name]({waitUntil: promise => {pending = promise;}});
  await pending;
}
async function request(url, method = 'GET') {
  let pending;
  handlers.fetch({request: new Request(url, {method}), respondWith: value => {pending = value;}});
  return pending ? await pending : null;
}
async function main() {
  await lifecycle('install');
  assert.deepEqual([...entries.get(current).keys()].sort(), shell.map(p => origin + p).sort());
  await lifecycle('activate');
  assert.equal(entries.has(old), false, 'Old shell cache, including any private response, is removed');
  assert.equal(entries.has('other-application'), true, 'Unrelated caches are untouched');
  assert.equal(claimed, true);
  let denied = 0;
  for (const url of ['/api/health', '/voiceprints/profiles.json', '/jobs/a/input.wav',
    '/logs/server.log', '/server.py', '/future.js', '/app.js/../server.py', '/%76oiceprints/profiles.json']) {
    assert.equal(await request(origin + url), null, 'No interception or offline fallback: ' + url);
    denied++;
  }
  assert.equal(await request('https://example.invalid/app.js'), null);
  assert.equal(await request(origin + '/app.js', 'POST'), null);
  assert.equal(await (await request(origin + '/app.js')).text(), 'fresh asset');
  assert.equal(await (await cache(current).match(origin + '/app.js')).text(), 'fresh asset');
  network = async () => new Response('missing', {status: 404});
  assert.equal((await request(origin + '/app.js')).status, 404);
  assert.equal(await (await cache(current).match(origin + '/app.js')).text(), 'fresh asset');
  network = async () => {throw new Error('offline fixture');};
  assert.equal(await (await request(origin + '/app.js')).text(), 'fresh asset');
  assert.equal((await request(origin + '/styles.css?uncached=1')).type, 'error', 'Never substitute HTML for a missing asset');
  console.log(`Passed service worker cache checks, including ${denied} private/API/boundary URLs; no models or real private data.`);
}
main().catch(error => {console.error(error); process.exitCode = 1;});
