import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, existsSync} from 'node:fs';
import {parseCatalog, lookURL} from '../.test-build/looks.js';

const source = () => JSON.parse(readFileSync(new URL('../../characters/ari/design/looks/catalog.json',import.meta.url),'utf8'));
test('the reference library contains three distinct outfits and three hairstyles with bundled artwork and sources', () => {
  const catalog = parseCatalog(source());
  assert.equal(catalog.looks.filter(l => l.category==='outfit').length,3);
  assert.equal(catalog.looks.filter(l => l.category==='hair').length,3);
  for(const look of catalog.looks) {
    assert.ok(existsSync(new URL(`../../characters/ari/design/${look.image}`,import.meta.url)));
    assert.ok(look.sources.length);
  }
});
test('catalog validation rejects traversal, remote assets, reserved/duplicate IDs, unknown source IDs and executable source URLs', () => {
  for(const image of ['../secret.png','looks/../../secret.png','https://example.com/track.png','looks/x.svg','looks/%2e%2e.png']) {
    const value=source();value.looks[0].image=image;assert.throws(()=>parseCatalog(value));
  }
  for(const mutate of [v=>v.looks.push(v.looks[0]), v=>v.looks[0].id='original', v=>v.looks[0].sources=['missing'], v=>v.sources[0].url='javascript:alert(1)', v=>v.version=2]) {
    const value=source();mutate(value);assert.throws(()=>parseCatalog(value));
  }
});
test('source revisions produce new URLs in both bundled and native live asset origins', () => {
  globalThis.document={baseURI:'file:///app/renderer/studio.html'};
  globalThis.window={};
  const look=parseCatalog(source()).looks[0];
  assert.equal(lookURL(look),'file:///app/renderer/ari/looks/tailored-v1.png?v=1');
  globalThis.window.ariLookBase='ari-design://assets/';
  assert.equal(lookURL({...look,revision:'new bytes'}),'ari-design://assets/looks/tailored-v1.png?v=new+bytes');
  delete globalThis.window;delete globalThis.document;
});
