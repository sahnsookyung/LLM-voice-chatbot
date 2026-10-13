import {asset} from './builtin';
import {lookURL, parseCatalog, type Look, type LookCatalog} from './looks';

export function mountLookLibrary(container: HTMLElement, selector: HTMLSelectElement, choose: (look: Look | null) => void) {
  container.innerHTML = `<div class="section-title"><div><span class="eyebrow">WARDROBE & HAIR</span><h2>New ways to wear Ari.</h2><p>Three outfits. Three hairstyles. The same familiar face.</p></div><div class="library-actions"><label class="check"><input id="watch-looks" type="checkbox" checked> Auto reload</label><button id="reload-looks">Reload artwork</button></div></div>
    <p class="hint">Select an illustration to compare with Ari. Use the separate Outfit and Hairstyle controls above to customize the animated 3D character. These reference images are complete illustrations.</p>
    <p id="look-status" role="status" aria-live="polite">Loading references…</p><div class="look-grid" aria-label="Clothing and hairstyle references"></div>
    <details class="design-sources"><summary>Design inspiration & sources</summary><ul></ul></details>`;
  const grid = container.querySelector<HTMLDivElement>('.look-grid')!;
  const status = container.querySelector<HTMLParagraphElement>('#look-status')!;
  const watch = container.querySelector<HTMLInputElement>('#watch-looks')!;
  let selected = 'original', catalog: LookCatalog | undefined, signature = '', selectedRevision = '';
  let pending: HTMLScriptElement | undefined, timeout: ReturnType<typeof setTimeout> | undefined;
  let force = false, disposed = false;
  try { selected = localStorage.getItem('ari-reference-look') ?? 'original'; } catch { /* File-origin storage can be unavailable. */ }
  const select = (id: string, reload = false) => {
    selected = id;
    selector.value=id;
    const look = catalog?.looks.find(value => value.id === id) ?? null;
    const next = look && reload ? {...look, revision:`${look.revision}-${Date.now()}`} : look;
    selectedRevision = look?.revision ?? '';
    try { localStorage.setItem('ari-reference-look', selected); } catch { /* Selection still works without storage. */ }
    for (const button of grid.querySelectorAll<HTMLButtonElement>('button')) button.setAttribute('aria-pressed', String(button.dataset.look === selected));
    choose(next);
    status.textContent = look ? `${look.name} · ${look.description} Static reference; expressions keep this look.` : 'Original Ari · includes the existing eight expression illustrations.';
  };
  const render = () => {
    if (!catalog) return;
    grid.replaceChildren();
    selector.replaceChildren(new Option('Original Ari', 'original'));
    for (const category of ['outfit','hair'] as const) {
      const group=document.createElement('optgroup'); group.label=category==='outfit'?'Outfits':'Hairstyles';
      for (const look of catalog.looks.filter(value => value.category===category)) group.append(new Option(look.name, look.id));
      selector.append(group);
    }
    selector.value=selected; selector.disabled=false;
    const entries = [{id:'original', name:'Original Ari', category:'original', description:'Ivory cardigan · long layered hair', image:asset('full-body-neutral-v1.png')}, ...catalog.looks.map(look => ({...look, image:lookURL(look)}))];
    for (const look of entries) {
      const button = document.createElement('button'); button.className='look-card'; button.dataset.look=look.id;
      button.setAttribute('aria-pressed', String(look.id === selected)); button.title=look.description;
      const img = new Image(); img.src=look.image; img.alt=look.description; img.loading='lazy';
      const label = document.createElement('strong'); label.textContent=look.name;
      const category = document.createElement('span'); category.textContent=look.category === 'hair' ? 'Hairstyle' : look.category === 'outfit' ? 'Outfit' : 'Original';
      button.append(img, category, label); button.addEventListener('click', () => select(look.id)); grid.append(button);
    }
    const list = container.querySelector('ul')!; list.replaceChildren();
    for (const source of catalog.sources) {
      const item = document.createElement('li'), link = document.createElement('a');
      link.href=source.url; link.textContent=source.title; link.target='_blank'; link.rel='noopener noreferrer';
      item.append(link, document.createTextNode(` — ${source.idea}`)); list.append(item);
    }
  };
  const finish = () => { clearTimeout(timeout); pending?.remove(); pending=undefined; };
  const receive = (event: Event) => {
    if (!pending || disposed) return;
    try {
      const next = parseCatalog((event as CustomEvent).detail);
      const nextSignature = JSON.stringify(next);
      const changed = nextSignature !== signature;
      catalog=next; signature=nextSignature;
      if (selected !== 'original' && !catalog.looks.some(look => look.id === selected)) selected='original';
      const revision = catalog.looks.find(look => look.id === selected)?.revision ?? '';
      if (changed) render();
      if (changed || force || revision !== selectedRevision) select(selected, force);
      else if (status.textContent?.startsWith('Reference library')) select(selected);
    } catch { status.textContent='Reference library could not reload. Check catalog.json, then try Reload artwork. The last good preview is retained.'; }
    finish();
  };
  const reload = (manual = false) => {
    if (pending || disposed) return;
    force=manual;
    const script = document.createElement('script'); pending=script;
    script.src=new URL(`looks/catalog.js?v=${Date.now()}`, window.ariLookBase ?? new URL('./ari/', document.baseURI).href).href;
    const failed = () => { status.textContent='Reference library unavailable. Try Reload artwork after the files are restored.'; finish(); };
    script.onerror=failed; timeout=setTimeout(failed, 8000); document.head.append(script);
  };
  window.addEventListener('ari:catalog', receive);
  selector.addEventListener('change', () => select(selector.value));
  container.querySelector('#reload-looks')!.addEventListener('click', () => reload(true));
  const interval = setInterval(() => {if (watch.checked && !document.hidden && window.characterStatus(false).visible !== false) reload();}, 3000);
  window.addEventListener('pagehide', () => {disposed=true; clearInterval(interval); finish(); window.removeEventListener('ari:catalog', receive);}, {once:true});
  reload();
}
