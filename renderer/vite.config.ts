import {defineConfig} from 'vite';
import {readFileSync, statSync, realpathSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {outfits, hairstyles, wardrobeLabels} from './src/wardrobe';
const design = new URL('../characters/ari/design/', import.meta.url);
const catalog = () => {
  const value = JSON.parse(readFileSync(new URL('looks/catalog.json',design),'utf8'));
  for (const look of value.looks) {
    if (!/^looks\/[a-z0-9-]+\.png$/.test(look.image)) throw new Error('Invalid reference asset path');
    const file = new URL(look.image,design);
    if (!realpathSync(file).startsWith(realpathSync(design)+'/')) throw new Error('Reference asset escapes design directory');
    const stat=statSync(file); look.revision=`${stat.mtimeMs}-${stat.size}`;
  }
  return value;
};
const catalogScript = () => `window.dispatchEvent(new CustomEvent('ari:catalog',{detail:${JSON.stringify(catalog())}}));`;

export default defineConfig({
  base: './',
  assetsInclude: ['**/*.glb', '**/*.vrm'],
  build: {
    target: 'es2022',
    lib: {entry: 'src/main.ts', name: 'CompanionCharacter', formats: ['iife'], fileName: () => 'character.js'},
    rollupOptions: {output: {inlineDynamicImports: true}},
  },
  plugins: [{
    name: 'native-file-html',
    configureServer(server) {
      // Serve source artwork directly in Vite as well as in the native studio.
      server.middlewares.use((req,res,next) => {
        const path=(req.url??'').split('?')[0];
        if (!path.startsWith('/ari/')) return next();
        const relative=path.slice(5);
        try {
          res.setHeader('Cache-Control','no-store');
          if(relative==='looks/catalog.js') {res.setHeader('Content-Type','text/javascript');res.end(catalogScript());return;}
          if(!/^(?:looks\/|emotions\/)?[a-z0-9-]+\.png$/.test(relative)) {res.statusCode=404;res.end();return;}
          const file=new URL(relative,design);
          if(!realpathSync(file).startsWith(realpathSync(design)+'/')) {res.statusCode=404;res.end();return;}
          res.setHeader('Content-Type','image/png');res.end(readFileSync(file));
        } catch {res.statusCode=404;res.end('Reference asset unavailable');}
      });
      // The library polls revisions and swaps only changed images. Do not restart
      // the whole studio (and its animation) when an artist saves artwork.
      server.watcher.add(fileURLToPath(new URL('looks/',design)));
    },
    handleHotUpdate(context) {
      if(context.file.startsWith(fileURLToPath(new URL('looks/',design)))) return [];
    },
    generateBundle() {
      // Native Character-tab choices use the renderer's exact IDs and labels.
      this.emitFile({type:'asset',fileName:'wardrobe-catalog.json',source:JSON.stringify({
        outfits:outfits.map(id=>({id,label:id==='original'?'Original cardigan':wardrobeLabels[id]})),
        hairstyles:hairstyles.map(id=>({id,label:id==='original'?'Long layered hair':wardrobeLabels[id]})),
      })});
      // WKWebView's file origin blocks ES modules. Ship one classic script;
      // keep the module entry in the source HTML for Vite development.
      for(const path of ['anime-reference-v2.png','full-body-neutral-v1.png','turnaround-v1.png',...['neutral','happy','curious','concerned','surprised','annoyed','embarrassed','sleepy'].map(e=>`emotions/${e}.png`)]) this.emitFile({type:'asset',fileName:`ari/${path}`,source:readFileSync(new URL(path,design))});
      for(const look of catalog().looks) this.emitFile({type:'asset',fileName:`ari/${look.image}`,source:readFileSync(new URL(look.image,design))});
      this.emitFile({type:'asset',fileName:'ari/looks/catalog.js',source:catalogScript()});
      this.emitFile({type:'asset',fileName:'ari/looks/catalog.json',source:JSON.stringify(catalog())});
      const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
      const bundledHtml=html.replace('<script type="module" src="/src/main.ts"></script>', '<script defer src="./character.js"></script>');
      this.emitFile({type:'asset',fileName:'index.html',source:bundledHtml});
      this.emitFile({type:'asset',fileName:'studio.html',source:bundledHtml.replace('<body>', '<body data-studio="true">')});
    },
  }],
});
