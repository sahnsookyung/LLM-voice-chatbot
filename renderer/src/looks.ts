export type Look = {
  id: string; name: string; category: 'outfit' | 'hair'; description: string;
  image: string; revision: string; sources: string[];
};
export type DesignSource = {id: string; title: string; url: string; idea: string};
export type LookCatalog = {version: 1; looks: Look[]; sources: DesignSource[]};

const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === 'string' && v.length > 0 && v.length <= 1000;
export function parseCatalog(value: unknown): LookCatalog {
  if (!record(value) || value.version !== 1 || !Array.isArray(value.looks) || value.looks.length > 60 || !Array.isArray(value.sources)) throw new Error('Unsupported reference library.');
  const ids = new Set<string>();
  const sources = value.sources.map((source): DesignSource => {
    if (!record(source) || !text(source.id) || !text(source.title) || !text(source.idea) || !text(source.url) || !/^https:\/\//.test(source.url)) throw new Error('Invalid design source.');
    return {id:source.id, title:source.title, idea:source.idea, url:source.url};
  });
  const looks = value.looks.map((look): Look => {
    if (!record(look) || !text(look.id) || !/^[a-z][a-z0-9-]{0,63}$/.test(look.id) || look.id === 'original' || ids.has(look.id) || !text(look.name) || !text(look.description) || !['outfit','hair'].includes(String(look.category)) || !text(look.image) || !/^looks\/[a-z0-9-]+\.png$/.test(look.image) || !Array.isArray(look.sources) || !look.sources.every(id => typeof id === 'string' && sources.some(source => source.id === id))) throw new Error('Invalid or duplicate reference look.');
    ids.add(look.id);
    return {id:look.id, name:look.name, description:look.description, category:look.category as Look['category'], image:look.image, revision:text(look.revision) ? look.revision : '1', sources:look.sources as string[]};
  });
  return {version:1, looks, sources};
}

export const lookURL = (look: Look) => {
  const base = window.ariLookBase ?? new URL('./ari/', document.baseURI).href;
  const url = new URL(look.image, base);
  url.searchParams.set('v', look.revision);
  return url.href;
};
