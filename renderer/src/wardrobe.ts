export const outfits = ['original', 'tailored', 'streetwear', 'linen'] as const;
export const hairstyles = ['original', 'bob', 'ponytail', 'braid'] as const;
export type Outfit = typeof outfits[number];
export type Hairstyle = typeof hairstyles[number];
export type WardrobeSelection = {outfit: Outfit; hair: Hairstyle};
export const defaultWardrobe: WardrobeSelection = {outfit:'original', hair:'original'};
export const wardrobeLabels: Record<Outfit | Hairstyle, string> = {
  original:'Original', tailored:'City tailoring', streetwear:'Utility weekend', linen:'Woodland linen',
  bob:'Soft tapered bob', ponytail:'High ponytail', braid:'Loose side braid',
};
export function parseWardrobe(value: unknown): WardrobeSelection {
  if (!value || typeof value !== 'object') throw new Error('Invalid wardrobe selection.');
  const v = value as Record<string, unknown>;
  if (!outfits.includes(v.outfit as Outfit) || !hairstyles.includes(v.hair as Hairstyle)) throw new Error('Unknown outfit or hairstyle.');
  return {outfit:v.outfit as Outfit, hair:v.hair as Hairstyle};
}
export const wardrobeKey = (value: WardrobeSelection) => `${value.outfit}/${value.hair}`;
export function restoreWardrobe(saved: string | null): WardrobeSelection {
  try { return parseWardrobe(JSON.parse(saved ?? 'null')); } catch { return {...defaultWardrobe}; }
}
export const wardrobeCombinations = () => outfits.flatMap(outfit => hairstyles.map(hair => ({outfit, hair})));
