import type {Outfit} from './wardrobe.js';

// Linear RGB garment colors. Accessories use the same palette rather than
// sampling screen pixels, where skin, hair and lighting would dominate.
export type FabricColor = [number, number, number];
type OutfitPalette = {outer:FabricColor;bottom:FabricColor;accent:FabricColor;dominant:'outer'|'bottom'};
export const outfitPalettes:Record<Outfit,OutfitPalette> = {
  original:{outer:[.68,.62,.50],bottom:[.035,.039,.043],accent:[.055,.06,.065],dominant:'outer'},
  tailored:{outer:[.43,.36,.30],bottom:[.065,.071,.075],accent:[.88,.84,.76],dominant:'outer'},
  streetwear:{outer:[.23,.29,.20],bottom:[.052,.066,.055],accent:[.84,.82,.74],dominant:'outer'},
  linen:{outer:[.68,.62,.50],bottom:[.27,.34,.23],accent:[.68,.62,.50],dominant:'bottom'},
  'cloud-sweater':{outer:[.54,.68,.76],bottom:[.13,.16,.20],accent:[.13,.16,.20],dominant:'outer'},
  'sunny-sweater':{outer:[.82,.72,.35],bottom:[.13,.16,.20],accent:[.13,.16,.20],dominant:'outer'},
  'school-cardigan':{outer:[.23,.25,.26],bottom:[.095,.105,.11],accent:[.055,.06,.065],dominant:'outer'},
  'school-cardigan-brown':{outer:[.32,.21,.14],bottom:[.14,.09,.065],accent:[.055,.06,.065],dominant:'outer'},
  'school-vest':{outer:[.77,.66,.47],bottom:[.085,.089,.09],accent:[.033,.039,.042],dominant:'outer'},
  'navy-blouse':{outer:[.86,.85,.78],bottom:[.035,.07,.09],accent:[.73,.72,.67],dominant:'bottom'},
  'pink-sailor':{outer:[.86,.85,.78],bottom:[.69,.33,.42],accent:[.88,.83,.78],dominant:'bottom'},
  'ruffle-dress':{outer:[.045,.039,.04],bottom:[.58,.44,.34],accent:[.035,.029,.033],dominant:'bottom'},
  'black-jacket':{outer:[.026,.031,.033],bottom:[.13,.16,.20],accent:[.85,.84,.78],dominant:'outer'},
  'butter-coat':{outer:[.82,.66,.29],bottom:[.82,.66,.29],accent:[.033,.039,.042],dominant:'outer'},
  'forest-pinafore':{outer:[.86,.85,.78],bottom:[.18,.25,.095],accent:[.73,.72,.67],dominant:'bottom'},
};

export function beretPalette(outfit:Outfit) {
  const palette=outfitPalettes[outfit],main=palette[palette.dominant];
  const shade=(color:FabricColor,factor:number)=>color.map(value=>value*factor) as FabricColor;
  return {crown:[...main] as FabricColor,band:shade(main,.42),seam:shade(palette.accent,.68),stem:shade(main,.88),ribbon:[...palette.accent] as FabricColor};
}
