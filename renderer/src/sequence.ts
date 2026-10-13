import {emotions, type Emotion} from './motion.js';
export type SequenceMode = 'shuffle' | 'pairs' | 'conversation' | 'interruptions' | 'custom' | 'movement';
export type Cue = {emotion: Emotion; duration: number; gesture?: string};
const shuffle = <T>(items: readonly T[], random: () => number) => {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.min(i, Math.floor(Math.max(0, random()) * (i + 1)));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
};

/** An Euler circuit visits each directed expression change once, without neutral resets. */
export function allExpressionPairs(random: () => number = Math.random): Emotion[] {
  const remaining = new Map(emotions.map(from => [from, shuffle(emotions.filter(to => to !== from), random)]));
  const stack: Emotion[] = ['neutral'], path: Emotion[] = [];
  while (stack.length) {
    const edges = remaining.get(stack[stack.length - 1])!;
    if (edges.length) stack.push(edges.pop()!);
    else path.push(stack.pop()!);
  }
  return path.reverse();
}

export function makeSequence(mode: SequenceMode, seconds = 4, custom = '', random: () => number = Math.random): Cue[] {
  if(mode==='movement')return [
    {emotion:'neutral',gesture:'hair-sway',duration:8},
    {emotion:'happy',gesture:'twirl',duration:7},
    {emotion:'happy',gesture:'raise-arms',duration:6},
    {emotion:'curious',gesture:'shrug',duration:5},
    {emotion:'curious',gesture:'point',duration:6},
    {emotion:'neutral',gesture:'cancel',duration:2},
  ];
  const duration = Number.isFinite(seconds) ? Math.max(2, Math.min(8, seconds)) : 4;
  let order: Emotion[];
  if (mode === 'pairs') order = allExpressionPairs(random);
  else if (mode === 'conversation') order = ['neutral','curious','happy','embarrassed','neutral','concerned','surprised','annoyed','concerned','happy','sleepy','neutral'];
  else if (mode === 'custom') {
    const values = custom.toLowerCase().split(/[\s,→]+/).filter(Boolean);
    if (values.length < 2 || values.length > 64 || values.some(value => !(emotions as readonly string[]).includes(value))) {
      throw new Error('Enter 2–64 emotion names separated by commas. Use the eight names below.');
    }
    order = values as Emotion[];
  } else order = shuffle(emotions, random);
  const gestures: Partial<Record<Emotion,string>> = {happy:'wave',curious:'explain',concerned:'nod',embarrassed:'shift',surprised:'step',sleepy:'shift'};
  return order.map((emotion, i) => ({emotion, duration: mode === 'interruptions' ? .4 : duration * (mode === 'pairs' || mode === 'custom' ? 1 : [.9,1.15,1,.85][i % 4]), gesture: gestures[emotion]}));
}

/** No wall-clock timers: pause/hidden time cannot skip cues or accumulate work. */
export class SequencePlayer {
  index = 0; elapsed = 0; playing = false; done = false;
  readonly visited = new Set<string>();
  constructor(readonly cues: Cue[]) { if (!cues.length) throw new Error('A sequence needs at least one cue.'); }
  get current() { return this.cues[this.index]; }
  next(): Cue | undefined {
    if (this.index + 1 === this.cues.length) { this.done = true; this.playing = false; return; }
    const previous = this.current.emotion;
    this.index++; this.elapsed = 0;
    if (previous !== this.current.emotion) this.visited.add(`${previous}→${this.current.emotion}`);
    return this.current;
  }
  step(seconds: number): Cue | undefined {
    if (!this.playing || !Number.isFinite(seconds) || seconds <= 0) return;
    this.elapsed += Math.min(seconds, .25);
    if (this.elapsed >= this.current.duration) return this.next();
  }
}
