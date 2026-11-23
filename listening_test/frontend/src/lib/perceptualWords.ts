export type WordHint = { word: string; hint?: string };

export const CATEGORY_WORDS: WordHint[] = [
  { word: "lead", hint: "the main melodic voice, usually bright and upfront" },
  { word: "bass", hint: "the low-end foundation of the sound" },
  { word: "fx", hint: "a sound effect or texture, not a musical note" },
  { word: "pad", hint: "a sustained, atmospheric background sound" },
  { word: "drum", hint: "a percussive hit, like a kick, snare or hi-hat" },
  { word: "keyboard", hint: "a piano or organ-like sustained tone" },
  { word: "arp", hint: "short for arpeggio: a fast repeating pattern of notes" },
  { word: "chord", hint: "multiple notes played together, a harmony" },
  { word: "pluck", hint: "a short attack, like a string being plucked" },
];

export const BRIGHTNESS_WORDS: WordHint[] = [
  { word: "bright", hint: "rich in high frequencies, vivid" },
  { word: "dark", hint: "lacking high frequencies, shadowy" },
  { word: "warm", hint: "rich in low-mid frequencies, comforting" },
  { word: "mellow", hint: "soft and smooth, not harsh" },
  { word: "sharp", hint: "cutting, with a hard edge" },
  { word: "muffled", hint: "as if heard through a wall or blanket" },
];

export const LOUDNESS_WORDS: WordHint[] = [
  { word: "soft", hint: "quiet, gentle in volume" },
  { word: "gentle", hint: "mild, not forceful" },
  { word: "loud", hint: "high in volume, prominent" },
  { word: "powerful", hint: "strong and commanding" },
  { word: "aggressive", hint: "forceful, intense, in-your-face" },
];

export const WEIGHT_WORDS: WordHint[] = [
  { word: "deep", hint: "has real weight and depth, substantial presence" },
  { word: "full", hint: "rich and dense, fills the space" },
  { word: "thin", hint: "lacking body, feels insubstantial" },
  { word: "light", hint: "delicate, airy, little weight" },
  { word: "heavy", hint: "dense and weighty, hard-hitting" },
];

export const TEXTURE_WORDS: WordHint[] = [
  { word: "smooth", hint: "even, without rough edges" },
  { word: "rough", hint: "uneven, gritty" },
  { word: "clean", hint: "pure, without distortion or noise" },
  { word: "harsh", hint: "unpleasantly rough or grating" },
  { word: "buzzy", hint: "with a vibrating, fuzzy edge" },
  { word: "metallic", hint: "ringing, like struck metal" },
  { word: "hollow", hint: "empty-sounding, like inside a tube" },
  { word: "lush", hint: "rich, dense and pleasant" },
];

export const TIME_SHAPE_WORDS: WordHint[] = [
  { word: "short", hint: "ends quickly" },
  { word: "long", hint: "lasts a long time" },
  { word: "slow", hint: "develops gradually, not abrupt" },
  { word: "fading", hint: "dying away gradually" },
  { word: "pulsing", hint: "rhythmically repeating, like a heartbeat" },
  { word: "evolving", hint: "slowly changing or transforming over time" },
];
