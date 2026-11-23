export type GuidedChoices = {
  category?: string;
  brightness?: string;
  loudness?: string;
  weight?: string;
  texture?: string;
  timeShape?: string;
};

function aOrAn(phrase: string): string {
  return phrase && /^[aeiou]/i.test(phrase) ? "an" : "a";
}

function joinAnd(items: (string | undefined)[]): string {
  const list = items.filter((i): i is string => Boolean(i));
  if (list.length === 0) return "";
  if (list.length === 1) return list[0];
  return list.slice(0, -1).join(", ") + " and " + list[list.length - 1];
}

export function composeSentence(choices: GuidedChoices): string {
  const category = choices.category?.trim() || "synthesizer";

  const categoryWords = new Set(category.toLowerCase().split(/\s+/));
  const descriptors = [
    choices.brightness,
    choices.texture,
    choices.loudness,
    choices.weight,
  ].filter((w): w is string => {
    if (!w) return false;
    return !categoryWords.has(w.toLowerCase());
  });
  const subject = joinAnd(descriptors);

  let sentence = subject
    ? `The sound of ${aOrAn(subject)} ${subject} ${category}`
    : `The sound of ${aOrAn(category)} ${category}`;

  if (choices.timeShape) sentence += `, ${choices.timeShape}`;

  return sentence;
}

export function hasAnyChoice(choices: GuidedChoices): boolean {
  return Object.values(choices).some((v) => v && v.trim().length > 0);
}
