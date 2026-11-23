import { useState } from "react";
import type { GuidedChoices } from "../lib/clapTemplate";
import {
  CATEGORY_WORDS,
  BRIGHTNESS_WORDS,
  LOUDNESS_WORDS,
  WEIGHT_WORDS,
  TEXTURE_WORDS,
  TIME_SHAPE_WORDS,
  type WordHint,
} from "../lib/perceptualWords";

type Props = {
  onChange: (choices: GuidedChoices) => void;
};

const FIELDS: { key: keyof GuidedChoices; label: string; words: WordHint[] }[] = [
  { key: "category", label: "Sound type", words: CATEGORY_WORDS },
  { key: "brightness", label: "Bright or dark?", words: BRIGHTNESS_WORDS },
  { key: "loudness", label: "Loud or soft?", words: LOUDNESS_WORDS },
  { key: "weight", label: "Weight or body?", words: WEIGHT_WORDS },
  { key: "texture", label: "Texture", words: TEXTURE_WORDS },
  { key: "timeShape", label: "How it moves over time", words: TIME_SHAPE_WORDS },
];

export default function GuidedBuilder({ onChange }: Props) {
  const [choices, setChoices] = useState<GuidedChoices>({});

  function setField(key: keyof GuidedChoices, value: string) {
    const next = { ...choices, [key]: value || undefined };
    setChoices(next);
    onChange(next);
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
      <p className="text-sm text-slate-500 mb-4">
        Pick a few words below — you don't have to fill every field, even
        just one or two are enough to build your sentence.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {FIELDS.map(({ key, label, words }) => (
          <label key={key} className="text-sm text-slate-600">
            {label}
            <select
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2.5 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-400"
              value={choices[key] ?? ""}
              onChange={(e) => setField(key, e.target.value)}
            >
              <option value="">—</option>
              {words.map(({ word, hint }) => (
                <option key={word} value={word}>
                  {hint ? `${word} — ${hint}` : word}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
    </div>
  );
}
