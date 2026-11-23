import { useState } from "react";
import type { Profile } from "../lib/api";

type Props = {
  onComplete: (profile: Profile) => void;
};

const EXPERTISE_OPTIONS: { value: 1 | 2 | 3; title: string; desc: string }[] = [
  {
    value: 1,
    title: "Expert / sound designer",
    desc: "I have professional or in-depth technical experience designing synthesizer sounds.",
  },
  {
    value: 2,
    title: "Knowledgeable listener",
    desc: "I listen to a lot of music and I'm reasonably familiar with this field.",
  },
  {
    value: 3,
    title: "Outside listener",
    desc: "I have no background in this field — I'm just describing what I perceive.",
  },
];

export default function ProfileGate({ onComplete }: Props) {
  const [expertise, setExpertise] = useState<1 | 2 | 3 | null>(null);
  const [note, setNote] = useState("");

  function handleContinue(e: React.FormEvent) {
    e.preventDefault();
    if (!expertise) return;
    onComplete({ expertise, note: note.trim() });
  }

  return (
    <div className="flex justify-center px-6 py-6 sm:py-10">
      <form
        onSubmit={handleContinue}
        className="max-w-lg w-full bg-white rounded-3xl border border-slate-200 shadow-sm p-6 sm:p-8 flex flex-col gap-5"
      >
        <div>
          <h1 className="text-xl font-semibold text-slate-800">🎧 Before you start</h1>
          <p className="text-sm text-slate-500 mt-1">
            Just one quick thing — this helps me understand how to weigh the
            descriptions I collect.
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            How familiar are you with sound design / synthesizers?
          </label>
          <div className="flex flex-col gap-2">
            {EXPERTISE_OPTIONS.map((opt) => (
              <label
                key={opt.value}
                className={`flex items-start gap-3 rounded-xl border p-3 cursor-pointer transition-colors ${
                  expertise === opt.value
                    ? "border-indigo-400 bg-indigo-50"
                    : "border-slate-200 hover:bg-slate-50"
                }`}
              >
                <input
                  type="radio"
                  name="expertise"
                  className="mt-1 accent-indigo-600"
                  checked={expertise === opt.value}
                  onChange={() => setExpertise(opt.value)}
                />
                <span>
                  <span className="block text-sm font-medium text-slate-800">
                    {opt.title}
                  </span>
                  <span className="block text-sm text-slate-500">{opt.desc}</span>
                </span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1.5">
            Say hi! 👋 (optional)
          </label>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Tell me a little about yourself, or just leave a friendly hello :)"
            rows={3}
            className="w-full rounded-xl border border-slate-300 px-3.5 py-3 text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-400 resize-none"
          />
        </div>

        <button
          type="submit"
          disabled={!expertise}
          className="self-start rounded-xl bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Continue
        </button>
      </form>
    </div>
  );
}
