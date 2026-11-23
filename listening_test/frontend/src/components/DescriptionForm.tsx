import { useState } from "react";
import GuidedBuilder from "./GuidedBuilder";
import { composeSentence, hasAnyChoice, type GuidedChoices } from "../lib/clapTemplate";

type Props = {
  onSubmit: (
    freeDescription: string,
    guidedDescription: string,
    guidedChoices: GuidedChoices | null
  ) => Promise<void>;
  onSkip: () => Promise<void>;
};

const PREFIX = "The sound of";
const CONTINUATION_PLACEHOLDER =
  "standing alone in an empty cathedral, cold and metallic, slowly pulsing...";

type Phase = "free" | "guided";

export default function DescriptionForm({ onSubmit, onSkip }: Props) {
  const [phase, setPhase] = useState<Phase>("free");
  const [freeContinuation, setFreeContinuation] = useState("");
  const [guidedChoices, setGuidedChoices] = useState<GuidedChoices>({});
  const [submitting, setSubmitting] = useState(false);
  const [skipping, setSkipping] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const guidedSentence = hasAnyChoice(guidedChoices) ? composeSentence(guidedChoices) : "";

  function handleContinue(e: React.FormEvent) {
    e.preventDefault();
    if (!freeContinuation.trim()) {
      setError('Please add a description after "The sound of...".');
      return;
    }
    setError(null);
    setPhase("guided");
  }

  async function handleFinalSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (submitting || skipping) return;
    if (!hasAnyChoice(guidedChoices)) {
      setError("Pick at least one word below to build your guided description.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await onSubmit(`${PREFIX} ${freeContinuation.trim()}`, guidedSentence, guidedChoices);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong while submitting.");
      setSubmitting(false);
    }
  }

  async function handleSkip() {
    if (submitting || skipping) return;
    setSkipping(true);
    setError(null);
    try {
      await onSkip();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong while skipping.");
      setSkipping(false);
    }
  }

  if (phase === "free") {
    return (
      <form onSubmit={handleContinue} className="flex flex-col gap-4">
        <div className="rounded-xl border border-indigo-200 bg-indigo-50/60 p-4">
          <p className="text-sm font-semibold text-indigo-700 mb-2">
            💡 Step 1 of 2 — first, describe it your way
          </p>
          <p className="text-sm text-slate-700 mb-2">
            Listen to the sound and write what <strong>you</strong> perceive —
            before seeing any suggested words, so they don't influence you.
            There's no wrong answer: it's completely fine if your description
            is abstract or unusual (e.g. "like a robot complaining", "a
            buzzing cloud"). Just trust what you hear. Write in{" "}
            <strong>English</strong>.
          </p>
          <p className="text-sm text-slate-700 mb-2">
            Forget technical words — think of a place, an emotion, a texture,
            or a movement:
          </p>
          <ul className="text-sm text-slate-600 list-disc list-inside space-y-1 mb-2">
            <li>Place or scene — "underwater", "an abandoned warehouse", "deep space"</li>
            <li>Emotion or atmosphere — "calming", "tense", "dreamy", "aggressive"</li>
            <li>Texture or sensation — "rough like sandpaper", "cold and metallic"</li>
            <li>Movement — "pulsing", "fading away", "floating"</li>
          </ul>
          <p className="text-sm text-slate-700">
            The sentence always starts with "The sound of..." — just continue
            from there. Avoid numbers, technical jargon (Hz, "low-pass
            filter"...) and smell/taste comparisons.
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1.5">
            How would you describe this sound?
          </label>
          <div className="flex items-start rounded-xl border border-slate-300 bg-white focus-within:ring-2 focus-within:ring-indigo-400">
            <span className="pl-3.5 py-3 text-slate-500 select-none whitespace-nowrap">
              {PREFIX}
            </span>
            <textarea
              value={freeContinuation}
              onChange={(e) => setFreeContinuation(e.target.value)}
              placeholder={CONTINUATION_PLACEHOLDER}
              rows={4}
              autoFocus
              className="flex-1 min-w-0 resize-none bg-transparent py-3 pr-3.5 pl-1.5 text-slate-800 placeholder:text-slate-400 focus:outline-none"
            />
          </div>
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={!freeContinuation.trim() || skipping}
            className="rounded-xl bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Continue to guided description →
          </button>
          <button
            type="button"
            onClick={handleSkip}
            disabled={skipping}
            className="rounded-xl border border-slate-300 px-5 py-2.5 text-sm font-medium text-slate-500 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {skipping ? "Skipping..." : "⏭ Skip this sound"}
          </button>
        </div>
      </form>
    );
  }

  return (
    <form onSubmit={handleFinalSubmit} className="flex flex-col gap-4">
      <div className="rounded-xl border border-indigo-200 bg-indigo-50/60 p-4">
        <p className="text-sm font-semibold text-indigo-700 mb-2">
          🪄 Step 2 of 2 — now, a guided description
        </p>
        <p className="text-sm text-slate-700">
          Here's a second way to describe the same sound: pick from common
          words below and I'll build the sentence for you — no typing this
          time. You don't have to use every word category, even just one or
          two are enough.
        </p>
      </div>

      <GuidedBuilder onChange={setGuidedChoices} />

      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1.5">
          Your guided description
        </label>
        <div className="flex items-start rounded-xl border border-slate-300 bg-slate-50 min-h-[3.5rem]">
          <span className="pl-3.5 py-3 text-slate-500 select-none whitespace-nowrap">
            {PREFIX}
          </span>
          <p className="flex-1 min-w-0 py-3 pr-3.5 pl-1.5 text-slate-800">
            {guidedSentence ? (
              guidedSentence.replace(/^the sound of\s*/i, "")
            ) : (
              <span className="text-slate-400">{CONTINUATION_PLACEHOLDER}</span>
            )}
          </p>
        </div>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={!hasAnyChoice(guidedChoices) || submitting || skipping}
          className="rounded-xl bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {submitting ? "Submitting..." : "Submit both descriptions"}
        </button>
        <button
          type="button"
          onClick={handleSkip}
          disabled={submitting || skipping}
          className="rounded-xl border border-slate-300 px-5 py-2.5 text-sm font-medium text-slate-500 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {skipping ? "Skipping..." : "⏭ Skip this sound"}
        </button>
      </div>
    </form>
  );
}
