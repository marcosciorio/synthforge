import { useEffect, useState } from "react";
import {
  fetchNext,
  getSessionId,
  loadProfile,
  saveProfile,
  skipSample,
  submitDescription,
  type Profile,
} from "./lib/api";
import type { GuidedChoices } from "./lib/clapTemplate";
import AudioPlayer from "./components/AudioPlayer";
import DescriptionForm from "./components/DescriptionForm";
import ProgressBar from "./components/ProgressBar";
import DoneScreen from "./components/DoneScreen";
import ProfileGate from "./components/ProfileGate";
import FaqSection from "./components/FaqSection";

type CurrentSample = {
  sample_id: string;
  audio_url: string;
  total: number;
  current_round: number;
  completed_this_round: number;
};

type Status = "loading" | "ready" | "done" | "error";

const sessionId = getSessionId();

export default function App() {
  const [profile, setProfile] = useState<Profile | null>(() => loadProfile());
  const [status, setStatus] = useState<Status>("loading");
  const [sample, setSample] = useState<CurrentSample | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [toast, setToast] = useState<string | null>(null);

  async function load() {
    setStatus("loading");
    try {
      const res = await fetchNext(sessionId);
      if (res.done) {
        setStatus("done");
      } else {
        setSample(res);
        setStatus("ready");
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Network error.");
      setStatus("error");
    }
  }

  useEffect(() => {
    if (!profile) return;
    load();
  }, [profile]);

  function handleProfileComplete(p: Profile) {
    saveProfile(p);
    setProfile(p);
  }

  async function handleSubmit(
    freeDescription: string,
    guidedDescription: string,
    guidedChoices: GuidedChoices | null
  ) {
    if (!sample || !profile) return;
    await submitDescription({
      session_id: sessionId,
      sample_id: sample.sample_id,
      free_description: freeDescription,
      guided_description: guidedDescription,
      guided_choices: guidedChoices,
      describer_expertise: profile.expertise,
      describer_note: profile.note || null,
    });
    setToast("Thanks! ✅ Moving to the next sound...");
    await new Promise((r) => setTimeout(r, 900));
    setToast(null);
    await load();
  }

  async function handleSkip() {
    if (!sample) return;
    const res = await skipSample({ session_id: sessionId, sample_id: sample.sample_id });
    if (res.done) {
      setStatus("done");
    } else {
      setSample(res);
      setStatus("ready");
    }
  }

  let content: React.ReactNode;

  if (!profile) {
    content = <ProfileGate onComplete={handleProfileComplete} />;
  } else if (status === "loading" && !sample) {
    content = (
      <div className="flex justify-center px-6 py-20 text-slate-400">Loading...</div>
    );
  } else if (status === "error") {
    content = (
      <div className="flex justify-center px-6 py-20">
        <div className="text-center max-w-md">
          <p className="text-slate-700 font-medium mb-2">Something went wrong</p>
          <p className="text-slate-500 text-sm mb-4">{errorMsg}</p>
          <button
            onClick={load}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  } else if (status === "done" || !sample) {
    content = <DoneScreen onRetry={load} />;
  } else {
    content = (
      <div className="flex flex-col">
        <header className="border-b border-slate-200 bg-white">
          <div className="max-w-5xl mx-auto w-full px-6 md:px-10 py-4 sm:py-5">
            <h1 className="text-lg font-semibold text-slate-800">🎧 Listening Test</h1>
            <p className="text-sm text-slate-500 mb-3">
              Listen to the sound and describe it in your own words — nothing
              technical, I just want to know what it suggests to your ear.
            </p>
            <ProgressBar
              total={sample.total}
              currentRound={sample.current_round}
              completedThisRound={sample.completed_this_round}
            />
          </div>
        </header>

        <main className="max-w-5xl mx-auto w-full px-6 md:px-10 py-6 md:py-10">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8 items-start">
            <AudioPlayer audioUrl={sample.audio_url} sampleId={sample.sample_id} />
            <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 md:p-8">
              <DescriptionForm
                key={sample.sample_id}
                onSubmit={handleSubmit}
                onSkip={handleSkip}
              />
            </div>
          </div>
        </main>

        {toast && (
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 rounded-full bg-slate-900 text-white text-sm px-5 py-2.5 shadow-lg">
            {toast}
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      {content}
      <FaqSection />
    </>
  );
}
