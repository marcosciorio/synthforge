type Props = {
  audioUrl: string;
  sampleId: string;
};

export default function AudioPlayer({ audioUrl, sampleId }: Props) {
  return (
    <div className="rounded-3xl bg-white border border-slate-200 shadow-sm p-6 sm:p-8 flex flex-col items-center gap-4 sm:gap-6">
      <div className="flex h-14 w-14 sm:h-20 sm:w-20 items-center justify-center rounded-full bg-indigo-50 text-2xl sm:text-3xl">
        🎧
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-slate-400 uppercase tracking-wide">
          Listen to the sound
        </p>
        <p className="mt-1 text-slate-500 text-sm">
          You can replay it as many times as you like before writing
        </p>
      </div>
      <audio key={sampleId} controls preload="auto" className="w-full" src={audioUrl}>
        Your browser does not support audio playback.
      </audio>
    </div>
  );
}
