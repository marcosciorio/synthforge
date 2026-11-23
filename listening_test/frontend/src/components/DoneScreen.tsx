type Props = {
  onRetry: () => void;
};

export default function DoneScreen({ onRetry }: Props) {
  return (
    <div className="flex justify-center px-6 py-10 sm:py-14">
      <div className="text-center max-w-md">
        <div className="text-5xl mb-4">🙏</div>
        <h1 className="text-2xl font-semibold text-slate-800 mb-2">
          No sound available right now
        </h1>
        <p className="text-slate-500 mb-5">
          Every sound in this round is currently being described by someone
          else. This is temporary — a new round starts automatically as soon
          as the last one is submitted, please try again in a moment.
        </p>
        <button
          onClick={onRetry}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
