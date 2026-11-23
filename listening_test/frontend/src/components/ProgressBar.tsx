type Props = {
  total: number;
  currentRound: number;
  completedThisRound: number;
};

export default function ProgressBar({ total, currentRound, completedThisRound }: Props) {
  const pct = total > 0 ? Math.round((completedThisRound / total) * 100) : 0;

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm text-slate-500 mb-1.5">
        <span>
          Round {currentRound} — {completedThisRound}/{total} sounds described
        </span>
        <span>{pct}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
        <div
          className="h-full rounded-full bg-indigo-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
