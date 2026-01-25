'use client';

interface ProgressBarProps {
  progress: number;
  currentIteration: number;
  totalIterations: number;
}

export default function ProgressBar({ progress, currentIteration, totalIterations }: ProgressBarProps) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs">
        <span className="text-[var(--text)]">Training Progress</span>
        <span className="font-mono text-[var(--middle)]">
          {currentIteration} / {totalIterations}
        </span>
      </div>
      <div className="w-full h-2 overflow-hidden bg-[var(--middle)]">
        <div
          className="h-full bg-[var(--text)] transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="text-right text-xs text-[var(--middle)]">
        {progress.toFixed(1)}%
      </div>
    </div>
  );
}
