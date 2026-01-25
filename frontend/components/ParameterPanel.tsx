'use client';

interface ParameterPanelProps {
  iterations: number;
  lr: number;
  diversityWeight: number;
  tempStart: number;
  tempEnd: number;
  onIterationsChange: (value: number) => void;
  onLrChange: (value: number) => void;
  onDiversityWeightChange: (value: number) => void;
  onTempStartChange: (value: number) => void;
  onTempEndChange: (value: number) => void;
}

export default function ParameterPanel({
  iterations,
  lr,
  diversityWeight,
  tempStart,
  tempEnd,
  onIterationsChange,
  onLrChange,
  onDiversityWeightChange,
  onTempStartChange,
  onTempEndChange,
}: ParameterPanelProps) {
  return (
    <div className="space-y-6">
      <h3 className="text-sm font-medium mb-4 text-[var(--text)]">Parameters</h3>

      <Slider
        label="Iterations"
        value={iterations}
        min={1000}
        max={20000}
        step={100}
        onChange={onIterationsChange}
      />

      <Slider
        label="Learning Rate"
        value={lr}
        min={0.001}
        max={0.1}
        step={0.001}
        onChange={onLrChange}
      />

      <Slider
        label="Diversity Weight"
        value={diversityWeight}
        min={0.0}
        max={0.1}
        step={0.001}
        onChange={onDiversityWeightChange}
      />

      <Slider
        label="Temp Start"
        value={tempStart}
        min={0.1}
        max={5.0}
        step={0.1}
        onChange={onTempStartChange}
      />

      <Slider
        label="Temp End"
        value={tempEnd}
        min={0.001}
        max={1.0}
        step={0.001}
        onChange={onTempEndChange}
      />
    </div>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function Slider({ label, value, min, max, step, onChange }: SliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs">
        <label className="text-[var(--text)]">{label}</label>
        <span className="text-[var(--text)] font-mono">{value.toFixed(step < 0.01 ? 3 : 2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 cursor-pointer appearance-none bg-transparent shake"
      />
    </div>
  );
}
