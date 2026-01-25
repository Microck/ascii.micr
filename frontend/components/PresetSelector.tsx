'use client';

export type Preset = 'epson' | 'discord';

interface PresetSelectorProps {
  preset: Preset;
  onPresetChange: (preset: Preset) => void;
}

export default function PresetSelector({ preset, onPresetChange }: PresetSelectorProps) {
  return (
    <form className="flex gap-4">
      <div>
        <input
          type="radio"
          id="preset-epson"
          name="preset"
          value="epson"
          checked={preset === 'epson'}
          onChange={() => onPresetChange('epson')}
          className="shake"
        />
        <label htmlFor="preset-epson" className="shake">
          Epson
        </label>
      </div>
      <div>
        <input
          type="radio"
          id="preset-discord"
          name="preset"
          value="discord"
          checked={preset === 'discord'}
          onChange={() => onPresetChange('discord')}
          className="shake"
        />
        <label htmlFor="preset-discord" className="shake">
          Discord
        </label>
      </div>
    </form>
  );
}
