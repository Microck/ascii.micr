'use client';

import { Download, Copy } from 'lucide-react';

interface ExportActionsProps {
  pngData?: string;
  textData?: string;
}

export default function ExportActions({ pngData, textData }: ExportActionsProps) {
  const handleDownloadPng = () => {
    if (!pngData) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${pngData}`;
    link.download = 'ascii-art.png';
    link.click();
  };

  const handleCopyText = () => {
    if (!textData) return;

    navigator.clipboard.writeText(textData);
  };

  return (
    <div className="flex gap-2">
      <button
        onClick={handleDownloadPng}
        disabled={!pngData}
        className="shake flex-1 flex items-center justify-center gap-2 px-4 py-2 border transition-colors text-[var(--text)] disabled:text-[var(--middle)] disabled:cursor-not-allowed hover:bg-[var(--bg25)] active:bg-[var(--bg50)]"
        style={{
          borderColor: 'var(--middle)',
          background: 'transparent',
        }}
      >
        <Download className="w-4 h-4" />
        <span>Download PNG</span>
      </button>

      <button
        onClick={handleCopyText}
        disabled={!textData}
        className="shake flex-1 flex items-center justify-center gap-2 px-4 py-2 border transition-colors text-[var(--text)] disabled:text-[var(--middle)] disabled:cursor-not-allowed hover:bg-[var(--bg25)] active:bg-[var(--bg50)]"
        style={{
          borderColor: 'var(--middle)',
          background: 'transparent',
        }}
      >
        <Copy className="w-4 h-4" />
        <span>Copy Text</span>
      </button>
    </div>
  );
}
