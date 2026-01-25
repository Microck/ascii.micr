'use client';

interface AsciiDisplayProps {
  originalImage?: string;
  asciiPng?: string;
  asciiText?: string;
  status?: 'idle' | 'generating' | 'complete' | 'error';
}

export default function AsciiDisplay({ originalImage, asciiPng, asciiText, status }: AsciiDisplayProps) {
  if (!originalImage) {
    return (
      <div className="w-full h-64 flex items-center justify-center border border-dashed border-[var(--middle)]">
        <p className="text-sm text-[var(--middle)]">Upload an image to preview</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="space-y-2">
        <h3 className="text-xs font-medium text-[var(--middle)] uppercase tracking-wider">Original</h3>
        <div className="relative aspect-square border border-[var(--middle)]">
          <img
            src={originalImage}
            alt="Original"
            className="w-full h-full object-contain"
          />
        </div>
      </div>

      <div className="space-y-2">
        <h3 className="text-xs font-medium text-[var(--middle)] uppercase tracking-wider">ASCII Preview</h3>
        {asciiPng ? (
          <div className="relative aspect-square border border-[var(--middle)] bg-[var(--bg)]">
            <img
              src={`data:image/png;base64,${asciiPng}`}
              alt="ASCII"
              className="w-full h-full object-contain"
            />
          </div>
        ) : (
          <div className="aspect-square flex items-center justify-center border border-[var(--middle)]">
            <p className="text-sm text-[var(--middle)]">
              {status === 'generating' ? 'Generating...' : 'Waiting...'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
