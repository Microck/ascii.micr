'use client';

import { useState, useRef } from 'react';
import { Upload, Image as ImageIcon } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
}

export default function ImageUploader({ onImageSelect }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      onImageSelect(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      onImageSelect(files[0]);
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          onImageSelect(file);
        }
        break;
      }
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onPaste={handlePaste}
      tabIndex={0}
      className={`
        shake relative w-full h-64 border border-dashed
        flex flex-col items-center justify-center gap-4
        transition-colors cursor-pointer outline-none
      `}
      style={{
        borderColor: 'var(--middle)',
        backgroundColor: isDragging ? 'var(--bg25)' : 'transparent',
      }}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />
      <div className="w-16 h-16 flex items-center justify-center">
        {isDragging ? <Upload className="w-8 h-8" style={{ color: 'var(--text)' }} /> : <ImageIcon className="w-8 h-8" style={{ color: 'var(--middle)' }} />}
      </div>
      <div className="text-center">
        <p className="text-sm mb-1" style={{ color: 'var(--text)' }}>
          {isDragging ? 'Drop image here' : 'Drop image or click to upload'}
        </p>
        <p className="text-xs" style={{ color: 'var(--middle)' }}>
          or paste (Ctrl+V)
        </p>
      </div>
    </div>
  );
}
