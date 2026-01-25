'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import ImageUploader from '@/components/ImageUploader';
import PresetSelector, { type Preset } from '@/components/PresetSelector';
import ParameterPanel from '@/components/ParameterPanel';
import AsciiDisplay from '@/components/AsciiDisplay';
import ProgressBar from '@/components/ProgressBar';
import ExportActions from '@/components/ExportActions';
import { useGeneration } from '@/hooks/useGeneration';
import { useToast } from '@/contexts/ToastContext';
import type { GenerationRequest, GenerationParams } from '@/lib/api';

export default function Home() {
  const { showToast } = useToast();
  const { state, generate, reset } = useGeneration();
  const [darkMode, setDarkMode] = useState(true);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [preset, setPreset] = useState<Preset>('epson');
  const [iterations, setIterations] = useState(10000);
  const [lr, setLr] = useState(0.01);
  const [diversityWeight, setDiversityWeight] = useState(0.01);
  const [tempStart, setTempStart] = useState(1.0);
  const [tempEnd, setTempEnd] = useState(0.01);

  const [activeSection, setActiveSection] = useState(0);
  const sectionsRef = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveSection(prev => Math.min(prev + 1, sectionsRef.current.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveSection(prev => Math.max(prev - 1, 0));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    if (sectionsRef.current[activeSection]) {
      sectionsRef.current[activeSection]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeSection]);

  const registerSection = (el: HTMLElement | null, index: number) => {
    sectionsRef.current[index] = el;
  };

  const handleImageSelect = useCallback((file: File) => {
    setSelectedImage(file);
    if (reset) reset();
  }, [reset]);

  const handleGenerate = useCallback(async () => {
    if (!selectedImage) return;

    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = reader.result as string;

      const params: GenerationParams = {
        iterations,
        lr,
        diversity_weight: diversityWeight,
        temp_start: tempStart,
        temp_end: tempEnd,
        optimize_alignment: false,
        dark_mode: preset === 'epson',
        encoding: preset === 'epson' ? 'cp437' : 'ascii',
        row_gap: preset === 'epson' ? 6 : 0,
      };

      const request: GenerationRequest = {
        image: base64,
        params,
      };

      try {
        await generate(request);
        showToast('ASCII art generated successfully', 'success');
      } catch (error) {
        showToast(error instanceof Error ? error.message : 'Failed to generate', 'error');
      }
    };
    reader.readAsDataURL(selectedImage);
  }, [selectedImage, preset, iterations, lr, diversityWeight, tempStart, tempEnd, generate, showToast]);

  return (
    <div className="flex gap-16 min-h-[calc(100vh-4rem)]">
      <div className="flex-1 max-w-[54ch] flex flex-col gap-8 pb-16">
        <div>
          <h1 className="text-xl mb-2 font-normal">gradscii-art</h1>
          <p className="text-[var(--middle)]">Gradient descent ASCII art generator</p>
        </div>

        <section ref={el => registerSection(el, 0)} className={`transition-opacity duration-300 ${activeSection === 0 ? 'opacity-100' : 'opacity-40'}`}>
          <h2 className="text-xs mb-4 text-[var(--text)]">Upload Image</h2>
          <ImageUploader onImageSelect={handleImageSelect} />
          {selectedImage && (
            <p className="text-xs mt-2 text-[var(--middle)]">{selectedImage.name}</p>
          )}
        </section>

        <section ref={el => registerSection(el, 1)} className={`transition-opacity duration-300 ${activeSection === 1 ? 'opacity-100' : 'opacity-40'}`}>
          <h2 className="text-xs mb-4 text-[var(--text)]">Preset</h2>
          <PresetSelector preset={preset} onPresetChange={setPreset} />
        </section>

        <section ref={el => registerSection(el, 2)} className={`transition-opacity duration-300 ${activeSection === 2 ? 'opacity-100' : 'opacity-40'}`}>
          <ParameterPanel
            iterations={iterations}
            lr={lr}
            diversityWeight={diversityWeight}
            tempStart={tempStart}
            tempEnd={tempEnd}
            onIterationsChange={setIterations}
            onLrChange={setLr}
            onDiversityWeightChange={setDiversityWeight}
            onTempStartChange={setTempStart}
            onTempEndChange={setTempEnd}
          />
        </section>

        <section ref={el => registerSection(el, 3)} className={`transition-opacity duration-300 ${activeSection === 3 ? 'opacity-100' : 'opacity-40'}`}>
        <button
          onClick={handleGenerate}
          disabled={!selectedImage || state.status === 'generating'}
          className="text-left w-fit disabled:text-[var(--middle)] disabled:cursor-default disabled:no-underline"
        >
          {state.status === 'generating' ? 'Generating...' : 'Generate ASCII'}
        </button>

        {state.status === 'generating' && (
          <div className="mt-4">
            <ProgressBar
              progress={state.progress}
              currentIteration={state.currentIteration}
              totalIterations={iterations}
            />
          </div>
        )}

        {state.status === 'error' && (
          <div className="mt-4 text-[var(--middle)]">
            {state.error}
          </div>
        )}
        </section>
      </div>

      <div className="flex-1 pb-16 min-w-[500px]">
        <section className="sticky top-20">
          <h2 className="text-xs mb-4 text-[var(--text)]">Preview</h2>
          <AsciiDisplay
            originalImage={selectedImage ? URL.createObjectURL(selectedImage) : undefined}
            asciiPng={state.result?.png}
            asciiText={state.result?.text}
            status={state.status}
          />

          {state.status === 'complete' && state.result && (
            <div className="mt-8">
              <h2 className="text-xs mb-4 text-[var(--text)]">Export</h2>
              <ExportActions
                pngData={state.result.png}
                textData={state.result.text}
              />
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
