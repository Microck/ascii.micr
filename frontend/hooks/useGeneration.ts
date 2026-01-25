import { useState, useCallback } from 'react';
import { generateAscii, type GenerationRequest } from '@/lib/api';

export type GenerationStatus = 'idle' | 'generating' | 'complete' | 'error';

export interface GenerationState {
  status: GenerationStatus;
  progress: number;
  currentIteration: number;
  result: {
    png: string;
    text: string;
  } | null;
  error: string | null;
}

export function useGeneration() {
  const [state, setState] = useState<GenerationState>({
    status: 'idle',
    progress: 0,
    currentIteration: 0,
    result: null,
    error: null,
  });

  const generate = useCallback(async (request: GenerationRequest) => {
    setState({
      status: 'generating',
      progress: 0,
      currentIteration: 0,
      result: null,
      error: null,
    });

    try {
      const result = await generateAscii(request);
      setState({
        status: 'complete',
        progress: 100,
        currentIteration: request.params.iterations,
        result,
        error: null,
      });
    } catch (error) {
      setState({
        status: 'error',
        progress: 0,
        currentIteration: 0,
        result: null,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }, []);

  const reset = useCallback(() => {
    setState({
      status: 'idle',
      progress: 0,
      currentIteration: 0,
      result: null,
      error: null,
    });
  }, []);

  return {
    state,
    generate,
    reset,
  };
}
