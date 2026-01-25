import type { Preset } from '@/components/PresetSelector';
import { getDemoData } from './demo';

export interface GenerationParams {
  iterations: number;
  lr: number;
  diversity_weight: number;
  temp_start: number;
  temp_end: number;
  optimize_alignment: boolean;
  dark_mode: boolean;
  encoding: string;
  row_gap: number;
}

export interface GenerationRequest {
  image: string;
  params: GenerationParams;
}

export interface GenerationResponse {
  png: string;
  text: string;
  steps?: {
    iteration: number;
    image: string;
  }[];
}

const RATE_LIMIT = 100;
const RATE_LIMIT_WINDOW = 60 * 1000;
const requestTimes: number[] = [];

function checkRateLimit(): boolean {
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW;

  const recentRequests = requestTimes.filter(time => time > windowStart);

  if (recentRequests.length >= RATE_LIMIT) {
    return false;
  }

  requestTimes.push(now);
  return true;
}

export async function generateAscii(request: GenerationRequest): Promise<GenerationResponse> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  if (!checkRateLimit()) {
    throw new Error('Rate limit exceeded. Please wait before generating again.');
  }

  try {
    const response = await fetch(`${apiUrl}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(120000),
    });

    if (!response.ok) {
      if (response.status === 0 || response.status === 502 || response.status === 503) {
        const preset = request.params.encoding === 'cp437' ? 'epson' : 'discord';
        const demo = getDemoData(preset);
        console.warn('API unavailable, using demo data');
        return demo;
      }

      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('API call failed:', error);

    if (error instanceof TypeError || error instanceof DOMException) {
      const preset = request.params.encoding === 'cp437' ? 'epson' : 'discord';
      return getDemoData(preset);
    }

    throw error instanceof Error ? error : new Error('API call failed');
  }
}
