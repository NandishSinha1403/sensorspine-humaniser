const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
const API_BASE = `${API_URL}/api`;

export interface MetricStats {
  burstiness: number;
  perplexity: number;
  ai_score: number;
}

export interface HumanizationMetrics {
  baseline: MetricStats;
  humanized: MetricStats;
  latency: number;
}

export interface DetectionResponse {
  score: number;
  label: string;
  sentences: any[];
}

export interface HumanizeResponse {
  original_text: string;
  humanized_text: string;
  original_score: number;
  humanized_score: number;
  passes_applied: string[];
  sentences: any[];
  metrics?: HumanizationMetrics;
  voice_profile?: {
    preferred_sentence_length: number;
    length_variance: number;
    favorite_openers: string[];
    connector_ratio: number;
    punctuation_habits: {
      comma_rate: number;
      semicolon_rate: number;
      question_rate: number;
      paren_rate: number;
      dash_rate: number;
    };
    formality_score: number;
    person: string;
  };
  changes_made: {
    phrases_replaced: number;
    sentences_restructured: number;
    paragraph_rhythm_fixes: number;
    rhythm_adjustments: number;
    burstiness_injections: number;
    lexical_substitutions: number;
    style_overlays: number;
    voice_conversions: number;
    clause_reorders: number;
    discourse_markers: number;
    audit_iterations: number;
    voice_adjustments: number;
    final_cleanups: number;
  };
}

export const detectText = async (text: string): Promise<DetectionResponse> => {
  const res = await fetch(`${API_BASE}/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error("Detection failed");
  const data = await res.json();
  return data;
};

export const humanizeText = async (text: string, intensity: number): Promise<HumanizeResponse> => {
  const res = await fetch(`${API_BASE}/humanize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, intensity }),
  });
  if (!res.ok) throw new Error("Humanization failed");
  const data = await res.json();
  return data;
};

