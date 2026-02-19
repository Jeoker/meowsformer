/**
 * Meowsformer — API Type Definitions
 * ====================================
 * TypeScript interfaces mirroring the backend Pydantic schemas.
 * These are the contracts between the FastAPI backend and the React frontend.
 */

/** Phase 0 — LLM analysis response */
export interface CatTranslationResponse {
  sound_id: string;
  pitch_adjust: number;
  human_interpretation: string;
  emotion_category: "Hungry" | "Angry" | "Happy" | "Alert";
  behavior_note: string;
}

/** NatureLM-audio-style preview description */
export interface PreviewDescription {
  summary: string;
  intent_label: string;
  vocalisation_type: string;
  confidence_score: number;
  confidence_level: string;
  va_distance: number;
  pitch_description: string;
  tempo_description: string;
  breed: string;
  source_context: string;
  detail: string;
}

/** Technical metadata about the synthesis */
export interface SynthesisMetadata {
  matched_sample_id: string;
  matched_breed: string;
  matched_context: string;
  target_valence: number;
  target_arousal: number;
  duration_seconds: number;
  sample_rate: number;
}

/** Phase 3 — Full synthesis response */
export interface MeowSynthesisResponse extends CatTranslationResponse {
  audio_base64: string | null;
  preview_description: PreviewDescription | null;
  synthesis_metadata: SynthesisMetadata | null;
  synthesis_ok: boolean;
}

/** Request parameters for the v1/translate endpoint */
export interface TranslateRequest {
  file: File;
  breed?: string;
  output_sr?: 16000 | 44100;
}

// ── Streaming / WebSocket Types ─────────────────────────────────────────

/** LLM target-tag set describing the ideal cat sound */
export interface TargetTagSet {
  emotion: string[];
  intent: string[];
  acoustic: string[];
  social_context: string[];
  reasoning: string;
}

/** Matched sample info returned from the matching engine */
export interface TaggedSampleInfo {
  sample_id: string;
  breed: string;
  context: string;
  tags: Record<string, string[]>;
  match_score: number;
  matched_tags: Record<string, string[]>;
}

/** Final streaming translation result */
export interface StreamingTranslationResult {
  transcription: string;
  target_tags: TargetTagSet;
  selected_sample: TaggedSampleInfo;
  audio_base64: string;
  reasoning: string;
}

// ── WebSocket message types ─────────────────────────────────────────────

export interface WSTranscriptionMessage {
  type: "transcription";
  text: string;
  is_final: boolean;
}

export interface WSAnalysisPreviewMessage {
  type: "analysis_preview";
  emotion: string;
  intent: string;
}

export interface WSResultMessage {
  type: "result";
  transcription: string;
  selected_category: TaggedSampleInfo;
  audio_base64: string;
  reasoning: string;
}

export interface WSErrorMessage {
  type: "error";
  detail: string;
}

export type WSServerMessage =
  | WSTranscriptionMessage
  | WSAnalysisPreviewMessage
  | WSResultMessage
  | WSErrorMessage;
