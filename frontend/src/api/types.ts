export type JobStatus =
  | "QUEUED"
  | "PREPARING"
  | "TRANSCRIBING"
  | "TRANSLATING"
  | "REVIEW"
  | "RENDER_QUEUED"
  | "SYNTHESIZING"
  | "MIXING"
  | "RENDERING"
  | "DONE"
  | "FAILED";

export type TargetLang = "en" | "vi";
export type SubtitleMode = "hard" | "soft" | "none";
export type Gender = "adam" | "female" | "male";

export interface Job {
  id: string;
  group_id: string;
  created_at: string;
  updated_at: string;
  original_filename: string;
  duration_sec: number | null;
  width: number | null;
  height: number | null;
  video_codec: string | null;
  source_lang: string | null;
  target_lang: TargetLang;
  voice_gender: Gender;
  subtitle_mode: SubtitleMode;
  auto_approve: boolean;
  orig_volume: number;
  status: JobStatus;
  progress: number;
  stage_message: string | null;
  error: string | null;
  failed_stage: string | null;
  has_output: boolean;
  segment_count: number;
  overflow_count: number;
}

export interface JobList {
  items: Job[];
  total: number;
}

export type FitStatus = "pending" | "ok" | "sped_up" | "shortened" | "overflow";

export interface Segment {
  id: number;
  job_id: string;
  idx: number;
  start: number;
  end: number;
  source_text: string;
  translated_text: string;
  max_chars: number;
  edited: boolean;
  tts_duration: number | null;
  placed_start: number | null;
  placed_end: number | null;
  speed: number | null;
  fit_status: FitStatus;
}

export interface GlossaryTerm {
  id: number;
  source_term: string;
  en: string | null;
  vi: string | null;
  note: string | null;
}

export interface AppConfig {
  target_languages: { code: TargetLang; label: string }[];
  max_upload_mb: number;
  max_duration_sec: number;
  default_orig_volume: number;
  tts_provider?: string;
  elevenlabs_configured?: boolean;
  elevenlabs_key_masked?: string;
}

export interface ElevenLabsSaveResp {
  configured: boolean;
  tts_provider: string;
  elevenlabs_key_masked: string;
  message: string;
  tier?: string;
  character_count?: number;
  character_limit?: number;
}

export const ACTIVE_STATUSES: JobStatus[] = [
  "QUEUED",
  "PREPARING",
  "TRANSCRIBING",
  "TRANSLATING",
  "RENDER_QUEUED",
  "SYNTHESIZING",
  "MIXING",
  "RENDERING",
];

export const RUNNING_STATUSES: JobStatus[] = [
  "PREPARING",
  "TRANSCRIBING",
  "TRANSLATING",
  "SYNTHESIZING",
  "MIXING",
  "RENDERING",
];
