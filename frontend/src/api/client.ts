import axios, { AxiosError } from "axios";
import type { AppConfig, ElevenLabsSaveResp, Gender, GlossaryTerm, Job, JobList, Segment, TargetLang } from "./types";

export const http = axios.create({ baseURL: "/api" });

export function errorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = (err.response?.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return "Dữ liệu gửi lên không hợp lệ";
    return err.message;
  }
  return String(err);
}

export const fileUrl = (jobId: string, kind: "source" | "output" | "srt") => `/api/jobs/${jobId}/files/${kind}`;
export const voicePreviewUrl = (voice: Gender, lang: TargetLang, engine = "edge", rev = 0) =>
  `/api/voices/preview?voice=${voice}&lang=${lang}&engine=${engine}&v=clean3_${rev}`;

export const VOICE_LABEL: Record<string, string> = {
  adam: "Adam (Dominant, Firm)",
  female: "Nữ",
  male: "Nam",
};

export const api = {
  config: async () => (await http.get<AppConfig>("/config")).data,
  saveElevenLabsKey: async (api_key: string) =>
    (await http.post<ElevenLabsSaveResp>("/settings/elevenlabs", { api_key })).data,

  listJobs: async (limit = 100, offset = 0) =>
    (await http.get<JobList>("/jobs", { params: { limit, offset } })).data,
  getJob: async (id: string) => (await http.get<Job>(`/jobs/${id}`)).data,
  createJobs: async (form: FormData, onProgress?: (pct: number) => void) =>
    (
      await http.post<Job[]>("/jobs", form, {
        onUploadProgress: (e) => {
          if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
        },
      })
    ).data,
  getSegments: async (id: string) => (await http.get<Segment[]>(`/jobs/${id}/segments`)).data,
  updateSegments: async (id: string, items: { idx: number; translated_text: string }[]) =>
    (await http.put<Segment[]>(`/jobs/${id}/segments`, { items })).data,
  approve: async (id: string) => (await http.post<Job>(`/jobs/${id}/approve`)).data,
  reopen: async (id: string) => (await http.post<Job>(`/jobs/${id}/reopen`)).data,
  retry: async (id: string) => (await http.post<Job>(`/jobs/${id}/retry`)).data,
  deleteJob: async (id: string) => (await http.delete(`/jobs/${id}`)).data,

  listGlossary: async () => (await http.get<GlossaryTerm[]>("/glossary")).data,
  createGlossary: async (t: Omit<GlossaryTerm, "id">) => (await http.post<GlossaryTerm>("/glossary", t)).data,
  updateGlossary: async (id: number, t: Omit<GlossaryTerm, "id">) =>
    (await http.put<GlossaryTerm>(`/glossary/${id}`, t)).data,
  deleteGlossary: async (id: number) => (await http.delete(`/glossary/${id}`)).data,
};

/** Must match backend app/text/budget.py count_chars */
export function countChars(text: string, lang: string): number {
  if (lang === "zh") {
    let n = 0;
    for (const ch of text) {
      if (/[\p{P}\p{Z}\s]/u.test(ch)) continue;
      n += 1;
    }
    return n;
  }
  return text.trim().length;
}

export function fmtTime(sec: number | null | undefined): string {
  if (sec == null) return "--:--";
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

export const LANG_TAG: Record<string, string> = { en: "EN", vi: "VI", zh: "中文" };
