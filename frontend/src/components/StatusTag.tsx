import { CheckCircle, CircleNotch, Clock, PencilSimple, WarningCircle } from "@phosphor-icons/react";
import type { JobStatus } from "../api/types";

type Kind = "queued" | "running" | "review" | "done" | "failed";

const MAP: Record<JobStatus, { label: string; kind: Kind }> = {
  QUEUED: { label: "Đang chờ", kind: "queued" },
  RENDER_QUEUED: { label: "Đang chờ", kind: "queued" },
  PREPARING: { label: "Đang xử lý", kind: "running" },
  TRANSCRIBING: { label: "Đang xử lý", kind: "running" },
  TRANSLATING: { label: "Đang xử lý", kind: "running" },
  SYNTHESIZING: { label: "Đang xử lý", kind: "running" },
  MIXING: { label: "Đang xử lý", kind: "running" },
  RENDERING: { label: "Đang xử lý", kind: "running" },
  REVIEW: { label: "Chờ duyệt", kind: "review" },
  DONE: { label: "Hoàn tất", kind: "done" },
  FAILED: { label: "Lỗi", kind: "failed" },
};

function Icon({ kind }: { kind: Kind }) {
  const p = { size: 14, weight: "bold" as const, "aria-hidden": true };
  switch (kind) {
    case "queued":
      return <Clock {...p} />;
    case "running":
      return <CircleNotch {...p} className="spin" />;
    case "review":
      return <PencilSimple {...p} />;
    case "done":
      return <CheckCircle {...p} weight="fill" />;
    case "failed":
      return <WarningCircle {...p} weight="fill" />;
  }
}

/** Status is conveyed by icon + text, never by color alone. */
export default function StatusTag({ status }: { status: JobStatus }) {
  const m = MAP[status] ?? { label: status, kind: "queued" as Kind };
  return (
    <span className={`status-pill ${m.kind}`}>
      <Icon kind={m.kind} />
      {m.label}
    </span>
  );
}
