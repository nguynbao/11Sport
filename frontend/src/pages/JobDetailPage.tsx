import {
  ArrowClockwise,
  ArrowLeft,
  ClosedCaptioning,
  DownloadSimple,
  FileText,
  GenderFemale,
  GenderMale,
  MicrophoneStage,
  PencilSimple,
  Timer,
  Translate,
} from "@phosphor-icons/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, App, Button, Card, Col, Progress, Result, Row, Skeleton, Steps, Typography } from "antd";
import { useRef, type ReactNode } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, errorMessage, fileUrl, LANG_TAG, VOICE_LABEL } from "../api/client";
import { ACTIVE_STATUSES, type JobStatus } from "../api/types";
import PageHeader from "../components/PageHeader";
import SegmentEditor from "../components/SegmentEditor";
import StatusTag from "../components/StatusTag";

const STEP_OF: Record<JobStatus, number> = {
  QUEUED: 0,
  PREPARING: 0,
  TRANSCRIBING: 1,
  TRANSLATING: 2,
  REVIEW: 3,
  RENDER_QUEUED: 4,
  SYNTHESIZING: 4,
  MIXING: 4,
  RENDERING: 5,
  DONE: 6,
  FAILED: 0,
};
const FAILED_STEP: Record<string, number> = { prepare: 0, transcribe: 1, translate: 2, synthesize: 4, mix: 4, render: 5 };
const SUB_LABEL = { hard: "Gắn cứng", soft: "Phụ đề rời", none: "Không phụ đề" } as const;

function Meta({ icon, label, value }: { icon: ReactNode; label: string; value: ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
      <span className="stat-icon" style={{ width: 34, height: 34, borderRadius: 10, background: "#EEF2FF", color: "#4F46E5" }}>
        {icon}
      </span>
      <div>
        <div className="file-meta">{label}</div>
        <div style={{ fontWeight: 600 }}>{value}</div>
      </div>
    </div>
  );
}

export default function JobDetailPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const { message } = App.useApp();
  const qc = useQueryClient();
  const videoRef = useRef<HTMLVideoElement>(null);

  const jobQ = useQuery({
    queryKey: ["job", id],
    queryFn: () => api.getJob(id),
    refetchInterval: (q) => (q.state.data && ACTIVE_STATUSES.includes(q.state.data.status) ? 2000 : false),
  });
  const job = jobQ.data;
  const segQ = useQuery({
    queryKey: ["segments", id, job?.status],
    queryFn: () => api.getSegments(id),
    enabled: !!job && job.segment_count > 0,
  });

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["job", id] });
    qc.invalidateQueries({ queryKey: ["segments", id] });
    qc.invalidateQueries({ queryKey: ["jobs"] });
  };
  const onError = (e: unknown) => message.error(errorMessage(e));

  const save = useMutation({
    mutationFn: async ({ items, approve }: { items: { idx: number; translated_text: string }[]; approve: boolean }) => {
      if (items.length) await api.updateSegments(id, items);
      if (approve) await api.approve(id);
      return approve;
    },
    onSuccess: (approved) => {
      message.success(approved ? "Đã gửi xuất video" : "Đã lưu nháp");
      refresh();
    },
    onError,
  });
  const reopen = useMutation({ mutationFn: () => api.reopen(id), onSuccess: refresh, onError });
  const retry = useMutation({ mutationFn: () => api.retry(id), onSuccess: refresh, onError });

  if (jobQ.isLoading)
    return (
      <Card className="soft-card">
        <Skeleton active paragraph={{ rows: 6 }} />
      </Card>
    );
  if (jobQ.isError || !job)
    return (
      <Result
        status="404"
        title="Không tìm thấy video"
        subTitle="Video có thể đã bị xóa hoặc hết hạn lưu trữ."
        extra={<Button onClick={() => navigate("/")}>Về danh sách</Button>}
      />
    );

  const step = job.status === "FAILED" ? FAILED_STEP[job.failed_stage ?? ""] ?? 0 : STEP_OF[job.status];
  const seek = (sec: number) => {
    const v = videoRef.current;
    if (v) {
      v.currentTime = sec;
      void v.play();
    }
  };
  const active = ACTIVE_STATUSES.includes(job.status);

  return (
    <>
      <PageHeader
        before={
          <Button
            icon={<ArrowLeft size={18} aria-hidden />}
            aria-label="Quay lại danh sách"
            onClick={() => navigate("/")}
            style={{ flex: "none" }}
          />
        }
        title={
          <span style={{ display: "inline-flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ overflowWrap: "anywhere" }}>{job.original_filename}</span>
            <StatusTag status={job.status} />
          </span>
        }
        subtitle={job.status === "REVIEW" ? "Kiểm tra bản dịch, sửa nếu cần, rồi bấm “Lưu & xuất video”." : job.stage_message}
        extra={
          job.status === "DONE" && (
            <>
              <Button icon={<PencilSimple size={16} aria-hidden />} onClick={() => reopen.mutate()} loading={reopen.isPending}>
                Sửa bản dịch & xuất lại
              </Button>
              <Button icon={<FileText size={16} aria-hidden />} href={fileUrl(id, "srt")} download>
                Tải SRT
              </Button>
              <Button type="primary" icon={<DownloadSimple size={16} aria-hidden />} href={fileUrl(id, "output")} download>
                Tải MP4
              </Button>
            </>
          )
        }
      />

      <Card className="soft-card" style={{ marginBottom: 20 }}>
        <Steps
          size="small"
          current={step}
          status={job.status === "FAILED" ? "error" : job.status === "DONE" ? "finish" : "process"}
          responsive
          items={[
            { title: "Phân tích" },
            { title: "Nhận diện" },
            { title: "Dịch" },
            { title: "Duyệt" },
            { title: "Lồng tiếng" },
            { title: "Xuất video" },
          ]}
        />
        {active && (
          <div style={{ marginTop: 18 }} aria-live="polite">
            <Progress percent={job.progress} status="active" />
          </div>
        )}
        <div
          style={{
            marginTop: 20,
            display: "grid",
            gap: 16,
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          }}
        >
          <Meta icon={<Translate size={18} aria-hidden />} label="Ngôn ngữ" value={`${job.source_lang ?? "?"} → ${LANG_TAG[job.target_lang]}`} />
          <Meta
            icon={
              job.voice_gender === "female" ? (
                <GenderFemale size={18} aria-hidden />
              ) : job.voice_gender === "adam" ? (
                <MicrophoneStage size={18} aria-hidden />
              ) : (
                <GenderMale size={18} aria-hidden />
              )
            }
            label="Giọng đọc"
            value={VOICE_LABEL[job.voice_gender] ?? job.voice_gender}
          />
          <Meta icon={<ClosedCaptioning size={18} aria-hidden />} label="Phụ đề" value={SUB_LABEL[job.subtitle_mode]} />
          <Meta
            icon={<Timer size={18} aria-hidden />}
            label="Thời lượng · số câu"
            value={`${job.duration_sec ? `${job.duration_sec.toFixed(1)}s` : "–"} · ${job.segment_count}`}
          />
        </div>
      </Card>

      {job.status === "FAILED" && (
        <Alert
          type="error"
          showIcon
          style={{ marginBottom: 20 }}
          message="Xử lý thất bại"
          description={<Typography.Paragraph style={{ whiteSpace: "pre-wrap", margin: 0 }}>{job.error}</Typography.Paragraph>}
          action={
            <Button icon={<ArrowClockwise size={16} aria-hidden />} onClick={() => retry.mutate()} loading={retry.isPending}>
              Thử lại
            </Button>
          }
        />
      )}

      {job.status === "REVIEW" && (
        <Row gutter={[20, 20]}>
          <Col xs={24} xl={8}>
            <Card className="soft-card" title="Video gốc" style={{ position: "sticky", top: 20 }}>
              <video ref={videoRef} src={fileUrl(id, "source")} controls preload="metadata" className="video-frame" />
              <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0, fontSize: 13 }}>
                Bấm nút ▶ ở cột thời gian để nghe lại đoạn tương ứng. Bản dịch càng ngắn gọn thì giọng đọc càng tự nhiên.
              </Typography.Paragraph>
            </Card>
          </Col>
          <Col xs={24} xl={16}>
            <Card className="soft-card" title={`Duyệt bản dịch (${job.segment_count} câu)`}>
              {segQ.data ? (
                <SegmentEditor
                  segments={segQ.data}
                  lang={job.target_lang}
                  saving={save.isPending}
                  onSeek={seek}
                  onSave={(items, approve) => save.mutate({ items, approve })}
                />
              ) : (
                <Skeleton active />
              )}
            </Card>
          </Col>
        </Row>
      )}

      {job.status === "DONE" && (
        <Card className="soft-card" title="Video kết quả">
          {job.overflow_count > 0 && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 14 }}
              message={`Có ${job.overflow_count} câu đọc vượt thời gian. Nên rút gọn bản dịch rồi xuất lại.`}
              action={
                <Button size="small" onClick={() => reopen.mutate()}>
                  Sửa ngay
                </Button>
              }
            />
          )}
          <video
            key={job.updated_at}
            src={`${fileUrl(id, "output")}?v=${encodeURIComponent(job.updated_at)}`}
            controls
            preload="metadata"
            className="video-frame"
            style={{ maxHeight: 620, objectFit: "contain" }}
          />
        </Card>
      )}

      {active && job.status !== "RENDER_QUEUED" && STEP_OF[job.status] < 3 && (
        <Card className="soft-card">
          <Skeleton active title={false} paragraph={{ rows: 4 }} />
          <Typography.Text type="secondary">Bản dịch sẽ hiện ở đây khi AI xử lý xong.</Typography.Text>
        </Card>
      )}
    </>
  );
}
