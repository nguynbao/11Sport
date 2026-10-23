import { FloppyDisk, PaperPlaneTilt, Play, WarningCircle } from "@phosphor-icons/react";
import { Button, Input, Space, Table, Tag, Tooltip, Typography } from "antd";
import { useEffect, useMemo, useState } from "react";
import { countChars, fmtTime } from "../api/client";
import type { Segment } from "../api/types";

interface Props {
  segments: Segment[];
  lang: string;
  saving: boolean;
  onSeek: (sec: number) => void;
  onSave: (items: { idx: number; translated_text: string }[], approve: boolean) => void;
}

const FIT_TAG: Record<string, { color: string; label: string } | undefined> = {
  ok: { color: "success", label: "Vừa khung" },
  sped_up: { color: "processing", label: "Đọc nhanh hơn" },
  shortened: { color: "warning", label: "Đã rút gọn" },
  overflow: { color: "error", label: "Vượt thời gian" },
};

export default function SegmentEditor({ segments, lang, saving, onSeek, onSave }: Props) {
  const [draft, setDraft] = useState<Record<number, string>>({});

  useEffect(() => {
    setDraft({});
  }, [segments]);

  const dirty = useMemo(
    () =>
      Object.entries(draft)
        .filter(([idx, text]) => segments.find((s) => s.idx === Number(idx))?.translated_text !== text)
        .map(([idx, text]) => ({ idx: Number(idx), translated_text: text })),
    [draft, segments],
  );
  const hasEmpty = dirty.some((d) => !d.translated_text.trim());

  useEffect(() => {
    if (!dirty.length) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty.length]);

  const overCount = segments.filter((s) => countChars(draft[s.idx] ?? s.translated_text, lang) > s.max_chars).length;

  return (
    <Space direction="vertical" size={14} style={{ width: "100%" }}>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <Button
          icon={<FloppyDisk size={16} aria-hidden />}
          onClick={() => onSave(dirty, false)}
          disabled={!dirty.length || hasEmpty}
          loading={saving}
        >
          Lưu nháp
        </Button>
        <Button
          type="primary"
          icon={<PaperPlaneTilt size={16} aria-hidden />}
          onClick={() => onSave(dirty, true)}
          disabled={hasEmpty}
          loading={saving}
        >
          Lưu & xuất video
        </Button>
        <span aria-live="polite" style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {dirty.length > 0 && <Typography.Text type="warning">{dirty.length} câu chưa lưu</Typography.Text>}
          {hasEmpty && <Typography.Text type="danger">Bản dịch không được để trống</Typography.Text>}
          {overCount > 0 && (
            <Typography.Text type="danger">
              <WarningCircle size={14} weight="fill" aria-hidden style={{ verticalAlign: -2 }} /> {overCount} câu dài hơn
              khung thời gian
            </Typography.Text>
          )}
        </span>
      </div>
      <Table<Segment>
        rowKey="id"
        size="middle"
        dataSource={segments}
        pagination={false}
        scroll={{ y: 600, x: 660 }}
        columns={[
          {
            title: "Thời gian",
            width: 108,
            render: (_: unknown, s) => (
              <Button
                type="text"
                size="small"
                icon={<Play size={14} weight="fill" aria-hidden />}
                onClick={() => onSeek(s.start)}
                aria-label={`Phát đoạn ${s.idx + 1} từ ${fmtTime(s.start)}`}
                className="seg-time"
                style={{ paddingInline: 6 }}
              >
                {fmtTime(s.start)}
              </Button>
            ),
          },
          {
            title: "Câu gốc",
            dataIndex: "source_text",
            width: "32%",
            render: (v: string) => <span className="seg-source">{v}</span>,
          },
          {
            title: "Bản dịch",
            render: (_: unknown, s) => {
              const value = draft[s.idx] ?? s.translated_text;
              const n = countChars(value, lang);
              const over = n > s.max_chars;
              return (
                <div>
                  <Input.TextArea
                    value={value}
                    autoSize={{ minRows: 1, maxRows: 6 }}
                    status={!value.trim() ? "error" : over ? "warning" : undefined}
                    aria-label={`Bản dịch câu ${s.idx + 1}`}
                    onChange={(e) => setDraft((d) => ({ ...d, [s.idx]: e.target.value }))}
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, gap: 8 }}>
                    <span className="char-count" style={{ color: over ? "#B91C1C" : "#64748B" }}>
                      {n}/{s.max_chars} ký tự{over ? " · nên rút gọn" : ""}
                    </span>
                    {s.edited && <Tag style={{ marginInlineEnd: 0 }}>Đã sửa</Tag>}
                  </div>
                </div>
              );
            },
          },
          {
            title: "Khớp giọng",
            width: 124,
            render: (_: unknown, s) => {
              const t = FIT_TAG[s.fit_status];
              if (!t) return <Typography.Text type="secondary">Chưa tạo</Typography.Text>;
              return (
                <Tooltip title={s.speed ? `Tốc độ đọc x${s.speed.toFixed(2)}` : undefined}>
                  <Tag color={t.color}>{t.label}</Tag>
                </Tooltip>
              );
            },
          },
        ]}
      />
    </Space>
  );
}
