import {
  ArrowClockwise,
  CheckCircle,
  CircleNotch,
  CloudArrowUp,
  DownloadSimple,
  Eye,
  FileVideo,
  MagnifyingGlass,
  PencilSimple,
  Stack,
  Trash,
  WarningCircle,
} from "@phosphor-icons/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { App, Button, Card, Empty, Input, Popconfirm, Progress, Space, Table, Tooltip, Typography } from "antd";
import dayjs from "dayjs";
import { useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { api, errorMessage, fileUrl, LANG_TAG } from "../api/client";
import { ACTIVE_STATUSES, RUNNING_STATUSES, type Job, type JobStatus } from "../api/types";
import PageHeader from "../components/PageHeader";
import StatusTag from "../components/StatusTag";

type Filter = "all" | "active" | "review" | "done" | "failed";

const FILTERS: { key: Filter; label: string; icon: ReactNode; color: string; bg: string; match: (s: JobStatus) => boolean }[] = [
  { key: "all", label: "Tất cả", icon: <Stack size={20} aria-hidden />, color: "#4F46E5", bg: "#EEF2FF", match: () => true },
  {
    key: "active",
    label: "Đang xử lý",
    icon: <CircleNotch size={20} aria-hidden />,
    color: "#4F46E5",
    bg: "#EEF2FF",
    match: (s) => ACTIVE_STATUSES.includes(s),
  },
  { key: "review", label: "Chờ duyệt", icon: <PencilSimple size={20} aria-hidden />, color: "#B45309", bg: "#FFFBEB", match: (s) => s === "REVIEW" },
  { key: "done", label: "Hoàn tất", icon: <CheckCircle size={20} aria-hidden />, color: "#047857", bg: "#ECFDF5", match: (s) => s === "DONE" },
  { key: "failed", label: "Lỗi", icon: <WarningCircle size={20} aria-hidden />, color: "#B91C1C", bg: "#FEF2F2", match: (s) => s === "FAILED" },
];

const fmtDur = (v: number | null) => (v ? `${Math.floor(v / 60)}:${String(Math.round(v % 60)).padStart(2, "0")}` : "–");

export default function JobsPage() {
  const { message } = App.useApp();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");

  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.listJobs(200),
    refetchInterval: (q) => (q.state.data?.items.some((j) => ACTIVE_STATUSES.includes(j.status)) ? 2000 : false),
  });

  const onError = (e: unknown) => message.error(errorMessage(e));
  const retry = useMutation({
    mutationFn: api.retry,
    onSuccess: () => {
      message.success("Đã gửi xử lý lại");
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError,
  });
  const del = useMutation({
    mutationFn: api.deleteJob,
    onSuccess: () => {
      message.success("Đã xóa video");
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError,
  });

  const items = data?.items ?? [];
  const counts = useMemo(
    () => Object.fromEntries(FILTERS.map((f) => [f.key, items.filter((j) => f.match(j.status)).length])) as Record<Filter, number>,
    [items],
  );
  const rows = useMemo(() => {
    const f = FILTERS.find((x) => x.key === filter)!;
    const q = search.trim().toLowerCase();
    return items.filter((j) => f.match(j.status) && (!q || j.original_filename.toLowerCase().includes(q)));
  }, [items, filter, search]);

  return (
    <>
      <PageHeader
        title="Danh sách video"
        subtitle="Theo dõi tiến độ, duyệt bản dịch và tải video đã lồng tiếng."
        extra={
          <>
            <Button icon={<ArrowClockwise size={16} aria-hidden />} onClick={() => refetch()} loading={isFetching}>
              Làm mới
            </Button>
            <Button type="primary" icon={<CloudArrowUp size={16} aria-hidden />} onClick={() => navigate("/new")}>
              Tạo video mới
            </Button>
          </>
        }
      />

      <div className="stat-grid" role="group" aria-label="Lọc theo trạng thái">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            type="button"
            className={`stat-card${filter === f.key ? " active" : ""}`}
            aria-pressed={filter === f.key}
            onClick={() => setFilter(f.key)}
          >
            <span className="stat-icon" style={{ background: f.bg, color: f.color }}>
              {f.icon}
            </span>
            <span>
              <div className="stat-value">{counts[f.key] ?? 0}</div>
              <div className="stat-label">{f.label}</div>
            </span>
          </button>
        ))}
      </div>

      <Card className="soft-card" styles={{ body: { padding: 0 } }}>
        <div style={{ padding: "16px 20px", display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <Input
            allowClear
            prefix={<MagnifyingGlass size={16} aria-hidden color="#64748B" />}
            placeholder="Tìm theo tên file…"
            aria-label="Tìm theo tên file"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ maxWidth: 320 }}
          />
          <Typography.Text type="secondary">{rows.length} video</Typography.Text>
        </div>
        <Table<Job>
          rowKey="id"
          loading={isLoading}
          dataSource={rows}
          pagination={rows.length > 20 ? { pageSize: 20, showSizeChanger: false } : false}
          scroll={{ x: 960 }}
          onRow={(r) => ({ onDoubleClick: () => navigate(`/jobs/${r.id}`) })}
          locale={{
            emptyText: (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={items.length ? "Không có video phù hợp bộ lọc" : "Chưa có video nào"}
              >
                {!items.length && (
                  <Button type="primary" icon={<CloudArrowUp size={16} aria-hidden />} onClick={() => navigate("/new")}>
                    Tải video đầu tiên
                  </Button>
                )}
              </Empty>
            ),
          }}
          columns={[
            {
              title: "Video",
              dataIndex: "original_filename",
              render: (_: string, r) => (
                <div className="file-cell">
                  <span className="file-thumb">
                    <FileVideo size={20} aria-hidden />
                  </span>
                  <span style={{ minWidth: 0 }}>
                    <Typography.Link className="file-name" onClick={() => navigate(`/jobs/${r.id}`)} title={r.original_filename}>
                      {r.original_filename}
                    </Typography.Link>
                    <span className="file-meta">
                      {fmtDur(r.duration_sec)} · {r.segment_count ? `${r.segment_count} câu` : "chưa phân tích"}
                    </span>
                  </span>
                </div>
              ),
            },
            {
              title: "Ngôn ngữ",
              dataIndex: "target_lang",
              width: 120,
              onHeaderCell: () => ({ style: { whiteSpace: "nowrap" } }),
              render: (v: string) => <span className={`lang-badge ${v}`}>{LANG_TAG[v] ?? v}</span>,
            },
            {
              title: "Trạng thái",
              dataIndex: "status",
              width: 260,
              render: (_: unknown, r) => (
                <Space direction="vertical" size={4} style={{ width: "100%" }}>
                  <StatusTag status={r.status} />
                  <Typography.Text
                    type={r.status === "FAILED" ? "danger" : "secondary"}
                    style={{ fontSize: 12.5 }}
                    ellipsis={{ tooltip: r.status === "FAILED" ? r.error : r.stage_message }}
                  >
                    {r.status === "FAILED" ? r.error : r.stage_message}
                  </Typography.Text>
                </Space>
              ),
            },
            {
              title: "Tiến độ",
              dataIndex: "progress",
              width: 150,
              render: (v: number, r) => (
                <Progress
                  percent={v}
                  size="small"
                  strokeColor={r.status === "DONE" ? "#059669" : undefined}
                  status={r.status === "FAILED" ? "exception" : RUNNING_STATUSES.includes(r.status) ? "active" : "normal"}
                  aria-label={`Tiến độ ${v}%`}
                />
              ),
            },
            {
              title: "Tạo lúc",
              dataIndex: "created_at",
              width: 140,
              render: (v: string) => (
                <Tooltip title={dayjs.utc(v).local().format("DD/MM/YYYY HH:mm")}>
                  <span>{dayjs.utc(v).local().fromNow()}</span>
                </Tooltip>
              ),
            },
            {
              title: <span className="sr-only">Hành động</span>,
              width: 190,
              align: "right",
              render: (_: unknown, r) => (
                <Space size={4}>
                  {r.status === "REVIEW" ? (
                    <Button size="small" type="primary" ghost onClick={() => navigate(`/jobs/${r.id}`)}>
                      Duyệt
                    </Button>
                  ) : (
                    <Tooltip title="Xem chi tiết">
                      <Button
                        type="text"
                        icon={<Eye size={18} aria-hidden />}
                        aria-label="Xem chi tiết"
                        onClick={() => navigate(`/jobs/${r.id}`)}
                      />
                    </Tooltip>
                  )}
                  <Tooltip title={r.has_output ? "Tải MP4" : "Chưa có video"}>
                    <Button
                      type="text"
                      icon={<DownloadSimple size={18} aria-hidden />}
                      aria-label="Tải MP4"
                      disabled={!r.has_output}
                      href={r.has_output ? fileUrl(r.id, "output") : undefined}
                      download
                    />
                  </Tooltip>
                  {r.status === "FAILED" && (
                    <Tooltip title="Thử lại">
                      <Button
                        type="text"
                        icon={<ArrowClockwise size={18} aria-hidden />}
                        aria-label="Thử lại"
                        onClick={() => retry.mutate(r.id)}
                      />
                    </Tooltip>
                  )}
                  <Popconfirm
                    title="Xóa video này?"
                    description="File gốc và kết quả sẽ bị xóa vĩnh viễn."
                    okText="Xóa"
                    okButtonProps={{ danger: true }}
                    cancelText="Hủy"
                    onConfirm={() => del.mutate(r.id)}
                  >
                    <Tooltip title={RUNNING_STATUSES.includes(r.status) ? "Đang xử lý, chưa thể xóa" : "Xóa"}>
                      <Button
                        type="text"
                        danger
                        icon={<Trash size={18} aria-hidden />}
                        aria-label="Xóa video"
                        disabled={RUNNING_STATUSES.includes(r.status)}
                      />
                    </Tooltip>
                  </Popconfirm>
                </Space>
              ),
            },
          ]}
        />
      </Card>
    </>
  );
}
