import { BookOpen, MagnifyingGlass, PencilSimple, Plus, Trash } from "@phosphor-icons/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { App, Button, Card, Empty, Form, Input, Modal, Popconfirm, Space, Table, Tooltip, Typography } from "antd";
import { useMemo, useState } from "react";
import { api, errorMessage } from "../api/client";
import type { GlossaryTerm } from "../api/types";
import PageHeader from "../components/PageHeader";

type TermForm = Omit<GlossaryTerm, "id">;

export default function GlossaryPage() {
  const { message } = App.useApp();
  const qc = useQueryClient();
  const [editing, setEditing] = useState<GlossaryTerm | null | "new">(null);
  const [search, setSearch] = useState("");
  const [form] = Form.useForm<TermForm>();
  const { data, isLoading } = useQuery({ queryKey: ["glossary"], queryFn: api.listGlossary });

  const done = (msg: string) => {
    message.success(msg);
    setEditing(null);
    qc.invalidateQueries({ queryKey: ["glossary"] });
  };
  const onError = (e: unknown) => message.error(errorMessage(e));
  const save = useMutation({
    mutationFn: (v: TermForm) => (editing && editing !== "new" ? api.updateGlossary(editing.id, v) : api.createGlossary(v)),
    onSuccess: () => done("Đã lưu thuật ngữ"),
    onError,
  });
  const del = useMutation({ mutationFn: api.deleteGlossary, onSuccess: () => done("Đã xóa thuật ngữ"), onError });

  const open = (t: GlossaryTerm | "new") => {
    setEditing(t);
    form.setFieldsValue(t === "new" ? { source_term: "", en: "", vi: "", note: "" } : t);
  };

  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();
    return (data ?? []).filter(
      (t) => !q || [t.source_term, t.en, t.vi, t.note].some((v) => (v ?? "").toLowerCase().includes(q)),
    );
  }, [data, search]);

  return (
    <>
      <PageHeader
        title="Bảng thuật ngữ"
        subtitle="Tên sản phẩm, thương hiệu và từ chuyên ngành. AI dùng bảng này để nhận diện giọng nói chính xác hơn và bắt buộc dịch đúng."
        extra={
          <Button type="primary" icon={<Plus size={16} weight="bold" aria-hidden />} onClick={() => open("new")}>
            Thêm thuật ngữ
          </Button>
        }
      />
      <Card className="soft-card" styles={{ body: { padding: 0 } }}>
        <div style={{ padding: "16px 20px", display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <Input
            allowClear
            prefix={<MagnifyingGlass size={16} aria-hidden color="#64748B" />}
            placeholder="Tìm thuật ngữ…"
            aria-label="Tìm thuật ngữ"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ maxWidth: 320 }}
          />
          <Typography.Text type="secondary">{rows.length} thuật ngữ</Typography.Text>
        </div>
        <Table<GlossaryTerm>
          rowKey="id"
          loading={isLoading}
          dataSource={rows}
          pagination={rows.length > 20 ? { pageSize: 20 } : false}
          scroll={{ x: 720 }}
          onRow={(r) => ({ onDoubleClick: () => open(r) })}
          locale={{
            emptyText: (
              <Empty image={<BookOpen size={48} color="#A5B4FC" aria-hidden />} description="Chưa có thuật ngữ nào">
                <Button type="primary" onClick={() => open("new")}>
                  Thêm thuật ngữ đầu tiên
                </Button>
              </Empty>
            ),
          }}
          columns={[
            { title: "Thuật ngữ gốc", dataIndex: "source_term", render: (v: string) => <Typography.Text strong>{v}</Typography.Text> },
            {
              title: <span className="lang-badge en">EN</span>,
              dataIndex: "en",
              render: (v: string | null) => v ?? <Typography.Text type="secondary">–</Typography.Text>,
            },
            {
              title: <span className="lang-badge vi">VI</span>,
              dataIndex: "vi",
              render: (v: string | null) => v ?? <Typography.Text type="secondary">–</Typography.Text>,
            },
            { title: "Ghi chú", dataIndex: "note", render: (v: string | null) => <Typography.Text type="secondary">{v}</Typography.Text> },
            {
              title: <span className="sr-only">Hành động</span>,
              width: 110,
              align: "right",
              render: (_: unknown, r) => (
                <Space size={4}>
                  <Tooltip title="Sửa">
                    <Button type="text" icon={<PencilSimple size={18} aria-hidden />} aria-label={`Sửa ${r.source_term}`} onClick={() => open(r)} />
                  </Tooltip>
                  <Popconfirm
                    title="Xóa thuật ngữ?"
                    okText="Xóa"
                    okButtonProps={{ danger: true }}
                    cancelText="Hủy"
                    onConfirm={() => del.mutate(r.id)}
                  >
                    <Tooltip title="Xóa">
                      <Button type="text" danger icon={<Trash size={18} aria-hidden />} aria-label={`Xóa ${r.source_term}`} />
                    </Tooltip>
                  </Popconfirm>
                </Space>
              ),
            },
          ]}
        />
      </Card>
      <Modal
        open={editing !== null}
        title={editing === "new" ? "Thêm thuật ngữ" : "Sửa thuật ngữ"}
        onCancel={() => setEditing(null)}
        onOk={() => form.submit()}
        okText="Lưu"
        cancelText="Hủy"
        confirmLoading={save.isPending}
        destroyOnHidden
      >
        <Form form={form} layout="vertical" onFinish={(v) => save.mutate(v)} requiredMark="optional">
          <Form.Item
            name="source_term"
            label="Thuật ngữ gốc"
            extra="Viết đúng như người nói trong video (ví dụ: 11Sport, giày chạy bộ)."
            rules={[{ required: true, whitespace: true, message: "Nhập thuật ngữ gốc" }]}
          >
            <Input autoFocus />
          </Form.Item>
          <Form.Item name="en" label="Bản dịch tiếng Anh">
            <Input />
          </Form.Item>
          <Form.Item name="vi" label="Bản dịch tiếng Việt">
            <Input />
          </Form.Item>
          <Form.Item name="note" label="Ghi chú">
            <Input />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}
