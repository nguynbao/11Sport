import {
  CaretRight,
  Check,
  CheckCircle,
  ClosedCaptioning,
  CloudArrowUp,
  FileVideo,
  GenderFemale,
  GenderMale,
  Key,
  Lightning,
  Microphone,
  MicrophoneStage,
  PencilSimple,
  Question,
  SpeakerHigh,
  SpeakerSimpleSlash,
  Stop,
  Subtitles,
  Trash,
  Translate,
  X,
} from "@phosphor-icons/react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, App, Button, Card, Collapse, Input, Progress, Slider, Space, Tag, Typography, Upload } from "antd";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { api, errorMessage, voicePreviewUrl } from "../api/client";
import type { Gender, SubtitleMode, TargetLang } from "../api/types";
import PageHeader from "../components/PageHeader";

const ACCEPT = ".mp4,.mov,.mkv,.webm,.avi";
const fmtSize = (b: number) => (b > 1024 * 1024 ? `${(b / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(b / 1024)} KB`);

interface ChoiceProps {
  checked: boolean;
  onClick: () => void;
  icon: ReactNode;
  title: string;
  desc?: string;
  role?: "checkbox" | "radio";
}

function Choice({ checked, onClick, icon, title, desc, role = "radio" }: ChoiceProps) {
  return (
    <button type="button" role={role} aria-checked={checked} className="choice-card" onClick={onClick}>
      <span className="choice-icon">{icon}</span>
      <span style={{ paddingRight: 18 }}>
        <div className="choice-title">{title}</div>
        {desc && <div className="choice-desc">{desc}</div>}
      </span>
      {checked && (
        <span className="choice-check">
          <CheckCircle size={18} weight="fill" aria-hidden />
        </span>
      )}
    </button>
  );
}

interface VoiceChoiceProps {
  voice: Gender;
  checked: boolean;
  onSelect: () => void;
  icon: ReactNode;
  title: string;
  badge?: string;
  desc: string;
  playingKey: string | null;
  onPreview: (voice: Gender, lang: TargetLang) => void;
}

function VoiceChoice({ voice, checked, onSelect, icon, title, badge, desc, playingKey, onPreview }: VoiceChoiceProps) {
  return (
    <div
      role="radio"
      tabIndex={0}
      aria-checked={checked}
      className="choice-card voice-card"
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
    >
      <span className="choice-icon">{icon}</span>
      <div className="voice-card-body">
        <div className="voice-card-head">
          <span className="choice-title">{title}</span>
          {badge && <span className="voice-badge">{badge}</span>}
        </div>
        <div className="choice-desc">{desc}</div>
        <div className="voice-preview-row">
          {(["en", "vi"] as TargetLang[]).map((l) => {
            const key = `${voice}:${l}`;
            const isPlaying = playingKey === key;
            return (
              <button
                key={l}
                type="button"
                className={`voice-preview-btn${isPlaying ? " playing" : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onPreview(voice, l);
                }}
              >
                {isPlaying ? <Stop size={13} weight="fill" aria-hidden /> : <SpeakerHigh size={14} weight="fill" aria-hidden />}
                <span>{isPlaying ? `Dừng (${l.toUpperCase()})` : `Nghe thử ${l.toUpperCase()}`}</span>
              </button>
            );
          })}
        </div>
      </div>
      {checked && (
        <span className="choice-check">
          <CheckCircle size={18} weight="fill" aria-hidden />
        </span>
      )}
    </div>
  );
}

function Section({ label, hint, children, id }: { label: string; hint?: string; children: ReactNode; id: string }) {
  return (
    <section aria-labelledby={id} style={{ marginBottom: 26 }}>
      <span id={id} className="section-label">
        {label}
      </span>
      {hint && (
        <Typography.Paragraph type="secondary" style={{ marginTop: -6, marginBottom: 10, fontSize: 13 }}>
          {hint}
        </Typography.Paragraph>
      )}
      {children}
    </section>
  );
}

export default function NewJobPage() {
  const { message } = App.useApp();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { data: cfg } = useQuery({ queryKey: ["config"], queryFn: api.config });

  const [file, setFile] = useState<File | null>(null);
  const [langs, setLangs] = useState<TargetLang[]>(["en"]);
  const [gender, setGender] = useState<Gender>("adam");
  const [subMode, setSubMode] = useState<SubtitleMode>("hard");
  const [review, setReview] = useState(true);
  const [volume, setVolume] = useState<number | null>(null);
  const [uploading, setUploading] = useState(false);
  const [pct, setPct] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const [elKey, setElKey] = useState("");
  const [savingKey, setSavingKey] = useState(false);
  const [previewRev, setPreviewRev] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  const activeEngine = cfg?.tts_provider ?? "edge";
  const hasElevenLabs = Boolean(cfg?.elevenlabs_configured);

  const handleSaveElevenLabs = async (keyToSave: string): Promise<boolean> => {
    setSavingKey(true);
    try {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
        setPlayingKey(null);
      }
      const res = await api.saveElevenLabsKey(keyToSave);
      setElKey("");
      setPreviewRev((r) => r + 1);
      await qc.invalidateQueries({ queryKey: ["config"] });
      if (res.configured && res.character_limit) {
        message.success(
          `${res.message} (Quota: ${(res.character_count ?? 0).toLocaleString()}/${res.character_limit.toLocaleString()} ký tự)`
        );
      } else {
        message.success(res.message);
      }
      return true;
    } catch (e) {
      message.error(errorMessage(e));
      return false;
    } finally {
      setSavingKey(false);
    }
  };

  const togglePreview = async (v: Gender, l: TargetLang) => {
    const key = `${v}:${l}`;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (playingKey === key) {
      setPlayingKey(null);
      return;
    }
    let engineForVoice = v === "adam" ? activeEngine : "edge";
    if (v === "adam" && elKey.trim()) {
      const ok = await handleSaveElevenLabs(elKey.trim());
      if (!ok) return;
      engineForVoice = "elevenlabs";
    }
    const audio = new Audio(voicePreviewUrl(v, l, engineForVoice, previewRev + 1));
    audioRef.current = audio;
    setPlayingKey(key);
    audio.onended = () => setPlayingKey((cur) => (cur === key ? null : cur));
    audio.onerror = () => {
      setPlayingKey((cur) => (cur === key ? null : cur));
      message.error("Không thể phát giọng mẫu");
    };
    void audio.play();
  };

  const vol = volume ?? Math.round((cfg?.default_orig_volume ?? 0.05) * 100);
  const maxMb = cfg?.max_upload_mb ?? 500;
  const maxMin = Math.round((cfg?.max_duration_sec ?? 600) / 60);

  const toggleLang = (l: TargetLang) =>
    setLangs((cur) => (cur.includes(l) ? cur.filter((x) => x !== l) : [...cur, l]));

  const pickFile = (f: File) => {
    setError(null);
    if (f.size > maxMb * 1024 * 1024) {
      setError(`File "${f.name}" vượt quá ${maxMb} MB.`);
      return;
    }
    setFile(f);
  };

  const submit = async () => {
    if (!file) return setError("Vui lòng chọn video.");
    if (!langs.length) return setError("Chọn ít nhất 1 ngôn ngữ đích.");
    setError(null);
    if (gender === "adam") {
      if (elKey.trim()) {
        const ok = await handleSaveElevenLabs(elKey.trim());
        if (!ok) return;
      } else if (!hasElevenLabs) {
        return setError("Vui lòng nhập ElevenLabs API Key cho giọng Adam (Dominant, Firm), hoặc chọn giọng Nữ / Nam để dùng miễn phí.");
      }
    }
    const fd = new FormData();
    fd.append("file", file);
    fd.append("target_langs", (["en", "vi"] as TargetLang[]).filter((l) => langs.includes(l)).join(","));
    fd.append("voice_gender", gender);
    fd.append("subtitle_mode", subMode);
    fd.append("auto_approve", review ? "false" : "true");
    fd.append("orig_volume", String(vol / 100));
    setUploading(true);
    setPct(0);
    try {
      const jobs = await api.createJobs(fd, setPct);
      message.success(`Đã tạo ${jobs.length} video, hệ thống đang xử lý`);
      navigate(jobs.length === 1 ? `/jobs/${jobs[0].id}` : "/");
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Tạo video lồng tiếng"
        subtitle="Tải video lên, chọn ngôn ngữ. AI sẽ nhận diện giọng nói, dịch, đọc thuyết minh và gắn phụ đề."
      />
      <div className="new-job-wrap">
        <Collapse
          className="soft-card how-collapse"
          bordered={false}
          expandIconPosition="end"
          expandIcon={({ isActive }) => (
            <CaretRight
              size={16}
              weight="bold"
              aria-hidden
              style={{ transform: isActive ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.2s ease" }}
            />
          )}
          items={[
            {
              key: "how",
              label: (
                <span className="how-collapse-title">
                  <span className="how-collapse-icon">
                    <Question size={18} weight="bold" aria-hidden />
                  </span>
                  <span>Hệ thống sẽ làm gì?</span>
                  <span className="how-collapse-sub">5 bước tự động · khoảng 1–3 phút</span>
                </span>
              ),
              children: (
                <>
                  <ol className="how-list">
                    {[
                      { icon: <Microphone size={16} aria-hidden />, t: "Nhận diện giọng nói", d: "Chuyển lời nói thành văn bản có mốc thời gian." },
                      { icon: <Translate size={16} aria-hidden />, t: "Dịch theo ngữ cảnh", d: "Ngắn gọn, vừa khung thời gian, đúng bảng thuật ngữ." },
                      { icon: <PencilSimple size={16} aria-hidden />, t: "Bạn duyệt bản dịch", d: "Sửa câu chữ nếu cần (có thể bỏ qua)." },
                      { icon: <Microphone size={16} aria-hidden />, t: "Đọc thuyết minh", d: "Giọng AI tự nhiên, tự giảm tiếng gốc." },
                      { icon: <Subtitles size={16} aria-hidden />, t: "Xuất video", d: "MP4 kèm phụ đề và file SRT." },
                    ].map((s, i) => (
                      <li key={s.t}>
                        <span className="how-num">{i + 1}</span>
                        <span>
                          <Typography.Text strong>{s.t}</Typography.Text>
                          <div className="file-meta">{s.d}</div>
                        </span>
                      </li>
                    ))}
                  </ol>
                  <Alert
                    style={{ marginTop: 14 }}
                    type="info"
                    showIcon
                    message="Video 1–3 phút thường xử lý xong trong khoảng 1–3 phút (không tính thời gian duyệt)."
                  />
                </>
              ),
            },
          ]}
        />

        <Card className="soft-card">
          <Section id="sec-file" label="1. Video nguồn">
            {file ? (
              <div className="selected-file">
                <span className="file-thumb">
                  <FileVideo size={22} aria-hidden />
                </span>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="file-name" title={file.name}>
                    {file.name}
                  </div>
                  <div className="file-meta">{fmtSize(file.size)}</div>
                </div>
                <Button
                  type="text"
                  icon={<X size={18} aria-hidden />}
                  aria-label="Bỏ chọn file"
                  onClick={() => setFile(null)}
                  disabled={uploading}
                />
              </div>
            ) : (
              <Upload.Dragger
                className="dropzone"
                accept={ACCEPT}
                multiple={false}
                showUploadList={false}
                beforeUpload={(f) => {
                  pickFile(f);
                  return false;
                }}
              >
                <div style={{ padding: "18px 12px" }}>
                  <div className="dropzone-icon">
                    <CloudArrowUp size={28} aria-hidden />
                  </div>
                  <Typography.Text strong style={{ fontSize: 15 }}>
                    Kéo thả video vào đây, hoặc <Typography.Link>chọn file</Typography.Link>
                  </Typography.Text>
                  <div className="file-meta" style={{ marginTop: 6 }}>
                    MP4, MOV, MKV, WEBM, AVI · tối đa {maxMb} MB · {maxMin} phút
                  </div>
                </div>
              </Upload.Dragger>
            )}
          </Section>

          <Section id="sec-lang" label="2. Ngôn ngữ đích" hint="Có thể chọn cả hai. Mỗi ngôn ngữ sẽ tạo ra một video riêng.">
            <div className="choice-grid" role="group" aria-labelledby="sec-lang">
              <Choice
                role="checkbox"
                checked={langs.includes("en")}
                onClick={() => toggleLang("en")}
                icon={<span className="lang-badge en" style={{ minWidth: 0, padding: 0, background: "none" }}>EN</span>}
                title="Tiếng Anh"
                desc="Giọng Mỹ tự nhiên"
              />
              <Choice
                role="checkbox"
                checked={langs.includes("vi")}
                onClick={() => toggleLang("vi")}
                icon={<span className="lang-badge vi" style={{ minWidth: 0, padding: 0, background: "none" }}>VI</span>}
                title="Tiếng Việt"
                desc="Giọng chuẩn, có dấu đầy đủ"
              />
            </div>
          </Section>

          <Section
            id="sec-voice"
            label="3. Giọng đọc"
            hint="Chọn giọng Adam (Dominant, Firm) sẽ dùng ElevenLabs (cần nhập API Key). Hai giọng Nữ / Nam dùng miễn phí không cần Key."
          >
            <div className="choice-grid" role="radiogroup" aria-labelledby="sec-voice">
              <VoiceChoice
                voice="adam"
                checked={gender === "adam"}
                onSelect={() => setGender("adam")}
                icon={<MicrophoneStage size={20} weight="duotone" aria-hidden />}
                title="Adam"
                badge="ElevenLabs"
                desc="Giọng Adam"
                playingKey={playingKey}
                onPreview={(v, l) => void togglePreview(v, l)}
              />
              <VoiceChoice
                voice="female"
                checked={gender === "female"}
                onSelect={() => setGender("female")}
                icon={<GenderFemale size={20} aria-hidden />}
                title="Nữ"
                badge="Miễn phí"
                desc="Giọng nữ tự nhiên, rõ chữ"
                playingKey={playingKey}
                onPreview={(v, l) => void togglePreview(v, l)}
              />
              <VoiceChoice
                voice="male"
                checked={gender === "male"}
                onSelect={() => setGender("male")}
                icon={<GenderMale size={20} aria-hidden />}
                title="Nam"
                badge="Miễn phí"
                desc="Giọng nam ấm, thân thiện"
                playingKey={playingKey}
                onPreview={(v, l) => void togglePreview(v, l)}
              />
            </div>

            {gender === "adam" && (
              <div className="elevenlabs-box">
                <div className="elevenlabs-box-head">
                  <Space size={8} wrap>
                    <Key size={16} weight="duotone" aria-hidden />
                    <Typography.Text strong style={{ fontSize: 13 }}>
                      ElevenLabs API Key cho giọng Adam (Dominant, Firm)
                    </Typography.Text>
                    {hasElevenLabs ? (
                      <Tag color="green" style={{ margin: 0 }}>
                        Đã kết nối ElevenLabs ({cfg?.elevenlabs_key_masked})
                      </Tag>
                    ) : (
                      <Tag color="warning" style={{ margin: 0 }}>
                        Chưa nhập API Key
                      </Tag>
                    )}
                  </Space>
                </div>
                <div className="elevenlabs-box-row">
                  <Input.Password
                    value={elKey}
                    onChange={(e) => setElKey(e.target.value)}
                    placeholder={
                      hasElevenLabs
                        ? `Đang dùng key (${cfg?.elevenlabs_key_masked}) — nhập key mới nếu muốn đổi`
                        : "Dán ElevenLabs API Key (sk_...) vào đây..."
                    }
                    onPressEnter={() => {
                      if (elKey.trim()) void handleSaveElevenLabs(elKey.trim());
                    }}
                  />
                  <Button
                    type="primary"
                    loading={savingKey}
                    disabled={!elKey.trim()}
                    onClick={() => void handleSaveElevenLabs(elKey.trim())}
                  >
                    Lưu & Kiểm tra Key
                  </Button>
                  {hasElevenLabs && (
                    <Button
                      danger
                      icon={<Trash size={15} aria-hidden />}
                      loading={savingKey}
                      onClick={() => void handleSaveElevenLabs("")}
                    >
                      Xóa Key
                    </Button>
                  )}
                </div>
              </div>
            )}
          </Section>

          <Section id="sec-sub" label="4. Phụ đề">
            <div className="choice-grid" role="radiogroup" aria-labelledby="sec-sub">
              <Choice
                checked={subMode === "hard"}
                onClick={() => setSubMode("hard")}
                icon={<Subtitles size={20} aria-hidden />}
                title="Gắn cứng"
                desc="Luôn hiển thị, phù hợp mạng xã hội"
              />
              <Choice
                checked={subMode === "soft"}
                onClick={() => setSubMode("soft")}
                icon={<ClosedCaptioning size={20} aria-hidden />}
                title="Phụ đề rời"
                desc="Người xem bật/tắt trong trình phát"
              />
              <Choice
                checked={subMode === "none"}
                onClick={() => setSubMode("none")}
                icon={<SpeakerSimpleSlash size={20} aria-hidden />}
                title="Không phụ đề"
                desc="Chỉ lồng tiếng"
              />
            </div>
          </Section>

          <Section id="sec-flow" label="5. Quy trình">
            <div className="choice-grid" role="radiogroup" aria-labelledby="sec-flow">
              <Choice
                checked={review}
                onClick={() => setReview(true)}
                icon={<PencilSimple size={20} aria-hidden />}
                title="Duyệt trước khi xuất"
                desc="Kiểm tra và sửa bản dịch, rồi mới tạo giọng đọc (khuyên dùng)"
              />
              <Choice
                checked={!review}
                onClick={() => setReview(false)}
                icon={<Lightning size={20} aria-hidden />}
                title="Tự động hoàn toàn"
                desc="Xuất video ngay, không cần duyệt"
              />
            </div>
          </Section>

          <Section id="sec-vol" label={`6. Âm lượng tiếng gốc khi có giọng đọc: ${vol}%`}>
            <Slider
              min={0}
              max={50}
              value={vol}
              onChange={(v) => setVolume(v)}
              marks={{ 0: "Tắt", 5: "5%", 50: "50%" }}
              aria-labelledby="sec-vol"
              tooltip={{ formatter: (v) => `${v}%` }}
              style={{ maxWidth: 420 }}
            />
          </Section>

          {error && (
            <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} role="alert" closable onClose={() => setError(null)} />
          )}
          {uploading && (
            <div style={{ marginBottom: 16 }} aria-live="polite">
              <Typography.Text>Đang tải lên… {pct}%</Typography.Text>
              <Progress percent={pct} showInfo={false} />
            </div>
          )}

          <Button
            type="primary"
            size="large"
            icon={<Check size={18} weight="bold" aria-hidden />}
            onClick={submit}
            loading={uploading}
            disabled={!file || !langs.length}
            style={{ minWidth: 200 }}
          >
            Bắt đầu xử lý{langs.length > 1 ? ` (${langs.length} video)` : ""}
          </Button>
        </Card>
      </div>
    </>
  );
}
