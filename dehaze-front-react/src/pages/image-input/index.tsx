import {
  ArrowLeftOutlined,
  DeleteOutlined,
  InboxOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Empty,
  Image as AntImage,
  message,
  Progress,
  Space,
  Tabs,
  Tag,
  Typography,
  Upload,
} from "antd";
import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import Camera from "@/components/Camera";
import ExampleImageSelect from "@/components/ExampleImageSelect";
import exampleImages from "@/pages/presentation/dehaze/exampleImages";
import { FileAPI } from "dehaze-sdk-js";

const { Dragger } = Upload;
const { Title } = Typography;

// 历史记录存储 key
const HISTORY_KEY = "dehaze:image-history";
// 最大文件大小 100MB
const MAX_FILE_SIZE = 100 * 1024 * 1024;
// 历史记录最大保存数量
const MAX_HISTORY_COUNT = 20;

// 图片来源类型
type ImageSource = "upload" | "camera" | "sample" | "history";

// 历史记录结构
interface HistoryRecord {
  id: string;
  imageUrl: string;
  source: Exclude<ImageSource, "history">;
  time: number;
}

// 样例分类
const SAMPLE_CATEGORIES = [
  "全部",
  "城市建筑",
  "自然风景",
  "人像",
  "夜景",
] as const;
type SampleCategory = (typeof SAMPLE_CATEGORIES)[number];

// 样例分类列表（不含"全部"）
const SAMPLE_CATEGORY_LIST: SampleCategory[] = [
  "城市建筑",
  "自然风景",
  "人像",
  "夜景",
];

// 为样例图片分配分类
const categorizedSamples = exampleImages.map((item, index) => ({
  ...item,
  category: SAMPLE_CATEGORY_LIST[index % SAMPLE_CATEGORY_LIST.length],
}));

// 来源标签映射
const SOURCE_LABELS: Record<Exclude<ImageSource, "history">, string> = {
  upload: "上传",
  camera: "拍照",
  sample: "样例",
};

// 读取历史记录
function getHistory(): HistoryRecord[] {
  const data = localStorage.getItem(HISTORY_KEY);
  if (!data) return [];
  try {
    return JSON.parse(data) as HistoryRecord[];
  } catch {
    return [];
  }
}

// 保存历史记录
function saveToHistory(
  imageUrl: string,
  source: Exclude<ImageSource, "history">
) {
  const records = getHistory();
  const newRecord: HistoryRecord = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    imageUrl,
    source,
    time: Date.now(),
  };
  const updated = [newRecord, ...records].slice(0, MAX_HISTORY_COUNT);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
}

// 按时间分组历史记录
function groupHistoryByTime(records: HistoryRecord[]) {
  const now = new Date();
  const todayStart = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate()
  ).getTime();
  const yesterdayStart = todayStart - 24 * 60 * 60 * 1000;
  const sevenDaysAgoStart = todayStart - 7 * 24 * 60 * 60 * 1000;

  const groups: { title: string; records: HistoryRecord[] }[] = [
    { title: "今天", records: [] },
    { title: "昨天", records: [] },
    { title: "最近7天", records: [] },
    { title: "更早", records: [] },
  ];

  for (const record of records) {
    if (record.time >= todayStart) {
      groups[0].records.push(record);
    } else if (record.time >= yesterdayStart) {
      groups[1].records.push(record);
    } else if (record.time >= sevenDaysAgoStart) {
      groups[2].records.push(record);
    } else {
      groups[3].records.push(record);
    }
  }

  return groups.filter((g) => g.records.length > 0);
}

// 格式化时间显示
function formatTime(time: number): string {
  const date = new Date(time);
  const now = new Date();
  const diff = now.getTime() - time;
  if (diff < 60 * 1000) return "刚刚";
  if (diff < 60 * 60 * 1000) return `${Math.floor(diff / (60 * 1000))}分钟前`;
  if (diff < 24 * 60 * 60 * 1000)
    return `${Math.floor(diff / (60 * 60 * 1000))}小时前`;
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const h = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  return `${y}-${m}-${d} ${h}:${min}`;
}

// ============ 上传面板 ============
interface UploadPanelProps {
  onImageSelected: (url: string, source: ImageSource) => void;
}

const UploadPanel: React.FC<UploadPanelProps> = ({ onImageSelected }) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [previewUrl, setPreviewUrl] = useState<string>("");

  // 上传文件，通过 onUploadProgress 回调显示进度
  const handleUpload = (file: File) => {
    setUploading(true);
    setProgress(0);
    FileAPI.upload(file, undefined, (progressEvent) => {
      const percent = progressEvent.total
        ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
        : 0;
      setProgress(percent);
    })
      .then((res) => {
        message.success("上传成功");
        onImageSelected(res.url), "upload";
      })
      .catch((err) => {
        message.error("上传失败: " + (err?.message || "未知错误"));
      })
      .finally(() => {
        setUploading(false);
      });
  };

  // 上传前校验：格式与大小
  const handleBeforeUpload = (file: File) => {
    // 格式校验
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      message.error("不支持该图片格式，请选择 JPG/PNG/WEBP 格式");
      return Upload.LIST_IGNORE;
    }
    // 大小校验
    if (file.size > MAX_FILE_SIZE) {
      message.error("图片大小超过 100MB，请选择较小的图片");
      return Upload.LIST_IGNORE;
    }
    // 生成本地预览
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target?.result as string);
    reader.readAsDataURL(file);
    // 开始上传
    handleUpload(file);
    return false;
  };

  return (
    <div style={{ padding: "24px 0" }}>
      <Dragger
        accept=".jpg,.jpeg,.png,.webp"
        beforeUpload={handleBeforeUpload}
        showUploadList={false}
        multiple={false}
        disabled={uploading}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
        <p className="ant-upload-hint">
          支持 JPG、PNG、WEBP 格式，单文件不超过 100MB
        </p>
      </Dragger>

      {previewUrl && (
        <div style={{ marginTop: 16, textAlign: "center" }}>
          <AntImage
            src={previewUrl}
            alt="预览"
            style={{ maxHeight: 240, objectFit: "contain" }}
          />
        </div>
      )}

      {uploading && (
        <div style={{ marginTop: 16 }}>
          <Progress percent={progress} status="active" />
        </div>
      )}
    </div>
  );
};

// ============ 拍照面板 ============
interface CameraPanelProps {
  onImageSelected: (url: string, source: ImageSource) => void;
}

const CameraPanel: React.FC<CameraPanelProps> = ({ onImageSelected }) => {
  // 拍照保存后上传
  const handleSave = (file: File) => {
    const hide = message.loading("正在上传...", 0);
    FileAPI.upload(file)
      .then((res) => {
        hide();
        message.success("拍照上传成功");
        onImageSelected(res.url), "camera";
      })
      .catch((err) => {
        hide();
        message.error("上传失败: " + (err?.message || "未知错误"));
      });
  };

  return (
    <div style={{ padding: "24px 0", maxWidth: 800, margin: "0 auto" }}>
      <Camera onSave={handleSave} onCancel={() => message.info("已取消拍照")} />
    </div>
  );
};

// ============ 样例画廊面板 ============
interface SamplePanelProps {
  onImageSelected: (url: string, source: ImageSource) => void;
}

const SamplePanel: React.FC<SamplePanelProps> = ({ onImageSelected }) => {
  const [category, setCategory] = useState<SampleCategory>("全部");

  // 按分类过滤样例图片
  const filteredUrls = useMemo(() => {
    if (category === "全部") {
      return categorizedSamples.map((s) => s.haze);
    }
    return categorizedSamples
      .filter((s) => s.category === category)
      .map((s) => s.haze);
  }, [category]);

  return (
    <div style={{ padding: "24px 0" }}>
      <Space style={{ marginBottom: 16 }}>
        {SAMPLE_CATEGORIES.map((cat) => (
          <Button
            key={cat}
            type={category === cat ? "primary" : "default"}
            onClick={() => setCategory(cat)}
          >
            {cat}
          </Button>
        ))}
      </Space>
      <ExampleImageSelect
        urls={filteredUrls}
        onExampleSelect={(url) => onImageSelected(url, "sample")}
      />
    </div>
  );
};

// ============ 历史记录面板 ============
interface HistoryPanelProps {
  onImageSelected: (url: string, source: ImageSource) => void;
}

const HistoryPanel: React.FC<HistoryPanelProps> = ({ onImageSelected }) => {
  const [records, setRecords] = useState<HistoryRecord[]>([]);

  // 加载 localStorage 中的历史记录
  useEffect(() => {
    setRecords(getHistory());
  }, []);

  // 删除单条历史记录
  const handleDelete = (id: string) => {
    const updated = records.filter((r) => r.id !== id);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
    setRecords(updated);
    message.success("已删除该记录");
  };

  // 按时间分组
  const groups = useMemo(() => groupHistoryByTime(records), [records]);

  if (records.length === 0) {
    return (
      <div style={{ padding: "48px 0" }}>
        <Empty description="暂无历史记录" />
      </div>
    );
  }

  return (
    <div style={{ padding: "24px 0" }}>
      {groups.map((group) => (
        <div key={group.title} style={{ marginBottom: 24 }}>
          <Title level={5} style={{ marginBottom: 12 }}>
            {group.title}
          </Title>
          <Space direction="vertical" style={{ width: "100%" }} size={12}>
            {group.records.map((record) => (
              <div
                key={record.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 16,
                  padding: 12,
                  border: "1px solid #f0f0f0",
                  borderRadius: 8,
                }}
              >
                <AntImage
                  src={record.imageUrl}
                  alt="历史图片"
                  width={80}
                  height={80}
                  style={{ objectFit: "cover", borderRadius: 4 }}
                />
                <div style={{ flex: 1 }}>
                  <Tag color="blue">{SOURCE_LABELS[record.source]}</Tag>
                  <span style={{ color: "#999", marginLeft: 8 }}>
                    {formatTime(record.time)}
                  </span>
                </div>
                <Space>
                  <Button
                    type="primary"
                    size="small"
                    icon={<ReloadOutlined />}
                    onClick={() => onImageSelected(record.imageUrl, "history")}
                  >
                    重新处理
                  </Button>
                  <Button
                    danger
                    size="small"
                    icon={<DeleteOutlined />}
                    onClick={() => handleDelete(record.id)}
                  >
                    删除
                  </Button>
                </Space>
              </div>
            ))}
          </Space>
        </div>
      ))}
    </div>
  );
};

// ============ 图像输入页面 ============
const ImageInputPage: React.FC = () => {
  const navigate = useNavigate();

  // 图片选择处理：保存历史记录并跳转到去雾页
  const handleImageSelected = (imageUrl: string, source: ImageSource) => {
    // 历史记录来源不重复保存
    if (source !== "history") {
      saveToHistory(imageUrl, source);
    }
    navigate(`/presentation/dehaze?imageUrl=${encodeURIComponent(imageUrl)}`);
  };

  return (
    <div
      style={{
        padding: 24,
        height: "calc(100vh - var(--navbar-height))",
        overflowY: "auto",
      }}
    >
      <Card>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            marginBottom: 16,
            gap: 8,
          }}
        >
          <Button
            type="text"
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate(-1)}
          />
          <Title level={4} style={{ margin: 0 }}>
            图像输入
          </Title>
        </div>
        <Tabs
          defaultActiveKey="upload"
          items={[
            {
              key: "upload",
              label: "上传",
              children: <UploadPanel onImageSelected={handleImageSelected} />,
            },
            {
              key: "camera",
              label: "拍照",
              children: <CameraPanel onImageSelected={handleImageSelected} />,
            },
            {
              key: "sample",
              label: "样例",
              children: <SamplePanel onImageSelected={handleImageSelected} />,
            },
            {
              key: "history",
              label: "历史",
              children: <HistoryPanel onImageSelected={handleImageSelected} />,
            },
          ]}
        />
      </Card>
    </div>
  );
};

export default ImageInputPage;
