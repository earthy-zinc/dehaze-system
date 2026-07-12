import {
  DatasetAPI,
  DatasetItemAPI,
  ItemFileAPI,
  TaskAPI,
  type Dataset,
  type DatasetItemVO,
  type DatasetStatistics,
  type ImageUrlVO,
} from "dehaze-sdk-js";
import Waterfall from "@/components/Waterfall";
import { useWindowSize } from "@/hooks/useWindowSize";
import {
  BarsOutlined,
  LeftOutlined,
  RightOutlined,
  AppstoreOutlined,
  PictureOutlined,
  PicRightOutlined,
  DownloadOutlined,
  DeleteOutlined,
  UploadOutlined,
  BarChartOutlined,
  DownOutlined,
  EyeOutlined,
} from "@ant-design/icons";
import {
  AutoComplete,
  Button,
  Card,
  Checkbox,
  Col,
  Descriptions,
  Divider,
  Dropdown,
  Empty,
  Form,
  Input,
  Modal,
  Popconfirm,
  Progress,
  Row,
  Segmented,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Tooltip,
  Upload,
  message,
  type TableColumnsType,
  type UploadFile,
  type MenuProps,
} from "antd";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useParams } from "react-router-dom";

// ==================== 类型定义 ====================

/** 展示模式 */
type DisplayMode = "list" | "waterfall" | "horizontal" | "grid";

/** 标注状态过滤：已标注/未标注二分 */
type AnnotationFilter = "annotated" | "unannotated";

/** 用于展示的图片信息 */
interface DisplayImage {
  /** 数据项ID */
  itemId: number;
  /** 图片文件ID */
  fileId: number;
  /** 缩略图URL */
  url: string;
  /** 原图URL */
  originUrl: string;
  /** 描述 */
  alt: string;
  /** 文件名 */
  fileName: string;
  /** 文件大小（字节） */
  sizeBytes: number;
  /** 格式化文件大小 */
  formattedSize: string;
  /** 文件格式 */
  format: string;
  /** 图片宽度 */
  width?: number;
  /** 图片高度 */
  height?: number;
  /** 图片类型：clear-清晰图，hazy-有雾图，trans-透射图，depth-深度图，segment-分割图 */
  type: string;
  /** 雾霾程度，支持多种规范：light/medium/heavy、beta=0.5、A=0.8,beta=0.2 等，可为空 */
  hazeLevel?: string;
  /** 场景类型 */
  sceneType?: string;
}

// ==================== 常量 ====================

/** 标注状态过滤选项（Tab 切换为"已标注/未标注"二分） */
const ANNOTATION_FILTERS: { key: AnnotationFilter; label: string }[] = [
  { key: "annotated", label: "已标注" },
  { key: "unannotated", label: "未标注" },
];

/** 图片类型标签映射（支持 clear/hazy/trans/depth/segment 五种类型，未知类型走兜底） */
const IMAGE_TYPE_LABELS: Record<string, { label: string; color: string }> = {
  clear: { label: "清晰图", color: "green" },
  hazy: { label: "有雾图", color: "orange" },
  trans: { label: "透射图", color: "blue" },
  depth: { label: "深度图", color: "purple" },
  segment: { label: "分割图", color: "cyan" },
};

/** 场景类型选项 */
const SCENE_TYPE_OPTIONS = [
  { label: "户外", value: "outdoor" },
  { label: "室内", value: "indoor" },
  { label: "街道", value: "street" },
  { label: "高速公路", value: "highway" },
  { label: "城市", value: "urban" },
  { label: "森林", value: "forest" },
  { label: "山区", value: "mountain" },
];

/** 雾霾程度预设选项（允许自由输入，支持 beta=X 等多种规范） */
const HAZE_LEVEL_OPTIONS = [
  { label: "轻度", value: "light" },
  { label: "中度", value: "medium" },
  { label: "重度", value: "heavy" },
];

/** 已知雾霾程度枚举的标签与颜色映射，未知值（如 beta=0.5）走兜底 */
const HAZE_LEVEL_MAP: Record<string, { label: string; color: string }> = {
  light: { label: "轻度", color: "green" },
  medium: { label: "中度", color: "orange" },
  heavy: { label: "重度", color: "red" },
};

/** 获取图片类型标签信息（未知类型走兜底） */
const getImageTypeInfo = (type: string) =>
  IMAGE_TYPE_LABELS[type] || { label: type || "未知", color: "default" };

/**
 * 格式化雾霾程度用于展示：
 * - light/medium/heavy → 轻度/中度/重度
 * - beta=X → β=X
 * - A=X,beta=Y → β=Y
 * - 其他 → 原值回显
 * - 空 → 未标注
 */
const formatHazeLevel = (level?: string): {
  label: string;
  color: string;
} | null => {
  if (!level) return null;
  if (HAZE_LEVEL_MAP[level]) return HAZE_LEVEL_MAP[level];
  // beta=X 格式
  const betaMatch = level.match(/beta=([\d.]+)/i);
  if (betaMatch) {
    return { label: `β=${betaMatch[1]}`, color: "blue" };
  }
  // 兜底：原值回显
  return { label: level, color: "default" };
};

/** 图表配色 */
const CHART_COLORS = [
  "#1677ff",
  "#52c41a",
  "#faad14",
  "#ff4d4f",
  "#722ed1",
  "#13c2c2",
  "#eb2f96",
  "#fa8c16",
];

/** 瀑布流断点 */
const BREAKPOINTS = [
  { minWidth: 0, columns: 1 },
  { minWidth: 768, columns: 2 },
  { minWidth: 1024, columns: 3 },
  { minWidth: 1280, columns: 4 },
];

// ==================== 工具函数 ====================

/** 格式化文件大小 */
const formatFileSize = (bytes: number): string => {
  if (!bytes || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${units[i]}`;
};

/** 数据项去重 */
const uniqueArray = (arr: DatasetItemVO[]) => {
  const seen = new Set<number>();
  return arr.filter((item) => {
    if (!seen.has(item.id)) {
      seen.add(item.id);
      return true;
    }
    return false;
  });
};

/** 将数据项下所有图片（含 clear/hazy/trans/depth/segment）展平为展示图片列表 */
const extractAllImagesFromItem = (item: DatasetItemVO): DisplayImage[] => {
  const result: DisplayImage[] = [];
  const pushImage = (img: ImageUrlVO | undefined, itemId: number, sceneType?: string) => {
    if (!img) return;
    result.push({
      itemId,
      fileId: img.id,
      url: img.thumbnailUrl || img.url,
      originUrl: img.originUrl || img.url,
      alt: img.description || img.fileName || "",
      fileName: img.fileName || "",
      sizeBytes: img.sizeBytes || 0,
      formattedSize: img.formattedSize || "",
      format: img.format || "",
      width: img.width,
      height: img.height,
      type: img.type,
      hazeLevel: img.hazeLevel,
      sceneType: img.sceneType || sceneType,
    });
  };
  pushImage(item.clearImage, item.id, item.sceneType);
  if (item.hazyImages) {
    item.hazyImages.forEach((img) => pushImage(img, item.id, item.sceneType));
  }
  return result;
};

/**
 * 将数据项列表转换为展示图片列表，按"已标注/未标注"过滤。
 * - annotated：hazeLevel 非空的图片
 * - unannotated：hazeLevel 为空的图片
 */
const buildDisplayImages = (
  items: DatasetItemVO[],
  filter: AnnotationFilter
): DisplayImage[] => {
  const result: DisplayImage[] = [];
  for (const item of items) {
    const allImages = extractAllImagesFromItem(item);
    for (const img of allImages) {
      const isAnnotated = Boolean(img.hazeLevel);
      if (filter === "annotated" && isAnnotated) {
        result.push(img);
      } else if (filter === "unannotated" && !isAnnotated) {
        result.push(img);
      }
    }
  }
  return result;
};

// ==================== 统计图表组件（echarts 不可用时使用 Ant Design 替代） ====================

/** 饼图替代组件：使用 CSS conic-gradient 绘制环形图 */
const DistributionPie: React.FC<{
  data: Record<string, number>;
  title: string;
}> = ({ data, title }) => {
  const entries = Object.entries(data).filter(([, v]) => v > 0);
  const total = entries.reduce((sum, [, v]) => sum + v, 0);

  if (total === 0) {
    return (
      <Card title={title} size="small">
        <Empty description="暂无数据" />
      </Card>
    );
  }

  // 构建 conic-gradient 各段
  let accumulated = 0;
  const gradientParts = entries.map(([key, value], i) => {
    const percentage = (value / total) * 100;
    const start = accumulated;
    accumulated += percentage;
    const color = CHART_COLORS[i % CHART_COLORS.length];
    return `${color} ${start}% ${accumulated}%`;
  });

  return (
    <Card title={title} size="small">
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <div
          style={{
            width: 120,
            height: 120,
            borderRadius: "50%",
            background: `conic-gradient(${gradientParts.join(", ")})`,
            position: "relative",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              width: 64,
              height: 64,
              borderRadius: "50%",
              backgroundColor: "#fff",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 18,
              fontWeight: "bold",
              color: "#333",
            }}
          >
            {total}
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          {entries.map(([key, value], i) => (
            <div
              key={key}
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 4,
                fontSize: 13,
              }}
            >
              <Space>
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    backgroundColor: CHART_COLORS[i % CHART_COLORS.length],
                    borderRadius: 2,
                  }}
                />
                <span>{key}</span>
              </Space>
              <span style={{ color: "#666" }}>
                {value} ({((value / total) * 100).toFixed(1)}%)
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};

/** 柱状图替代组件：使用 Ant Design Progress 组件 */
const DistributionBar: React.FC<{
  data: Record<string, number>;
  title: string;
}> = ({ data, title }) => {
  const entries = Object.entries(data).filter(([, v]) => v > 0);

  if (entries.length === 0) {
    return (
      <Card title={title} size="small">
        <Empty description="暂无数据" />
      </Card>
    );
  }

  const max = Math.max(...entries.map(([, v]) => v), 1);

  return (
    <Card title={title} size="small">
      {entries.map(([key, value]) => (
        <div key={key} style={{ marginBottom: 12 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 4,
              fontSize: 13,
            }}
          >
            <span>{key}</span>
            <span style={{ color: "#666" }}>{value}</span>
          </div>
          <Progress
            percent={(value / max) * 100}
            showInfo={false}
            strokeColor="#1677ff"
          />
        </div>
      ))}
    </Card>
  );
};

// ==================== 统计分析弹窗 ====================

const StatisticsModal: React.FC<{
  visible: boolean;
  statistics: DatasetStatistics | undefined;
  onClose: () => void;
}> = ({ visible, statistics, onClose }) => {
  return (
    <Modal
      title="统计分析"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={900}
    >
      {!statistics ? (
        <Empty description="暂无统计数据" />
      ) : (
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <DistributionPie
              data={statistics.sceneDistribution}
              title="场景类型分布"
            />
          </Col>
          <Col span={12}>
            <DistributionBar
              data={statistics.hazeDistribution}
              title="雾霾程度分布"
            />
          </Col>
          <Col span={12}>
            <DistributionPie
              data={statistics.formatDistribution}
              title="文件格式分布"
            />
          </Col>
          <Col span={12}>
            <DistributionBar
              data={statistics.resolutionDistribution || {}}
              title="分辨率分布"
            />
          </Col>
        </Row>
      )}
    </Modal>
  );
};

// ==================== 配对上传弹窗 ====================

const PairUploadDialog: React.FC<{
  visible: boolean;
  datasetId: number;
  onClose: () => void;
  onSuccess: () => void;
}> = ({ visible, datasetId, onClose, onSuccess }) => {
  const [clearFileList, setClearFileList] = useState<UploadFile[]>([]);
  const [hazyFileList, setHazyFileList] = useState<UploadFile[]>([]);
  const [hazeLevel, setHazeLevel] = useState<string>("");
  const [sceneType, setSceneType] = useState<string>();
  const [name, setName] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);

  // 重置状态
  const resetState = () => {
    setClearFileList([]);
    setHazyFileList([]);
    setHazeLevel("");
    setSceneType(undefined);
    setName("");
  };

  const handleCancel = () => {
    resetState();
    onClose();
  };

  const handleSubmit = async () => {
    // 清晰图和有雾图均为可选（适配不同数据集规范），但至少上传一张图片
    if (clearFileList.length === 0 && hazyFileList.length === 0) {
      message.warning("请至少上传一张图片（清晰图或有雾图）");
      return;
    }

    const clearFile = clearFileList[0]?.originFileObj as File | undefined;
    const hazyFiles = hazyFileList
      .map((f) => f.originFileObj as File)
      .filter(Boolean);

    const formData = new FormData();
    formData.append("datasetId", String(datasetId));
    if (name) formData.append("name", name);
    if (clearFile) {
      formData.append("clearImage", clearFile);
    }
    if (hazyFiles.length > 0) {
      hazyFiles.forEach((file) => formData.append("hazyImages", file));
      // 每张有雾图对应一个雾霾程度（支持空字符串表示未标注）
      hazyFiles.forEach(() => formData.append("hazeLevels", hazeLevel));
    }
    if (sceneType) formData.append("sceneType", sceneType);

    setSubmitting(true);
    try {
      await DatasetItemAPI.uploadImagePair(formData);
      message.success("配对上传成功");
      resetState();
      onClose();
      onSuccess();
    } catch (error: any) {
      message.error(error?.message || "上传失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title="配对上传"
      open={visible}
      onCancel={handleCancel}
      onOk={handleSubmit}
      confirmLoading={submitting}
      okText="上传"
      cancelText="取消"
      width={680}
      destroyOnClose
    >
      <Form layout="vertical">
        <Form.Item label="数据项名称">
          <Input
            placeholder="请输入数据项名称（可选）"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </Form.Item>
        <Form.Item label="清晰图（限1张，可选）">
          <Upload
            listType="picture-card"
            maxCount={1}
            fileList={clearFileList}
            beforeUpload={() => false}
            onChange={({ fileList }) => setClearFileList(fileList)}
            onRemove={() => {
              setClearFileList([]);
              return true;
            }}
          >
            {clearFileList.length < 1 && (
              <div>
                <UploadOutlined />
                <div style={{ marginTop: 4 }}>上传清晰图</div>
              </div>
            )}
          </Upload>
        </Form.Item>
        <Form.Item label="有雾图（可多张，可选）">
          <Upload
            listType="picture-card"
            multiple
            fileList={hazyFileList}
            beforeUpload={() => false}
            onChange={({ fileList }) => setHazyFileList(fileList)}
          >
            <div>
              <UploadOutlined />
              <div style={{ marginTop: 4 }}>上传有雾图</div>
            </div>
          </Upload>
        </Form.Item>
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item label="雾霾程度（可选，支持 beta=0.5 等格式）">
              <AutoComplete
                value={hazeLevel}
                onChange={setHazeLevel}
                options={HAZE_LEVEL_OPTIONS}
                placeholder="如 light/medium/heavy/beta=0.5"
                filterOption={false}
              />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="场景类型">
              <Select
                value={sceneType}
                onChange={setSceneType}
                options={SCENE_TYPE_OPTIONS}
                allowClear
                placeholder="请选择场景类型"
              />
            </Form.Item>
          </Col>
        </Row>
      </Form>
    </Modal>
  );
};

// ==================== 批量上传弹窗 ====================

const BatchUploadDialog: React.FC<{
  visible: boolean;
  datasetId: number;
  onClose: () => void;
  onSuccess: () => void;
}> = ({ visible, datasetId, onClose, onSuccess }) => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [sceneType, setSceneType] = useState<string>();
  const [submitting, setSubmitting] = useState(false);

  const resetState = () => {
    setFileList([]);
    setSceneType(undefined);
  };

  const handleCancel = () => {
    resetState();
    onClose();
  };

  const handleSubmit = async () => {
    if (fileList.length === 0) {
      message.warning("请选择至少一个文件");
      return;
    }

    const files = fileList.map((f) => f.originFileObj as File).filter(Boolean);

    if (files.length === 0) {
      message.warning("文件无效");
      return;
    }

    const formData = new FormData();
    formData.append("datasetId", String(datasetId));
    files.forEach((file) => formData.append("files", file));
    if (sceneType) formData.append("sceneType", sceneType);

    setSubmitting(true);
    try {
      const result = await DatasetItemAPI.batchUpload(formData);
      message.success(
        `批量上传完成：成功 ${result.succeeded} 个，失败 ${result.failed} 个`
      );
      resetState();
      onClose();
      onSuccess();
    } catch (error: any) {
      message.error(error?.message || "上传失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title="批量上传"
      open={visible}
      onCancel={handleCancel}
      onOk={handleSubmit}
      confirmLoading={submitting}
      okText="上传"
      cancelText="取消"
      width={680}
      destroyOnClose
    >
      <Form layout="vertical">
        <Form.Item label="场景类型">
          <Select
            value={sceneType}
            onChange={setSceneType}
            options={SCENE_TYPE_OPTIONS}
            allowClear
            placeholder="请选择场景类型（可选）"
          />
        </Form.Item>
        <Form.Item label="选择文件" required>
          <Upload
            listType="picture-card"
            multiple
            fileList={fileList}
            beforeUpload={() => false}
            onChange={({ fileList }) => setFileList(fileList)}
          >
            <div>
              <UploadOutlined />
              <div style={{ marginTop: 4 }}>选择文件</div>
            </div>
          </Upload>
        </Form.Item>
      </Form>
    </Modal>
  );
};

// ==================== 图片详情查看弹窗 ====================

const ImageViewerModal: React.FC<{
  visible: boolean;
  image: DisplayImage | null;
  index: number;
  total: number;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
  onDownload: (image: DisplayImage) => void;
  onDelete: (image: DisplayImage) => void;
}> = ({
  visible,
  image,
  index,
  total,
  onClose,
  onPrev,
  onNext,
  onDownload,
  onDelete,
}) => {
  if (!image) return null;

  const hazeInfo = formatHazeLevel(image.hazeLevel);
  const typeInfo = getImageTypeInfo(image.type);

  return (
    <Modal
      title="图片详情"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={960}
      destroyOnClose
    >
      <Row gutter={16}>
        {/* 左侧：图片预览 + 上一张/下一张 */}
        <Col span={16}>
          <div
            style={{
              position: "relative",
              textAlign: "center",
              minHeight: 400,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "#fafafa",
              borderRadius: 8,
            }}
          >
            <img
              src={image.originUrl || image.url}
              alt={image.alt}
              style={{
                maxWidth: "100%",
                maxHeight: "70vh",
                objectFit: "contain",
              }}
            />
            <Button
              shape="circle"
              icon={<LeftOutlined />}
              onClick={onPrev}
              disabled={index === 0}
              style={{ position: "absolute", left: 8, top: "50%" }}
            />
            <Button
              shape="circle"
              icon={<RightOutlined />}
              onClick={onNext}
              disabled={index === total - 1}
              style={{ position: "absolute", right: 8, top: "50%" }}
            />
          </div>
        </Col>
        {/* 右侧：信息面板 */}
        <Col span={8}>
          <Descriptions column={1} size="small" bordered>
            <Descriptions.Item label="文件名">
              <Tooltip title={image.fileName}>
                <span
                  style={{
                    display: "inline-block",
                    maxWidth: 200,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    verticalAlign: "bottom",
                  }}
                >
                  {image.fileName || "-"}
                </span>
              </Tooltip>
            </Descriptions.Item>
            <Descriptions.Item label="类型">
              <Tag color={typeInfo.color}>{typeInfo.label}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="大小">
              {image.formattedSize || formatFileSize(image.sizeBytes)}
            </Descriptions.Item>
            <Descriptions.Item label="格式">
              {image.format ? <Tag>{image.format.toUpperCase()}</Tag> : "-"}
            </Descriptions.Item>
            <Descriptions.Item label="分辨率">
              {image.width && image.height
                ? `${image.width} × ${image.height}`
                : "-"}
            </Descriptions.Item>
            <Descriptions.Item label="雾霾程度">
              {hazeInfo ? (
                <Tag color={hazeInfo.color}>{hazeInfo.label}</Tag>
              ) : (
                "-"
              )}
            </Descriptions.Item>
            <Descriptions.Item label="场景类型">
              {image.sceneType || "-"}
            </Descriptions.Item>
          </Descriptions>

          <Space style={{ marginTop: 16, width: "100%" }}>
            <Button
              icon={<DownloadOutlined />}
              onClick={() => onDownload(image)}
            >
              下载
            </Button>
            <Popconfirm
              title="确认删除此图片？"
              onConfirm={() => onDelete(image)}
              okText="确定"
              cancelText="取消"
              okType="danger"
            >
              <Button danger icon={<DeleteOutlined />}>
                删除
              </Button>
            </Popconfirm>
          </Space>

          <div
            style={{
              marginTop: 16,
              textAlign: "center",
              color: "#999",
              fontSize: 13,
            }}
          >
            第 {index + 1} / {total} 张
          </div>
        </Col>
      </Row>
    </Modal>
  );
};

// ==================== 主组件 ====================

export default function DatasetDetail() {
  const { id } = useParams<{ id: string }>();
  const datasetId = Number(id) || 0;
  const [queryParams, setQueryParams] = useState({
    pageNum: 1,
    pageSize: 10,
    keywords: "",
  });
  const [totalPages, setTotalPages] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(false);

  const [datasetInfo, setDatasetInfo] = useState<Dataset | null>(null);
  const [imageData, setImageData] = useState<DatasetItemVO[]>([]);
  const [annotationFilter, setAnnotationFilter] = useState<AnnotationFilter>("annotated");

  // 展示模式
  const [mode, setMode] = useState<DisplayMode>("waterfall");
  // 选择模式（用于瀑布流点击选择）
  const [selectMode, setSelectMode] = useState(false);
  // 选中的文件ID集合
  const [selectedFileIds, setSelectedFileIds] = useState<Set<number>>(
    new Set()
  );

  // 弹窗状态
  const [statsVisible, setStatsVisible] = useState(false);
  const [pairUploadVisible, setPairUploadVisible] = useState(false);
  const [batchUploadVisible, setBatchUploadVisible] = useState(false);
  const [viewerVisible, setViewerVisible] = useState(false);
  const [viewerIndex, setViewerIndex] = useState(0);

  const { width } = useWindowSize();
  const loadingBarRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  // 计算瀑布流图片宽度
  const itemWidth = useMemo(() => {
    return BREAKPOINTS.reduce(
      (acc, breakpoint) =>
        width >= breakpoint.minWidth
          ? Math.floor((width - 60) / breakpoint.columns)
          : acc,
      400
    );
  }, [width]);

  // 获取数据集信息
  useEffect(() => {
    DatasetAPI.getDatasetInfoById(datasetId)
      .then((info) => {
        setDatasetInfo(info);
      })
      .catch((error) => {
        message.error(error?.message || "获取数据集信息失败");
      });
  }, [datasetId]);

  // 获取图片数据
  useEffect(() => {
    setLoading(true);
    DatasetItemAPI.getList({
      datasetId,
      pageNum: queryParams.pageNum,
      pageSize: queryParams.pageSize,
      keyword: queryParams.keywords || undefined,
    })
      .then((data) => {
        if (queryParams.pageNum === 1) {
          setImageData(data.list);
          setTotalPages(Math.ceil(data.total / queryParams.pageSize));
        } else {
          setImageData((prev) => uniqueArray([...prev, ...data.list]));
        }
      })
      .catch((error) => {
        message.error(error?.message || "获取图片列表失败");
      })
      .finally(() => {
        setLoading(false);
      });
  }, [datasetId, queryParams]);

  // 构建展示图片列表（按"已标注/未标注"过滤）
  const displayImages = useMemo(
    () => buildDisplayImages(imageData, annotationFilter),
    [imageData, annotationFilter]
  );

  // 瀑布流组件所需的数据格式
  const waterfallList = useMemo(
    () =>
      displayImages.map((img) => ({
        id: img.fileId,
        src: img.url,
      })),
    [displayImages]
  );

  // 统计信息
  const statistics = datasetInfo?.statistics;

  // 无限滚动观察器
  useEffect(() => {
    if (observerRef.current) observerRef.current.disconnect();
    observerRef.current = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && queryParams.pageNum < totalPages) {
          const nextPage = queryParams.pageNum + 1;
          setQueryParams((prev) => ({ ...prev, pageNum: nextPage }));
        }
      });
    });

    if (loadingBarRef.current) {
      const loadingBarEl = loadingBarRef.current;
      loadingBarEl.style.transform = "translate3d(0, 3000px, 0)";
      observerRef.current.observe(loadingBarEl);
      setTimeout(() => (loadingBarEl.style.transform = "none"), 2000);
    }

    return () => observerRef.current?.disconnect();
  }, [queryParams.pageNum, totalPages]);

  // ==================== 事件处理 ====================

  /** 刷新图片列表 */
  const refreshImages = useCallback(() => {
    setQueryParams((prev) => ({ ...prev, pageNum: 1 }));
  }, []);

  /** 搜索 */
  const handleSearch = () => {
    setQueryParams((prev) => ({ ...prev, pageNum: 1 }));
  };

  /** 重置查询 */
  const resetQuery = () => {
    setQueryParams({ pageNum: 1, pageSize: 10, keywords: "" });
  };

  /** 切换标注状态过滤 */
  const handleAnnotationFilterChange = (filter: AnnotationFilter) => {
    setAnnotationFilter(filter);
    setSelectedFileIds(new Set());
  };

  /** 切换选择 */
  const toggleSelect = (fileId: number, checked: boolean) => {
    setSelectedFileIds((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(fileId);
      } else {
        next.delete(fileId);
      }
      return next;
    });
  };

  /** 全选/取消全选 */
  const handleSelectAll = () => {
    if (selectedFileIds.size === displayImages.length) {
      setSelectedFileIds(new Set());
    } else {
      setSelectedFileIds(new Set(displayImages.map((img) => img.fileId)));
    }
  };

  /** 查看图片大图 */
  const handleViewImage = (image: DisplayImage) => {
    const idx = displayImages.findIndex((img) => img.fileId === image.fileId);
    setViewerIndex(idx >= 0 ? idx : 0);
    setViewerVisible(true);
  };

  /** 瀑布流点击处理 */
  const handleWaterfallClick = (fileId: number) => {
    if (selectMode) {
      toggleSelect(fileId, !selectedFileIds.has(fileId));
    } else {
      const img = displayImages.find((i) => i.fileId === fileId);
      if (img) handleViewImage(img);
    }
  };

  /** 上一张 */
  const handlePrev = () => {
    setViewerIndex((i) => Math.max(0, i - 1));
  };

  /** 下一张 */
  const handleNext = () => {
    setViewerIndex((i) => Math.min(displayImages.length - 1, i + 1));
  };

  /** 下载单张图片 */
  const handleDownloadOne = async (image: DisplayImage) => {
    try {
      const task = await TaskAPI.create({
        type: "item_download",
        targetId: image.itemId,
      });
      if (task.downloadUrl) {
        window.open(task.downloadUrl);
      } else {
        message.success(`下载任务已创建，任务ID: ${task.taskId}`);
      }
    } catch (error: any) {
      message.error(error?.message || "下载失败");
    }
  };

  /** 删除单张图片 */
  const handleDeleteOne = async (image: DisplayImage) => {
    try {
      await ItemFileAPI.deleteById(image.fileId);
      message.success("删除成功");
      setViewerVisible(false);
      refreshImages();
    } catch (error: any) {
      message.error(error?.message || "删除失败");
    }
  };

  /** 批量下载 */
  const handleBatchDownload = async () => {
    const selectedImages = displayImages.filter((img) =>
      selectedFileIds.has(img.fileId)
    );
    if (selectedImages.length === 0) {
      message.warning("请先选择要下载的图片");
      return;
    }
    // 统一任务接口 batch_download 的 targetIds 为数据项ID列表
    const itemIds = Array.from(
      new Set(selectedImages.map((img) => img.itemId))
    );
    try {
      const task = await TaskAPI.create({
        type: "batch_download",
        targetIds: itemIds,
      });
      if (task.downloadUrl) {
        window.open(task.downloadUrl);
      } else {
        message.success(`下载任务已创建，任务ID: ${task.taskId}`);
      }
      setSelectedFileIds(new Set());
    } catch (error: any) {
      message.error(error?.message || "下载失败");
    }
  };

  /** 批量删除 */
  const handleBatchDelete = async () => {
    const ids = Array.from(selectedFileIds);
    if (ids.length === 0) {
      message.warning("请先选择要删除的图片");
      return;
    }
    try {
      const result = await ItemFileAPI.batchDelete({ ids });
      message.success(`成功删除 ${result.successCount} 张图片`);
      setSelectedFileIds(new Set());
      refreshImages();
    } catch (error: any) {
      message.error(error?.message || "删除失败");
    }
  };

  /** 上传下拉菜单 */
  const uploadMenuItems: MenuProps["items"] = [
    { key: "pair", label: "配对上传", icon: <UploadOutlined /> },
    { key: "batch", label: "批量上传", icon: <UploadOutlined /> },
  ];

  const handleUploadMenuClick: MenuProps["onClick"] = ({ key }) => {
    if (key === "pair") setPairUploadVisible(true);
    if (key === "batch") setBatchUploadVisible(true);
  };

  // ==================== 表格列定义 ====================

  const columns: TableColumnsType<DisplayImage> = [
    {
      title: "缩略图",
      dataIndex: "url",
      key: "url",
      width: 80,
      render: (url: string, record: DisplayImage) => (
        <img
          src={url}
          alt={record.alt}
          style={{
            width: 50,
            height: 50,
            objectFit: "cover",
            borderRadius: 4,
            cursor: "pointer",
          }}
          onClick={() => handleViewImage(record)}
        />
      ),
    },
    {
      title: "文件名",
      dataIndex: "fileName",
      key: "fileName",
      render: (text: string) => text || "-",
    },
    {
      title: "类型",
      dataIndex: "type",
      key: "type",
      width: 80,
      render: (type: string) => {
        const info = getImageTypeInfo(type);
        return <Tag color={info.color}>{info.label}</Tag>;
      },
    },
    {
      title: "雾霾程度",
      dataIndex: "hazeLevel",
      key: "hazeLevel",
      width: 100,
      render: (level: string) => {
        const info = formatHazeLevel(level);
        return info ? <Tag color={info.color}>{info.label}</Tag> : "-";
      },
    },
    {
      title: "大小",
      dataIndex: "formattedSize",
      key: "formattedSize",
      width: 100,
      render: (text: string, record: DisplayImage) =>
        text || formatFileSize(record.sizeBytes),
    },
    {
      title: "格式",
      dataIndex: "format",
      key: "format",
      width: 80,
      render: (text: string) => (text ? <Tag>{text.toUpperCase()}</Tag> : "-"),
    },
    {
      title: "分辨率",
      key: "resolution",
      width: 120,
      render: (_: unknown, record: DisplayImage) =>
        record.width && record.height
          ? `${record.width} × ${record.height}`
          : "-",
    },
    {
      title: "操作",
      key: "action",
      width: 160,
      align: "center",
      render: (_: unknown, record: DisplayImage) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewImage(record)}
          >
            查看
          </Button>
          <Button
            type="link"
            size="small"
            icon={<DownloadOutlined />}
            onClick={() => handleDownloadOne(record)}
          >
            下载
          </Button>
        </Space>
      ),
    },
  ];

  // ==================== 渲染 ====================

  return (
    <div className="app-container">
      <Card>
        {/* 头部信息区 */}
        <h1 className="mt-2 mb-3" style={{ textAlign: "center" }}>
          {datasetInfo?.name} {datasetInfo?.type} 数据集
        </h1>
        <p className="mr-3 ml-3 mb-6" style={{ textIndent: "2em" }}>
          {datasetInfo?.description}
        </p>

        {/* 统计摘要 */}
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Card size="small">
              <Statistic title="总图片数" value={statistics?.fileCount ?? 0} />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="已标注"
                value={statistics?.annotatedCount ?? 0}
                valueStyle={{ color: "#52c41a" }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="未标注"
                value={statistics?.unannotatedCount ?? 0}
                valueStyle={{ color: "#fa8c16" }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="文件大小"
                value={formatFileSize(statistics?.totalSize ?? 0)}
              />
            </Card>
          </Col>
        </Row>

        {/* 工具栏第一行：图片类型 + 搜索 + 统计分析 */}
        <div
          className="mb-3"
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 8,
          }}
        >
          <Button.Group>
            {ANNOTATION_FILTERS.map((t) => (
              <Button
                key={t.key}
                type={annotationFilter === t.key ? "primary" : "default"}
                onClick={() => handleAnnotationFilterChange(t.key)}
              >
                {t.label}
              </Button>
            ))}
          </Button.Group>
          <Space>
            <Form layout="inline" component="div">
              <Form.Item label="关键字">
                <Input
                  value={queryParams.keywords}
                  onChange={(e) =>
                    setQueryParams((prev) => ({
                      ...prev,
                      keywords: e.target.value,
                    }))
                  }
                  placeholder="图片名称"
                  onPressEnter={handleSearch}
                  style={{ width: 160 }}
                />
              </Form.Item>
              <Form.Item>
                <Button type="primary" onClick={handleSearch}>
                  搜索
                </Button>
                <Button onClick={resetQuery}>重置</Button>
              </Form.Item>
            </Form>
            <Button
              icon={<BarChartOutlined />}
              onClick={() => setStatsVisible(true)}
            >
              统计分析
            </Button>
          </Space>
        </div>

        {/* 工具栏第二行：展示模式 + 上传/下载/删除 + 选择模式 */}
        <div
          className="mb-3"
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 8,
          }}
        >
          <Space>
            <Segmented
              value={mode}
              onChange={(value) => setMode(value as DisplayMode)}
              options={[
                { label: "列表", value: "list", icon: <BarsOutlined /> },
                {
                  label: "纵向瀑布",
                  value: "waterfall",
                  icon: <PictureOutlined />,
                },
                {
                  label: "横向瀑布",
                  value: "horizontal",
                  icon: <PicRightOutlined />,
                },
                { label: "网格", value: "grid", icon: <AppstoreOutlined /> },
              ]}
            />
            {mode === "waterfall" && (
              <Button
                type={selectMode ? "primary" : "default"}
                onClick={() => setSelectMode(!selectMode)}
              >
                {selectMode ? "退出选择" : "选择模式"}
              </Button>
            )}
            {selectedFileIds.size > 0 && (
              <>
                <Tag color="blue">已选择 {selectedFileIds.size} 项</Tag>
                <Button
                  size="small"
                  onClick={() => setSelectedFileIds(new Set())}
                >
                  清除选择
                </Button>
              </>
            )}
          </Space>
          <Space>
            {mode !== "list" && mode !== "waterfall" && (
              <Button onClick={handleSelectAll}>
                {selectedFileIds.size === displayImages.length &&
                displayImages.length > 0
                  ? "取消全选"
                  : "全选"}
              </Button>
            )}
            <Dropdown
              menu={{ items: uploadMenuItems, onClick: handleUploadMenuClick }}
            >
              <Button type="primary" icon={<UploadOutlined />}>
                上传 <DownOutlined />
              </Button>
            </Dropdown>
            <Button
              icon={<DownloadOutlined />}
              disabled={selectedFileIds.size === 0}
              onClick={handleBatchDownload}
            >
              下载
            </Button>
            <Popconfirm
              title={`确认删除选中的 ${selectedFileIds.size} 张图片吗？`}
              onConfirm={handleBatchDelete}
              okText="确定"
              cancelText="取消"
              okType="danger"
              disabled={selectedFileIds.size === 0}
            >
              <Button
                danger
                icon={<DeleteOutlined />}
                disabled={selectedFileIds.size === 0}
              >
                删除
              </Button>
            </Popconfirm>
          </Space>
        </div>

        {/* 图片展示区域 */}
        <Spin spinning={loading && queryParams.pageNum === 1}>
          {displayImages.length === 0 && !loading ? (
            <Empty description="暂无图片数据" />
          ) : (
            <>
              {/* 列表模式 */}
              {mode === "list" && (
                <Table
                  columns={columns}
                  dataSource={displayImages}
                  rowKey="fileId"
                  loading={loading && queryParams.pageNum === 1}
                  rowSelection={{
                    selectedRowKeys: Array.from(selectedFileIds),
                    onChange: (keys) =>
                      setSelectedFileIds(new Set(keys as number[])),
                  }}
                  pagination={false}
                  scroll={{ y: 500 }}
                  size="small"
                />
              )}

              {/* 纵向瀑布流模式 */}
              {mode === "waterfall" && (
                <>
                  {selectMode && (
                    <div style={{ marginBottom: 8 }}>
                      <Button onClick={handleSelectAll}>
                        {selectedFileIds.size === displayImages.length &&
                        displayImages.length > 0
                          ? "取消全选"
                          : "全选"}
                      </Button>
                    </div>
                  )}
                  <Waterfall
                    list={waterfallList}
                    width={itemWidth}
                    onClickItem={handleWaterfallClick}
                  />
                </>
              )}

              {/* 横向瀑布流模式 */}
              {mode === "horizontal" && (
                <div
                  style={{
                    display: "flex",
                    gap: 12,
                    overflowX: "auto",
                    padding: "8px 0",
                    minHeight: 200,
                  }}
                >
                  {displayImages.map((img) => (
                    <div
                      key={img.fileId}
                      style={{
                        flexShrink: 0,
                        width: 220,
                        position: "relative",
                        border: selectedFileIds.has(img.fileId)
                          ? "2px solid #1677ff"
                          : "2px solid transparent",
                        borderRadius: 8,
                        overflow: "hidden",
                      }}
                    >
                      <img
                        src={img.url}
                        alt={img.alt}
                        style={{
                          width: "100%",
                          height: 200,
                          objectFit: "cover",
                          cursor: "pointer",
                          display: "block",
                        }}
                        onClick={() => handleViewImage(img)}
                      />
                      <Checkbox
                        checked={selectedFileIds.has(img.fileId)}
                        onChange={(e) =>
                          toggleSelect(img.fileId, e.target.checked)
                        }
                        style={{
                          position: "absolute",
                          top: 6,
                          left: 6,
                          backgroundColor: "rgba(255,255,255,0.8)",
                          borderRadius: 4,
                          padding: "0 4px",
                        }}
                      />
                      <div
                        style={{
                          padding: "4px 8px",
                          fontSize: 12,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                          backgroundColor: "#fff",
                        }}
                      >
                        {img.fileName || "未命名"}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* 网格模式 */}
              {mode === "grid" && (
                <Row gutter={[16, 16]}>
                  {displayImages.map((img) => (
                    <Col key={img.fileId} xs={12} sm={8} md={6} lg={4} xl={4}>
                      <Card
                        size="small"
                        hoverable
                        bodyStyle={{ padding: 0 }}
                        style={{
                          border: selectedFileIds.has(img.fileId)
                            ? "2px solid #1677ff"
                            : "1px solid #f0f0f0",
                        }}
                      >
                        <div style={{ position: "relative" }}>
                          <img
                            src={img.url}
                            alt={img.alt}
                            style={{
                              width: "100%",
                              height: 150,
                              objectFit: "cover",
                              cursor: "pointer",
                              display: "block",
                            }}
                            onClick={() => handleViewImage(img)}
                          />
                          <Checkbox
                            checked={selectedFileIds.has(img.fileId)}
                            onChange={(e) =>
                              toggleSelect(img.fileId, e.target.checked)
                            }
                            style={{
                              position: "absolute",
                              top: 6,
                              left: 6,
                              backgroundColor: "rgba(255,255,255,0.8)",
                              borderRadius: 4,
                              padding: "0 4px",
                            }}
                          />
                        </div>
                        <div
                          style={{
                            padding: "4px 8px",
                            fontSize: 12,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                        >
                          <Tooltip title={img.fileName}>
                            {img.fileName || "未命名"}
                          </Tooltip>
                        </div>
                      </Card>
                    </Col>
                  ))}
                </Row>
              )}
            </>
          )}
        </Spin>

        {/* 无限滚动加载触发器 */}
        <div ref={loadingBarRef}>
          {queryParams.pageNum < totalPages && (
            <Divider>正在加载，请稍后</Divider>
          )}
        </div>
      </Card>

      {/* 统计分析弹窗 */}
      <StatisticsModal
        visible={statsVisible}
        statistics={statistics}
        onClose={() => setStatsVisible(false)}
      />

      {/* 配对上传弹窗 */}
      <PairUploadDialog
        visible={pairUploadVisible}
        datasetId={datasetId}
        onClose={() => setPairUploadVisible(false)}
        onSuccess={refreshImages}
      />

      {/* 批量上传弹窗 */}
      <BatchUploadDialog
        visible={batchUploadVisible}
        datasetId={datasetId}
        onClose={() => setBatchUploadVisible(false)}
        onSuccess={refreshImages}
      />

      {/* 图片详情查看弹窗 */}
      <ImageViewerModal
        visible={viewerVisible}
        image={displayImages[viewerIndex] || null}
        index={viewerIndex}
        total={displayImages.length}
        onClose={() => setViewerVisible(false)}
        onPrev={handlePrev}
        onNext={handleNext}
        onDownload={handleDownloadOne}
        onDelete={handleDeleteOne}
      />
    </div>
  );
}
