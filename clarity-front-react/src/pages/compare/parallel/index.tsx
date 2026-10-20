import { RootState } from "@/store";
import {
  setBrightness,
  setContrast,
  setMagnifierShape,
  setMagnifierSize,
  setMagnifierZoomLevel,
  setSaturate,
  toggleMagnifierShow,
} from "@/store/modules/imageShowSlice";
import {
  CloseOutlined,
  DownloadOutlined,
  InboxOutlined,
  PlusOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Form,
  Modal,
  Radio,
  Slider,
  Switch,
  Upload,
  message,
} from "antd";
import React, { useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import styles from "./index.module.scss";

/** 单张对比图片项 */
interface ParallelImage {
  id: number;
  url: string;
  name: string;
}

const Parallel: React.FC = () => {
  const dispatch = useDispatch();

  // 从 Redux 读取滤镜和放大镜配置
  const brightness = useSelector(
    (state: RootState) => state.imageShow.brightness
  );
  const contrast = useSelector((state: RootState) => state.imageShow.contrast);
  const saturate = useSelector((state: RootState) => state.imageShow.saturate);
  const magnifierEnabled = useSelector(
    (state: RootState) => state.imageShow.magnifier.enabled
  );
  const magnifierShape = useSelector(
    (state: RootState) => state.imageShow.magnifier.shape
  );
  const magnifierZoomLevel = useSelector(
    (state: RootState) => state.imageShow.magnifier.zoomLevel
  );
  const magnifierWidth = useSelector(
    (state: RootState) => state.imageShow.magnifier.width
  );
  const magnifierHeight = useSelector(
    (state: RootState) => state.imageShow.magnifier.height
  );

  // 图片列表
  const [images, setImages] = useState<ParallelImage[]>([]);
  const nextIdRef = useRef(0);

  // 放大镜状态：当前悬停的图片索引及鼠标在图片内的相对坐标
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [lensPos, setLensPos] = useState({ x: 0, y: 0 });
  // 缓存当前悬停图片的显示尺寸，用于放大镜背景计算
  const [hoveredRect, setHoveredRect] = useState<{
    width: number;
    height: number;
  } | null>(null);

  /** 处理图片上传，将File转为URL加入列表 */
  const handleAddImages = (files: File[]) => {
    const newImages: ParallelImage[] = files.map((file) => ({
      id: nextIdRef.current++,
      url: URL.createObjectURL(file),
      name: file.name,
    }));
    setImages((prev) => [...prev, ...newImages]);
  };

  /** 删除指定图片 */
  const handleRemoveImage = (id: number) => {
    setImages((prev) => {
      const target = prev.find((item) => item.id === id);
      if (target) URL.revokeObjectURL(target.url);
      return prev.filter((item) => item.id !== id);
    });
  };

  // 导出报告相关状态
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [reportFormat, setReportFormat] = useState<"pdf" | "image">("pdf");
  const [includeMetrics, setIncludeMetrics] = useState(true);
  const [exporting, setExporting] = useState(false);

  /** 清空所有图片并重置滤镜 */
  const handleReset = () => {
    images.forEach((item) => URL.revokeObjectURL(item.url));
    setImages([]);
    setHoveredIndex(null);
    dispatch(setBrightness(100));
    dispatch(setContrast(100));
    dispatch(setSaturate(100));
  };

  /** 鼠标在图片上移动时更新放大镜位置 */
  const handleMouseMove = (
    e: React.MouseEvent,
    index: number,
    img: ParallelImage
  ) => {
    if (!magnifierEnabled) return;
    const target = e.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setHoveredRect({ width: rect.width, height: rect.height });
    setHoveredIndex(index);
    setLensPos({ x, y });
  };

  /** 鼠标离开图片时隐藏放大镜 */
  const handleMouseLeave = () => {
    setHoveredIndex(null);
    setHoveredRect(null);
  };

  /** 计算放大镜透镜样式 */
  const getLensStyle = (img: ParallelImage): React.CSSProperties => {
    if (!hoveredRect) return { display: "none" };
    // 背景图按放大倍数缩放
    const bgWidth = hoveredRect.width * magnifierZoomLevel;
    const bgHeight = hoveredRect.height * magnifierZoomLevel;
    // 背景位置：使透镜中心对准鼠标所在点
    const bgX = lensPos.x * magnifierZoomLevel - magnifierWidth / 2;
    const bgY = lensPos.y * magnifierZoomLevel - magnifierHeight / 2;
    return {
      left: lensPos.x - magnifierWidth / 2,
      top: lensPos.y - magnifierHeight / 2,
      width: magnifierWidth,
      height: magnifierHeight,
      borderRadius: magnifierShape === "circle" ? "50%" : "0",
      backgroundImage: `url(${img.url})`,
      backgroundSize: `${bgWidth}px ${bgHeight}px`,
      backgroundPosition: `-${bgX}px -${bgY}px`,
    };
  };

  // 打开导出报告弹窗
  const handleOpenReportModal = () => {
    setReportModalOpen(true);
  };

  // 关闭导出报告弹窗
  const handleCloseReportModal = () => {
    setReportModalOpen(false);
    setExporting(false);
  };

  // 生成并下载对比报告（parallel 视图无预测日志，需先切换到 overlap 视图完成预测）
  const handleGenerateReport = async () => {
    message.warning("请先在重叠对比视图中完成去雾处理，再返回导出报告");
    handleCloseReportModal();
  };

  // 滤镜CSS字符串
  const filterStyle = `brightness(${brightness}%) contrast(${contrast}%) saturate(${saturate}%)`;

  return (
    <div className={styles["app-container"]}>
      {/* 左侧控制面板：放大镜与滤镜调节 */}
      <div className={styles["sidebar"]}>
        <Card className={styles["sidebar-card"]}>
          <h3>对比工具面板</h3>

          {/* 放大镜控制 */}
          <Form layout="vertical">
            <Form.Item label="放大镜">
              <Switch
                checked={magnifierEnabled}
                onChange={() => dispatch(toggleMagnifierShow())}
              />
            </Form.Item>
            {magnifierEnabled && (
              <>
                <Form.Item label="放大镜形状">
                  <Radio.Group
                    value={magnifierShape}
                    onChange={(e) =>
                      dispatch(setMagnifierShape(e.target.value))
                    }
                  >
                    <Radio value="square">正方形</Radio>
                    <Radio value="circle">圆形</Radio>
                  </Radio.Group>
                </Form.Item>
                <Form.Item
                  label={`放大倍数：${magnifierZoomLevel.toFixed(1)}x`}
                >
                  <Slider
                    min={2}
                    max={20}
                    step={0.5}
                    value={magnifierZoomLevel}
                    onChange={(v) => dispatch(setMagnifierZoomLevel(v))}
                  />
                </Form.Item>
                <Form.Item label={`透镜宽度：${magnifierWidth}px`}>
                  <Slider
                    min={100}
                    max={1000}
                    value={magnifierWidth}
                    onChange={(v) =>
                      dispatch(
                        setMagnifierSize({
                          width: v,
                          height: magnifierHeight,
                        })
                      )
                    }
                  />
                </Form.Item>
                <Form.Item label={`透镜高度：${magnifierHeight}px`}>
                  <Slider
                    min={100}
                    max={1000}
                    value={magnifierHeight}
                    onChange={(v) =>
                      dispatch(
                        setMagnifierSize({
                          width: magnifierWidth,
                          height: v,
                        })
                      )
                    }
                  />
                </Form.Item>
              </>
            )}

            {/* 滤镜控制 */}
            <Form.Item label={`亮度：${brightness}%`}>
              <Slider
                min={0}
                max={200}
                value={brightness}
                onChange={(v) => dispatch(setBrightness(v))}
              />
            </Form.Item>
            <Form.Item label={`对比度：${contrast}%`}>
              <Slider
                min={0}
                max={200}
                value={contrast}
                onChange={(v) => dispatch(setContrast(v))}
              />
            </Form.Item>
            <Form.Item label={`饱和度：${saturate}%`}>
              <Slider
                min={0}
                max={200}
                value={saturate}
                onChange={(v) => dispatch(setSaturate(v))}
              />
            </Form.Item>
          </Form>

          {images.length > 0 && (
            <>
              <Button
                block
                icon={<ReloadOutlined />}
                onClick={handleReset}
                style={{ marginTop: 8 }}
              >
                清空并重置
              </Button>
              <Button
                block
                type="dashed"
                icon={<DownloadOutlined />}
                onClick={handleOpenReportModal}
                style={{ marginTop: 8 }}
              >
                导出对比报告
              </Button>
            </>
          )}
        </Card>
      </div>

      {/* 右侧展示区域 */}
      <Card className={styles["flex-center"]}>
        {images.length === 0 ? (
          // 空状态：上传入口
          <div className={styles["upload-area"]}>
            <Upload.Dragger
              multiple
              beforeUpload={() => false}
              accept="image/*"
              showUploadList={false}
              onChange={(info) => {
                const files = info.fileList
                  .map((f) => f.originFileObj)
                  .filter((f) => !!f) as File[];
                if (files.length > 0) handleAddImages(files);
              }}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">
                点击或拖拽上传多张图片进行并排对比
              </p>
              <p className="ant-upload-hint">支持同时上传多张图片</p>
            </Upload.Dragger>
          </div>
        ) : (
          <div className={styles["parallel-container"]}>
            {images.map((img, index) => (
              <div key={img.id} className={styles["image-panel"]}>
                <span className={styles["image-label"]}>
                  {img.name || `图片${index + 1}`}
                </span>
                <CloseOutlined
                  className={styles["remove-btn"]}
                  onClick={() => handleRemoveImage(img.id)}
                />
                <div
                  className={styles["image-wrapper"]}
                  onMouseMove={(e) => handleMouseMove(e, index, img)}
                  onMouseLeave={handleMouseLeave}
                >
                  <img
                    src={img.url}
                    alt={img.name}
                    style={{ filter: filterStyle }}
                  />
                  {/* 放大镜透镜 */}
                  {magnifierEnabled && hoveredIndex === index && (
                    <div
                      className={styles["magnifier-lens"]}
                      style={getLensStyle(img)}
                    />
                  )}
                </div>
              </div>
            ))}
            {/* 继续添加图片入口 */}
            <div className={styles["add-panel"]}>
              <Upload
                multiple
                beforeUpload={() => false}
                accept="image/*"
                showUploadList={false}
                onChange={(info) => {
                  const files = info.fileList
                    .map((f) => f.originFileObj)
                    .filter((f) => !!f) as File[];
                  if (files.length > 0) handleAddImages(files);
                }}
              >
                <Button type="dashed" icon={<PlusOutlined />}>
                  添加更多图片
                </Button>
              </Upload>
            </div>
          </div>
        )}
      </Card>

      {/* 导出对比报告弹窗 */}
      <Modal
        title="导出对比报告"
        open={reportModalOpen}
        onCancel={handleCloseReportModal}
        footer={null}
        width={500}
      >
        <div style={{ padding: "8px 0" }}>
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontWeight: 500, marginBottom: 8 }}>报告格式</div>
            <Radio.Group
              value={reportFormat}
              onChange={(e) => setReportFormat(e.target.value)}
              style={{ display: "block" }}
            >
              <Radio value="pdf">PDF 文档</Radio>
              <Radio value="image">图片 (PNG)</Radio>
            </Radio.Group>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontWeight: 500, marginBottom: 8 }}>报告选项</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Switch checked={includeMetrics} onChange={setIncludeMetrics} />
                <span>包含性能指标</span>
              </div>
            </div>
          </div>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
            <Button onClick={handleCloseReportModal}>取消</Button>
            <Button
              type="primary"
              onClick={handleGenerateReport}
              loading={exporting}
            >
              生成报告
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default Parallel;
