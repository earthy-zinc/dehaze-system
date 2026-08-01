import {
  AlgorithmAPI,
  FileAPI,
  ModelAPI,
  type CompareReportForm,
  type CompareReportResultVO,
  type FileInfo,
  type OptionType,
} from "dehaze-sdk-js";
import AlgorithmToolBar from "@/components/AlgorithmToolBar";
import { MagnifierInfo, Point } from "@/components/AlgorithmToolBar/types";
import ExampleImageSelect from "@/components/ExampleImageSelect";
import Loading from "@/components/Loading";
import OverlapImageShow from "@/components/OverlapImageShow";
import SingleImageShow from "@/components/SingleImageShow";
import { useWindowSize } from "@/hooks/useWindowSize";
import { calculateFileMd5 } from "@/utils/md5";
import {
  Button,
  Card,
  Cascader,
  message,
  Modal,
  Select,
  Switch,
  Tag,
} from "antd";
import React, { useEffect, useMemo, useState } from "react";
import { DownloadOutlined, InfoCircleOutlined } from "@ant-design/icons";
import styles from "./index.module.scss";

type ActivePage =
  "singleImage" | "example" | "loading" | "overlap" | "effect" | "camera";

export default function Overlap() {
  const [image1, setImage1] = useState(
    "http://192.168.31.3:8989/api/v1/files/dataset/thumbnail/Dense-Haze/hazy/01_hazy.png"
  );
  const [image2, setImage2] = useState(
    "http://localhost:9000/trained-models/20241123/5cf1637dd4f74f9187429aeb6ed1f772.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=admin%2F20241123%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241123T101336Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=ace226d1564ef3450d050fa15700613623e3d28f3caa5352dca421f117493198"
  );
  const [showMask, setShowMask] = useState(false);
  const [contrast, setContrast] = useState(0);
  const [brightness, setBrightness] = useState(0);
  const [originScale, setOriginScale] = useState(1);
  const [point, setPoint] = useState<Point>({ x: 0, y: 0 });
  const [exampleImageUrls, setExampleImageUrls] = useState<string[]>([
    "http://192.168.31.3:8989/api/v1/files/dataset/thumbnail/Dense-Haze/hazy/01_hazy.png",
  ]);
  const [modelOptions, setModelOptions] = useState<OptionType[]>([]);
  const [selectedModel, setSelectedModel] = useState<OptionType>({
    value: 1,
    label: "模型名称",
  });
  const [activePage, setActivePage] = useState<ActivePage>("overlap");
  const [algorithmInfo, setAlgorithmInfo] = useState<any>(null);
  const [algorithmLoading, setAlgorithmLoading] = useState(false);
  const [lastLogId, setLastLogId] = useState<number | null>(null);

  // 导出报告相关状态
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [reportFormat, setReportFormat] = useState<"pdf" | "image">("pdf");
  const [includeMetrics, setIncludeMetrics] = useState(true);
  const [includeFilters, setIncludeFilters] = useState(false);
  const [exporting, setExporting] = useState(false);

  const { width } = useWindowSize();

  const disableMore = useMemo(() => activePage !== "overlap", [activePage]);

  const magnifier = useMemo(() => {
    return {
      imgUrls: [image1, image2],
      radius: Math.floor((width * 0.3 - 90) / 4),
      originScale: originScale,
      point: point,
    } as MagnifierInfo;
  }, [width, point, originScale]);

  const handleCameraSave = (file: File) => {
    // 上传文件
    setActivePage("camera");
  };

  const handleImageUpload = async (file: File) => {
    setActivePage("loading");
    try {
      // 计算文件哈希进行秒传校验
      const md5 = await calculateFileMd5(file);
      const existing = await FileAPI.uploadCheck(md5);
      let res: FileInfo;
      if (existing) {
        // 秒传命中，直接复用已有文件
        res = existing;
        message.success("文件秒传成功");
      } else {
        // 未命中，执行实际上传
        res = await FileAPI.upload(file);
      }
      // 文件上传成功后拿到服务器返回的 url 地址在右侧渲染
      setImage1(res.url);
      // 将文件显示到 SingleImageShow 组件中
      setActivePage("singleImage");
    } catch (err: any) {
      setActivePage("example");
      message.error(err.message);
    }
  };

  const handleReset = () => {
    setImage1("");
    setImage2("");
    setShowMask(false);
    setActivePage("example");
  };

  const handleGenerateImage = () => {
    setActivePage("loading");
    ModelAPI.predictAndWait({
      algorithmId: Number(selectedModel?.value) || 1,
      imageUrl: image1,
    })
      .then((res) => {
        setLastLogId(res.logId || null);
        if (res.status === 3) {
          throw new Error(res.errorMessage || "去雾处理失败");
        }
        setImage2(res.resultUrl || "");
      })
      .then(() => setActivePage("overlap"))
      .catch((err) => {
        message.error(err.message);
        setActivePage("singleImage");
      });
  };

  const handleExampleImageClick = (url: string) => {
    setImage1(url);
    setActivePage("singleImage");
  };

  // 获取算法信息
  useEffect(() => {
    if (!selectedModel?.value) return;
    setAlgorithmLoading(true);
    AlgorithmAPI.getAlgorithmInfoById(Number(selectedModel.value))
      .then(setAlgorithmInfo)
      .catch(() => message.error("获取算法信息失败"))
      .finally(() => setAlgorithmLoading(false));
  }, [selectedModel?.value]);

  // 打开导出报告弹窗
  const handleOpenReportModal = () => {
    setReportModalOpen(true);
  };

  // 关闭导出报告弹窗
  const handleCloseReportModal = () => {
    setReportModalOpen(false);
    setExporting(false);
  };

  // 生成并下载对比报告
  const handleGenerateReport = async () => {
    if (!disableMore) return;
    if (!lastLogId) {
      message.warning("请先进行去雾处理以生成对比报告");
      return;
    }
    setExporting(true);
    try {
      message.info("正在生成报告...");
      const reportForm: CompareReportForm = {
        logId: lastLogId,
        format: reportFormat,
        includeMetrics,
        includeFilters,
      };

      const reportTask = await ModelAPI.generateReport(reportForm);
      message.info(`报告生成任务已启动，任务ID: ${reportTask.taskId}`);

      // 轮询报告状态
      let reportStatus: CompareReportResultVO = reportTask;
      const pollInterval = setInterval(async () => {
        reportStatus = await ModelAPI.getReportStatus(reportTask.taskId);
        if (reportStatus.status === 2) {
          // 完成
          clearInterval(pollInterval);
          if (reportStatus.downloadUrl) {
            window.open(reportStatus.downloadUrl, "_blank");
            message.success("报告生成成功");
          } else {
            message.error("报告生成成功但下载链接为空");
          }
          handleCloseReportModal();
        } else if (reportStatus.status === 3) {
          // 失败
          clearInterval(pollInterval);
          message.error(reportStatus.errorMessage || "报告生成失败");
          handleCloseReportModal();
        }
      }, 2000);
    } catch (err: any) {
      message.error("报告生成失败: " + (err?.message || "未知错误"));
      setExporting(false);
    }
  };

  const handleMouseover = (p: Point) => {
    setPoint({ x: p.x, y: p.y });
  };

  useEffect(() => {
    AlgorithmAPI.getOption().then((options) => {
      setModelOptions(options);
      if (options.length > 0) {
        setSelectedModel(options[0]);
      }
    });
  }, []);

  const handleChange = (
    value: (string | number)[],
    selectedOptions: OptionType[]
  ) => {
    if (selectedOptions.length > 0) {
      setSelectedModel(selectedOptions[selectedOptions.length - 1]);
    }
  };
  const showActivePage = () => {
    switch (activePage) {
      case "singleImage":
        return <SingleImageShow src={image1} />;
      case "example":
        return (
          <ExampleImageSelect
            urls={exampleImageUrls}
            onExampleSelect={(url: string) => handleExampleImageClick(url)}
          />
        );
      case "camera":
        return <div>相机</div>;
      case "overlap":
        return (
          <div
            style={{ display: "flex", flexDirection: "column", height: "100%" }}
          >
            {/* 算法信息区域 */}
            <Card style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <InfoCircleOutlined
                  style={{ fontSize: 20, color: "#1890ff" }}
                />
                <div style={{ flex: 1 }}>
                  {algorithmLoading ? (
                    <span style={{ color: "#999" }}>加载中...</span>
                  ) : algorithmInfo ? (
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 16 }}>
                        {algorithmInfo.name}
                        {algorithmInfo.version && (
                          <Tag style={{ marginLeft: 8 }}>
                            v{algorithmInfo.version}
                          </Tag>
                        )}
                      </div>
                      <div
                        style={{ fontSize: 13, color: "#666", marginTop: 4 }}
                      >
                        {algorithmInfo.description || "暂无描述"}
                      </div>
                      <div
                        style={{ fontSize: 13, color: "#999", marginTop: 4 }}
                      >
                        {algorithmInfo.params && (
                          <span>参数: {algorithmInfo.params}</span>
                        )}
                      </div>
                    </div>
                  ) : (
                    <span style={{ color: "#999" }}>暂无算法信息</span>
                  )}
                </div>
              </div>
            </Card>

            {/* 导出报告按钮 */}
            <div
              style={{
                display: "flex",
                justifyContent: "flex-end",
                marginBottom: 8,
              }}
            >
              <Button
                type="primary"
                icon={<DownloadOutlined />}
                onClick={handleOpenReportModal}
                disabled={disableMore || !lastLogId}
                loading={exporting}
              >
                导出对比报告
              </Button>
            </div>

            {/* 重叠对比图片 */}
            <div style={{ flex: 1, overflow: "hidden" }}>
              <OverlapImageShow />
            </div>
          </div>
        );
      case "effect":
        return <div>特效</div>;
      case "loading":
        return <Loading />;
      default:
        return null;
    }
  };
  return (
    <div className={styles["app-container"]}>
      {/* 左侧工具栏 */}
      <AlgorithmToolBar
        disableMore={disableMore}
        onUpload={handleImageUpload}
        onTakePhoto={() => setActivePage("camera")}
        onReset={handleReset}
        onGenerate={handleGenerateImage}
        onEval={() => {}}
        onSelectFromDataset={() => {}}
      >
        {/* 选择模型区域 */}
        <div className={styles["select-wrap"]}>
          <h3 className={"text-align-center"}>选择去雾模型</h3>
          <Cascader
            className={"ml-20"}
            defaultValue={[selectedModel?.value || 1]}
            options={modelOptions}
            onChange={handleChange}
          />
        </div>
      </AlgorithmToolBar>
      {/* 右侧展示栏 */}
      <Card className={styles["flex-center"]}>{showActivePage()}</Card>

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
            <Select
              value={reportFormat}
              onChange={(v) => setReportFormat(v)}
              options={[
                { label: "PDF 文档", value: "pdf" },
                { label: "图片 (PNG)", value: "image" },
              ]}
              style={{ width: "100%" }}
            />
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontWeight: 500, marginBottom: 8 }}>报告选项</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Switch checked={includeMetrics} onChange={setIncludeMetrics} />
                <span>包含性能指标</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Switch checked={includeFilters} onChange={setIncludeFilters} />
                <span>包含滤镜参数</span>
              </div>
            </div>
          </div>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
            <Button onClick={handleCloseReportModal}>取消</Button>
            <Button
              type="primary"
              onClick={handleGenerateReport}
              loading={exporting}
              disabled={!lastLogId}
            >
              生成报告
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
