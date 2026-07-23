import {
  AlgorithmAPI,
  FileAPI,
  ModelAPI,
  type FileInfo,
  type OptionType,
} from "dehaze-sdk-js";

import AlgorithmToolBar from "@/components/AlgorithmToolBar";
import Camera from "@/components/Camera";
import DatasetImageSelect from "@/components/DatasetImageSelect";
import ExampleImageSelect from "@/components/ExampleImageSelect";
import OverlapImageShow from "@/components/OverlapImageShow";
import SingleImageShow from "@/components/SingleImageShow";
import { ImageTypeEnum } from "@/enums/ImageTypeEnum";

import { RootState } from "@/store";
import {
  setImageUrl,
  setImageUrls,
  setLoading,
  setModelId,
} from "@/store/modules/imageShowSlice";

import { changeUrl } from "@/utils";
import { calculateFileMd5 } from "@/utils/md5";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  FileImageOutlined,
  LoadingOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  List,
  message,
  Modal,
  Progress,
  Select,
  Steps,
  Tag,
} from "antd";
import React, { useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate, useSearchParams } from "react-router-dom";
import exampleImages from "./exampleImages";

import styles from "./index.module.scss";

/** 处理阶段定义，对应需求规格中的5个阶段及其进度区间 */
const PROCESS_STAGES = [
  { title: "预处理", description: "图片格式转换、尺寸归一化", min: 0, max: 10 },
  {
    title: "算法初始化",
    description: "加载算法模型、资源分配",
    min: 10,
    max: 20,
  },
  {
    title: "去雾处理",
    description: "执行去雾算法、实时进度更新",
    min: 20,
    max: 90,
  },
  {
    title: "后处理",
    description: "色彩校正、对比度增强、锐化",
    min: 90,
    max: 95,
  },
  {
    title: "保存",
    description: "生成结果图像、保存到服务器",
    min: 95,
    max: 100,
  },
];

/** 根据进度百分比计算当前所处阶段索引 */
const getStageIndex = (progress: number): number => {
  for (let i = 0; i < PROCESS_STAGES.length; i++) {
    if (progress < PROCESS_STAGES[i].max) return i;
  }
  return PROCESS_STAGES.length - 1;
};

/** 批量任务项 */
interface BatchTask {
  id: number;
  fileName: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  resultUrl?: string;
  error?: string;
}

const Dehaze: React.FC = () => {
  const [show, setShow] = useState({
    camera: false,
    singleImage: false,
    example: true,
    overlap: false,
  });

  const [selectedModel, setSelectedModel] = useState<number>();
  const [algorithmOptions, setAlgorithmOptions] = useState<OptionType[]>([]);
  const [dialogVisible, setDialogVisible] = useState(false);
  const [cleanUrl, setCleanUrl] = useState("");

  // 处理进度相关状态
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const cancelFlagRef = useRef(false);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 算法参数
  const [dehazeIntensity, setDehazeIntensity] = useState(50);
  const [sharpenLevel, setSharpenLevel] = useState(30);

  // 批量处理状态
  const [batchTasks, setBatchTasks] = useState<BatchTask[]>([]);
  const [batchMode, setBatchMode] = useState(false);
  const batchCancelRef = useRef(false);

  // 保存结果状态
  const [saving, setSaving] = useState(false);

  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // 使用选择器获取状态
  const urls = useSelector((state: RootState) => state.imageShow.urls);
  const modelId = useSelector((state: RootState) => state.imageShow.modelId);

  // 获取模型可选项，并默认选择第一个模型展示
  useEffect(() => {
    // 检查 URL 查询参数 imageUrl，若有则自动加载图片（来自图像输入页跳转）
    const imageUrl = searchParams.get("imageUrl");
    if (imageUrl) {
      dispatch(setImageUrl({ url: imageUrl, type: ImageTypeEnum.HAZE }));
      setShow((prev) => ({
        ...prev,
        singleImage: true,
        example: false,
        overlap: false,
      }));
    }
    const fetchData = async () => {
      const options = await AlgorithmAPI.getOption();
      setAlgorithmOptions(options);
    };
    fetchData().then();
  }, []);

  // 组件卸载时清理定时器
  useEffect(() => {
    return () => {
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
      }
    };
  }, []);

  const handleCameraSave = (file: File) => {
    handleImageUpload(file);
  };

  const handleImageUpload = async (file: File) => {
    dispatch(setLoading(true));
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
        res = await FileAPI.upload(file, modelId);
      }
      dispatch(
        setImageUrl({
          url: changeUrl(res.url),
          type: ImageTypeEnum.HAZE,
        })
      );
      setBatchMode(false);
      setShow((prev) => ({
        ...prev,
        singleImage: true,
        example: false,
        overlap: false,
      }));
    } catch (err) {
      message.error(err instanceof Error ? err.message : "上传失败");
    } finally {
      dispatch(setLoading(false));
    }
  };

  const handleReset = () => {
    dispatch(setImageUrls([])); // 重置 urls
    setBatchMode(false);
    setBatchTasks([]);
    setShow((prev) => ({
      ...prev,
      example: true,
      singleImage: false,
      overlap: false,
    }));
  };

  /** 启动模拟进度定时器，递增进度直到90%等待API完成 */
  const startProgressSimulation = () => {
    progressTimerRef.current = setInterval(() => {
      setProgress((prev) => {
        if (cancelFlagRef.current) return prev;
        // 预处理和算法初始化阶段快速通过
        if (prev < 20) return Math.min(prev + 2, 20);
        // 去雾处理阶段缓慢递增，在90%处等待API完成
        if (prev < 90) return Math.min(prev + Math.random() * 3, 90);
        return prev;
      });
    }, 200);
  };

  const handleGenerateImage = async () => {
    if (!selectedModel) return message.error("请选择模型");
    if (!urls[0]) return message.error("请先上传图片");

    // 重置进度状态
    cancelFlagRef.current = false;
    setProcessing(true);
    setProgress(0);
    dispatch(setLoading(true));
    setShow((prev) => ({ ...prev, overlap: false, singleImage: false }));

    // 启动模拟进度定时器
    startProgressSimulation();

    try {
      const response = await ModelAPI.prediction({
        modelId: selectedModel,
        url: urls[0].url,
      });

      // 已取消则忽略结果
      if (cancelFlagRef.current) return;

      // 后处理阶段
      setProgress(95);
      await new Promise((resolve) => setTimeout(resolve, 300));
      if (cancelFlagRef.current) return;

      // 保存阶段
      setProgress(100);

      dispatch(
        setImageUrl({
          url: changeUrl(response.hazeUrl),
          type: ImageTypeEnum.HAZE,
        })
      );
      dispatch(
        setImageUrl({
          url: changeUrl(response.predUrl),
          type: ImageTypeEnum.PRED,
        })
      );
      setShow((prev) => ({ ...prev, overlap: true, singleImage: false }));
      message.success("去雾处理完成");
    } catch (error) {
      if (!cancelFlagRef.current) {
        message.error("生成失败");
        setShow((prev) => ({ ...prev, singleImage: true, overlap: false }));
      }
    } finally {
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
        progressTimerRef.current = null;
      }
      setProcessing(false);
      dispatch(setLoading(false));
    }
  };

  /** 取消正在进行的处理 */
  const handleCancel = () => {
    cancelFlagRef.current = true;
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
    setProcessing(false);
    setProgress(0);
    dispatch(setLoading(false));
    setShow((prev) => ({ ...prev, singleImage: true, overlap: false }));
    message.info("已取消处理");
  };

  /** 保存处理结果：获取预测图后上传保存 */
  const handleSaveResult = async () => {
    const predUrl = urls.find((u) => u.label.text === ImageTypeEnum.PRED)?.url;
    if (!predUrl) return message.error("没有可保存的结果");

    setSaving(true);
    try {
      const res = await fetch(predUrl);
      const blob = await res.blob();
      const file = new File([blob], `dehaze_result_${Date.now()}.png`, {
        type: blob.type,
      });
      // 计算文件哈希进行秒传校验
      const md5 = await calculateFileMd5(file);
      const existingResult = await FileAPI.uploadCheck(md5);
      if (!existingResult) {
        await FileAPI.upload(file, modelId);
      }
      message.success("结果保存成功");
    } catch (error) {
      message.error("保存失败");
    } finally {
      setSaving(false);
    }
  };

  /** 批量处理：串行处理每张图片 */
  const handleBatchUpload = async (files: File[]) => {
    if (files.length > 20) {
      message.warning("最多支持20张图片");
      return;
    }
    if (!selectedModel) {
      message.error("请先选择模型");
      return;
    }

    setBatchMode(true);
    batchCancelRef.current = false;
    setShow((prev) => ({
      ...prev,
      example: false,
      singleImage: false,
      overlap: false,
    }));

    // 初始化批量任务列表
    const tasks: BatchTask[] = files.map((f, i) => ({
      id: i,
      fileName: f.name,
      status: "pending",
      progress: 0,
    }));
    setBatchTasks(tasks);

    // 串行处理每张图片
    for (let i = 0; i < files.length; i++) {
      if (batchCancelRef.current) break;

      setBatchTasks((prev) =>
        prev.map((t, idx) => (idx === i ? { ...t, status: "processing" } : t))
      );

      try {
        // 计算文件哈希进行秒传校验
        const md5 = await calculateFileMd5(files[i]);
        const existingFile = await FileAPI.uploadCheck(md5);
        const uploadRes = existingFile
          ? existingFile
          : await FileAPI.upload(files[i], modelId);
        if (batchCancelRef.current) break;

        // 调用预测接口
        const predRes = await ModelAPI.prediction({
          modelId: selectedModel,
          url: changeUrl(uploadRes.url),
        });
        if (batchCancelRef.current) break;

        // 模拟单张进度递增
        for (let p = 0; p <= 100; p += 25) {
          if (batchCancelRef.current) break;
          setBatchTasks((prev) =>
            prev.map((t, idx) =>
              idx === i ? { ...t, progress: Math.min(p, 100) } : t
            )
          );
          await new Promise((resolve) => setTimeout(resolve, 100));
        }

        setBatchTasks((prev) =>
          prev.map((t, idx) =>
            idx === i
              ? {
                  ...t,
                  status: "completed",
                  progress: 100,
                  resultUrl: changeUrl(predRes.predUrl),
                }
              : t
          )
        );
      } catch (error) {
        setBatchTasks((prev) =>
          prev.map((t, idx) =>
            idx === i ? { ...t, status: "failed", error: "处理失败" } : t
          )
        );
      }
    }

    if (!batchCancelRef.current) {
      message.success("批量处理完成");
    }
  };

  /** 取消批量处理 */
  const handleBatchCancel = () => {
    batchCancelRef.current = true;
    setBatchTasks((prev) =>
      prev.map((t) =>
        t.status === "pending" || t.status === "processing"
          ? { ...t, status: "failed", error: "已取消" }
          : t
      )
    );
    message.info("已取消批量处理");
  };

  const handleExampleImageClick = (url: string) => {
    const selectedExample = exampleImages.find((item) => item.haze === url);
    dispatch(
      setImageUrl({
        url,
        type: ImageTypeEnum.HAZE,
      })
    );
    setCleanUrl(selectedExample?.clean || "");
    setBatchMode(false);
    setShow((prev) => ({
      ...prev,
      singleImage: true,
      example: false,
      overlap: false,
    }));
  };

  const handleSelectModel = (id: number) => {
    setSelectedModel(id);
    dispatch(setModelId(id));
  };

  const handleDatasetImageSelect = (haze: string, clean: string) => {
    dispatch(setImageUrl({ url: haze, type: ImageTypeEnum.HAZE }));
    setCleanUrl(clean);
    setDialogVisible(false);
    setBatchMode(false);
    setShow((prev) => ({
      ...prev,
      singleImage: true,
      example: false,
      overlap: false,
    }));
  };

  const handleEval = () => {
    navigate("/evaluation", {
      state: {
        modelId: selectedModel,
        images: urls,
        cleanUrl,
      },
    });
  };

  const currentStage = getStageIndex(progress);

  // 渲染批量任务状态标签
  const renderBatchStatus = (task: BatchTask) => {
    switch (task.status) {
      case "pending":
        return <Tag color="default">等待中</Tag>;
      case "processing":
        return (
          <div style={{ width: 200 }}>
            <Progress percent={task.progress} size="small" />
          </div>
        );
      case "completed":
        return (
          <Tag icon={<CheckCircleOutlined />} color="success">
            完成
          </Tag>
        );
      case "failed":
        return (
          <Tag icon={<CloseCircleOutlined />} color="error">
            {task.error}
          </Tag>
        );
      default:
        return null;
    }
  };

  return (
    <div className={styles["app-container"]}>
      <AlgorithmToolBar
        disableMore={!show.overlap}
        onUpload={handleImageUpload}
        onEval={handleEval}
        onTakePhoto={() => setShow((prev) => ({ ...prev, camera: true }))}
        onReset={handleReset}
        onGenerate={handleGenerateImage}
        onSelectFromDataset={() => setDialogVisible(true)}
        dehazeIntensity={dehazeIntensity}
        sharpenLevel={sharpenLevel}
        onDehazeIntensityChange={setDehazeIntensity}
        onSharpenLevelChange={setSharpenLevel}
        onBatchUpload={handleBatchUpload}
        processing={processing}
        onCancel={handleCancel}
        showSave={show.overlap}
        onSave={handleSaveResult}
      >
        <div className={styles["select-wrap"]}>
          <span>选择去雾模型</span>
          <Select
            value={selectedModel}
            options={algorithmOptions}
            onChange={handleSelectModel}
            style={{ width: 240 }}
          />
        </div>
      </AlgorithmToolBar>

      <Card className={styles["flex-center"]}>
        {/* 处理进度展示：5阶段 Steps + Progress */}
        {processing && (
          <div className={styles["progress-wrap"]}>
            <Steps
              current={currentStage}
              size="small"
              direction="vertical"
              items={PROCESS_STAGES.map((s) => ({
                title: s.title,
                description: s.description,
              }))}
            />
            <Progress
              percent={progress}
              status="active"
              strokeColor={{ from: "#108ee9", to: "#87d068" }}
            />
            <p className={styles["progress-tip"]}>
              当前阶段：{PROCESS_STAGES[currentStage].title}（
              {PROCESS_STAGES[currentStage].min}%-
              {PROCESS_STAGES[currentStage].max}%）
            </p>
          </div>
        )}

        {/* 批量处理任务列表 */}
        {!processing && batchMode && (
          <div className={styles["batch-wrap"]}>
            <div className={styles["batch-header"]}>
              <h3>批量处理任务（{batchTasks.length}张）</h3>
              <div className={styles["batch-actions"]}>
                <Button
                  size="small"
                  onClick={handleBatchCancel}
                  disabled={batchTasks.every(
                    (t) => t.status === "completed" || t.status === "failed"
                  )}
                >
                  取消批量
                </Button>
                <Button
                  size="small"
                  type="primary"
                  ghost
                  onClick={() => {
                    setBatchMode(false);
                    setShow((prev) => ({ ...prev, example: true }));
                  }}
                >
                  返回
                </Button>
              </div>
            </div>
            <List
              dataSource={batchTasks}
              renderItem={(task) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={<FileImageOutlined style={{ fontSize: 24 }} />}
                    title={task.fileName}
                    description={renderBatchStatus(task)}
                  />
                  {task.resultUrl && (
                    <img
                      src={task.resultUrl}
                      alt="结果"
                      style={{
                        width: 80,
                        height: 80,
                        objectFit: "cover",
                        borderRadius: 4,
                      }}
                    />
                  )}
                </List.Item>
              )}
            />
          </div>
        )}

        {/* 常规内容展示 */}
        {!processing && !batchMode && show.example && (
          <ExampleImageSelect
            className={styles["example"]}
            urls={exampleImages.map((item) => item.haze)}
            onExampleSelect={handleExampleImageClick}
          />
        )}
        {!processing && !batchMode && show.camera && (
          <Camera
            onSave={handleCameraSave}
            onCancel={() => setShow((prev) => ({ ...prev, camera: false }))}
          />
        )}
        {!processing && !batchMode && show.singleImage && (
          <SingleImageShow src={urls[0]?.url || ""} />
        )}
        {!processing && !batchMode && show.overlap && <OverlapImageShow />}
        {saving && (
          <div className={styles["saving-tip"]}>
            <LoadingOutlined /> 正在保存结果...
          </div>
        )}
      </Card>

      <Modal
        title="选择数据集图片"
        open={dialogVisible}
        onCancel={() => setDialogVisible(false)}
        footer={null}
      >
        <DatasetImageSelect onSelected={handleDatasetImageSelect} />
      </Modal>
    </div>
  );
};

export default Dehaze;
