import { AiImageAPI, AiModelAPI, FileAPI, type AiModelVO } from "clarity-sdk-js";

import AlgorithmToolBar from "@/components/AlgorithmToolBar";
import SingleImageShow from "@/components/SingleImageShow";

import { setLoading } from "@/store/modules/imageShowSlice";

import { InboxOutlined } from "@ant-design/icons";
import { Card, message, Select, Upload } from "antd";
import React, { useEffect, useState } from "react";
import { useDispatch } from "react-redux";
import styles from "./index.module.scss";

/** 页面状态 */
type SegStage = "upload" | "preview" | "result";

const Segmentation: React.FC = () => {
  const dispatch = useDispatch();

  const [stage, setStage] = useState<SegStage>("upload");
  const [selectedModel, setSelectedModel] = useState<string>();
  const [models, setModels] = useState<AiModelVO[]>([]);
  const [originUrl, setOriginUrl] = useState("");
  const [resultUrl, setResultUrl] = useState("");

  // 模型下拉数据源：当前用户可用的图像模型
  useEffect(() => {
    AiModelAPI.listEnabledModels("image_edit").then(setModels);
  }, []);

  /** 上传图片 */
  const handleImageUpload = (file: File) => {
    dispatch(setLoading(true));
    FileAPI.upload(file)
      .then((res) => {
        setOriginUrl(res.url);
        setResultUrl("");
        setStage("preview");
      })
      .catch((err) => message.error(err))
      .finally(() => dispatch(setLoading(false)));
  };

  /** 执行分割处理 */
  const handleGenerate = async () => {
    if (!selectedModel) return message.error("请选择分割算法");
    if (!originUrl) return message.error("请先上传图片");

    dispatch(setLoading(true));
    try {
      const response = await AiImageAPI.submitAndWait({
        model: selectedModel,
        imageUrl: originUrl,
      });
      if (response.status === 3) {
        throw new Error(response.errorMessage || "分割处理失败");
      }
      setResultUrl(response.resultUrl || "");
      setStage("result");
      message.success("分割处理完成");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "分割处理失败");
    } finally {
      dispatch(setLoading(false));
    }
  };

  /** 重置 */
  const handleReset = () => {
    setOriginUrl("");
    setResultUrl("");
    setStage("upload");
  };

  const handleSelectModel = (slug: string) => {
    setSelectedModel(slug);
  };

  return (
    <div className={styles["app-container"]}>
      <AlgorithmToolBar
        title="图像分割"
        description="对图像进行语义分割处理，识别并标记图像中的不同区域，输出分割掩码图"
        disableMore={stage !== "result"}
        onUpload={handleImageUpload}
        onTakePhoto={() => {}}
        onEval={() => message.info("暂不支持评估")}
        onReset={handleReset}
        onGenerate={handleGenerate}
        onSelectFromDataset={() => message.info("暂不支持从数据集选择")}
      >
        <div className={styles["select-wrap"]}>
          <span>选择分割模型</span>
          <Select
            value={selectedModel}
            options={models.map((m) => ({
              value: m.modelId,
              label: m.displayName,
            }))}
            onChange={handleSelectModel}
            style={{ width: 240 }}
            placeholder="请选择模型"
          />
        </div>
      </AlgorithmToolBar>

      <Card className={styles["flex-center"]}>
        {stage === "upload" && (
          <div className={styles["upload-area"]}>
            <Upload.Dragger
              beforeUpload={() => false}
              accept="image/*"
              showUploadList={false}
              onChange={(info) => {
                const file = info.fileList[0]?.originFileObj;
                if (file) handleImageUpload(file);
              }}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽上传图片进行分割处理</p>
              <p className="ant-upload-hint">支持单张图片上传</p>
            </Upload.Dragger>
          </div>
        )}

        {stage === "preview" && (
          <div className={styles["result-container"]}>
            <div className={styles["result-panel"]}>
              <div className={styles["panel-title"]}>原始图像</div>
              <SingleImageShow src={originUrl} />
            </div>
          </div>
        )}

        {stage === "result" && (
          <div className={styles["result-container"]}>
            <div className={styles["result-panel"]}>
              <div className={styles["panel-title"]}>原始图像</div>
              <img
                className={styles["panel-image"]}
                src={originUrl}
                alt="原图"
              />
            </div>
            <div className={styles["result-panel"]}>
              <div className={styles["panel-title"]}>分割结果</div>
              <img
                className={styles["panel-image"]}
                src={resultUrl}
                alt="分割结果"
              />
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

export default Segmentation;
