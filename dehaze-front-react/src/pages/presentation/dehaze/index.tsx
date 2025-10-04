import AlgorithmAPI from "@/api/algorithm";
import FileAPI from "@/api/file";
import ModelAPI from "@/api/model";

import AlgorithmToolBar from "@/components/AlgorithmToolBar";
import Camera from "@/components/Camera";
import DatasetImageSelect from "@/components/DatasetImageSelect";
import ExampleImageSelect from "@/components/ExampleImageSelect";
import Loading from "@/components/Loading";
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
import { Card, message, Modal, Select } from "antd";
import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import exampleImages from "./exampleImages";

import styles from "./index.module.scss";

const Dehaze: React.FC = () => {
  const [show, setShow] = useState({
    camera: false,
    singleImage: false,
    example: true,
    loading: false,
    overlap: false,
    effect: false,
  });

  const [selectedModel, setSelectedModel] = useState<number>();
  const [algorithmOptions, setAlgorithmOptions] = useState<OptionType[]>([]);
  const [dialogVisible, setDialogVisible] = useState(false);
  const [cleanUrl, setCleanUrl] = useState("");

  const dispatch = useDispatch();
  const navigate = useNavigate();

  // 使用选择器获取状态
  const urls = useSelector((state: RootState) => state.imageShow.urls);
  const modelId = useSelector((state: RootState) => state.imageShow.modelId);

  // 获取模型可选项，并默认选择第一个模型展示
  useEffect(() => {
    const fetchData = async () => {
      const options = await AlgorithmAPI.getOption();
      setAlgorithmOptions(options);
    };
    fetchData().then();
  }, []);
  const handleCameraSave = (file: File) => {
    handleImageUpload(file);
  };

  const handleImageUpload = (file: File) => {
    dispatch(setLoading(true));
    FileAPI.upload(file, modelId)
      .then((res) => {
        dispatch(
          setImageUrl({
            url: changeUrl(res.url),
            type: ImageTypeEnum.HAZE,
          })
        );
        setShow({ ...show, singleImage: true, example: false });
      })
      .catch((err) => message.error(err))
      .finally(() => dispatch(setLoading(false)));
  };

  const handleReset = () => {
    dispatch(setImageUrls([])); // 重置 urls
    setShow({ ...show, example: true });
  };

  const handleGenerateImage = async () => {
    if (!selectedModel) return message.error("请选择模型");
    if (!urls[0]) return message.error("请先上传图片");

    dispatch(setLoading(true));
    try {
      const response = await ModelAPI.prediction({
        modelId: selectedModel,
        url: urls[0].url,
      });
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
      setShow({ ...show, overlap: true });
    } catch (error) {
      message.error("生成失败");
      setShow({ ...show, singleImage: true });
    } finally {
      dispatch(setLoading(false));
    }
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
    setShow({ ...show, singleImage: true });
  };

  useEffect(() => {
    const fetchModels = async () => {};
    fetchModels();
  }, []);

  const handleSelectModel = (id: number) => {
    setSelectedModel(id);
    dispatch(setModelId(id));
  };

  const handleDatasetImageSelect = (haze: string, clean: string) => {
    dispatch(setImageUrl({ url: haze, type: ImageTypeEnum.HAZE }));
    setCleanUrl(clean);
    setDialogVisible(false);
    setShow({ ...show, singleImage: true });
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

  return (
    <div className={styles["app-container"]}>
      <AlgorithmToolBar
        disableMore={!show.overlap}
        onUpload={handleImageUpload}
        onEval={handleEval}
        onTakePhoto={() => setShow({ ...show, camera: true })}
        onReset={handleReset}
        onGenerate={handleGenerateImage}
        onSelectFromDataset={() => setDialogVisible(true)}
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
        {show.example && (
          <ExampleImageSelect
            className={styles["example"]}
            urls={exampleImages.map((item) => item.haze)}
            onExampleSelect={handleExampleImageClick}
          />
        )}
        {show.camera && (
          <Camera
            onSave={handleCameraSave}
            onCancel={() => setShow({ ...show, camera: false })}
          />
        )}
        {show.singleImage && <SingleImageShow src={urls[0]?.url || ""} />}
        {show.loading && <Loading />}
        {show.overlap && <OverlapImageShow />}
      </Card>

      <Modal
        title="选择数据集图片"
        visible={dialogVisible}
        onCancel={() => setDialogVisible(false)}
        footer={null}
      >
        <DatasetImageSelect onSelected={handleDatasetImageSelect} />
      </Modal>
    </div>
  );
};

export default Dehaze;
