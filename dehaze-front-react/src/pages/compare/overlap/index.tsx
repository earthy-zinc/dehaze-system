import AlgorithmAPI from "@/api/algorithm";
import FileAPI from "@/api/file";
import ModelAPI from "@/api/model";
import AlgorithmToolBar from "@/components/AlgorithmToolBar";
import { MagnifierInfo, Point } from "@/components/AlgorithmToolBar/types";
import ExampleImageSelect from "@/components/ExampleImageSelect";
import Loading from "@/components/Loading";
import OverlapImageShow from "@/components/OverlapImageShow";
import SingleImageShow from "@/components/SingleImageShow";
import { useWindowSize } from "@/hooks/useWindowSize";
import { Card, Cascader, message } from "antd";
import React, { useEffect, useMemo, useState } from "react";
import styles from "./index.module.scss";

type ActivePage =
  | "singleImage"
  | "example"
  | "loading"
  | "overlap"
  | "effect"
  | "camera";

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

  const handleImageUpload = (file: File) => {
    setActivePage("loading");
    // 上传文件
    FileAPI.upload(file)
      .then((res) => {
        // 文件上传成功后拿到服务器返回的 url 地址在右侧渲染
        setImage1(res.url);
      })
      .then(() => {
        // 将文件显示到 SingleImageShow 组件中
        setActivePage("singleImage");
      })
      .catch((err) => {
        setActivePage("example");
        message.error(err.message);
      });
  };

  const handleReset = () => {
    setImage1("");
    setImage2("");
    setShowMask(false);
    setActivePage("example");
  };

  const handleGenerateImage = () => {
    setActivePage("loading");
    ModelAPI.prediction({
      modelId: Number(selectedModel.value) || 1,
      url: image1,
    })
      .then((res) => {
        // 获取生成后的图片url
        setImage2(res.predUrl);
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

  const handleMouseover = (p: Point) => {
    setPoint({ x: p.x, y: p.y });
  };

  useEffect(() => {
    AlgorithmAPI.getOption().then((options) => {
      setModelOptions(options);
      setSelectedModel(options[0]);
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
        return <OverlapImageShow />;
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
            defaultValue={[selectedModel.value || 1]}
            options={modelOptions}
            onChange={handleChange}
          />
        </div>
      </AlgorithmToolBar>
      {/* 右侧展示栏 */}
      <Card className={styles["flex-center"]}>{showActivePage()}</Card>
    </div>
  );
}
