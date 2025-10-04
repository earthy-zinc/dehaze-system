import { RootState } from "@/store";
import {
  setBrightness,
  setContrast,
  setMagnifierShape,
  setMagnifierSize,
  setMagnifierZoomLevel,
  setSaturate,
  toggleDividerShow,
  toggleMagnifierShow,
} from "@/store/modules/imageShowSlice";
import {
  selectDisableGenerate,
  selectDividerEnabled,
  selectMagnifierZoomLevel,
} from "@/store/selector/imageShowSelector";
import { DownOutlined } from "@ant-design/icons";
import {
  Button,
  Card,
  Dropdown,
  Form,
  Menu,
  Radio,
  Slider,
  Typography,
} from "antd";
import React, { useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import styles from "./index.module.scss";

interface AlgorithmToolBarProps {
  title?: string;
  description?: string;
  disableMore: boolean;
  onUpload: (file: File) => void;
  onTakePhoto: () => void;
  onEval: () => void;
  onReset: () => void;
  onGenerate: () => void;
  onSelectFromDataset: () => void;
  children: React.ReactNode;
}

const AlgorithmToolBar: React.FC<AlgorithmToolBarProps> = ({
  title = "图像去雾",
  description = "通过对遭受雾气影响的图像进行相应处理，恢复图像原本的纹理结构和细节信息，进而提升图像的能见度",
  disableMore,
  onUpload,
  onTakePhoto,
  onEval,
  onReset,
  onGenerate,
  onSelectFromDataset,
  children,
}) => {
  const dispatch = useDispatch();
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
  const magnifierZoomLevel = useSelector(selectMagnifierZoomLevel);
  const magnifierWidth = useSelector(
    (state: RootState) => state.imageShow.magnifier.width
  );
  const magnifierHeight = useSelector(
    (state: RootState) => state.imageShow.magnifier.height
  );
  const dividerEnabled = useSelector(selectDividerEnabled);
  const disableGenerate = useSelector(selectDisableGenerate);
  const loading = useSelector((state: RootState) => state.imageShow.loading);

  // 本地状态管理滤镜开关
  const [filterStates, setFilterStates] = useState({
    brightnessEnabled: true,
    contrastEnabled: true,
    saturateEnabled: true,
  });

  const windowWidth = useRef(window.innerWidth);

  useEffect(() => {
    const handleResize = () => (windowWidth.current = window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleMagnifierToggle = () => {
    dispatch(toggleMagnifierShow());
  };

  const handleDividerToggle = () => {
    dispatch(toggleDividerShow());
  };

  const handleFilterToggle = (type: "brightness" | "contrast" | "saturate") => {
    setFilterStates((prev) => ({
      ...prev,
      [`${type}Enabled`]: !prev[`${type}Enabled`],
    }));
  };

  const handleMagnifierChange = (
    type: "shape" | "zoomLevel" | "width" | "height",
    value: any
  ) => {
    switch (type) {
      case "shape":
        dispatch(setMagnifierShape(value));
        break;
      case "zoomLevel":
        dispatch(setMagnifierZoomLevel(value));
        break;
      case "width":
        dispatch(setMagnifierSize({ width: value, height: magnifierHeight }));
        break;
      case "height":
        dispatch(setMagnifierSize({ width: magnifierWidth, height: value }));
        break;
      default:
        break;
    }
  };

  const handleFilterChange = (
    type: "brightness" | "contrast" | "saturate",
    value: number
  ) => {
    switch (type) {
      case "brightness":
        dispatch(setBrightness(value));
        break;
      case "contrast":
        dispatch(setContrast(value));
        break;
      case "saturate":
        dispatch(setSaturate(value));
        break;
      default:
        break;
    }
  };

  const renderMoreMenu = () => (
    <Menu>
      <Menu.Item onClick={handleMagnifierToggle}>
        {magnifierEnabled ? "关闭" : "开启"}放大镜
      </Menu.Item>
      <Menu.Item onClick={handleDividerToggle}>
        {dividerEnabled ? "关闭" : "开启"}拖拽线
      </Menu.Item>
      <Menu.Item onClick={() => handleFilterToggle("brightness")}>
        {filterStates.brightnessEnabled ? "关闭" : "开启"}亮度调整
      </Menu.Item>
      <Menu.Item onClick={() => handleFilterToggle("contrast")}>
        {filterStates.contrastEnabled ? "关闭" : "开启"}对比度调整
      </Menu.Item>
      <Menu.Item onClick={() => handleFilterToggle("saturate")}>
        {filterStates.saturateEnabled ? "关闭" : "开启"}饱和度调整
      </Menu.Item>
      <Menu.Item onClick={onSelectFromDataset}>从现有数据集中选择</Menu.Item>
      <Menu.Item onClick={onEval}>评估结果</Menu.Item>
    </Menu>
  );

  return (
    <div className={styles.sidebarContainer}>
      <Card className={styles.sidebarCard}>
        <Typography.Title level={3} className={styles.title}>
          {title}
        </Typography.Title>
        <Typography.Text className={styles.description}>
          {description}
        </Typography.Text>

        <div className={styles.buttonGroup}>
          <Dropdown overlay={renderMoreMenu} trigger={["click"]}>
            <Button>
              更多功能 <DownOutlined />
            </Button>
          </Dropdown>
          <Button onClick={onReset}>清除结果</Button>
        </div>

        {children}

        <div className={styles.generateButtonGroup}>
          <Button
            type="primary"
            disabled={disableGenerate}
            loading={loading}
            onClick={onGenerate}
          >
            {loading ? "正在生成" : "立即生成"}
          </Button>
          <Button onClick={onEval}>评估结果</Button>
        </div>

        <Form className={styles.form}>
          {magnifierEnabled && !disableMore && (
            <div className={styles.magnifierOptions}>
              <Form.Item label="放大镜形状">
                <Radio.Group
                  value={magnifierShape}
                  onChange={(e) =>
                    handleMagnifierChange("shape", e.target.value)
                  }
                >
                  <Radio value="square">正方形</Radio>
                  <Radio value="circle">圆形</Radio>
                </Radio.Group>
              </Form.Item>
              <Form.Item label="放大倍数">
                <Slider
                  min={2}
                  max={20}
                  value={magnifierZoomLevel}
                  onChange={(v) => handleMagnifierChange("zoomLevel", v)}
                />
              </Form.Item>
              <Form.Item label="放大镜宽度">
                <Slider
                  min={100}
                  max={1000}
                  value={magnifierWidth}
                  onChange={(v) => handleMagnifierChange("width", v)}
                />
              </Form.Item>
              <Form.Item label="放大镜高度">
                <Slider
                  min={100}
                  max={1000}
                  value={magnifierHeight}
                  onChange={(v) => handleMagnifierChange("height", v)}
                />
              </Form.Item>
            </div>
          )}

          {filterStates.brightnessEnabled && !disableMore && (
            <Form.Item label="亮度">
              <Slider
                min={-100}
                max={100}
                value={brightness}
                onChange={(v) => handleFilterChange("brightness", v)}
              />
            </Form.Item>
          )}

          {filterStates.contrastEnabled && !disableMore && (
            <Form.Item label="对比度">
              <Slider
                min={-100}
                max={100}
                value={contrast}
                onChange={(v) => handleFilterChange("contrast", v)}
              />
            </Form.Item>
          )}

          {filterStates.saturateEnabled && !disableMore && (
            <Form.Item label="饱和度">
              <Slider
                min={-100}
                max={100}
                value={saturate}
                onChange={(v) => handleFilterChange("saturate", v)}
              />
            </Form.Item>
          )}
        </Form>
      </Card>
    </div>
  );
};

export default AlgorithmToolBar;
