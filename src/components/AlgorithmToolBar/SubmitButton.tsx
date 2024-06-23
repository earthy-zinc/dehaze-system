import { MagnifierInfo } from "@/components/AlgorithmToolBar/types";
import Magnifier from "@/components/Magnifier";
import { RootState } from "@/store";
import { Button, Dropdown, Form, Radio, Slider } from "antd";
import React, { useState } from "react";
import { useSelector } from "react-redux";

interface SubmitButtonProps {
  disableMore: boolean;
  magnifier: MagnifierInfo;
  onGenerate: () => void;
  onMagnifierChange: () => void;
  onBrightnessChange: (value: number) => void;
  onContrastChange: (value: number) => void;
}

const SubmitButton: React.FC<SubmitButtonProps> = ({
  disableMore = false,
  magnifier = {
    imgUrls: [],
    radius: 100,
    originScale: 1,
    point: {
      x: 0,
      y: 0,
    },
  },
  onGenerate,
  onMagnifierChange,
  onBrightnessChange,
  onContrastChange,
}) => {
  const themeColor = useSelector(
    (state: RootState) => state.settings.themeColor
  );
  const [showMagnifier, setShowMagnifier] = useState(false);
  const [showContrast, setShowContrast] = useState(false);
  const [showBrightness, setShowBrightness] = useState(false);
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(0);
  const [magnifierShape, setMagnifierShape] = useState<"square" | "circle">(
    "square"
  );
  const [magnifierScale, setMagnifierScale] = useState(8);
  const [magnifierLabels, setMagnifierLabels] = useState([
    { text: "原图", color: "white", backgroundColor: "black" },
    { text: "对比图", color: "white", backgroundColor: themeColor },
  ]);
  const [menuItems, setMenuItems] = useState([
    {
      label: "开启放大镜",
      key: "magnifier",
      active: false,
    },
    {
      label: "开启对比度",
      key: "contrast",
      active: false,
    },
    {
      label: "开启亮度",
      key: "brightness",
      active: false,
    },
  ]);
  const handleItemClick = ({ key }: { key: string }) => {
    switch (key) {
      case "magnifier":
        setShowMagnifier((show) => !show);
        onMagnifierChange();
        break;
      case "contrast":
        setShowContrast((show) => !show);
        break;
      case "brightness":
        setShowBrightness((show) => !show);
        break;
    }
    setMenuItems((items) =>
      items.map((item) =>
        item.key === key
          ? {
              ...item,
              label: item.label.includes("开启")
                ? "关闭" + item.label.slice(2)
                : "开启" + item.label.slice(2),
            }
          : item
      )
    );
  };
  const moreMenu = {
    items: menuItems,
    onClick: handleItemClick,
  };

  return (
    <>
      <div
        style={{
          display: "flex",
          justifyContent: "space-evenly",
          margin: "16px",
        }}
      >
        <Button
          type="primary"
          onClick={onGenerate}
          style={{ marginRight: "6px" }}
        >
          立即生成
        </Button>
        <Dropdown.Button menu={moreMenu} disabled={disableMore}>
          更多功能
        </Dropdown.Button>
      </div>
      {showMagnifier &&
        magnifier.imgUrls.map((url, index) => (
          <Magnifier
            src={url}
            key={index}
            label={magnifierLabels[index]}
            brightness={brightness}
            contrast={contrast}
            shape={magnifierShape}
            scale={magnifierScale}
            point={magnifier.point}
            radius={magnifier.radius}
            originScale={magnifier.originScale}
          />
        ))}
      <Form>
        {showMagnifier && (
          <>
            <Form.Item label="放大镜形状">
              <Radio.Group
                value={magnifierShape}
                onChange={(e) => setMagnifierShape(e.target.value)}
              >
                <Radio value="square">正方形</Radio>
                <Radio value="circle">圆形</Radio>
              </Radio.Group>
            </Form.Item>
            <Form.Item label="放大倍数">
              <Slider
                min={2}
                max={16}
                value={magnifierScale}
                onChange={(value) => setMagnifierScale(value)}
              />
            </Form.Item>
          </>
        )}
        {showBrightness && (
          <Form.Item label="亮度">
            <Slider
              min={-100}
              max={100}
              value={brightness}
              onChange={(value) => {
                setBrightness(value);
                onBrightnessChange(value);
              }}
            />
          </Form.Item>
        )}

        {showContrast && (
          <Form.Item label="对比度">
            <Slider
              min={-100}
              max={100}
              value={contrast}
              onChange={(value) => {
                setContrast(value);
                onContrastChange(value);
              }}
            />
          </Form.Item>
        )}
      </Form>
    </>
  );
};

export default SubmitButton;
