import AlgorithmHeader from "@/components/AlgorithmToolBar/AlgorithmHeader";
import ImageUploadButton from "@/components/AlgorithmToolBar/ImageUploadButton";
import SubmitButton from "@/components/AlgorithmToolBar/SubmitButton";

import { MagnifierInfo } from "@/components/AlgorithmToolBar/types";
import { Card } from "antd";
import React from "react";
import "./index.scss";

interface AlgorithmToolBarProps {
  title?: string;
  description?: string;
  disableMore: boolean;
  magnifier: MagnifierInfo;
  onUpload: (file: File) => void;
  onTakePhoto: () => void;
  onReset: () => void;
  onGenerate: () => void;
  onMagnifierChange: () => void;
  onBrightnessChange: (value: number) => void;
  onContrastChange: (value: number) => void;
  children: React.ReactNode;
}

const AlgorithmToolBar: React.FC<AlgorithmToolBarProps> = ({
  title = "图像去雾",
  description = "通过对遭受雾气影响的图像进行相应处理，恢复图像原本的纹理结构和细节信息，进而提升图像的能见度",
  disableMore,
  magnifier,
  onUpload,
  onTakePhoto,
  onReset,
  onGenerate,
  onMagnifierChange,
  onBrightnessChange,
  onContrastChange,
  children,
}) => {
  return (
    <div className="mr-3">
      <Card className="sidebar-card">
        <AlgorithmHeader title={title} description={description} />
        <ImageUploadButton
          onUpload={onUpload}
          onTakePhoto={onTakePhoto}
          onReset={onReset}
        />
        {children}
        <SubmitButton
          disableMore={disableMore}
          magnifier={magnifier}
          onGenerate={onGenerate}
          onMagnifierChange={onMagnifierChange}
          onBrightnessChange={onBrightnessChange}
          onContrastChange={onContrastChange}
        />
      </Card>
    </div>
  );
};

export default AlgorithmToolBar;
