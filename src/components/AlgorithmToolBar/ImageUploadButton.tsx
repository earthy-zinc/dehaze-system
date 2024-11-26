import { ClearOutlined, UploadOutlined, UserOutlined } from "@ant-design/icons";
import { Button, Dropdown, Upload, UploadFile } from "antd";
import { UploadChangeParam } from "antd/es/upload";
import React from "react";

interface ImageUploadButtonProps {
  onUpload: (file: File) => void;
  onTakePhoto: () => void;
  onReset: () => void;
}

const ImageUploadButton: React.FC<ImageUploadButtonProps> = ({
  onUpload,
  onTakePhoto,
  onReset,
}) => {
  const menuProps = {
    items: [
      {
        label: "拍照上传",
        key: "1",
        icon: <UserOutlined />,
      },
    ],
    onClick: onTakePhoto,
  };
  const handleUploadChange = (info: UploadChangeParam<UploadFile>) => {
    const fileList = [...info.fileList];
    const file = fileList.slice(-1);
    if (file[0].status === "done") {
      onUpload(file[0].originFileObj as File);
    }
  };
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-evenly",
        margin: "16px",
      }}
    >
      <Dropdown.Button type="primary" menu={menuProps}>
        <Upload onChange={handleUploadChange}>
          <UploadOutlined
            style={{
              marginRight: "6px",
              color: "var(--ant-button-primary-color)",
            }}
          />
          <span style={{ color: "var(--ant-button-primary-color)" }}>
            本地上传
          </span>
        </Upload>
      </Dropdown.Button>
      <Button icon={<ClearOutlined />} onClick={onReset}>
        清除结果
      </Button>
    </div>
  );
};

export default ImageUploadButton;
