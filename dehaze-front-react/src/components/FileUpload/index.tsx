import { FileAPI, type FileInfo } from "dehaze-sdk-js";
import { InboxOutlined } from "@ant-design/icons";
import { message, Modal, Upload } from "antd";
import type { UploadFile, UploadProps } from "antd";
import React, { useRef, useState } from "react";

import { calculateFileMd5 } from "@/utils/md5";

const { Dragger } = Upload;

/** 文件大小上限：100MB */
const MAX_FILE_SIZE = 100 * 1024 * 1024;

export interface FileUploadProps {
  /** 接受的文件类型，例如 "image/*" */
  accept?: string;
  /** 最大上传数量，默认 1 */
  maxCount?: number;
  /** 模型ID，上传时关联模型 */
  modelId?: number;
  /** 受控值：已上传的文件信息列表 */
  value?: FileInfo[];
  /** 文件列表变化回调 */
  onChange?: (files: FileInfo[]) => void;
}

interface UploadingItem {
  uid: string;
  name: string;
  percent: number;
}

/**
 * 通用文件上传组件
 *
 * 支持：拖拽上传、点击选择、文件大小校验（100MB）、格式校验（accept）、
 * MD5 秒传、上传进度显示、删除前确认。
 */
const FileUpload: React.FC<FileUploadProps> = ({
  accept,
  maxCount = 1,
  modelId,
  value = [],
  onChange,
}) => {
  // 上传中的文件（uid + 名称 + 进度）
  const [uploadingItems, setUploadingItems] = useState<UploadingItem[]>([]);
  // 已取消上传的 uid 集合（用于阻止已移除文件的上传完成后写入）
  const cancelledUids = useRef<Set<string>>(new Set());
  // value 的最新引用，供异步回调使用
  const valueRef = useRef(value);
  valueRef.current = value;

  // 由受控值 + 上传中文件 组合出 Upload 可用的 fileList
  const fileList: UploadFile[] = [
    ...value.map<UploadFile>((info) => ({
      uid: String(info.id),
      name: info.name,
      status: "done",
      url: info.url,
    })),
    ...uploadingItems.map<UploadFile>((item) => ({
      uid: item.uid,
      name: item.name,
      status: "uploading",
      percent: item.percent,
    })),
  ];

  // 上传前校验文件大小
  const beforeUpload: UploadProps["beforeUpload"] = (file) => {
    if (file.size > MAX_FILE_SIZE) {
      message.error("文件大小不能超过 100MB");
      return Upload.LIST_IGNORE;
    }
    return true;
  };

  // 自定义上传：MD5 秒传 + 进度上传
  const handleCustomRequest: UploadProps["customRequest"] = async (options) => {
    const { file, onSuccess, onError } = options;
    const rawFile = file as File;
    const uid = (file as { uid?: string }).uid || String(Date.now());

    setUploadingItems((prev) => [
      ...prev,
      { uid, name: rawFile.name, percent: 0 },
    ]);

    try {
      // 计算文件哈希进行秒传校验
      const md5 = await calculateFileMd5(rawFile);
      const existing = await FileAPI.uploadCheck(md5);

      let fileInfo: FileInfo;
      if (existing) {
        // 秒传命中，直接复用已有文件
        fileInfo = existing;
        message.success("文件秒传成功");
      } else {
        // 未命中，执行实际上传
        fileInfo = await FileAPI.upload(rawFile, modelId, (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round(
              (progressEvent.loaded / progressEvent.total) * 100
            );
            setUploadingItems((prev) =>
              prev.map((item) =>
                item.uid === uid ? { ...item, percent } : item
              )
            );
          }
        });
      }

      // 若上传过程中已被移除，则丢弃结果
      if (cancelledUids.current.has(uid)) {
        cancelledUids.current.delete(uid);
        return;
      }

      setUploadingItems((prev) => prev.filter((item) => item.uid !== uid));
      onSuccess?.(fileInfo, rawFile);
      const newFiles = [...valueRef.current, fileInfo].slice(-maxCount);
      onChange?.(newFiles);
    } catch (err) {
      setUploadingItems((prev) => prev.filter((item) => item.uid !== uid));
      onError?.(err as any);
      message.error("文件上传失败");
    }
  };

  // 删除文件前确认
  const handleRemove: UploadProps["onRemove"] = (file) => {
    const fileInfo = valueRef.current.find(
      (info) => String(info.id) === file.uid
    );

    // 上传中的文件：直接移除展示，并标记取消
    if (!fileInfo) {
      cancelledUids.current.add(file.uid);
      setUploadingItems((prev) => prev.filter((item) => item.uid !== file.uid));
      return false;
    }

    // 已上传文件：弹窗确认后调用删除接口
    Modal.confirm({
      title: "确认删除",
      content: "确认删除该文件吗？删除后不可恢复。",
      okText: "确认",
      cancelText: "取消",
      onOk: async () => {
        try {
          await FileAPI.deleteById(fileInfo.id);
          onChange?.(
            valueRef.current.filter((info) => String(info.id) !== file.uid)
          );
        } catch {
          message.error("删除失败");
        }
      },
    });
    return false;
  };

  const draggerProps: UploadProps = {
    accept,
    maxCount,
    fileList,
    beforeUpload,
    customRequest: handleCustomRequest,
    onRemove: handleRemove,
    multiple: maxCount > 1,
  };

  return (
    <Dragger {...draggerProps}>
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
      <p className="ant-upload-hint">
        支持单个或批量上传，文件大小不超过 100MB
      </p>
    </Dragger>
  );
};

export default FileUpload;
