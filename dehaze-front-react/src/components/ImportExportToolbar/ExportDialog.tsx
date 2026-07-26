import { useImportExport } from "@/hooks/useImportExport";
import { DownloadOutlined } from "@ant-design/icons";
import {
  Alert,
  Button,
  Divider,
  Modal,
  Radio,
  Space,
  Switch,
  Typography,
  message,
} from "antd";
import React, { useEffect, useState } from "react";
import type { ExportDialogProps } from "./types";

const { Text } = Typography;

const ExportDialog: React.FC<ExportDialogProps> = ({
  open,
  module,
  queryParams,
  initialFormat,
  onClose,
}) => {
  const { exportLoading, exportData, downloadExportBlob } = useImportExport({
    module,
    queryParams,
  });

  const [format, setFormat] = useState<"excel" | "csv">("excel");
  const [forceAsync, setForceAsync] = useState(false);
  const [asyncTaskId, setAsyncTaskId] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setFormat(initialFormat ?? "excel");
      setForceAsync(false);
      setAsyncTaskId(null);
    }
  }, [open, initialFormat]);

  const handleClose = () => {
    setForceAsync(false);
    setAsyncTaskId(null);
    onClose();
  };

  const handleSubmit = async () => {
    try {
      const result = await exportData(format, undefined, forceAsync);
      if (result.isAsync) {
        setAsyncTaskId(result.taskId);
        message.success("数据量较大，已创建导出任务，可在任务列表查看进度");
        handleClose();
      } else {
        downloadExportBlob(result.blob, format);
        message.success("导出成功");
        handleClose();
      }
    } catch (error: unknown) {
      message.error((error as Error)?.message || "导出失败");
    }
  };

  return (
    <Modal
      title="导出数据"
      open={open}
      onCancel={handleClose}
      width={560}
      destroyOnHidden
      footer={
        <Space>
          <Button onClick={handleClose}>{asyncTaskId ? "关闭" : "取消"}</Button>
          {!asyncTaskId && (
            <Button
              type="primary"
              icon={<DownloadOutlined />}
              loading={exportLoading}
              onClick={handleSubmit}
            >
              确定导出
            </Button>
          )}
        </Space>
      }
    >
      {!asyncTaskId && (
        <Space direction="vertical" size="middle" style={{ width: "100%" }}>
          <div>
            <Text strong>文件格式</Text>
            <Radio.Group
              value={format}
              onChange={(e) => setFormat(e.target.value)}
              style={{ marginLeft: 12 }}
            >
              <Radio value="excel">Excel (.xlsx)</Radio>
              <Radio value="csv">CSV (.csv)</Radio>
            </Radio.Group>
          </div>

          <Divider style={{ margin: "8px 0" }} />

          <div>
            <Space align="center">
              <Text strong>异步导出</Text>
              <Switch checked={forceAsync} onChange={setForceAsync} />
            </Space>
            <div style={{ marginTop: 4 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                开启后强制走异步任务，适用于大数据量导出（单次最多 10 万条）
              </Text>
            </div>
          </div>
        </Space>
      )}

      {asyncTaskId && (
        <Alert
          type="info"
          message="已创建导出任务"
          description="数据量较大，已转为异步任务，可在任务列表查看进度"
          showIcon
        />
      )}
    </Modal>
  );
};

export default ExportDialog;
