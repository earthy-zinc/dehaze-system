import { downloadBlob, useImportExport } from "@/hooks/useImportExport";
import {
  DownloadOutlined,
  InboxOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Modal,
  Radio,
  Space,
  Statistic,
  Table,
  Typography,
  Upload,
  type UploadFile,
  type UploadProps,
  message,
} from "antd";
import React, { useMemo, useState } from "react";
import type { ImportError, ImportModule, ImportResult } from "dehaze-sdk-js";
import { MODULE_LABEL_MAP } from "./types";

const { Dragger } = Upload;
const { Text } = Typography;

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ACCEPT_EXTENSIONS = [".xlsx", ".xls", ".csv"];

interface ImportDialogProps {
  open: boolean;
  module: ImportModule;
  extraImportParams?: Record<string, unknown>;
  onClose: () => void;
  onImportComplete: () => void;
}

const ImportDialog: React.FC<ImportDialogProps> = ({
  open,
  module,
  extraImportParams,
  onClose,
  onImportComplete,
}) => {
  const { importLoading, downloadTemplate, importData } = useImportExport({
    module: module,
    queryParams: {},
    extraImportParams,
  });

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [importMode, setImportMode] = useState<"all" | "partial">("all");
  const [syncResult, setSyncResult] = useState<ImportResult | null>(null);
  const [asyncTaskId, setAsyncTaskId] = useState<string | null>(null);
  const [errorReportLoading, setErrorReportLoading] = useState(false);

  const dialogTitle = `导入${MODULE_LABEL_MAP[module] ?? module}`;

  const resetState = () => {
    setSelectedFile(null);
    setImportMode("all");
    setSyncResult(null);
    setAsyncTaskId(null);
  };

  const handleClose = () => {
    resetState();
    onClose();
  };

  const validateFile = (file: File): string | null => {
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPT_EXTENSIONS.includes(ext)) {
      return "仅支持 .xlsx、.xls、.csv 格式文件";
    }
    if (file.size > MAX_FILE_SIZE) {
      return "文件大小不能超过 20MB";
    }
    return null;
  };

  const draggerProps: UploadProps = {
    accept: ".xlsx,.xls,.csv",
    multiple: false,
    maxCount: 1,
    fileList: selectedFile
      ? [
          {
            uid: selectedFile.name,
            name: selectedFile.name,
            status: "done",
          } as UploadFile,
        ]
      : [],
    beforeUpload: (file) => {
      const error = validateFile(file);
      if (error) {
        message.warning(error);
        setSelectedFile(null);
        return Upload.LIST_IGNORE;
      }
      setSelectedFile(file);
      setSyncResult(null);
      setAsyncTaskId(null);
      return false;
    },
    onRemove: () => {
      setSelectedFile(null);
    },
  };

  const handleDownloadTemplate = (format: "excel" | "csv") => {
    downloadTemplate(format);
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      message.warning("请先选择文件");
      return;
    }
    try {
      const result = await importData(selectedFile, importMode);
      if (result.isAsync) {
        setAsyncTaskId(result.taskId);
        message.success("数据量较大，已创建导入任务，可在任务列表查看进度");
        onImportComplete();
      } else {
        setSyncResult(result.result);
        message.success("导入完成");
        onImportComplete();
      }
    } catch (error: unknown) {
      message.error((error as Error)?.message || "导入失败");
    }
  };

  const handleDownloadErrorReport = async () => {
    if (!syncResult?.errorReportUrl) {
      message.warning("无错误报告可下载");
      return;
    }
    setErrorReportLoading(true);
    try {
      const res = await fetch(syncResult.errorReportUrl);
      if (!res.ok) throw new Error("下载失败");
      const blob = await res.blob();
      downloadBlob(blob, "import_error_report.xlsx");
    } catch (error: unknown) {
      message.error((error as Error)?.message || "错误报告下载失败");
    } finally {
      setErrorReportLoading(false);
    }
  };

  const errorColumns = useMemo(
    () => [
      {
        title: "行号",
        dataIndex: "row",
        key: "row",
        width: 80,
        align: "center" as const,
      },
      { title: "字段", dataIndex: "field", key: "field", width: 140 },
      {
        title: "错误信息",
        dataIndex: "message",
        key: "message",
        ellipsis: true,
      },
    ],
    []
  );

  const hasErrors = !!syncResult && syncResult.failureCount > 0;

  return (
    <Modal
      title={dialogTitle}
      open={open}
      onCancel={handleClose}
      width={640}
      destroyOnHidden
      footer={
        <Space>
          <Button onClick={handleClose}>
            {syncResult || asyncTaskId ? "关闭" : "取消"}
          </Button>
          {!syncResult && !asyncTaskId && (
            <Button
              type="primary"
              icon={<UploadOutlined />}
              loading={importLoading}
              disabled={!selectedFile}
              onClick={handleSubmit}
            >
              确定导入
            </Button>
          )}
        </Space>
      }
    >
      {!syncResult && !asyncTaskId && (
        <Space direction="vertical" size="middle" style={{ width: "100%" }}>
          <div>
            <Text strong>导入模式</Text>
            <Radio.Group
              value={importMode}
              onChange={(e) => setImportMode(e.target.value)}
              style={{ marginLeft: 12 }}
            >
              <Radio value="all">全量导入</Radio>
              <Radio value="partial">部分导入</Radio>
            </Radio.Group>
            <div style={{ marginTop: 4 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {importMode === "all"
                  ? "全量导入：覆盖更新已存在的记录，新增不存在的记录"
                  : "部分导入：仅新增不存在的记录，已存在的记录跳过"}
              </Text>
            </div>
          </div>

          <div>
            <Text strong>下载模板</Text>
            <Space style={{ marginLeft: 12 }}>
              <Button
                size="small"
                icon={<DownloadOutlined />}
                onClick={() => handleDownloadTemplate("excel")}
              >
                Excel 模板
              </Button>
              <Button
                size="small"
                icon={<DownloadOutlined />}
                onClick={() => handleDownloadTemplate("csv")}
              >
                CSV 模板
              </Button>
            </Space>
          </div>

          <Dragger {...draggerProps} style={{ padding: 16 }}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">将文件拖到此处，或点击上传</p>
            <p className="ant-upload-hint">
              支持 .xlsx、.xls、.csv 格式，文件大小 ≤ 20MB
            </p>
          </Dragger>
        </Space>
      )}

      {syncResult && (
        <Space direction="vertical" size="middle" style={{ width: "100%" }}>
          <Alert
            type={hasErrors ? "warning" : "success"}
            message={
              hasErrors
                ? `导入完成，${syncResult.failureCount} 条数据失败`
                : "导入完成"
            }
            showIcon
          />
          <Space
            size="large"
            style={{ justifyContent: "space-around", width: "100%" }}
          >
            <Statistic title="总行数" value={syncResult.totalRows} />
            <Statistic
              title="成功"
              value={syncResult.successCount}
              valueStyle={{ color: "#3f8600" }}
            />
            <Statistic
              title="失败"
              value={syncResult.failureCount}
              valueStyle={{ color: "#cf1322" }}
            />
            <Statistic
              title="跳过"
              value={syncResult.skippedCount}
              valueStyle={{ color: "#d48806" }}
            />
          </Space>
          {syncResult.errorReportUrl && (
            <div style={{ textAlign: "center" }}>
              <Button
                type="primary"
                ghost
                icon={<DownloadOutlined />}
                loading={errorReportLoading}
                onClick={handleDownloadErrorReport}
              >
                下载错误报告
              </Button>
            </div>
          )}
          {syncResult.errors && syncResult.errors.length > 0 && (
            <Table<ImportError>
              columns={errorColumns}
              dataSource={syncResult.errors}
              rowKey={(record, index) => `${record.row}-${index}`}
              size="small"
              pagination={{ pageSize: 5 }}
              scroll={{ y: 240 }}
            />
          )}
        </Space>
      )}

      {asyncTaskId && (
        <Alert
          type="info"
          message="已创建导入任务"
          description="数据量较大，已转为异步任务，可在任务列表查看进度"
          showIcon
        />
      )}
    </Modal>
  );
};

export default ImportDialog;
