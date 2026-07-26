import {
  DownloadOutlined,
  DownOutlined,
  ImportOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { Button, Dropdown, Space, type MenuProps } from "antd";
import React, { useState } from "react";
import type { ExportModule, ImportModule } from "dehaze-sdk-js";
import { useImportExport } from "@/hooks/useImportExport";
import ExportDialog from "./ExportDialog";
import ImportDialog from "./ImportDialog";
import TaskListDrawer from "./TaskListDrawer";
import type { ImportExportToolbarProps } from "./types";

const ImportExportToolbar: React.FC<ImportExportToolbarProps> = ({
  module,
  importable = true,
  queryParams,
  extraImportParams,
  onImportComplete,
}) => {
  const { templateLoading, downloadTemplate } = useImportExport({
    module,
    queryParams,
    extraImportParams,
  });

  const [importDialogVisible, setImportDialogVisible] = useState(false);
  const [exportDialogVisible, setExportDialogVisible] = useState(false);
  const [exportInitialFormat, setExportInitialFormat] = useState<
    "excel" | "csv"
  >("excel");
  const [taskDrawerVisible, setTaskDrawerVisible] = useState(false);

  const templateMenuItems: MenuProps["items"] = [
    { key: "excel", label: "下载 Excel 模板", icon: <DownloadOutlined /> },
    { key: "csv", label: "下载 CSV 模板", icon: <DownloadOutlined /> },
  ];

  const exportMenuItems: MenuProps["items"] = [
    { key: "excel", label: "导出为 Excel" },
    { key: "csv", label: "导出为 CSV" },
  ];

  const handleTemplateMenuClick: MenuProps["onClick"] = ({ key }) => {
    downloadTemplate(key as "excel" | "csv");
  };

  const handleExportMenuClick: MenuProps["onClick"] = ({ key }) => {
    setExportInitialFormat(key as "excel" | "csv");
    setExportDialogVisible(true);
  };

  return (
    <Space>
      {importable && (
        <Space.Compact>
          <Button
            icon={<ImportOutlined />}
            onClick={() => setImportDialogVisible(true)}
          >
            导入
          </Button>
          <Dropdown
            menu={{
              items: templateMenuItems,
              onClick: handleTemplateMenuClick,
            }}
          >
            <Button
              icon={<DownOutlined />}
              loading={templateLoading}
              style={{ paddingInline: 8 }}
            />
          </Dropdown>
        </Space.Compact>
      )}
      <Dropdown
        menu={{ items: exportMenuItems, onClick: handleExportMenuClick }}
      >
        <Button icon={<DownloadOutlined />}>
          导出 <DownOutlined />
        </Button>
      </Dropdown>
      <Button
        icon={<UploadOutlined />}
        onClick={() => setTaskDrawerVisible(true)}
      >
        任务列表
      </Button>

      {importable && (
        <ImportDialog
          open={importDialogVisible}
          module={module as ImportModule}
          extraImportParams={extraImportParams}
          onClose={() => setImportDialogVisible(false)}
          onImportComplete={() => {
            setImportDialogVisible(false);
            onImportComplete?.();
          }}
        />
      )}

      <ExportDialog
        open={exportDialogVisible}
        module={module as ExportModule}
        queryParams={queryParams}
        initialFormat={exportInitialFormat}
        onClose={() => setExportDialogVisible(false)}
      />

      <TaskListDrawer
        open={taskDrawerVisible}
        module={module}
        onClose={() => setTaskDrawerVisible(false)}
      />
    </Space>
  );
};

export default ImportExportToolbar;
