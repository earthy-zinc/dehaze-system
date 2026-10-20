import { render, screen, fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ImportExportToolbar from "../index";
import React from "react";

const mockDownloadTemplate = vi.fn();
const mockUseImportExport = vi.fn();

vi.mock("@/hooks/useImportExport", () => ({
  useImportExport: (...args: unknown[]) => {
    mockUseImportExport(...args);
    return {
      exportLoading: false,
      importLoading: false,
      templateLoading: false,
      downloadTemplate: mockDownloadTemplate,
      exportData: vi.fn(),
      downloadExportBlob: vi.fn(),
      importData: vi.fn(),
    };
  },
}));

vi.mock("../ImportDialog", () => ({
  __esModule: true,
  default: ({ open, module, onClose, onImportComplete }: any) =>
    open ? (
      <div data-testid="import-dialog" data-module={module}>
        <button onClick={onClose}>close-import</button>
        <button onClick={onImportComplete}>complete-import</button>
      </div>
    ) : null,
}));

vi.mock("../ExportDialog", () => ({
  __esModule: true,
  default: ({ open, module, queryParams, initialFormat, onClose }: any) =>
    open ? (
      <div
        data-testid="export-dialog"
        data-module={module}
        data-format={initialFormat}
      >
        <button onClick={onClose}>close-export</button>
      </div>
    ) : null,
}));

vi.mock("../TaskListDrawer", () => ({
  __esModule: true,
  default: ({ open, module, onClose }: any) =>
    open ? (
      <div data-testid="task-drawer" data-module={module}>
        <button onClick={onClose}>close-drawer</button>
      </div>
    ) : null,
}));

describe("ImportExportToolbar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("渲染导入、导出、任务列表按钮", () => {
    render(<ImportExportToolbar module="user" queryParams={{}} />);
    expect(screen.getByText("导入")).not.toBeNull();
    expect(screen.getByText("导出")).not.toBeNull();
    expect(screen.getByText("任务列表")).not.toBeNull();
  });

  it("importable=false 时不渲染导入按钮", () => {
    render(
      <ImportExportToolbar module="user" queryParams={{}} importable={false} />
    );
    expect(screen.queryByText("导入")).toBeNull();
    expect(screen.getByText("导出")).not.toBeNull();
  });

  it("importable=true 时渲染导入按钮", () => {
    render(
      <ImportExportToolbar module="user" queryParams={{}} importable={true} />
    );
    expect(screen.getByText("导入")).not.toBeNull();
  });

  it("点击导入按钮打开 ImportDialog", () => {
    render(<ImportExportToolbar module="user" queryParams={{}} />);
    fireEvent.click(screen.getByText("导入"));
    expect(screen.getByTestId("import-dialog")).not.toBeNull();
  });

  it("点击任务列表按钮打开 TaskListDrawer", () => {
    render(<ImportExportToolbar module="user" queryParams={{}} />);
    fireEvent.click(screen.getByText("任务列表"));
    expect(screen.getByTestId("task-drawer")).not.toBeNull();
  });

  it("传递 module 给 useImportExport", () => {
    render(<ImportExportToolbar module="role" queryParams={{}} />);
    expect(mockUseImportExport).toHaveBeenCalledWith(
      expect.objectContaining({ module: "role" })
    );
  });

  it("传递 queryParams 给 useImportExport", () => {
    render(
      <ImportExportToolbar module="user" queryParams={{ keywords: "admin" }} />
    );
    expect(mockUseImportExport).toHaveBeenCalledWith(
      expect.objectContaining({ queryParams: { keywords: "admin" } })
    );
  });

  it("ImportDialog 完成后调用 onImportComplete 回调", () => {
    const onImportComplete = vi.fn();
    render(
      <ImportExportToolbar
        module="user"
        queryParams={{}}
        onImportComplete={onImportComplete}
      />
    );
    fireEvent.click(screen.getByText("导入"));
    fireEvent.click(screen.getByText("complete-import"));
    expect(onImportComplete).toHaveBeenCalled();
  });

  it("传递 extraImportParams 给 useImportExport", () => {
    render(
      <ImportExportToolbar
        module="user"
        queryParams={{}}
        extraImportParams={{ deptId: 1 }}
      />
    );
    expect(mockUseImportExport).toHaveBeenCalledWith(
      expect.objectContaining({ extraImportParams: { deptId: 1 } })
    );
  });
});
