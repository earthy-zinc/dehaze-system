import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ExportDialog from "../ExportDialog";

const { mockExportData, mockDownloadExportBlob, mockMessage } = vi.hoisted(
  () => ({
    mockExportData: vi.fn(),
    mockDownloadExportBlob: vi.fn(),
    mockMessage: {
      success: vi.fn(),
      warning: vi.fn(),
      error: vi.fn(),
      info: vi.fn(),
    },
  })
);

vi.mock("@/hooks/useImportExport", () => ({
  useImportExport: () => ({
    exportLoading: false,
    importLoading: false,
    templateLoading: false,
    downloadTemplate: vi.fn(),
    exportData: mockExportData,
    downloadExportBlob: mockDownloadExportBlob,
    importData: vi.fn(),
  }),
}));

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  return {
    ...actual,
    message: mockMessage,
  };
});

const renderDialog = (props: Record<string, any> = {}) =>
  render(
    <ExportDialog
      open={true}
      module="user"
      queryParams={{}}
      onClose={vi.fn()}
      {...props}
    />
  );

describe("ExportDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("dialog 打开时渲染标题与文件格式选项", () => {
    renderDialog();
    expect(screen.getByText("导出数据")).not.toBeNull();
    expect(screen.getByText("Excel (.xlsx)")).not.toBeNull();
    expect(screen.getByText("CSV (.csv)")).not.toBeNull();
  });

  it("渲染异步导出开关", () => {
    renderDialog();
    expect(screen.getByText("异步导出")).not.toBeNull();
    expect(screen.getByText(/强制走异步任务/)).not.toBeNull();
  });

  it("module 为 role 时也正常渲染", () => {
    renderDialog({ module: "role" });
    expect(screen.getByText("导出数据")).not.toBeNull();
  });

  it("点击确定导出按钮调用 exportData (同步)", async () => {
    const blob = new Blob(["test"], { type: "text/csv" });
    mockExportData.mockResolvedValueOnce({ isAsync: false, blob });
    renderDialog();
    fireEvent.click(screen.getByText("确定导出"));
    await waitFor(() => {
      expect(mockExportData).toHaveBeenCalled();
    });
  });

  it("异步导出返回 taskId 时显示成功消息并关闭", async () => {
    mockExportData.mockResolvedValueOnce({ isAsync: true, taskId: "task-123" });
    const onClose = vi.fn();
    renderDialog({ onClose });
    fireEvent.click(screen.getByText("确定导出"));
    await waitFor(() => {
      expect(mockMessage.success).toHaveBeenCalled();
      expect(onClose).toHaveBeenCalled();
    });
  });

  it("导出失败时调用 message.error", async () => {
    mockExportData.mockRejectedValueOnce(new Error("导出失败"));
    renderDialog();
    fireEvent.click(screen.getByText("确定导出"));
    await waitFor(() => {
      expect(mockMessage.error).toHaveBeenCalledWith("导出失败");
    });
  });

  it("open=false 时不渲染内容", () => {
    renderDialog({ open: false });
    expect(screen.queryByText("导出数据")).toBeNull();
  });

  it("开启异步导出开关后调用 exportData 带 forceAsync", async () => {
    const blob = new Blob(["test"], { type: "text/csv" });
    mockExportData.mockResolvedValueOnce({ isAsync: false, blob });
    renderDialog();
    const switchInput = screen.getByRole("switch");
    fireEvent.click(switchInput);
    fireEvent.click(screen.getByText("确定导出"));
    await waitFor(() => {
      expect(mockExportData).toHaveBeenCalledWith("excel", undefined, true);
    });
  });

  it("切换格式为 CSV 后导出调用 format=csv", async () => {
    const blob = new Blob(["test"], { type: "text/csv" });
    mockExportData.mockResolvedValueOnce({ isAsync: false, blob });
    renderDialog();
    fireEvent.click(screen.getByText("CSV (.csv)"));
    fireEvent.click(screen.getByText("确定导出"));
    await waitFor(() => {
      expect(mockExportData).toHaveBeenCalledWith("csv", undefined, false);
    });
  });
});
