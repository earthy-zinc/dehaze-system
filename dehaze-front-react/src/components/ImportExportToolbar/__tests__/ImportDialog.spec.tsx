import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ImportDialog from "../ImportDialog";
import React from "react";

const { mockDownloadTemplate, mockImportData, mockMessage } = vi.hoisted(
  () => ({
    mockDownloadTemplate: vi.fn(),
    mockImportData: vi.fn(),
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
    downloadTemplate: mockDownloadTemplate,
    exportData: vi.fn(),
    downloadExportBlob: vi.fn(),
    importData: mockImportData,
  }),
  downloadBlob: vi.fn(),
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
    <ImportDialog
      open={true}
      module="user"
      onClose={vi.fn()}
      onImportComplete={vi.fn()}
      {...props}
    />
  );

describe("ImportDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("dialog 打开时渲染标题与导入模式选项", () => {
    renderDialog();
    expect(screen.getByText("导入用户")).not.toBeNull();
    expect(screen.getByText("全量导入")).not.toBeNull();
    expect(screen.getByText("部分导入")).not.toBeNull();
  });

  it("渲染模板下载按钮(Excel/CSV)", () => {
    renderDialog();
    expect(screen.getByText("Excel 模板")).not.toBeNull();
    expect(screen.getByText("CSV 模板")).not.toBeNull();
  });

  it("渲染文件上传区域", () => {
    renderDialog();
    expect(screen.getByText("将文件拖到此处，或点击上传")).not.toBeNull();
  });

  it("渲染文件格式提示", () => {
    renderDialog();
    expect(
      screen.getByText(/支持 .xlsx、.xls、.csv 格式，文件大小 ≤ 20MB/)
    ).not.toBeNull();
  });

  it("module 为 role 时标题为导入角色", () => {
    renderDialog({ module: "role" });
    expect(screen.getByText("导入角色")).not.toBeNull();
  });

  it("module 为 algorithm 时标题为导入算法", () => {
    renderDialog({ module: "algorithm" });
    expect(screen.getByText("导入算法")).not.toBeNull();
  });

  it("点击 Excel 模板按钮调用 downloadTemplate", () => {
    renderDialog();
    fireEvent.click(screen.getByText("Excel 模板"));
    expect(mockDownloadTemplate).toHaveBeenCalledWith("excel");
  });

  it("点击 CSV 模板按钮调用 downloadTemplate", () => {
    renderDialog();
    fireEvent.click(screen.getByText("CSV 模板"));
    expect(mockDownloadTemplate).toHaveBeenCalledWith("csv");
  });

  it("未选择文件时确定导入按钮禁用", () => {
    renderDialog();
    const submitBtn = screen.getByText("确定导入").closest("button");
    expect(submitBtn).not.toBeNull();
    expect(submitBtn!.disabled).toBe(true);
  });

  it("全量导入模式显示对应提示", () => {
    renderDialog();
    expect(
      screen.getByText(/全量导入：覆盖更新已存在的记录，新增不存在的记录/)
    ).not.toBeNull();
  });

  it("切换到部分导入模式显示对应提示", () => {
    renderDialog();
    fireEvent.click(screen.getByText("部分导入"));
    expect(
      screen.getByText(/部分导入：仅新增不存在的记录，已存在的记录跳过/)
    ).not.toBeNull();
  });

  it("open=false 时不渲染内容", () => {
    renderDialog({ open: false });
    expect(screen.queryByText("导入用户")).toBeNull();
  });
});
