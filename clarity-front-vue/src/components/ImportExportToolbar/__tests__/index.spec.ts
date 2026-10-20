import { mount } from "@vue/test-utils";
import { ref } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ImportExportToolbar from "../index.vue";

vi.mock("element-plus/es", () => ({
  ElMessage: {
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
  ElMessageBox: { confirm: vi.fn() },
  ElButton: { name: "ElButton", template: "<div />" },
  ElDropdown: { name: "ElDropdown", template: "<div />" },
  ElDropdownMenu: { name: "ElDropdownMenu", template: "<div />" },
  ElDropdownItem: { name: "ElDropdownItem", template: "<div />" },
  ElIcon: { name: "ElIcon", template: "<span />" },
}));

const mockDownloadTemplate = vi.fn();
const mockExportData = vi.fn();
const mockImportData = vi.fn();
const mockDownloadExportBlob = vi.fn();

vi.mock("@/composables/useImportExport", () => ({
  useImportExport: () => ({
    exportLoading: ref(false),
    importLoading: ref(false),
    templateLoading: ref(false),
    downloadTemplate: mockDownloadTemplate,
    exportData: mockExportData,
    downloadExportBlob: mockDownloadExportBlob,
    importData: mockImportData,
  }),
  downloadBlob: vi.fn(),
  buildFileName: vi.fn((p: string, f: string) => `${p}.${f}`),
}));

vi.mock("@element-plus/icons-vue", () => ({
  ArrowDown: { template: "<i />" },
  Download: { template: "<i />" },
  Top: { template: "<i />" },
  Upload: { template: "<i />" },
}));

const elStubs = {
  "el-button": {
    template: "<button @click=\"$emit('click')\"><slot /></button>",
    props: ["type", "loading", "disabled", "size", "plain"],
    emits: ["click"],
  },
  "el-dropdown": {
    template: '<div><slot /><slot name="dropdown" /></div>',
    props: ["trigger"],
    emits: ["command"],
  },
  "el-dropdown-menu": { template: "<div><slot /></div>" },
  "el-dropdown-item": {
    template: "<div @click=\"$emit('click')\"><slot /></div>",
    props: ["command"],
    emits: ["click"],
  },
  "el-icon": { template: "<span><slot /></span>" },
};

const mountToolbar = (props: Record<string, any> = {}) =>
  mount(ImportExportToolbar, {
    props: {
      module: "user",
      queryParams: {},
      ...props,
    },
    global: {
      stubs: {
        ...elStubs,
        ImportDialog: true,
        ExportDialog: true,
        TaskListDrawer: true,
      },
    },
  });

describe("ImportExportToolbar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("渲染导入、导出、任务列表按钮", () => {
    const wrapper = mountToolbar();
    expect(wrapper.text()).toContain("导入");
    expect(wrapper.text()).toContain("导出");
    expect(wrapper.text()).toContain("任务列表");
  });

  it("importable=false 时不渲染 ImportDialog 子组件", () => {
    const wrapper = mountToolbar({ importable: false });
    expect(wrapper.findComponent({ name: "ImportDialog" }).exists()).toBe(
      false
    );
  });

  it("importable=true 时渲染 ImportDialog 子组件", () => {
    const wrapper = mountToolbar({ importable: true });
    expect(wrapper.findComponent({ name: "ImportDialog" }).exists()).toBe(true);
  });

  it("点击任务列表按钮将 TaskListDrawer 的 modelValue 设为 true", async () => {
    const wrapper = mountToolbar();
    const buttons = wrapper.findAll("button");
    const taskBtn = buttons.find((b) => b.text().includes("任务列表"));
    expect(taskBtn).toBeDefined();
    await taskBtn!.trigger("click");
    const drawer = wrapper.findComponent({ name: "TaskListDrawer" });
    expect(drawer.props("modelValue")).toBe(true);
  });

  it("ImportDialog 触发 import-complete 事件后向父组件冒泡", async () => {
    const wrapper = mountToolbar();
    await wrapper
      .findComponent({ name: "ImportDialog" })
      .vm.$emit("import-complete");
    expect(wrapper.emitted("import-complete")).toBeTruthy();
  });

  it("传递 module 和 queryParams 给 ExportDialog", () => {
    const wrapper = mountToolbar({
      module: "role",
      queryParams: { keywords: "admin" },
    });
    const exportDialog = wrapper.findComponent({ name: "ExportDialog" });
    expect(exportDialog.props("module")).toBe("role");
    expect(exportDialog.props("queryParams")).toEqual({ keywords: "admin" });
  });

  it("传递 module 给 TaskListDrawer", () => {
    const wrapper = mountToolbar({ module: "dict" });
    const drawer = wrapper.findComponent({ name: "TaskListDrawer" });
    expect(drawer.props("module")).toBe("dict");
  });
});
