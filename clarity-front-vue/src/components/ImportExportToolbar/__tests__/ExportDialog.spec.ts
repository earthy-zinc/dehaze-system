import { mount } from "@vue/test-utils";
import { computed, provide, inject, ref } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ExportDialog from "../ExportDialog.vue";

const mockExportData = vi.fn();
const mockDownloadExportBlob = vi.fn();

vi.mock("@/composables/useImportExport", () => ({
  useImportExport: () => ({
    exportLoading: ref(false),
    importLoading: ref(false),
    templateLoading: ref(false),
    downloadTemplate: vi.fn(),
    exportData: mockExportData,
    downloadExportBlob: mockDownloadExportBlob,
    importData: vi.fn(),
  }),
  downloadBlob: vi.fn(),
  buildFileName: vi.fn(),
}));

const elStubs = {
  "el-dialog": {
    template: '<div><h2>{{ title }}</h2><slot /><slot name="footer" /></div>',
    props: ["modelValue", "title", "width"],
  },
  "el-form": { template: "<div><slot /></div>" },
  "el-form-item": {
    template: "<div><label>{{ label }}</label><slot /></div>",
    props: ["label"],
  },
  "el-radio-group": {
    template: "<div><slot /></div>",
    props: ["modelValue"],
    emits: ["update:modelValue"],
    setup(props: any, { emit }: { emit: (e: string, v: unknown) => void }) {
      const currentValue = computed(() => props.modelValue);
      const updateValue = (val: unknown) => emit("update:modelValue", val);
      provide("radioGroupValue", currentValue);
      provide("updateRadioGroupValue", updateValue);
    },
  },
  "el-radio": {
    template:
      '<label><input type="radio" :value="value" :checked="isChecked" @change="onChange" /><slot /></label>',
    props: ["value", "label"],
    setup(props: any) {
      const currentValue = inject("radioGroupValue", ref(null));
      const updateValue = inject("updateRadioGroupValue", (_v: unknown) => {});
      const isChecked = computed(() => currentValue.value === props.value);
      const onChange = () => updateValue(props.value);
      return { isChecked, onChange };
    },
  },
  "el-button": {
    template: "<button @click=\"$emit('click')\"><slot /></button>",
    props: ["type", "loading", "disabled", "size", "plain"],
    emits: ["click"],
  },
  "el-switch": {
    template: '<input type="checkbox" />',
    props: ["modelValue"],
    emits: ["update:modelValue"],
  },
  "el-checkbox": {
    template:
      '<label><input type="checkbox" :checked="modelValue" />{{ label }}<slot /></label>',
    props: ["modelValue", "indeterminate", "label", "value"],
    emits: ["update:modelValue", "change"],
  },
  "el-checkbox-group": {
    template: "<div><slot /></div>",
    props: ["modelValue"],
    emits: ["update:modelValue"],
  },
  "el-divider": { template: "<hr />" },
  "el-result": {
    template: '<div class="result"><slot /></div>',
    props: ["icon", "title", "subTitle"],
  },
};

const mountDialog = (props: Record<string, any> = {}) =>
  mount(ExportDialog, {
    props: {
      modelValue: true,
      module: "user",
      queryParams: {},
      ...props,
    },
    global: {
      stubs: elStubs,
    },
  });

describe("ExportDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("dialog 打开时渲染标题与文件格式选项", () => {
    const wrapper = mountDialog();
    expect(wrapper.text()).toContain("导出数据");
    expect(wrapper.text()).toContain("Excel (.xlsx)");
    expect(wrapper.text()).toContain("CSV (.csv)");
  });

  it("渲染异步导出开关", () => {
    const wrapper = mountDialog();
    expect(wrapper.text()).toContain("异步导出");
    expect(wrapper.text()).toContain("强制走异步任务");
  });

  it("未传 fields 时不渲染字段选择区域", () => {
    const wrapper = mountDialog();
    expect(wrapper.text()).not.toContain("导出字段");
  });

  it("传入 fields 时渲染字段选择区域", () => {
    const wrapper = mountDialog({
      fields: [
        { label: "用户名", value: "username" },
        { label: "昵称", value: "nickname" },
        { label: "邮箱", value: "email" },
      ],
    });
    expect(wrapper.text()).toContain("导出字段");
    expect(wrapper.text()).toContain("用户名");
    expect(wrapper.text()).toContain("昵称");
    expect(wrapper.text()).toContain("邮箱");
    expect(wrapper.text()).toContain("全选");
  });

  it("传入 fields 时显示不勾选导出全部字段提示", () => {
    const wrapper = mountDialog({
      fields: [{ label: "用户名", value: "username" }],
    });
    expect(wrapper.text()).toContain("不勾选则导出全部字段");
  });

  it("initialFormat=csv 时默认选中 CSV", async () => {
    const wrapper = mountDialog({ initialFormat: "csv" });
    await wrapper.vm.$nextTick();
    const csvRadio = wrapper.find('input[value="csv"]');
    expect((csvRadio.element as HTMLInputElement).checked).toBe(true);
  });

  it("module 为 role 时也正常渲染", () => {
    const wrapper = mountDialog({ module: "role" });
    expect(wrapper.text()).toContain("导出数据");
  });

  it("点击确定导出按钮调用 exportData", async () => {
    const blob = new Blob(["test"], { type: "text/csv" });
    mockExportData.mockResolvedValueOnce({ isAsync: false, blob });
    const wrapper = mountDialog();
    const submitBtn = wrapper
      .findAll("button")
      .find((b) => b.text().includes("确定导出"));
    expect(submitBtn).toBeDefined();
    await submitBtn!.trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.vm.$nextTick();
    expect(mockExportData).toHaveBeenCalled();
  });

  it("异步导出返回 taskId 时触发 async-task-created 事件", async () => {
    mockExportData.mockResolvedValueOnce({ isAsync: true, taskId: "task-123" });
    const wrapper = mountDialog();
    const submitBtn = wrapper
      .findAll("button")
      .find((b) => b.text().includes("确定导出"));
    await submitBtn!.trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("async-task-created")).toBeTruthy();
    expect(wrapper.emitted("async-task-created")![0]).toEqual(["task-123"]);
  });

  it("导出失败时不触发 async-task-created", async () => {
    mockExportData.mockRejectedValueOnce(new Error("导出失败"));
    const wrapper = mountDialog();
    const submitBtn = wrapper
      .findAll("button")
      .find((b) => b.text().includes("确定导出"));
    await submitBtn!.trigger("click");
    await wrapper.vm.$nextTick();
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("async-task-created")).toBeFalsy();
  });
});
