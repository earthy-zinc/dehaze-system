import { mount } from "@vue/test-utils";
import { computed, provide, inject, ref } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ImportDialog from "../ImportDialog.vue";

const mockDownloadTemplate = vi.fn();
const mockImportData = vi.fn();

vi.mock("@/composables/useImportExport", () => ({
  useImportExport: () => ({
    exportLoading: ref(false),
    importLoading: ref(false),
    templateLoading: ref(false),
    downloadTemplate: mockDownloadTemplate,
    exportData: vi.fn(),
    downloadExportBlob: vi.fn(),
    importData: mockImportData,
  }),
  downloadBlob: vi.fn(),
  buildFileName: vi.fn(),
}));

vi.mock("element-plus", () => ({
  genFileId: vi.fn(() => 1),
  ElRadio: { name: "ElRadio", template: "<div />" },
  ElRadioGroup: { name: "ElRadioGroup", template: "<div />" },
  ElButton: { name: "ElButton", template: "<div />" },
  ElIcon: { name: "ElIcon", template: "<span />" },
  ElForm: { name: "ElForm", template: "<div />" },
  ElFormItem: { name: "ElFormItem", template: "<div />" },
  ElUpload: { name: "ElUpload", template: "<div />" },
  ElResult: { name: "ElResult", template: "<div />" },
  ElDialog: { name: "ElDialog", template: "<div />" },
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
    name: "el-radio-group",
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
  "el-icon": { template: "<span><slot /></span>" },
  "el-upload": {
    template: '<div class="upload"><slot /><slot name="tip" /></div>',
    props: ["autoUpload", "limit", "accept", "action", "drag"],
  },
  "el-result": {
    template: '<div class="result"><slot /></div>',
    props: ["icon", "title", "subTitle"],
  },
};

const mountDialog = (props: Record<string, any> = {}) =>
  mount(ImportDialog, {
    props: {
      modelValue: true,
      module: "user",
      ...props,
    },
    global: {
      stubs: {
        ...elStubs,
        ImportResultPanel: true,
      },
    },
  });

describe("ImportDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("dialog 打开时渲染标题与导入模式选项", () => {
    const wrapper = mountDialog();
    expect(wrapper.text()).toContain("导入用户");
    expect(wrapper.text()).toContain("全量导入");
    expect(wrapper.text()).toContain("部分导入");
  });

  it("渲染模板下载按钮(Excel/CSV)", () => {
    const wrapper = mountDialog();
    expect(wrapper.text()).toContain("Excel 模板");
    expect(wrapper.text()).toContain("CSV 模板");
  });

  it("渲染文件上传区域", () => {
    const wrapper = mountDialog();
    expect(wrapper.text()).toContain("将文件拖到此处");
  });

  it("切换到部分导入模式显示对应提示", async () => {
    const wrapper = mountDialog();
    const radioGroup = wrapper.findComponent({ name: "el-radio-group" });
    expect(radioGroup.exists()).toBe(true);
    await radioGroup.vm.$emit("update:modelValue", "partial");
    expect(wrapper.text()).toContain("仅新增不存在的记录");
  });

  it("module 为 role 时标题为导入角色", () => {
    const wrapper = mountDialog({ module: "role" });
    expect(wrapper.text()).toContain("导入角色");
  });

  it("module 为 algorithm 时标题为导入算法", () => {
    const wrapper = mountDialog({ module: "algorithm" });
    expect(wrapper.text()).toContain("导入算法");
  });

  it("传入 extraImportParams 时作为 prop 接收", () => {
    const wrapper = mountDialog({
      extraImportParams: { deptId: 1 },
    });
    expect(wrapper.props("extraImportParams")).toEqual({ deptId: 1 });
  });

  it("全量导入模式显示对应提示", () => {
    const wrapper = mountDialog();
    expect(wrapper.text()).toContain("覆盖更新已存在的记录");
  });
});
