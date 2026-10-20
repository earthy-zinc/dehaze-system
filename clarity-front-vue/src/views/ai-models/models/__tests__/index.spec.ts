// 管理端模型列表页表单弹窗回归：编辑回显 / 新增空表单 / 编辑态不可改字段禁用。
// 背景：store.openFormDialog 是原地改 formDialog 属性，watch 若监听 computed 本身则永不触发、表单空白。
import { flushPromises, mount } from "@vue/test-utils";
import { defineComponent, h, nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AiModelAPI, AiProviderAPI, type AiModelVO } from "clarity-sdk-js";
import { elStubs } from "@/components/ai-chat/__tests__/stubs";
import { useAdminModelStore } from "@/store/modules/adminModel";
import ModelsPage from "../index.vue";

vi.mock("clarity-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("clarity-sdk-js")>();
  return {
    ...actual,
    AiModelAPI: {
      ...actual.AiModelAPI,
      listModels: vi.fn(),
      listEnabledModels: vi.fn(),
    },
    AiProviderAPI: {
      ...actual.AiProviderAPI,
      listProviders: vi.fn(),
      getUsageStats: vi.fn(),
    },
  };
});

const passthrough = (name: string) =>
  defineComponent({
    name,
    inheritAttrs: false,
    template: '<div class="stub"><slot /><slot name="dropdown" /></div>',
  });

/** 弹窗桩：modelValue 为真时渲染标题与默认插槽 */
const DialogStub = defineComponent({
  name: "ElDialog",
  props: ["modelValue", "title"],
  setup:
    (props, { slots }) =>
    () =>
      props.modelValue
        ? h("div", { class: "el-dialog-stub" }, [
            h("div", { class: "el-dialog-stub__title" }, String(props.title)),
            slots.default?.(),
          ])
        : null,
});

/** 下拉桩：暴露选中值与禁用态，供断言 */
const SelectStub = defineComponent({
  name: "ElSelect",
  props: ["modelValue", "disabled", "multiple", "placeholder"],
  setup:
    (props, { slots }) =>
    () =>
      h(
        "div",
        {
          class: "el-select-stub",
          "data-value": String(props.modelValue),
          "data-disabled": String(!!props.disabled),
        },
        slots.default?.()
      ),
});

const global = {
  stubs: {
    ...elStubs,
    "el-dialog": DialogStub,
    "el-select": SelectStub,
    "el-form": passthrough("ElForm"),
    "el-form-item": passthrough("ElFormItem"),
    "el-card": passthrough("ElCard"),
    "el-divider": passthrough("ElDivider"),
    "el-tabs": passthrough("ElTabs"),
    "el-tab-pane": passthrough("ElTabPane"),
    // 表格桩不渲染列插槽：作用域插槽需要 row 数据，本用例只关心表单弹窗
    "el-table": defineComponent({
      name: "ElTable",
      template: '<div class="el-table-stub"></div>',
    }),
    "el-table-column": passthrough("ElTableColumn"),
    "el-switch": passthrough("ElSwitch"),
    "el-checkbox": passthrough("ElCheckbox"),
    "el-input-number": passthrough("ElInputNumber"),
    "el-icon": passthrough("ElIcon"),
    "el-option": passthrough("ElOption"),
  },
  directives: { loading: {}, hasPerm: () => {} },
};

const modelRow = {
  id: 9,
  providerId: 3,
  modelId: "o-haze-image-edit",
  modelType: "image_edit",
  taskTags: ["dehaze"],
  displayName: "O-HAZE 去雾",
  maxContextTokens: 8192,
  maxOutputTokens: 2048,
  supportsMultimodal: 0,
  supportsToolCall: 0,
  supportsStreaming: 1,
  supportsPromptCache: 0,
  supportsStructuredOutput: 0,
  promptCachePrefixLen: 0,
  fallbackModelId: null,
  vipLevel: 1,
  status: 1,
} as unknown as AiModelVO;

async function mountPage() {
  const wrapper = mount(ModelsPage, { global });
  await flushPromises();
  await nextTick();
  return wrapper;
}

type Wrapper = Awaited<ReturnType<typeof mountPage>>;

const dialogTitle = (wrapper: Wrapper) =>
  wrapper.find(".el-dialog-stub__title").text();

/** 表单里的下拉：按选中值定位（同一弹窗内值可区分） */
const selectByValue = (wrapper: Wrapper, value: string) =>
  wrapper
    .findAll(".el-select-stub")
    .find((s) => s.attributes("data-value") === value);

/** 弹窗表单内输入框顺序：模型标识在前、显示名称其次（el-input 桩不会渲染 placeholder，按位置取） */
const dialogInputs = (wrapper: Wrapper) =>
  wrapper.find(".el-dialog-stub").findAll("input");
const modelIdInput = (wrapper: Wrapper) => dialogInputs(wrapper)[0];
const displayNameInput = (wrapper: Wrapper) => dialogInputs(wrapper)[1];

describe("管理端模型管理页 · 表单弹窗", () => {
  beforeEach(() => {
    vi.mocked(AiModelAPI.listModels).mockResolvedValue({
      list: [],
      total: 0,
    } as never);
    vi.mocked(AiModelAPI.listEnabledModels).mockResolvedValue([] as never);
    vi.mocked(AiProviderAPI.listProviders).mockResolvedValue({
      list: [],
      total: 0,
    } as never);
    vi.mocked(AiProviderAPI.getUsageStats).mockResolvedValue({} as never);
  });

  it("编辑：打开弹窗后回显该行数据，且创建后不可改字段禁用", async () => {
    const wrapper = await mountPage();
    const store = useAdminModelStore();

    // store.openFormDialog 为原地改属性，此处按同一路径触发
    store.formDialog.model = modelRow;
    store.formDialog.visible = true;
    await nextTick();

    expect(dialogTitle(wrapper)).toBe("编辑模型");
    expect(selectByValue(wrapper, "3")).toBeTruthy(); // 供应商
    expect(selectByValue(wrapper, "dehaze")).toBeTruthy(); // 任务标签

    const typeSelect = selectByValue(wrapper, "image_edit")!;
    expect(typeSelect.attributes("data-disabled")).toBe("true");

    const idInput = modelIdInput(wrapper);
    expect(idInput.element.value).toBe("o-haze-image-edit");
    expect(idInput.attributes("disabled")).toBeDefined();

    expect(displayNameInput(wrapper).element.value).toBe("O-HAZE 去雾");
  });

  it("新增：字段为默认空值，模型类型与模型标识可编辑", async () => {
    const wrapper = await mountPage();
    const store = useAdminModelStore();

    store.formDialog.model = null;
    store.formDialog.visible = true;
    await nextTick();

    expect(dialogTitle(wrapper)).toBe("新增模型");
    const typeSelect = selectByValue(wrapper, "chat")!;
    expect(typeSelect.attributes("data-disabled")).toBe("false");

    const idInput = modelIdInput(wrapper);
    expect(idInput.element.value).toBe("");
    expect(idInput.attributes("disabled")).toBeUndefined();
    // 显示名称为空
    expect(displayNameInput(wrapper).element.value).toBe("");
  });

  it("关闭后重开另一行：按新行回显，不残留上一次表单", async () => {
    const wrapper = await mountPage();
    const store = useAdminModelStore();

    store.formDialog.model = modelRow;
    store.formDialog.visible = true;
    await nextTick();
    expect(modelIdInput(wrapper).element.value).toBe("o-haze-image-edit");

    store.formDialog.visible = false;
    await nextTick();

    store.formDialog.model = {
      ...modelRow,
      modelId: "aecrnet",
      displayName: "AECR-NET",
      modelType: "chat",
    } as AiModelVO;
    store.formDialog.visible = true;
    await nextTick();

    expect(modelIdInput(wrapper).element.value).toBe("aecrnet");
    expect(displayNameInput(wrapper).element.value).toBe("AECR-NET");
    expect(selectByValue(wrapper, "chat")).toBeTruthy();
  });
});
