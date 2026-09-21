// 组件单测共用的 Element Plus 轻量桩：透传插槽、转发 click / v-model，避免加载真实组件。
import { h, inject, provide } from "vue";

const RADIO_GROUP_KEY = Symbol("radioGroupSet");

export const elStubs = {
  "el-button": {
    props: ["type", "size", "plain", "link", "disabled", "circle", "text"],
    emits: ["click"],
    template:
      '<button class="el-button-stub" :disabled="disabled" @click="$emit(\'click\')"><slot /></button>',
  },
  "el-tag": {
    props: ["type", "size", "effect"],
    template: '<span class="el-tag-stub"><slot /></span>',
  },
  "el-alert": {
    props: ["title", "type", "closable"],
    template: '<div class="el-alert-stub">{{ title }}</div>',
  },
  "el-empty": {
    props: ["description", "imageSize"],
    template: '<div class="el-empty-stub">{{ description }}</div>',
  },
  "el-collapse": { template: '<div class="el-collapse-stub"><slot /></div>' },
  "el-collapse-item": {
    template:
      '<div class="el-collapse-item-stub"><slot name="title" /><slot /></div>',
  },
  "el-tooltip": { template: '<span class="el-tooltip-stub"><slot /></span>' },
  "el-radio-group": {
    props: ["modelValue"],
    emits: ["update:modelValue"],
    setup(_props: unknown, { emit, slots }: any) {
      // 模拟 Element Plus：向子 radio-button 注入选中回调，经 v-model 上抛
      provide(RADIO_GROUP_KEY, (value: unknown) =>
        emit("update:modelValue", value)
      );
      return () =>
        h("div", { class: "el-radio-group-stub" }, slots.default?.());
    },
  },
  "el-radio-button": {
    props: ["value"],
    setup(props: { value?: unknown }, { slots }: any) {
      const select = inject(RADIO_GROUP_KEY, null) as
        ((value: unknown) => void) | null;
      return () =>
        h(
          "button",
          {
            class: "el-radio-button-stub",
            onClick: () => select?.(props.value),
          },
          slots.default?.()
        );
    },
  },
  "el-input": {
    props: ["modelValue", "type", "rows", "maxlength", "placeholder", "size"],
    emits: ["update:modelValue"],
    template:
      '<input class="el-input-stub" :value="modelValue" @input="$emit(\'update:modelValue\', $event.target.value)" />',
  },
};

/** 按文本查找按钮桩 */
export function findButton(
  wrapper: { findAll: (s: string) => any[] },
  text: string
) {
  return wrapper.findAll("button").find((btn) => btn.text().includes(text));
}
