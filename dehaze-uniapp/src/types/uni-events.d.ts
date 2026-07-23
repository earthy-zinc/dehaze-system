/**
 * uni-app 原生组件事件类型定义
 *
 * 用于补齐 uni-app 内联事件类型，避免在模板中重复声明内联类型。
 */

/** slider 组件 change 事件 */
export interface SliderChangeEvent {
  detail: { value: number };
}
