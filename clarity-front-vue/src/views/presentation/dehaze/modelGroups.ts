import type { AiModelVO } from "clarity-sdk-js";

/** 图像模型任务标签中文名；未知标签原样输出，避免维护封闭词表 */
const MODEL_TASK_TAG_LABELS: Record<string, string> = {
  clarity: "去雾",
  derain: "去雨",
  desnow: "去雪",
  deblur: "去模糊",
  lowlight: "低光增强",
  super_resolution: "超分辨率",
  denoise: "去噪",
  inpaint: "图像修复",
};

/** 兜底分组：模型未填 task_tags 时归入 */
export const OTHER_TASK_TAG_LABEL = "其他";

export interface ModelGroup {
  label: string;
  models: AiModelVO[];
}

/**
 * 按 taskTags 首标签分组（分组展示优于按标签过滤，避免隐藏模型）。
 * 与 flutter 同口径（`taskTags.firstOrNull ?? '其他'`）；taro/uniapp/RN/android/front-react 目前是扁平列表。
 */
export function groupModelsByTaskTag(models: AiModelVO[]): ModelGroup[] {
  const groups = new Map<string, AiModelVO[]>();
  for (const model of models) {
    // 空数组与未填等价，一并落到兜底分组
    const tag = model.taskTags?.[0] ?? OTHER_TASK_TAG_LABEL;
    groups.set(tag, [...(groups.get(tag) ?? []), model]);
  }
  return [...groups.entries()].map(([tag, models]) => ({
    label: MODEL_TASK_TAG_LABELS[tag] ?? tag,
    models,
  }));
}
