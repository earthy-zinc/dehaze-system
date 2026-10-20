import { AiImageAPI, AiModelAPI } from "clarity-sdk-js";
import type { AiModelVO } from "clarity-sdk-js";

/** 历史最近使用模型的回看条数 */
const RECENT_TASK_PAGE_SIZE = 20;

export interface ImageModelOptions {
  models: AiModelVO[];
  preselected: AiModelVO | null;
}

/**
 * 加载启用的图像编辑模型，并给出预选模型
 *
 * 预选次序：历史最近使用（且仍在启用列表内）→ 后端标记的平台默认模型
 * （`isDefault` ← 字典 `ai_model.default_model.image_edit`）；后端未配置平台默认时
 * 不预选（由用户在模型下拉中显式选择），不再回落到"启用列表首项"猜值。
 */
export async function loadImageEditModels(): Promise<ImageModelOptions> {
  const models = (await AiModelAPI.listEnabledModels("image_edit")) || [];
  const preselected = await findRecentlyUsedModel(models);
  return {
    models,
    preselected: preselected ?? models.find((m) => m.isDefault) ?? null,
  };
}

/** 历史最近使用的模型（按任务时间倒序取首个仍在启用列表内的） */
async function findRecentlyUsedModel(
  models: AiModelVO[]
): Promise<AiModelVO | null> {
  try {
    const tasks = await AiImageAPI.listImageTasks({
      pageNum: 1,
      pageSize: RECENT_TASK_PAGE_SIZE,
    });
    for (const task of tasks.list) {
      const match = models.find((m) => m.modelId === task.model);
      if (match) return match;
    }
  } catch {
    // 历史查询失败时交由调用方取后端平台默认标记
  }
  return null;
}
