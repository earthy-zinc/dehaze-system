/**
 * 样例图片库服务
 * 从数据集管理模块获取公开展示的数据项作为样例图片
 */
import { DatasetItemAPI } from "dehaze-sdk-js";
import type { DatasetItemVO } from "dehaze-sdk-js";
import type { SampleImage, FogLevel } from "../data/imageInputData";

const SCENE_CATEGORY_MAP: Record<string, Exclude<FogLevel, "all">> = {
  城市: "light",
  城市建筑: "light",
  建筑: "light",
  街道: "light",
  道路: "light",
  自然: "medium",
  自然风景: "medium",
  风景: "medium",
  森林: "medium",
  山脉: "medium",
  湖泊: "medium",
  海岸: "medium",
  乡村: "medium",
  人像: "heavy",
  人物: "heavy",
  夜景: "special",
  夜景雾霾: "special",
};

const convertItemToSample = (item: DatasetItemVO): SampleImage | null => {
  const hazyImage = item.hazyImages?.[0];
  const clearImage = item.clearImage;
  const image = hazyImage || clearImage;
  if (!image?.url) return null;

  const sceneType = item.sceneType || image.sceneType || "";
  const category = SCENE_CATEGORY_MAP[sceneType] || "medium";

  return {
    id: item.id,
    name: item.name || "未命名",
    url: image.url,
    category,
    scene: sceneType,
    difficulty:
      hazyImage?.hazeLevel === "heavy"
        ? "困难"
        : hazyImage?.hazeLevel === "medium"
          ? "中等"
          : "简单",
    recommendAlgorithm: undefined,
  };
};

export async function fetchSampleImages(
  category: FogLevel
): Promise<SampleImage[]> {
  try {
    const res = await DatasetItemAPI.getList({
      pageNum: 1,
      pageSize: 50,
      sortBy: "usageCount",
      sortOrder: "desc",
    });

    const items = (res.list as unknown as DatasetItemVO[]) || [];
    const samples = items
      .map(convertItemToSample)
      .filter((s): s is SampleImage => s !== null);

    if (category === "all") return samples;
    return samples.filter((s) => s.category === category);
  } catch {
    return [];
  }
}
