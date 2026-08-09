/**
 * 样例图片库服务
 * 对齐设计文档需求规格 §2.3.1：样例图片来源于数据集管理模块中标记为"公开展示"的数据项
 * 通过 DatasetItemAPI.getList 获取公开展示的数据项，按场景类型分类
 */

import { DatasetItemAPI } from "dehaze-sdk-js";
import type { DatasetItemVO, DatasetItemQuery } from "dehaze-sdk-js";
import { SampleImage, SampleCategory } from "./types";

// 场景类型到分类的映射
const SCENE_CATEGORY_MAP: Record<string, Exclude<SampleCategory, "all">> = {
  城市: "city",
  城市建筑: "city",
  建筑: "city",
  街道: "city",
  道路: "city",
  自然: "nature",
  自然风景: "nature",
  风景: "nature",
  森林: "nature",
  山脉: "nature",
  湖泊: "nature",
  海岸: "nature",
  乡村: "nature",
  人像: "portrait",
  人物: "portrait",
  夜景: "night",
  夜景雾霾: "night",
};

// 分类标签配置
export const categoryTabs = [
  { key: "all" as const, label: "全部" },
  { key: "city" as const, label: "城市建筑" },
  { key: "nature" as const, label: "自然风景" },
  { key: "portrait" as const, label: "人像场景" },
  { key: "night" as const, label: "夜景雾霾" },
];

/**
 * 将数据项转换为样例图片
 * 优先使用有雾图（去雾处理的输入），若无则用清晰图
 */
const convertItemToSample = (item: DatasetItemVO): SampleImage | null => {
  // 优先取有雾图作为去雾输入样例
  const hazyImage = item.hazyImages?.[0];
  const clearImage = item.clearImage;
  const image = hazyImage || clearImage;

  if (!image?.url) return null;

  const sceneType = item.sceneType || image.sceneType || "";
  const category = SCENE_CATEGORY_MAP[sceneType] || "nature";

  return {
    id: item.id,
    name: item.name || "未命名",
    url: image.url,
    thumbnailUrl: image.thumbnailUrl || image.url,
    category,
    sceneType,
    hazeLevel: hazyImage?.hazeLevel as "light" | "medium" | "heavy" | undefined,
    cleanUrl: clearImage?.url,
  };
};

/**
 * 从数据集管理模块获取公开展示的样例图片
 * @param category 场景分类筛选
 */
export const fetchSampleImages = async (
  category: SampleCategory
): Promise<SampleImage[]> => {
  try {
    // 获取公开展示的数据项，按使用次数排序
    const res = await DatasetItemAPI.getList({
      pageNum: 1,
      pageSize: 50,
      isPublic: true,
      sortBy: "usageCount",
      sortOrder: "desc",
    } as DatasetItemQuery);

    const items = res.list || [];
    const samples = items
      .map(convertItemToSample)
      .filter((s): s is SampleImage => s !== null);

    if (category === "all") {
      return samples;
    }
    return samples.filter((s) => s.category === category);
  } catch (error) {
    console.error("获取样例图片失败:", error);
    return [];
  }
};
