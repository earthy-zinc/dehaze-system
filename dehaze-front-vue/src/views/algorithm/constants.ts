import type { TagType } from "@/enums/TagType";

/** 算法状态：1 草稿 / 2 测试中 / 3 待审核 / 4 已发布 / 5 已停用 / 6 已归档 */
export const algorithmStatusMap: Record<
  number,
  { label: string; tag: TagType }
> = {
  1: { label: "草稿", tag: "info" },
  2: { label: "测试中", tag: "warning" },
  3: { label: "待审核", tag: "warning" },
  4: { label: "已发布", tag: "success" },
  5: { label: "已停用", tag: "danger" },
  6: { label: "已归档", tag: "info" },
};
