// 去雾页模型下拉分组：兜底分组是防御分支（当前启用的 image_edit 模型都带 task_tags，
// 但模型管理允许不填，将来必然出现），用确定性单测覆盖，不靠造库数据。
import { describe, expect, it } from "vitest";
import type { AiModelVO } from "clarity-sdk-js";
import { groupModelsByTaskTag, OTHER_TASK_TAG_LABEL } from "../modelGroups";

const model = (over: Partial<AiModelVO>): AiModelVO =>
  ({ modelId: "x", displayName: "x", ...over }) as AiModelVO;

describe("groupModelsByTaskTag", () => {
  it("按首标签分组并映射中文组名，组内保持原顺序", () => {
    const groups = groupModelsByTaskTag([
      model({ modelId: "a", taskTags: ["dehaze"] }),
      model({ modelId: "b", taskTags: ["derain"] }),
      model({ modelId: "c", taskTags: ["dehaze"] }),
    ]);

    expect(groups.map((g) => g.label)).toEqual(["去雾", "去雨"]);
    expect(groups[0].models.map((m) => m.modelId)).toEqual(["a", "c"]);
  });

  it("未填 taskTags 与空数组都落入兜底分组", () => {
    const groups = groupModelsByTaskTag([
      model({ modelId: "no-tags", taskTags: undefined }),
      model({ modelId: "empty-tags", taskTags: [] }),
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0].label).toBe(OTHER_TASK_TAG_LABEL);
    expect(groups[0].models.map((m) => m.modelId)).toEqual([
      "no-tags",
      "empty-tags",
    ]);
  });

  it("未知标签原样输出，多标签只取首标签", () => {
    const groups = groupModelsByTaskTag([
      model({ modelId: "unknown", taskTags: ["foo"] }),
      model({ modelId: "multi", taskTags: ["dehaze", "derain"] }),
    ]);

    expect(groups.map((g) => g.label)).toEqual(["foo", "去雾"]);
  });

  it("带标签与无标签混排：兜底组按出现顺序插入，模型不丢失", () => {
    const models = [
      model({ modelId: "a", taskTags: ["dehaze"] }),
      model({ modelId: "b", taskTags: undefined }),
      model({ modelId: "c", taskTags: ["derain"] }),
      model({ modelId: "d", taskTags: [] }),
    ];
    const groups = groupModelsByTaskTag(models);

    expect(groups.map((g) => g.label)).toEqual(["去雾", "其他", "去雨"]);
    expect(groups[1].models.map((m) => m.modelId)).toEqual(["b", "d"]);
    expect(groups.flatMap((g) => g.models)).toHaveLength(models.length);
  });
});
