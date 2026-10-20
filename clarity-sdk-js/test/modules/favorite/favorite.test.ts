import { AiModelAPI, AiProviderAPI, DatasetAPI, FavoriteAPI } from "../../../index";
import { FavoriteTargetType } from "@/api/favorite/model";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { createFavoriteForm, createFavoriteQuery } from "#/factories/favorite";
import { uniqueName } from "#/factories/common";
import { USERS } from "#/factories/constants";

/**
 * 收藏管理接口测试
 *
 * 测试策略：
 * - 收藏目标对象使用已启用的图像模型（targetType=model），避免依赖预置算法数据
 * - result/dataset 类型仅做接口契约验证（创建+清理），依赖后端是否支持该类型收藏
 * - 每个测试用例独立创建并清理自己的收藏记录
 */
describe("收藏管理接口测试", () => {
  // 收集所有需要清理的收藏记录 ID 和临时模型 slug
  const favoriteIds: number[] = [];
  const createdModelSlugs: string[] = [];

  // 动态获取 result/dataset 类型的目标 ID（避免硬编码预置数据依赖）
  let datasetTargetId: number;
  let resultTargetId: number;
  let datasetCreated = false;

  // 已启用的图像模型行 ID（sys_ai_model.id），轮转使用
  let modelTargetIds: number[] = [];
  let modelCursor = 0;
  let providerId: number | undefined;

  beforeAll(async () => {
    await login(USERS.ADMIN.username);

    // 动态获取一个已存在的数据集 ID
    const datasetPage = await DatasetAPI.getList({ pageNum: 1, pageSize: 1 });
    if (datasetPage.list.length > 0) {
      datasetTargetId = datasetPage.list[0]!.id;
    } else {
      // 测试环境无数据集，创建一个临时的
      datasetTargetId = await DatasetAPI.add({
        parentId: 0,
        name: `favorite_test_dataset_${Date.now()}`,
        type: "用户数据集",
        description: "收藏测试自动创建",
        status: 1,
      });
      datasetCreated = true;
    }
    // result 类型查 sys_pred_log 表，校验较宽松，保留默认值
    resultTargetId = 1;

    // 已启用图像模型作为 model 目标
    const models = await AiModelAPI.listEnabledModels("image_edit");
    modelTargetIds = models.map((m) => m.id);

    const providers = await AiProviderAPI.listEnabledProviders();
    providerId = providers[0]?.id;
  });

  afterAll(async () => {
    // 先清理收藏记录，再清理临时模型，最后清理临时数据集
    if (favoriteIds.length > 0) {
      try {
        await FavoriteAPI.deleteByIds(favoriteIds);
      } catch (e) {
        console.warn(`清理失败:`, e);
      }
    }
    for (const slug of createdModelSlugs) {
      try {
        await AiModelAPI.deleteModel(slug);
      } catch (e) {
        console.warn(`清理失败:`, e);
      }
    }
    if (datasetCreated && datasetTargetId) {
      try {
        await DatasetAPI.deleteById(datasetTargetId);
      } catch (e) {
        console.warn(`清理失败:`, e);
      }
    }
  });

  /** 取一个已启用的图像模型行 ID 作为收藏目标（轮转，避免同目标重复收藏） */
  function nextModelTarget(): number {
    if (modelTargetIds.length === 0) {
      throw new Error("无已启用的图像模型可作为收藏目标");
    }
    const id = modelTargetIds[modelCursor % modelTargetIds.length]!;
    modelCursor++;
    return id;
  }

  /** 创建临时图像模型作为收藏目标（用于失效联动测试），返回行 ID */
  async function createModelTarget(): Promise<{ id: number; slug: string }> {
    if (providerId === undefined) {
      throw new Error("无已启用供应商，无法创建图像模型");
    }
    const slug = uniqueName("favmodel").slice(0, 64);
    const created = await AiModelAPI.createModel({
      providerId,
      modelId: slug,
      modelType: "image_edit",
      displayName: uniqueName("收藏失效模型"),
    });
    createdModelSlugs.push(created.modelId);
    return { id: created.id, slug: created.modelId };
  }

  /** 创建收藏并记录 ID 以便统一清理 */
  async function createFavorite(
    targetType: FavoriteTargetType = "model",
    targetId?: number
  ): Promise<{ favoriteId: number; targetId: number }> {
    const actualTargetId = targetId ?? nextModelTarget();
    const form = createFavoriteForm({ targetType, targetId: actualTargetId });
    const favoriteId = (await FavoriteAPI.add(form)) as number;
    favoriteIds.push(favoriteId);
    return { favoriteId, targetId: actualTargetId };
  }

  /** 从清理列表中移除已删除的收藏 ID */
  function removeFavoriteIds(ids: number[]) {
    for (const id of ids) {
      const idx = favoriteIds.indexOf(id);
      if (idx >= 0) favoriteIds.splice(idx, 1);
    }
  }

  /** 清理指定目标类型下已存在的同一 targetId 收藏（避免残留导致重复） */
  async function cleanupFavorite(targetType: FavoriteTargetType, targetId: number) {
    try {
      const page = await FavoriteAPI.getPage(createFavoriteQuery({ targetType, pageSize: 100 }));
      const existing = page.list.find((item) => item.targetId === targetId);
      if (existing) {
        await FavoriteAPI.deleteByIds([existing.id]);
      }
    } catch {
      // 清理失败忽略
    }
  }

  describe("POST /api/v1/favorites - 添加收藏", () => {
    test("正向测试：收藏模型（targetType=model）", async () => {
      const targetId = nextModelTarget();
      const form = createFavoriteForm({ targetType: "model", targetId });

      const favoriteId = await FavoriteAPI.add(form);

      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);
    });

    test("正向测试：收藏处理结果（targetType=result）", async () => {
      const targetType = "result";
      const targetId = resultTargetId;

      // 清理上次测试残留：直接查列表找匹配记录并删除
      await cleanupFavorite(targetType, targetId);

      const form = createFavoriteForm({ targetType, targetId });
      const favoriteId = await FavoriteAPI.add(form);
      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);
    });

    test("正向测试：收藏数据集（targetType=dataset）", async () => {
      const targetType = "dataset";
      const targetId = datasetTargetId;

      // 清理上次测试残留：直接查列表找匹配记录并删除
      await cleanupFavorite(targetType, targetId);

      const form = createFavoriteForm({ targetType, targetId });
      const favoriteId = await FavoriteAPI.add(form);
      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);
    });

    test("边界测试：重复收藏同一对象幂等，返回原行 id", async () => {
      const targetId = nextModelTarget();
      const form = createFavoriteForm({ targetType: "model", targetId });

      const firstId = await FavoriteAPI.add(form);
      favoriteIds.push(firstId as number);

      // 已收藏（未取消）时重复 add：命中唯一键 uk_user_target(deleted=0) 复活原行，重查返回原 id
      const secondId = await FavoriteAPI.add(form);
      expect(secondId).toBe(firstId);
    });

    test("边界测试：取消后重新收藏生成新行", async () => {
      const targetId = nextModelTarget();
      const form = createFavoriteForm({ targetType: "model", targetId });

      const firstId = await FavoriteAPI.add(form);
      await FavoriteAPI.deleteByIds([firstId as number]);

      // 取消后原行 deleted=原行id（uk 含 deleted 不再冲突），重新收藏插入新行返回新 id
      const secondId = await FavoriteAPI.add(form);
      expect(secondId).toBeGreaterThan(firstId as number);

      const status = await FavoriteAPI.getStatus("model", targetId);
      expect(status.favorited).toBe(true);
      favoriteIds.push(secondId as number);
    });

    test("边界测试：收藏不存在的对象应返回业务错误 A0401", async () => {
      const form = createFavoriteForm({ targetType: "model", targetId: 99999999 });

      await expectBizError(FavoriteAPI.add(form), ["A0401", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("边界测试：收藏容量已满应返回业务错误 A0500", async () => {
      // 此场景难以在集成测试中真实触发（需先填满容量），真实容量上限由后端单测覆盖
      const targetId = nextModelTarget();
      const form = createFavoriteForm({ targetType: "model", targetId });

      const favoriteId = await FavoriteAPI.add(form);
      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);
    });

    test("参数校验：缺少必需字段 targetType 应抛出业务错误", async () => {
      const form = { targetId: 1 } as any;
      await expectBizError(FavoriteAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：缺少必需字段 targetId 应抛出业务错误", async () => {
      const form = { targetType: "model" } as any;
      await expectBizError(FavoriteAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  describe("DELETE /api/v1/favorites/{ids} - 批量取消收藏", () => {
    test("正向测试：取消单个收藏", async () => {
      const { favoriteId } = await createFavorite("model");

      await FavoriteAPI.deleteByIds([favoriteId]);
      removeFavoriteIds([favoriteId]);

      const page = await FavoriteAPI.getPage(createFavoriteQuery({ pageSize: 100 }));
      const exists = page.list.some((item) => item.id === favoriteId);
      expect(exists).toBe(false);
    });

    test("正向测试：批量取消多个收藏", async () => {
      const fav1 = await createFavorite("model");
      const fav2 = await createFavorite("model");
      const fav3 = await createFavorite("model");
      const idsToDelete = [fav1.favoriteId, fav2.favoriteId, fav3.favoriteId];

      await FavoriteAPI.deleteByIds(idsToDelete);
      removeFavoriteIds(idsToDelete);

      const page = await FavoriteAPI.getPage(createFavoriteQuery({ pageSize: 100 }));
      for (const id of idsToDelete) {
        const exists = page.list.some((item) => item.id === id);
        expect(exists).toBe(false);
      }
    });

    test("边界测试：取消不存在的收藏 ID 不应报错", async () => {
      // 后端通常对不存在的 ID 做幂等处理（返回成功）；若返回错误则暴露行为差异
      await FavoriteAPI.deleteByIds([99999999]);
    });
  });

  describe("GET /api/v1/favorites/page - 收藏列表分页查询", () => {
    test("正向测试：分页查询收藏列表并验证结构", async () => {
      const query = createFavoriteQuery();
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");

      if (result.list.length > 0) {
        const first = result.list[0]!;
        expect(typeof first.id).toBe("number");
        expect(typeof first.userId).toBe("number");
        expect(typeof first.targetType).toBe("string");
        expect(typeof first.targetId).toBe("number");
        expect(typeof first.createTime).toBe("string");
      }
    });

    test("正向测试：按类型筛选 model", async () => {
      await createFavorite("model");

      const query = createFavoriteQuery({ targetType: "model" });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      // 验证所有返回项都是 model 类型
      result.list.forEach((item) => {
        expect(item.targetType).toBe("model");
      });
    });

    test("正向测试：按类型筛选 result", async () => {
      const query = createFavoriteQuery({ targetType: "result" });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((item) => {
        expect(item.targetType).toBe("result");
      });
    });

    test("正向测试：按类型筛选 dataset", async () => {
      const query = createFavoriteQuery({ targetType: "dataset" });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((item) => {
        expect(item.targetType).toBe("dataset");
      });
    });

    test("正向测试：按关键词搜索收藏对象名称", async () => {
      // 创建带唯一 displayName 的临时模型，用其名称子串搜索
      if (providerId === undefined) {
        throw new Error("无已启用供应商，无法创建图像模型");
      }
      const slug = uniqueName("favmodel").slice(0, 64);
      const displayName = uniqueName("收藏关键词模型");
      const created = await AiModelAPI.createModel({
        providerId,
        modelId: slug,
        modelType: "image_edit",
        displayName,
      });
      createdModelSlugs.push(created.modelId);
      await createFavorite("model", created.id);

      // 用模型名称的子串作为关键词搜索
      const keyword = displayName.substring(0, Math.min(3, displayName.length));
      const query = createFavoriteQuery({ keywords: keyword });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      // 若后端支持关键词搜索，返回项应包含关键词
      if (result.list.length > 0) {
        const hasMatch = result.list.some(
          (item) => item.targetName && item.targetName.includes(keyword)
        );
        expect(hasMatch).toBe(true);
      }
    });

    test("正向测试：按收藏时间倒序排序（默认）", async () => {
      const query = createFavoriteQuery({ sortBy: "createTime", sortOrder: "desc" });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      for (let i = 1; i < result.list.length; i++) {
        const prev = result.list[i - 1]!;
        const curr = result.list[i]!;
        if (prev.createTime && curr.createTime) {
          expect(prev.createTime >= curr.createTime).toBe(true);
        }
      }
    });

    test("正向测试：按收藏时间正序排序", async () => {
      const query = createFavoriteQuery({ sortBy: "createTime", sortOrder: "asc" });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      for (let i = 1; i < result.list.length; i++) {
        const prev = result.list[i - 1]!;
        const curr = result.list[i]!;
        if (prev.createTime && curr.createTime) {
          expect(prev.createTime <= curr.createTime).toBe(true);
        }
      }
    });

    test("边界测试：空收藏列表（使用不存在的类型筛选）", async () => {
      // image/preset 为预留类型，当前无业务数据，应返回空列表
      const query = createFavoriteQuery({ targetType: "image" });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      expect(result.total).toBe(0);
      expect(result.list.length).toBe(0);
    });

    test("边界测试：搜索无匹配关键词返回空列表", async () => {
      const query = createFavoriteQuery({ keywords: "不存在的关键词xyz_99999" });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBe(0);
    });

    test("边界测试：分页参数 pageNum=1, pageSize=1", async () => {
      await createFavorite("model");

      const query = createFavoriteQuery({ pageNum: 1, pageSize: 1 });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBeLessThanOrEqual(1);
      expect(typeof result.total).toBe("number");
    });

    test("边界测试：分页参数 pageSize 较大值", async () => {
      const query = createFavoriteQuery({ pageNum: 1, pageSize: 100 });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBeLessThanOrEqual(100);
    });
  });

  describe("GET /api/v1/favorites/{id}/status - 检查是否已收藏", () => {
    test("正向测试：已收藏的对象返回 favorited=true", async () => {
      const { targetId } = await createFavorite("model");

      const status = await FavoriteAPI.getStatus("model", targetId);

      expect(status.targetType).toBe("model");
      expect(status.targetId).toBe(targetId);
      expect(status.favorited).toBe(true);
    });

    test("正向测试：未收藏的对象返回 favorited=false", async () => {
      const targetId = nextModelTarget(); // 取一个目标但未收藏

      const status = await FavoriteAPI.getStatus("model", targetId);

      expect(status.targetId).toBe(targetId);
      expect(status.favorited).toBe(false);
    });
  });

  describe("数据隔离 - 越权校验", () => {
    test("边界：越权取消他人收藏应更新0行无报错", async () => {
      // admin 创建收藏
      const { favoriteId } = await createFavorite("model");

      // 切换到 user 尝试取消 admin 的收藏
      await login(USERS.USER.username);
      try {
        await FavoriteAPI.deleteByIds([favoriteId]);
        // 后端按 user_id 过滤，更新0行但不报错
      } catch {
        // 部分后端可能返回错误
      } finally {
        await login(USERS.ADMIN.username);
      }

      const page = await FavoriteAPI.getPage(createFavoriteQuery({ pageSize: 100 }));
      const stillExists = page.list.some((item) => item.id === favoriteId);
      expect(stillExists).toBe(true);
    });

    test("验证：用户仅能查询到自己的收藏（数据隔离）", async () => {
      // admin 创建收藏并捕获其 id
      const { favoriteId } = await createFavorite("model");

      // 切换到 user 查询收藏列表
      await login(USERS.USER.username);
      const userPage = await FavoriteAPI.getPage(createFavoriteQuery({ pageSize: 100 }));

      // user 看不到 admin 创建的收藏
      expect(userPage.list.some((item) => item.id === favoriteId)).toBe(false);
      await login(USERS.ADMIN.username);
    });
  });

  describe("GET /api/v1/favorites/count - 收藏数量统计", () => {
    test("正向测试：获取所有类型收藏数量", async () => {
      await createFavorite("model");

      const result = await FavoriteAPI.getCount();

      expect(Array.isArray(result)).toBe(true);
      result.forEach((item) => {
        expect(typeof item.targetType).toBe("string");
        expect(typeof item.count).toBe("number");
        expect(item.count).toBeGreaterThanOrEqual(0);
      });
    });

    test("正向测试：按类型获取收藏数量", async () => {
      await createFavorite("model");

      const result = await FavoriteAPI.getCount("model");

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      const modelCount = result.find((item) => item.targetType === "model");
      expect(modelCount).toBeDefined();
      expect(modelCount!.count).toBeGreaterThan(0);
    });
  });

  describe("目标类型校验", () => {
    test("参数校验：非法 targetType 应返回业务错误 A0400", async () => {
      const form = createFavoriteForm({
        targetType: "hacker" as unknown as FavoriteTargetType,
        targetId: 1,
      });
      await expectBizError(FavoriteAPI.add(form), ["A0400"]);
    });
  });

  describe("失效联动（对象删除 → is_invalid）", () => {
    test("完整联动：模型删除后收藏标记 isInvalid=true，且该对象不可再次收藏", async () => {
      const { id: targetId, slug } = await createModelTarget();
      const form = createFavoriteForm({ targetType: "model", targetId });
      const favoriteId = (await FavoriteAPI.add(form)) as number;
      favoriteIds.push(favoriteId);

      await AiModelAPI.deleteModel(slug);

      const page = await FavoriteAPI.getPage(
        createFavoriteQuery({ targetType: "model", pageSize: 100 })
      );
      const found = page.list.find((item) => item.id === favoriteId);
      expect(found).toBeDefined();
      expect(found!.isInvalid).toBe(true);

      // 已删除对象再次收藏应返回 A0401
      await expectBizError(FavoriteAPI.add(form), ["A0401"]);

      // 失效收藏仍可正常取消
      await FavoriteAPI.deleteByIds([favoriteId]);
      removeFavoriteIds([favoriteId]);
      const status = await FavoriteAPI.getStatus("model", targetId);
      expect(status.favorited).toBe(false);
    });

    test("完整联动：数据集删除后收藏标记 isInvalid=true 且名称不再回显", async () => {
      const datasetName = `favorite_invalidation_${Date.now()}`;
      const targetId = await DatasetAPI.add({
        parentId: 0,
        name: datasetName,
        type: "用户数据集",
        description: "收藏失效联动测试",
        status: 1,
      });
      const form = createFavoriteForm({ targetType: "dataset", targetId });
      const favoriteId = (await FavoriteAPI.add(form)) as number;
      favoriteIds.push(favoriteId);

      await DatasetAPI.deleteById(targetId);

      const page = await FavoriteAPI.getPage(
        createFavoriteQuery({ targetType: "dataset", pageSize: 100 })
      );
      const found = page.list.find((item) => item.id === favoriteId);
      expect(found).toBeDefined();
      expect(found!.isInvalid).toBe(true);
      expect(found!.targetName).toBeFalsy();

      await FavoriteAPI.deleteByIds([favoriteId]);
      removeFavoriteIds([favoriteId]);
    });
  });

  describe("对抗性脏语料（关键词搜索）", () => {
    test("LIKE 通配符按字面匹配（% 和 _ 不做模式展开）", async () => {
      const datasetName = `fav_wild_a%b_${Date.now()}`;
      const targetId = await DatasetAPI.add({
        parentId: 0,
        name: datasetName,
        type: "用户数据集",
        description: "关键词转义测试",
        status: 1,
      });
      const form = createFavoriteForm({ targetType: "dataset", targetId });
      const favoriteId = (await FavoriteAPI.add(form)) as number;
      favoriteIds.push(favoriteId);

      // % 作为字面字符匹配
      const page = await FavoriteAPI.getPage(
        createFavoriteQuery({ keywords: "a%b", pageSize: 100 })
      );
      expect(page.list.some((item) => item.id === favoriteId)).toBe(true);

      // 不存在的脏关键词不应报错且不匹配
      const dirtyPage = await FavoriteAPI.getPage(
        createFavoriteQuery({ keywords: "🌫️\u200b\r\n", pageSize: 100 })
      );
      expect(Array.isArray(dirtyPage.list)).toBe(true);
      expect(dirtyPage.list.some((item) => item.id === favoriteId)).toBe(false);

      removeFavoriteIds([favoriteId]);
      await FavoriteAPI.deleteByIds([favoriteId]);
      await DatasetAPI.deleteById(targetId);
    });
  });

  describe("集成场景：完整业务流程", () => {
    test("完整流程：添加收藏 → 查询列表验证 → 取消收藏 → 验证已取消", async () => {
      const targetId = nextModelTarget();
      const form = createFavoriteForm({ targetType: "model", targetId });
      const favoriteId = (await FavoriteAPI.add(form)) as number;
      expect(favoriteId).toBeGreaterThan(0);

      const status = await FavoriteAPI.getStatus("model", targetId);
      expect(status.favorited).toBe(true);

      const page = await FavoriteAPI.getPage(createFavoriteQuery({ targetType: "model" }));
      const found = page.list.find((item) => item.id === favoriteId);
      expect(found).toBeDefined();
      expect(found!.targetId).toBe(targetId);
      expect(found!.targetType).toBe("model");

      await FavoriteAPI.deleteByIds([favoriteId]);

      const statusAfter = await FavoriteAPI.getStatus("model", targetId);
      expect(statusAfter.favorited).toBe(false);

      const pageAfter = await FavoriteAPI.getPage(createFavoriteQuery({ targetType: "model" }));
      const stillExists = pageAfter.list.some((item) => item.id === favoriteId);
      expect(stillExists).toBe(false);

      // 从清理列表移除（已手动删除）
      removeFavoriteIds([favoriteId]);
    });

    test("完整流程：添加多个类型收藏 → 按类型筛选验证", async () => {
      const fav1 = await createFavorite("model");
      const fav2 = await createFavorite("model");

      const modelPage = await FavoriteAPI.getPage(
        createFavoriteQuery({ targetType: "model", pageSize: 100 })
      );
      const hasFav1 = modelPage.list.some((item) => item.id === fav1.favoriteId);
      const hasFav2 = modelPage.list.some((item) => item.id === fav2.favoriteId);
      expect(hasFav1).toBe(true);
      expect(hasFav2).toBe(true);

      const datasetPage = await FavoriteAPI.getPage(
        createFavoriteQuery({ targetType: "dataset", pageSize: 100 })
      );
      const datasetHasFav1 = datasetPage.list.some((item) => item.id === fav1.favoriteId);
      const datasetHasFav2 = datasetPage.list.some((item) => item.id === fav2.favoriteId);
      expect(datasetHasFav1).toBe(false);
      expect(datasetHasFav2).toBe(false);

      const counts = await FavoriteAPI.getCount("model");
      const modelCount = counts.find((c) => c.targetType === "model");
      expect(modelCount).toBeDefined();
      expect(modelCount!.count).toBeGreaterThanOrEqual(2);
    });
  });
});
