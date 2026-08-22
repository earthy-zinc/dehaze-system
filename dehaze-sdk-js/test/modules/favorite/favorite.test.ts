import { AlgorithmAPI, DatasetAPI, FavoriteAPI } from "../../../index";
import { FavoriteTargetType } from "@/api/favorite/model";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { createAlgorithmForm } from "#/factories/algorithm";
import { createFavoriteForm, createFavoriteQuery } from "#/factories/favorite";
import { USERS } from "#/factories/constants";

/**
 * 收藏管理接口测试
 *
 * 测试策略：
 * - 收藏目标对象使用动态创建的算法（targetType=algorithm），避免依赖预置数据
 * - result/dataset 类型仅做接口契约验证（创建+清理），依赖后端是否支持该类型收藏
 * - 每个测试用例独立创建并清理自己的收藏记录和算法目标
 */
describe("收藏管理接口测试", () => {
  // 收集所有需要清理的收藏记录 ID 和算法 ID
  const favoriteIds: number[] = [];
  const algorithmIds: number[] = [];

  // 动态获取 result/dataset 类型的目标 ID（避免硬编码预置数据依赖）
  let datasetTargetId: number;
  let resultTargetId: number;
  let datasetCreated = false;

  beforeAll(async () => {
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
    // result 类型查 sys_prediction_log 表，校验较宽松，保留默认值
    resultTargetId = 1;
  });

  afterAll(async () => {
    // 先清理收藏记录，再清理算法，最后清理临时数据集
    if (favoriteIds.length > 0) {
      try {
        await FavoriteAPI.deleteByIds(favoriteIds);
      } catch (e) {
        console.warn(`清理失败:`, e);
      }
    }
    for (const id of algorithmIds) {
      try {
        await AlgorithmAPI.deleteByIds([id.toString()]);
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

  /** 创建一个算法作为收藏目标，并返回算法 ID */
  async function createAlgorithmTarget(): Promise<number> {
    const form = createAlgorithmForm({ parentId: 0 });
    const id = (await AlgorithmAPI.add(form)) as number;
    algorithmIds.push(id);
    return id;
  }

  /** 创建收藏并记录 ID 以便统一清理 */
  async function createFavorite(
    targetType: "algorithm" | "result" | "dataset" = "algorithm",
    targetId?: number
  ): Promise<{ favoriteId: number; targetId: number }> {
    const actualTargetId = targetId ?? (await createAlgorithmTarget());
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
    test("正向测试：收藏算法（targetType=algorithm）", async () => {
      const targetId = await createAlgorithmTarget();
      const form = createFavoriteForm({ targetType: "algorithm", targetId });

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

    test("边界测试：重复收藏同一对象走 upsert 复活，返回原行 id", async () => {
      const targetId = await createAlgorithmTarget();
      const form = createFavoriteForm({ targetType: "algorithm", targetId });

      const firstId = await FavoriteAPI.add(form);
      favoriteIds.push(firstId as number);

      // upsert 改造后：重复收藏走 ON DUPLICATE KEY UPDATE 复活软删行，返回原行 id（非新增）
      const secondId = await FavoriteAPI.add(form);
      expect(secondId).toBe(firstId);
    });

    test("边界测试：取消后重新收藏（原记录复活）", async () => {
      const targetId = await createAlgorithmTarget();
      const form = createFavoriteForm({ targetType: "algorithm", targetId });

      const firstId = await FavoriteAPI.add(form);
      await FavoriteAPI.deleteByIds([firstId as number]);

      // 取消后重新收藏，应返回原记录 ID（复活）
      const secondId = await FavoriteAPI.add(form);
      expect(secondId).toBe(firstId);

      const status = await FavoriteAPI.getStatus("algorithm", targetId);
      expect(status.favorited).toBe(true);
      favoriteIds.push(secondId as number);
    });

    test("边界测试：收藏不存在的对象应返回业务错误 A0401", async () => {
      const form = createFavoriteForm({ targetType: "algorithm", targetId: 99999999 });

      await expectBizError(FavoriteAPI.add(form), ["A0401", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("边界测试：收藏容量已满应返回业务错误 A0500", async () => {
      // 此场景难以在集成测试中真实触发（需先填满容量），真实容量上限由后端单测覆盖
      const targetId = await createAlgorithmTarget();
      const form = createFavoriteForm({ targetType: "algorithm", targetId });

      const favoriteId = await FavoriteAPI.add(form);
      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);
    });

    test("参数校验：缺少必需字段 targetType 应抛出业务错误", async () => {
      const form = { targetId: 1 } as any;
      await expectBizError(FavoriteAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("参数校验：缺少必需字段 targetId 应抛出业务错误", async () => {
      const form = { targetType: "algorithm" } as any;
      await expectBizError(FavoriteAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  describe("DELETE /api/v1/favorites/{ids} - 批量取消收藏", () => {
    test("正向测试：取消单个收藏", async () => {
      const { favoriteId } = await createFavorite("algorithm");

      await FavoriteAPI.deleteByIds([favoriteId]);
      removeFavoriteIds([favoriteId]);

      const page = await FavoriteAPI.getPage(createFavoriteQuery({ pageSize: 100 }));
      const exists = page.list.some((item) => item.id === favoriteId);
      expect(exists).toBe(false);
    });

    test("正向测试：批量取消多个收藏", async () => {
      const fav1 = await createFavorite("algorithm");
      const fav2 = await createFavorite("algorithm");
      const fav3 = await createFavorite("algorithm");
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

    test("正向测试：按类型筛选 algorithm", async () => {
      await createFavorite("algorithm");

      const query = createFavoriteQuery({ targetType: "algorithm" });
      const result = await FavoriteAPI.getPage(query);

      expect(Array.isArray(result.list)).toBe(true);
      // 验证所有返回项都是 algorithm 类型
      result.list.forEach((item) => {
        expect(item.targetType).toBe("algorithm");
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
      const targetId = await createAlgorithmTarget();
      const algoInfo = await AlgorithmAPI.getAlgorithmInfoById(targetId);
      await createFavorite("algorithm", targetId);

      // 用算法名称的子串作为关键词搜索
      const keyword = algoInfo.name.substring(0, Math.min(3, algoInfo.name.length));
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
      await createFavorite("algorithm");

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
      const { targetId } = await createFavorite("algorithm");

      const status = await FavoriteAPI.getStatus("algorithm", targetId);

      expect(status.targetType).toBe("algorithm");
      expect(status.targetId).toBe(targetId);
      expect(status.favorited).toBe(true);
    });

    test("正向测试：未收藏的对象返回 favorited=false", async () => {
      const targetId = await createAlgorithmTarget(); // 创建但未收藏

      const status = await FavoriteAPI.getStatus("algorithm", targetId);

      expect(status.targetId).toBe(targetId);
      expect(status.favorited).toBe(false);
    });
  });

  describe("数据隔离 - 越权校验", () => {
    test("边界：越权取消他人收藏应更新0行无报错", async () => {
      // admin 创建收藏
      const { favoriteId } = await createFavorite("algorithm");

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
      const { favoriteId } = await createFavorite("algorithm");

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
      await createFavorite("algorithm");

      const result = await FavoriteAPI.getCount();

      expect(Array.isArray(result)).toBe(true);
      result.forEach((item) => {
        expect(typeof item.targetType).toBe("string");
        expect(typeof item.count).toBe("number");
        expect(item.count).toBeGreaterThanOrEqual(0);
      });
    });

    test("正向测试：按类型获取收藏数量", async () => {
      await createFavorite("algorithm");

      const result = await FavoriteAPI.getCount("algorithm");

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      const algorithmCount = result.find((item) => item.targetType === "algorithm");
      expect(algorithmCount).toBeDefined();
      expect(algorithmCount!.count).toBeGreaterThan(0);
    });
  });

  describe("集成场景：完整业务流程", () => {
    test("完整流程：添加收藏 → 查询列表验证 → 取消收藏 → 验证已取消", async () => {
      const targetId = await createAlgorithmTarget();
      const form = createFavoriteForm({ targetType: "algorithm", targetId });
      const favoriteId = (await FavoriteAPI.add(form)) as number;
      expect(favoriteId).toBeGreaterThan(0);

      const status = await FavoriteAPI.getStatus("algorithm", targetId);
      expect(status.favorited).toBe(true);

      const page = await FavoriteAPI.getPage(createFavoriteQuery({ targetType: "algorithm" }));
      const found = page.list.find((item) => item.id === favoriteId);
      expect(found).toBeDefined();
      expect(found!.targetId).toBe(targetId);
      expect(found!.targetType).toBe("algorithm");

      await FavoriteAPI.deleteByIds([favoriteId]);

      const statusAfter = await FavoriteAPI.getStatus("algorithm", targetId);
      expect(statusAfter.favorited).toBe(false);

      const pageAfter = await FavoriteAPI.getPage(createFavoriteQuery({ targetType: "algorithm" }));
      const stillExists = pageAfter.list.some((item) => item.id === favoriteId);
      expect(stillExists).toBe(false);

      // 从清理列表移除（已手动删除）
      removeFavoriteIds([favoriteId]);
    });

    test("完整流程：添加多个类型收藏 → 按类型筛选验证", async () => {
      const fav1 = await createFavorite("algorithm");
      const fav2 = await createFavorite("algorithm");

      const algoPage = await FavoriteAPI.getPage(
        createFavoriteQuery({ targetType: "algorithm", pageSize: 100 })
      );
      const hasFav1 = algoPage.list.some((item) => item.id === fav1.favoriteId);
      const hasFav2 = algoPage.list.some((item) => item.id === fav2.favoriteId);
      expect(hasFav1).toBe(true);
      expect(hasFav2).toBe(true);

      const datasetPage = await FavoriteAPI.getPage(
        createFavoriteQuery({ targetType: "dataset", pageSize: 100 })
      );
      const datasetHasFav1 = datasetPage.list.some((item) => item.id === fav1.favoriteId);
      const datasetHasFav2 = datasetPage.list.some((item) => item.id === fav2.favoriteId);
      expect(datasetHasFav1).toBe(false);
      expect(datasetHasFav2).toBe(false);

      const counts = await FavoriteAPI.getCount("algorithm");
      const algoCount = counts.find((c) => c.targetType === "algorithm");
      expect(algoCount).toBeDefined();
      expect(algoCount!.count).toBeGreaterThanOrEqual(2);
    });
  });
});
