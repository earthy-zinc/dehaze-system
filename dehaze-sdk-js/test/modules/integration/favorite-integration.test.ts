import { AlgorithmAPI, ModelAPI, FileAPI, DatasetAPI } from "../../../index";
import FavoriteAPI from "@/api/favorite";
import type { FavoriteTargetType } from "@/api/favorite/model";
import { createFavoriteForm } from "#/factories/favorite";
import { createPredictionForm } from "#/factories/model";
import { login, logout } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import * as fs from "fs";
import * as path from "path";

/**
 * 收藏跨模块集成测试
 *
 * 验证收藏管理模块与算法管理、去雾处理、数据集模块之间的协作。
 * 重点测试不同 targetType（algorithm/result/dataset）的完整收藏生命周期。
 *
 * 收藏目标使用已存在的预置算法/数据集，避免需要管理员权限创建目标。
 */
describe("收藏跨模块集成测试", () => {
  const favoriteIds: number[] = [];
  let uploadedFileId: number;
  let algorithmTargetId: number;
  let datasetTargetId: number;
  /** 标记 dataset 是否由本套件创建，afterAll 只清理自己创建的，避免误删并行套件的数据 */
  let datasetCreated = false;

  beforeAll(async () => {
    await login(USERS.USER.username);

    const uploadFile = async (relativePath: string): Promise<{ id: number; url: string }> => {
      const filePath = path.resolve(__dirname, relativePath);
      const fileData = fs.readFileSync(filePath);
      const blob = new Blob([fileData]);
      const fileName = path.basename(relativePath);
      const formFile = new File([blob], fileName, { type: "image/jpeg" });
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          return await FileAPI.upload(formFile);
        } catch (e: any) {
          const code = e?.response?.data?.code;
          if (code === "A0002" && attempt < 2) {
            // TTL 5 秒，等待 3 秒覆盖一半 TTL，最多 3 次重试（总等待 9 秒）
            await new Promise((resolve) => setTimeout(resolve, 3000));
            continue;
          }
          if (code === "A0501") {
            const page = await FileAPI.getPage({ pageNum: 1, pageSize: 5, keywords: fileName });
            const found = page.list.find((f: any) => f.name === fileName && f.url);
            if (found) {
              return { id: found.id, url: found.url };
            }
            // 查到的文件 url 为空（可能已删除），等待后重试上传
            if (attempt < 2) {
              await new Promise((resolve) => setTimeout(resolve, 3000));
              continue;
            }
            throw new Error(`文件 ${fileName} 已存在但无法在列表中查到有效记录，请检查数据库`);
          }
          throw e;
        }
      }
      return { id: 0, url: "" };
    };

    const hazyInfo = await uploadFile("../../resources/test/model/hazy.jpg");
    uploadedFileId = hazyInfo.id;

    // 预取目标 ID 并清理残留收藏（上次异常中断可能留下未清理的数据）
    const algorithms = await AlgorithmAPI.listAll();
    if (algorithms.length > 0) {
      algorithmTargetId = algorithms[0]!.id;
      await cleanupResidualFavorites("algorithm", algorithmTargetId);
    }

    // 始终创建专用数据集（不复用列表中的，避免并行套件误删别人的 dataset）
    // 创建数据集需要 admin 权限，临时切换到 admin 账号
    await login(USERS.ADMIN.username);
    datasetTargetId = await DatasetAPI.add({
      parentId: 0,
      name: `favorite_test_dataset_${Date.now()}`,
      type: "用户数据集",
      description: "收藏集成测试自动创建",
      status: "1",
    });
    datasetCreated = true;
    // 切回 user 账号进行后续测试
    await login(USERS.USER.username);
    await cleanupResidualFavorites("dataset", datasetTargetId);
  });

  /** 清理指定 targetType+targetId 的残留收藏记录 */
  async function cleanupResidualFavorites(targetType: FavoriteTargetType, targetId: number) {
    try {
      const page = await FavoriteAPI.getPage({
        pageNum: 1,
        pageSize: 100,
        targetType,
      });
      const residual = page.list.filter((item) => item.targetId === targetId);
      if (residual.length > 0) {
        await FavoriteAPI.deleteByIds(residual.map((item) => item.id));
      }
    } catch {
      // 忽略清理错误
    }
  }

  afterAll(async () => {
    if (favoriteIds.length > 0) {
      try {
        await FavoriteAPI.deleteByIds(favoriteIds);
      } catch {}
    }
    // 只清理本套件创建的数据集（避免误删并行套件创建的共享数据）
    // 删除数据集需要 admin 权限，临时切换到 admin 账号
    if (datasetCreated && datasetTargetId) {
      try {
        await login(USERS.ADMIN.username);
        await DatasetAPI.deleteById(datasetTargetId);
      } catch {}
    }

    await logout();
  });

  /**
   * 获取一个已存在的算法 ID 作为收藏目标
   */
  function getAlgorithmTargetId(): number {
    if (!algorithmTargetId) {
      throw new Error("无可用算法作为收藏目标");
    }
    return algorithmTargetId;
  }

  // ============================================================
  // 场景1：算法收藏完整流程
  // ============================================================
  describe("场景1：算法收藏完整流程", () => {
    test("正向流程：收藏算法→验证状态→验证列表→取消收藏", async () => {
      // 1. 获取已存在的算法作为收藏目标
      const targetId = getAlgorithmTargetId();

      // 2. 清理残留收藏后收藏算法
      await cleanupResidualFavorites("algorithm", targetId);
      const form = createFavoriteForm({ targetType: "algorithm", targetId });
      const favoriteId = (await FavoriteAPI.add(form)) as number;
      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);

      // 3. 验证收藏状态
      const status = await FavoriteAPI.getStatus("algorithm", targetId);
      expect(status.favorited).toBe(true);
      expect(status.targetType).toBe("algorithm");
      expect(status.targetId).toBe(targetId);

      // 4. 在收藏列表中验证
      const page = await FavoriteAPI.getPage({
        pageNum: 1,
        pageSize: 100,
        targetType: "algorithm",
      });
      const found = page.list.find((item) => item.id === favoriteId);
      expect(found).toBeDefined();
      expect(found!.targetId).toBe(targetId);
      expect(found!.targetType).toBe("algorithm");

      // 5. 取消收藏
      await FavoriteAPI.deleteByIds([favoriteId]);
      const idx = favoriteIds.indexOf(favoriteId);
      if (idx >= 0) favoriteIds.splice(idx, 1);

      // 6. 验证已取消
      const statusAfter = await FavoriteAPI.getStatus("algorithm", targetId);
      expect(statusAfter.favorited).toBe(false);

      const pageAfter = await FavoriteAPI.getPage({
        pageNum: 1,
        pageSize: 100,
        targetType: "algorithm",
      });
      const stillExists = pageAfter.list.some((item) => item.id === favoriteId);
      expect(stillExists).toBe(false);
    });
  });

  // ============================================================
  // 场景2：处理结果收藏完整流程
  // ============================================================
  describe("场景2：处理结果收藏完整流程", () => {
    test("正向流程：去雾处理→收藏结果→验证列表→取消收藏", async () => {
      // 1. 执行去雾处理
      const predForm = createPredictionForm({
        algorithmId: 13,
        fileId: uploadedFileId,
      });
      const predResult = await ModelAPI.predictAndWait(predForm, {
        intervalMs: 2000,
        timeoutMs: 120000,
      });
      expect(predResult.status).toBe(2);
      expect(predResult.logId).toBeDefined();

      // 2. 收藏处理结果
      const form = createFavoriteForm({
        targetType: "result",
        targetId: predResult.logId!,
      });
      const favoriteId = (await FavoriteAPI.add(form)) as number;
      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);

      // 3. 验证收藏列表
      const page = await FavoriteAPI.getPage({
        pageNum: 1,
        pageSize: 100,
        targetType: "result",
      });
      const found = page.list.find((item) => item.id === favoriteId);
      expect(found).toBeDefined();
      expect(found!.targetType).toBe("result");

      // 4. 验证收藏状态
      const status = await FavoriteAPI.getStatus("result", predResult.logId!);
      expect(status.favorited).toBe(true);

      // 5. 取消收藏
      await FavoriteAPI.deleteByIds([favoriteId]);
      const idx = favoriteIds.indexOf(favoriteId);
      if (idx >= 0) favoriteIds.splice(idx, 1);

      // 6. 验证已取消
      const statusAfter = await FavoriteAPI.getStatus("result", predResult.logId!);
      expect(statusAfter.favorited).toBe(false);
    });
  });

  // ============================================================
  // 场景3：数据集收藏完整流程
  // ============================================================
  describe("场景3：数据集收藏完整流程", () => {
    test("正向流程：获取数据集→收藏→验证列表→取消收藏", async () => {
      // 1. 复用 beforeAll 中预取的 datasetTargetId（预置数据集不会被并行套件删除）
      if (!datasetTargetId) {
        throw new Error("无可用数据集作为收藏目标");
      }
      const targetId = datasetTargetId;

      // 2. 清理残留收藏后收藏数据集
      await cleanupResidualFavorites("dataset", targetId);
      const form = createFavoriteForm({ targetType: "dataset", targetId });
      const favoriteId = (await FavoriteAPI.add(form)) as number;
      expect(favoriteId).toBeGreaterThan(0);
      favoriteIds.push(favoriteId);

      // 3. 验证收藏列表
      const page = await FavoriteAPI.getPage({
        pageNum: 1,
        pageSize: 100,
        targetType: "dataset",
      });
      const found = page.list.find((item) => item.id === favoriteId);
      expect(found).toBeDefined();
      expect(found!.targetType).toBe("dataset");
      expect(found!.targetId).toBe(targetId);

      // 4. 验证收藏状态
      const status = await FavoriteAPI.getStatus("dataset", targetId);
      expect(status.favorited).toBe(true);

      // 5. 取消收藏
      await FavoriteAPI.deleteByIds([favoriteId]);
      const idx = favoriteIds.indexOf(favoriteId);
      if (idx >= 0) favoriteIds.splice(idx, 1);

      // 6. 验证已取消
      const statusAfter = await FavoriteAPI.getStatus("dataset", targetId);
      expect(statusAfter.favorited).toBe(false);
    });
  });
});
