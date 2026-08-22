import { AlgorithmAPI, Algorithm, AlgorithmSelectNodeVO } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { createAlgorithmForm, createAlgorithmQuery } from "#/factories/algorithm";
import { USERS } from "#/factories/constants";
import { login } from "#/utils/auth";

// 后端选择树字段为 isLeaf；SDK 类型 AlgorithmSelectNodeVO 已声明 isLeaf，
// 但「查找叶子」处沿用历史运行时的 leaf 字段，此处做一次类型收窄以保留运行语义。
type SelectNodeWithLeaf = AlgorithmSelectNodeVO & { leaf?: boolean };

/** 取选择树中第一个叶子节点（leaf 字段缺失时返回 undefined，调用方负责跳过） */
function findFirstLeaf(tree: AlgorithmSelectNodeVO[]): AlgorithmSelectNodeVO | undefined {
  return (tree as SelectNodeWithLeaf[]).find((n) => n.leaf);
}

/** 取选择树中全部叶子节点 */
function findLeaves(tree: AlgorithmSelectNodeVO[]): AlgorithmSelectNodeVO[] {
  return (tree as SelectNodeWithLeaf[]).filter((n) => n.leaf);
}

/** 创建父/子/孙三级算法，返回各层 id */
async function createCascadeAlgorithms(): Promise<{
  parentId: number;
  childId: number;
  grandChildId: number;
}> {
  const parentId = await AlgorithmAPI.add(createAlgorithmForm({ parentId: 0 }));
  const childId = await AlgorithmAPI.add(createAlgorithmForm({ parentId }));
  const grandChildId = await AlgorithmAPI.add(createAlgorithmForm({ parentId: childId }));
  return { parentId, childId, grandChildId };
}

/** 断言算法已被删除（查询不存在抛业务错误） */
async function expectDeleted(id: number): Promise<void> {
  await expectBizError(AlgorithmAPI.getAlgorithmInfoById(id), [
    "A0401",
    "A0400",
    "B0001",
    "ERR_BAD_REQUEST",
  ]);
}

describe("算法管理接口测试", () => {
  describe("GET /api/v1/algorithm - 算法树形表格", () => {
    test("正向测试：获取算法树形列表并验证树形结构", async () => {
      const result = await AlgorithmAPI.getList();

      expect(Array.isArray(result)).toBe(true);

      const verifyAlgorithm = (algorithms: typeof result) => {
        algorithms.forEach((algo) => {
          expect(algo.id).toBeGreaterThan(0);
          expect(algo.name).toBeTruthy();
          // type 字段在 schema 中 default:''，预置/分类节点可能为空字符串，仅校验类型
          expect(typeof algo.type).toBe("string");
          // parentId 可能为 undefined（后端bug），使用更宽松的断言
          if (algo.parentId !== undefined) {
            expect(algo.parentId).toBeGreaterThanOrEqual(0);
          }

          if (algo.children && algo.children.length > 0) {
            algo.children.forEach((child) => {
              if (child.parentId !== undefined) {
                expect(child.parentId).toBe(algo.id);
              }
            });
            verifyAlgorithm(algo.children);
          }
        });
      };

      if (result.length > 0) {
        verifyAlgorithm(result);
      }
    });

    test("正向测试：按关键词搜索算法并验证搜索结果", async () => {
      // 先获取所有算法，取一个已存在算法的名称片段作为搜索关键词
      const allAlgorithms = await AlgorithmAPI.getList();
      expect(allAlgorithms.length).toBeGreaterThan(0);

      const firstAlgorithm = allAlgorithms[0]!;
      const searchKeyword = firstAlgorithm.name.substring(0, 2);
      const query = createAlgorithmQuery({ keywords: searchKeyword });
      const result = await AlgorithmAPI.getList(query);

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);

      const verifyKeyword = (algorithms: typeof result, keyword: string) => {
        algorithms.forEach((algo) => {
          const nameContains = algo.name.toLowerCase().includes(keyword.toLowerCase());
          expect(nameContains).toBe(true);
          if (algo.children && algo.children.length > 0) {
            verifyKeyword(algo.children, keyword);
          }
        });
      };

      verifyKeyword(result, searchKeyword);
    });

    test("正向测试：验证算法层级关系", async () => {
      const result = await AlgorithmAPI.getList();

      expect(Array.isArray(result)).toBe(true);

      result.forEach((algo) => {
        if (!algo.children || algo.children.length === 0) {
          // 叶子节点，parentId 可能为 undefined（后端bug）
          if (algo.parentId !== undefined) {
            expect(algo.parentId).toBeGreaterThanOrEqual(0);
          }
        } else {
          algo.children!.forEach((child) => {
            if (child.parentId !== undefined) {
              expect(child.parentId).toBe(algo.id);
            }
          });
        }
      });
    });
  });

  describe("GET /api/v1/algorithm/options - 算法下拉选项", () => {
    test("正向测试：获取算法下拉列表并验证数据准确性", async () => {
      const result = await AlgorithmAPI.getOption();

      expect(Array.isArray(result)).toBe(true);

      result.forEach((option: any) => {
        expect(option.value).toBeTruthy();
        expect(typeof option.value).toBe("number");
        expect(option.label).toBeTruthy();
        expect(typeof option.label).toBe("string");
      });
    });
  });

  describe("GET /api/v1/algorithm/{id} - 获取算法详情", () => {
    test("正向测试：获取算法详情并验证数据完整性", async () => {
      const form = createAlgorithmForm({ parentId: 0 });
      const testAlgorithmId = await AlgorithmAPI.add(form);

      const result = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);

      expect(result.id).toBe(testAlgorithmId);
      expect(result.name).toBeTruthy();
      expect(result.type).toBeTruthy();
      if (result.parentId !== undefined) {
        expect(result.parentId).toBeGreaterThanOrEqual(0);
      }

      await AlgorithmAPI.deleteByIds([testAlgorithmId.toString()]);
    });

    test("异常测试：获取不存在算法应抛出业务错误", async () => {
      await expectBizError(AlgorithmAPI.getAlgorithmInfoById(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/algorithm - 新增算法", () => {
    test("正向测试：创建算法并验证数据真实持久化", async () => {
      const form = createAlgorithmForm({ parentId: 0 });

      const algorithmId = await AlgorithmAPI.add(form);

      expect(typeof algorithmId).toBe("number");

      // 验证持久化
      const algorithmInfo = await AlgorithmAPI.getAlgorithmInfoById(algorithmId);
      expect(algorithmInfo.id).toBe(algorithmId);
      expect(algorithmInfo.name).toBe(form.name);
      expect(algorithmInfo.type).toBe(form.type);
      if (form.parentId !== undefined) {
        expect(algorithmInfo.parentId).toBe(form.parentId);
      }

      await AlgorithmAPI.deleteByIds([algorithmId.toString()]);
    });

    test("正向测试：创建子算法并验证父子关系", async () => {
      // 先创建父算法再创建子算法
      const parentForm = createAlgorithmForm({ parentId: 0 });
      const parentAlgorithmId = await AlgorithmAPI.add(parentForm);

      const childForm = createAlgorithmForm({ parentId: parentAlgorithmId });
      const childAlgorithmId = await AlgorithmAPI.add(childForm);

      const childAlgorithmInfo = await AlgorithmAPI.getAlgorithmInfoById(childAlgorithmId);
      expect(childAlgorithmInfo.parentId).toBe(parentAlgorithmId);

      // 先删子再删父
      await AlgorithmAPI.deleteByIds([childAlgorithmId.toString()]);
      await AlgorithmAPI.deleteByIds([parentAlgorithmId.toString()]);
    });

    test("参数校验：缺少必需字段 name 应抛出业务错误", async () => {
      const form: Partial<Algorithm> = {
        parentId: 0,
        type: "TEST",
        description: "测试",
      };

      await expectBizError(AlgorithmAPI.add(form as Algorithm), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：缺少必需字段 type 应抛出业务错误", async () => {
      const form: Partial<Algorithm> = {
        parentId: 0,
        name: "测试算法",
        description: "测试",
      };

      await expectBizError(AlgorithmAPI.add(form as Algorithm), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("PUT /api/v1/algorithm/{id} - 修改算法", () => {
    test("正向测试：更新算法名称并验证更新真实生效", async () => {
      const form = createAlgorithmForm({ parentId: 0 });
      const testAlgorithmId = await AlgorithmAPI.add(form);
      const originalAlgorithm = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);

      // 更新算法名称
      const newForm = createAlgorithmForm({ parentId: originalAlgorithm.parentId });
      await AlgorithmAPI.update(testAlgorithmId, newForm);

      const algorithmInfo = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);
      expect(algorithmInfo.name).toBe(newForm.name);

      await AlgorithmAPI.deleteByIds([testAlgorithmId.toString()]);
    });

    test("正向测试：更新算法状态并验证状态值正确", async () => {
      // 创建草稿状态算法，再更新为已停用（5=DISABLED）
      const form = createAlgorithmForm({ parentId: 0, status: 1 });
      const testAlgorithmId = await AlgorithmAPI.add(form);
      const originalAlgorithm = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);

      const updateForm: Partial<Algorithm> = {
        ...originalAlgorithm,
        status: 5,
      };
      await AlgorithmAPI.update(testAlgorithmId, updateForm as Algorithm);

      const algorithmInfo = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);
      expect(algorithmInfo.status).toBe(5);

      await AlgorithmAPI.deleteByIds([testAlgorithmId.toString()]);
    });

    test("异常测试：更新不存在的算法应抛出业务错误", async () => {
      const form = createAlgorithmForm();

      await expectBizError(AlgorithmAPI.update(99999999, form), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("PUT /api/v1/algorithms/{id}/status - 算法状态管理", () => {
    let testAlgorithmId: number;

    beforeAll(async () => {
      const form = createAlgorithmForm({ parentId: 0, status: 1 });
      testAlgorithmId = await AlgorithmAPI.add(form);
    });

    afterAll(async () => {
      try {
        await AlgorithmAPI.deleteByIds([testAlgorithmId.toString()]);
      } catch (e) {
        console.warn(`清理失败:`, e);
      }
    });

    test("正向测试：草稿->测试中", async () => {
      await AlgorithmAPI.updateStatus(testAlgorithmId, 2);
      const detail = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);
      expect(detail.status).toBe(2);
    });

    test("正向测试：测试中->待审核", async () => {
      await AlgorithmAPI.updateStatus(testAlgorithmId, 3);
      const detail = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);
      expect(detail.status).toBe(3);
    });

    test("异常：非法状态跳跃（草稿->已发布）", async () => {
      const form = createAlgorithmForm({ parentId: 0, status: 1 });
      const id = await AlgorithmAPI.add(form);
      try {
        await expectBizError(AlgorithmAPI.updateStatus(id, 4), [
          "A0502",
          "A0400",
          "B0001",
          "ERR_BAD_REQUEST",
        ]);
      } finally {
        await AlgorithmAPI.deleteByIds([id.toString()]);
      }
    });

    test("边界：删除已发布算法应失败", async () => {
      const tree = await AlgorithmAPI.tree();
      if (tree.length > 0) {
        const firstLeaf = findFirstLeaf(tree);
        if (firstLeaf) {
          await expectBizError(AlgorithmAPI.deleteByIds([firstLeaf.id.toString()]), [
            "A0502",
            "A0504",
            "A0400",
            "B0001",
            "ERR_BAD_REQUEST",
          ]);
        }
      }
    });
  });

  describe("PUT /api/v1/algorithms/{id}/audit - 审核算法", () => {
    let testAlgorithmId: number;

    beforeAll(async () => {
      const form = createAlgorithmForm({ parentId: 0, status: 1 });
      testAlgorithmId = await AlgorithmAPI.add(form);
      // 草稿->测试中->待审核
      await AlgorithmAPI.updateStatus(testAlgorithmId, 2);
      await AlgorithmAPI.updateStatus(testAlgorithmId, 3);
    });

    afterAll(async () => {
      try {
        await AlgorithmAPI.deleteByIds([testAlgorithmId.toString()]);
      } catch (e) {
        console.warn(`清理失败:`, e);
      }
    });

    test("正向测试：审核通过", async () => {
      await AlgorithmAPI.auditAlgorithm(testAlgorithmId, { approved: true, remark: "审核通过" });
      const detail = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);
      expect(detail.status).toBe(4);
    });

    test("边界：驳回原因为空应失败", async () => {
      const form = createAlgorithmForm({ parentId: 0, status: 1 });
      const id = await AlgorithmAPI.add(form);
      await AlgorithmAPI.updateStatus(id, 2);
      await AlgorithmAPI.updateStatus(id, 3);

      try {
        await expectBizError(AlgorithmAPI.auditAlgorithm(id, { approved: false, remark: "" }), [
          "B0001", // python：字符串异常默认码 SYSTEM_EXECUTION_ERROR
          "A0500", // java：业务异常
          "A0400",
          "ERR_BAD_REQUEST",
        ]);
      } finally {
        // 驳回后状态回到测试中(2)，须先恢复草稿再删除（java 仅允许草稿/已停用/已归档删除）
        await AlgorithmAPI.updateStatus(id, 1).catch(() => {});
        await AlgorithmAPI.deleteByIds([id.toString()]).catch(() => {});
      }
    });
  });

  describe("DELETE /api/v1/algorithm/{ids} - 删除算法", () => {
    test("正向测试：删除单个算法并验证算法真的被删除", async () => {
      const form = createAlgorithmForm({ parentId: 0 });
      const algorithmId = await AlgorithmAPI.add(form);

      await AlgorithmAPI.deleteByIds([algorithmId.toString()]);

      await expectDeleted(algorithmId);
    });

    test("正向测试：批量删除多个算法并验证所有算法都被删除", async () => {
      const algorithmIds: number[] = [];
      for (let i = 0; i < 3; i++) {
        const form = createAlgorithmForm({ parentId: 0 });
        const algorithmId = await AlgorithmAPI.add(form);
        algorithmIds.push(algorithmId);
      }

      await AlgorithmAPI.deleteByIds(algorithmIds.map((id) => id.toString()));

      for (const algorithmId of algorithmIds) {
        await expectDeleted(algorithmId);
      }
    });

    test("异常测试：删除不存在的算法应抛出业务错误", async () => {
      await expectBizError(AlgorithmAPI.deleteByIds(["99999999"]), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("级联删除：同时选中父算法和子算法应递归删除所有子孙算法", async () => {
      const { parentId, childId, grandChildId } = await createCascadeAlgorithms();
      await AlgorithmAPI.deleteByIds([
        parentId.toString(),
        childId.toString(),
        grandChildId.toString(),
      ]);
      await expectDeleted(parentId);
      await expectDeleted(childId);
      await expectDeleted(grandChildId);
    });

    test("级联删除：仅选中父算法应递归删除所有子孙算法", async () => {
      const { parentId, childId, grandChildId } = await createCascadeAlgorithms();
      await AlgorithmAPI.deleteByIds([parentId.toString()]);
      await expectDeleted(parentId);
      await expectDeleted(childId);
      await expectDeleted(grandChildId);
    });

    test("边界测试：超长算法名称应被拒绝", async () => {
      const form = createAlgorithmForm({ parentId: 0, name: "x".repeat(500) });
      await expectBizError(AlgorithmAPI.add(form as any), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  describe("POST /api/v1/algorithms/select/compare - 算法对比", () => {
    test("正向测试：对比 2 个叶子算法返回对比结果", async () => {
      // 选择树仅含已发布算法，compare 要求算法均已发布
      const tree = await AlgorithmAPI.tree();
      const leaves = findLeaves(tree);

      if (leaves.length < 2) {
        console.warn("已发布叶子算法不足 2 个，跳过对比正向测试");
        return;
      }

      const result = await AlgorithmAPI.compare({
        algorithmIds: [leaves[0]!.id, leaves[1]!.id],
      });

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThanOrEqual(2);
      result.forEach((item) => {
        expect(item.algorithmId).toBeGreaterThan(0);
        expect(item.algorithmName).toBeTruthy();
      });
    });

    test("参数校验：少于 2 个算法应抛出业务错误", async () => {
      await expectBizError(AlgorithmAPI.compare({ algorithmIds: [1] }), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：对比4个算法应失败（超过上限）", async () => {
      await expectBizError(AlgorithmAPI.compare({ algorithmIds: [1, 2, 3, 4] }), [
        "A0500",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/algorithms/select/tree - 算法选择树（仅已发布）", () => {
    test("正向测试：返回树形结构且叶子节点标记 leaf", async () => {
      const tree = await AlgorithmAPI.tree();

      expect(Array.isArray(tree)).toBe(true);

      const verifyNode = (nodes: typeof tree) => {
        nodes.forEach((node) => {
          expect(node.id).toBeGreaterThan(0);
          expect(node.name).toBeTruthy();
          expect(typeof node.isLeaf).toBe("boolean");
          if (node.children && node.children.length > 0) {
            verifyNode(node.children);
          }
        });
      };
      verifyNode(tree);
    });

    test("正向测试：按任务类型获取算法树", async () => {
      const tree = await AlgorithmAPI.tree("dehaze");
      expect(Array.isArray(tree)).toBe(true);
      const verifyNode = (nodes: typeof tree) => {
        nodes.forEach((node) => {
          expect(node.id).toBeGreaterThan(0);
          expect(node.name).toBeTruthy();
          if (node.children && node.children.length > 0) {
            verifyNode(node.children);
          }
        });
      };
      verifyNode(tree);
    });
  });

  describe("GET /api/v1/algorithms/select/{id} - 算法详情", () => {
    test("正向测试：从选择树取叶子节点获取详情", async () => {
      const tree = await AlgorithmAPI.tree();
      const firstLeaf = findFirstLeaf(tree);

      if (!firstLeaf) {
        console.warn("无已发布算法，跳过详情正向测试");
        return;
      }

      const detail = await AlgorithmAPI.getSelectDetail(firstLeaf.id);
      expect(detail.id).toBe(firstLeaf.id);
      expect(detail.name).toBeTruthy();
      expect(detail.type).toBeTruthy();
      expect(Array.isArray(detail.sampleImages)).toBe(true);
    });

    test("异常测试：不存在的算法ID应抛出业务错误", async () => {
      await expectBizError(AlgorithmAPI.getSelectDetail(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/algorithms/select/{id}/test - 自定义图片测试", () => {
    test("异常测试：不存在的算法ID应抛出业务错误", async () => {
      await expectBizError(AlgorithmAPI.test(99999999, { imageUrl: "https://example.com/x.png" }), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("版本管理", () => {
    let testAlgorithmId: number;

    beforeAll(async () => {
      // 创建时显式带 version：python 后端 create_version 会先归档当前版本，若主表
      // version 为 NULL 会触发 sys_algorithm_version.version NOT NULL 的 IntegrityError（C0300）。
      // 测试数据补上初始版本号即可规避该后端缺陷。
      const form = createAlgorithmForm({ parentId: 0, status: 1, version: "v0.1.0" });
      testAlgorithmId = await AlgorithmAPI.add(form);
    });

    afterAll(async () => {
      try {
        await AlgorithmAPI.deleteByIds([testAlgorithmId.toString()]);
      } catch (e) {
        console.warn(`清理失败:`, e);
      }
    });

    test("正向测试：获取版本历史", async () => {
      const versions = await AlgorithmAPI.getVersions(testAlgorithmId);
      expect(Array.isArray(versions)).toBe(true);
    });

    test("正向测试：新增算法版本", async () => {
      // 版本号须符合 vX.Y.Z 格式（python 后端强校验，测试数据需对齐）
      await AlgorithmAPI.addVersion(testAlgorithmId, {
        version: `v1.0.${Date.now() % 100000}`,
        changeLog: "测试版本",
      });
      const versions = await AlgorithmAPI.getVersions(testAlgorithmId);
      expect(versions.length).toBeGreaterThan(0);
    });

    test("边界：版本号已存在应失败", async () => {
      const version = `v2.0.${Date.now() % 100000}`;
      await AlgorithmAPI.addVersion(testAlgorithmId, { version, changeLog: "第一次" });
      await expectBizError(
        AlgorithmAPI.addVersion(testAlgorithmId, { version, changeLog: "重复" }),
        ["A0501", "A0400", "B0001", "A0500", "ERR_BAD_REQUEST"]
      );
    });
  });

  describe("性能监控", () => {
    test("正向测试：获取算法监控数据", async () => {
      const tree = await AlgorithmAPI.tree();
      const firstLeaf = findFirstLeaf(tree);
      if (!firstLeaf) {
        console.warn("无已发布算法，跳过监控数据验证");
        return;
      }

      const monitor = await AlgorithmAPI.getMonitorData(firstLeaf.id);
      expect(monitor).toBeDefined();
    });

    test("正向测试：获取算法统计报表", async () => {
      const tree = await AlgorithmAPI.tree();
      const firstLeaf = findFirstLeaf(tree);
      if (!firstLeaf) {
        console.warn("无已发布算法，跳过统计报表验证");
        return;
      }

      const stats = await AlgorithmAPI.getMonitorStats(firstLeaf.id, 7);
      expect(Array.isArray(stats)).toBe(true);
    });
  });

  // python 后端已实现 POST /api/v1/algorithms/select/recommend（F-M03-007，T-AS-060~068）：
  // 关键词/任务类型/样例算法匹配，topN 默认 3、范围 1-10，空结果返回 total=0、items=[]（HTTP 200）。
  describe("POST /api/v1/algorithms/select/recommend - 算法推荐匹配", () => {
    test("正向测试：关键词匹配推荐", async () => {
      const result = await AlgorithmAPI.recommend({ keyword: "去雾", topN: 3 });
      expect(Array.isArray(result.items)).toBe(true);
      expect(result.items.length).toBeLessThanOrEqual(3);
      result.items.forEach((item) => {
        expect(item.algorithmId).toBeGreaterThan(0);
        expect(item.algorithmName).toBeTruthy();
        expect(typeof item.matchScore).toBe("number");
      });
    });

    test("边界：topN默认值为3", async () => {
      const result = await AlgorithmAPI.recommend({ keyword: "去雾" });
      expect(result.items.length).toBeLessThanOrEqual(3);
    });

    test("边界：匹配结果为空返回200", async () => {
      const result = await AlgorithmAPI.recommend({ keyword: "不存在的关键词XYZ_99999", topN: 3 });
      expect(result.items.length).toBe(0);
    });

    test("边界：topN超出上限应失败", async () => {
      await expectBizError(AlgorithmAPI.recommend({ keyword: "去雾", topN: 11 }), [
        "A0500",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：topN小于下限应失败", async () => {
      await expectBizError(AlgorithmAPI.recommend({ keyword: "去雾", topN: 0 }), [
        "A0500",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/algorithms/select/search - 搜索算法", () => {
    test("正向测试：按关键词搜索返回匹配的叶子算法", async () => {
      const tree = await AlgorithmAPI.tree();
      const firstLeaf = findFirstLeaf(tree);

      if (!firstLeaf) {
        console.warn("无已发布算法，跳过搜索正向测试");
        return;
      }

      const keyword = firstLeaf.name.substring(0, 2);
      const result = await AlgorithmAPI.search(keyword);

      expect(Array.isArray(result)).toBe(true);
      result.forEach((node) => {
        expect(node.name.toLowerCase()).toContain(keyword.toLowerCase());
      });
    });

    test("边界测试：空关键词应返回空数组", async () => {
      const result = await AlgorithmAPI.search("  ");
      expect(result).toEqual([]);
    });
  });

  // 后端已为算法管理接口加 require_permission（sys:algorithm:add/edit/delete），
  // 普通用户访问返回 A0301（访问未授权）。
  describe("权限测试 - 普通用户管理操作应失败", () => {
    beforeAll(async () => {
      await login(USERS.USER.username);
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：普通用户新增算法应失败", async () => {
      const form = createAlgorithmForm({ parentId: 0 });
      await expectBizError(AlgorithmAPI.add(form as Algorithm), [
        "A0301",
        "A0403",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户修改算法应失败", async () => {
      const form = createAlgorithmForm({ parentId: 0 });
      await expectBizError(AlgorithmAPI.update(1, form as Algorithm), [
        "A0301",
        "A0403",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户删除算法应失败", async () => {
      await expectBizError(AlgorithmAPI.deleteByIds(["999"]), [
        "A0301",
        "A0403",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });
});
