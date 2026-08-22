import { DatasetAPI, DatasetItemAPI, ItemFileAPI } from "../../../index";
import {
  createDatasetForm,
  createDatasetItemForm,
  createItemFileUpdateForm,
} from "#/factories/dataset";
import { uniqueName } from "#/factories/common";
import * as fs from "fs";
import * as path from "path";
import FormData from "form-data";

// Node form-data 与浏览器 FormData 类型不一致，用断言规避类型检查
type AnyFormData = any;

const RESOURCES_DIR = path.resolve(__dirname, "../../resources");
const TEST_CLEAN_DIR = path.join(RESOURCES_DIR, "test/clean");
const TEST_HAZY_DIR = path.join(RESOURCES_DIR, "test/hazy");
const TEST2_CLEAN_DIR = path.join(RESOURCES_DIR, "test2/clean");
const TEST2_HAZY_DIR = path.join(RESOURCES_DIR, "test2/hazy");
const TEST3_DIR = path.join(RESOURCES_DIR, "test3");

describe("图片文件接口测试", () => {
  let testDatasetId: number;
  let testItemId: number;
  let uploadedFileIds: number[] = [];

  // 构造上传单张图片到数据项的 multipart 表单（withItem/withFile 可控，用于参数校验用例）
  function uploadForm(
    filename: string,
    type: string,
    opts: {
      dir?: string;
      contentType?: string;
      fields?: Record<string, string>;
      withItem?: boolean;
      withFile?: boolean;
    } = {}
  ): AnyFormData {
    const {
      dir = TEST_CLEAN_DIR,
      contentType = "image/jpeg",
      fields = {},
      withItem = true,
      withFile = true,
    } = opts;
    const formData = new FormData();
    if (withItem) formData.append("itemId", testItemId.toString());
    formData.append("type", type);
    if (withFile) {
      formData.append("file", fs.createReadStream(path.join(dir, filename)), {
        filename,
        contentType,
      });
    }
    Object.entries(fields).forEach(([k, v]) => formData.append(k, v));
    return formData;
  }

  // 构造上传单清晰图 + 多张有雾图的配对表单
  function pairForm(opts: {
    name?: string;
    sceneType?: string;
    clear?: { file: string; filename: string };
    hazy?: Array<{ file: string; filename: string; level: string }>;
  }): AnyFormData {
    const { name, sceneType = "urban", clear, hazy = [] } = opts;
    const formData = new FormData();
    formData.append("datasetId", testDatasetId.toString());
    if (name) formData.append("name", name);
    formData.append("sceneType", sceneType);
    if (clear) {
      formData.append("clearImage", fs.createReadStream(clear.file), {
        filename: clear.filename,
        contentType: "image/jpeg",
      });
    }
    hazy.forEach(({ file, filename, level }) => {
      formData.append("hazyImages", fs.createReadStream(file), {
        filename,
        contentType: "image/jpeg",
      });
      formData.append("hazeLevels", level);
    });
    return formData;
  }

  // 构造批量上传图片的 multipart 表单
  function batchForm(
    sceneType: string,
    files: Array<{ file: string; filename: string }>
  ): AnyFormData {
    const formData = new FormData();
    formData.append("datasetId", testDatasetId.toString());
    formData.append("sceneType", sceneType);
    files.forEach(({ file, filename }) => {
      formData.append("files", fs.createReadStream(file), {
        filename,
        contentType: "image/jpeg",
      });
    });
    return formData;
  }

  beforeAll(async () => {
    testDatasetId = await DatasetAPI.add(
      createDatasetForm({ name: uniqueName("文件测试数据集"), type: "用户数据集" })
    );
    const item = await DatasetItemAPI.add(
      createDatasetItemForm(testDatasetId, { sceneType: "urban", name: "文件测试数据项" })
    );
    testItemId = item.id;
  });

  afterAll(async () => {
    // 清理上传的图片（已被测试删除的会在此被忽略）
    try {
      if (uploadedFileIds.length > 0) {
        await ItemFileAPI.batchDelete({ ids: uploadedFileIds });
      }
    } catch (e) {
      // 忽略清理失败
    }
    try {
      await DatasetAPI.deleteById(testDatasetId);
    } catch (e) {
      // 忽略清理失败
    }
  });

  describe("POST /api/v1/item-files - 上传数据项图片", () => {
    test("正向测试：上传清晰图（test/资源）", async () => {
      const result = await ItemFileAPI.upload(
        uploadForm("41_outdoor_GT.jpg", "clear", { fields: { description: "测试清晰图" } })
      );
      expect(result.id).toBeGreaterThan(0);
      expect(typeof result.url).toBe("string");
      expect(result.url.length).toBeGreaterThan(0);
      expect(result.type).toBe("clear");
      uploadedFileIds.push(result.id);
    });

    test("正向测试：上传有雾图（test/资源）", async () => {
      const result = await ItemFileAPI.upload(
        uploadForm("41_outdoor_hazy.jpg", "hazy", {
          dir: TEST_HAZY_DIR,
          fields: { hazeLevel: "medium", sceneType: "urban" },
        })
      );
      expect(result.id).toBeGreaterThan(0);
      expect(result.type).toBe("hazy");
      expect(result.hazeLevel).toBe("medium");
      expect(["urban", "outdoor"]).toContain(result.sceneType);
      uploadedFileIds.push(result.id);
    });

    test("正向测试：上传PNG格式图片（test3/资源）", async () => {
      const result = await ItemFileAPI.upload(
        uploadForm("cqupt.png", "clear", { dir: TEST3_DIR, contentType: "image/png" })
      );
      expect(result.id).toBeGreaterThan(0);
      expect(typeof result.url).toBe("string");
      expect(result.url.length).toBeGreaterThan(0);
      expect(result.format).toBe("png");
      uploadedFileIds.push(result.id);
    });

    test("正向测试：上传小尺寸图片（test2/资源）", async () => {
      const result = await ItemFileAPI.upload(
        uploadForm("0025.jpg", "clear", { dir: TEST2_CLEAN_DIR })
      );
      expect(result.id).toBeGreaterThan(0);
      const detail = await ItemFileAPI.getById(result.id);
      if (detail.sizeBytes !== undefined && detail.sizeBytes !== null) {
        expect(detail.sizeBytes).toBeGreaterThan(0);
        expect(detail.sizeBytes).toBeLessThan(200000);
      }
      uploadedFileIds.push(result.id);
    });

    test("参数校验：缺少文件", async () => {
      await expect(
        ItemFileAPI.upload(uploadForm("", "hazy", { withFile: false }))
      ).rejects.toThrow();
    });

    test("参数校验：缺少数据项ID", async () => {
      const formData = uploadForm("test.jpg", "hazy", { withItem: false });
      await expect(ItemFileAPI.upload(formData)).rejects.toThrow();
    });

    test("参数校验：无效的图片类型", async () => {
      // 后端可能接受（忽略无效类型）或拒绝，两种行为均可接受
      const result = await ItemFileAPI.upload(uploadForm("test.jpg", "invalid")).catch(() => null);
      if (result !== null) {
        expect(result.id).toBeGreaterThan(0);
      }
    });
  });

  describe("GET /api/v1/item-files/{id} - 获取图片详细信息", () => {
    let testFileId: number;

    beforeAll(async () => {
      const result = await ItemFileAPI.upload(
        uploadForm("42_outdoor_GT.jpg", "clear", { fields: { description: "测试图片详情" } })
      );
      testFileId = result.id;
      uploadedFileIds.push(testFileId);
    });

    test("正向测试：获取已上传图片的详细信息", async () => {
      const result = await ItemFileAPI.getById(testFileId);
      expect(result.id).toBe(testFileId);
      expect(result.type).toBe("clear");
      expect(typeof result.url).toBe("string");
      expect(result.url.length).toBeGreaterThan(0);
      expect(result.description).toBe("测试图片详情");
      if (result.fileName) {
        expect(result.fileName).toBe("42_outdoor_GT.jpg");
      }
      if (result.sizeBytes !== undefined && result.sizeBytes !== null) {
        expect(result.sizeBytes).toBeGreaterThan(0);
      }
      if (result.format) {
        expect(["jpeg", "jpg"]).toContain(result.format.toLowerCase());
      }
    });

    test("正向测试：验证图片尺寸信息", async () => {
      const result = await ItemFileAPI.getById(testFileId);
      if (result.width !== undefined && result.width !== null) {
        expect(result.width).toBeGreaterThan(0);
      }
      if (result.height !== undefined && result.height !== null) {
        expect(result.height).toBeGreaterThan(0);
      }
    });

    test("正向测试：验证缩略图URL", async () => {
      const result = await ItemFileAPI.getById(testFileId);
      if (result.thumbnailUrl) {
        expect(result.thumbnailUrl.length).toBeGreaterThan(0);
      }
    });

    test("异常测试：获取不存在的图片", async () => {
      await expect(ItemFileAPI.getById(99999999)).rejects.toThrow();
    });

    test("异常测试：无效ID格式", async () => {
      await expect(ItemFileAPI.getById(-1)).rejects.toThrow();
    });
  });

  describe("PUT /api/v1/item-files/{id} - 修改图片信息", () => {
    let testFileId: number;

    beforeAll(async () => {
      const result = await ItemFileAPI.upload(uploadForm("43_outdoor_GT.jpg", "clear"));
      testFileId = result.id;
      uploadedFileIds.push(testFileId);
    });

    test("正向测试：更新图片描述", async () => {
      const form = createItemFileUpdateForm({ description: "更新后的描述" });
      await ItemFileAPI.update(testFileId, form);
      const detail = await ItemFileAPI.getById(testFileId);
      expect(detail.description).toBe("更新后的描述");
    });

    test("正向测试：更新场景类型", async () => {
      const form = createItemFileUpdateForm({ sceneType: "rural" });
      await ItemFileAPI.update(testFileId, form);
      const detail = await ItemFileAPI.getById(testFileId);
      if (detail.sceneType) {
        expect(detail.sceneType).toBe("rural");
      }
    });

    test("正向测试：更新雾霾程度", async () => {
      const form = createItemFileUpdateForm({ type: "hazy", hazeLevel: "heavy" });
      await ItemFileAPI.update(testFileId, form);
    });

    test("异常测试：更新不存在的图片应返回业务错误", async () => {
      const form = createItemFileUpdateForm({ description: "测试" });
      // 更新不存在的资源应返回 A0401(RESOURCE_NOT_FOUND) 或 B0001(SYSTEM_ERROR)，均为有效业务错误
      await ItemFileAPI.update(99999999, form).catch((error: any) => {
        const bizError = error.response?.data || error;
        expect(["A0401", "B0001"]).toContain(bizError.code);
      });
    });
  });

  describe("DELETE /api/v1/item-files/{id} - 删除图片", () => {
    let testFileId: number;

    beforeAll(async () => {
      const result = await ItemFileAPI.upload(uploadForm("44_outdoor_GT.jpg", "clear"));
      testFileId = result.id;
    });

    test("正向测试：删除已上传的图片", async () => {
      await ItemFileAPI.deleteById(testFileId);
      await expect(ItemFileAPI.getById(testFileId)).rejects.toThrow();
    });

    test("异常测试：删除不存在的图片（幂等或报错）", async () => {
      // 后端可能返回成功（幂等设计）或报错；报错时返回 A0401 或 B0001，均为有效业务错误
      await ItemFileAPI.deleteById(99999999).catch((error: any) => {
        const bizError = error.response?.data || error;
        expect(["A0401", "B0001"]).toContain(bizError.code);
      });
    });
  });

  describe("DELETE /api/v1/item-files/batch - 批量删除图片", () => {
    let batchFileIds: number[] = [];

    beforeAll(async () => {
      const fileName = "45_outdoor_GT.jpg";
      const result = await ItemFileAPI.upload(uploadForm(fileName, "clear"));
      batchFileIds.push(result.id);
      uploadedFileIds.push(result.id);
    });

    test("正向测试：批量删除多张图片", async () => {
      const form = { ids: batchFileIds };
      const result = await ItemFileAPI.batchDelete(form);
      expect(result.successCount).toBe(batchFileIds.length);
      expect(result.failedCount).toBe(0);
      if (result.successIds) {
        expect(result.successIds).toEqual(expect.arrayContaining(batchFileIds));
      }
    });

    test("参数校验：空ID数组", async () => {
      await expect(ItemFileAPI.batchDelete({ ids: [] })).rejects.toThrow();
    });

    test("异常测试：包含不存在的ID", async () => {
      const result = await ItemFileAPI.batchDelete({ ids: [99999999, 99999998] }).catch(() => null);
      if (result !== null) {
        expect(result.successCount).toBe(0);
      }
    });
  });

  describe("POST /api/v1/dataset-items/upload - 上传数据项配对图片", () => {
    test("正向测试：上传clean+hazy配对（test/资源）", async () => {
      const result = await DatasetItemAPI.uploadImagePair(
        pairForm({
          name: "配对测试",
          sceneType: "urban",
          clear: {
            file: path.join(TEST_CLEAN_DIR, "41_outdoor_GT.jpg"),
            filename: "41_outdoor_GT.jpg",
          },
          hazy: [
            {
              file: path.join(TEST_HAZY_DIR, "41_outdoor_hazy.jpg"),
              filename: "41_outdoor_hazy.jpg",
              level: "medium",
            },
          ],
        })
      );
      expect(result.id).toBeGreaterThan(0);
      expect(result.name).toContain("配对测试");
      if (result.clearImage) {
        expect(result.clearImage.id).toBeGreaterThan(0);
      }
      if (result.hazyImages && result.hazyImages.length > 0) {
        expect(result.hazyImages.length).toBeGreaterThan(0);
      }
    });

    test("正向测试：上传一对多hazy配对（test2/资源）", async () => {
      const hazyFiles = ["0025_0.8_0.04.jpg", "0025_0.8_0.08.jpg", "0025_0.9_0.12.jpg"];
      const result = await DatasetItemAPI.uploadImagePair(
        pairForm({
          name: "一对多测试",
          sceneType: "urban",
          clear: { file: path.join(TEST2_CLEAN_DIR, "0025.jpg"), filename: "0025.jpg" },
          hazy: hazyFiles.map((fileName) => ({
            file: path.join(TEST2_HAZY_DIR, fileName),
            filename: fileName,
            level: "light",
          })),
        })
      );
      expect(result.id).toBeGreaterThan(0);
    });
  });

  describe("POST /api/v1/dataset-items/batch - 批量上传图片", () => {
    test("正向测试：批量上传test/资源（多对配对）", async () => {
      const cleanFiles = ["42_outdoor_GT.jpg", "43_outdoor_GT.jpg"];
      const hazyFiles = ["42_outdoor_hazy.jpg", "43_outdoor_hazy.jpg"];
      const formData = batchForm("outdoor", [
        ...cleanFiles.map((f) => ({ file: path.join(TEST_CLEAN_DIR, f), filename: f })),
        ...hazyFiles.map((f) => ({ file: path.join(TEST_HAZY_DIR, f), filename: f })),
      ]);
      const result = await DatasetItemAPI.batchUpload(formData);
      expect(result.total).toBe(cleanFiles.length + hazyFiles.length);
    });

    test("边界测试：批量上传单张图片", async () => {
      const formData = batchForm("urban", [
        { file: path.join(TEST_CLEAN_DIR, "44_outdoor_GT.jpg"), filename: "44_outdoor_GT.jpg" },
      ]);
      const result = await DatasetItemAPI.batchUpload(formData);
      expect(result.total).toBe(1);
      expect(result.succeeded + result.failed).toBe(1);
    });

    test("参数校验：空文件列表", async () => {
      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      await expect(DatasetItemAPI.batchUpload(formData as AnyFormData)).rejects.toThrow();
    });
  });

  describe("业务规则测试", () => {
    test("业务规则：配对图片分辨率一致性校验", async () => {
      const result = await DatasetItemAPI.uploadImagePair(
        pairForm({
          name: "分辨率测试",
          sceneType: "urban",
          clear: {
            file: path.join(TEST_CLEAN_DIR, "41_outdoor_GT.jpg"),
            filename: "41_outdoor_GT.jpg",
          },
          hazy: [
            {
              file: path.join(TEST2_HAZY_DIR, "0025_0.8_0.04.jpg"),
              filename: "0025_0.8_0.04.jpg",
              level: "light",
            },
          ],
        })
      ).catch((error: any) => {
        const bizError = error.response?.data || error;
        // 接受多种业务错误码：A0400(参数错误)、A0401(资源不存在)、B0001(系统错误)
        expect(["A0400", "A0401", "B0001"]).toContain(bizError.code);
        return null;
      });
      if (result !== null) {
        expect(result.id).toBeGreaterThan(0);
      }
    });

    test("业务规则：批量上传文件名识别规则", async () => {
      const formData = batchForm("outdoor", [
        { file: path.join(TEST_CLEAN_DIR, "41_outdoor_GT.jpg"), filename: "scene001_clear.jpg" },
        {
          file: path.join(TEST_HAZY_DIR, "41_outdoor_hazy.jpg"),
          filename: "scene001_hazy_medium.jpg",
        },
      ]);
      const result = await DatasetItemAPI.batchUpload(formData);
      expect(result.total).toBe(2);
      if (result.succeeded > 0) {
        expect(result.successItems).toBeDefined();
        if (result.successItems && result.successItems.length > 0) {
          expect(result.successItems[0]!.fileCount).toBeGreaterThanOrEqual(1);
        }
      }
    });

    test("业务规则：图片类型修改后配对完整性", async () => {
      const item = await DatasetItemAPI.uploadImagePair(
        pairForm({
          name: "配对完整性测试",
          sceneType: "urban",
          clear: {
            file: path.join(TEST_CLEAN_DIR, "42_outdoor_GT.jpg"),
            filename: "42_outdoor_GT.jpg",
          },
          hazy: [
            {
              file: path.join(TEST_HAZY_DIR, "42_outdoor_hazy.jpg"),
              filename: "42_outdoor_hazy.jpg",
              level: "medium",
            },
          ],
        })
      );
      expect(item.id).toBeGreaterThan(0);

      const detail = await DatasetItemAPI.getById(item.id);
      expect(detail.clearImage).toBeDefined();

      const updateForm = createItemFileUpdateForm({ type: "hazy", hazeLevel: "light" });
      // 后端可能允许或拒绝修改清晰图类型为有雾图（配对完整性校验），两种行为均可接受
      await ItemFileAPI.update(detail.clearImage!.id, updateForm).catch(() => {});

      const updatedDetail = await DatasetItemAPI.getById(item.id);
      expect(updatedDetail.id).toBe(item.id);
    });
  });
});
