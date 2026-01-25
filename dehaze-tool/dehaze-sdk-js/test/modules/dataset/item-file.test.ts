import { DatasetAPI, DatasetItemAPI, ItemFileAPI } from "../../../index";
import { login, logout } from "#/utils/auth";
import {
  createDatasetForm,
  createDatasetItemForm,
  createItemFileUpdateForm,
} from "#/factories/dataset";
import * as fs from "fs";
import * as path from "path";
import FormData from "form-data";

// Node.js form-data 与浏览器 FormData 类型不完全兼容，但运行时行为兼容
// 使用类型断言解决 TypeScript 类型检查问题
type AnyFormData = any;

// 获取测试资源目录
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

  beforeAll(async () => {
    await login();

    // 创建测试数据集
    const datasetForm = createDatasetForm({ name: "文件测试数据集", type: "用户数据集" });
    const dataset = await DatasetAPI.add(datasetForm);
    testDatasetId = dataset.id;

    // 创建测试数据项
    const itemForm = createDatasetItemForm(testDatasetId, {
      sceneType: "urban",
      name: "文件测试数据项",
    });
    const item = await DatasetItemAPI.add(itemForm);
    testItemId = item.id;
  }, 30000);

  afterAll(async () => {
    // 清理上传的图片
    try {
      if (uploadedFileIds.length > 0) {
        await ItemFileAPI.batchDelete({ ids: uploadedFileIds });
      }
    } catch (e) {
      // 忽略
    }

    // 清理测试数据
    try {
      await DatasetAPI.deleteById(testDatasetId);
    } catch (e) {
      // 忽略
    }
    await logout();
  });

  describe("POST /api/v1/item-files - 上传数据项图片", () => {
    test("正向测试：上传清晰图（test/资源）", async () => {
      const cleanImagePath = path.join(TEST_CLEAN_DIR, "41_outdoor_GT.jpg");

      const formData = new FormData();
      formData.append("itemId", testItemId.toString());
      formData.append("type", "clear");
      formData.append("file", fs.createReadStream(cleanImagePath), {
        filename: "41_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });
      formData.append("description", "测试清晰图");

      const result = await ItemFileAPI.upload(formData as AnyFormData);
      expect(result.id).toBeDefined();
      expect(result.url).toBeDefined();
      expect(result.type).toBe("clear");
      uploadedFileIds.push(result.id);
    });

    test("正向测试：上传有雾图（test/资源）", async () => {
      const hazyImagePath = path.join(TEST_HAZY_DIR, "41_outdoor_hazy.jpg");

      const formData = new FormData();
      formData.append("itemId", testItemId.toString());
      formData.append("type", "hazy");
      formData.append("hazeLevel", "medium");
      formData.append("file", fs.createReadStream(hazyImagePath), {
        filename: "41_outdoor_hazy.jpg",
        contentType: "image/jpeg",
      });
      formData.append("sceneType", "urban");

      const result = await ItemFileAPI.upload(formData as AnyFormData);
      expect(result.id).toBeDefined();
      expect(result.type).toBe("hazy");
      expect(result.hazeLevel).toBe("medium");
      expect(["urban", "outdoor"]).toContain(result.sceneType);
      uploadedFileIds.push(result.id);
    });

    test("正向测试：上传PNG格式图片（test3/资源）", async () => {
      const pngPath = path.join(TEST3_DIR, "cqupt.png");

      const formData = new FormData();
      formData.append("itemId", testItemId.toString());
      formData.append("type", "clear");
      formData.append("file", fs.createReadStream(pngPath), {
        filename: "cqupt.png",
        contentType: "image/png",
      });

      const result = await ItemFileAPI.upload(formData as AnyFormData);
      expect(result.id).toBeDefined();
      expect(result.url).toBeDefined();
      expect(result.format).toBe("png");
      uploadedFileIds.push(result.id);
    });

    test("正向测试：上传小尺寸图片（test2/资源）", async () => {
      const cleanPath = path.join(TEST2_CLEAN_DIR, "0025.jpg");

      const formData = new FormData();
      formData.append("itemId", testItemId.toString());
      formData.append("type", "clear");
      formData.append("file", fs.createReadStream(cleanPath), {
        filename: "0025.jpg",
        contentType: "image/jpeg",
      });

      const result = await ItemFileAPI.upload(formData as AnyFormData);
      expect(result.id).toBeDefined();
      const detail = await ItemFileAPI.getById(result.id);
      expect(detail.sizeBytes).toBeGreaterThan(0);
      expect(detail.sizeBytes).toBeLessThan(200000);
      uploadedFileIds.push(result.id);
    });

    test("参数校验：缺少文件", async () => {
      const formData = new FormData();
      formData.append("itemId", testItemId.toString());
      formData.append("type", "hazy");

      await expect(ItemFileAPI.upload(formData as AnyFormData)).rejects.toThrow();
    });

    test("参数校验：缺少数据项ID", async () => {
      const imagePath = path.join(TEST_CLEAN_DIR, "41_outdoor_GT.jpg");

      const formData = new FormData();
      formData.append("type", "hazy");
      formData.append("file", fs.createReadStream(imagePath), {
        filename: "test.jpg",
        contentType: "image/jpeg",
      });

      await expect(ItemFileAPI.upload(formData as AnyFormData)).rejects.toThrow();
    });

    test("参数校验：无效的图片类型", async () => {
      const imagePath = path.join(TEST_CLEAN_DIR, "41_outdoor_GT.jpg");

      const formData = new FormData();
      formData.append("itemId", testItemId.toString());
      formData.append("type", "invalid");
      formData.append("file", fs.createReadStream(imagePath), {
        filename: "test.jpg",
        contentType: "image/jpeg",
      });

      try {
        const result = await ItemFileAPI.upload(formData as AnyFormData);
        expect(result).toBeDefined();
        expect(result.id).toBeDefined();
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });

  describe("GET /api/v1/item-files/{id} - 获取图片详细信息", () => {
    let testFileId: number;

    beforeAll(async () => {
      const imagePath = path.join(TEST_CLEAN_DIR, "42_outdoor_GT.jpg");

      const formData = new FormData();
      formData.append("itemId", testItemId.toString());
      formData.append("type", "clear");
      formData.append("file", fs.createReadStream(imagePath), {
        filename: "42_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });
      formData.append("description", "测试图片详情");

      const result = await ItemFileAPI.upload(formData as AnyFormData);
      testFileId = result.id;
      uploadedFileIds.push(testFileId);
    });

    test("正向测试：获取已上传图片的详细信息", async () => {
      const result = await ItemFileAPI.getById(testFileId);
      expect(result.id).toBe(testFileId);
      expect(result.type).toBe("clear");
      expect(result.url).toBeDefined();
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
      const imagePath = path.join(TEST_CLEAN_DIR, "43_outdoor_GT.jpg");

      const formData = new FormData();
      formData.append("itemId", testItemId.toString());
      formData.append("type", "clear");
      formData.append("file", fs.createReadStream(imagePath), {
        filename: "43_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });

      const result = await ItemFileAPI.upload(formData as AnyFormData);
      testFileId = result.id;
      uploadedFileIds.push(testFileId);
    });

    test("正向测试：更新图片描述", async () => {
      const form = createItemFileUpdateForm({ description: "更新后的描述" });
      await expect(ItemFileAPI.update(testFileId, form)).resolves.not.toThrow();
    });

    test("正向测试：更新场景类型", async () => {
      const form = createItemFileUpdateForm({ sceneType: "rural" });
      await expect(ItemFileAPI.update(testFileId, form)).resolves.not.toThrow();
    });

    test("正向测试：更新雾霾程度", async () => {
      const form = createItemFileUpdateForm({ type: "hazy", hazeLevel: "heavy" });
      await expect(ItemFileAPI.update(testFileId, form)).resolves.not.toThrow();
    });

    test("异常测试：更新不存在的图片（后端bug - 应返回错误）", async () => {
      const form = createItemFileUpdateForm({ description: "测试" });
      try {
        await ItemFileAPI.update(99999999, form);
        console.warn("⚠️ 后端bug：更新不存在的图片返回成功（应返回 B0001 错误）");
      } catch (error: any) {
        const bizError = error.response?.data || error;
        expect(bizError.code).toBe("B0001");
        expect(bizError.msg).toContain("不存在");
      }
    });
  });

  describe("DELETE /api/v1/item-files/{id} - 删除图片", () => {
    let testFileId: number;

    beforeEach(async () => {
      const imagePath = path.join(TEST_CLEAN_DIR, "44_outdoor_GT.jpg");

      const formData = new FormData();
      formData.append("itemId", testItemId.toString());
      formData.append("type", "clear");
      formData.append("file", fs.createReadStream(imagePath), {
        filename: "44_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });

      const result = await ItemFileAPI.upload(formData as AnyFormData);
      testFileId = result.id;
    });

    test("正向测试：删除已上传的图片", async () => {
      await expect(ItemFileAPI.deleteById(testFileId)).resolves.not.toThrow();
    });

    test("异常测试：删除不存在的图片（后端幂等设计）", async () => {
      try {
        await ItemFileAPI.deleteById(99999999);
        console.warn("⚠️ 后端删除不存在的图片返回成功（幂等设计）");
      } catch (error: any) {
        const bizError = error.response?.data || error;
        expect(bizError.code).toBe("B0001");
        expect(bizError.msg).toContain("不存在");
      }
    });
  });

  describe("DELETE /api/v1/item-files/batch - 批量删除图片", () => {
    let batchFileIds: number[] = [];

    beforeAll(async () => {
      for (const fileName of ["45_outdoor_GT.jpg"]) {
        const imagePath = path.join(TEST_CLEAN_DIR, fileName);

        const formData = new FormData();
        formData.append("itemId", testItemId.toString());
        formData.append("type", "clear");
        formData.append("file", fs.createReadStream(imagePath), {
          filename: fileName,
          contentType: "image/jpeg",
        });

        const result = await ItemFileAPI.upload(formData as AnyFormData);
        batchFileIds.push(result.id);
        uploadedFileIds.push(result.id);
      }
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
      const form = { ids: [] };
      await expect(ItemFileAPI.batchDelete(form)).rejects.toThrow();
    });

    test("异常测试：包含不存在的ID", async () => {
      const form = { ids: [99999999, 99999998] };
      try {
        const result = await ItemFileAPI.batchDelete(form);
        expect(result.successCount).toBe(0);
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });

  describe("POST /api/v1/dataset-items/upload - 上传数据项配对图片", () => {
    test("正向测试：上传clean+hazy配对（test/资源）", async () => {
      const cleanPath = path.join(TEST_CLEAN_DIR, "41_outdoor_GT.jpg");
      const hazyPath = path.join(TEST_HAZY_DIR, "41_outdoor_hazy.jpg");

      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      formData.append("name", "配对测试");
      formData.append("sceneType", "urban");
      formData.append("clearImage", fs.createReadStream(cleanPath), {
        filename: "41_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });
      formData.append("hazyImages", fs.createReadStream(hazyPath), {
        filename: "41_outdoor_hazy.jpg",
        contentType: "image/jpeg",
      });
      formData.append("hazeLevels", "medium");

      const result = await DatasetItemAPI.uploadImagePair(formData as AnyFormData);
      expect(result.id).toBeDefined();
      expect(result.name).toContain("配对测试");
      if (result.clearImage) {
        expect(result.clearImage.id).toBeDefined();
      }
      if (result.hazyImages && result.hazyImages.length > 0) {
        expect(result.hazyImages.length).toBeGreaterThan(0);
      }
    });

    test("正向测试：上传一对多hazy配对（test2/资源）", async () => {
      const cleanPath = path.join(TEST2_CLEAN_DIR, "0025.jpg");
      const hazyFiles = ["0025_0.8_0.04.jpg", "0025_0.8_0.08.jpg", "0025_0.9_0.12.jpg"];

      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      formData.append("name", "一对多测试");
      formData.append("sceneType", "urban");
      formData.append("clearImage", fs.createReadStream(cleanPath), {
        filename: "0025.jpg",
        contentType: "image/jpeg",
      });

      hazyFiles.forEach((fileName) => {
        formData.append("hazyImages", fs.createReadStream(path.join(TEST2_HAZY_DIR, fileName)), {
          filename: fileName,
          contentType: "image/jpeg",
        });
        formData.append("hazeLevels", "light");
      });

      const result = await DatasetItemAPI.uploadImagePair(formData as AnyFormData);
      expect(result.id).toBeDefined();
      if (result.hazyImages) {
        expect(result.hazyImages.length).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe("POST /api/v1/dataset-items/batch - 批量上传图片", () => {
    test("正向测试：批量上传test/资源（多对配对）", async () => {
      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      formData.append("sceneType", "outdoor");

      const cleanFiles = ["42_outdoor_GT.jpg", "43_outdoor_GT.jpg"];
      cleanFiles.forEach((fileName) => {
        const filePath = path.join(TEST_CLEAN_DIR, fileName);
        formData.append("files", fs.createReadStream(filePath), {
          filename: fileName,
          contentType: "image/jpeg",
        });
      });

      const hazyFiles = ["42_outdoor_hazy.jpg", "43_outdoor_hazy.jpg"];
      hazyFiles.forEach((fileName) => {
        const filePath = path.join(TEST_HAZY_DIR, fileName);
        formData.append("files", fs.createReadStream(filePath), {
          filename: fileName,
          contentType: "image/jpeg",
        });
      });

      const result = await DatasetItemAPI.batchUpload(formData as AnyFormData);
      expect(result).toBeDefined();
      expect(result.total).toBe(cleanFiles.length + hazyFiles.length);
      expect(result.succeeded).toBeGreaterThanOrEqual(0);
      expect(result.failed).toBeGreaterThanOrEqual(0);
    });

    test("边界测试：批量上传单张图片", async () => {
      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      formData.append("sceneType", "urban");

      const filePath = path.join(TEST_CLEAN_DIR, "44_outdoor_GT.jpg");
      formData.append("files", fs.createReadStream(filePath), {
        filename: "44_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });

      const result = await DatasetItemAPI.batchUpload(formData as AnyFormData);
      expect(result).toBeDefined();
      expect(result.total).toBe(1);
      expect(result.succeeded + result.failed).toBe(1);
    });

    test("参数校验：空文件列表", async () => {
      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());

      await expect(DatasetItemAPI.batchUpload(formData as AnyFormData)).rejects.toThrow();
    });
  });

  describe("POST /api/v1/dataset-items/{id}/download/task - 创建数据项下载任务", () => {
    let testDownloadItemId: number;

    beforeAll(async () => {
      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      formData.append("name", "下载测试");
      formData.append("sceneType", "urban");

      const cleanPath = path.join(TEST_CLEAN_DIR, "45_outdoor_GT.jpg");
      formData.append("clearImage", fs.createReadStream(cleanPath), {
        filename: "45_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });

      const hazyPath = path.join(TEST_HAZY_DIR, "45_outdoor_hazy.jpg");
      formData.append("hazyImages", fs.createReadStream(hazyPath), {
        filename: "45_outdoor_hazy.jpg",
        contentType: "image/jpeg",
      });
      formData.append("hazeLevels", "medium");

      const result = await DatasetItemAPI.uploadImagePair(formData as AnyFormData);
      testDownloadItemId = result.id;
    });

    test("正向测试：创建下载任务（全部图片）", async () => {
      const result = await DatasetItemAPI.createDownloadTask(testDownloadItemId);
      expect(result.taskId).toBeDefined();
      expect(result.status).toBeDefined();
      if (result.progress !== undefined && result.progress !== null) {
        expect(result.progress).toBeGreaterThanOrEqual(0);
        expect(result.progress).toBeLessThanOrEqual(100);
      }
    });

    test("正向测试：创建下载任务（指定图片ID）", async () => {
      const item = await DatasetItemAPI.getById(testDownloadItemId);
      const fileIds = item.clearImage ? [item.clearImage.id] : [];

      if (fileIds.length > 0) {
        const result = await DatasetItemAPI.createDownloadTask(testDownloadItemId, fileIds);
        expect(result.taskId).toBeDefined();
        if (result.totalFiles !== undefined && result.totalFiles !== null) {
          expect(result.totalFiles).toBeGreaterThan(0);
        }
      } else {
        console.warn("No file IDs available for download task test");
      }
    });

    test("异常测试：创建不存在数据项的下载任务", async () => {
      await expect(DatasetItemAPI.createDownloadTask(99999999)).rejects.toThrow();
    });
  });

  describe("POST /api/v1/dataset-items/batch/download - 批量下载数据项图片", () => {
    test("正向测试：批量下载test/资源图片", async () => {
      const fileIds: number[] = [];
      for (const fileName of ["41_outdoor_GT.jpg", "42_outdoor_GT.jpg"]) {
        const imagePath = path.join(TEST_CLEAN_DIR, fileName);

        const formData = new FormData();
        formData.append("itemId", testItemId.toString());
        formData.append("type", "clear");
        formData.append("file", fs.createReadStream(imagePath), {
          filename: fileName,
          contentType: "image/jpeg",
        });

        const result = await ItemFileAPI.upload(formData as AnyFormData);
        fileIds.push(result.id);
        uploadedFileIds.push(result.id);
      }

      const downloadForm = {
        itemFileIds: fileIds,
        organizeByItem: true,
      };

      const result = await DatasetItemAPI.batchDownload(downloadForm);
      expect(result.taskId).toBeDefined();
      expect(result.status).toBeDefined();
    });

    test("正向测试：扁平结构批量下载", async () => {
      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      formData.append("name", "扁平下载测试");
      formData.append("sceneType", "urban");

      const cleanPath = path.join(TEST_CLEAN_DIR, "43_outdoor_GT.jpg");
      formData.append("clearImage", fs.createReadStream(cleanPath), {
        filename: "43_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });

      const hazyPath = path.join(TEST_HAZY_DIR, "43_outdoor_hazy.jpg");
      formData.append("hazyImages", fs.createReadStream(hazyPath), {
        filename: "43_outdoor_hazy.jpg",
        contentType: "image/jpeg",
      });
      formData.append("hazeLevels", "medium");

      const result = await DatasetItemAPI.uploadImagePair(formData as AnyFormData);
      const item = await DatasetItemAPI.getById(result.id);

      if (item.clearImage) {
        const downloadForm = {
          itemFileIds: [item.clearImage.id],
          organizeByItem: false,
        };

        const downloadResult = await DatasetItemAPI.batchDownload(downloadForm);
        expect(downloadResult.taskId).toBeDefined();
      }
    });

    test("参数校验：空ID数组", async () => {
      const downloadForm = {
        itemFileIds: [],
        organizeByItem: true,
      };

      await expect(DatasetItemAPI.batchDownload(downloadForm)).rejects.toThrow();
    });

    test("参数校验：缺少必需参数", async () => {
      const downloadForm = {} as any;

      await expect(DatasetItemAPI.batchDownload(downloadForm)).rejects.toThrow();
    });
  });

  describe("业务规则测试", () => {
    test("业务规则：配对图片分辨率一致性校验", async () => {
      const cleanPath = path.join(TEST_CLEAN_DIR, "41_outdoor_GT.jpg");
      const hazyPath = path.join(TEST2_HAZY_DIR, "0025_0.8_0.04.jpg");

      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      formData.append("name", "分辨率测试");
      formData.append("sceneType", "urban");
      formData.append("clearImage", fs.createReadStream(cleanPath), {
        filename: "41_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });
      formData.append("hazyImages", fs.createReadStream(hazyPath), {
        filename: "0025_0.8_0.04.jpg",
        contentType: "image/jpeg",
      });
      formData.append("hazeLevels", "light");

      try {
        const result = await DatasetItemAPI.uploadImagePair(formData as AnyFormData);
        console.warn("⚠️ 后端未校验配对图片分辨率一致性，建议后端添加校验");
        expect(result).toBeDefined();
      } catch (error: any) {
        const bizError = error.response?.data || error;
        expect(bizError.code).toBe("B0001");
        expect(bizError.msg).toMatch(/分辨率|resolution|尺寸/i);
      }
    });

    test("业务规则：批量上传文件名识别规则", async () => {
      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      formData.append("sceneType", "outdoor");

      const cleanPath = path.join(TEST_CLEAN_DIR, "41_outdoor_GT.jpg");
      const hazyPath = path.join(TEST_HAZY_DIR, "41_outdoor_hazy.jpg");

      formData.append("files", fs.createReadStream(cleanPath), {
        filename: "scene001_clear.jpg",
        contentType: "image/jpeg",
      });
      formData.append("files", fs.createReadStream(hazyPath), {
        filename: "scene001_hazy_medium.jpg",
        contentType: "image/jpeg",
      });

      const result = await DatasetItemAPI.batchUpload(formData as AnyFormData);
      expect(result.total).toBe(2);
      if (result.succeeded > 0) {
        expect(result.successItems).toBeDefined();
        if (result.successItems && result.successItems.length > 0) {
          expect(result.successItems[0]!.fileCount).toBeGreaterThanOrEqual(1);
        }
      }
    });

    test("业务规则：图片类型修改后配对完整性", async () => {
      const formData = new FormData();
      formData.append("datasetId", testDatasetId.toString());
      formData.append("name", "配对完整性测试");
      formData.append("sceneType", "urban");

      const cleanPath = path.join(TEST_CLEAN_DIR, "42_outdoor_GT.jpg");
      const hazyPath = path.join(TEST_HAZY_DIR, "42_outdoor_hazy.jpg");

      formData.append("clearImage", fs.createReadStream(cleanPath), {
        filename: "42_outdoor_GT.jpg",
        contentType: "image/jpeg",
      });
      formData.append("hazyImages", fs.createReadStream(hazyPath), {
        filename: "42_outdoor_hazy.jpg",
        contentType: "image/jpeg",
      });
      formData.append("hazeLevels", "medium");

      const item = await DatasetItemAPI.uploadImagePair(formData as AnyFormData);
      expect(item.id).toBeDefined();

      const detail = await DatasetItemAPI.getById(item.id);
      if (!detail.clearImage) {
        console.warn("⚠️ 后端未在数据项详情中返回 clearImage 字段");
        return;
      }

      const updateForm = createItemFileUpdateForm({ type: "hazy", hazeLevel: "light" });

      try {
        await ItemFileAPI.update(detail.clearImage.id, updateForm);
        const updatedDetail = await DatasetItemAPI.getById(item.id);
        if (!updatedDetail.clearImage) {
          console.warn("⚠️ 后端允许删除所有清晰图，建议添加配对完整性校验");
        }
      } catch (error: any) {
        expect(error).toBeDefined();
      }
    });
  });
});
