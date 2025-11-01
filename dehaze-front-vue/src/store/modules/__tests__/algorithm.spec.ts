import { useAlgorithmStore } from "../algorithm";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AlgorithmAPI } from "dehaze-sdk-js";
import type { Algorithm, AlgorithmQuery } from "dehaze-sdk-js";

// Mock AlgorithmAPI
vi.mock("dehaze-sdk-js", () => ({
  AlgorithmAPI: {
    getList: vi.fn(),
    getOption: vi.fn(),
    add: vi.fn(),
    update: vi.fn(),
    deleteByIds: vi.fn(),
  },
}));

describe("useAlgorithmStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("初始化状态", () => {
    it("应该初始化空的算法列表", () => {
      const store = useAlgorithmStore();

      expect(store.algorithmList).toEqual([]);
      expect(Array.isArray(store.algorithmList)).toBe(true);
    });

    it("应该初始化空的算法选项列表", () => {
      const store = useAlgorithmStore();

      expect(store.algorithmOptions).toEqual([]);
      expect(Array.isArray(store.algorithmOptions)).toBe(true);
    });
  });

  describe("getAlgorithmList", () => {
    it("应该成功获取算法列表", async () => {
      // Arrange
      const mockAlgorithms: Algorithm[] = [
        {
          id: 1,
          parentId: 0,
          type: "DCP",
          name: "DCP",
          description: "Dark Channel Prior",
        },
        {
          id: 2,
          parentId: 0,
          type: "DCP",
          name: "DCP",
          description: "Dark Channel Prior",
        },
      ];

      vi.mocked(AlgorithmAPI.getList).mockResolvedValue(mockAlgorithms);

      const store = useAlgorithmStore();

      // Act
      await store.getAlgorithmList();

      // Assert
      expect(AlgorithmAPI.getList).toHaveBeenCalledWith(undefined);
      expect(store.algorithmList).toEqual(mockAlgorithms);
      expect(store.algorithmList.length).toBe(2);
    });

    it("应该使用查询参数获取算法列表", async () => {
      // Arrange
      const queryParams: AlgorithmQuery = {
        keywords: "DCP",
      };

      const mockAlgorithms: Algorithm[] = [
        {
          id: 1,
          parentId: 0,
          type: "DCP",
          name: "DCP",
          description: "Dark Channel Prior",
        },
      ];

      vi.mocked(AlgorithmAPI.getList).mockResolvedValue(mockAlgorithms);

      const store = useAlgorithmStore();

      // Act
      await store.getAlgorithmList(queryParams);

      // Assert
      expect(AlgorithmAPI.getList).toHaveBeenCalledWith(queryParams);
      expect(store.algorithmList).toEqual(mockAlgorithms);
    });

    it("应该处理空的算法列表", async () => {
      // Arrange
      vi.mocked(AlgorithmAPI.getList).mockResolvedValue([]);

      const store = useAlgorithmStore();

      // Act
      await store.getAlgorithmList();

      // Assert
      expect(store.algorithmList).toEqual([]);
      expect(store.algorithmList.length).toBe(0);
    });

    it("应该处理API调用失败", async () => {
      // Arrange
      const error = new Error("Network error");
      vi.mocked(AlgorithmAPI.getList).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(store.getAlgorithmList()).rejects.toThrow("Network error");
      expect(store.algorithmList).toEqual([]); // 应该保持为空
    });

    it("应该处理部分参数的查询", async () => {
      // Arrange
      const queryParams: AlgorithmQuery = {
        keywords: "AOD",
      };

      const mockAlgorithms: Algorithm[] = [
        {
          id: 1,
          parentId: 0,
          type: "DCP",
          name: "DCP",
          description: "Dark Channel Prior",
        },
      ];

      vi.mocked(AlgorithmAPI.getList).mockResolvedValue(mockAlgorithms);

      const store = useAlgorithmStore();

      // Act
      await store.getAlgorithmList(queryParams);

      // Assert
      expect(AlgorithmAPI.getList).toHaveBeenCalledWith(queryParams);
      expect(store.algorithmList).toEqual(mockAlgorithms);
    });
  });

  describe("getAlgorithmOptions", () => {
    it("应该成功获取算法选项列表", async () => {
      // Arrange
      const mockOptions = [
        { value: "1", label: "DCP" },
        { value: "2", label: "AOD-Net" },
        { value: "3", label: "FFA-Net" },
      ];

      vi.mocked(AlgorithmAPI.getOption).mockResolvedValue(mockOptions);

      const store = useAlgorithmStore();

      // Act
      await store.getAlgorithmOptions();

      // Assert
      expect(AlgorithmAPI.getOption).toHaveBeenCalled();
      expect(store.algorithmOptions).toEqual(mockOptions);
      expect(store.algorithmOptions.length).toBe(3);
    });

    it("应该处理空的算法选项列表", async () => {
      // Arrange
      vi.mocked(AlgorithmAPI.getOption).mockResolvedValue([]);

      const store = useAlgorithmStore();

      // Act
      await store.getAlgorithmOptions();

      // Assert
      expect(store.algorithmOptions).toEqual([]);
      expect(store.algorithmOptions.length).toBe(0);
    });

    it("应该处理API调用失败", async () => {
      // Arrange
      const error = new Error("Failed to fetch options");
      vi.mocked(AlgorithmAPI.getOption).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(store.getAlgorithmOptions()).rejects.toThrow(
        "Failed to fetch options"
      );
      expect(store.algorithmOptions).toEqual([]); // 应该保持为空
    });
  });

  describe("addAlgorithm", () => {
    it("应该成功添加新算法", async () => {
      // Arrange
      const newAlgorithm: Algorithm = {
        id: 1,
        parentId: 0,
        type: "DCP",
        name: "DCP",
        description: "Dark Channel Prior",
      };

      const mockResult = { success: true, id: 123 };
      vi.mocked(AlgorithmAPI.add).mockResolvedValue(mockResult);

      const store = useAlgorithmStore();

      // Act
      const result = await store.addAlgorithm(newAlgorithm);

      // Assert
      expect(AlgorithmAPI.add).toHaveBeenCalledWith(newAlgorithm);
      expect(result).toEqual(mockResult);
    });

    it("应该处理添加算法失败", async () => {
      // Arrange
      const newAlgorithm: Algorithm = {
        name: "Invalid Algorithm",
        description: "Invalid data",
      } as Algorithm;

      const error = new Error("Validation failed");
      vi.mocked(AlgorithmAPI.add).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(store.addAlgorithm(newAlgorithm)).rejects.toThrow(
        "Validation failed"
      );
    });

    it("应该处理空的算法数据", async () => {
      // Arrange
      const emptyAlgorithm = {} as Algorithm;
      const error = new Error("Empty algorithm data");
      vi.mocked(AlgorithmAPI.add).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(store.addAlgorithm(emptyAlgorithm)).rejects.toThrow(
        "Empty algorithm data"
      );
    });
  });

  describe("updateAlgorithm", () => {
    it("应该成功更新算法", async () => {
      // Arrange
      const algorithmId = 1;
      const updatedAlgorithm: Algorithm = {
        id: algorithmId,
        parentId: 0,
        name: "Updated Algorithm",
        type: "Updated Type",
        description: "Updated description",
      };

      const mockResult = { success: true };
      vi.mocked(AlgorithmAPI.update).mockResolvedValue(mockResult);

      const store = useAlgorithmStore();

      // Act
      const result = await store.updateAlgorithm(algorithmId, updatedAlgorithm);

      // Assert
      expect(AlgorithmAPI.update).toHaveBeenCalledWith(
        algorithmId,
        updatedAlgorithm
      );
      expect(result).toEqual(mockResult);
    });

    it("应该处理更新不存在的算法", async () => {
      // Arrange
      const nonExistentId = 999;
      const algorithmData: Algorithm = {
        name: "Updated Algorithm",
      } as Algorithm;

      const error = new Error("Algorithm not found");
      vi.mocked(AlgorithmAPI.update).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(
        store.updateAlgorithm(nonExistentId, algorithmData)
      ).rejects.toThrow("Algorithm not found");
    });

    it("应该处理无效的ID", async () => {
      // Arrange
      const invalidId = -1;
      const algorithmData: Algorithm = {
        name: "Updated Algorithm",
      } as Algorithm;

      const error = new Error("Invalid ID");
      vi.mocked(AlgorithmAPI.update).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(
        store.updateAlgorithm(invalidId, algorithmData)
      ).rejects.toThrow("Invalid ID");
    });
  });

  describe("deleteAlgorithmByIds", () => {
    it("应该成功删除单个算法", async () => {
      // Arrange
      const ids = ["1"];
      const mockResult = { success: true, deletedCount: 1 };
      vi.mocked(AlgorithmAPI.deleteByIds).mockResolvedValue(mockResult);

      const store = useAlgorithmStore();

      // Act
      const result = await store.deleteAlgorithmByIds(ids);

      // Assert
      expect(AlgorithmAPI.deleteByIds).toHaveBeenCalledWith(ids);
      expect(result).toEqual(mockResult);
    });

    it("应该成功删除多个算法", async () => {
      // Arrange
      const ids = ["1", "2", "3"];
      const mockResult = { success: true, deletedCount: 3 };
      vi.mocked(AlgorithmAPI.deleteByIds).mockResolvedValue(mockResult);

      const store = useAlgorithmStore();

      // Act
      const result = await store.deleteAlgorithmByIds(ids);

      // Assert
      expect(AlgorithmAPI.deleteByIds).toHaveBeenCalledWith(ids);
      expect(result).toEqual(mockResult);
    });

    it("应该处理删除不存在的算法", async () => {
      // Arrange
      const nonExistentIds = ["999", "1000"];
      const mockResult = { success: true, deletedCount: 0 };
      vi.mocked(AlgorithmAPI.deleteByIds).mockResolvedValue(mockResult);

      const store = useAlgorithmStore();

      // Act
      const result = await store.deleteAlgorithmByIds(nonExistentIds);

      // Assert
      expect(AlgorithmAPI.deleteByIds).toHaveBeenCalledWith(nonExistentIds);
      expect(result.deletedCount).toBe(0);
    });

    it("应该处理空的ID数组", async () => {
      // Arrange
      const emptyIds: string[] = [];
      const error = new Error("No IDs provided");
      vi.mocked(AlgorithmAPI.deleteByIds).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(store.deleteAlgorithmByIds(emptyIds)).rejects.toThrow(
        "No IDs provided"
      );
    });

    it("应该处理删除操作失败", async () => {
      // Arrange
      const ids = ["1"];
      const error = new Error("Delete operation failed");
      vi.mocked(AlgorithmAPI.deleteByIds).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(store.deleteAlgorithmByIds(ids)).rejects.toThrow(
        "Delete operation failed"
      );
    });
  });

  describe("状态隔离", () => {
    it("多个store实例应该有独立的状态", async () => {
      // Arrange
      const store1 = useAlgorithmStore();
      const store2 = useAlgorithmStore();

      const mockAlgorithms: Algorithm[] = [
        { id: 1, name: "Algorithm 1" } as Algorithm,
      ];

      vi.mocked(AlgorithmAPI.getList).mockResolvedValue(mockAlgorithms);

      // Act
      await store1.getAlgorithmList();

      // Assert
      expect(store1.algorithmList).toEqual(mockAlgorithms);
      expect(store2.algorithmList).toEqual([]); // store2应该不受影响
    });
  });

  describe("边界情况", () => {
    it("应该处理null查询参数", async () => {
      // Arrange
      vi.mocked(AlgorithmAPI.getList).mockResolvedValue([]);

      const store = useAlgorithmStore();

      // Act
      await store.getAlgorithmList(null as any);

      // Assert
      expect(AlgorithmAPI.getList).toHaveBeenCalledWith(null);
      expect(store.algorithmList).toEqual([]);
    });

    it("应该处理undefined查询参数", async () => {
      // Arrange
      vi.mocked(AlgorithmAPI.getList).mockResolvedValue([]);

      const store = useAlgorithmStore();

      // Act
      await store.getAlgorithmList(undefined);

      // Assert
      expect(AlgorithmAPI.getList).toHaveBeenCalledWith(undefined);
      expect(store.algorithmList).toEqual([]);
    });

    it("应该处理null算法数据", async () => {
      // Arrange
      const error = new Error("Null algorithm data");
      vi.mocked(AlgorithmAPI.add).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(store.addAlgorithm(null as any)).rejects.toThrow(
        "Null algorithm data"
      );
    });

    it("应该处理undefined算法数据", async () => {
      // Arrange
      const error = new Error("Undefined algorithm data");
      vi.mocked(AlgorithmAPI.add).mockRejectedValue(error);

      const store = useAlgorithmStore();

      // Act & Assert
      await expect(store.addAlgorithm(undefined as any)).rejects.toThrow(
        "Undefined algorithm data"
      );
    });
  });

  describe("完整工作流", () => {
    it("应该支持完整的CRUD操作流程", async () => {
      // Arrange
      const store = useAlgorithmStore();

      // Mock API responses
      const mockAlgorithms: Algorithm[] = [
        { id: 1, name: "DCP" } as Algorithm,
        { id: 2, name: "AOD-Net" } as Algorithm,
      ];
      const mockOptions = [
        { value: "1", label: "DCP" },
        { value: "2", label: "AOD-Net" },
      ];

      vi.mocked(AlgorithmAPI.getList).mockResolvedValue(mockAlgorithms);
      vi.mocked(AlgorithmAPI.getOption).mockResolvedValue(mockOptions);
      vi.mocked(AlgorithmAPI.add).mockResolvedValue({ success: true, id: 3 });
      vi.mocked(AlgorithmAPI.update).mockResolvedValue({ success: true });
      vi.mocked(AlgorithmAPI.deleteByIds).mockResolvedValue({
        success: true,
        deletedCount: 1,
      });

      // Act & Assert - 获取列表
      await store.getAlgorithmList();
      expect(store.algorithmList).toEqual(mockAlgorithms);

      // Act & Assert - 获取选项
      await store.getAlgorithmOptions();
      expect(store.algorithmOptions).toEqual(mockOptions);

      // Act & Assert - 添加算法
      const newAlgorithm = { name: "New Algorithm" } as Algorithm;
      await store.addAlgorithm(newAlgorithm);
      expect(AlgorithmAPI.add).toHaveBeenCalledWith(newAlgorithm);

      // Act & Assert - 更新算法
      const updatedAlgorithm = { name: "Updated Algorithm" } as Algorithm;
      await store.updateAlgorithm(1, updatedAlgorithm);
      expect(AlgorithmAPI.update).toHaveBeenCalledWith(1, updatedAlgorithm);

      // Act & Assert - 删除算法
      await store.deleteAlgorithmByIds(["2"]);
      expect(AlgorithmAPI.deleteByIds).toHaveBeenCalledWith(["2"]);
    });
  });
});
