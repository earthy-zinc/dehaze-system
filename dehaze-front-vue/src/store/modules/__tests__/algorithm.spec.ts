import { useAlgorithmStore } from "../algorithm";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AlgorithmAPI } from "dehaze-sdk-js";

// Mock AlgorithmAPI
vi.mock("dehaze-sdk-js", () => ({
  AlgorithmAPI: {
    getOption: vi.fn(),
  },
}));

describe("useAlgorithmStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("初始化状态", () => {
    it("应该初始化空的算法选项列表", () => {
      const store = useAlgorithmStore();

      expect(store.algorithmOptions).toEqual([]);
      expect(Array.isArray(store.algorithmOptions)).toBe(true);
    });
  });

  describe("getAlgorithmOptions", () => {
    it("应该成功获取算法选项列表", async () => {
      const mockOptions = [
        { value: "1", label: "DCP" },
        { value: "2", label: "AOD-Net" },
        { value: "3", label: "FFA-Net" },
      ];

      vi.mocked(AlgorithmAPI.getOption).mockResolvedValue(mockOptions);

      const store = useAlgorithmStore();

      await store.getAlgorithmOptions();

      expect(AlgorithmAPI.getOption).toHaveBeenCalled();
      expect(store.algorithmOptions).toEqual(mockOptions);
      expect(store.algorithmOptions.length).toBe(3);
    });

    it("应该处理空的算法选项列表", async () => {
      vi.mocked(AlgorithmAPI.getOption).mockResolvedValue([]);

      const store = useAlgorithmStore();

      await store.getAlgorithmOptions();

      expect(store.algorithmOptions).toEqual([]);
      expect(store.algorithmOptions.length).toBe(0);
    });

    it("应该处理API调用失败", async () => {
      const error = new Error("Failed to fetch options");
      vi.mocked(AlgorithmAPI.getOption).mockRejectedValue(error);

      const store = useAlgorithmStore();

      await expect(store.getAlgorithmOptions()).rejects.toThrow(
        "Failed to fetch options"
      );
      expect(store.algorithmOptions).toEqual([]);
    });
  });
});
