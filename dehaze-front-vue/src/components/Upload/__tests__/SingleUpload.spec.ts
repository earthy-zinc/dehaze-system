import { mount } from "@vue/test-utils";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ElMessage, ElMessageBox } from "element-plus";
import SingleUpload from "../SingleUpload.vue";
import { FileAPI } from "dehaze-sdk-js";

// Mock SDK
vi.mock("dehaze-sdk-js", () => ({
  FileAPI: {
    upload: vi.fn(),
  },
}));

// Mock Element Plus
vi.mock("element-plus", async () => {
  const actual = await vi.importActual("element-plus");
  return {
    ...actual,
    ElMessage: {
      warning: vi.fn(),
      error: vi.fn(),
      success: vi.fn(),
    },
    ElMessageBox: {
      confirm: vi.fn(),
    },
  };
});

// Mock useImageShowStore
vi.mock("@/store/modules/imageShow", () => ({
  useImageShowStore: () => ({
    modelId: "test-model-id",
  }),
}));

describe("SingleUpload Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("组件渲染", () => {
    it("应该正确渲染上传组件", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      expect(wrapper.find(".single-uploader").exists()).toBe(true);
      expect(wrapper.find("img.single-uploader__image").exists()).toBe(false);
      expect(wrapper.text()).toContain("上传图片");
    });

    it("应该显示自定义的提示文本", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
          tooltip: "请选择图片文件",
        },
      });

      expect(wrapper.text()).toContain("请选择图片文件");
    });

    it("当有图片URL时应该显示图片", () => {
      const imageUrl = "https://example.com/test.jpg";
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: imageUrl,
        },
      });

      const img = wrapper.find("img.single-uploader__image");
      expect(img.exists()).toBe(true);
      expect(img.attributes("src")).toBe(imageUrl);
      expect(img.attributes("alt")).toBe("图片解析失败");
    });
  });

  describe("文件上传验证", () => {
    it("应该接受大小小于10MB的文件", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      const file = new File(["test"], "test.jpg", { type: "image/jpeg" });
      Object.defineProperty(file, "size", { value: 5 * 1024 * 1024 }); // 5MB

      const result = (wrapper.vm as any).handleBeforeUpload(file);
      expect(result).toBe(true);
    });

    it("应该拒绝大小大于10MB的文件并显示警告", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      const file = new File(["test"], "large.jpg", { type: "image/jpeg" });
      Object.defineProperty(file, "size", { value: 15 * 1024 * 1024 }); // 15MB

      const result = (wrapper.vm as any).handleBeforeUpload(file);
      expect(result).toBe(false);
      expect(ElMessage.warning).toHaveBeenCalledWith("上传图片不能大于10M");
    });

    it("应该处理边界值文件大小（10MB）", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      const file = new File(["test"], "boundary.jpg", { type: "image/jpeg" });
      Object.defineProperty(file, "size", { value: 10 * 1024 * 1024 }); // 正好10MB

      const result = (wrapper.vm as any).handleBeforeUpload(file);
      expect(result).toBe(true);
    });
  });

  describe("文件上传功能", () => {
    it("应该成功上传文件并更新URL", async () => {
      const mockUploadResponse = {
        fileId: 0,
        name: "",
        url: "https://example.com/uploaded.jpg",
      };
      vi.mocked(FileAPI.upload).mockResolvedValue(mockUploadResponse);

      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      const file = new File(["test"], "test.jpg", { type: "image/jpeg" });
      const uploadOptions = {
        file,
        onProgress: vi.fn(),
        onSuccess: vi.fn(),
        onError: vi.fn(),
      };

      await (wrapper.vm as any).uploadFile(uploadOptions);

      expect(FileAPI.upload).toHaveBeenCalledWith(file, "test-model-id");
      expect(wrapper.emitted("update:modelValue")).toBeTruthy();
      expect(wrapper.emitted("update:modelValue")?.[0]).toEqual([
        "https://example.com/uploaded.jpg",
      ]);
      expect(wrapper.emitted("onChange")).toBeTruthy();
      expect(wrapper.emitted("onChange")?.[0]).toEqual([
        "https://example.com/uploaded.jpg",
      ]);
    });

    it("应该在上传失败时保持原有URL", async () => {
      const initialUrl = "https://example.com/initial.jpg";
      vi.mocked(FileAPI.upload).mockRejectedValue(new Error("Upload failed"));

      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: initialUrl,
        },
      });

      const file = new File(["test"], "test.jpg", { type: "image/jpeg" });
      const uploadOptions = {
        file,
        onProgress: vi.fn(),
        onSuccess: vi.fn(),
        onError: vi.fn(),
      };

      await expect(
        (wrapper.vm as any).uploadFile(uploadOptions)
      ).rejects.toThrow("Upload failed");

      // 验证原有URL没有改变
      expect(wrapper.props("modelValue")).toBe(initialUrl);
    });

    it("应该处理网络错误", async () => {
      vi.mocked(FileAPI.upload).mockRejectedValue(new Error("Network error"));

      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      const file = new File(["test"], "test.jpg", { type: "image/jpeg" });
      const uploadOptions = {
        file,
        onProgress: vi.fn(),
        onSuccess: vi.fn(),
        onError: vi.fn(),
      };

      await expect(
        (wrapper.vm as any).uploadFile(uploadOptions)
      ).rejects.toThrow("Network error");
    });
  });

  describe("双向绑定", () => {
    it("应该正确响应modelValue的变化", async () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      expect(wrapper.find("img.single-uploader__image").exists()).toBe(false);

      await wrapper.setProps({
        modelValue: "https://example.com/new-image.jpg",
      });

      expect(wrapper.find("img.single-uploader__image").exists()).toBe(true);
      expect(wrapper.find("img.single-uploader__image").attributes("src")).toBe(
        "https://example.com/new-image.jpg"
      );
    });

    it("应该正确处理从有值到空的变化", async () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "https://example.com/initial.jpg",
        },
      });

      expect(wrapper.find("img.single-uploader__image").exists()).toBe(true);

      await wrapper.setProps({ modelValue: "" });

      expect(wrapper.find("img.single-uploader__image").exists()).toBe(false);
    });
  });

  describe("组件样式和类名", () => {
    it("应该包含正确的CSS类", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      expect(wrapper.find(".single-uploader").exists()).toBe(true);
    });

    it("图片应该包含正确的CSS类", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "https://example.com/test.jpg",
        },
      });

      const img = wrapper.find("img");
      expect(img.classes()).toContain("single-uploader__image");
    });
  });

  describe("Props验证", () => {
    it("应该设置默认的props值", () => {
      const wrapper = mount(SingleUpload);

      expect(wrapper.props("modelValue")).toBe("");
      expect(wrapper.props("tooltip")).toBe("上传图片");
    });

    it("应该接受自定义的props值", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "https://example.com/custom.jpg",
          tooltip: "自定义提示",
        },
      });

      expect(wrapper.props("modelValue")).toBe(
        "https://example.com/custom.jpg"
      );
      expect(wrapper.props("tooltip")).toBe("自定义提示");
    });
  });

  describe("事件处理", () => {
    it("应该在成功上传后触发正确的事件", async () => {
      const mockUploadResponse = {
        fileId: 0,
        name: "",
        url: "https://example.com/uploaded.jpg",
      };
      vi.mocked(FileAPI.upload).mockResolvedValue(mockUploadResponse);

      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      const file = new File(["test"], "test.jpg", { type: "image/jpeg" });
      const uploadOptions = {
        file,
        onProgress: vi.fn(),
        onSuccess: vi.fn(),
        onError: vi.fn(),
      };

      await (wrapper.vm as any).uploadFile(uploadOptions);

      // 验证事件被触发
      expect(wrapper.emitted("update:modelValue")).toBeDefined();
      expect(wrapper.emitted("onChange")).toBeDefined();
    });
  });

  describe("边界情况", () => {
    it("应该处理空文件对象", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      // 创建一个空的文件对象
      const file = new File([""], "empty.jpg", { type: "image/jpeg" });
      Object.defineProperty(file, "size", { value: 0 });

      const result = (wrapper.vm as any).handleBeforeUpload(file);
      expect(result).toBe(true); // 0字节的文件应该通过大小检查
    });

    it("应该处理不支持的文件类型（通过大小检查）", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      // 文件类型不由beforeUpload检查，只检查大小
      const file = new File(["test"], "test.txt", { type: "text/plain" });
      Object.defineProperty(file, "size", { value: 1024 });

      const result = (wrapper.vm as any).handleBeforeUpload(file);
      expect(result).toBe(true);
    });

    it("应该处理非常大的文件", () => {
      const wrapper = mount(SingleUpload, {
        props: {
          modelValue: "",
        },
      });

      const file = new File(["test"], "huge.jpg", { type: "image/jpeg" });
      Object.defineProperty(file, "size", { value: Number.MAX_SAFE_INTEGER });

      const result = (wrapper.vm as any).handleBeforeUpload(file);
      expect(result).toBe(false);
      expect(ElMessage.warning).toHaveBeenCalledWith("上传图片不能大于10M");
    });
  });
});
