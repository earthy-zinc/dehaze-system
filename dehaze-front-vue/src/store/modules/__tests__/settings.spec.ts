import { useSettingsStore } from "../settings";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ThemeEnum } from "@/enums/ThemeEnum";

// Mock defaultSettings
vi.mock("@/settings", () => ({
  default: {
    tagsView: true,
    sidebarLogo: true,
    fixedHeader: true,
    layout: "left",
    themeColor: "#409EFF",
    theme: "light",
    watermarkEnabled: false,
  },
}));

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
Object.defineProperty(window, "localStorage", {
  value: localStorageMock,
});

describe("useSettingsStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("初始化状态", () => {
    it("应该初始化默认的设置状态", () => {
      const store = useSettingsStore();

      expect(store.settingsVisible).toBe(false);
      expect(store.tagsView).toBe(true);
      expect(store.sidebarLogo).toBe(true);
      expect(store.fixedHeader).toBe(true);
      expect(store.layout).toBe("left");
      expect(store.themeColor).toBe("#409EFF");
      expect(store.theme).toBe(ThemeEnum.LIGHT);
      expect(store.watermarkEnabled).toBe(false);
    });
  });

  describe("settingsVisible", () => {
    it("应该有settingsVisible状态", () => {
      const store = useSettingsStore();

      expect(store.settingsVisible).toBeDefined();
      expect(store.settingsVisible).toBe(false); // ref的值
    });

    it("应该能修改settingsVisible", () => {
      const store = useSettingsStore();

      store.settingsVisible = true;

      expect(store.settingsVisible).toBe(true);
    });
  });

  describe("changeSetting", () => {
    it("应该能改变fixedHeader设置", () => {
      const store = useSettingsStore();

      store.changeSetting({ key: "fixedHeader", value: false });

      expect(store.fixedHeader).toBe(false);
    });

    it("应该能改变tagsView设置", () => {
      const store = useSettingsStore();

      store.changeSetting({ key: "tagsView", value: false });

      expect(store.tagsView).toBe(false);
    });

    it("应该能改变sidebarLogo设置", () => {
      const store = useSettingsStore();

      store.changeSetting({ key: "sidebarLogo", value: false });

      expect(store.sidebarLogo).toBe(false);
    });

    it("应该能改变layout设置", () => {
      const store = useSettingsStore();

      store.changeSetting({ key: "layout", value: "top" });

      expect(store.layout).toBe("top");
    });

    it("应该能改变watermarkEnabled设置", () => {
      const store = useSettingsStore();

      store.changeSetting({ key: "watermarkEnabled", value: true });

      expect(store.watermarkEnabled).toBe(true);
    });

    it("应该忽略无效的设置键", () => {
      const store = useSettingsStore();
      const originalThemeColor = store.themeColor;

      store.changeSetting({ key: "invalidKey", value: "invalidValue" });

      expect(store.themeColor).toBe(originalThemeColor);
    });

    it("应该处理所有支持的设置类型", () => {
      const store = useSettingsStore();

      // 测试布尔值
      store.changeSetting({ key: "tagsView", value: false });
      expect(store.tagsView).toBe(false);

      store.changeSetting({ key: "tagsView", value: true });
      expect(store.tagsView).toBe(true);

      // 测试字符串
      store.changeSetting({ key: "layout", value: "mix" });
      expect(store.layout).toBe("mix");
    });
  });

  describe("changeTheme", () => {
    it("应该能改变主题", () => {
      const store = useSettingsStore();

      store.changeTheme(ThemeEnum.DARK);

      expect(store.theme).toBe(ThemeEnum.DARK);
    });

    it("应该支持所有主题类型", () => {
      const store = useSettingsStore();

      store.changeTheme(ThemeEnum.LIGHT);
      expect(store.theme).toBe(ThemeEnum.LIGHT);

      store.changeTheme(ThemeEnum.DARK);
      expect(store.theme).toBe(ThemeEnum.DARK);
    });
  });

  describe("changeThemeColor", () => {
    it("应该能改变主题颜色", () => {
      const store = useSettingsStore();

      store.changeThemeColor("#FF6B6B");

      expect(store.themeColor).toBe("#FF6B6B");
    });
  });

  describe("changeLayout", () => {
    it("应该能改变布局模式", () => {
      const store = useSettingsStore();

      store.changeLayout("top");

      expect(store.layout).toBe("top");
    });

    it("应该支持所有布局模式", () => {
      const store = useSettingsStore();
      const layouts = ["left", "top", "mix"];

      layouts.forEach((layout) => {
        store.changeLayout(layout);
        expect(store.layout).toBe(layout);
      });
    });
  });

  describe("边界情况", () => {
    it("应该处理空的颜色值", () => {
      const store = useSettingsStore();

      store.changeThemeColor("");

      expect(store.themeColor).toBe("");
    });

    it("应该处理无效的颜色值", () => {
      const store = useSettingsStore();

      store.changeThemeColor("invalid-color");

      expect(store.themeColor).toBe("invalid-color");
    });

    it("应该处理无效的布局值", () => {
      const store = useSettingsStore();

      store.changeLayout("invalid-layout");

      expect(store.layout).toBe("invalid-layout");
    });

    it("应该处理空的主题值", () => {
      const store = useSettingsStore();

      store.changeTheme("");

      expect(store.theme).toBe("");
    });

    it("应该处理changeSetting的空参数", () => {
      const store = useSettingsStore();
      const originalTagsView = store.tagsView;

      // @ts-ignore - 测试运行时行为
      store.changeSetting({});

      expect(store.tagsView).toBe(originalTagsView);
    });

    it("应该处理changeSetting的undefined值", () => {
      const store = useSettingsStore();

      store.changeSetting({ key: "tagsView", value: "1" });

      expect(store.tagsView).toBeUndefined();
    });
  });

  describe("状态组合", () => {
    it("应该支持同时修改多个设置", () => {
      const store = useSettingsStore();

      store.changeSetting({ key: "tagsView", value: false });
      store.changeSetting({ key: "fixedHeader", value: false });
      store.changeLayout("top");
      store.changeTheme(ThemeEnum.DARK);
      store.changeThemeColor("#FF6B6B");

      expect(store.tagsView).toBe(false);
      expect(store.fixedHeader).toBe(false);
      expect(store.layout).toBe("top");
      expect(store.theme).toBe(ThemeEnum.DARK);
      expect(store.themeColor).toBe("#FF6B6B");
    });

    it("应该支持主题和颜色的组合变化", () => {
      const store = useSettingsStore();

      // 同时改变主题和颜色
      store.changeTheme(ThemeEnum.DARK);
      store.changeThemeColor("#95D475");

      expect(store.theme).toBe(ThemeEnum.DARK);
      expect(store.themeColor).toBe("#95D475");
    });
  });
});
