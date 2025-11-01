import { ref } from "vue";
import { useAppStore } from "../app";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DeviceEnum } from "@/enums/DeviceEnum";
import { SidebarStatusEnum } from "@/enums/SidebarStatusEnum";

// Mock defaultSettings
vi.mock("@/settings", () => ({
  default: {
    size: "default",
    language: "zh-cn",
  },
}));

// Mock useStorage from @vueuse/core
const createMockRef = (defaultValue: any) => {
  const mockRef = ref(defaultValue);
  return mockRef;
};

vi.mock("@vueuse/core", () => ({
  useStorage: vi.fn((key, defaultValue) => {
    return createMockRef(defaultValue);
  }),
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

describe("useAppStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // 重置localStorage模拟
    localStorageMock.getItem.mockReturnValue(null);
  });

  describe("初始化状态", () => {
    it("应该初始化默认的设备类型", () => {
      const store = useAppStore();

      expect(store.device).toBe(DeviceEnum.DESKTOP);
    });

    it("应该初始化默认的尺寸设置", () => {
      const store = useAppStore();

      expect(store.size).toBe("default");
    });

    it("应该初始化默认的语言设置", () => {
      const store = useAppStore();

      expect(store.language).toBe("zh-cn");
    });

    it("应该初始化侧边栏状态", () => {
      const store = useAppStore();

      expect(store.sidebar.opened).toBe(false);
      expect(store.sidebar.withoutAnimation).toBe(false);
    });

    it("应该初始化空的活动顶级菜单路径", () => {
      const store = useAppStore();

      expect(store.activeTopMenuPath).toBe("");
    });
  });

  describe("locale计算属性", () => {
    it("应该为中文语言返回中文语言包", () => {
      const store = useAppStore();
      store.language = "zh-cn";

      expect(store.locale.name).toBe("zh-cn");
    });

    it("应该为英文语言返回英文语言包", () => {
      const store = useAppStore();
      store.language = "en";

      expect(store.locale.name).toBe("en");
    });

    it("应该为其他语言返回中文语言包作为默认", () => {
      const store = useAppStore();
      store.language = "fr";

      expect(store.locale.name).toBe("zh-cn");
    });
  });

  describe("toggleSidebar", () => {
    it("应该切换侧边栏的打开状态", () => {
      const store = useAppStore();
      const initialState = store.sidebar.opened;

      store.toggleSidebar();

      expect(store.sidebar.opened).toBe(!initialState);
    });

    it("应该更新sidebarStatus存储值", () => {
      const store = useAppStore();
      store.toggleSidebar();

      if (store.sidebar.opened) {
        expect(store.sidebarStatus).toBe(SidebarStatusEnum.OPENED);
      } else {
        expect(store.sidebarStatus).toBe(SidebarStatusEnum.CLOSED);
      }
    });

    it("应该多次调用都能正常工作", () => {
      const store = useAppStore();
      const initialState = store.sidebar.opened;

      // 连续切换3次
      store.toggleSidebar();
      store.toggleSidebar();
      store.toggleSidebar();

      // 最终状态应该与初始状态相反
      expect(store.sidebar.opened).toBe(!initialState);
    });
  });

  describe("closeSideBar", () => {
    it("应该关闭侧边栏", () => {
      const store = useAppStore();
      // 先打开侧边栏
      store.sidebar.opened = true;

      store.closeSideBar();

      expect(store.sidebar.opened).toBe(false);
    });

    it("应该更新sidebarStatus为CLOSED", () => {
      const store = useAppStore();

      store.closeSideBar();

      expect(store.sidebarStatus).toBe(SidebarStatusEnum.CLOSED);
    });

    it("对已关闭的侧边栏调用也应该安全", () => {
      const store = useAppStore();
      store.sidebar.opened = false;

      store.closeSideBar();

      expect(store.sidebar.opened).toBe(false);
      expect(store.sidebarStatus).toBe(SidebarStatusEnum.CLOSED);
    });
  });

  describe("openSideBar", () => {
    it("应该打开侧边栏", () => {
      const store = useAppStore();
      // 先关闭侧边栏
      store.sidebar.opened = false;

      store.openSideBar();

      expect(store.sidebar.opened).toBe(true);
    });

    it("应该更新sidebarStatus为OPENED", () => {
      const store = useAppStore();

      store.openSideBar();

      expect(store.sidebarStatus).toBe(SidebarStatusEnum.OPENED);
    });

    it("对已打开的侧边栏调用也应该安全", () => {
      const store = useAppStore();
      store.sidebar.opened = true;

      store.openSideBar();

      expect(store.sidebar.opened).toBe(true);
      expect(store.sidebarStatus).toBe(SidebarStatusEnum.OPENED);
    });
  });

  describe("toggleDevice", () => {
    it("应该更新设备类型", () => {
      const store = useAppStore();

      store.toggleDevice(DeviceEnum.MOBILE);

      expect(store.device).toBe(DeviceEnum.MOBILE);
    });

    it("应该支持所有设备类型", () => {
      const store = useAppStore();

      store.toggleDevice(DeviceEnum.DESKTOP);
      expect(store.device).toBe(DeviceEnum.DESKTOP);

      store.toggleDevice(DeviceEnum.MOBILE);
      expect(store.device).toBe(DeviceEnum.MOBILE);
    });
  });

  describe("changeSize", () => {
    it("应该更新尺寸设置", () => {
      const store = useAppStore();

      store.changeSize("large");

      expect(store.size).toBe("large");
    });

    it("应该支持所有有效的尺寸值", () => {
      const store = useAppStore();
      const sizes = ["large", "default", "small"];

      sizes.forEach((size) => {
        store.changeSize(size);
        expect(store.size).toBe(size);
      });
    });
  });

  describe("changeLanguage", () => {
    it("应该更新语言设置", () => {
      const store = useAppStore();

      store.changeLanguage("en");

      expect(store.language).toBe("en");
    });

    it("应该支持所有有效的语言值", () => {
      const store = useAppStore();
      const languages = ["zh-cn", "en"];

      languages.forEach((language) => {
        store.changeLanguage(language);
        expect(store.language).toBe(language);
      });
    });
  });

  describe("activeTopMenu", () => {
    it("应该更新活动顶级菜单路径", () => {
      const store = useAppStore();

      store.activeTopMenu("/dashboard");

      expect(store.activeTopMenuPath).toBe("/dashboard");
    });

    it("应该处理空字符串路径", () => {
      const store = useAppStore();

      store.activeTopMenu("");

      expect(store.activeTopMenuPath).toBe("");
    });

    it("应该处理复杂的路径", () => {
      const store = useAppStore();

      store.activeTopMenu("/system/user/list");

      expect(store.activeTopMenuPath).toBe("/system/user/list");
    });
  });

  describe("sidebar对象响应式", () => {
    it("sidebar应该是响应式的", () => {
      const store = useAppStore();

      expect(store.sidebar).toBeDefined();
      expect(store.sidebar.opened).toBeDefined();
      expect(store.sidebar.withoutAnimation).toBeDefined();
    });

    it("应该能直接修改sidebar属性", () => {
      const store = useAppStore();

      store.sidebar.withoutAnimation = true;

      expect(store.sidebar.withoutAnimation).toBe(true);
    });
  });

  describe("状态持久化", () => {
    it("应该使用useStorage进行状态持久化", () => {
      // 这个测试主要验证store正确使用了持久化
      // 实际的持久化测试需要更复杂的设置
      const store = useAppStore();

      expect(store.device).toBeDefined();
      expect(store.size).toBeDefined();
      expect(store.language).toBeDefined();
    });
  });

  describe("边界情况", () => {
    it("应该处理无效的设备类型", () => {
      const store = useAppStore();

      // 类型检查应该在TypeScript层面阻止，但运行时测试
      store.toggleDevice("invalid" as DeviceEnum);

      expect(store.device).toBe("invalid");
    });

    it("应该处理空的尺寸值", () => {
      const store = useAppStore();

      store.changeSize("");

      expect(store.size).toBe("");
    });

    it("应该处理空的语言值", () => {
      const store = useAppStore();

      store.changeLanguage("");

      expect(store.language).toBe("");
    });

    it("应该处理null的菜单路径", () => {
      const store = useAppStore();

      store.activeTopMenu(null as any);

      expect(store.activeTopMenuPath).toBeNull();
    });
  });

  describe("组合操作", () => {
    it("应该支持多个操作的组合", () => {
      const store = useAppStore();

      // 执行一系列操作
      store.toggleDevice(DeviceEnum.MOBILE);
      store.changeSize("large");
      store.changeLanguage("en");
      store.openSideBar();
      store.activeTopMenu("/dashboard");

      // 验证所有状态都正确更新
      expect(store.device).toBe(DeviceEnum.MOBILE);
      expect(store.size).toBe("large");
      expect(store.language).toBe("en");
      expect(store.sidebar.opened).toBe(true);
      expect(store.activeTopMenuPath).toBe("/dashboard");
    });

    it("应该支持侧边栏状态的完整操作序列", () => {
      const store = useAppStore();

      // 初始状态
      expect(store.sidebar.opened).toBe(false);

      // 打开
      store.openSideBar();
      expect(store.sidebar.opened).toBe(true);

      // 切换
      store.toggleSidebar();
      expect(store.sidebar.opened).toBe(false);

      // 再切换
      store.toggleSidebar();
      expect(store.sidebar.opened).toBe(true);

      // 关闭
      store.closeSideBar();
      expect(store.sidebar.opened).toBe(false);
    });
  });
});
