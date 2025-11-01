import { translateRouteTitle } from "../i18n";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock i18n - 避免hoisting问题，直接在vi.mock中定义
vi.mock("@/lang/index", () => ({
  default: {
    global: {
      te: vi.fn(),
      t: vi.fn(),
    },
  },
}));

// 导入mock类型以便在测试中使用
import i18n from "@/lang/index";

describe("translateRouteTitle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("国际化标题翻译", () => {
    it("应该返回翻译后的标题当存在国际化配置时", () => {
      // Arrange
      const title = "Dashboard";
      const translatedTitle = "仪表板";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.Dashboard");
      expect(i18n.global.t).toHaveBeenCalledWith("route.Dashboard");
      expect(result).toBe(translatedTitle);
    });

    it("应该返回原始标题当不存在国际化配置时", () => {
      // Arrange
      const title = "Dashboard";
      (i18n.global.te as any).mockReturnValue(false);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.Dashboard");
      expect(i18n.global.t).not.toHaveBeenCalled();
      expect(result).toBe(title);
    });

    it("应该处理空字符串标题", () => {
      // Arrange
      const title = "";
      (i18n.global.te as any).mockReturnValue(false);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.");
      expect(result).toBe("");
    });

    it("应该处理特殊字符的标题", () => {
      // Arrange
      const title = "User & Profile";
      const translatedTitle = "用户与资料";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.User & Profile");
      expect(i18n.global.t).toHaveBeenCalledWith("route.User & Profile");
      expect(result).toBe(translatedTitle);
    });

    it("应该处理数字标题", () => {
      // Arrange
      const title = "404";
      const translatedTitle = "页面未找到";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.404");
      expect(i18n.global.t).toHaveBeenCalledWith("route.404");
      expect(result).toBe(translatedTitle);
    });

    it("应该处理带空格的标题", () => {
      // Arrange
      const title = "User Management";
      const translatedTitle = "用户管理";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.User Management");
      expect(i18n.global.t).toHaveBeenCalledWith("route.User Management");
      expect(result).toBe(translatedTitle);
    });

    it("应该处理null和undefined标题", () => {
      // Arrange
      (i18n.global.te as any).mockReturnValue(false);

      // Act & Assert - null
      const resultNull = translateRouteTitle(null as any);
      expect(i18n.global.te).toHaveBeenCalledWith("route.null");
      expect(resultNull).toBeNull();

      // Act & Assert - undefined
      const resultUndefined = translateRouteTitle(undefined as any);
      expect(i18n.global.te).toHaveBeenCalledWith("route.undefined");
      expect(resultUndefined).toBeUndefined();
    });

    it("应该处理复杂的路由标题", () => {
      // Arrange
      const title = "system.user.profile";
      const translatedTitle = "系统用户资料";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.system.user.profile");
      expect(i18n.global.t).toHaveBeenCalledWith("route.system.user.profile");
      expect(result).toBe(translatedTitle);
    });

    it("应该处理驼峰命名的标题", () => {
      // Arrange
      const title = "userProfile";
      const translatedTitle = "用户资料";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.userProfile");
      expect(i18n.global.t).toHaveBeenCalledWith("route.userProfile");
      expect(result).toBe(translatedTitle);
    });

    it("应该处理中英文混合的标题", () => {
      // Arrange
      const title = "用户 User";
      (i18n.global.te as any).mockReturnValue(false);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.用户 User");
      expect(result).toBe(title);
    });
  });

  describe("i18n API 交互", () => {
    it("应该正确构造国际化键名", () => {
      // Arrange
      const title = "Login";
      (i18n.global.te as any).mockReturnValue(false);

      // Act
      translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledTimes(1);
      expect(i18n.global.te).toHaveBeenCalledWith("route.Login");
    });

    it("应该在有国际化配置时调用翻译方法", () => {
      // Arrange
      const title = "Logout";
      const translatedTitle = "登出";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      translateRouteTitle(title);

      // Assert
      expect(i18n.global.t).toHaveBeenCalledTimes(1);
      expect(i18n.global.t).toHaveBeenCalledWith("route.Logout");
    });

    it("应该在没有国际化配置时不调用翻译方法", () => {
      // Arrange
      const title = "Settings";
      (i18n.global.te as any).mockReturnValue(false);

      // Act
      translateRouteTitle(title);

      // Assert
      expect(i18n.global.t).not.toHaveBeenCalled();
    });
  });

  describe("边界情况", () => {
    it("应该处理非常长的标题", () => {
      // Arrange
      const longTitle =
        "This is a very long title that might be used in some applications for testing purposes";
      (i18n.global.te as any).mockReturnValue(false);

      // Act
      const result = translateRouteTitle(longTitle);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith(`route.${longTitle}`);
      expect(result).toBe(longTitle);
    });

    it("应该处理只有空格的标题", () => {
      // Arrange
      const title = "   ";
      (i18n.global.te as any).mockReturnValue(false);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.   ");
      expect(result).toBe("   ");
    });

    it("应该处理特殊符号标题", () => {
      // Arrange
      const title = "!@#$%^&*()";
      (i18n.global.te as any).mockReturnValue(false);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.!@#$%^&*()");
      expect(result).toBe(title);
    });

    it("应该处理emoji标题", () => {
      // Arrange
      const title = "🏠 Home";
      const translatedTitle = "🏠 主页";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      const result = translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledWith("route.🏠 Home");
      expect(i18n.global.t).toHaveBeenCalledWith("route.🏠 Home");
      expect(result).toBe(translatedTitle);
    });
  });

  describe("性能和效率", () => {
    it("应该只调用一次te方法", () => {
      // Arrange
      const title = "Test";
      (i18n.global.te as any).mockReturnValue(false);

      // Act
      translateRouteTitle(title);

      // Assert
      expect(i18n.global.te).toHaveBeenCalledTimes(1);
    });

    it("应该在需要时只调用一次t方法", () => {
      // Arrange
      const title = "Test";
      const translatedTitle = "测试";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      translateRouteTitle(title);

      // Assert
      expect(i18n.global.t).toHaveBeenCalledTimes(1);
    });
  });

  describe("类型安全", () => {
    it("应该处理各种类型的输入", () => {
      // Arrange
      (i18n.global.te as any).mockReturnValue(false);

      // Test different types
      expect(translateRouteTitle("string" as any)).toBe("string");
      expect(translateRouteTitle(123 as any)).toBe(123);
      expect(translateRouteTitle(true as any)).toBe(true);
      expect(translateRouteTitle(false as any)).toBe(false);
      expect(translateRouteTitle({} as any)).toEqual({});
    });
  });

  describe("实际使用场景", () => {
    it("应该处理常见的路由标题", () => {
      const commonTitles = [
        "Dashboard",
        "Profile",
        "Settings",
        "Login",
        "Logout",
        "Register",
        "404",
        "500",
      ];
      (i18n.global.te as any).mockReturnValue(false);

      commonTitles.forEach((title) => {
        // Act
        const result = translateRouteTitle(title);

        // Assert
        expect(result).toBe(title);
        expect(i18n.global.te).toHaveBeenCalledWith(`route.${title}`);
      });
    });

    it("应该处理嵌套路由标题", () => {
      // Arrange
      const nestedTitle = "system.permission.role";
      const translatedTitle = "系统权限角色";
      (i18n.global.te as any).mockReturnValue(true);
      (i18n.global.t as any).mockReturnValue(translatedTitle);

      // Act
      const result = translateRouteTitle(nestedTitle);

      // Assert
      expect(result).toBe(translatedTitle);
      expect(i18n.global.te).toHaveBeenCalledWith(`route.${nestedTitle}`);
    });
  });
});
