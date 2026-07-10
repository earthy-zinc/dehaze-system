// 布局管理模块 - 设备检测和自适应布局
export class LayoutManager {
  constructor() {
    this.deviceType = this.detectDeviceType();
    this.orientation = this.detectOrientation();
    this.inputType = this.detectInputType();
    this.performance = this.detectPerformance();

    this.init();
  }

  init() {
    // 监听窗口变化
    window.addEventListener("resize", () => {
      this.handleResize();
    });

    // 监听方向变化
    window.addEventListener("orientationchange", () => {
      this.handleOrientationChange();
    });

    // 应用初始布局
    this.applyLayout();

    // 设置性能优化
    this.applyPerformanceOptimizations();
  }

  // 检测设备类型
  detectDeviceType() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const userAgent = navigator.userAgent.toLowerCase();

    // 检测是否为移动设备
    const isMobile =
      /mobile|android|iphone|ipod|blackberry|iemobile|opera mini/i.test(
        userAgent
      );
    const isTablet = /ipad|android(?!.*mobile)|tablet/i.test(userAgent);

    if (width <= 375) {
      return "phone-small";
    } else if (width <= 767) {
      return "phone";
    } else if (width <= 1023) {
      return isTablet || (isMobile && width >= 768) ? "tablet" : "phone-large";
    } else if (width <= 1439) {
      return "desktop-small";
    } else {
      return "desktop";
    }
  }

  // 检测屏幕方向
  detectOrientation() {
    return window.innerWidth > window.innerHeight ? "landscape" : "portrait";
  }

  // 检测输入类型
  detectInputType() {
    // 检测是否支持触摸
    const hasTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0;

    // 检测是否有鼠标
    const hasMouse = window.matchMedia("(pointer: fine)").matches;

    if (hasTouch && !hasMouse) {
      return "touch";
    } else if (hasMouse && !hasTouch) {
      return "mouse";
    } else {
      return "hybrid";
    }
  }

  // 检测设备性能
  detectPerformance() {
    // 基于设备类型和硬件并发数估算性能
    const cores = navigator.hardwareConcurrency || 2;
    const memory = navigator.deviceMemory || 4;

    if (this.deviceType.includes("desktop") && cores >= 4 && memory >= 8) {
      return "high";
    } else if (cores >= 2 && memory >= 4) {
      return "medium";
    } else {
      return "low";
    }
  }

  // 应用布局
  applyLayout() {
    const body = document.body;

    // 清除旧的类名
    body.className = body.className.replace(
      /device-\S+|orientation-\S+|input-\S+|performance-\S+/g,
      ""
    );

    // 添加新的类名
    body.classList.add(`device-${this.deviceType}`);
    body.classList.add(`orientation-${this.orientation}`);
    body.classList.add(`input-${this.inputType}`);
    body.classList.add(`performance-${this.performance}`);

    // 应用特定布局
    this.applyDeviceSpecificLayout();
  }

  // 应用设备特定布局
  applyDeviceSpecificLayout() {
    const main = document.querySelector("main");
    const nav = document.querySelector("nav");

    if (!main) return;

    switch (this.deviceType) {
      case "desktop":
      case "desktop-small":
        this.applyDesktopLayout(main, nav);
        break;

      case "tablet":
        if (this.orientation === "landscape") {
          this.applyTabletLandscapeLayout(main, nav);
        } else {
          this.applyTabletPortraitLayout(main, nav);
        }
        break;

      case "phone":
      case "phone-small":
      case "phone-large":
        if (this.orientation === "landscape") {
          this.applyPhoneLandscapeLayout(main, nav);
        } else {
          this.applyPhonePortraitLayout(main, nav);
        }
        break;
    }
  }

  // PC端布局
  applyDesktopLayout(main, nav) {
    // 显示底部导航
    if (nav) {
      nav.style.display = "flex";
    }

    // // 移除桌面侧边栏（首页不需要）
    // const existingSidebar = document.querySelector('.desktop-sidebar');
    // if (existingSidebar) {
    //     existingSidebar.remove();
    // }

    // 启用悬停效果
    document.body.classList.add("enable-hover-effects");
  }

  // 平板横屏布局
  applyTabletLandscapeLayout(main, nav) {
    // 显示底部导航
    if (nav) {
      nav.style.display = "flex";
    }

    // 移除桌面侧边栏
    const sidebar = document.querySelector(".desktop-sidebar");
    if (sidebar) {
      sidebar.remove();
    }

    // 启用触控优化
    document.body.classList.add("touch-optimized");
  }

  // 平板竖屏布局
  applyTabletPortraitLayout(main, nav) {
    // 显示底部导航
    if (nav) {
      nav.style.display = "flex";
    }

    // 启用触控优化
    document.body.classList.add("touch-optimized");
  }

  // 手机横屏布局
  applyPhoneLandscapeLayout(main, nav) {
    // 显示底部导航（紧凑模式）
    if (nav) {
      nav.style.display = "flex";
      nav.classList.add("compact-mode");
    }

    // 启用手势操作
    this.enableSwipeGestures(main);

    // 启用触控优化
    document.body.classList.add("touch-optimized");
  }

  // 手机竖屏布局
  applyPhonePortraitLayout(main, nav) {
    // 显示底部导航
    if (nav) {
      nav.style.display = "flex";
      nav.classList.remove("compact-mode");
    }

    // 启用手势操作
    this.enableSwipeGestures(main);

    // 启用触控优化
    document.body.classList.add("touch-optimized");
  }

  // 启用手势操作
  enableSwipeGestures(container) {
    let startX = 0;
    let startY = 0;
    let currentX = 0;
    let currentY = 0;

    container.addEventListener(
      "touchstart",
      (e) => {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
      },
      { passive: true }
    );

    container.addEventListener(
      "touchmove",
      (e) => {
        currentX = e.touches[0].clientX;
        currentY = e.touches[0].clientY;
      },
      { passive: true }
    );

    container.addEventListener("touchend", () => {
      const diffX = currentX - startX;
      const diffY = currentY - startY;

      // 判断是否为有效滑动
      if (Math.abs(diffX) > 50 && Math.abs(diffX) > Math.abs(diffY)) {
        if (diffX > 0) {
          // 向右滑动
          window.dispatchEvent(
            new CustomEvent("swipe", { detail: { direction: "right" } })
          );
        } else {
          // 向左滑动
          window.dispatchEvent(
            new CustomEvent("swipe", { detail: { direction: "left" } })
          );
        }
      }
    });
  }

  // 应用性能优化
  applyPerformanceOptimizations() {
    const body = document.body;

    switch (this.performance) {
      case "low":
        // 低性能设备：禁用复杂动画
        body.classList.add("reduce-animations");
        this.disableComplexAnimations();
        break;

      case "medium":
        // 中等性能：简化动画
        body.classList.add("simplified-animations");
        break;

      case "high":
        // 高性能：启用所有效果
        body.classList.add("full-animations");
        break;
    }
  }

  // 禁用复杂动画
  disableComplexAnimations() {
    const style = document.createElement("style");
    style.textContent = `
            .reduce-animations * {
                animation-duration: 0.1s !important;
                transition-duration: 0.1s !important;
            }
          
            .reduce-animations .feature-card:hover {
                transform: none !important;
            }
        `;
    document.head.appendChild(style);
  }

  // 处理窗口大小变化
  handleResize() {
    const oldDeviceType = this.deviceType;
    const oldOrientation = this.orientation;

    this.deviceType = this.detectDeviceType();
    this.orientation = this.detectOrientation();

    // 如果设备类型或方向改变，重新应用布局
    if (
      oldDeviceType !== this.deviceType ||
      oldOrientation !== this.orientation
    ) {
      this.applyLayout();

      // 触发布局变化事件
      window.dispatchEvent(
        new CustomEvent("layoutchange", {
          detail: {
            deviceType: this.deviceType,
            orientation: this.orientation,
          },
        })
      );
    }
  }

  // 处理方向变化
  handleOrientationChange() {
    setTimeout(() => {
      this.handleResize();
    }, 100);
  }

  // 获取当前布局信息
  getLayoutInfo() {
    return {
      deviceType: this.deviceType,
      orientation: this.orientation,
      inputType: this.inputType,
      performance: this.performance,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
    };
  }

  // 判断是否为移动设备
  isMobile() {
    return this.deviceType.includes("phone");
  }

  // 判断是否为平板
  isTablet() {
    return this.deviceType === "tablet";
  }

  // 判断是否为桌面
  isDesktop() {
    return this.deviceType.includes("desktop");
  }

  // 判断是否为触控设备
  isTouchDevice() {
    return this.inputType === "touch" || this.inputType === "hybrid";
  }
}

// 导出单例
export const layoutManager = new LayoutManager();
