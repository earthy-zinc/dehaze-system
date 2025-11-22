import { initImageInput } from "./modules/imageInput.js";
import { initAlgorithmSelect } from "./modules/algorithmSelect.js";
import { initProcessing } from "./modules/processing.js";
import { initSideBySide } from "./modules/sideBySide.js";
import { initOverlay } from "./modules/overlay.js";
import { initMagnifier } from "./modules/magnifier.js";
import { initFilter } from "./modules/filter.js";
import { initMetrics } from "./modules/metrics.js";
import { initAlgorithm } from "./modules/algorithm.js";
import { initDataset } from "./modules/dataset.js";
import { layoutManager } from "./modules/layoutManager.js";

// 全局状态管理
const AppState = {
  currentPage: "home",
  images: [],
  settings: {
    magnifierSize: 150,
    magnifierZoom: 2.5,
    magnifierShape: "circle",
  },
};

// Toast提示
export function showToast(message, duration = 2000) {
  const toast = document.getElementById("toast");
  const toastMessage = document.getElementById("toastMessage");
  toastMessage.textContent = message;
  toast.classList.remove("hidden");

  setTimeout(() => {
    toast.classList.add("hidden");
  }, duration);
}

// 加载提示
export function showLoading() {
  document.getElementById("loadingOverlay").classList.remove("hidden");
}

export function hideLoading() {
  document.getElementById("loadingOverlay").classList.add("hidden");
}

// 页面导航
function navigateToPage(pageId) {
  // 隐藏所有页面
  document.querySelectorAll(".page-section").forEach((section) => {
    section.classList.remove("active");
  });

  // 显示目标页面
  const targetPage = document.getElementById(pageId);
  if (targetPage) {
    targetPage.classList.add("active");
    AppState.currentPage = pageId;

    // 更新导航状态
    updateNavigation(pageId);

    // 初始化页面内容
    initPageContent(pageId);
  }

  // 关闭侧边菜单
  closeSideMenu();
}

// 更新导航状态
function updateNavigation(pageId) {
  // 更新底部导航
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.remove("active");
    if (item.getAttribute("href") === `#${pageId}`) {
      item.classList.add("active");
    }
  });

  // 更新侧边菜单
  document.querySelectorAll(".menu-item").forEach((item) => {
    item.classList.remove("active");
    if (item.getAttribute("href") === `#${pageId}`) {
      item.classList.add("active");
    }
  });
}

// 初始化页面内容
function initPageContent(pageId) {
  const page = document.getElementById(pageId);

  // 如果页面已经初始化过，不重复初始化
  if (page.dataset.initialized === "true") {
    return;
  }

  switch (pageId) {
    case "image-input":
      initImageInput(page);
      break;
    case "algorithm-select":
      initAlgorithmSelect(page);
      break;
    case "processing":
      initProcessing(page);
      break;
    case "side-by-side":
      initSideBySide(page);
      break;
    case "overlay":
      initOverlay(page);
      break;
    case "magnifier":
      initMagnifier(page);
      break;
    case "filter":
      initFilter(page);
      break;
    case "metrics":
      initMetrics(page);
      break;
    case "algorithm":
      initAlgorithm(page);
      break;
    case "dataset":
      initDataset(page);
      break;
  }

  page.dataset.initialized = "true";
}

// 侧边菜单控制
function openSideMenu() {
  const sideMenu = document.getElementById("sideMenu");
  const menuPanel = document.getElementById("menuPanel");
  sideMenu.classList.remove("hidden");
  setTimeout(() => {
    menuPanel.style.transform = "translateX(0)";
  }, 10);
}

function closeSideMenu() {
  const menuPanel = document.getElementById("menuPanel");
  menuPanel.style.transform = "translateX(100%)";
  setTimeout(() => {
    const sideMenu = document.getElementById("sideMenu");
    sideMenu.classList.add("hidden");
  }, 300);
}

// 图片加载工具
export function loadImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve({ file, url: e.target.result, img });
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// 创建图片上传区域
export function createUploadArea(container, onUpload, options = {}) {
  const {
    multiple = false,
    accept = "image/*",
    text = "点击或拖拽上传图片",
  } = options;

  const uploadArea = document.createElement("div");
  uploadArea.className = "upload-area";
  uploadArea.innerHTML = `
        <i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-3"></i>
        <p class="text-gray-600 font-medium">${text}</p>
        <p class="text-sm text-gray-400 mt-2">支持 JPG、PNG、WebP 格式</p>
    `;

  const input = document.createElement("input");
  input.type = "file";
  input.accept = accept;
  input.multiple = multiple;
  input.style.display = "none";

  // 点击上传区域触发文件选择
  uploadArea.addEventListener("click", () => input.click());

  // 拖拽上传事件处理
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", async (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");

    const files = Array.from(e.dataTransfer.files).filter((file) =>
      file.type.startsWith("image/")
    );

    if (files.length > 0) {
      showLoading();
      try {
        const images = await Promise.all(files.map(loadImage));
        onUpload(images);
      } catch (error) {
        showToast("图片加载失败");
      } finally {
        hideLoading();
      }
    }
  });

  // 文件输入变化事件（点击上传）
  input.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      showLoading();
      try {
        const images = await Promise.all(files.map(loadImage));
        onUpload(images);
      } catch (error) {
        showToast("图片加载失败");
      } finally {
        hideLoading();
      }
    }
    input.value = ""; // 清空输入框，防止重复上传
  });

  container.appendChild(uploadArea);
  container.appendChild(input);

  return { uploadArea, input };
}

// 创建图片预览
export function createImagePreview(imageData, onRemove) {
  const preview = document.createElement("div");
  preview.className = "image-preview";

  const img = document.createElement("img");
  img.src = imageData.url;
  img.alt = imageData.file.name;

  const removeBtn = document.createElement("div");
  removeBtn.className = "remove-btn";
  removeBtn.innerHTML = '<i class="fas fa-times"></i>';
  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    onRemove();
  });

  preview.appendChild(img);
  preview.appendChild(removeBtn);

  return preview;
}

// 计算图像指标（模拟）
export function calculateMetrics(img1, img2) {
  // 这里是模拟计算，实际应用中需要使用真实的图像处理算法
  return {
    ssim: (0.85 + Math.random() * 0.1).toFixed(4),
    psnr: (28 + Math.random() * 5).toFixed(2),
    entropy: (6.5 + Math.random() * 1.5).toFixed(3),
    gradient: (12 + Math.random() * 8).toFixed(2),
    runtime: (50 + Math.random() * 150).toFixed(0),
  };
}

// 获取应用状态
export function getAppState() {
  return AppState;
}

// 更新应用状态
export function updateAppState(updates) {
  Object.assign(AppState, updates);
}

// 初始化应用
function initApp() {
  // 菜单按钮事件
  document.getElementById("menuBtn").addEventListener("click", openSideMenu);
  document
    .getElementById("closeMenuBtn")
    .addEventListener("click", closeSideMenu);

  // 点击遮罩关闭菜单
  document.getElementById("sideMenu").addEventListener("click", (e) => {
    if (e.target.id === "sideMenu") {
      closeSideMenu();
    }
  });

  // 导航事件
  document.querySelectorAll(".nav-item, .menu-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const pageId = e.currentTarget.getAttribute("href").substring(1);
      navigateToPage(pageId);
    });
  });

  // 功能卡片点击事件
  document.querySelectorAll(".feature-card").forEach((card) => {
    card.addEventListener("click", () => {
      const target = card.dataset.target;
      if (target) {
        navigateToPage(target);
      }
    });
  });

  // 处理 URL hash 变化
  window.addEventListener("hashchange", () => {
    const hash = window.location.hash.substring(1);
    if (hash) {
      navigateToPage(hash);
    }
  });

  // 初始化页面
  const initialHash = window.location.hash.substring(1);
  if (initialHash) {
    navigateToPage(initialHash);
  }

  // 阻止双击缩放（移动端优化）
  let lastTouchEnd = 0;
  document.addEventListener(
    "touchend",
    (e) => {
      const now = Date.now();
      if (now - lastTouchEnd <= 300) {
        e.preventDefault();
      }
      lastTouchEnd = now;
    },
    false
  );

  // 初始化首页交互动画
  initHomePageAnimations();

  console.log("图像去雾对比系统已启动");
}

function initHomePageAnimations() {
  // 滚动视差效果
  let ticking = false;

  window.addEventListener("scroll", () => {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        handleScrollAnimations();
        ticking = false;
      });
    }
    ticking = true;
  });

  function handleScrollAnimations() {
    const scrollY = window.scrollY;

    // Hero区域视差效果
    const heroSection = document.querySelector(".hero-section");
    if (heroSection) {
      const heroContent = heroSection.querySelector(".hero-content");
      if (heroContent) {
        heroContent.style.transform = `translateY(${scrollY * 0.3}px)`;
        heroContent.style.opacity = Math.max(0, 1 - scrollY / 500);
      }
    }

    // 元素进入视口动画
    const animateElements = document.querySelectorAll(
      ".showcase-section, .workflow-step, .tool-card, .spec-card"
    );
    animateElements.forEach((element) => {
      const rect = element.getBoundingClientRect();
      const isVisible = rect.top < window.innerHeight * 0.8;

      if (isVisible && !element.classList.contains("animated")) {
        element.classList.add("animated");
        element.style.animation = "fadeInUp 0.6s ease-out forwards";
      }
    });
  }

  // 添加 fadeInUp 动画
  const style = document.createElement("style");
  style.textContent = `
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .workflow-step, .tool-card, .spec-card {
            opacity: 0;
        }

        .workflow-step.animated, .tool-card.animated, .spec-card.animated {
            opacity: 1;
        }
    `;
  document.head.appendChild(style);

  // 鼠标跟随效果（仅桌面端）
  if (window.innerWidth > 768) {
    let mouseX = 0;
    let mouseY = 0;
    let currentX = 0;
    let currentY = 0;

    document.addEventListener("mousemove", (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    });

    function animateMouseFollow() {
      currentX += (mouseX - currentX) * 0.1;
      currentY += (mouseY - currentY) * 0.1;

      const workflowSteps = document.querySelectorAll(".workflow-step");
      workflowSteps.forEach((step, index) => {
        const rect = step.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        const deltaX = (currentX - centerX) / 50;
        const deltaY = (currentY - centerY) / 50;

        step.style.transform = `perspective(1000px) rotateY(${deltaX}deg) rotateX(${-deltaY}deg)`;
      });

      requestAnimationFrame(animateMouseFollow);
    }

    animateMouseFollow();
  }

  // 数字滚动动画
  function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        current = target;
        clearInterval(timer);
      }
      element.textContent = Math.floor(current);
    }, 16);
  }

  // 监听规格卡片进入视口
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && entry.target.dataset.animated) {
          entry.target.dataset.animated = "true";
          const valueElement = entry.target.querySelector(".spec-value");
          if (valueElement) {
            const text = valueElement.textContent;
            const number = parseInt(text);
            if (!isNaN(number)) {
              valueElement.textContent = "0";
              setTimeout(() => {
                animateCounter(valueElement, number);
              }, 200);
            }
          }
        }
      });
    },
    { threshold: 0.5 }
  );

  document.querySelectorAll(".spec-card").forEach((card) => {
    observer.observe(card);
  });

  // 平滑滚动到锚点
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      const href = this.getAttribute("href");
      if (href !== "#" && href.length > 1) {
        const targetId = href.substring(1);
        const targetElement = document.getElementById(targetId);

        if (targetElement && targetElement.classList.contains("page-section")) {
          return;
        }

        if (targetElement) {
          e.preventDefault();
          targetElement.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }
      }
    });
  });

  setTimeout(() => {
    handleScrollAnimations();
  }, 100);
}

// 监听布局变化事件
window.addEventListener("layoutchange", (e) => {
  const { deviceType, orientation } = e.detail;
  console.log(`布局变化: ${deviceType} - ${orientation}`);

  // 根据设备类型调整 UI
  if (layoutManager.isDesktop()) {
    document.body.classList.add("desktop-mode");
  } else {
    document.body.classList.remove("desktop-mode");
  }
});

// 监听导航事件（来自桌面侧边栏）
window.addEventListener("navigate", (e) => {
  const { page } = e.detail;
  navigateToPage(page);
});

// 监听滑动手势事件
window.addEventListener("swipe", (e) => {
  const { direction } = e.detail;

  // 获取当前页面索引
  const pages = ["home", "processing", "compare", "dataset"];
  const currentIndex = pages.indexOf(AppState.currentPage);

  if (direction === "left" && currentIndex < pages.length - 1) {
    // 向左滑动，切换到下一页
    navigateToPage(pages[currentIndex + 1]);
  } else if (direction === "right" && currentIndex > 0) {
    // 向右滑动，切换到上一页
    navigateToPage(pages[currentIndex - 1]);
  }
});

// 启动应用
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}

// 输出布局信息（调试用）
console.log("布局信息:", layoutManager.getLayoutInfo());
