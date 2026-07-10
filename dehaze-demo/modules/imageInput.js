// 图像输入模块
import {
  createUploadArea,
  showToast,
  loadImage,
  showLoading,
  hideLoading,
} from "../main.js";

// 样例图片库数据
const sampleImages = {
  light: [
    {
      id: 1,
      name: "轻度雾霾-城市街道",
      url: "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=800",
      difficulty: "简单",
    },
    {
      id: 2,
      name: "轻度雾霾-公园景观",
      url: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
      difficulty: "简单",
    },
    {
      id: 3,
      name: "轻度雾霾-建筑物",
      url: "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
      difficulty: "简单",
    },
    {
      id: 4,
      name: "轻度雾霾-山景",
      url: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
      difficulty: "简单",
    },
    {
      id: 5,
      name: "轻度雾霾-湖泊",
      url: "https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=800",
      difficulty: "简单",
    },
  ],
  medium: [
    {
      id: 6,
      name: "中度雾霾-城市天际线",
      url: "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800",
      difficulty: "中等",
    },
    {
      id: 7,
      name: "中度雾霾-道路",
      url: "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=800",
      difficulty: "中等",
    },
    {
      id: 8,
      name: "中度雾霾-森林",
      url: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
      difficulty: "中等",
    },
    {
      id: 9,
      name: "中度雾霾-海岸",
      url: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800",
      difficulty: "中等",
    },
    {
      id: 10,
      name: "中度雾霾-乡村",
      url: "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800",
      difficulty: "中等",
    },
  ],
  heavy: [
    {
      id: 11,
      name: "重度雾霾-城市中心",
      url: "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800",
      difficulty: "困难",
    },
    {
      id: 12,
      name: "重度雾霾-高速公路",
      url: "https://images.unsplash.com/photo-1465447142348-e9952c393450?w=800",
      difficulty: "困难",
    },
    {
      id: 13,
      name: "重度雾霾-山区",
      url: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
      difficulty: "困难",
    },
    {
      id: 14,
      name: "重度雾霾-港口",
      url: "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=800",
      difficulty: "困难",
    },
    {
      id: 15,
      name: "重度雾霾-工业区",
      url: "https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?w=800",
      difficulty: "困难",
    },
  ],
  special: [
    {
      id: 16,
      name: "特殊场景-夜景雾霾",
      url: "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=800",
      difficulty: "困难",
    },
    {
      id: 17,
      name: "特殊场景-逆光雾霾",
      url: "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800",
      difficulty: "困难",
    },
    {
      id: 18,
      name: "特殊场景-雨雾",
      url: "https://images.unsplash.com/photo-1428908728789-d2de25dbd4e2?w=800",
      difficulty: "中等",
    },
    {
      id: 19,
      name: "特殊场景-晨雾",
      url: "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800",
      difficulty: "简单",
    },
    {
      id: 20,
      name: "特殊场景-雪雾",
      url: "https://images.unsplash.com/photo-1491002052546-bf38f186af56?w=800",
      difficulty: "中等",
    },
  ],
};

// 历史记录管理
class HistoryManager {
  constructor() {
    this.maxRecords = 20;
    this.storageKey = "dehaze_history";
  }

  getHistory() {
    try {
      const history = localStorage.getItem(this.storageKey);
      return history ? JSON.parse(history) : [];
    } catch (e) {
      return [];
    }
  }

  addRecord(record) {
    const history = this.getHistory();
    history.unshift({
      ...record,
      id: Date.now(),
      timestamp: new Date().toISOString(),
    });

    // 限制记录数量
    if (history.length > this.maxRecords) {
      history.splice(this.maxRecords);
    }

    localStorage.setItem(this.storageKey, JSON.stringify(history));
  }

  deleteRecord(id) {
    const history = this.getHistory();
    const filtered = history.filter((record) => record.id !== id);
    localStorage.setItem(this.storageKey, JSON.stringify(filtered));
  }

  clearHistory() {
    localStorage.removeItem(this.storageKey);
  }
}

const historyManager = new HistoryManager();

export function initImageInput(container) {
  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-image text-blue-500"></i>
                图像输入
            </h2>
            <p class="text-gray-600 text-sm mb-4">选择图片开始去雾处理</p>
          
            <!-- 输入方式选择 -->
            <div class="grid grid-cols-2 gap-3 mb-6">
                <button class="input-method-btn active" data-method="upload">
                    <i class="fas fa-cloud-upload-alt text-3xl mb-2"></i>
                    <span class="font-medium">上传图片</span>
                    <span class="text-xs opacity-75">从相册选择</span>
                </button>
              
                <button class="input-method-btn" data-method="camera">
                    <i class="fas fa-camera text-3xl mb-2"></i>
                    <span class="font-medium">拍照</span>
                    <span class="text-xs opacity-75">实时拍摄</span>
                </button>
              
                <button class="input-method-btn" data-method="sample">
                    <i class="fas fa-images text-3xl mb-2"></i>
                    <span class="font-medium">样例图片</span>
                    <span class="text-xs opacity-75">快速体验</span>
                </button>
              
                <button class="input-method-btn" data-method="history">
                    <i class="fas fa-history text-3xl mb-2"></i>
                    <span class="font-medium">历史记录</span>
                    <span class="text-xs opacity-75">最近处理</span>
                </button>
            </div>
          
            <!-- 上传区域 -->
            <div id="uploadArea" class="input-content active"></div>
          
            <!-- 拍照区域 -->
            <div id="cameraArea" class="input-content hidden">
                <div class="bg-gray-100 rounded-xl p-6 text-center">
                    <i class="fas fa-camera text-6xl text-gray-400 mb-4"></i>
                    <p class="text-gray-600 mb-4">点击下方按钮打开相机</p>
                    <button id="openCameraBtn" class="btn btn-primary">
                        <i class="fas fa-camera"></i> 打开相机
                    </button>
                    <input type="file" id="cameraInput" accept="image/*" capture="environment" class="hidden">
                </div>
            </div>
          
            <!-- 样例图片库 -->
            <div id="sampleArea" class="input-content hidden">
                <div class="mb-4">
                    <div class="flex space-x-2 overflow-x-auto pb-2">
                        <button class="category-btn active" data-category="all">全部</button>
                        <button class="category-btn" data-category="light">轻度雾霾</button>
                        <button class="category-btn" data-category="medium">中度雾霾</button>
                        <button class="category-btn" data-category="heavy">重度雾霾</button>
                        <button class="category-btn" data-category="special">特殊场景</button>
                    </div>
                </div>
                <div id="sampleGrid" class="grid grid-cols-2 md:grid-cols-3 gap-3"></div>
            </div>
          
            <!-- 历史记录 -->
            <div id="historyArea" class="input-content hidden">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-sm text-gray-600">最近处理的图片</span>
                    <button id="clearHistoryBtn" class="text-sm text-red-500 hover:text-red-600">
                        <i class="fas fa-trash"></i> 清空
                    </button>
                </div>
                <div id="historyList"></div>
            </div>
          
            <!-- 图片预览 -->
            <div id="previewSection" class="hidden mt-6">
                <div class="bg-white rounded-xl shadow-lg p-4">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="font-semibold text-gray-800">图片预览</h3>
                        <button id="removePreviewBtn" class="text-red-500 hover:text-red-600">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                  
                    <div class="bg-gray-100 rounded-lg overflow-hidden mb-3" style="height: 300px;">
                        <img id="previewImage" class="w-full h-full object-contain">
                    </div>
                  
                    <div class="grid grid-cols-2 gap-3 text-sm text-gray-600 mb-4">
                        <div>
                            <i class="fas fa-file-image text-blue-500"></i>
                            <span id="imageSize">-</span>
                        </div>
                        <div>
                            <i class="fas fa-expand-arrows-alt text-green-500"></i>
                            <span id="imageDimensions">-</span>
                        </div>
                    </div>
                  
                    <button id="nextStepBtn" class="btn btn-primary w-full">
                        <i class="fas fa-arrow-right"></i> 下一步：选择算法
                    </button>
                </div>
            </div>
        </div>
      
        <!-- 快速体验 -->
        <div class="card mt-4">
            <div class="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl p-6 text-white">
                <h3 class="text-lg font-bold mb-2">
                    <i class="fas fa-bolt"></i> 快速体验
                </h3>
                <p class="text-sm opacity-90 mb-4">使用样例图片快速体验去雾效果</p>
                <button id="quickStartBtn" class="bg-white text-blue-600 px-6 py-2 rounded-lg font-medium hover:bg-blue-50 transition-colors">
                    立即体验
                </button>
            </div>
        </div>
    `;

  let currentImage = null;
  let currentMethod = "upload";

  // 初始化上传区域
  const uploadArea = container.querySelector("#uploadArea");
  createUploadArea(uploadArea, handleImageUpload, {
    text: "点击或拖拽上传图片",
    accept: "image/jpeg,image/png,image/webp",
  });

  // 输入方式切换
  container.querySelectorAll(".input-method-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentMethod = btn.dataset.method;

      // 更新按钮状态
      container.querySelectorAll(".input-method-btn").forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");

      // 切换内容区域
      container.querySelectorAll(".input-content").forEach((content) => {
        content.classList.add("hidden");
      });

      const targetArea = container.querySelector(`#${currentMethod}Area`);
      if (targetArea) {
        targetArea.classList.remove("hidden");
      }

      // 特殊处理
      if (currentMethod === "sample") {
        renderSampleImages("all");
      } else if (currentMethod === "history") {
        renderHistory();
      }
    });
  });

  // 拍照功能
  const cameraInput = container.querySelector("#cameraInput");
  container.querySelector("#openCameraBtn").addEventListener("click", () => {
    cameraInput.click();
  });

  cameraInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) {
      showLoading();
      try {
        const imageData = await loadImage(file);
        handleImageUpload([imageData]);
      } catch (error) {
        showToast("图片加载失败");
      } finally {
        hideLoading();
      }
    }
  });

  // 样例分类切换
  container.querySelectorAll(".category-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const category = btn.dataset.category;

      container.querySelectorAll(".category-btn").forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");

      renderSampleImages(category);
    });
  });

  // 清空历史
  container.querySelector("#clearHistoryBtn").addEventListener("click", () => {
    if (confirm("确定要清空所有历史记录吗？")) {
      historyManager.clearHistory();
      renderHistory();
      showToast("历史记录已清空");
    }
  });

  // 移除预览
  container.querySelector("#removePreviewBtn").addEventListener("click", () => {
    currentImage = null;
    container.querySelector("#previewSection").classList.add("hidden");
  });

  // 下一步按钮
  container.querySelector("#nextStepBtn").addEventListener("click", () => {
    if (currentImage) {
      // 保存当前图片到全局状态
      window.dehazeApp = window.dehazeApp || {};
      window.dehazeApp.currentImage = currentImage;

      // 跳转到算法选择页面
      window.location.hash = "#algorithm-select";
    }
  });

  // 快速体验
  container.querySelector("#quickStartBtn").addEventListener("click", () => {
    // 随机选择一张样例图片
    const allSamples = [
      ...sampleImages.light,
      ...sampleImages.medium,
      ...sampleImages.heavy,
    ];
    const randomSample =
      allSamples[Math.floor(Math.random() * allSamples.length)];

    loadSampleImage(randomSample);
  });

  function handleImageUpload(images) {
    if (images.length > 0) {
      currentImage = images[0];
      showPreview(currentImage);
    }
  }

  function showPreview(imageData) {
    const previewSection = container.querySelector("#previewSection");
    const previewImage = container.querySelector("#previewImage");
    const imageSize = container.querySelector("#imageSize");
    const imageDimensions = container.querySelector("#imageDimensions");

    previewImage.src = imageData.url;

    // 显示文件大小
    const sizeInMB = (imageData.file.size / 1024 / 1024).toFixed(2);
    imageSize.textContent = `${sizeInMB} MB`;

    // 显示图片尺寸
    const img = imageData.img;
    imageDimensions.textContent = `${img.width} × ${img.height}`;

    previewSection.classList.remove("hidden");

    // 滚动到预览区域
    previewSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderSampleImages(category) {
    const sampleGrid = container.querySelector("#sampleGrid");
    sampleGrid.innerHTML = "";

    let samples = [];
    if (category === "all") {
      samples = [
        ...sampleImages.light,
        ...sampleImages.medium,
        ...sampleImages.heavy,
        ...sampleImages.special,
      ];
    } else {
      samples = sampleImages[category] || [];
    }

    samples.forEach((sample) => {
      const card = document.createElement("div");
      card.className = "sample-card";
      card.innerHTML = `
                <div class="relative bg-white rounded-lg shadow-md overflow-hidden cursor-pointer hover:shadow-xl transition-shadow">
                    <img src="${sample.url}" alt="${
        sample.name
      }" class="w-full h-32 object-cover">
                    <div class="p-3">
                        <p class="text-sm font-medium text-gray-800 truncate">${
                          sample.name
                        }</p>
                        <div class="flex items-center justify-between mt-2">
                            <span class="text-xs px-2 py-1 rounded-full ${getDifficultyClass(
                              sample.difficulty
                            )}">
                                ${sample.difficulty}
                            </span>
                            <button class="text-blue-500 hover:text-blue-600 text-sm">
                                <i class="fas fa-arrow-right"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;

      card.addEventListener("click", () => {
        loadSampleImage(sample);
      });

      sampleGrid.appendChild(card);
    });
  }

  function getDifficultyClass(difficulty) {
    const classes = {
      简单: "bg-green-100 text-green-700",
      中等: "bg-yellow-100 text-yellow-700",
      困难: "bg-red-100 text-red-700",
    };
    return classes[difficulty] || "bg-gray-100 text-gray-700";
  }

  async function loadSampleImage(sample) {
    showLoading();
    try {
      const response = await fetch(sample.url);
      const blob = await response.blob();
      const file = new File([blob], sample.name + ".jpg", {
        type: "image/jpeg",
      });
      const imageData = await loadImage(file);

      currentImage = {
        ...imageData,
        sampleInfo: sample,
      };

      showPreview(currentImage);
      showToast("样例图片加载成功");
    } catch (error) {
      showToast("样例图片加载失败，请重试");
    } finally {
      hideLoading();
    }
  }

  function renderHistory() {
    const historyList = container.querySelector("#historyList");
    const history = historyManager.getHistory();

    if (history.length === 0) {
      historyList.innerHTML = `
                <div class="text-center py-8 text-gray-400">
                    <i class="fas fa-inbox text-4xl mb-2"></i>
                    <p>暂无历史记录</p>
                </div>
            `;
      return;
    }

    historyList.innerHTML = history
      .map(
        (record) => `
            <div class="history-item bg-white rounded-lg shadow-md p-3 mb-3 flex items-center space-x-3">
                <img src="${
                  record.thumbnail
                }" class="w-16 h-16 object-cover rounded-lg">
                <div class="flex-1 min-w-0">
                    <p class="text-sm font-medium text-gray-800 truncate">${
                      record.fileName
                    }</p>
                    <p class="text-xs text-gray-500">${formatTime(
                      record.timestamp
                    )}</p>
                    <p class="text-xs text-blue-600">${
                      record.algorithm || "未知算法"
                    }</p>
                </div>
                <div class="flex space-x-2">
                    <button class="history-load-btn text-blue-500 hover:text-blue-600" data-id="${
                      record.id
                    }">
                        <i class="fas fa-redo"></i>
                    </button>
                    <button class="history-delete-btn text-red-500 hover:text-red-600" data-id="${
                      record.id
                    }">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `
      )
      .join("");

    // 绑定事件
    historyList.querySelectorAll(".history-load-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = parseInt(btn.dataset.id);
        const record = history.find((r) => r.id === id);
        if (record) {
          loadHistoryRecord(record);
        }
      });
    });

    historyList.querySelectorAll(".history-delete-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = parseInt(btn.dataset.id);
        historyManager.deleteRecord(id);
        renderHistory();
        showToast("记录已删除");
      });
    });
  }

  function loadHistoryRecord(record) {
    // 从历史记录加载图片
    showToast("正在加载历史记录...");
    // 这里简化处理，实际应该从缓存或服务器加载
  }

  function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return "刚刚";
    if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
    return date.toLocaleDateString();
  }
}

// 导出历史管理器供其他模块使用
export { historyManager };
