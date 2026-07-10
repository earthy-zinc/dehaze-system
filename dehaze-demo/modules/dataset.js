// 数据集管理模块 - 纯前端实现
import { showToast, showLoading, hideLoading } from "../main.js";

// Mock数据
const MOCK_DATASETS = [
  {
    id: 1,
    name: "RESIDE数据集",
    description: "大规模真实场景图像去雾数据集，包含室内外多种场景",
    creator: "Li Boyi",
    thumbnail:
      "https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=400&h=400&fit=crop",
    total_images: 13990,
    foggy_count: 6995,
    clear_count: 6995,
    annotated_count: 0,
    created_at: "2024-01-15T10:30:00Z",
    updated_at: "2024-01-15T10:30:00Z",
  },
  {
    id: 2,
    name: "O-HAZE数据集",
    description: "户外真实雾霾图像数据集，包含45对有雾/无雾图像",
    creator: "Ancuti Codruta",
    thumbnail:
      "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
    total_images: 90,
    foggy_count: 45,
    clear_count: 45,
    annotated_count: 0,
    created_at: "2024-01-10T14:20:00Z",
    updated_at: "2024-01-10T14:20:00Z",
  },
  {
    id: 3,
    name: "I-HAZE数据集",
    description: "室内真实雾霾图像数据集，包含35对有雾/无雾图像",
    creator: "Ancuti Codruta",
    thumbnail:
      "https://images.unsplash.com/photo-1497366216548-37526070297c?w=400&h=400&fit=crop",
    total_images: 70,
    foggy_count: 35,
    clear_count: 35,
    annotated_count: 0,
    created_at: "2024-01-08T09:15:00Z",
    updated_at: "2024-01-08T09:15:00Z",
  },
  {
    id: 4,
    name: "Dense-Haze数据集",
    description: "密集雾霾场景数据集，专注于极端雾霾条件",
    creator: "Ancuti Codruta",
    thumbnail:
      "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=400&h=400&fit=crop",
    total_images: 110,
    foggy_count: 55,
    clear_count: 55,
    annotated_count: 0,
    created_at: "2024-01-05T16:45:00Z",
    updated_at: "2024-01-05T16:45:00Z",
  },
  {
    id: 5,
    name: "NH-HAZE数据集",
    description: "非均匀雾霾数据集，模拟真实世界的复杂雾霾分布",
    creator: "Ancuti Codruta",
    thumbnail:
      "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=400&fit=crop",
    total_images: 110,
    foggy_count: 55,
    clear_count: 55,
    annotated_count: 0,
    created_at: "2024-01-03T11:30:00Z",
    updated_at: "2024-01-03T11:30:00Z",
  },
  {
    id: 6,
    name: "SOTS数据集",
    description: "合成雾霾数据集，包含室内外场景",
    creator: "Li Boyi",
    thumbnail:
      "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=400&h=400&fit=crop",
    total_images: 1000,
    foggy_count: 500,
    clear_count: 500,
    annotated_count: 0,
    created_at: "2024-01-01T08:00:00Z",
    updated_at: "2024-01-01T08:00:00Z",
  },
];

// 生成Mock图片数据
function generateMockImages(datasetId, count) {
  const images = [];
  const dataset = MOCK_DATASETS.find((d) => d.id === datasetId);
  if (!dataset) return images;

  const imageTypes = ["foggy", "clear", "annotated"];
  const sampleImages = [
    "https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1475924156734-496f6cac6ec1?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1418065460487-3e41a6c84dc5?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1511593358241-7eea1f3c84e5?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1542435503-956c469947f6?w=800&h=600&fit=crop",
    "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=800&h=600&fit=crop",
  ];

  for (let i = 0; i < count; i++) {
    const type = imageTypes[i % 3];
    const typeCount =
      type === "foggy"
        ? dataset.foggy_count
        : type === "clear"
        ? dataset.clear_count
        : dataset.annotated_count;

    if (typeCount === 0) continue;

    images.push({
      id: datasetId * 1000 + i,
      dataset_id: datasetId,
      filename: `${dataset.name.replace(/\s+/g, "_")}_${type}_${String(
        i + 1
      ).padStart(4, "0")}.jpg`,
      image_url: sampleImages[i % sampleImages.length],
      image_type: type,
      width: 1920,
      height: 1080,
      file_size: Math.floor(Math.random() * 2000000) + 500000,
      tags: `${type},${dataset.name}`,
      description: `${dataset.name}中的${
        type === "foggy" ? "有雾" : type === "clear" ? "无雾" : "标注"
      }图像`,
      created_at: new Date(
        Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000
      ).toISOString(),
    });
  }

  return images;
}

// 模块状态
let currentView = "list"; // list, detail
let currentDatasetId = null;
let currentImageType = "all";
let datasets = [];
let images = [];
let currentPage = 1;
let hasMore = true;
let isLoading = false;
let searchKeyword = "";

// 工具函数
function formatDate(dateString) {
  const date = new Date(dateString);
  const now = new Date();
  const diff = now - date;
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (days === 0) return "今天";
  if (days === 1) return "昨天";
  if (days < 7) return `${days}天前`;

  return date.toLocaleDateString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

function formatFileSize(bytes) {
  if (!bytes) return "-";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// Mock API请求
async function fetchDatasets(page = 1, search = "") {
  // 模拟网络延迟
  await new Promise((resolve) => setTimeout(resolve, 300));

  let filteredDatasets = [...MOCK_DATASETS];

  // 搜索过滤
  if (search) {
    const keyword = search.toLowerCase();
    filteredDatasets = filteredDatasets.filter(
      (d) =>
        d.name.toLowerCase().includes(keyword) ||
        (d.description && d.description.toLowerCase().includes(keyword))
    );
  }

  const pageSize = 10;
  const start = (page - 1) * pageSize;
  const end = start + pageSize;
  const list = filteredDatasets.slice(start, end);

  return {
    code: 0,
    data: {
      list,
      total: filteredDatasets.length,
      page,
      page_size: pageSize,
      total_pages: Math.ceil(filteredDatasets.length / pageSize),
    },
  };
}

async function fetchDatasetDetail(datasetId) {
  // 模拟网络延迟
  await new Promise((resolve) => setTimeout(resolve, 200));

  const dataset = MOCK_DATASETS.find((d) => d.id === datasetId);

  if (!dataset) {
    return { code: 404, message: "数据集不存在" };
  }

  return {
    code: 0,
    data: dataset,
  };
}

async function fetchDatasetImages(
  datasetId,
  page = 1,
  imageType = "all",
  search = ""
) {
  // 模拟网络延迟
  await new Promise((resolve) => setTimeout(resolve, 400));

  // 生成该数据集的所有图片
  let allImages = generateMockImages(datasetId, 60);

  // 类型过滤
  if (imageType !== "all") {
    allImages = allImages.filter((img) => img.image_type === imageType);
  }

  // 搜索过滤
  if (search) {
    const keyword = search.toLowerCase();
    allImages = allImages.filter(
      (img) =>
        img.filename.toLowerCase().includes(keyword) ||
        (img.tags && img.tags.toLowerCase().includes(keyword)) ||
        (img.description && img.description.toLowerCase().includes(keyword))
    );
  }

  const pageSize = 20;
  const start = (page - 1) * pageSize;
  const end = start + pageSize;
  const list = allImages.slice(start, end);

  return {
    code: 0,
    data: {
      list,
      total: allImages.length,
      page,
      page_size: pageSize,
      total_pages: Math.ceil(allImages.length / pageSize),
    },
  };
}

// 初始化模块
export function initDataset(container) {
  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-database text-teal-500"></i>
                数据集管理
            </h2>
            <p class="text-gray-600 text-sm mb-4">浏览和管理图像去雾数据集</p>
          
            <!-- 搜索栏 -->
            <div class="mb-4">
                <div class="relative">
                    <input 
                        type="text" 
                        id="datasetSearch" 
                        placeholder="搜索数据集或图片..."
                        class="w-full px-4 py-2 pl-10 pr-10 border border-gray-300 rounded-xl focus:outline-none focus:border-teal-500 transition-colors"
                    >
                    <i class="fas fa-search absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"></i>
                    <button id="clearSearch" class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hidden">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
          
            <!-- 数据集列表视图 -->
            <div id="datasetListView">
                <div id="datasetsList" class="space-y-3"></div>
                <div id="datasetsLoading" class="text-center py-8 hidden">
                    <i class="fas fa-spinner fa-spin text-3xl text-teal-500"></i>
                    <p class="text-gray-500 mt-2">加载中...</p>
                </div>
                <div id="datasetsEmpty" class="text-center py-12 hidden">
                    <i class="fas fa-folder-open text-6xl text-gray-300 mb-3"></i>
                    <p class="text-gray-500">暂无数据集</p>
                </div>
            </div>
          
            <!-- 数据集详情视图 -->
            <div id="datasetDetailView" class="hidden">
                <!-- 返回按钮 -->
                <button id="backToList" class="btn btn-secondary btn-sm mb-4">
                    <i class="fas fa-arrow-left"></i> 返回列表
                </button>
              
                <!-- 数据集信息 -->
                <div id="datasetInfo" class="bg-gradient-to-r from-teal-500 to-cyan-500 rounded-xl p-5 text-white mb-4"></div>
              
                <!-- 图片类型切换 -->
                <div class="flex overflow-x-auto space-x-2 mb-4 pb-2">
                    <button class="type-filter-btn active" data-type="all">
                        全部 <span class="count">0</span>
                    </button>
                    <button class="type-filter-btn" data-type="foggy">
                        有雾 <span class="count">0</span>
                    </button>
                    <button class="type-filter-btn" data-type="clear">
                        无雾 <span class="count">0</span>
                    </button>
                    <button class="type-filter-btn" data-type="annotated">
                        标注 <span class="count">0</span>
                    </button>
                </div>
              
                <!-- 图片网格 -->
                <div id="imagesGrid" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3"></div>
                <div id="imagesLoading" class="text-center py-8 hidden">
                    <i class="fas fa-spinner fa-spin text-3xl text-teal-500"></i>
                    <p class="text-gray-500 mt-2">加载中...</p>
                </div>
                <div id="imagesEmpty" class="text-center py-12 hidden">
                    <i class="fas fa-image text-6xl text-gray-300 mb-3"></i>
                    <p class="text-gray-500">暂无图片</p>
                </div>
                <div id="loadMoreTrigger" class="h-10"></div>
            </div>
        </div>
      
        <!-- 使用提示 -->
        <div class="bg-teal-50 border border-teal-200 rounded-xl p-4 mt-4">
            <div class="flex items-start space-x-3">
                <i class="fas fa-info-circle text-teal-500 mt-1"></i>
                <div class="text-sm text-teal-800">
                    <p class="font-semibold mb-1">使用提示</p>
                    <ul class="space-y-1 text-teal-700">
                        <li>• 点击数据集卡片查看详细信息</li>
                        <li>• 支持按类型筛选图片</li>
                        <li>• 点击图片可查看大图</li>
                        <li>• 支持搜索功能快速定位</li>
                    </ul>
                </div>
            </div>
        </div>
    `;

  // 初始化事件监听
  initEventListeners(container);

  // 加载数据集列表
  loadDatasets(container);
}

// 初始化事件监听
function initEventListeners(container) {
  // 搜索功能
  const searchInput = container.querySelector("#datasetSearch");
  const clearSearchBtn = container.querySelector("#clearSearch");

  let searchTimeout;
  searchInput.addEventListener("input", (e) => {
    clearTimeout(searchTimeout);
    searchKeyword = e.target.value.trim();
    clearSearchBtn.classList.toggle("hidden", !searchKeyword);

    searchTimeout = setTimeout(() => {
      currentPage = 1;
      if (currentView === "list") {
        loadDatasets(container);
      } else {
        loadImages(container);
      }
    }, 500);
  });

  clearSearchBtn.addEventListener("click", () => {
    searchInput.value = "";
    searchKeyword = "";
    clearSearchBtn.classList.add("hidden");
    currentPage = 1;
    if (currentView === "list") {
      loadDatasets(container);
    } else {
      loadImages(container);
    }
  });

  // 返回按钮
  container.querySelector("#backToList").addEventListener("click", () => {
    showListView(container);
  });

  // 类型筛选
  container.querySelectorAll(".type-filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const type = btn.dataset.type;
      if (currentImageType === type) return;

      currentImageType = type;
      currentPage = 1;

      container.querySelectorAll(".type-filter-btn").forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");

      loadImages(container);
    });
  });

  // 无限滚动
  const loadMoreTrigger = container.querySelector("#loadMoreTrigger");
  const observer = new IntersectionObserver(
    (entries) => {
      if (
        entries[0].isIntersecting &&
        hasMore &&
        !isLoading &&
        currentView === "detail"
      ) {
        currentPage++;
        loadImages(container, true);
      }
    },
    { threshold: 0.1 }
  );

  observer.observe(loadMoreTrigger);
}

// 加载数据集列表
async function loadDatasets(container) {
  if (isLoading) return;

  isLoading = true;
  const loadingEl = container.querySelector("#datasetsLoading");
  const emptyEl = container.querySelector("#datasetsEmpty");
  const listEl = container.querySelector("#datasetsList");

  loadingEl.classList.remove("hidden");
  emptyEl.classList.add("hidden");

  try {
    const result = await fetchDatasets(1, searchKeyword);

    if (result.code === 0) {
      datasets = result.data.list;

      if (datasets.length === 0) {
        emptyEl.classList.remove("hidden");
        listEl.innerHTML = "";
      } else {
        renderDatasets(container, datasets);
      }
    }
  } catch (error) {
    console.error("加载数据集失败:", error);
    showToast("加载失败，请重试");
  } finally {
    isLoading = false;
    loadingEl.classList.add("hidden");
  }
}

// 渲染数据集列表
function renderDatasets(container, datasets) {
  const listEl = container.querySelector("#datasetsList");

  listEl.innerHTML = datasets
    .map(
      (dataset) => `
        <div class="dataset-card bg-white rounded-xl shadow-md hover:shadow-xl transition-all cursor-pointer overflow-hidden" data-id="${
          dataset.id
        }">
            <div class="flex">
                <div class="w-32 h-32 flex-shrink-0 bg-gradient-to-br from-teal-400 to-cyan-500">
                    <img src="${dataset.thumbnail}" alt="${
        dataset.name
      }" class="w-full h-full object-cover">
                </div>
                <div class="flex-1 p-4">
                    <h3 class="font-bold text-gray-800 mb-1 line-clamp-1">${
                      dataset.name
                    }</h3>
                    <p class="text-sm text-gray-600 mb-3 line-clamp-2">${
                      dataset.description || "暂无描述"
                    }</p>
                    <div class="flex items-center space-x-4 text-xs text-gray-500">
                        <span><i class="fas fa-images text-teal-500"></i> ${
                          dataset.total_images
                        }</span>
                        <span><i class="fas fa-clock text-gray-400"></i> ${formatDate(
                          dataset.created_at
                        )}</span>
                    </div>
                </div>
            </div>
        </div>
    `
    )
    .join("");

  // 绑定点击事件
  listEl.querySelectorAll(".dataset-card").forEach((card) => {
    card.addEventListener("click", () => {
      const datasetId = parseInt(card.dataset.id);
      showDetailView(container, datasetId);
    });
  });
}

// 显示详情视图
async function showDetailView(container, datasetId) {
  currentView = "detail";
  currentDatasetId = datasetId;
  currentImageType = "all";
  currentPage = 1;
  images = [];

  container.querySelector("#datasetListView").classList.add("hidden");
  container.querySelector("#datasetDetailView").classList.remove("hidden");

  showLoading();

  try {
    // 加载数据集详情
    const result = await fetchDatasetDetail(datasetId);

    if (result.code === 0) {
      const dataset = result.data;
      renderDatasetInfo(container, dataset);

      // 更新类型计数
      container.querySelector('[data-type="all"] .count').textContent =
        dataset.total_images;
      container.querySelector('[data-type="foggy"] .count').textContent =
        dataset.foggy_count;
      container.querySelector('[data-type="clear"] .count').textContent =
        dataset.clear_count;
      container.querySelector('[data-type="annotated"] .count').textContent =
        dataset.annotated_count;
    }

    // 加载图片列表
    await loadImages(container);
  } catch (error) {
    console.error("加载详情失败:", error);
    showToast("加载失败，请重试");
  } finally {
    hideLoading();
  }
}

// 渲染数据集信息
function renderDatasetInfo(container, dataset) {
  const infoEl = container.querySelector("#datasetInfo");

  infoEl.innerHTML = `
        <h3 class="text-xl font-bold mb-2">${dataset.name}</h3>
        <p class="text-sm opacity-90 mb-4">${
          dataset.description || "暂无描述"
        }</p>
        <div class="grid grid-cols-4 gap-3">
            <div class="text-center">
                <div class="text-2xl font-bold">${dataset.total_images}</div>
                <div class="text-xs opacity-80">总计</div>
            </div>
            <div class="text-center">
                <div class="text-2xl font-bold">${dataset.foggy_count}</div>
                <div class="text-xs opacity-80">有雾</div>
            </div>
            <div class="text-center">
                <div class="text-2xl font-bold">${dataset.clear_count}</div>
                <div class="text-xs opacity-80">无雾</div>
            </div>
            <div class="text-center">
                <div class="text-2xl font-bold">${dataset.annotated_count}</div>
                <div class="text-xs opacity-80">标注</div>
            </div>
        </div>
    `;
}

// 加载图片列表
async function loadImages(container, append = false) {
  if (isLoading) return;

  isLoading = true;
  const loadingEl = container.querySelector("#imagesLoading");
  const emptyEl = container.querySelector("#imagesEmpty");
  const gridEl = container.querySelector("#imagesGrid");

  loadingEl.classList.remove("hidden");

  if (!append) {
    gridEl.innerHTML = "";
    emptyEl.classList.add("hidden");
  }

  try {
    const result = await fetchDatasetImages(
      currentDatasetId,
      currentPage,
      currentImageType,
      searchKeyword
    );

    if (result.code === 0) {
      const newImages = result.data.list;

      if (newImages.length === 0 && currentPage === 1) {
        emptyEl.classList.remove("hidden");
      } else {
        if (append) {
          images = [...images, ...newImages];
        } else {
          images = newImages;
        }

        renderImages(container, newImages, append);
        hasMore = result.data.page < result.data.total_pages;
      }
    }
  } catch (error) {
    console.error("加载图片失败:", error);
    showToast("加载失败，请重试");
  } finally {
    isLoading = false;
    loadingEl.classList.add("hidden");
  }
}

// 渲染图片网格
function renderImages(container, images, append = false) {
  const gridEl = container.querySelector("#imagesGrid");

  const html = images
    .map((image) => {
      const typeLabels = {
        foggy: "有雾",
        clear: "无雾",
        annotated: "标注",
      };
      const typeColors = {
        foggy: "bg-gray-500",
        clear: "bg-blue-500",
        annotated: "bg-green-500",
      };

      return `
            <div class="image-card relative bg-white rounded-lg shadow-md overflow-hidden cursor-pointer hover:shadow-xl transition-all" data-id="${
              image.id
            }">
                <div class="relative aspect-square">
                    <img src="${image.image_url}" alt="${
        image.filename
      }" class="w-full h-full object-cover">
                    <span class="absolute top-2 right-2 ${
                      typeColors[image.image_type]
                    } text-white text-xs px-2 py-1 rounded-full">
                        ${typeLabels[image.image_type]}
                    </span>
                </div>
                <div class="p-2">
                    <p class="text-xs text-gray-600 truncate">${
                      image.filename
                    }</p>
                </div>
            </div>
        `;
    })
    .join("");

  if (append) {
    gridEl.insertAdjacentHTML("beforeend", html);
  } else {
    gridEl.innerHTML = html;
  }

  // 绑定点击事件
  gridEl.querySelectorAll(".image-card").forEach((card) => {
    card.addEventListener("click", () => {
      const imageId = parseInt(card.dataset.id);
      showImageViewer(container, imageId);
    });
  });
}

// 显示图片查看器
function showImageViewer(container, imageId) {
  const imageIndex = images.findIndex((img) => img.id === imageId);
  if (imageIndex === -1) return;

  const image = images[imageIndex];
  const typeLabels = {
    foggy: "有雾图像",
    clear: "无雾图像",
    annotated: "标注图像",
  };

  // 创建模态框
  const modal = document.createElement("div");
  modal.className =
    "fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center p-4";
  modal.innerHTML = `
        <div class="relative max-w-4xl w-full">
            <button class="absolute top-4 right-4 text-white text-2xl hover:text-gray-300 z-10">
                <i class="fas fa-times"></i>
            </button>
            <img src="${image.image_url}" alt="${
    image.filename
  }" class="w-full h-auto rounded-lg">
            <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-6 text-white">
                <p class="font-semibold mb-1">${image.filename}</p>
                <div class="text-sm opacity-90 space-y-1">
                    <p>类型: ${typeLabels[image.image_type]}</p>
                    <p>尺寸: ${image.width} × ${image.height}</p>
                    <p>大小: ${formatFileSize(image.file_size)}</p>
                    ${image.tags ? `<p>标签: ${image.tags}</p>` : ""}
                </div>
            </div>
        </div>
    `;

  document.body.appendChild(modal);

  // 关闭模态框
  modal.addEventListener("click", (e) => {
    if (e.target === modal || e.target.closest("button")) {
      modal.remove();
    }
  });
}

// 显示列表视图
function showListView(container) {
  currentView = "list";
  currentDatasetId = null;
  currentImageType = "all";
  currentPage = 1;
  images = [];

  container.querySelector("#datasetListView").classList.remove("hidden");
  container.querySelector("#datasetDetailView").classList.add("hidden");

  // 重置类型筛选
  container.querySelectorAll(".type-filter-btn").forEach((btn) => {
    btn.classList.remove("active");
  });
  container.querySelector('[data-type="all"]').classList.add("active");
}
