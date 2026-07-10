// 算法选择模块
import { showToast } from "../main.js";

// 算法数据库
const algorithms = {
  traditional: {
    name: "传统算法",
    icon: "fas fa-cogs",
    algorithms: [
      {
        id: "dcp",
        name: "暗通道先验",
        nameEn: "Dark Channel Prior (DCP)",
        type: "traditional",
        rating: 4,
        speed: "medium",
        description: "基于暗通道先验理论的经典去雾算法",
        params: { omega: 0.95, t0: 0.1, radius: 15 },
        performance: { psnr: 28.5, ssim: 0.89, time: 120 },
        scenarios: ["轻度雾霾", "中度雾霾"],
        author: "Kaiming He et al.",
        year: 2011,
        paper: "https://ieeexplore.ieee.org/document/5567108",
        favorite: false,
        usageCount: 1250,
      },
      {
        id: "cap",
        name: "色彩衰减先验",
        nameEn: "Color Attenuation Prior (CAP)",
        type: "traditional",
        rating: 4,
        speed: "fast",
        description: "利用色彩衰减特性进行去雾处理",
        params: { lambda: 0.5, sigma: 0.8 },
        performance: { psnr: 27.8, ssim: 0.87, time: 80 },
        scenarios: ["轻度雾霾", "户外场景"],
        author: "Qingsong Zhu et al.",
        year: 2015,
        favorite: false,
        usageCount: 890,
      },
      {
        id: "bcr",
        name: "边界约束与上下文正则化",
        nameEn: "Boundary Constraint and Context Regularization",
        type: "traditional",
        rating: 3,
        speed: "slow",
        description: "结合边界约束和上下文信息的去雾方法",
        params: { alpha: 0.7, beta: 0.3 },
        performance: { psnr: 26.5, ssim: 0.85, time: 200 },
        scenarios: ["中度雾霾"],
        author: "Gaofeng Meng et al.",
        year: 2013,
        favorite: false,
        usageCount: 450,
      },
    ],
  },
  deeplearning: {
    name: "深度学习算法",
    icon: "fas fa-brain",
    children: {
      cnn: {
        name: "CNN系列",
        algorithms: [
          {
            id: "dehazenet",
            name: "DehazeNet",
            nameEn: "DehazeNet",
            type: "cnn",
            rating: 5,
            speed: "fast",
            description: "端到端的深度卷积神经网络去雾算法",
            params: { layers: 16, filters: 64 },
            performance: { psnr: 32.5, ssim: 0.94, time: 50 },
            scenarios: ["轻度雾霾", "中度雾霾", "重度雾霾"],
            author: "Bolun Cai et al.",
            year: 2016,
            favorite: false,
            usageCount: 2100,
          },
          {
            id: "aodnet",
            name: "AOD-Net",
            nameEn: "All-in-One Dehazing Network",
            type: "cnn",
            rating: 5,
            speed: "fast",
            description: "一体化去雾网络，直接生成去雾图像",
            params: { depth: 20, channels: 128 },
            performance: { psnr: 33.2, ssim: 0.95, time: 45 },
            scenarios: ["所有场景"],
            author: "Boyi Li et al.",
            year: 2017,
            favorite: false,
            usageCount: 2500,
          },
          {
            id: "griddehazenet",
            name: "GridDehazeNet",
            nameEn: "GridDehazeNet",
            type: "cnn",
            rating: 5,
            speed: "medium",
            description: "基于网格结构的注意力去雾网络",
            params: { grid_size: 3, attention: true },
            performance: { psnr: 34.8, ssim: 0.96, time: 80 },
            scenarios: ["中度雾霾", "重度雾霾"],
            author: "Xiaohong Liu et al.",
            year: 2019,
            favorite: false,
            usageCount: 1800,
          },
        ],
      },
      gan: {
        name: "GAN系列",
        algorithms: [
          {
            id: "cyclegan",
            name: "CycleGAN",
            nameEn: "Cycle-Consistent GAN",
            type: "gan",
            rating: 4,
            speed: "slow",
            description: "循环一致性生成对抗网络去雾",
            params: { epochs: 200, lr: 0.0002 },
            performance: { psnr: 31.5, ssim: 0.92, time: 150 },
            scenarios: ["特殊场景", "艺术风格"],
            author: "Jun-Yan Zhu et al.",
            year: 2017,
            favorite: false,
            usageCount: 1200,
          },
          {
            id: "pix2pix",
            name: "Pix2Pix",
            nameEn: "Image-to-Image Translation",
            type: "gan",
            rating: 4,
            speed: "medium",
            description: "图像到图像转换的条件GAN",
            params: { lambda: 100, gan_mode: "vanilla" },
            performance: { psnr: 30.8, ssim: 0.91, time: 100 },
            scenarios: ["配对数据训练"],
            author: "Phillip Isola et al.",
            year: 2017,
            favorite: false,
            usageCount: 950,
          },
        ],
      },
      transformer: {
        name: "Transformer系列",
        algorithms: [
          {
            id: "dehazeformer",
            name: "DehazeFormer",
            nameEn: "DehazeFormer",
            type: "transformer",
            rating: 5,
            speed: "medium",
            description: "基于Transformer的去雾网络",
            params: { heads: 8, layers: 12 },
            performance: { psnr: 35.5, ssim: 0.97, time: 90 },
            scenarios: ["所有场景"],
            author: "Song et al.",
            year: 2022,
            favorite: false,
            usageCount: 1500,
          },
          {
            id: "vit",
            name: "Vision Transformer",
            nameEn: "Vision Transformer for Dehazing",
            type: "transformer",
            rating: 5,
            speed: "slow",
            description: "视觉Transformer应用于图像去雾",
            params: { patch_size: 16, embed_dim: 768 },
            performance: { psnr: 36.2, ssim: 0.98, time: 120 },
            scenarios: ["高质量要求"],
            author: "Dosovitskiy et al.",
            year: 2021,
            favorite: false,
            usageCount: 1100,
          },
        ],
      },
    },
  },
  hybrid: {
    name: "混合算法",
    icon: "fas fa-layer-group",
    algorithms: [
      {
        id: "hybrid1",
        name: "传统+深度学习融合",
        nameEn: "Traditional-DL Hybrid",
        type: "hybrid",
        rating: 5,
        speed: "medium",
        description: "结合传统方法和深度学习的优势",
        params: { weight_traditional: 0.3, weight_dl: 0.7 },
        performance: { psnr: 34.5, ssim: 0.96, time: 100 },
        scenarios: ["所有场景"],
        author: "Research Team",
        year: 2023,
        favorite: false,
        usageCount: 800,
      },
      {
        id: "ensemble",
        name: "多模型融合",
        nameEn: "Ensemble Model",
        type: "hybrid",
        rating: 5,
        speed: "slow",
        description: "集成多个模型的预测结果",
        params: { models: 3, fusion_method: "weighted" },
        performance: { psnr: 35.8, ssim: 0.97, time: 180 },
        scenarios: ["高质量要求"],
        author: "Research Team",
        year: 2023,
        favorite: false,
        usageCount: 600,
      },
    ],
  },
};

// 获取所有算法的扁平列表
function getAllAlgorithms() {
  const allAlgs = [];

  // 传统算法
  allAlgs.push(...algorithms.traditional.algorithms);

  // 深度学习算法
  Object.values(algorithms.deeplearning.children).forEach((category) => {
    allAlgs.push(...category.algorithms);
  });

  // 混合算法
  allAlgs.push(...algorithms.hybrid.algorithms);

  return allAlgs;
}

export function initAlgorithmSelect(container) {
  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-brain text-indigo-500"></i>
                算法选择
            </h2>
            <p class="text-gray-600 text-sm mb-4">选择合适的去雾算法处理图片</p>
          
            <!-- 智能推荐 -->
            <div id="recommendSection" class="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-4 mb-4 text-white">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="font-bold">
                        <i class="fas fa-magic"></i> 智能推荐
                    </h3>
                    <span class="text-xs bg-white bg-opacity-20 px-2 py-1 rounded-full">AI推荐</span>
                </div>
                <p class="text-sm opacity-90 mb-3">根据图像特征为您推荐最佳算法</p>
                <div id="recommendedAlgorithm" class="bg-white bg-opacity-10 rounded-lg p-3">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="font-medium">AOD-Net</p>
                            <p class="text-xs opacity-75">推荐理由：适合中度雾霾，处理速度快</p>
                        </div>
                        <button class="bg-white text-indigo-600 px-4 py-2 rounded-lg font-medium hover:bg-opacity-90 transition-colors" id="useRecommendedBtn">
                            使用
                        </button>
                    </div>
                </div>
            </div>
          
            <!-- 搜索和筛选 -->
            <div class="mb-4">
                <div class="relative mb-3">
                    <input type="text" id="searchInput" placeholder="搜索算法名称或关键词..." 
                           class="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    <i class="fas fa-search absolute left-3 top-3 text-gray-400"></i>
                </div>
              
                <div class="flex items-center space-x-2 overflow-x-auto pb-2">
                    <button class="filter-btn active" data-filter="all">
                        <i class="fas fa-th"></i> 全部
                    </button>
                    <button class="filter-btn" data-filter="traditional">
                        <i class="fas fa-cogs"></i> 传统
                    </button>
                    <button class="filter-btn" data-filter="cnn">
                        <i class="fas fa-network-wired"></i> CNN
                    </button>
                    <button class="filter-btn" data-filter="gan">
                        <i class="fas fa-random"></i> GAN
                    </button>
                    <button class="filter-btn" data-filter="transformer">
                        <i class="fas fa-cube"></i> Transformer
                    </button>
                    <button class="filter-btn" data-filter="favorite">
                        <i class="fas fa-star"></i> 收藏
                    </button>
                </div>
              
                <div class="flex items-center space-x-2 mt-2">
                    <select id="sortSelect" class="text-sm border border-gray-300 rounded-lg px-3 py-1">
                        <option value="recommended">推荐度排序</option>
                        <option value="speed">速度排序</option>
                        <option value="usage">使用次数</option>
                        <option value="name">名称排序</option>
                    </select>
                  
                    <select id="speedFilter" class="text-sm border border-gray-300 rounded-lg px-3 py-1">
                        <option value="all">所有速度</option>
                        <option value="fast">快速</option>
                        <option value="medium">中等</option>
                        <option value="slow">较慢</option>
                    </select>
                </div>
            </div>
          
            <!-- 算法树形列表 -->
            <div id="algorithmTree" class="space-y-3"></div>
        </div>
      
        <!-- 算法详情模态框 -->
        <div id="algorithmModal" class="modal hidden">
            <div class="modal-content max-w-2xl">
                <div class="modal-header">
                    <h3 class="text-lg font-bold">算法详情</h3>
                    <button id="closeModalBtn" class="text-gray-500 hover:text-gray-700">
                        <i class="fas fa-times text-xl"></i>
                    </button>
                </div>
                <div class="modal-body" id="algorithmDetails"></div>
            </div>
        </div>
    `;

  let currentFilter = "all";
  let currentSort = "recommended";
  let searchQuery = "";
  let speedFilter = "all";

  // 渲染算法树
  renderAlgorithmTree();

  // 搜索功能
  const searchInput = container.querySelector("#searchInput");
  searchInput.addEventListener("input", (e) => {
    searchQuery = e.target.value.toLowerCase();
    renderAlgorithmTree();
  });

  // 筛选按钮
  container.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentFilter = btn.dataset.filter;

      container.querySelectorAll(".filter-btn").forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");

      renderAlgorithmTree();
    });
  });

  // 排序选择
  container.querySelector("#sortSelect").addEventListener("change", (e) => {
    currentSort = e.target.value;
    renderAlgorithmTree();
  });

  // 速度筛选
  container.querySelector("#speedFilter").addEventListener("change", (e) => {
    speedFilter = e.target.value;
    renderAlgorithmTree();
  });

  // 使用推荐算法
  container
    .querySelector("#useRecommendedBtn")
    .addEventListener("click", () => {
      const aodnet = getAllAlgorithms().find((alg) => alg.id === "aodnet");
      selectAlgorithm(aodnet);
    });

  // 关闭模态框
  container.querySelector("#closeModalBtn").addEventListener("click", () => {
    container.querySelector("#algorithmModal").classList.add("hidden");
  });

  function renderAlgorithmTree() {
    const treeContainer = container.querySelector("#algorithmTree");
    treeContainer.innerHTML = "";

    const allAlgs = getAllAlgorithms();
    let filteredAlgs = allAlgs;

    // 应用筛选
    if (currentFilter !== "all") {
      if (currentFilter === "favorite") {
        filteredAlgs = allAlgs.filter((alg) => alg.favorite);
      } else {
        filteredAlgs = allAlgs.filter((alg) => alg.type === currentFilter);
      }
    }

    // 应用搜索
    if (searchQuery) {
      filteredAlgs = filteredAlgs.filter(
        (alg) =>
          alg.name.toLowerCase().includes(searchQuery) ||
          alg.nameEn.toLowerCase().includes(searchQuery) ||
          alg.description.toLowerCase().includes(searchQuery)
      );
    }

    // 应用速度筛选
    if (speedFilter !== "all") {
      filteredAlgs = filteredAlgs.filter((alg) => alg.speed === speedFilter);
    }

    // 排序
    filteredAlgs = sortAlgorithms(filteredAlgs, currentSort);

    if (filteredAlgs.length === 0) {
      treeContainer.innerHTML = `
                <div class="text-center py-8 text-gray-400">
                    <i class="fas fa-search text-4xl mb-2"></i>
                    <p>未找到匹配的算法</p>
                </div>
            `;
      return;
    }

    // 按类型分组显示
    const grouped = groupByType(filteredAlgs);

    Object.entries(grouped).forEach(([type, algs]) => {
      const categoryDiv = document.createElement("div");
      categoryDiv.className = "algorithm-category";

      const categoryHeader = document.createElement("div");
      categoryHeader.className = "category-header";
      categoryHeader.innerHTML = `
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100">
                    <div class="flex items-center space-x-2">
                        <i class="${getTypeIcon(type)} text-blue-500"></i>
                        <span class="font-medium text-gray-800">${getTypeName(
                          type
                        )}</span>
                        <span class="text-xs text-gray-500">(${
                          algs.length
                        })</span>
                    </div>
                    <i class="fas fa-chevron-down text-gray-400"></i>
                </div>
            `;

      const categoryContent = document.createElement("div");
      categoryContent.className = "category-content mt-2 space-y-2";

      algs.forEach((alg) => {
        const algCard = createAlgorithmCard(alg);
        categoryContent.appendChild(algCard);
      });

      categoryDiv.appendChild(categoryHeader);
      categoryDiv.appendChild(categoryContent);
      treeContainer.appendChild(categoryDiv);

      // 折叠/展开功能
      categoryHeader.addEventListener("click", () => {
        categoryContent.classList.toggle("hidden");
        const icon = categoryHeader.querySelector(".fa-chevron-down");
        icon.classList.toggle("fa-chevron-down");
        icon.classList.toggle("fa-chevron-up");
      });
    });
  }

  function createAlgorithmCard(alg) {
    const card = document.createElement("div");
    card.className =
      "algorithm-card bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow";
    card.innerHTML = `
            <div class="flex items-start justify-between mb-2">
                <div class="flex-1">
                    <div class="flex items-center space-x-2 mb-1">
                        <h4 class="font-bold text-gray-800">${alg.name}</h4>
                        <button class="favorite-btn ${
                          alg.favorite ? "text-yellow-500" : "text-gray-300"
                        }" data-id="${alg.id}">
                            <i class="fas fa-star"></i>
                        </button>
                    </div>
                    <p class="text-xs text-gray-500 mb-2">${alg.nameEn}</p>
                    <p class="text-sm text-gray-600">${alg.description}</p>
                </div>
            </div>
          
            <div class="flex items-center space-x-3 mb-3 text-xs">
                <div class="flex items-center space-x-1">
                    ${generateStars(alg.rating)}
                </div>
                <span class="px-2 py-1 rounded-full ${getSpeedClass(
                  alg.speed
                )}">
                    ${getSpeedText(alg.speed)}
                </span>
                <span class="text-gray-500">
                    <i class="fas fa-users"></i> ${alg.usageCount}
                </span>
            </div>
          
            <div class="flex items-center space-x-2">
                <button class="btn btn-primary btn-sm flex-1 select-alg-btn" data-id="${
                  alg.id
                }">
                    <i class="fas fa-check"></i> 选择
                </button>
                <button class="btn btn-secondary btn-sm details-btn" data-id="${
                  alg.id
                }">
                    <i class="fas fa-info-circle"></i> 详情
                </button>
            </div>
        `;

    // 收藏按钮
    card.querySelector(".favorite-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      toggleFavorite(alg.id);
    });

    // 选择按钮
    card.querySelector(".select-alg-btn").addEventListener("click", () => {
      selectAlgorithm(alg);
    });

    // 详情按钮
    card.querySelector(".details-btn").addEventListener("click", () => {
      showAlgorithmDetails(alg);
    });

    return card;
  }

  function generateStars(rating) {
    let stars = "";
    for (let i = 0; i < 5; i++) {
      if (i < rating) {
        stars += '<i class="fas fa-star text-yellow-500"></i>';
      } else {
        stars += '<i class="far fa-star text-gray-300"></i>';
      }
    }
    return stars;
  }

  function getSpeedClass(speed) {
    const classes = {
      fast: "bg-green-100 text-green-700",
      medium: "bg-yellow-100 text-yellow-700",
      slow: "bg-red-100 text-red-700",
    };
    return classes[speed] || "bg-gray-100 text-gray-700";
  }

  function getSpeedText(speed) {
    const texts = {
      fast: "快速",
      medium: "中等",
      slow: "较慢",
    };
    return texts[speed] || speed;
  }

  function getTypeIcon(type) {
    const icons = {
      traditional: "fas fa-cogs",
      cnn: "fas fa-network-wired",
      gan: "fas fa-random",
      transformer: "fas fa-cube",
      hybrid: "fas fa-layer-group",
    };
    return icons[type] || "fas fa-brain";
  }

  function getTypeName(type) {
    const names = {
      traditional: "传统算法",
      cnn: "CNN系列",
      gan: "GAN系列",
      transformer: "Transformer系列",
      hybrid: "混合算法",
    };
    return names[type] || type;
  }

  function groupByType(algs) {
    const grouped = {};
    algs.forEach((alg) => {
      if (!grouped[alg.type]) {
        grouped[alg.type] = [];
      }
      grouped[alg.type].push(alg);
    });
    return grouped;
  }

  function sortAlgorithms(algs, sortBy) {
    const sorted = [...algs];
    switch (sortBy) {
      case "recommended":
        return sorted.sort((a, b) => b.rating - a.rating);
      case "speed":
        const speedOrder = { fast: 0, medium: 1, slow: 2 };
        return sorted.sort((a, b) => speedOrder[a.speed] - speedOrder[b.speed]);
      case "usage":
        return sorted.sort((a, b) => b.usageCount - a.usageCount);
      case "name":
        return sorted.sort((a, b) => a.name.localeCompare(b.name));
      default:
        return sorted;
    }
  }

  function toggleFavorite(algId) {
    const allAlgs = getAllAlgorithms();
    const alg = allAlgs.find((a) => a.id === algId);
    if (alg) {
      alg.favorite = !alg.favorite;
      renderAlgorithmTree();
      showToast(alg.favorite ? "已添加到收藏" : "已取消收藏");
    }
  }

  function selectAlgorithm(alg) {
    // 保存选中的算法
    window.dehazeApp = window.dehazeApp || {};
    window.dehazeApp.selectedAlgorithm = alg;

    showToast(`已选择算法：${alg.name}`);

    // 跳转到处理页面
    setTimeout(() => {
      window.location.hash = "#processing";
    }, 500);
  }

  function showAlgorithmDetails(alg) {
    const modal = container.querySelector("#algorithmModal");
    const details = container.querySelector("#algorithmDetails");

    details.innerHTML = `
            <div class="space-y-4">
                <!-- 基本信息 -->
                <div>
                    <h4 class="font-bold text-gray-800 mb-2">${alg.name}</h4>
                    <p class="text-sm text-gray-600 mb-2">${alg.nameEn}</p>
                    <p class="text-sm text-gray-700">${alg.description}</p>
                </div>
              
                <!-- 评分和速度 -->
                <div class="flex items-center space-x-4">
                    <div>
                        <span class="text-sm text-gray-600">推荐指数：</span>
                        ${generateStars(alg.rating)}
                    </div>
                    <div>
                        <span class="text-sm text-gray-600">处理速度：</span>
                        <span class="px-2 py-1 rounded-full text-xs ${getSpeedClass(
                          alg.speed
                        )}">
                            ${getSpeedText(alg.speed)}
                        </span>
                    </div>
                </div>
              
                <!-- 性能指标 -->
                <div class="bg-gray-50 rounded-lg p-4">
                    <h5 class="font-semibold text-gray-800 mb-3">性能指标</h5>
                    <div class="grid grid-cols-3 gap-4 text-center">
                        <div>
                            <p class="text-2xl font-bold text-blue-600">${
                              alg.performance.psnr
                            }</p>
                            <p class="text-xs text-gray-600">PSNR (dB)</p>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-green-600">${
                              alg.performance.ssim
                            }</p>
                            <p class="text-xs text-gray-600">SSIM</p>
                        </div>
                        <div>
                            <p class="text-2xl font-bold text-orange-600">${
                              alg.performance.time
                            }</p>
                            <p class="text-xs text-gray-600">时间 (ms)</p>
                        </div>
                    </div>
                </div>
              
                <!-- 适用场景 -->
                <div>
                    <h5 class="font-semibold text-gray-800 mb-2">适用场景</h5>
                    <div class="flex flex-wrap gap-2">
                        ${alg.scenarios
                          .map(
                            (s) => `
                            <span class="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                                ${s}
                            </span>
                        `
                          )
                          .join("")}
                    </div>
                </div>
              
                <!-- 作者信息 -->
                <div class="border-t pt-4">
                    <div class="grid grid-cols-2 gap-3 text-sm">
                        <div>
                            <span class="text-gray-600">作者：</span>
                            <span class="font-medium">${alg.author}</span>
                        </div>
                        <div>
                            <span class="text-gray-600">年份：</span>
                            <span class="font-medium">${alg.year}</span>
                        </div>
                        <div class="col-span-2">
                            <span class="text-gray-600">使用次数：</span>
                            <span class="font-medium">${alg.usageCount}</span>
                        </div>
                    </div>
                </div>
              
                <!-- 操作按钮 -->
                <div class="flex space-x-3 pt-4">
                    <button class="btn btn-primary flex-1" onclick="document.querySelector('.select-alg-btn[data-id=\\'${
                      alg.id
                    }\\']').click(); document.querySelector('#closeModalBtn').click();">
                        <i class="fas fa-check"></i> 使用此算法
                    </button>
                    <button class="btn btn-secondary" onclick="document.querySelector('.favorite-btn[data-id=\\'${
                      alg.id
                    }\\']').click();">
                        <i class="fas fa-star"></i> ${
                          alg.favorite ? "取消收藏" : "收藏"
                        }
                    </button>
                </div>
            </div>
        `;

    modal.classList.remove("hidden");
  }
}

// 导出算法数据供其他模块使用
export { algorithms, getAllAlgorithms };
