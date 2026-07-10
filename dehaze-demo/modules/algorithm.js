// 算法信息模块
import { showToast } from "../main.js";

// 模拟算法数据
const algorithms = [
  {
    id: "dehazenet",
    name: "DehazeNet",
    version: "v2.1.0",
    developer: "图像处理实验室",
    releaseDate: "2024-01-15",
    description: "基于深度学习的端到端去雾网络，采用多尺度特征融合技术",
    parameters: "2.3M",
    complexity: "O(n²)",
    memory: "156MB",
    formats: ["JPG", "PNG", "WebP", "BMP"],
    architecture: {
      layers: 18,
      type: "CNN",
      input: "256x256x3",
      output: "256x256x3",
    },
    features: [
      "多尺度特征提取",
      "注意力机制增强",
      "残差连接优化",
      "自适应参数调节",
    ],
    advantages: [
      "处理速度快，适合实时应用",
      "对浓雾场景效果显著",
      "保持图像细节完整",
      "内存占用低",
    ],
    limitations: [
      "对极端光照条件敏感",
      "夜间场景效果一般",
      "需要GPU加速以达到最佳性能",
    ],
    performance: {
      cpu: "180ms",
      gpu: "45ms",
      mobile: "320ms",
    },
  },
  {
    id: "aodnet",
    name: "AOD-Net",
    version: "v1.5.2",
    developer: "计算机视觉研究所",
    releaseDate: "2023-11-20",
    description: "全对一去雾网络，通过端到端学习直接生成去雾图像",
    parameters: "1.8M",
    complexity: "O(n)",
    memory: "128MB",
    formats: ["JPG", "PNG", "WebP"],
    architecture: {
      layers: 12,
      type: "CNN",
      input: "512x512x3",
      output: "512x512x3",
    },
    features: ["轻量级网络结构", "端到端训练", "无需先验知识", "快速推理"],
    advantages: [
      "模型体积小，易于部署",
      "训练收敛快",
      "适合移动端应用",
      "泛化能力强",
    ],
    limitations: [
      "对复杂场景处理能力有限",
      "色彩还原度中等",
      "需要大量训练数据",
    ],
    performance: {
      cpu: "120ms",
      gpu: "28ms",
      mobile: "210ms",
    },
  },
  {
    id: "ffa-net",
    name: "FFA-Net",
    version: "v3.0.1",
    developer: "深度学习实验室",
    releaseDate: "2024-03-10",
    description: "特征融合注意力网络，结合通道注意力和像素注意力机制",
    parameters: "4.5M",
    complexity: "O(n² log n)",
    memory: "256MB",
    formats: ["JPG", "PNG", "WebP", "TIFF"],
    architecture: {
      layers: 24,
      type: "Attention-CNN",
      input: "512x512x3",
      output: "512x512x3",
    },
    features: [
      "双重注意力机制",
      "特征金字塔融合",
      "自适应权重学习",
      "多任务联合训练",
    ],
    advantages: [
      "去雾效果最佳",
      "细节保留完整",
      "色彩还原度高",
      "适应多种雾霾程度",
    ],
    limitations: [
      "计算复杂度较高",
      "需要较大内存",
      "移动端性能受限",
      "训练时间较长",
    ],
    performance: {
      cpu: "350ms",
      gpu: "68ms",
      mobile: "580ms",
    },
  },
];

let currentAlgorithm = algorithms[0];

export function initAlgorithm(container) {
  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-brain text-indigo-500"></i>
                算法信息
            </h2>
            <p class="text-gray-600 text-sm mb-4">查看去雾算法的详细技术参数和性能特点</p>
          
            <!-- 算法选择 -->
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-2">选择算法</label>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-3" id="algorithmSelector">
                    ${algorithms
                      .map(
                        (algo, index) => `
                        <button class="algorithm-card ${
                          index === 0 ? "active" : ""
                        }" data-id="${algo.id}">
                            <div class="flex items-center justify-between mb-2">
                                <span class="font-semibold text-gray-800">${
                                  algo.name
                                }</span>
                                <span class="badge badge-blue">${
                                  algo.version
                                }</span>
                            </div>
                            <p class="text-xs text-gray-600">${
                              algo.developer
                            }</p>
                        </button>
                    `
                      )
                      .join("")}
                </div>
            </div>
        </div>
      
        <!-- 算法详情 -->
        <div class="card mt-4">
            <h3 class="font-semibold text-gray-800 mb-4">
                <i class="fas fa-info-circle text-blue-500"></i>
                基本信息
            </h3>
            <div class="grid grid-cols-2 gap-4 mb-6">
                <div class="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4">
                    <div class="text-sm text-gray-600 mb-1">算法名称</div>
                    <div class="font-semibold text-gray-800" id="algoName">-</div>
                </div>
                <div class="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-4">
                    <div class="text-sm text-gray-600 mb-1">版本号</div>
                    <div class="font-semibold text-gray-800" id="algoVersion">-</div>
                </div>
                <div class="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-4">
                    <div class="text-sm text-gray-600 mb-1">开发者</div>
                    <div class="font-semibold text-gray-800" id="algoDeveloper">-</div>
                </div>
                <div class="bg-gradient-to-br from-orange-50 to-red-50 rounded-xl p-4">
                    <div class="text-sm text-gray-600 mb-1">发布日期</div>
                    <div class="font-semibold text-gray-800" id="algoDate">-</div>
                </div>
            </div>
          
            <div class="bg-gray-50 rounded-xl p-4 mb-4">
                <div class="text-sm font-medium text-gray-700 mb-2">算法描述</div>
                <p class="text-gray-600" id="algoDescription">-</p>
            </div>
        </div>
      
        <!-- 技术参数 -->
        <div class="card mt-4">
            <h3 class="font-semibold text-gray-800 mb-4">
                <i class="fas fa-cog text-purple-500"></i>
                技术参数
            </h3>
            <div class="space-y-3">
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div class="flex items-center space-x-3">
                        <i class="fas fa-database text-blue-500"></i>
                        <span class="text-sm font-medium text-gray-700">模型参数量</span>
                    </div>
                    <span class="font-semibold text-gray-800" id="algoParams">-</span>
                </div>
              
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div class="flex items-center space-x-3">
                        <i class="fas fa-tachometer-alt text-green-500"></i>
                        <span class="text-sm font-medium text-gray-700">计算复杂度</span>
                    </div>
                    <span class="font-semibold text-gray-800" id="algoComplexity">-</span>
                </div>
              
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div class="flex items-center space-x-3">
                        <i class="fas fa-memory text-purple-500"></i>
                        <span class="text-sm font-medium text-gray-700">内存占用</span>
                    </div>
                    <span class="font-semibold text-gray-800" id="algoMemory">-</span>
                </div>
              
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div class="flex items-center space-x-3">
                        <i class="fas fa-file-image text-orange-500"></i>
                        <span class="text-sm font-medium text-gray-700">支持格式</span>
                    </div>
                    <span class="font-semibold text-gray-800" id="algoFormats">-</span>
                </div>
            </div>
        </div>
      
        <!-- 模型架构 -->
        <div class="card mt-4">
            <h3 class="font-semibold text-gray-800 mb-4">
                <i class="fas fa-project-diagram text-green-500"></i>
                模型架构
            </h3>
            <div class="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-6">
                <div class="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <div class="text-sm text-gray-600 mb-1">网络层数</div>
                        <div class="text-2xl font-bold text-indigo-600" id="algoLayers">-</div>
                    </div>
                    <div>
                        <div class="text-sm text-gray-600 mb-1">网络类型</div>
                        <div class="text-2xl font-bold text-purple-600" id="algoType">-</div>
                    </div>
                </div>
              
                <div class="flex items-center justify-center space-x-4 text-sm">
                    <div class="bg-white rounded-lg px-4 py-2 shadow">
                        <div class="text-gray-600 mb-1">输入</div>
                        <div class="font-semibold text-gray-800" id="algoInput">-</div>
                    </div>
                    <i class="fas fa-arrow-right text-gray-400 text-xl"></i>
                    <div class="bg-indigo-500 text-white rounded-lg px-4 py-2 shadow">
                        <div class="opacity-90 mb-1">网络处理</div>
                        <div class="font-semibold">Deep Learning</div>
                    </div>
                    <i class="fas fa-arrow-right text-gray-400 text-xl"></i>
                    <div class="bg-white rounded-lg px-4 py-2 shadow">
                        <div class="text-gray-600 mb-1">输出</div>
                        <div class="font-semibold text-gray-800" id="algoOutput">-</div>
                    </div>
                </div>
            </div>
          
            <!-- 关键特性 -->
            <div class="mt-4">
                <h4 class="text-sm font-medium text-gray-700 mb-2">关键特性</h4>
                <div class="grid grid-cols-2 gap-2" id="algoFeatures"></div>
            </div>
        </div>
      
        <!-- 性能特点 -->
        <div class="card mt-4">
            <h3 class="font-semibold text-gray-800 mb-4">
                <i class="fas fa-chart-line text-orange-500"></i>
                性能特点
            </h3>
          
            <!-- 运行性能 -->
            <div class="mb-4">
                <h4 class="text-sm font-medium text-gray-700 mb-3">运行性能</h4>
                <div class="space-y-3">
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-sm text-gray-600">
                                <i class="fas fa-microchip text-blue-500"></i> CPU
                            </span>
                            <span class="text-sm font-semibold text-gray-800" id="perfCPU">-</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div id="perfCPUBar" class="bg-blue-500 h-2 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                  
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-sm text-gray-600">
                                <i class="fas fa-bolt text-green-500"></i> GPU
                            </span>
                            <span class="text-sm font-semibold text-gray-800" id="perfGPU">-</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div id="perfGPUBar" class="bg-green-500 h-2 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                  
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-sm text-gray-600">
                                <i class="fas fa-mobile-alt text-purple-500"></i> Mobile
                            </span>
                            <span class="text-sm font-semibold text-gray-800" id="perfMobile">-</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div id="perfMobileBar" class="bg-purple-500 h-2 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
          
            <!-- 优势 -->
            <div class="mb-4">
                <h4 class="text-sm font-medium text-gray-700 mb-2">
                    <i class="fas fa-check-circle text-green-500"></i> 优势特点
                </h4>
                <div id="algoAdvantages" class="space-y-2"></div>
            </div>
          
            <!-- 局限性 -->
            <div>
                <h4 class="text-sm font-medium text-gray-700 mb-2">
                    <i class="fas fa-exclamation-triangle text-orange-500"></i> 局限性
                </h4>
                <div id="algoLimitations" class="space-y-2"></div>
            </div>
        </div>
    `;

  // 算法选择事件
  container.querySelectorAll(".algorithm-card").forEach((card) => {
    card.addEventListener("click", () => {
      const algoId = card.dataset.id;
      currentAlgorithm = algorithms.find((a) => a.id === algoId);

      // 更新选中状态
      container.querySelectorAll(".algorithm-card").forEach((c) => {
        c.classList.remove("active");
      });
      card.classList.add("active");

      // 显示算法信息
      displayAlgorithmInfo();
    });
  });

  // 初始显示第一个算法
  displayAlgorithmInfo();

  function displayAlgorithmInfo() {
    const algo = currentAlgorithm;

    // 基本信息
    container.querySelector("#algoName").textContent = algo.name;
    container.querySelector("#algoVersion").textContent = algo.version;
    container.querySelector("#algoDeveloper").textContent = algo.developer;
    container.querySelector("#algoDate").textContent = algo.releaseDate;
    container.querySelector("#algoDescription").textContent = algo.description;

    // 技术参数
    container.querySelector("#algoParams").textContent = algo.parameters;
    container.querySelector("#algoComplexity").textContent = algo.complexity;
    container.querySelector("#algoMemory").textContent = algo.memory;
    container.querySelector("#algoFormats").textContent =
      algo.formats.join(", ");

    // 模型架构
    container.querySelector("#algoLayers").textContent =
      algo.architecture.layers;
    container.querySelector("#algoType").textContent = algo.architecture.type;
    container.querySelector("#algoInput").textContent = algo.architecture.input;
    container.querySelector("#algoOutput").textContent =
      algo.architecture.output;

    // 关键特性
    const featuresHTML = algo.features
      .map(
        (feature) => `
            <div class="bg-white border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700">
                <i class="fas fa-star text-yellow-500 mr-1"></i> ${feature}
            </div>
        `
      )
      .join("");
    container.querySelector("#algoFeatures").innerHTML = featuresHTML;

    // 性能数据
    const maxTime = 600; // 最大时间用于计算百分比
    container.querySelector("#perfCPU").textContent = algo.performance.cpu;
    container.querySelector("#perfGPU").textContent = algo.performance.gpu;
    container.querySelector("#perfMobile").textContent =
      algo.performance.mobile;

    setTimeout(() => {
      container.querySelector("#perfCPUBar").style.width =
        100 - (parseInt(algo.performance.cpu) / maxTime) * 100 + "%";
      container.querySelector("#perfGPUBar").style.width =
        100 - (parseInt(algo.performance.gpu) / maxTime) * 100 + "%";
      container.querySelector("#perfMobileBar").style.width =
        100 - (parseInt(algo.performance.mobile) / maxTime) * 100 + "%";
    }, 100);

    // 优势
    const advantagesHTML = algo.advantages
      .map(
        (adv) => `
            <div class="flex items-start space-x-2 text-sm text-gray-700">
                <i class="fas fa-check text-green-500 mt-1"></i>
                <span>${adv}</span>
            </div>
        `
      )
      .join("");
    container.querySelector("#algoAdvantages").innerHTML = advantagesHTML;

    // 局限性
    const limitationsHTML = algo.limitations
      .map(
        (lim) => `
            <div class="flex items-start space-x-2 text-sm text-gray-700">
                <i class="fas fa-minus text-orange-500 mt-1"></i>
                <span>${lim}</span>
            </div>
        `
      )
      .join("");
    container.querySelector("#algoLimitations").innerHTML = limitationsHTML;
  }
}

// 添加算法卡片样式
const style = document.createElement("style");
style.textContent = `
    .algorithm-card {
        padding: 16px;
        border-radius: 12px;
        background: white;
        border: 2px solid #E5E7EB;
        cursor: pointer;
        transition: all 0.2s;
        text-align: left;
    }
  
    .algorithm-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }
  
    .algorithm-card.active {
        border-color: #6366F1;
        background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }
`;
document.head.appendChild(style);
