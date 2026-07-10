// 指标展示模块
import { createUploadArea, calculateMetrics, showToast } from "../main.js";

let beforeImage = null;
let afterImage = null;
let metricsData = null;

export function initMetrics(container) {
  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-chart-bar text-red-500"></i>
                指标评估
            </h2>
            <p class="text-gray-600 text-sm mb-4">定量评估去雾效果，提供多维度评价指标</p>
          
            <!-- 图片上传 -->
            <div id="uploadSection">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">去雾前图片</label>
                        <div id="beforeUpload"></div>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">去雾后图片</label>
                        <div id="afterUpload"></div>
                    </div>
                </div>
            </div>
          
            <!-- 指标展示区域 -->
            <div id="metricsContainer" class="hidden">
                <!-- 操作按钮 -->
                <div class="flex justify-between items-center mb-4">
                    <h3 class="font-semibold text-gray-800">评估结果</h3>
                    <div class="flex space-x-2">
                        <button id="exportBtn" class="btn btn-secondary btn-sm">
                            <i class="fas fa-download"></i> 导出
                        </button>
                        <button id="resetBtn" class="btn btn-secondary btn-sm">
                            <i class="fas fa-redo"></i> 重新评估
                        </button>
                    </div>
                </div>
              
                <!-- 指标卡片 -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <!-- SSIM -->
                    <div class="metric-card" style="background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);">
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="metric-label">结构相似性</div>
                                <div class="metric-value">
                                    <span id="ssimValue">-</span>
                                </div>
                                <div class="text-xs opacity-80">SSIM (0-1, 越高越好)</div>
                            </div>
                            <i class="fas fa-image text-4xl opacity-30"></i>
                        </div>
                    </div>
                  
                    <!-- PSNR -->
                    <div class="metric-card" style="background: linear-gradient(135deg, #F093FB 0%, #F5576C 100%);">
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="metric-label">峰值信噪比</div>
                                <div class="metric-value">
                                    <span id="psnrValue">-</span>
                                    <span class="metric-unit">dB</span>
                                </div>
                                <div class="text-xs opacity-80">PSNR (越高越好)</div>
                            </div>
                            <i class="fas fa-signal text-4xl opacity-30"></i>
                        </div>
                    </div>
                  
                    <!-- 信息熵 -->
                    <div class="metric-card" style="background: linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%);">
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="metric-label">信息熵</div>
                                <div class="metric-value">
                                    <span id="entropyValue">-</span>
                                </div>
                                <div class="text-xs opacity-80">Entropy (信息量)</div>
                            </div>
                            <i class="fas fa-database text-4xl opacity-30"></i>
                        </div>
                    </div>
                  
                    <!-- 平均梯度 -->
                    <div class="metric-card" style="background: linear-gradient(135deg, #43E97B 0%, #38F9D7 100%);">
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="metric-label">平均梯度</div>
                                <div class="metric-value">
                                    <span id="gradientValue">-</span>
                                </div>
                                <div class="text-xs opacity-80">Gradient (清晰度)</div>
                            </div>
                            <i class="fas fa-chart-line text-4xl opacity-30"></i>
                        </div>
                    </div>
                </div>
              
                <!-- 运行时间 -->
                <div class="bg-gradient-to-r from-orange-400 to-pink-500 rounded-xl p-5 text-white mb-6">
                    <div class="flex items-center justify-between">
                        <div>
                            <div class="text-sm opacity-90">算法运行时间</div>
                            <div class="text-3xl font-bold mt-1">
                                <span id="runtimeValue">-</span>
                                <span class="text-lg ml-1">ms</span>
                            </div>
                        </div>
                        <i class="fas fa-clock text-5xl opacity-30"></i>
                    </div>
                </div>
              
                <!-- 图表展示 -->
                <div class="bg-white rounded-xl p-5 border border-gray-200">
                    <h4 class="font-semibold text-gray-800 mb-4">指标对比图</h4>
                    <div id="chartContainer" class="h-64"></div>
                </div>
              
                <!-- 图片对比 -->
                <div class="mt-6">
                    <h4 class="font-semibold text-gray-800 mb-3">图片对比</h4>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <p class="text-sm text-gray-600 mb-2 text-center">去雾前</p>
                            <div class="bg-gray-100 rounded-lg overflow-hidden" style="height: 200px;">
                                <img id="beforePreview" class="w-full h-full object-contain">
                            </div>
                        </div>
                        <div>
                            <p class="text-sm text-gray-600 mb-2 text-center">去雾后</p>
                            <div class="bg-gray-100 rounded-lg overflow-hidden" style="height: 200px;">
                                <img id="afterPreview" class="w-full h-full object-contain">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
      
        <!-- 指标说明 -->
        <div class="card mt-4">
            <h3 class="font-semibold text-gray-800 mb-3">
                <i class="fas fa-question-circle text-blue-500"></i>
                指标说明
            </h3>
            <div class="space-y-3 text-sm">
                <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                        <i class="fas fa-image text-purple-600"></i>
                    </div>
                    <div>
                        <p class="font-medium text-gray-800">SSIM (结构相似性)</p>
                        <p class="text-gray-600">衡量两张图片的结构相似程度，范围0-1，值越大表示结构保持越好</p>
                    </div>
                </div>
              
                <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-pink-100 rounded-lg flex items-center justify-center flex-shrink-0">
                        <i class="fas fa-signal text-pink-600"></i>
                    </div>
                    <div>
                        <p class="font-medium text-gray-800">PSNR (峰值信噪比)</p>
                        <p class="text-gray-600">衡量图像质量的客观标准，单位dB，值越大表示图像质量越好</p>
                    </div>
                </div>
              
                <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-cyan-100 rounded-lg flex items-center justify-center flex-shrink-0">
                        <i class="fas fa-database text-cyan-600"></i>
                    </div>
                    <div>
                        <p class="font-medium text-gray-800">信息熵 (Entropy)</p>
                        <p class="text-gray-600">反映图像的信息量，值越大表示图像包含的信息越丰富</p>
                    </div>
                </div>
              
                <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center flex-shrink-0">
                        <i class="fas fa-chart-line text-green-600"></i>
                    </div>
                    <div>
                        <p class="font-medium text-gray-800">平均梯度 (Gradient)</p>
                        <p class="text-gray-600">反映图像的清晰度，值越大表示图像边缘越清晰</p>
                    </div>
                </div>
            </div>
        </div>
    `;

  // 初始化上传区域
  const beforeUpload = container.querySelector("#beforeUpload");
  const afterUpload = container.querySelector("#afterUpload");

  createUploadArea(
    beforeUpload,
    (images) => {
      beforeImage = images[0];
      checkAndCalculate();
    },
    { text: "上传去雾前图片" }
  );

  createUploadArea(
    afterUpload,
    (images) => {
      afterImage = images[0];
      checkAndCalculate();
    },
    { text: "上传去雾后图片" }
  );

  // 重置按钮
  container.querySelector("#resetBtn").addEventListener("click", () => {
    beforeImage = null;
    afterImage = null;
    metricsData = null;
    container.querySelector("#uploadSection").classList.remove("hidden");
    container.querySelector("#metricsContainer").classList.add("hidden");
  });

  // 导出按钮
  container
    .querySelector("#exportBtn")
    .addEventListener("click", exportMetrics);

  function checkAndCalculate() {
    if (beforeImage && afterImage) {
      calculateAndDisplay();
    }
  }

  function calculateAndDisplay() {
    showToast("正在计算指标...");

    // 模拟计算延迟
    setTimeout(() => {
      metricsData = calculateMetrics(beforeImage, afterImage);
      displayMetrics();
      showToast("指标计算完成");
    }, 1000);
  }

  function displayMetrics() {
    container.querySelector("#uploadSection").classList.add("hidden");
    container.querySelector("#metricsContainer").classList.remove("hidden");

    // 显示指标值
    container.querySelector("#ssimValue").textContent = metricsData.ssim;
    container.querySelector("#psnrValue").textContent = metricsData.psnr;
    container.querySelector("#entropyValue").textContent = metricsData.entropy;
    container.querySelector("#gradientValue").textContent =
      metricsData.gradient;
    container.querySelector("#runtimeValue").textContent = metricsData.runtime;

    // 显示图片预览
    container.querySelector("#beforePreview").src = beforeImage.url;
    container.querySelector("#afterPreview").src = afterImage.url;

    // 绘制图表
    drawChart();
  }

  function drawChart() {
    const chartContainer = container.querySelector("#chartContainer");
    chartContainer.innerHTML = "";

    const metrics = [
      {
        name: "SSIM",
        value: parseFloat(metricsData.ssim) * 100,
        max: 100,
        color: "#667EEA",
      },
      {
        name: "PSNR",
        value: (parseFloat(metricsData.psnr) / 40) * 100,
        max: 100,
        color: "#F5576C",
      },
      {
        name: "信息熵",
        value: (parseFloat(metricsData.entropy) / 8) * 100,
        max: 100,
        color: "#00F2FE",
      },
      {
        name: "平均梯度",
        value: (parseFloat(metricsData.gradient) / 20) * 100,
        max: 100,
        color: "#38F9D7",
      },
    ];

    const chartHTML = metrics
      .map(
        (metric) => `
            <div class="mb-4">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-sm font-medium text-gray-700">${
                      metric.name
                    }</span>
                    <span class="text-sm font-semibold text-gray-800">${metric.value.toFixed(
                      1
                    )}%</span>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div class="h-full rounded-full transition-all duration-1000" 
                         style="width: ${metric.value}%; background: ${
          metric.color
        };">
                    </div>
                </div>
            </div>
        `
      )
      .join("");

    chartContainer.innerHTML = chartHTML;
  }

  function exportMetrics() {
    if (!metricsData) {
      showToast("暂无数据可导出");
      return;
    }

    const data = {
      timestamp: new Date().toISOString(),
      metrics: metricsData,
      images: {
        before: beforeImage.file.name,
        after: afterImage.file.name,
      },
    };

    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `metrics_${Date.now()}.json`;
    a.click();

    URL.revokeObjectURL(url);
    showToast("数据已导出");
  }
}
