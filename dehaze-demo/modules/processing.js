// 去雾处理模块
import { showToast, showLoading, hideLoading } from "../main.js";
import { historyManager } from "./imageInput.js";

export function initProcessing(container) {
  // 获取当前图片和算法
  const app = window.dehazeApp || {};
  const currentImage = app.currentImage;
  const selectedAlgorithm = app.selectedAlgorithm;

  if (!currentImage || !selectedAlgorithm) {
    container.innerHTML = `
            <div class="card">
                <div class="text-center py-12">
                    <i class="fas fa-exclamation-circle text-6xl text-gray-300 mb-4"></i>
                    <p class="text-gray-600 mb-4">请先上传图片并选择算法</p>
                    <button class="btn btn-primary" onclick="window.location.hash='#image-input'">
                        返回图像输入
                    </button>
                </div>
            </div>
        `;
    return;
  }

  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-cog text-blue-500"></i>
                去雾处理
            </h2>
            <p class="text-gray-600 text-sm mb-4">使用 ${
              selectedAlgorithm.name
            } 算法处理图片</p>
          
            <!-- 图片预览 -->
            <div class="bg-gray-100 rounded-xl overflow-hidden mb-4" style="height: 250px;">
                <img src="${
                  currentImage.url
                }" class="w-full h-full object-contain">
            </div>
          
            <!-- 算法信息 -->
            <div class="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl p-4 mb-4 text-white">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="font-bold">${selectedAlgorithm.name}</h3>
                    <span class="px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">
                        ${selectedAlgorithm.type.toUpperCase()}
                    </span>
                </div>
                <p class="text-sm opacity-90 mb-2">${
                  selectedAlgorithm.description
                }</p>
                <div class="flex items-center space-x-4 text-xs">
                    <span><i class="fas fa-clock"></i> 预计 ${
                      selectedAlgorithm.performance.time
                    }ms</span>
                    <span><i class="fas fa-star"></i> ${
                      selectedAlgorithm.rating
                    }/5</span>
                </div>
            </div>
          
            <!-- 处理模式选择 -->
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-2">处理模式</label>
                <div class="grid grid-cols-3 gap-2">
                    <button class="mode-btn active" data-mode="conservative">
                        <i class="fas fa-shield-alt"></i>
                        <span>保守</span>
                        <span class="text-xs opacity-75">保留细节</span>
                    </button>
                    <button class="mode-btn" data-mode="standard">
                        <i class="fas fa-balance-scale"></i>
                        <span>标准</span>
                        <span class="text-xs opacity-75">推荐</span>
                    </button>
                    <button class="mode-btn" data-mode="aggressive">
                        <i class="fas fa-bolt"></i>
                        <span>激进</span>
                        <span class="text-xs opacity-75">最大去雾</span>
                    </button>
                </div>
            </div>
          
            <!-- 高级参数调节 -->
            <div class="mb-4">
                <div class="flex items-center justify-between mb-2">
                    <label class="text-sm font-medium text-gray-700">高级参数</label>
                    <button id="toggleAdvanced" class="text-sm text-blue-600 hover:text-blue-700">
                        <i class="fas fa-chevron-down"></i> 展开
                    </button>
                </div>
                <div id="advancedParams" class="hidden space-y-3 bg-gray-50 rounded-lg p-4">
                    <div>
                        <label class="text-xs text-gray-600">去雾强度</label>
                        <input type="range" class="slider" min="0" max="100" value="80" id="strengthSlider">
                        <div class="flex justify-between text-xs text-gray-500">
                            <span>弱</span>
                            <span id="strengthValue">80%</span>
                            <span>强</span>
                        </div>
                    </div>
                    <div>
                        <label class="text-xs text-gray-600">细节保留</label>
                        <input type="range" class="slider" min="0" max="100" value="70" id="detailSlider">
                        <div class="flex justify-between text-xs text-gray-500">
                            <span>低</span>
                            <span id="detailValue">70%</span>
                            <span>高</span>
                        </div>
                    </div>
                    <div>
                        <label class="text-xs text-gray-600">色彩饱和度</label>
                        <input type="range" class="slider" min="0" max="100" value="60" id="saturationSlider">
                        <div class="flex justify-between text-xs text-gray-500">
                            <span>低</span>
                            <span id="saturationValue">60%</span>
                            <span>高</span>
                        </div>
                    </div>
                </div>
            </div>
          
            <!-- 处理按钮 -->
            <button id="startProcessBtn" class="btn btn-primary w-full mb-3">
                <i class="fas fa-play"></i> 开始去雾处理
            </button>
          
            <div class="flex space-x-2">
                <button class="btn btn-secondary flex-1" onclick="window.location.hash='#algorithm-select'">
                    <i class="fas fa-arrow-left"></i> 更换算法
                </button>
                <button class="btn btn-secondary flex-1" onclick="window.location.hash='#image-input'">
                    <i class="fas fa-image"></i> 更换图片
                </button>
            </div>
          
            <!-- 处理进度 -->
            <div id="progressSection" class="hidden mt-4">
                <div class="bg-white rounded-xl shadow-lg p-4">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="font-semibold text-gray-800">处理中...</h3>
                        <button id="cancelBtn" class="text-red-500 hover:text-red-600 text-sm">
                            <i class="fas fa-times"></i> 取消
                        </button>
                    </div>
                  
                    <div class="mb-3">
                        <div class="flex justify-between text-sm text-gray-600 mb-1">
                            <span id="progressStage">初始化...</span>
                            <span id="progressPercent">0%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                            <div id="progressBar" class="bg-gradient-to-r from-blue-500 to-indigo-600 h-full rounded-full transition-all duration-300" style="width: 0%"></div>
                        </div>
                    </div>
                  
                    <div class="flex justify-between text-xs text-gray-500">
                        <span>已用时间: <span id="elapsedTime">0s</span></span>
                        <span>预计剩余: <span id="remainingTime">-</span></span>
                    </div>
                </div>
            </div>
          
            <!-- 处理结果 -->
            <div id="resultSection" class="hidden mt-4">
                <div class="bg-white rounded-xl shadow-lg p-4">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="font-semibold text-gray-800">
                            <i class="fas fa-check-circle text-green-500"></i> 处理完成
                        </h3>
                        <span class="text-sm text-gray-600">用时: <span id="totalTime">-</span></span>
                    </div>
                  
                    <div class="grid grid-cols-2 gap-3 mb-4">
                        <div>
                            <p class="text-xs text-gray-600 mb-1 text-center">原图</p>
                            <div class="bg-gray-100 rounded-lg overflow-hidden" style="height: 150px;">
                                <img src="${
                                  currentImage.url
                                }" class="w-full h-full object-contain">
                            </div>
                        </div>
                        <div>
                            <p class="text-xs text-gray-600 mb-1 text-center">去雾后</p>
                            <div class="bg-gray-100 rounded-lg overflow-hidden" style="height: 150px;">
                                <img id="resultImage" class="w-full h-full object-contain">
                            </div>
                        </div>
                    </div>
                  
                    <div class="grid grid-cols-2 gap-2 mb-3">
                        <button id="viewComparisonBtn" class="btn btn-primary">
                            <i class="fas fa-columns"></i> 查看对比
                        </button>
                        <button id="saveResultBtn" class="btn btn-secondary">
                            <i class="fas fa-download"></i> 保存结果
                        </button>
                    </div>
                  
                    <div class="flex space-x-2">
                        <button id="reprocessBtn" class="btn btn-secondary flex-1">
                            <i class="fas fa-redo"></i> 重新处理
                        </button>
                        <button id="shareBtn" class="btn btn-secondary flex-1">
                            <i class="fas fa-share-alt"></i> 分享
                        </button>
                    </div>
                </div>
            </div>
        </div>
      
        <!-- 处理提示 -->
        <div class="card mt-4">
            <div class="bg-blue-50 border border-blue-200 rounded-xl p-4">
                <div class="flex items-start space-x-3">
                    <i class="fas fa-lightbulb text-blue-500 mt-1"></i>
                    <div class="text-sm text-blue-800">
                        <p class="font-semibold mb-1">处理提示</p>
                        <ul class="space-y-1 text-blue-700">
                            <li>• 保守模式适合保留更多原图细节</li>
                            <li>• 标准模式适合大多数场景</li>
                            <li>• 激进模式适合重度雾霾图片</li>
                            <li>• 可在高级参数中微调处理效果</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    `;

  let processingMode = "standard";
  let processingParams = {
    strength: 80,
    detail: 70,
    saturation: 60,
  };
  let isProcessing = false;
  let processTimer = null;

  // 模式切换
  container.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      processingMode = btn.dataset.mode;

      container.querySelectorAll(".mode-btn").forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");

      // 根据模式调整参数
      updateParamsByMode(processingMode);
    });
  });

  // 高级参数展开/收起
  const toggleAdvanced = container.querySelector("#toggleAdvanced");
  const advancedParams = container.querySelector("#advancedParams");

  toggleAdvanced.addEventListener("click", () => {
    advancedParams.classList.toggle("hidden");
    const icon = toggleAdvanced.querySelector("i");
    icon.classList.toggle("fa-chevron-down");
    icon.classList.toggle("fa-chevron-up");
    toggleAdvanced.innerHTML = advancedParams.classList.contains("hidden")
      ? '<i class="fas fa-chevron-down"></i> 展开'
      : '<i class="fas fa-chevron-up"></i> 收起';
  });

  // 参数滑块
  const strengthSlider = container.querySelector("#strengthSlider");
  const detailSlider = container.querySelector("#detailSlider");
  const saturationSlider = container.querySelector("#saturationSlider");

  strengthSlider.addEventListener("input", (e) => {
    processingParams.strength = parseInt(e.target.value);
    container.querySelector("#strengthValue").textContent =
      processingParams.strength + "%";
  });

  detailSlider.addEventListener("input", (e) => {
    processingParams.detail = parseInt(e.target.value);
    container.querySelector("#detailValue").textContent =
      processingParams.detail + "%";
  });

  saturationSlider.addEventListener("input", (e) => {
    processingParams.saturation = parseInt(e.target.value);
    container.querySelector("#saturationValue").textContent =
      processingParams.saturation + "%";
  });

  // 开始处理
  container.querySelector("#startProcessBtn").addEventListener("click", () => {
    if (!isProcessing) {
      startProcessing();
    }
  });

  // 取消处理
  container.querySelector("#cancelBtn").addEventListener("click", () => {
    cancelProcessing();
  });

  // 查看对比
  container
    .querySelector("#viewComparisonBtn")
    .addEventListener("click", () => {
      window.location.hash = "#side-by-side";
    });

  // 保存结果
  container.querySelector("#saveResultBtn").addEventListener("click", () => {
    saveResult();
  });

  // 重新处理
  container.querySelector("#reprocessBtn").addEventListener("click", () => {
    container.querySelector("#resultSection").classList.add("hidden");
    container.querySelector("#startProcessBtn").disabled = false;
  });

  // 分享
  container.querySelector("#shareBtn").addEventListener("click", () => {
    showToast("分享功能开发中...");
  });

  function updateParamsByMode(mode) {
    const presets = {
      conservative: { strength: 60, detail: 85, saturation: 50 },
      standard: { strength: 80, detail: 70, saturation: 60 },
      aggressive: { strength: 95, detail: 50, saturation: 75 },
    };

    const preset = presets[mode];
    processingParams = { ...preset };

    strengthSlider.value = preset.strength;
    detailSlider.value = preset.detail;
    saturationSlider.value = preset.saturation;

    container.querySelector("#strengthValue").textContent =
      preset.strength + "%";
    container.querySelector("#detailValue").textContent = preset.detail + "%";
    container.querySelector("#saturationValue").textContent =
      preset.saturation + "%";
  }

  function startProcessing() {
    isProcessing = true;

    // 隐藏开始按钮，显示进度
    container.querySelector("#startProcessBtn").disabled = true;
    container.querySelector("#progressSection").classList.remove("hidden");

    // 模拟处理过程
    const stages = [
      { name: "图像预处理...", duration: 500 },
      { name: "特征提取...", duration: 800 },
      { name: "去雾计算...", duration: 1200 },
      { name: "后处理优化...", duration: 600 },
      { name: "生成结果...", duration: 400 },
    ];

    let currentStage = 0;
    let totalProgress = 0;
    let startTime = Date.now();

    const progressBar = container.querySelector("#progressBar");
    const progressStage = container.querySelector("#progressStage");
    const progressPercent = container.querySelector("#progressPercent");
    const elapsedTime = container.querySelector("#elapsedTime");
    const remainingTime = container.querySelector("#remainingTime");

    function processStage() {
      if (!isProcessing || currentStage >= stages.length) {
        if (isProcessing) {
          completeProcessing();
        }
        return;
      }

      const stage = stages[currentStage];
      progressStage.textContent = stage.name;

      const stageProgress = 100 / stages.length;
      const targetProgress = (currentStage + 1) * stageProgress;

      const interval = setInterval(() => {
        if (!isProcessing) {
          clearInterval(interval);
          return;
        }

        totalProgress += 2;
        if (totalProgress >= targetProgress) {
          totalProgress = targetProgress;
          clearInterval(interval);
          currentStage++;
          setTimeout(processStage, 100);
        }

        progressBar.style.width = totalProgress + "%";
        progressPercent.textContent = Math.round(totalProgress) + "%";

        // 更新时间
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        elapsedTime.textContent = elapsed + "s";

        if (totalProgress > 0) {
          const estimated = Math.ceil(
            (elapsed / totalProgress) * (100 - totalProgress)
          );
          remainingTime.textContent = estimated + "s";
        }
      }, 50);
    }

    processStage();
  }

  function cancelProcessing() {
    isProcessing = false;
    container.querySelector("#progressSection").classList.add("hidden");
    container.querySelector("#startProcessBtn").disabled = false;
    showToast("处理已取消");
  }

  function completeProcessing() {
    isProcessing = false;

    // 隐藏进度，显示结果
    container.querySelector("#progressSection").classList.add("hidden");
    container.querySelector("#resultSection").classList.remove("hidden");

    // 模拟生成去雾后的图片（实际应该是算法处理结果）
    const resultImage = container.querySelector("#resultImage");
    resultImage.src = currentImage.url; // 这里应该是处理后的图片

    // 显示处理时间
    const totalTime = container.querySelector("#totalTime");
    totalTime.textContent =
      (selectedAlgorithm.performance.time / 1000).toFixed(2) + "s";

    // 保存到全局状态
    window.dehazeApp.processedImage = {
      original: currentImage,
      processed: { url: currentImage.url }, // 实际应该是处理后的图片
      algorithm: selectedAlgorithm,
      params: processingParams,
      mode: processingMode,
    };

    // 添加到历史记录
    historyManager.addRecord({
      fileName: currentImage.file.name,
      thumbnail: currentImage.url,
      algorithm: selectedAlgorithm.name,
      mode: processingMode,
    });

    showToast("处理完成！");
  }

  function saveResult() {
    const resultImage = container.querySelector("#resultImage");
    const link = document.createElement("a");
    link.href = resultImage.src;
    link.download = `dehazed_${Date.now()}.png`;
    link.click();
    showToast("图片已保存");
  }
}
