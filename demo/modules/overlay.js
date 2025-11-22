// 重叠对比模块
import { createUploadArea, showToast } from "../main.js";

let beforeImage = null;
let afterImage = null;
let comparisonMode = "vertical"; // vertical, horizontal, opacity
let dividerPosition = 50; // 百分比
let opacity = 50; // 透明度百分比

export function initOverlay(container) {
  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-layer-group text-purple-500"></i>
                重叠对比
            </h2>
            <p class="text-gray-600 text-sm mb-4">通过分割线或透明度叠加对比两张图片</p>
          
            <!-- 对比模式选择 -->
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-2">对比模式</label>
                <div class="toolbar">
                    <button class="toolbar-btn active" data-mode="vertical">
                        <i class="fas fa-grip-lines-vertical"></i> 左右分割
                    </button>
                    <button class="toolbar-btn" data-mode="horizontal">
                        <i class="fas fa-grip-lines"></i> 上下分割
                    </button>
                    <button class="toolbar-btn" data-mode="opacity">
                        <i class="fas fa-adjust"></i> 透明度
                    </button>
                </div>
            </div>
          
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
          
            <!-- 对比区域 -->
            <div id="comparisonContainer" class="hidden mt-4">
                <div class="bg-white rounded-xl shadow-lg p-4">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="font-semibold text-gray-800">对比视图</h3>
                        <button id="resetBtn" class="btn btn-secondary btn-sm">
                            <i class="fas fa-redo"></i> 重新上传
                        </button>
                    </div>
                  
                    <!-- 透明度控制（仅在透明度模式显示） -->
                    <div id="opacityControl" class="mb-4 hidden">
                        <label class="block text-sm font-medium text-gray-700 mb-2">
                            透明度: <span id="opacityValue">50</span>%
                        </label>
                        <input type="range" id="opacitySlider" class="slider" min="0" max="100" value="50">
                    </div>
                  
                    <!-- 图片对比容器 -->
                    <div id="imageComparison" class="relative bg-gray-100 rounded-lg overflow-hidden" style="height: 400px;">
                        <!-- 底层图片（去雾前） -->
                        <div id="beforeLayer" class="absolute inset-0">
                            <img id="beforeImg" class="w-full h-full object-contain">
                        </div>
                      
                        <!-- 顶层图片（去雾后） -->
                        <div id="afterLayer" class="absolute inset-0">
                            <img id="afterImg" class="w-full h-full object-contain">
                        </div>
                      
                        <!-- 分割线 -->
                        <div id="divider" class="divider-line vertical" style="left: 50%;">
                            <div class="divider-handle">
                                <i class="fas fa-arrows-alt-h"></i>
                            </div>
                        </div>
                    </div>
                  
                    <!-- 图片标签 -->
                    <div class="flex justify-between mt-3 text-sm">
                        <div class="flex items-center space-x-2">
                            <div class="w-4 h-4 bg-blue-500 rounded"></div>
                            <span class="text-gray-600">去雾前</span>
                        </div>
                        <div class="flex items-center space-x-2">
                            <div class="w-4 h-4 bg-green-500 rounded"></div>
                            <span class="text-gray-600">去雾后</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
      
        <!-- 使用提示 -->
        <div class="bg-purple-50 border border-purple-200 rounded-xl p-4 mt-4">
            <div class="flex items-start space-x-3">
                <i class="fas fa-info-circle text-purple-500 mt-1"></i>
                <div class="text-sm text-purple-800">
                    <p class="font-semibold mb-1">使用提示</p>
                    <ul class="space-y-1 text-purple-700">
                        <li>• 拖动分割线可调整对比区域</li>
                        <li>• 透明度模式支持实时调节叠加效果</li>
                        <li>• 建议上传相同尺寸的图片以获得最佳效果</li>
                    </ul>
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
      checkAndShowComparison();
    },
    { text: "上传去雾前图片" }
  );

  createUploadArea(
    afterUpload,
    (images) => {
      afterImage = images[0];
      checkAndShowComparison();
    },
    { text: "上传去雾后图片" }
  );

  // 模式切换
  container.querySelectorAll("[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      comparisonMode = btn.dataset.mode;

      container.querySelectorAll("[data-mode]").forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");

      updateComparisonMode();
    });
  });

  // 重置按钮
  container.querySelector("#resetBtn").addEventListener("click", () => {
    beforeImage = null;
    afterImage = null;
    container.querySelector("#uploadSection").classList.remove("hidden");
    container.querySelector("#comparisonContainer").classList.add("hidden");
  });

  // 透明度滑块
  const opacitySlider = container.querySelector("#opacitySlider");
  const opacityValue = container.querySelector("#opacityValue");

  opacitySlider.addEventListener("input", (e) => {
    opacity = e.target.value;
    opacityValue.textContent = opacity;
    updateOpacity();
  });

  function checkAndShowComparison() {
    if (beforeImage && afterImage) {
      showComparison();
    }
  }

  function showComparison() {
    container.querySelector("#uploadSection").classList.add("hidden");
    container.querySelector("#comparisonContainer").classList.remove("hidden");

    const beforeImg = container.querySelector("#beforeImg");
    const afterImg = container.querySelector("#afterImg");

    beforeImg.src = beforeImage.url;
    afterImg.src = afterImage.url;

    updateComparisonMode();
    initDividerDrag();
  }

  function updateComparisonMode() {
    if (!beforeImage || !afterImage) return;

    const divider = container.querySelector("#divider");
    const afterLayer = container.querySelector("#afterLayer");
    const opacityControl = container.querySelector("#opacityControl");
    const dividerHandle = divider.querySelector(".divider-handle i");

    // 重置样式
    afterLayer.style.clipPath = "";
    afterLayer.style.opacity = "";
    divider.className = "divider-line";

    switch (comparisonMode) {
      case "vertical":
        divider.classList.add("vertical");
        divider.style.left = dividerPosition + "%";
        divider.style.top = "0";
        afterLayer.style.clipPath = `inset(0 0 0 ${dividerPosition}%)`;
        dividerHandle.className = "fas fa-arrows-alt-h";
        opacityControl.classList.add("hidden");
        divider.classList.remove("hidden");
        break;

      case "horizontal":
        divider.classList.add("horizontal");
        divider.style.top = dividerPosition + "%";
        divider.style.left = "0";
        afterLayer.style.clipPath = `inset(${dividerPosition}% 0 0 0)`;
        dividerHandle.className = "fas fa-arrows-alt-v";
        opacityControl.classList.add("hidden");
        divider.classList.remove("hidden");
        break;

      case "opacity":
        afterLayer.style.opacity = opacity / 100;
        opacityControl.classList.remove("hidden");
        divider.classList.add("hidden");
        break;
    }
  }

  function updateOpacity() {
    if (comparisonMode === "opacity") {
      const afterLayer = container.querySelector("#afterLayer");
      afterLayer.style.opacity = opacity / 100;
    }
  }

  function initDividerDrag() {
    const divider = container.querySelector("#divider");
    const imageComparison = container.querySelector("#imageComparison");
    let isDragging = false;

    function startDrag(e) {
      if (comparisonMode === "opacity") return;
      isDragging = true;
      e.preventDefault();
    }

    function drag(e) {
      if (!isDragging) return;

      const rect = imageComparison.getBoundingClientRect();
      let position;

      if (comparisonMode === "vertical") {
        const clientX = e.type.includes("touch")
          ? e.touches[0].clientX
          : e.clientX;
        position = ((clientX - rect.left) / rect.width) * 100;
      } else if (comparisonMode === "horizontal") {
        const clientY = e.type.includes("touch")
          ? e.touches[0].clientY
          : e.clientY;
        position = ((clientY - rect.top) / rect.height) * 100;
      }

      position = Math.max(0, Math.min(100, position));
      dividerPosition = position;
      updateComparisonMode();
    }

    function stopDrag() {
      isDragging = false;
    }

    // 鼠标事件
    divider.addEventListener("mousedown", startDrag);
    document.addEventListener("mousemove", drag);
    document.addEventListener("mouseup", stopDrag);

    // 触摸事件
    divider.addEventListener("touchstart", startDrag);
    document.addEventListener("touchmove", drag);
    document.addEventListener("touchend", stopDrag);
  }
}
