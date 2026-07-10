// 滤镜调节模块
import { createUploadArea, showToast } from "../main.js";

let currentImage = null;
let filters = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  hue: 0,
};

const presets = {
  original: { brightness: 0, contrast: 0, saturation: 0, hue: 0 },
  vivid: { brightness: 10, contrast: 20, saturation: 30, hue: 0 },
  soft: { brightness: 5, contrast: -10, saturation: -15, hue: 0 },
  cool: { brightness: 0, contrast: 10, saturation: 0, hue: -20 },
  warm: { brightness: 5, contrast: 5, saturation: 10, hue: 20 },
};

export function initFilter(container) {
  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-adjust text-orange-500"></i>
                滤镜调节
            </h2>
            <p class="text-gray-600 text-sm mb-4">实时调节图片亮度、对比度、饱和度等参数</p>
          
            <!-- 图片上传 -->
            <div id="uploadSection"></div>
          
            <!-- 滤镜调节区域 -->
            <div id="filterContainer" class="hidden">
                <!-- 预设滤镜 -->
                <div class="mb-4">
                    <label class="block text-sm font-medium text-gray-700 mb-2">预设滤镜</label>
                    <div class="grid grid-cols-3 gap-2">
                        <button class="preset-btn active" data-preset="original">
                            <i class="fas fa-undo"></i> 原图
                        </button>
                        <button class="preset-btn" data-preset="vivid">
                            <i class="fas fa-sun"></i> 鲜艳
                        </button>
                        <button class="preset-btn" data-preset="soft">
                            <i class="fas fa-cloud"></i> 柔和
                        </button>
                        <button class="preset-btn" data-preset="cool">
                            <i class="fas fa-snowflake"></i> 冷色
                        </button>
                        <button class="preset-btn" data-preset="warm">
                            <i class="fas fa-fire"></i> 暖色
                        </button>
                        <button class="preset-btn" id="resetBtn">
                            <i class="fas fa-sync"></i> 重置
                        </button>
                    </div>
                </div>
              
                <!-- 图片预览 -->
                <div class="bg-gray-100 rounded-xl overflow-hidden mb-4" style="height: 300px;">
                    <img id="previewImage" class="w-full h-full object-contain">
                </div>
              
                <!-- 滤镜参数调节 -->
                <div class="space-y-4">
                    <!-- 亮度 -->
                    <div class="filter-control">
                        <div class="flex items-center justify-between mb-2">
                            <label class="text-sm font-medium text-gray-700">
                                <i class="fas fa-sun text-yellow-500"></i> 亮度
                            </label>
                            <span class="text-sm font-semibold text-gray-800">
                                <span id="brightnessValue">0</span>%
                            </span>
                        </div>
                        <input type="range" id="brightnessSlider" class="slider" min="-100" max="100" value="0">
                    </div>
                  
                    <!-- 对比度 -->
                    <div class="filter-control">
                        <div class="flex items-center justify-between mb-2">
                            <label class="text-sm font-medium text-gray-700">
                                <i class="fas fa-adjust text-blue-500"></i> 对比度
                            </label>
                            <span class="text-sm font-semibold text-gray-800">
                                <span id="contrastValue">0</span>%
                            </span>
                        </div>
                        <input type="range" id="contrastSlider" class="slider" min="-100" max="100" value="0">
                    </div>
                  
                    <!-- 饱和度 -->
                    <div class="filter-control">
                        <div class="flex items-center justify-between mb-2">
                            <label class="text-sm font-medium text-gray-700">
                                <i class="fas fa-palette text-purple-500"></i> 饱和度
                            </label>
                            <span class="text-sm font-semibold text-gray-800">
                                <span id="saturationValue">0</span>%
                            </span>
                        </div>
                        <input type="range" id="saturationSlider" class="slider" min="-100" max="100" value="0">
                    </div>
                  
                    <!-- 色相 -->
                    <div class="filter-control">
                        <div class="flex items-center justify-between mb-2">
                            <label class="text-sm font-medium text-gray-700">
                                <i class="fas fa-circle-notch text-pink-500"></i> 色相
                            </label>
                            <span class="text-sm font-semibold text-gray-800">
                                <span id="hueValue">0</span>°
                            </span>
                        </div>
                        <input type="range" id="hueSlider" class="slider" min="-180" max="180" value="0">
                    </div>
                </div>
              
                <!-- 操作按钮 -->
                <div class="flex space-x-3 mt-6">
                    <button id="downloadBtn" class="btn btn-primary flex-1">
                        <i class="fas fa-download"></i> 下载图片
                    </button>
                    <button id="changeImageBtn" class="btn btn-secondary flex-1">
                        <i class="fas fa-image"></i> 更换图片
                    </button>
                </div>
            </div>
        </div>
      
        <!-- 对比视图 -->
        <div id="comparisonView" class="card mt-4 hidden">
            <h3 class="font-semibold text-gray-800 mb-3">原图对比</h3>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <p class="text-sm text-gray-600 mb-2 text-center">原图</p>
                    <div class="bg-gray-100 rounded-lg overflow-hidden" style="height: 200px;">
                        <img id="originalImage" class="w-full h-full object-contain">
                    </div>
                </div>
                <div>
                    <p class="text-sm text-gray-600 mb-2 text-center">调节后</p>
                    <div class="bg-gray-100 rounded-lg overflow-hidden" style="height: 200px;">
                        <img id="filteredImage" class="w-full h-full object-contain">
                    </div>
                </div>
            </div>
        </div>
      
        <!-- 使用提示 -->
        <div class="bg-orange-50 border border-orange-200 rounded-xl p-4 mt-4">
            <div class="flex items-start space-x-3">
                <i class="fas fa-info-circle text-orange-500 mt-1"></i>
                <div class="text-sm text-orange-800">
                    <p class="font-semibold mb-1">使用提示</p>
                    <ul class="space-y-1 text-orange-700">
                        <li>• 拖动滑块实时预览滤镜效果</li>
                        <li>• 可使用预设滤镜快速应用常用效果</li>
                        <li>• 调节完成后可下载处理后的图片</li>
                        <li>• 支持查看原图对比效果</li>
                    </ul>
                </div>
            </div>
        </div>
    `;

  // 初始化上传区域
  const uploadSection = container.querySelector("#uploadSection");
  createUploadArea(uploadSection, handleImageUpload, {
    text: "上传图片进行滤镜调节",
  });

  // 初始化滑块
  const sliders = {
    brightness: container.querySelector("#brightnessSlider"),
    contrast: container.querySelector("#contrastSlider"),
    saturation: container.querySelector("#saturationSlider"),
    hue: container.querySelector("#hueSlider"),
  };

  const values = {
    brightness: container.querySelector("#brightnessValue"),
    contrast: container.querySelector("#contrastValue"),
    saturation: container.querySelector("#saturationValue"),
    hue: container.querySelector("#hueValue"),
  };

  // 滑块事件
  Object.keys(sliders).forEach((key) => {
    sliders[key].addEventListener("input", (e) => {
      filters[key] = parseInt(e.target.value);
      values[key].textContent = filters[key];
      applyFilters();

      // 取消预设按钮的激活状态
      container.querySelectorAll(".preset-btn").forEach((btn) => {
        btn.classList.remove("active");
      });
    });
  });

  // 预设滤镜
  container.querySelectorAll(".preset-btn[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = btn.dataset.preset;
      applyPreset(preset);

      container.querySelectorAll(".preset-btn").forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");
    });
  });

  // 重置按钮
  container.querySelector("#resetBtn").addEventListener("click", () => {
    applyPreset("original");
    container.querySelectorAll(".preset-btn").forEach((btn) => {
      btn.classList.remove("active");
    });
    container.querySelector('[data-preset="original"]').classList.add("active");
  });

  // 下载按钮
  container
    .querySelector("#downloadBtn")
    .addEventListener("click", downloadImage);

  // 更换图片按钮
  container.querySelector("#changeImageBtn").addEventListener("click", () => {
    currentImage = null;
    container.querySelector("#uploadSection").classList.remove("hidden");
    container.querySelector("#filterContainer").classList.add("hidden");
    container.querySelector("#comparisonView").classList.add("hidden");
  });

  function handleImageUpload(images) {
    currentImage = images[0];
    showFilterInterface();
  }

  function showFilterInterface() {
    container.querySelector("#uploadSection").classList.add("hidden");
    container.querySelector("#filterContainer").classList.remove("hidden");
    container.querySelector("#comparisonView").classList.remove("hidden");

    const previewImage = container.querySelector("#previewImage");
    const originalImage = container.querySelector("#originalImage");
    const filteredImage = container.querySelector("#filteredImage");

    previewImage.src = currentImage.url;
    originalImage.src = currentImage.url;
    filteredImage.src = currentImage.url;

    applyPreset("original");
  }

  function applyPreset(presetName) {
    const preset = presets[presetName];
    filters = { ...preset };

    // 更新滑块和显示值
    Object.keys(filters).forEach((key) => {
      sliders[key].value = filters[key];
      values[key].textContent = filters[key];
    });

    applyFilters();
  }

  function applyFilters() {
    const previewImage = container.querySelector("#previewImage");
    const filteredImage = container.querySelector("#filteredImage");

    const filterString = `
            brightness(${100 + filters.brightness}%)
            contrast(${100 + filters.contrast}%)
            saturate(${100 + filters.saturation}%)
            hue-rotate(${filters.hue}deg)
        `;

    previewImage.style.filter = filterString;
    filteredImage.style.filter = filterString;
  }

  function downloadImage() {
    showToast("正在生成图片...");

    // 创建canvas来生成处理后的图片
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      // 应用滤镜
      ctx.filter = `
                brightness(${100 + filters.brightness}%)
                contrast(${100 + filters.contrast}%)
                saturate(${100 + filters.saturation}%)
                hue-rotate(${filters.hue}deg)
            `;

      ctx.drawImage(img, 0, 0);

      // 下载
      canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `filtered_${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(url);
        showToast("图片已下载");
      });
    };

    img.src = currentImage.url;
  }
}

// 添加预设按钮样式
const style = document.createElement("style");
style.textContent = `
    .preset-btn {
        padding: 10px 12px;
        border-radius: 10px;
        background: white;
        border: 2px solid #E5E7EB;
        color: #6B7280;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 13px;
        font-weight: 500;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }
  
    .preset-btn i {
        font-size: 18px;
    }
  
    .preset-btn:hover {
        border-color: #F97316;
        color: #F97316;
        transform: translateY(-2px);
    }
  
    .preset-btn.active {
        background: linear-gradient(135deg, #F97316, #FB923C);
        border-color: #F97316;
        color: white;
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
    }
  
    .filter-control {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
    }
`;
document.head.appendChild(style);
