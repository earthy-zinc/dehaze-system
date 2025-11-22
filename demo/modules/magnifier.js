// 放大镜模块
import { createUploadArea, showToast, getAppState } from "../main.js";

let images = [];
let magnifierSettings = {
  size: 150,
  zoom: 2.5,
  shape: "circle",
  borderColor: "#3B82F6",
};

export function initMagnifier(container) {
  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-search-plus text-green-500"></i>
                放大镜功能
            </h2>
            <p class="text-gray-600 text-sm mb-4">长按图片查看局部细节，支持多图联动放大</p>
          
            <!-- 放大镜设置 -->
            <div class="bg-gray-50 rounded-xl p-4 mb-4">
                <h3 class="font-semibold text-gray-800 mb-3">放大镜设置</h3>
              
                <!-- 放大倍数 -->
                <div class="mb-4">
                    <label class="block text-sm font-medium text-gray-700 mb-2">
                        放大倍数: <span id="zoomValue">2.5</span>x
                    </label>
                    <input type="range" id="zoomSlider" class="slider" min="1.5" max="5" step="0.5" value="2.5">
                </div>
              
                <!-- 放大镜大小 -->
                <div class="mb-4">
                    <label class="block text-sm font-medium text-gray-700 mb-2">
                        放大镜大小: <span id="sizeValue">150</span>px
                    </label>
                    <input type="range" id="sizeSlider" class="slider" min="100" max="300" step="10" value="150">
                </div>
              
                <!-- 形状选择 -->
                <div class="mb-4">
                    <label class="block text-sm font-medium text-gray-700 mb-2">形状</label>
                    <div class="flex space-x-3">
                        <button class="shape-btn active" data-shape="circle">
                            <i class="fas fa-circle"></i> 圆形
                        </button>
                        <button class="shape-btn" data-shape="square">
                            <i class="fas fa-square"></i> 方形
                        </button>
                    </div>
                </div>
              
                <!-- 联动模式 -->
                <div class="flex items-center justify-between">
                    <span class="text-sm font-medium text-gray-700">多图联动</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="linkToggle" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>
          
            <!-- 图片上传 -->
            <div id="uploadSection"></div>
          
            <!-- 图片展示区域 -->
            <div id="imagesContainer" class="hidden">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="font-semibold text-gray-800">图片列表</h3>
                    <button id="clearBtn" class="btn btn-secondary btn-sm">
                        <i class="fas fa-trash"></i> 清空
                    </button>
                </div>
                <div id="imageGrid" class="grid grid-cols-1 md:grid-cols-2 gap-4"></div>
            </div>
        </div>
      
        <!-- 使用提示 -->
        <div class="bg-green-50 border border-green-200 rounded-xl p-4 mt-4">
            <div class="flex items-start space-x-3">
                <i class="fas fa-info-circle text-green-500 mt-1"></i>
                <div class="text-sm text-green-800">
                    <p class="font-semibold mb-1">使用提示</p>
                    <ul class="space-y-1 text-green-700">
                        <li>• 长按图片任意位置触发放大镜</li>
                        <li>• 移动手指查看不同区域的细节</li>
                        <li>• 开启联动模式后，多张图片会同步放大相同位置</li>
                        <li>• 可自定义放大倍数、大小和形状</li>
                    </ul>
                </div>
            </div>
        </div>
    `;

  // 初始化设置控件
  const zoomSlider = container.querySelector("#zoomSlider");
  const zoomValue = container.querySelector("#zoomValue");
  const sizeSlider = container.querySelector("#sizeSlider");
  const sizeValue = container.querySelector("#sizeValue");
  const linkToggle = container.querySelector("#linkToggle");

  zoomSlider.addEventListener("input", (e) => {
    magnifierSettings.zoom = parseFloat(e.target.value);
    zoomValue.textContent = magnifierSettings.zoom;
  });

  sizeSlider.addEventListener("input", (e) => {
    magnifierSettings.size = parseInt(e.target.value);
    sizeValue.textContent = magnifierSettings.size;
  });

  // 形状选择
  container.querySelectorAll(".shape-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      magnifierSettings.shape = btn.dataset.shape;
      container.querySelectorAll(".shape-btn").forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");
    });
  });

  // 初始化上传区域
  const uploadSection = container.querySelector("#uploadSection");
  createUploadArea(uploadSection, handleImageUpload, {
    multiple: true,
    text: "上传图片（支持多张）",
  });

  // 清空按钮
  container.querySelector("#clearBtn").addEventListener("click", () => {
    images = [];
    updateDisplay();
  });

  function handleImageUpload(uploadedImages) {
    images.push(...uploadedImages);
    updateDisplay();
  }

  function updateDisplay() {
    const imagesContainer = container.querySelector("#imagesContainer");
    const uploadSection = container.querySelector("#uploadSection");
    const imageGrid = container.querySelector("#imageGrid");

    if (images.length === 0) {
      imagesContainer.classList.add("hidden");
      uploadSection.classList.remove("hidden");
    } else {
      imagesContainer.classList.remove("hidden");
      uploadSection.classList.add("hidden");

      imageGrid.innerHTML = "";
      images.forEach((imageData, index) => {
        const imageContainer = createImageWithMagnifier(imageData, index);
        imageGrid.appendChild(imageContainer);
      });
    }
  }

  function createImageWithMagnifier(imageData, index) {
    const wrapper = document.createElement("div");
    wrapper.className =
      "relative bg-white rounded-xl shadow-lg overflow-hidden";

    const imageContainer = document.createElement("div");
    imageContainer.className = "image-container relative";
    imageContainer.style.height = "300px";
    imageContainer.dataset.index = index;

    const img = document.createElement("img");
    img.src = imageData.url;
    img.className = "w-full h-full object-contain";
    img.draggable = false;

    const label = document.createElement("div");
    label.className =
      "absolute top-2 left-2 bg-black bg-opacity-60 text-white px-3 py-1 rounded-full text-sm font-medium z-10";
    label.textContent = `图片 ${index + 1}`;

    const removeBtn = document.createElement("button");
    removeBtn.className =
      "absolute top-2 right-2 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors z-10";
    removeBtn.innerHTML = '<i class="fas fa-times"></i>';
    removeBtn.addEventListener("click", () => {
      images.splice(index, 1);
      updateDisplay();
    });

    imageContainer.appendChild(img);
    wrapper.appendChild(imageContainer);
    wrapper.appendChild(label);
    wrapper.appendChild(removeBtn);

    // 添加放大镜功能
    addMagnifierToImage(imageContainer, img);

    return wrapper;
  }

  function addMagnifierToImage(imageContainer, img) {
    let magnifier = null;
    let longPressTimer = null;
    let isActive = false;

    function createMagnifier() {
      magnifier = document.createElement("div");
      magnifier.className = `magnifier ${magnifierSettings.shape}`;
      magnifier.style.width = magnifierSettings.size + "px";
      magnifier.style.height = magnifierSettings.size + "px";
      magnifier.style.borderColor = magnifierSettings.borderColor;

      const magnifierImg = document.createElement("img");
      magnifierImg.src = img.src;
      magnifier.appendChild(magnifierImg);

      imageContainer.appendChild(magnifier);
      return magnifierImg;
    }

    function showMagnifier(e) {
      if (!magnifier) {
        const magnifierImg = createMagnifier();

        // 计算放大图片的尺寸
        const rect = imageContainer.getBoundingClientRect();
        const imgRect = img.getBoundingClientRect();
        magnifierImg.style.width =
          imgRect.width * magnifierSettings.zoom + "px";
        magnifierImg.style.height =
          imgRect.height * magnifierSettings.zoom + "px";
      }

      magnifier.style.display = "block";
      isActive = true;
    }

    function moveMagnifier(e) {
      if (!isActive || !magnifier) return;

      const touch = e.type.includes("touch") ? e.touches[0] : e;
      const rect = imageContainer.getBoundingClientRect();
      const imgRect = img.getBoundingClientRect();

      // 计算鼠标在图片上的相对位置
      let x = touch.clientX - imgRect.left;
      let y = touch.clientY - imgRect.top;

      // 限制在图片范围内
      x = Math.max(0, Math.min(x, imgRect.width));
      y = Math.max(0, Math.min(y, imgRect.height));

      // 放大镜位置
      const magnifierX = touch.clientX - rect.left;
      const magnifierY = touch.clientY - rect.top;

      magnifier.style.left = magnifierX + "px";
      magnifier.style.top = magnifierY + "px";
      magnifier.style.transform = "translate(-50%, -50%)";

      // 计算放大图片的偏移
      const magnifierImg = magnifier.querySelector("img");
      const offsetX = x * magnifierSettings.zoom - magnifierSettings.size / 2;
      const offsetY = y * magnifierSettings.zoom - magnifierSettings.size / 2;

      magnifierImg.style.left = -offsetX + "px";
      magnifierImg.style.top = -offsetY + "px";

      // 联动模式
      if (linkToggle.checked) {
        syncMagnifiers(
          x / imgRect.width,
          y / imgRect.height,
          imageContainer.dataset.index
        );
      }
    }

    function hideMagnifier() {
      if (magnifier) {
        magnifier.style.display = "none";
      }
      isActive = false;
      clearTimeout(longPressTimer);
    }

    // 触摸事件
    imageContainer.addEventListener("touchstart", (e) => {
      e.preventDefault();
      longPressTimer = setTimeout(() => {
        showMagnifier(e);
        moveMagnifier(e);
      }, 300);
    });

    imageContainer.addEventListener("touchmove", (e) => {
      e.preventDefault();
      if (isActive) {
        moveMagnifier(e);
      }
    });

    imageContainer.addEventListener("touchend", (e) => {
      e.preventDefault();
      hideMagnifier();
    });

    // 鼠标事件（桌面端）
    imageContainer.addEventListener("mousedown", (e) => {
      longPressTimer = setTimeout(() => {
        showMagnifier(e);
        moveMagnifier(e);
      }, 300);
    });

    imageContainer.addEventListener("mousemove", (e) => {
      if (isActive) {
        moveMagnifier(e);
      }
    });

    imageContainer.addEventListener("mouseup", hideMagnifier);
    imageContainer.addEventListener("mouseleave", hideMagnifier);
  }

  function syncMagnifiers(relativeX, relativeY, excludeIndex) {
    const allContainers = container.querySelectorAll(".image-container");

    allContainers.forEach((cont, index) => {
      if (index.toString() === excludeIndex) return;

      const img = cont.querySelector("img");
      const magnifier = cont.querySelector(".magnifier");

      if (!magnifier) {
        // 创建联动的放大镜
        const newMagnifier = document.createElement("div");
        newMagnifier.className = `magnifier ${magnifierSettings.shape}`;
        newMagnifier.style.width = magnifierSettings.size + "px";
        newMagnifier.style.height = magnifierSettings.size + "px";
        newMagnifier.style.borderColor = "#10B981"; // 绿色表示联动

        const magnifierImg = document.createElement("img");
        magnifierImg.src = img.src;
        const imgRect = img.getBoundingClientRect();
        magnifierImg.style.width =
          imgRect.width * magnifierSettings.zoom + "px";
        magnifierImg.style.height =
          imgRect.height * magnifierSettings.zoom + "px";

        newMagnifier.appendChild(magnifierImg);
        cont.appendChild(newMagnifier);
      }

      const syncMagnifier = cont.querySelector(".magnifier");
      const imgRect = img.getBoundingClientRect();
      const contRect = cont.getBoundingClientRect();

      // 计算对应位置
      const x = relativeX * imgRect.width;
      const y = relativeY * imgRect.height;

      syncMagnifier.style.display = "block";
      syncMagnifier.style.left = imgRect.left - contRect.left + x + "px";
      syncMagnifier.style.top = imgRect.top - contRect.top + y + "px";
      syncMagnifier.style.transform = "translate(-50%, -50%)";

      const magnifierImg = syncMagnifier.querySelector("img");
      const offsetX = x * magnifierSettings.zoom - magnifierSettings.size / 2;
      const offsetY = y * magnifierSettings.zoom - magnifierSettings.size / 2;

      magnifierImg.style.left = -offsetX + "px";
      magnifierImg.style.top = -offsetY + "px";

      // 自动隐藏联动放大镜
      setTimeout(() => {
        syncMagnifier.style.display = "none";
      }, 100);
    });
  }
}

// 添加形状按钮样式
const style = document.createElement("style");
style.textContent = `
    .shape-btn {
        flex: 1;
        padding: 8px 16px;
        border-radius: 8px;
        background: white;
        border: 2px solid #E5E7EB;
        color: #6B7280;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 14px;
        font-weight: 500;
    }
  
    .shape-btn:hover {
        border-color: #3B82F6;
        color: #3B82F6;
    }
  
    .shape-btn.active {
        background: #3B82F6;
        border-color: #3B82F6;
        color: white;
    }
`;
document.head.appendChild(style);
