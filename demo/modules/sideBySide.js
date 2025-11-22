// 并排对比模块
import { createUploadArea, createImagePreview, showToast } from "../main.js";

let images = [];
let currentLayout = "horizontal-2";

export function initSideBySide(container) {
  container.innerHTML = `
        <div class="card">
            <h2 class="card-title">
                <i class="fas fa-columns text-blue-500"></i>
                并排对比
            </h2>
            <p class="text-gray-600 text-sm mb-4">支持多张图片并排展示，可切换不同布局模式</p>
          
            <!-- 布局选择 -->
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700 mb-2">布局模式</label>
                <div class="toolbar">
                    <button class="toolbar-btn active" data-layout="horizontal-2">
                        <i class="fas fa-grip-horizontal"></i> 水平2列
                    </button>
                    <button class="toolbar-btn" data-layout="vertical-2">
                        <i class="fas fa-grip-vertical"></i> 垂直2列
                    </button>
                    <button class="toolbar-btn" data-layout="grid-4" id="grid4Btn">
                        <i class="fas fa-th"></i> 网格4列
                    </button>
                </div>
            </div>
          
            <!-- 图片上传区域 -->
            <div id="uploadSection"></div>
          
            <!-- 图片对比区域 -->
            <div id="comparisonArea" class="hidden">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="font-semibold text-gray-800">对比视图</h3>
                    <button id="clearBtn" class="btn btn-secondary btn-sm">
                        <i class="fas fa-trash"></i> 清空
                    </button>
                </div>
                <div id="imageGrid" class="image-grid grid-2"></div>
            </div>
        </div>
      
        <!-- 操作提示 -->
        <div class="bg-blue-50 border border-blue-200 rounded-xl p-4 mt-4">
            <div class="flex items-start space-x-3">
                <i class="fas fa-info-circle text-blue-500 mt-1"></i>
                <div class="text-sm text-blue-800">
                    <p class="font-semibold mb-1">使用提示</p>
                    <ul class="space-y-1 text-blue-700">
                        <li>• 手机端最多支持2张图片同屏对比</li>
                        <li>• 平板设备支持最多4张图片网格布局</li>
                        <li>• 点击图片可查看大图</li>
                        <li>• 支持拖拽上传图片</li>
                    </ul>
                </div>
            </div>
        </div>
    `;

  // 检测设备类型
  const isTablet = window.innerWidth >= 768;
  const grid4Btn = container.querySelector("#grid4Btn");

  if (!isTablet) {
    grid4Btn.disabled = true;
    grid4Btn.classList.add("opacity-50", "cursor-not-allowed");
    grid4Btn.title = "平板设备专用";
  }

  // 初始化上传区域
  const uploadSection = container.querySelector("#uploadSection");
  const maxImages = isTablet ? 4 : 2;

  createUploadArea(uploadSection, handleImageUpload, {
    multiple: true,
    text: `上传图片进行对比（最多${maxImages}张）`,
  });

  // 布局切换
  container.querySelectorAll("[data-layout]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const layout = btn.dataset.layout;

      // 检查是否是网格布局且设备不支持
      if (layout === "grid-4" && !isTablet) {
        showToast("网格布局仅支持平板设备");
        return;
      }

      // 检查图片数量
      if (layout === "grid-4" && images.length < 3) {
        showToast("网格布局需要至少3张图片");
        return;
      }

      currentLayout = layout;

      // 更新按钮状态
      container.querySelectorAll("[data-layout]").forEach((b) => {
        b.classList.remove("active");
      });
      btn.classList.add("active");

      // 更新布局
      updateLayout();
    });
  });

  // 清空按钮
  container.querySelector("#clearBtn").addEventListener("click", () => {
    images = [];
    updateDisplay();
  });

  function handleImageUpload(uploadedImages) {
    // 限制图片数量
    const availableSlots = maxImages - images.length;
    const newImages = uploadedImages.slice(0, availableSlots);

    if (uploadedImages.length > availableSlots) {
      showToast(`最多只能上传${maxImages}张图片`);
    }

    images.push(...newImages);
    updateDisplay();
  }

  function updateDisplay() {
    const comparisonArea = container.querySelector("#comparisonArea");
    const uploadSection = container.querySelector("#uploadSection");

    if (images.length === 0) {
      comparisonArea.classList.add("hidden");
      uploadSection.classList.remove("hidden");
    } else {
      comparisonArea.classList.remove("hidden");
      uploadSection.classList.add("hidden");
      updateLayout();
    }
  }

  function updateLayout() {
    const imageGrid = container.querySelector("#imageGrid");
    imageGrid.innerHTML = "";

    // 设置网格类
    imageGrid.className = "image-grid";

    switch (currentLayout) {
      case "horizontal-2":
        imageGrid.classList.add("grid-2");
        imageGrid.style.gridTemplateColumns = "repeat(2, 1fr)";
        break;
      case "vertical-2":
        imageGrid.classList.add("grid-2");
        imageGrid.style.gridTemplateColumns = "1fr";
        imageGrid.style.gridTemplateRows = "repeat(2, 1fr)";
        break;
      case "grid-4":
        imageGrid.classList.add("grid-4");
        if (window.innerWidth >= 768) {
          imageGrid.style.gridTemplateColumns = "repeat(2, 1fr)";
        }
        break;
    }

    // 渲染图片
    images.forEach((imageData, index) => {
      const imageContainer = document.createElement("div");
      imageContainer.className =
        "relative bg-white rounded-xl shadow-lg overflow-hidden";

      const img = document.createElement("img");
      img.src = imageData.url;
      img.className = "w-full h-full object-contain";
      img.style.minHeight = "200px";
      img.style.maxHeight = currentLayout === "vertical-2" ? "300px" : "400px";

      const label = document.createElement("div");
      label.className =
        "absolute top-2 left-2 bg-black bg-opacity-60 text-white px-3 py-1 rounded-full text-sm font-medium";
      label.textContent = `图片 ${index + 1}`;

      const removeBtn = document.createElement("button");
      removeBtn.className =
        "absolute top-2 right-2 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors";
      removeBtn.innerHTML = '<i class="fas fa-times"></i>';
      removeBtn.addEventListener("click", () => {
        images.splice(index, 1);
        updateDisplay();
      });

      // 点击查看大图
      img.addEventListener("click", () => {
        showImageModal(imageData.url);
      });

      imageContainer.appendChild(img);
      imageContainer.appendChild(label);
      imageContainer.appendChild(removeBtn);
      imageGrid.appendChild(imageContainer);
    });

    // 如果图片不足，显示添加按钮
    if (images.length < maxImages) {
      const addBtn = document.createElement("div");
      addBtn.className =
        "flex items-center justify-center bg-gray-100 rounded-xl border-2 border-dashed border-gray-300 cursor-pointer hover:bg-gray-200 transition-colors";
      addBtn.style.minHeight = "200px";
      addBtn.innerHTML = `
                <div class="text-center text-gray-500">
                    <i class="fas fa-plus text-3xl mb-2"></i>
                    <p class="text-sm">添加图片</p>
                </div>
            `;

      addBtn.addEventListener("click", () => {
        uploadSection.querySelector('input[type="file"]').click();
      });

      imageGrid.appendChild(addBtn);
    }
  }

  function showImageModal(imageUrl) {
    const modal = document.createElement("div");
    modal.className = "modal";
    modal.innerHTML = `
            <div class="modal-content max-w-4xl">
                <div class="modal-header">
                    <h3 class="text-lg font-bold">图片预览</h3>
                    <button class="text-gray-500 hover:text-gray-700">
                        <i class="fas fa-times text-xl"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <img src="${imageUrl}" class="w-full h-auto rounded-lg">
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
}
