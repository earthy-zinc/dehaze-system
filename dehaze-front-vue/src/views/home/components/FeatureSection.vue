<template>
  <section class="feature-section">
    <div class="feature-container">
      <div class="section-head">
        <h2 class="section-title">核心能力</h2>
        <p class="section-desc">从单张图片到批量数据集，全链路去雾能力覆盖</p>
      </div>

      <!-- 核心能力卡片 -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div v-for="f in features" :key="f.title" class="feature-card">
          <div class="feature-card__head">
            <div class="feature-icon">
              <el-icon><component :is="f.icon" /></el-icon>
            </div>
            <h3 class="feature-card__title">{{ f.title }}</h3>
          </div>
          <p class="feature-card__desc">{{ f.desc }}</p>
          <div class="feature-card__tags">
            <span v-for="t in f.tags" :key="t" class="feature-tag">{{
              t
            }}</span>
          </div>
        </div>
      </div>

      <!-- 全链路处理流程 -->
      <div class="flow-panel">
        <div class="flow-panel__head">
          <h3 class="flow-panel__title">全链路处理流程</h3>
          <span class="flow-panel__summary"
            >上传 → 匹配 → 去雾 → 对比 → 导出</span
          >
        </div>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-3 md:gap-4">
          <div v-for="s in flowSteps" :key="s.title" class="flow-step">
            <div class="flow-step__icon">
              <el-icon v-if="s.icon"><component :is="s.icon" /></el-icon>
              <svg-icon v-else :icon-class="s.svgIcon" size="20px" />
            </div>
            <div class="flow-step__title">{{ s.title }}</div>
            <div class="flow-step__desc">{{ s.desc }}</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script lang="ts" setup>
import {
  MagicStick,
  Files,
  View,
  Cpu,
  Upload,
  Lightning,
  DataAnalysis,
  Download,
} from "@element-plus/icons-vue";

defineOptions({
  name: "FeatureSection",
});

const features = [
  {
    icon: MagicStick,
    title: "智能算法推荐",
    desc: "根据场景自动匹配最佳去雾模型，无需专业知识。",
    tags: ["场景识别", "自动选模型", "零配置"],
  },
  {
    icon: Files,
    title: "高清批量处理",
    desc: "支持数据集级批量上传与处理，提升工作效率。",
    tags: ["数据集导入", "并行加速", "原画质保"],
  },
  {
    icon: View,
    title: "多维度效果对比",
    desc: "并排、重叠、放大镜等多种对比模式与量化指标。",
    tags: ["并排/滑块", "PSNR/SSIM", "局部放大"],
  },
  {
    icon: Cpu,
    title: "算法实验室",
    desc: "内置多种 SOTA 去雾算法，支持自定义上传与测评。",
    tags: ["30+ SOTA", "自定义上传", "A/B 测评"],
  },
];

// 流程步骤：icon 为 Element Plus 图标，svgIcon 为项目内置 SVG 图标
const flowSteps = [
  { icon: Upload, title: "上传图像", desc: "单张/批量/数据集" },
  { svgIcon: "bulb", title: "场景识别", desc: "自动匹配算法" },
  { icon: Lightning, title: "智能去雾", desc: "深度学习还原" },
  { icon: DataAnalysis, title: "效果对比", desc: "量化指标评估" },
  { icon: Download, title: "导出结果", desc: "原图/报告导出" },
];
</script>

<style lang="scss" scoped>
.feature-section {
  padding: 80px 0 96px;
  background: linear-gradient(
    to bottom,
    rgba(239, 246, 255, 0.5) 0%,
    var(--el-bg-color) 40%,
    rgba(239, 246, 255, 0.5) 100%
  );

  html.dark & {
    background: var(--el-bg-color);
  }
}

.feature-container {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 24px;
}

.section-head {
  margin-bottom: 56px;
  text-align: center;
}

.section-title {
  margin-bottom: 12px;
  font-size: clamp(24px, 2.4vw, 34px);
  font-weight: 600;
  line-height: 1.25;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.section-desc {
  max-width: 36rem;
  margin: 0 auto;
  font-size: 16px;
  line-height: 1.6;
  color: var(--el-text-color-secondary);
}

.feature-card {
  position: relative;
  padding: 24px;
  background-color: var(--el-bg-color-overlay);
  border-radius: 16px;
  box-shadow: 0 2px 8px rgb(0 0 0 / 8%);
  transition:
    transform 0.25s ease,
    box-shadow 0.25s ease;

  // 蓝色渐变描边
  &::before {
    content: "";
    position: absolute;
    inset: 0;
    padding: 1.5px;
    background: linear-gradient(
      135deg,
      rgba(59, 130, 246, 0.55),
      rgba(99, 102, 241, 0.35),
      rgba(6, 182, 212, 0.25)
    );
    border-radius: inherit;
    -webkit-mask:
      linear-gradient(#fff 0 0) content-box,
      linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask:
      linear-gradient(#fff 0 0) content-box,
      linear-gradient(#fff 0 0);
    mask-composite: exclude;
    pointer-events: none;
  }

  &:hover {
    box-shadow:
      0 8px 24px rgb(0 0 0 / 12%),
      0 0 0 1px rgb(59 130 246 / 12%);
    transform: translateY(-6px);
  }

  &__head {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 12px;
  }

  &__title {
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  &__desc {
    margin-bottom: 16px;
    padding-left: 60px;
    font-size: 14px;
    line-height: 1.5;
    color: var(--el-text-color-secondary);
  }

  &__tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding-left: 60px;
  }
}

.feature-icon {
  display: flex;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  font-size: 24px;
  color: var(--home-primary-600);
  background-color: var(--home-primary-50);
  border-radius: 12px;
  box-shadow: 0 0 0 4px rgb(59 130 246 / 8%);
  transition: box-shadow 0.25s ease;

  .feature-card:hover & {
    box-shadow: 0 0 0 6px rgb(59 130 246 / 10%);
  }
}

.feature-tag {
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 500;
  color: var(--home-primary-700);
  background-color: var(--home-primary-50);
  border-radius: 6px;
}

.flow-panel {
  margin-top: 48px;
  padding: 28px 36px;
  background-color: var(--el-bg-color-overlay);
  border: 1px solid var(--el-border-color-light);
  border-radius: 16px;

  &__head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
  }

  &__title {
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  &__summary {
    font-size: 14px;
    font-weight: 500;
    color: var(--el-text-color-secondary);
  }
}

.flow-step {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: center;
  text-align: center;

  &__icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    font-size: 20px;
    color: var(--home-primary-600);
    background-color: var(--home-primary-50);
    border-radius: 12px;
  }

  &__title {
    font-size: 14px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  &__desc {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}
</style>
