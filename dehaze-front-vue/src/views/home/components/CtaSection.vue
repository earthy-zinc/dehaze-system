<template>
  <section class="cta-section">
    <div class="cta-container">
      <div class="cta-panel">
        <!-- 左侧：价值主张 + CTA -->
        <div class="cta-main">
          <div class="cta-glow" aria-hidden="true"></div>
          <div class="cta-badge">
            <el-icon><Star /></el-icon>
            免费体验
          </div>
          <h2 class="cta-title">准备好让画面<br />回归清晰了吗？</h2>
          <p class="cta-desc">
            注册即可使用全部去雾算法，支持单张与批量处理，立即查看你的图像修复效果。
          </p>
          <div class="cta-actions">
            <button
              class="cta-btn cta-btn--primary"
              @click="router.push('/register')"
            >
              免费注册
            </button>
            <button
              class="cta-btn cta-btn--ghost"
              @click="router.push('/presentation/dehaze')"
            >
              查看去雾演示
            </button>
          </div>
          <div class="cta-trust">
            <span class="cta-trust__item">
              <el-icon><Check /></el-icon>
              无需信用卡
            </span>
            <span class="cta-trust__item">
              <el-icon><Check /></el-icon>
              30+ 算法可选
            </span>
          </div>
        </div>

        <!-- 右侧：去雾效果可视化 -->
        <div class="cta-visual">
          <img :src="visualUrl" alt="去雾效果展示" />
          <div class="cta-visual__overlay"></div>
          <div class="cta-visual__haze">
            <span class="cta-visual__tag">Before</span>
          </div>
          <div class="cta-visual__clear">
            <span class="cta-visual__tag">After</span>
          </div>
          <div class="cta-visual__metric">
            <el-icon><Lightning /></el-icon>
            PSNR 42 dB
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script lang="ts" setup>
import { Star, Check, Lightning } from "@element-plus/icons-vue";

defineOptions({
  name: "CtaSection",
});

const router = useRouter();

// 去雾效果配图（真实生成，非占位图）
const visualUrl =
  "https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=" +
  encodeURIComponent(
    "A clear urban street scene photo, modern city architecture, bright vivid colors, high detail, professional photography"
  ) +
  "&image_size=landscape_16_9";
</script>

<style lang="scss" scoped>
// 深色 CTA 区亮暗主题一致，不随 html.dark 变化
.cta-section {
  padding: 80px 0 96px;
  background-color: var(--el-bg-color-page);
}

.cta-container {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 24px;
}

.cta-panel {
  display: grid;
  overflow: hidden;
  border-radius: 16px;
  box-shadow: 0 16px 32px rgb(0 0 0 / 14%);

  @media (min-width: 768px) {
    grid-template-columns: 1fr 1fr;
  }
}

.cta-main {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 40px 56px;
  color: #fff;
  background: linear-gradient(to bottom right, #1e40af, #2563eb, #4338ca);

  @media (min-width: 768px) {
    padding: 56px;
  }
}

.cta-glow {
  position: absolute;
  top: -64px;
  left: -64px;
  width: 256px;
  height: 256px;
  background: radial-gradient(
    circle,
    rgba(147, 197, 253, 0.5),
    transparent 70%
  );
  border-radius: 50%;
  filter: blur(50px);
  pointer-events: none;
}

.cta-badge {
  position: relative;
  display: inline-flex;
  gap: 8px;
  align-items: center;
  align-self: flex-start;
  margin-bottom: 20px;
  padding: 4px 12px;
  font-size: 14px;
  font-weight: 500;
  color: #fff;
  background-color: rgb(255 255 255 / 15%);
  border: 1px solid rgb(255 255 255 / 25%);
  border-radius: 9999px;
  backdrop-filter: blur(4px);

  .el-icon {
    font-size: 14px;
    color: var(--home-primary-200);
  }
}

.cta-title {
  margin-bottom: 16px;
  font-size: clamp(24px, 2.4vw, 34px);
  font-weight: 600;
  line-height: 1.25;
}

.cta-desc {
  max-width: 28rem;
  margin-bottom: 32px;
  font-weight: 500;
  line-height: 1.6;
  color: rgb(255 255 255 / 90%);
}

.cta-actions {
  display: flex;
  flex-direction: column;
  gap: 12px;

  @media (min-width: 640px) {
    flex-direction: row;
  }
}

.cta-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 14px 28px;
  font-size: 16px;
  font-weight: 600;
  font-family: inherit;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition:
    background-color 0.15s,
    box-shadow 0.15s;

  &--primary {
    color: var(--home-primary-700);
    background-color: #fff;
    box-shadow: 0 8px 24px rgb(0 0 0 / 16%);

    &:hover {
      background-color: var(--el-fill-color-lighter);
    }
  }

  &--ghost {
    color: #fff;
    background-color: rgb(255 255 255 / 10%);
    border: 1px solid rgb(255 255 255 / 30%);

    &:hover {
      background-color: rgb(255 255 255 / 20%);
    }
  }
}

.cta-trust {
  display: flex;
  gap: 20px;
  margin-top: 28px;
  font-size: 14px;
  color: rgb(255 255 255 / 70%);

  &__item {
    display: flex;
    gap: 6px;
    align-items: center;

    .el-icon {
      font-size: 16px;
    }
  }
}

.cta-visual {
  position: relative;
  min-height: 280px;
  background-color: var(--el-fill-color-dark);

  @media (min-width: 768px) {
    min-height: 100%;
  }

  img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  &__overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
      90deg,
      rgba(30, 58, 138, 0.4) 0%,
      transparent 50%
    );
  }

  &__haze,
  &__clear {
    position: absolute;
    top: 0;
    bottom: 0;
    display: flex;
    align-items: center;
  }

  &__haze {
    left: 0;
    width: 50%;

    &::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(
        90deg,
        rgba(200, 215, 230, 0.55) 0%,
        rgba(200, 215, 230, 0.55) 62%,
        rgba(200, 215, 230, 0.2) 88%,
        rgba(200, 215, 230, 0) 100%
      );
      backdrop-filter: blur(2px);
    }
  }

  &__clear {
    right: 0;
    width: 50%;
  }

  &__tag {
    position: relative;
    z-index: 1;
    margin-left: 20px;
    padding: 6px 12px;
    font-size: 14px;
    font-weight: 500;
    color: #fff;
    border-radius: 8px;

    .cta-visual__haze & {
      background-color: rgb(0 0 0 / 40%);
      border: 1px solid rgb(255 255 255 / 10%);
      backdrop-filter: blur(8px);
    }

    .cta-visual__clear & {
      background-color: var(--home-primary-500);
      box-shadow: 0 4px 12px rgb(0 0 0 / 24%);
    }
  }

  &__metric {
    position: absolute;
    right: 20px;
    bottom: 20px;
    display: flex;
    gap: 6px;
    align-items: center;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 500;
    color: #fff;
    background-color: rgb(0 0 0 / 50%);
    border: 1px solid rgb(255 255 255 / 10%);
    border-radius: 8px;
    backdrop-filter: blur(8px);

    .el-icon {
      font-size: 14px;
      color: var(--home-primary-300);
    }
  }
}
</style>
