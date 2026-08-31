<template>
  <section class="hero-section">
    <img :src="bgUrl" alt="" class="hero-bg" aria-hidden="true" />
    <div class="hero-bg-mask"></div>
    <div class="hero-content">
      <div class="hero-badge">
        <span class="hero-badge__dot"></span>
        AI 驱动的图像去雾平台
      </div>
      <h1 class="hero-title">让每一张图片<br />都清晰可见</h1>
      <p class="hero-subtitle">
        基于深度学习的智能去雾算法，自动识别雾霾、水汽与低光照场景，一键还原真实色彩与细节。
      </p>
      <div class="hero-actions">
        <button
          class="hero-btn hero-btn--primary"
          @click="router.push('/presentation/dehaze')"
        >
          <el-icon class="hero-btn__icon"><Lightning /></el-icon>
          立即体验去雾
        </button>
        <button
          class="hero-btn hero-btn--secondary"
          @click="router.push('/dashboard')"
        >
          进入工作台
        </button>
      </div>
    </div>
  </section>
</template>

<script lang="ts" setup>
import { Lightning } from "@element-plus/icons-vue";

defineOptions({
  name: "HeroSection",
});

const router = useRouter();

// 背景配图（真实生成，非占位图）
const bgUrl =
  "https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=" +
  encodeURIComponent(
    "Soft morning mist over a serene mountain landscape, light blue and white tones, hazy atmosphere, wide shot, professional photography"
  ) +
  "&image_size=landscape_16_9";
</script>

<style lang="scss" scoped>
.hero-section {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #eff6ff 0%, #f0fdfa 50%, #f5f3ff 100%);

  html.dark & {
    background: linear-gradient(
      135deg,
      var(--el-bg-color) 0%,
      var(--el-bg-color-overlay) 50%,
      var(--el-bg-color) 100%
    );
  }
}

.hero-bg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  object-fit: cover;
}

.hero-bg-mask {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(
    to bottom,
    rgb(255 255 255 / 80%) 0%,
    rgb(255 255 255 / 55%) 50%,
    rgb(255 255 255 / 92%) 100%
  );

  html.dark & {
    // 暗色下不叠加白色蒙版，避免设计稿暗色发灰的问题，直接透出暗色背景
    background: linear-gradient(
      to bottom,
      var(--el-bg-color) 0%,
      rgb(24 32 48 / 55%) 50%,
      var(--el-bg-color) 100%
    );
  }
}

.hero-content {
  position: relative;
  max-width: 1280px;
  padding: 96px 24px 128px;
  margin: 0 auto;
  text-align: center;

  @media (width >= 768px) {
    padding: 128px 24px;
  }
}

.hero-badge {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  padding: 4px 12px;
  margin-bottom: 24px;
  font-size: 14px;
  font-weight: 500;
  color: var(--home-primary-700);
  background-color: rgb(239 246 255 / 80%);
  border: 1px solid var(--home-primary-100);
  border-radius: 9999px;
  backdrop-filter: blur(4px);

  html.dark & {
    background-color: rgb(59 130 246 / 15%);
    border-color: rgb(59 130 246 / 30%);
  }

  &__dot {
    width: 8px;
    height: 8px;
    background-color: var(--home-primary-500);
    border-radius: 50%;
  }
}

.hero-title {
  margin-bottom: 24px;
  font-size: clamp(40px, 5vw, 72px);
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: -0.03em;
  background: linear-gradient(100deg, #3b82f6 0%, #06b6d4 45%, #8b5cf6 100%);
  background-clip: text;
  -webkit-text-fill-color: transparent;
  transform: perspective(900px) rotateX(2deg) skewY(-1deg);
}

.hero-subtitle {
  max-width: 42rem;
  margin: 0 auto 40px;
  font-size: 16px;
  font-weight: 500;
  line-height: 1.6;
  color: var(--el-text-color-regular);
}

.hero-actions {
  display: flex;
  gap: 16px;
  justify-content: center;
}

.hero-btn {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  justify-content: center;
  padding: 14px 32px;
  font-family: inherit;
  font-size: 16px;
  cursor: pointer;
  border: none;
  border-radius: 8px;
  transition:
    transform 0.15s ease,
    box-shadow 0.15s ease,
    background-color 0.15s ease;

  &:active {
    transform: scale(0.97);
  }

  &__icon {
    font-size: 20px;
  }

  &--primary {
    color: #fff;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    box-shadow: 0 10px 24px rgb(59 130 246 / 35%);

    &:hover {
      box-shadow: 0 12px 28px rgb(59 130 246 / 45%);
      transform: translateY(-1px);
    }
  }

  &--secondary {
    color: var(--el-text-color-primary);
    background-color: var(--el-fill-color-light);

    &:hover {
      background-color: var(--el-fill-color);
    }
  }
}
</style>
