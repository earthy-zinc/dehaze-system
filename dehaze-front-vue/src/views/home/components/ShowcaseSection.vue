<template>
  <section class="showcase-section">
    <div class="showcase-glow" aria-hidden="true"></div>
    <div class="showcase-container">
      <div class="showcase-grid">
        <!-- 左侧：去雾实证视觉 -->
        <div>
          <div class="showcase-image">
            <img :src="compareUrl" alt="去雾效果对比" />
            <div class="showcase-image__haze">
              <span class="showcase-image__label">雾化输入</span>
            </div>
            <div class="showcase-image__clear">
              <span class="showcase-image__label">Dehaze 还原</span>
            </div>
            <div class="showcase-image__divider">
              <div class="showcase-image__divider--haze"></div>
              <div class="showcase-image__divider--clear"></div>
            </div>
          </div>
          <p class="showcase-caption">
            <el-icon><CircleCheck /></el-icon>
            真实场景去雾前后对比 · 画质细节完整还原
          </p>
        </div>

        <!-- 右侧：能力数据 -->
        <div>
          <div class="showcase-badge">
            <span class="showcase-badge__dot"></span>
            平台实力
          </div>
          <h2 class="showcase-title">数据驱动的<br />去雾引擎</h2>
          <p class="showcase-desc">
            覆盖 30+ SOTA 算法，千万级真实图像训练，量化指标行业领先。
          </p>
          <div class="grid grid-cols-2 gap-5">
            <div v-for="m in metrics" :key="m.label" class="metric-card">
              <div class="metric-card__value">
                <span
                  :class="
                    m.highlight
                      ? 'metric-card__num--primary'
                      : 'metric-card__num'
                  "
                  >{{ m.value }}</span
                >
                <span
                  class="metric-card__unit"
                  :class="{ 'metric-card__unit--primary': m.highlight }"
                  >{{ m.unit }}</span
                >
              </div>
              <div class="metric-card__label">{{ m.label }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- 行业认可 + 客户证言 -->
      <div class="showcase-feedback">
        <div class="showcase-feedback__head">
          <div class="showcase-badge showcase-badge--center">
            <span class="showcase-badge__dot"></span>
            行业认可
          </div>
          <h3 class="showcase-feedback__title">被领先机构与团队信赖</h3>
        </div>
        <div class="showcase-orgs">
          <template v-for="(org, i) in organizations" :key="org">
            <span class="showcase-orgs__item">{{ org }}</span>
            <span
              v-if="i < organizations.length - 1"
              class="showcase-orgs__divider"
            ></span>
          </template>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div v-for="t in testimonials" :key="t.name" class="testimonial-card">
            <el-icon class="testimonial-card__quote"><ChatDotRound /></el-icon>
            <p class="testimonial-card__text">{{ t.quote }}</p>
            <div class="testimonial-card__author">
              <div
                class="testimonial-card__avatar"
                :style="{ background: t.avatarBg }"
              >
                {{ t.avatar }}
              </div>
              <div>
                <div class="testimonial-card__name">{{ t.name }}</div>
                <div class="testimonial-card__org">{{ t.org }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script lang="ts" setup>
import { CircleCheck, ChatDotRound } from "@element-plus/icons-vue";

defineOptions({
  name: "ShowcaseSection",
});

// 效果对比配图（真实生成，非占位图）
const compareUrl =
  "https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=" +
  encodeURIComponent(
    "A crystal clear scenic landscape photo, snow-capped mountains and turquoise lake, vivid colors, high detail, professional photography"
  ) +
  "&image_size=landscape_16_9";

const metrics = [
  { value: "30", unit: "+", label: "SOTA 去雾算法", highlight: false },
  { value: "1000", unit: "万+", label: "累计处理图像", highlight: false },
  { value: "42", unit: "dB", label: "平均 PSNR · 行业领先", highlight: true },
  { value: "4.9", unit: "/5", label: "用户综合评分", highlight: false },
];

const organizations = [
  "中科院自动化所",
  "清华大学视觉实验室",
  "华为云生态",
  "商汤研究院",
  "旷视科技",
];

const testimonials = [
  {
    quote:
      "Dehaze 的批量去雾管线让我们的遥感数据预处理效率提升了 6 倍，PSNR 稳定在 41dB 以上，已成为实验室标配工具。",
    avatar: "王",
    name: "王博士",
    org: "中科院自动化所 · 遥感组",
    avatarBg: "linear-gradient(135deg,#60a5fa,#2563eb)",
  },
  {
    quote:
      "算法市场覆盖主流 SOTA 方案，对比评估一目了然，为我们的模型选型节省了大量调研时间。",
    avatar: "李",
    name: "李研究员",
    org: "清华大学 · 视觉实验室",
    avatarBg: "linear-gradient(135deg,#818cf8,#6366f1)",
  },
  {
    quote:
      "指标评估体系完善，从 PSNR 到 SSIM 全链路可追溯，对接我们的生产流水线毫无障碍。",
    avatar: "张",
    name: "张工程师",
    org: "商汤研究院 · 平台组",
    avatarBg: "linear-gradient(135deg,#34d399,#059669)",
  },
];
</script>

<style lang="scss" scoped>
// 深色实景区亮暗主题一致，不随 html.dark 变化
.showcase-section {
  position: relative;
  padding: 96px 0 128px;
  overflow: hidden;
  color: #fff;
  background: linear-gradient(to bottom right, #1e3a8a, #1e40af, #312e81);
}

.showcase-glow {
  position: absolute;
  top: -80px;
  right: -80px;
  width: 384px;
  height: 384px;
  pointer-events: none;
  background: radial-gradient(circle, rgb(96 165 250 / 50%), transparent 70%);
  border-radius: 50%;
  filter: blur(60px);
}

.showcase-container {
  position: relative;
  max-width: 1280px;
  padding: 0 24px;
  margin: 0 auto;
}

.showcase-grid {
  display: grid;
  gap: 40px 56px;
  align-items: center;

  @media (width >= 1024px) {
    grid-template-columns: 1fr 1fr;
  }
}

.showcase-image {
  position: relative;
  overflow: hidden;
  border: 1px solid rgb(255 255 255 / 10%);
  border-radius: 16px;
  box-shadow: 0 16px 32px rgb(0 0 0 / 30%);

  img {
    display: block;
    width: 100%;
    height: 288px;
    object-fit: cover;

    @media (width >= 768px) {
      height: 320px;
    }
  }

  // 左侧雾化蒙版
  &__haze {
    position: absolute;
    inset: 0 auto 0 0;
    width: 50%;
    background: linear-gradient(
      90deg,
      rgb(180 200 220 / 60%) 0%,
      rgb(180 200 220 / 60%) 62%,
      rgb(180 200 220 / 22%) 88%,
      rgb(180 200 220 / 0%) 100%
    );
    backdrop-filter: blur(1.5px);
  }

  &__clear {
    position: absolute;
    inset: 0 0 0 50%;
  }

  &__label {
    position: absolute;
    top: 16px;
    left: 16px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 500;
    color: #fff;
    border-radius: 6px;

    .showcase-image__haze & {
      background-color: rgb(0 0 0 / 35%);
      backdrop-filter: blur(4px);
    }

    .showcase-image__clear & {
      background-color: rgb(59 130 246 / 90%);
      backdrop-filter: blur(4px);
    }
  }

  &__divider {
    position: absolute;
    right: 0;
    bottom: 0;
    left: 0;
    display: flex;
    height: 4px;

    &--haze {
      width: 50%;
      background-color: rgb(255 255 255 / 30%);
    }

    &--clear {
      width: 50%;
      background-color: var(--home-primary-400);
    }
  }
}

.showcase-caption {
  display: flex;
  gap: 12px;
  align-items: center;
  margin: 16px 4px 0;
  font-size: 14px;
  font-weight: 500;
  color: rgb(255 255 255 / 80%);

  .el-icon {
    flex-shrink: 0;
    font-size: 16px;
    color: var(--home-primary-300);
  }
}

.showcase-badge {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  padding: 4px 12px;
  margin-bottom: 20px;
  font-size: 14px;
  font-weight: 500;
  color: rgb(255 255 255 / 90%);
  background-color: rgb(255 255 255 / 10%);
  border: 1px solid rgb(255 255 255 / 20%);
  border-radius: 9999px;
  backdrop-filter: blur(4px);

  &__dot {
    width: 8px;
    height: 8px;
    background-color: var(--home-primary-300);
    border-radius: 50%;
  }

  &--center {
    margin-bottom: 16px;
  }
}

.showcase-title {
  margin-bottom: 12px;
  font-size: clamp(24px, 2.4vw, 34px);
  font-weight: 600;
  line-height: 1.25;
}

.showcase-desc {
  max-width: 28rem;
  margin-bottom: 32px;
  color: rgb(255 255 255 / 75%);
}

.metric-card {
  padding: 20px;
  background-color: rgb(255 255 255 / 5%);
  border: 1px solid rgb(255 255 255 / 10%);
  border-radius: 12px;
  backdrop-filter: blur(4px);
  transition: background-color 0.2s;

  &:hover {
    background-color: rgb(255 255 255 / 8%);
  }

  &__value {
    display: flex;
    gap: 4px;
    align-items: baseline;
    margin-bottom: 6px;
  }

  &__num {
    font-size: 36px;
    font-weight: 700;
    color: #fff;

    &--primary {
      color: var(--home-primary-300);
    }
  }

  &__unit {
    font-size: 18px;
    font-weight: 600;
    color: var(--home-primary-300);

    &--primary {
      color: rgb(255 255 255 / 80%);
    }
  }

  &__label {
    font-size: 14px;
    font-weight: 500;
    color: rgb(255 255 255 / 65%);
  }
}

.showcase-feedback {
  padding-top: 48px;
  margin-top: 56px;
  border-top: 1px solid rgb(255 255 255 / 10%);

  &__head {
    margin-bottom: 40px;
    text-align: center;
  }

  &__title {
    font-size: clamp(18px, 1.8vw, 24px);
    font-weight: 600;
  }
}

.showcase-orgs {
  display: flex;
  flex-wrap: wrap;
  gap: 24px 40px;
  align-items: center;
  justify-content: center;
  margin-bottom: 48px;
  opacity: 0.7;

  &__item {
    font-size: 18px;
    font-weight: 600;
    color: rgb(255 255 255 / 80%);
    letter-spacing: 0.05em;
  }

  &__divider {
    width: 1px;
    height: 20px;
    background-color: rgb(255 255 255 / 15%);
  }
}

.testimonial-card {
  padding: 24px;
  background-color: rgb(255 255 255 / 5%);
  border: 1px solid rgb(255 255 255 / 10%);
  border-radius: 12px;
  backdrop-filter: blur(4px);

  &__quote {
    margin-bottom: 16px;
    font-size: 32px;
    color: var(--home-primary-300);
  }

  &__text {
    margin-bottom: 20px;
    font-size: 14px;
    line-height: 1.6;
    color: rgb(255 255 255 / 85%);
  }

  &__author {
    display: flex;
    gap: 12px;
    align-items: center;
    padding-top: 16px;
    border-top: 1px solid rgb(255 255 255 / 10%);
  }

  &__avatar {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    font-size: 14px;
    font-weight: 700;
    color: #fff;
    border-radius: 50%;
  }

  &__name {
    font-size: 14px;
    font-weight: 600;
    color: #fff;
  }

  &__org {
    margin-top: 2px;
    font-size: 12px;
    color: rgb(255 255 255 / 55%);
  }
}
</style>
