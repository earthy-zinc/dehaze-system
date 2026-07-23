<template>
  <view class="hero-section">
    <!-- 装饰圆形 -->
    <view class="hero-decoration hero-decoration-1" />
    <view class="hero-decoration hero-decoration-2" />
    <view class="hero-decoration hero-decoration-3" />

    <view class="hero-content">
      <view class="hero-badge">
        <text class="badge-dot" />
        <text class="badge-text">AI 智能去雾引擎</text>
      </view>

      <text class="hero-title">图像去雾</text>
      <text class="hero-subtitle">专业级图像处理系统</text>
      <text class="hero-description">
        采用先进的深度学习算法，一键还原清晰视界
        从图像输入到效果评估的完整闭环体验
      </text>

      <view class="hero-cta">
        <view class="cta-primary" @click="handlePrimaryClick">
          <text class="cta-primary-text">立即开始</text>
          <u-icon name="arrow-right" size="16" color="#ffffff" />
        </view>
        <view class="cta-secondary" @click="handleSecondaryClick">
          <text class="cta-secondary-text">浏览数据集</text>
        </view>
      </view>

      <!-- 数据指标 -->
      <view class="hero-stats">
        <view class="stat-item">
          <text class="stat-value">{{ algorithmCountDisplay }}</text>
          <text class="stat-label">去雾算法</text>
        </view>
        <view class="stat-divider" />
        <view class="stat-item">
          <text class="stat-value">GPU</text>
          <text class="stat-label">推理加速</text>
        </view>
        <view class="stat-divider" />
        <view class="stat-item">
          <text class="stat-value">4项</text>
          <text class="stat-label">评估指标</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { computed } from "vue";

interface Props {
  algorithmCount?: number | string;
}

interface Emits {
  (e: "primary-click"): void;
  (e: "secondary-click"): void;
}

const props = withDefaults(defineProps<Props>(), {
  algorithmCount: 0,
});

const emit = defineEmits<Emits>();

/** 算法数量展示：字符串原样显示，数字按 `${n}+` 格式（0 显示 "--"） */
const algorithmCountDisplay = computed(() => {
  const v = props.algorithmCount;
  if (typeof v === "string") return v;
  return v > 0 ? `${v}+` : "--";
});

const handlePrimaryClick = () => {
  emit("primary-click");
};

const handleSecondaryClick = () => {
  emit("secondary-click");
};
</script>

<style lang="scss" scoped>
$brand-primary: #3b82f6;
$brand-secondary: #6366f1;
$brand-gradient: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);

.hero-section {
  position: relative;
  padding: 120rpx 40rpx 100rpx;
  text-align: center;
  width: 100%;
  background: $brand-gradient;
  overflow: hidden;
}

/* 装饰圆形 */
.hero-decoration {
  position: absolute;
  border-radius: 50%;
  pointer-events: none;
}

.hero-decoration-1 {
  width: 480rpx;
  height: 480rpx;
  background: rgba(255, 255, 255, 0.08);
  top: -160rpx;
  right: -120rpx;
}

.hero-decoration-2 {
  width: 320rpx;
  height: 320rpx;
  background: rgba(255, 255, 255, 0.06);
  bottom: -80rpx;
  left: -80rpx;
}

.hero-decoration-3 {
  width: 200rpx;
  height: 200rpx;
  background: rgba(255, 255, 255, 0.05);
  top: 40%;
  left: 10%;
}

.hero-content {
  position: relative;
  z-index: 1;
  max-width: 720rpx;
  margin: 0 auto;
}

/* 顶部徽章 */
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 12rpx;
  padding: 10rpx 24rpx;
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  border: 2rpx solid rgba(255, 255, 255, 0.25);
  border-radius: 100rpx;
  margin-bottom: 32rpx;
}

.badge-dot {
  width: 12rpx;
  height: 12rpx;
  background: #34d399;
  border-radius: 50%;
  box-shadow: 0 0 8rpx rgba(52, 211, 153, 0.8);
}

.badge-text {
  font-size: 22rpx;
  color: #ffffff;
  font-weight: 500;
  letter-spacing: 0.5rpx;
}

.hero-title {
  display: block;
  font-size: 88rpx;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #ffffff;
  margin-bottom: 16rpx;
  line-height: 1.05;
  text-shadow: 0 4rpx 24rpx rgba(0, 0, 0, 0.1);
}

.hero-subtitle {
  display: block;
  font-size: 36rpx;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.95);
  margin-bottom: 28rpx;
  letter-spacing: -0.01em;
}

.hero-description {
  display: block;
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.85);
  line-height: 1.7;
  max-width: 560rpx;
  margin: 0 auto 56rpx;
  white-space: pre-line;
}

/* CTA 按钮 */
.hero-cta {
  display: flex;
  gap: 24rpx;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 72rpx;
}

.cta-primary {
  display: inline-flex;
  align-items: center;
  gap: 12rpx;
  padding: 24rpx 56rpx;
  background: #ffffff;
  border-radius: 14rpx;
  box-shadow: 0 12rpx 32rpx rgba(0, 0, 0, 0.15);
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.97);
    box-shadow: 0 6rpx 16rpx rgba(0, 0, 0, 0.2);
  }
}

.cta-primary-text {
  font-size: 32rpx;
  font-weight: 700;
  color: $brand-primary;
}

.cta-secondary {
  display: inline-flex;
  align-items: center;
  padding: 24rpx 48rpx;
  background: rgba(255, 255, 255, 0.1);
  border: 2rpx solid rgba(255, 255, 255, 0.3);
  backdrop-filter: blur(10px);
  border-radius: 14rpx;
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.97);
    background: rgba(255, 255, 255, 0.2);
  }
}

.cta-secondary-text {
  font-size: 30rpx;
  font-weight: 600;
  color: #ffffff;
}

/* 数据指标 */
.hero-stats {
  display: inline-flex;
  align-items: center;
  gap: 32rpx;
  padding: 24rpx 40rpx;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 2rpx solid rgba(255, 255, 255, 0.15);
  border-radius: 20rpx;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4rpx;
}

.stat-value {
  font-size: 32rpx;
  font-weight: 700;
  color: #ffffff;
  line-height: 1.2;
}

.stat-label {
  font-size: 22rpx;
  color: rgba(255, 255, 255, 0.75);
}

.stat-divider {
  width: 2rpx;
  height: 36rpx;
  background: rgba(255, 255, 255, 0.2);
}

/* 响应式适配 */
@media screen and (max-width: 768rpx) {
  .hero-section {
    padding: 80rpx 32rpx 80rpx;
  }

  .hero-title {
    font-size: 64rpx;
  }

  .hero-subtitle {
    font-size: 28rpx;
  }

  .hero-description {
    font-size: 24rpx;
  }

  .hero-cta {
    flex-direction: column;
    align-items: stretch;
    gap: 16rpx;
  }

  .cta-primary,
  .cta-secondary {
    justify-content: center;
  }

  .hero-stats {
    gap: 20rpx;
    padding: 20rpx 28rpx;
  }

  .stat-value {
    font-size: 28rpx;
  }

  .stat-label {
    font-size: 20rpx;
  }
}
</style>
