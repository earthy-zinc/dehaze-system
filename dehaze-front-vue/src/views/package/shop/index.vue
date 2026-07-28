<template>
  <div class="package-shop">
    <div class="shop-container">
      <!-- 顶部促销横幅 -->
      <transition name="banner-fade">
        <div v-if="bannerVisible" class="promo-banner">
          <div class="banner-content">
            <el-icon class="banner-icon"><Ticket /></el-icon>
            <span class="banner-text">
              开通会员，解锁高清去雾、批量处理、专业评估等全部能力
            </span>
          </div>
          <el-icon class="banner-close" @click="bannerVisible = false">
            <Close />
          </el-icon>
        </div>
      </transition>

      <!-- 页面标题 -->
      <div class="page-header">
        <h2 class="header-title">会员套餐</h2>
        <p class="header-subtitle">选择适合您的套餐，开启专业去雾体验</p>
      </div>

      <!-- 套餐卡片列表 -->
      <div v-loading="loading" class="package-cards">
        <template v-if="packages.length > 0">
          <div class="cards-row">
            <div
              v-for="pkg in packages"
              :key="pkg.id"
              :class="['package-card', `level-${pkg.levelCode}`]"
            >
              <div class="card-glow"></div>
              <div class="card-inner">
                <div class="card-header">
                  <div :class="['level-icon', `level-icon-${pkg.levelCode}`]">
                    <el-icon><Check /></el-icon>
                  </div>
                  <div class="level-info">
                    <div class="level-name">{{ pkg.levelName }}</div>
                    <div class="package-name">{{ pkg.name }}</div>
                  </div>
                  <div
                    v-if="pkg.levelCode === currentLevelCode"
                    class="current-badge"
                  >
                    当前
                  </div>
                </div>

                <div class="price-section">
                  <div class="sale-price">
                    <span class="currency">¥</span>
                    <span class="price-num">
                      {{ pkg.salePrice.toFixed(2) }}
                    </span>
                  </div>
                  <div class="original-price">
                    原价 ¥{{ pkg.originalPrice.toFixed(2) }}
                  </div>
                  <div class="daily-price">
                    ¥{{ pkg.dailyPrice.toFixed(2) }}/天 ·
                    {{ periodLabel(pkg.period) }}
                  </div>
                </div>

                <div class="benefits-list">
                  <div
                    v-for="(value, key) in pkg.benefits"
                    :key="key"
                    class="benefit-item"
                  >
                    <el-icon class="benefit-check"><Check /></el-icon>
                    <span class="benefit-label">
                      {{ benefitLabel(String(key)) }}
                    </span>
                    <span class="benefit-value">
                      {{ formatBenefitValue(String(key), Number(value)) }}
                    </span>
                  </div>
                </div>

                <div v-if="pkg.description" class="package-desc">
                  {{ pkg.description }}
                </div>

                <el-button
                  :type="buttonType(pkg.levelCode)"
                  class="action-btn"
                  :loading="purchasingId === pkg.id"
                  @click="handlePurchase(pkg)"
                >
                  {{ buttonText(pkg.levelCode) }}
                  <el-icon class="btn-icon"><ArrowRight /></el-icon>
                </el-button>
              </div>
            </div>
          </div>
        </template>

        <el-empty
          v-else-if="!loading"
          description="暂无在售套餐"
          :image-size="120"
        />
      </div>

      <!-- 权益对比表 -->
      <div v-if="packages.length > 0" class="comparison-section">
        <h3 class="section-title">权益对比</h3>
        <el-table
          :data="comparisonRows"
          border
          :span-method="spanMethod"
          class="comparison-table"
        >
          <el-table-column label="权益项" prop="label" min-width="160" fixed />
          <el-table-column
            v-for="pkg in comparisonPackages"
            :key="pkg.id"
            :label="pkg.levelName"
            align="center"
            min-width="140"
          >
            <template #default="scope">
              <span
                :class="['compare-cell', { highlight: scope.row.isHighlight }]"
              >
                {{ scope.row[`pkg_${pkg.id}`] ?? "—" }}
              </span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import {
  PackageAPI,
  MemberAPI,
  OrderAPI,
  type PackageDetailVO,
  type MemberProfileVO,
} from "dehaze-sdk-js";
import { Ticket, Close, Check, ArrowRight } from "@element-plus/icons-vue";

defineOptions({
  name: "PackageShop",
  inheritAttrs: false,
});

const router = useRouter();
const loading = ref(false);
const purchasingId = ref<number>(0);
const bannerVisible = ref(true);

const packages = ref<PackageDetailVO[]>([]);
const profile = ref<MemberProfileVO>();

const currentLevelCode = computed(() => profile.value?.levelCode ?? "level_0");

const levelOrder: Record<string, number> = {
  level_0: 0,
  level_1: 1,
  level_2: 2,
  level_3: 3,
};

const benefitLabels: Record<string, string> = {
  monthlyDehazeQuota: "去雾配额",
  monthlyEvaluateQuota: "评估配额",
  historyRetention: "历史保留",
  batchLimit: "批量上限",
  priority: "优先级",
  advancedParams: "高级参数",
  hdExport: "高清导出",
  reportExport: "报告导出",
  batchDownload: "批量下载",
};

const benefitUnits: Record<string, string> = {
  monthlyDehazeQuota: "次/月",
  monthlyEvaluateQuota: "次/月",
  historyRetention: "天",
  batchLimit: "张",
  priority: "级",
  advancedParams: "项",
  hdExport: "次",
  reportExport: "次",
  batchDownload: "次",
};

function periodLabel(period: string) {
  const map: Record<string, string> = {
    monthly: "月卡",
    quarterly: "季卡",
    yearly: "年卡",
  };
  return map[period] ?? period;
}

function benefitLabel(key: string) {
  return benefitLabels[key] ?? key;
}

function formatBenefitValue(key: string, value: number) {
  const unit = benefitUnits[key];
  if (key === "historyRetention") {
    return value === 0 ? "—" : `${value} ${unit}`;
  }
  if (key === "hdExport" || key === "reportExport" || key === "batchDownload") {
    if (value === 0) return "不支持";
    if (value === 1) return "支持";
  }
  if (key === "priority" || key === "advancedParams") {
    return value === 1 ? "支持" : "—";
  }
  if (unit) {
    return value === 0 ? "—" : `${value} ${unit}`;
  }
  return String(value);
}

function buttonText(pkgLevel: string) {
  const current = levelOrder[currentLevelCode.value] ?? 0;
  const target = levelOrder[pkgLevel] ?? 0;
  if (currentLevelCode.value === "level_0") {
    return "立即开通";
  }
  if (target === current) {
    return "续费";
  }
  if (target > current) {
    return "升级至此";
  }
  return "续费";
}

function buttonType(pkgLevel: string) {
  if (pkgLevel === "level_3") return "warning";
  if (pkgLevel === "level_2") return "primary";
  return "success";
}

const comparisonPackages = computed(() => {
  const levelMap = new Map<string, PackageDetailVO>();
  packages.value.forEach((pkg) => {
    if (!levelMap.has(pkg.levelCode)) {
      levelMap.set(pkg.levelCode, pkg);
    }
  });
  const ordered: PackageDetailVO[] = [];
  ["level_1", "level_2", "level_3"].forEach((code) => {
    const pkg = levelMap.get(code);
    if (pkg) ordered.push(pkg);
  });
  return ordered;
});

const comparisonRows = computed(() => {
  const allKeys = new Set<string>();
  comparisonPackages.value.forEach((pkg) => {
    Object.keys(pkg.benefits).forEach((k) => allKeys.add(k));
  });
  return Array.from(allKeys).map((key) => {
    const row: Record<string, any> = {
      label: benefitLabels[key] ?? key,
      benefitKey: key,
      isHighlight: false,
    };
    comparisonPackages.value.forEach((pkg) => {
      const val = pkg.benefits[key];
      row[`pkg_${pkg.id}`] =
        val !== undefined ? formatBenefitValue(key, Number(val)) : "—";
    });
    return row;
  });
});

function spanMethod(_params: any) {
  return {
    rowspan: 1,
    colspan: 1,
  };
}

function handlePurchase(pkg: PackageDetailVO) {
  ElMessageBox.confirm(
    `确认开通「${pkg.name}」，将使用余额支付 ¥${pkg.salePrice.toFixed(2)}？`,
    "开通确认",
    {
      confirmButtonText: "确认开通",
      cancelButtonText: "取消",
      type: "info",
    }
  )
    .then(() => {
      purchasingId.value = pkg.id;
      return OrderAPI.create({
        packageId: pkg.id,
        payMethod: "balance",
      });
    })
    .then((res) => {
      ElMessage.success("订单创建成功");
      router.push(`/order/detail?orderNo=${res.orderNo}`);
    })
    .catch(() => {})
    .finally(() => {
      purchasingId.value = 0;
    });
}

function loadData() {
  loading.value = true;
  Promise.all([
    PackageAPI.listOnSale(),
    MemberAPI.getProfile().catch(() => undefined),
  ])
    .then(([list, profileData]) => {
      packages.value = list;
      if (profileData) {
        profile.value = profileData;
      }
    })
    .finally(() => {
      loading.value = false;
    });
}

onMounted(() => {
  loadData();
});
</script>

<style lang="scss" scoped>
.package-shop {
  min-height: calc(100vh - 60px);
  padding: 24px 20px 48px;
  background: linear-gradient(180deg, #f5f7fa 0%, #e8edf3 100%);
}

.shop-container {
  max-width: 1200px;
  margin: 0 auto;
}

.promo-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  margin-bottom: 24px;
  color: #fff;
  background: linear-gradient(90deg, #409eff 0%, #722ed1 100%);
  border-radius: 10px;
  box-shadow: 0 4px 16px rgb(64 158 255 / 25%);

  .banner-content {
    display: flex;
    gap: 10px;
    align-items: center;

    .banner-icon {
      font-size: 20px;
    }

    .banner-text {
      font-size: 14px;
      font-weight: 500;
    }
  }

  .banner-close {
    font-size: 16px;
    cursor: pointer;
    opacity: 0.85;
    transition: opacity 0.2s;

    &:hover {
      opacity: 1;
    }
  }
}

.banner-fade-enter-active,
.banner-fade-leave-active {
  transition: all 0.3s ease;
}

.banner-fade-enter-from,
.banner-fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

.page-header {
  margin-bottom: 32px;
  text-align: center;

  .header-title {
    margin: 0 0 8px;
    font-size: 28px;
    font-weight: 700;
    color: var(--el-text-color-primary);
    letter-spacing: 1px;
  }

  .header-subtitle {
    margin: 0;
    font-size: 14px;
    color: var(--el-text-color-secondary);
  }
}

.package-cards {
  min-height: 240px;
}

.cards-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  align-items: stretch;
}

.package-card {
  position: relative;
  overflow: hidden;
  border-radius: 16px;
  transition:
    transform 0.3s ease,
    box-shadow 0.3s ease;

  &:hover {
    box-shadow: 0 12px 32px rgb(0 0 0 / 12%);
    transform: translateY(-6px);

    .card-glow {
      opacity: 1;
    }
  }

  .card-glow {
    position: absolute;
    top: 0;
    right: 0;
    left: 0;
    height: 6px;
    opacity: 0.85;
    transition: opacity 0.3s;
  }

  &.level-level_1 .card-glow {
    background: linear-gradient(90deg, #409eff, #79bbff);
  }

  &.level-level_2 .card-glow {
    background: linear-gradient(90deg, #722ed1, #9254de);
  }

  &.level-level_3 .card-glow {
    background: linear-gradient(90deg, #fa8c16, #ffa940);
  }

  .card-inner {
    position: relative;
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 24px;
    background: #fff;
    border: 1px solid var(--el-border-color-lighter);
    border-top: none;
    border-radius: 0 0 16px 16px;
  }
}

.card-header {
  position: relative;
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 20px;

  .level-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    font-size: 22px;
    color: #fff;
    border-radius: 12px;

    &.level-icon-level_1 {
      background: linear-gradient(135deg, #409eff, #79bbff);
    }

    &.level-icon-level_2 {
      background: linear-gradient(135deg, #722ed1, #9254de);
    }

    &.level-icon-level_3 {
      background: linear-gradient(135deg, #fa8c16, #ffa940);
    }
  }

  .level-info {
    flex: 1;

    .level-name {
      font-size: 16px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }

    .package-name {
      margin-top: 2px;
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }

  .current-badge {
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 500;
    color: #fff;
    background: var(--el-color-success);
    border-radius: 10px;
  }
}

.price-section {
  padding-bottom: 16px;
  margin-bottom: 20px;
  border-bottom: 1px dashed var(--el-border-color-lighter);

  .sale-price {
    display: flex;
    align-items: baseline;
    color: var(--el-color-danger);

    .currency {
      font-size: 18px;
      font-weight: 500;
    }

    .price-num {
      margin-left: 2px;
      font-size: 36px;
      font-weight: 700;
      line-height: 1;
    }
  }

  .original-price {
    margin-top: 6px;
    font-size: 13px;
    color: var(--el-text-color-secondary);
    text-decoration: line-through;
  }

  .daily-price {
    margin-top: 4px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}

.benefits-list {
  flex: 1;
  margin-bottom: 16px;

  .benefit-item {
    display: flex;
    gap: 6px;
    align-items: center;
    padding: 6px 0;
    font-size: 13px;

    .benefit-check {
      font-size: 14px;
      color: var(--el-color-success);
    }

    .benefit-label {
      flex: 1;
      color: var(--el-text-color-regular);
    }

    .benefit-value {
      font-weight: 600;
      color: var(--el-text-color-primary);
    }
  }
}

.package-desc {
  padding: 10px 12px;
  margin-bottom: 16px;
  font-size: 12px;
  line-height: 1.5;
  color: var(--el-text-color-secondary);
  background: var(--el-fill-color-light);
  border-radius: 6px;
}

.action-btn {
  width: 100%;
  height: 42px;
  font-size: 15px;
  font-weight: 600;

  .btn-icon {
    margin-left: 4px;
  }
}

.comparison-section {
  margin-top: 48px;

  .section-title {
    margin: 0 0 16px;
    font-size: 20px;
    font-weight: 700;
    color: var(--el-text-color-primary);
    text-align: center;
  }
}

.comparison-table {
  .compare-cell {
    font-size: 13px;

    &.highlight {
      font-weight: 600;
      color: var(--el-color-primary);
    }
  }
}

@media (width <= 768px) {
  .cards-row {
    grid-template-columns: 1fr;
  }

  .page-header .header-title {
    font-size: 22px;
  }
}
</style>
