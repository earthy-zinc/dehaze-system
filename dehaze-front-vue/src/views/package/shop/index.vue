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

      <!-- 商品分类 Tab -->
      <div class="type-tabs">
        <div
          :class="['type-tab', { active: activeTab === 'vip' }]"
          @click="activeTab = 'vip'"
        >
          会员卡
        </div>
        <div
          :class="['type-tab', { active: activeTab === 'credit' }]"
          @click="activeTab = 'credit'"
        >
          积分卡
        </div>
      </div>

      <!-- 套餐卡片列表 -->
      <div v-loading="loading" class="package-cards">
        <template v-if="filteredPackages.length > 0">
          <div class="cards-row">
            <div
              v-for="pkg in filteredPackages"
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
                    <div class="level-name">
                      {{
                        pkg.packageType === "credit" ? "积分卡" : pkg.levelName
                      }}
                    </div>
                    <div class="package-name">{{ pkg.name }}</div>
                  </div>
                  <div
                    v-if="pkg.levelCode === currentLevelCode"
                    class="current-badge"
                  >
                    当前
                  </div>
                </div>

                <!-- 积分卡：可得积分 + 单价提示 -->
                <template v-if="pkg.packageType === 'credit'">
                  <div class="price-section">
                    <div class="credit-amount">
                      {{ pkg.creditAmount ?? 0 }}
                      <span class="credit-unit">积分</span>
                    </div>
                    <div class="credit-unit-price">
                      ¥{{ yuan(pkg.creditUnitPrice) }}/积分
                    </div>
                    <div class="price-row">
                      <span class="currency">¥</span>
                      <span class="price-num">{{ yuan(pkg.salePrice) }}</span>
                      <span class="original-price">
                        原价 ¥{{ yuan(pkg.originalPrice) }}
                      </span>
                    </div>
                  </div>
                </template>

                <!-- 会员卡：价格 + 日均价 + 权益 -->
                <template v-else>
                  <div class="price-section">
                    <div class="sale-price">
                      <span class="currency">¥</span>
                      <span class="price-num">{{ yuan(pkg.salePrice) }}</span>
                    </div>
                    <div class="original-price">
                      原价 ¥{{ yuan(pkg.originalPrice) }}
                    </div>
                    <div class="daily-price">
                      ¥{{ yuan(pkg.dailyPrice) }}/天 ·
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
                </template>

                <div v-if="pkg.description" class="package-desc">
                  {{ pkg.description }}
                </div>

                <el-button
                  :type="buttonType(pkg)"
                  class="action-btn"
                  @click="handlePurchase(pkg)"
                >
                  {{ buttonText(pkg) }}
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

      <!-- 权益对比表（会员卡区） -->
      <div v-if="comparisonPackages.length > 0" class="comparison-section">
        <h3 class="section-title">权益对比</h3>
        <el-table :data="comparisonRows" border class="comparison-table">
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
                :class="[
                  'compare-cell',
                  { highlight: pkg.levelCode === currentLevelCode },
                ]"
              >
                {{ scope.row[`pkg_${pkg.id}`] ?? "—" }}
              </span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <!-- 购买确认弹窗：价格明细 + 优惠券选择 -->
    <el-dialog
      v-model="purchaseDialogVisible"
      title="购买确认"
      width="480px"
      :close-on-click-modal="false"
    >
      <div v-loading="priceLoading" class="purchase-detail">
        <div class="detail-row">
          <span class="detail-label">商品</span>
          <span class="detail-value">{{ currentPackage?.name }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">原价</span>
          <span class="detail-value"
            >¥{{ yuan(purchasePrice?.originalPrice) }}</span
          >
        </div>
        <div class="detail-row">
          <span class="detail-label">促销优惠</span>
          <span class="detail-value discount">
            -¥{{ yuan(purchasePrice?.discountAmount) }}
          </span>
        </div>
        <div class="detail-row">
          <span class="detail-label">优惠券抵扣</span>
          <span class="detail-value discount">
            {{
              purchasePrice?.couponAmount
                ? `-¥${yuan(purchasePrice.couponAmount)}`
                : "—"
            }}
          </span>
        </div>
        <div class="detail-row coupon-select-row">
          <span class="detail-label">优惠券</span>
          <el-select
            v-model="selectedCouponId"
            class="coupon-select"
            placeholder="不使用优惠券"
            clearable
            :disabled="couponLoading"
            @change="recalcPrice"
          >
            <el-option
              v-for="coupon in applicableCoupons"
              :key="coupon.id"
              :label="couponOptionLabel(coupon)"
              :value="coupon.id"
            />
          </el-select>
        </div>
        <div class="detail-row total">
          <span class="detail-label">应付</span>
          <span class="detail-value payable"
            >¥{{ yuan(purchasePrice?.payableAmount) }}</span
          >
        </div>
      </div>
      <template #footer>
        <el-button @click="purchaseDialogVisible = false">取消</el-button>
        <el-button
          type="primary"
          :loading="purchasing"
          :disabled="priceLoading"
          @click="confirmPurchase"
        >
          确认购买
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  PackageAPI,
  CouponAPI,
  OrderAPI,
  type PackageDetailVO,
  type PriceResult,
  type UserCouponVO,
} from "dehaze-sdk-js";
import { useMemberStoreHook } from "@/store";
import { Ticket, Close, Check, ArrowRight } from "@element-plus/icons-vue";
import { periodOptions } from "../constants";

defineOptions({
  name: "PackageShop",
  inheritAttrs: false,
});

const router = useRouter();
const loading = ref(false);
const bannerVisible = ref(true);
const activeTab = ref<"vip" | "credit">("vip");

/** 购买弹窗状态 */
const purchaseDialogVisible = ref(false);
const currentPackage = ref<PackageDetailVO | null>(null);
const purchasePrice = ref<PriceResult | null>(null);
const priceLoading = ref(false);
const couponLoading = ref(false);
const purchasing = ref(false);
const myCoupons = ref<UserCouponVO[]>([]);
const selectedCouponId = ref<number | undefined>(undefined);

const packages = ref<PackageDetailVO[]>([]);
const memberStore = useMemberStoreHook();
const profile = computed(() => memberStore.profile);

const currentLevelCode = computed(() => profile.value?.levelCode ?? "level_0");

/** 后端金额单位为分，前端展示用元 */
function yuan(cents?: number) {
  return ((cents ?? 0) / 100).toFixed(2);
}

const filteredPackages = computed(() =>
  packages.value.filter((pkg) => pkg.packageType === activeTab.value)
);

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

function periodLabel(period?: string) {
  return (period && periodOptions.find((o) => o.value === period)?.label) || "";
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

function buttonText(pkg: PackageDetailVO) {
  if (pkg.packageType === "credit") {
    return "立即购买";
  }
  const current = levelOrder[currentLevelCode.value] ?? 0;
  const target = levelOrder[pkg.levelCode ?? ""] ?? 0;
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

function buttonType(pkg: PackageDetailVO) {
  if (pkg.levelCode === "level_3") return "warning";
  if (pkg.levelCode === "level_2") return "primary";
  return "success";
}

const comparisonPackages = computed(() => {
  const levelMap = new Map<string, PackageDetailVO>();
  packages.value
    .filter((pkg) => pkg.packageType === "vip")
    .forEach((pkg) => {
      if (pkg.levelCode && !levelMap.has(pkg.levelCode)) {
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
    };
    comparisonPackages.value.forEach((pkg) => {
      const val = pkg.benefits[key];
      row[`pkg_${pkg.id}`] =
        val !== undefined ? formatBenefitValue(key, Number(val)) : "—";
    });
    return row;
  });
});

/** 用户未使用优惠券中适用于当前套餐的（applicableScope 为空表示全部适用） */
const applicableCoupons = computed(() => {
  const pkg = currentPackage.value;
  if (!pkg) return [];
  return myCoupons.value.filter((coupon) => {
    const scope = coupon.applicableScope;
    if (!scope || scope.length === 0) return true;
    return scope.includes(pkg.id) || scope.includes(pkg.packageType);
  });
});

function couponOptionLabel(coupon: UserCouponVO) {
  const threshold =
    coupon.threshold && coupon.threshold > 0
      ? `（满¥${yuan(coupon.threshold)}可用）`
      : "";
  return `${coupon.couponName}${threshold}`;
}

function recalcPrice() {
  const pkg = currentPackage.value;
  if (!pkg) return Promise.resolve();
  priceLoading.value = true;
  return PackageAPI.calculatePrice(pkg.id, selectedCouponId.value)
    .then((res) => {
      purchasePrice.value = res;
    })
    .catch(() => {
      purchasePrice.value = null;
    })
    .finally(() => {
      priceLoading.value = false;
    });
}

function handlePurchase(pkg: PackageDetailVO) {
  currentPackage.value = pkg;
  selectedCouponId.value = undefined;
  purchasePrice.value = null;
  myCoupons.value = [];
  purchaseDialogVisible.value = true;
  couponLoading.value = true;
  CouponAPI.listMy(1)
    .then((coupons) => {
      myCoupons.value = coupons;
    })
    .catch(() => {
      myCoupons.value = [];
    })
    .finally(() => {
      couponLoading.value = false;
    });
  recalcPrice();
}

function confirmPurchase() {
  const pkg = currentPackage.value;
  if (!pkg || priceLoading.value || purchasing.value) return;
  purchasing.value = true;
  OrderAPI.create({
    packageId: pkg.id,
    couponId: selectedCouponId.value,
    payMethod: "balance",
  })
    .then((res) => {
      ElMessage.success("订单创建成功");
      purchaseDialogVisible.value = false;
      router.push(`/order/detail?orderNo=${res.orderNo}`);
    })
    .catch(() => {})
    .finally(() => {
      purchasing.value = false;
    });
}

function loadData() {
  loading.value = true;
  Promise.all([
    PackageAPI.listOnSale(),
    memberStore.loadProfile().catch(() => undefined),
  ])
    .then(([list]) => {
      packages.value = list;
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

.type-tabs {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin-bottom: 28px;

  .type-tab {
    padding: 8px 28px;
    font-size: 15px;
    color: var(--el-text-color-regular);
    cursor: pointer;
    background: #fff;
    border: 1px solid var(--el-border-color-lighter);
    border-radius: 20px;
    transition: all 0.2s;

    &.active {
      font-weight: 600;
      color: #fff;
      background: var(--el-color-primary);
      border-color: var(--el-color-primary);
    }
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

  .credit-amount {
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    color: var(--el-color-danger);

    .credit-unit {
      font-size: 16px;
      font-weight: 500;
    }
  }

  .credit-unit-price {
    margin-top: 6px;
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  .price-row {
    display: flex;
    gap: 10px;
    align-items: baseline;
    margin-top: 12px;
    color: var(--el-color-danger);

    .currency {
      font-size: 18px;
      font-weight: 500;
    }

    .price-num {
      font-size: 28px;
      font-weight: 700;
      line-height: 1;
    }

    .original-price {
      font-size: 13px;
      color: var(--el-text-color-secondary);
      text-decoration: line-through;
    }
  }

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
  margin-top: auto;
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

.purchase-detail {
  min-height: 120px;

  .detail-row {
    display: flex;
    gap: 12px;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;

    .detail-label {
      color: var(--el-text-color-secondary);
    }

    .detail-value.discount {
      color: var(--el-color-success);
    }

    &.coupon-select-row {
      justify-content: flex-start;

      .coupon-select {
        width: 260px;
      }
    }

    &.total {
      padding-top: 12px;
      margin-top: 4px;
      border-top: 1px dashed var(--el-border-color-lighter);

      .detail-label {
        font-weight: 600;
        color: var(--el-text-color-primary);
      }

      .detail-value.payable {
        font-size: 22px;
        font-weight: 700;
        color: var(--el-color-danger);
      }
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
