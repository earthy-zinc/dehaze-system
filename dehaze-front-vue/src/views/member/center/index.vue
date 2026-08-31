<template>
  <div class="member-center">
    <!-- 顶部等级卡片 -->
    <div
      v-loading="loading"
      class="level-card"
      :style="{ background: levelBackground }"
    >
      <div class="level-info">
        <div class="level-icon">
          <el-icon :size="56"><Trophy /></el-icon>
        </div>
        <div class="level-detail">
          <div class="level-name">{{ profile?.levelName || "未开通会员" }}</div>
          <div class="level-meta">
            <template v-if="profile">
              <span v-if="profile.expireTime" class="meta-item">
                <el-icon><Calendar /></el-icon>
                到期时间：{{ profile.expireTime }}
              </span>
              <span v-else class="meta-item growth-maintain">
                <el-icon><Star /></el-icon>
                成长值维持
              </span>
              <span class="meta-item">
                <el-icon><ArrowUp /></el-icon>
                成长值：{{ profile.growthValue }}
              </span>
            </template>
          </div>
        </div>
      </div>
      <div class="level-actions">
        <el-button plain @click="router.push('/member/growth-logs')">
          <el-icon><View /></el-icon>
          成长值明细
        </el-button>
        <el-button
          v-if="!isMaxLevel && profile"
          type="primary"
          @click="handleUpgrade"
        >
          <el-icon><ArrowUp /></el-icon>
          升级
        </el-button>
      </div>
    </div>

    <!-- 成长值进度条 -->
    <el-card v-if="profile" class="growth-card" shadow="never">
      <div class="growth-header">
        <span class="growth-title">成长值进度</span>
        <span class="growth-current">{{ profile.growthValue }}</span>
      </div>
      <template v-if="!isMaxLevel">
        <el-progress
          :percentage="profile.progressPercent"
          :stroke-width="14"
          :color="levelColor"
          :text-inside="true"
        />
        <div class="growth-footer">
          距下一等级还需
          <strong class="growth-gap">{{ growthToNext }}</strong>
          成长值
        </div>
      </template>
      <div v-else class="max-level-text">
        <el-icon><Trophy /></el-icon>
        已达最高等级
      </div>
    </el-card>

    <!-- 权益概览 -->
    <el-card v-if="profile" class="benefit-card" shadow="never">
      <template #header>
        <span class="card-title">权益概览（本月）</span>
      </template>
      <el-row :gutter="20">
        <el-col :sm="6" :xs="12">
          <div class="benefit-item">
            <div class="benefit-label">图像处理剩余</div>
            <div class="benefit-value">{{ imageRemaining }}</div>
            <div class="benefit-sub">7 类图像任务取最低剩余</div>
          </div>
        </el-col>
        <el-col :sm="6" :xs="12">
          <div class="benefit-item">
            <div class="benefit-label">评估剩余</div>
            <div class="benefit-value">{{ evaluateRemaining }}</div>
            <div class="benefit-sub">本月剩余评估次数</div>
          </div>
        </el-col>
        <el-col :sm="6" :xs="12">
          <div class="benefit-item">
            <div class="benefit-label">批量上限</div>
            <div class="benefit-value">
              {{ profile.benefits?.batchLimit ?? 0 }}
            </div>
            <div class="benefit-sub">单次批量处理数量</div>
          </div>
        </el-col>
        <el-col :sm="6" :xs="12">
          <div class="benefit-item">
            <div class="benefit-label">历史保留</div>
            <div class="benefit-value">
              {{ profile.benefits?.historyRetention ?? 0 }}
            </div>
            <div class="benefit-sub">历史记录保留天数</div>
          </div>
        </el-col>
      </el-row>

      <div class="unlocked-features">
        <div class="features-title">已解锁功能</div>
        <div class="features-list">
          <el-tag
            v-if="profile.benefits?.hdExport"
            type="success"
            effect="light"
            round
          >
            <el-icon><Star /></el-icon>高清导出
          </el-tag>
          <el-tag
            v-if="profile.benefits?.reportExport"
            type="success"
            effect="light"
            round
          >
            <el-icon><Star /></el-icon>报告导出
          </el-tag>
          <el-tag
            v-if="profile.benefits?.batchDownload"
            type="success"
            effect="light"
            round
          >
            <el-icon><Star /></el-icon>批量下载
          </el-tag>
          <el-tag
            v-if="profile.benefits?.advancedParams"
            type="success"
            effect="light"
            round
          >
            <el-icon><Star /></el-icon>高级参数
          </el-tag>
          <el-tag
            v-if="!hasAnyUnlockedFeature"
            type="info"
            effect="plain"
            round
          >
            暂无解锁功能
          </el-tag>
        </div>
      </div>
    </el-card>

    <!-- 签到模块 -->
    <el-card class="signin-card" shadow="never">
      <template #header>
        <div class="signin-header">
          <span class="card-title">每日签到</span>
          <el-button
            type="primary"
            :loading="signInLoading"
            :disabled="hasSignedToday"
            @click="handleSignIn"
          >
            <el-icon><Calendar /></el-icon>
            {{ hasSignedToday ? "今日已签到" : "立即签到" }}
          </el-button>
        </div>
      </template>

      <div class="signin-stats">
        <div class="stat-item">
          <span class="stat-label">连续签到</span>
          <span class="stat-value">{{ calendar?.continuousDays ?? 0 }} 天</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">累计签到</span>
          <span class="stat-value">{{ calendar?.totalDays ?? 0 }} 天</span>
        </div>
      </div>

      <el-calendar v-model="calendarDate">
        <template #date-cell="{ data }">
          <div
            :class="[
              'calendar-cell',
              { signed: isSignedDate(data.day), today: isToday(data.day) },
            ]"
          >
            <span class="cell-day">{{
              Number(data.day.split("-").pop())
            }}</span>
            <el-icon
              v-if="isSignedDate(data.day)"
              class="signed-icon"
              :size="14"
            >
              <Star />
            </el-icon>
          </div>
        </template>
      </el-calendar>
    </el-card>

    <!-- 签到成功动画提示 -->
    <transition name="bonus-float">
      <div v-if="bonusVisible" class="bonus-tip">
        <el-icon><Star /></el-icon>
        <span>+{{ bonusGrowth }} 成长值</span>
      </div>
    </transition>
  </div>
</template>

<script lang="ts" setup>
import {
  MemberAPI,
  MemberProfileVO,
  BenefitSummaryVO,
  SignInCalendarVO,
  MemberLevelCode,
} from "dehaze-sdk-js";
import { Trophy, Star, Calendar, ArrowUp, View } from "@element-plus/icons-vue";

defineOptions({
  name: "MemberCenter",
  inheritAttrs: false,
});

const router = useRouter();

const loading = ref(false);
const profile = ref<MemberProfileVO>();
const summary = ref<BenefitSummaryVO>();
const calendar = ref<SignInCalendarVO>();
const calendarDate = ref(new Date());
const signInLoading = ref(false);
const bonusVisible = ref(false);
const bonusGrowth = ref(0);

const levelColorMap: Record<MemberLevelCode, string> = {
  level_0: "#8c8c8c",
  level_1: "#409eff",
  level_2: "#722ed1",
  level_3: "#fa8c16",
};

const levelGradientMap: Record<MemberLevelCode, string> = {
  level_0: "linear-gradient(135deg, #8c8c8c 0%, #595959 100%)",
  level_1: "linear-gradient(135deg, #409eff 0%, #1677ff 100%)",
  level_2: "linear-gradient(135deg, #722ed1 0%, #531dab 100%)",
  level_3: "linear-gradient(135deg, #fa8c16 0%, #d4380d 100%)",
};

const levelColor = computed(() => {
  return profile.value ? levelColorMap[profile.value.levelCode] : "#8c8c8c";
});

const levelBackground = computed(() => {
  return profile.value
    ? levelGradientMap[profile.value.levelCode]
    : levelGradientMap.level_0;
});

const isMaxLevel = computed(() => {
  return profile.value?.levelCode === "level_3";
});

const growthToNext = computed(() => {
  if (!profile.value || !profile.value.nextLevelGrowth) return 0;
  return Math.max(0, profile.value.nextLevelGrowth - profile.value.growthValue);
});

const imageRemaining = computed(
  () => summary.value?.imageCategory?.remaining ?? 0
);

const evaluateRemaining = computed(
  () => summary.value?.evaluateCategory?.remaining ?? 0
);

const hasAnyUnlockedFeature = computed(() => {
  if (!profile.value?.benefits) return false;
  const b = profile.value.benefits;
  return !!(
    b.hdExport ||
    b.reportExport ||
    b.batchDownload ||
    b.advancedParams
  );
});

const todayStr = computed(() => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
});

const hasSignedToday = computed(() => {
  if (!calendar.value?.signDates) return false;
  return calendar.value.signDates.includes(todayStr.value);
});

function isSignedDate(day: string) {
  return calendar.value?.signDates?.includes(day) ?? false;
}

function isToday(day: string) {
  return day === todayStr.value;
}

function loadProfile() {
  loading.value = true;
  Promise.all([MemberAPI.getProfile(), MemberAPI.getBenefitSummary()])
    .then(([profileData, summaryData]) => {
      profile.value = profileData;
      summary.value = summaryData;
    })
    .finally(() => {
      loading.value = false;
    });
}

function loadCalendar(year: number, month: number) {
  MemberAPI.getSignInCalendar(year, month).then((data) => {
    calendar.value = data;
  });
}

watch(calendarDate, (val) => {
  loadCalendar(val.getFullYear(), val.getMonth() + 1);
});

function handleSignIn() {
  signInLoading.value = true;
  MemberAPI.signIn()
    .then((res) => {
      bonusGrowth.value = res.growthValue + res.bonusGrowth;
      bonusVisible.value = true;
      setTimeout(() => {
        bonusVisible.value = false;
      }, 2000);
      ElMessage.success(
        `签到成功！连续签到 ${res.continuousDays} 天，获得 ${bonusGrowth.value} 成长值`
      );
      loadProfile();
      loadCalendar(
        calendarDate.value.getFullYear(),
        calendarDate.value.getMonth() + 1
      );
    })
    .finally(() => {
      signInLoading.value = false;
    });
}

function handleUpgrade() {
  ElMessage.info("升级功能即将开放，敬请期待");
}

onMounted(() => {
  loadProfile();
  loadCalendar(
    calendarDate.value.getFullYear(),
    calendarDate.value.getMonth() + 1
  );
});

onActivated(() => {
  loadProfile();
  loadCalendar(
    calendarDate.value.getFullYear(),
    calendarDate.value.getMonth() + 1
  );
});
</script>

<style lang="scss" scoped>
.member-center {
  max-width: 960px;
  padding: 24px 20px 40px;
  margin: 0 auto;
}

/* 等级卡片 */
.level-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 28px 32px;
  margin-bottom: 20px;
  color: #fff;
  border-radius: 16px;
  box-shadow: 0 8px 24px rgb(0 0 0 / 12%);

  .level-info {
    display: flex;
    gap: 20px;
    align-items: center;
  }

  .level-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 80px;
    height: 80px;
    background: rgb(255 255 255 / 18%);
    border-radius: 50%;
    backdrop-filter: blur(4px);
  }

  .level-detail {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .level-name {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 1px;
    text-shadow: 0 2px 8px rgb(0 0 0 / 15%);
  }

  .level-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    font-size: 13px;
    color: rgb(255 255 255 / 92%);

    .meta-item {
      display: inline-flex;
      gap: 4px;
      align-items: center;
    }

    .growth-maintain {
      color: rgb(255 255 255 / 75%);
    }
  }

  .level-actions {
    display: flex;
    gap: 10px;

    :deep(.el-button) {
      backdrop-filter: blur(4px);
    }

    :deep(.el-button--plain) {
      color: #fff;
      background: rgb(255 255 255 / 18%);
      border-color: rgb(255 255 255 / 35%);

      &:hover {
        color: #fff;
        background: rgb(255 255 255 / 28%);
        border-color: rgb(255 255 255 / 50%);
      }
    }
  }
}

/* 成长值卡片 */
.growth-card {
  margin-bottom: 20px;
  border-radius: 12px;

  .growth-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .growth-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  .growth-current {
    font-size: 24px;
    font-weight: 700;
    color: var(--el-color-primary);
  }

  .growth-footer {
    margin-top: 12px;
    font-size: 13px;
    color: var(--el-text-color-secondary);

    .growth-gap {
      margin: 0 4px;
      font-size: 15px;
      color: var(--el-color-primary);
    }
  }

  .max-level-text {
    display: flex;
    gap: 8px;
    align-items: center;
    justify-content: center;
    padding: 12px 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--el-color-warning);
  }
}

/* 权益概览卡片 */
.benefit-card {
  margin-bottom: 20px;
  border-radius: 12px;

  .card-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  .benefit-item {
    padding: 16px 12px;
    text-align: center;
    background: var(--el-fill-color-light);
    border-radius: 10px;
    transition: all 0.2s ease;

    &:hover {
      box-shadow: 0 4px 12px rgb(0 0 0 / 6%);
      transform: translateY(-2px);
    }

    .benefit-label {
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--el-text-color-secondary);
    }

    .benefit-value {
      font-size: 26px;
      font-weight: 700;
      line-height: 1.2;
      color: var(--el-color-primary);
    }

    .benefit-sub {
      margin-top: 6px;
      font-size: 12px;
      color: var(--el-text-color-placeholder);
    }
  }

  .unlocked-features {
    padding-top: 18px;
    margin-top: 18px;
    border-top: 1px dashed var(--el-border-color);

    .features-title {
      margin-bottom: 10px;
      font-size: 13px;
      color: var(--el-text-color-secondary);
    }

    .features-list {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;

      .el-tag {
        display: inline-flex;
        gap: 4px;
        align-items: center;
      }
    }
  }
}

/* 签到卡片 */
.signin-card {
  border-radius: 12px;

  .signin-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .card-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  .signin-stats {
    display: flex;
    gap: 32px;
    padding: 12px 16px;
    margin-bottom: 16px;
    background: var(--el-fill-color-light);
    border-radius: 8px;

    .stat-item {
      display: flex;
      flex-direction: column;
      gap: 4px;

      .stat-label {
        font-size: 12px;
        color: var(--el-text-color-secondary);
      }

      .stat-value {
        font-size: 18px;
        font-weight: 600;
        color: var(--el-color-primary);
      }
    }
  }

  :deep(.el-calendar) {
    --el-calendar-cell-width: 60px;

    .el-calendar__header {
      padding-top: 8px;
    }

    .el-calendar-table .el-calendar-day {
      height: 64px;
      padding: 4px;
    }
  }

  .calendar-cell {
    display: flex;
    flex-direction: column;
    gap: 2px;
    align-items: center;
    justify-content: center;
    height: 100%;

    .cell-day {
      font-size: 14px;
      color: var(--el-text-color-primary);
    }

    .signed-icon {
      color: var(--el-color-warning);
    }

    &.signed {
      .cell-day {
        font-weight: 600;
        color: var(--el-color-warning);
      }
    }

    &.today {
      .cell-day {
        font-weight: 700;
        color: var(--el-color-primary);
      }
    }
  }
}

/* 签到成功动画提示 */
.bonus-tip {
  position: fixed;
  top: 50%;
  left: 50%;
  z-index: 2000;
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 16px 28px;
  font-size: 22px;
  font-weight: 700;
  color: #fff;
  pointer-events: none;
  background: linear-gradient(135deg, #fa8c16, #d4380d);
  border-radius: 50px;
  box-shadow: 0 8px 32px rgb(250 140 22 / 40%);
  transform: translate(-50%, -50%);
}

.bonus-float-enter-active {
  animation: bonus-pop 0.4s ease-out;
}

.bonus-float-leave-active {
  animation: bonus-fade 0.6s ease-in forwards;
}

@keyframes bonus-pop {
  0% {
    opacity: 0;
    transform: translate(-50%, -30%) scale(0.5);
  }

  60% {
    transform: translate(-50%, -50%) scale(1.1);
  }

  100% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes bonus-fade {
  0% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }

  100% {
    opacity: 0;
    transform: translate(-50%, -120%) scale(1);
  }
}

@media (width <= 768px) {
  .level-card {
    flex-direction: column;
    gap: 18px;
    padding: 24px 20px;
    text-align: center;

    .level-info {
      flex-direction: column;
    }

    .level-meta {
      justify-content: center;
    }
  }

  .signin-stats {
    flex-wrap: wrap;
    gap: 16px !important;
  }
}
</style>
