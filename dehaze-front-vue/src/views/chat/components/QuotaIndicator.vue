<!-- 配额提示条：日/月已用与限额，接近或达到限额、欠费时引导升级 -->
<script lang="ts" setup>
import { computed } from "vue";
import { useRouter } from "vue-router";
import { useChatUserStore } from "@/store/modules/chatUser";

defineOptions({ name: "QuotaIndicator" });

const router = useRouter();
const chatUserStore = useChatUserStore();

const quota = computed(() => chatUserStore.quota);

/** 限额为 0 视为不设限 */
function percent(used: number, limit: number) {
  if (limit <= 0) return 0;
  return Math.min(100, Math.round((used / limit) * 100));
}

const dailyPercent = computed(() =>
  quota.value ? percent(quota.value.dailyUsed, quota.value.dailyLimit) : 0
);
const monthlyPercent = computed(() =>
  quota.value ? percent(quota.value.monthlyUsed, quota.value.monthlyLimit) : 0
);

const alertText = computed(() => {
  if (!quota.value) return "";
  if (quota.value.arrearsStatus) return "积分余额已欠费，请前往充值或升级会员";
  if (dailyPercent.value >= 100 || monthlyPercent.value >= 100) {
    return "今日/本月配额已达上限，升级会员可提升限额";
  }
  if (dailyPercent.value >= 80) return "今日配额即将用尽，注意合理使用";
  return "";
});

function goUpgrade() {
  router.push("/package/shop");
}
</script>

<template>
  <div
    v-if="quota && (alertText || dailyPercent > 0 || monthlyPercent > 0)"
    class="quota-indicator"
    :class="{ 'quota-indicator--alert': alertText }"
  >
    <template v-if="alertText">
      <el-alert
        :title="alertText"
        type="warning"
        :closable="false"
        class="flex-1"
      >
        <el-button size="small" type="warning" @click="goUpgrade">
          升级会员
        </el-button>
      </el-alert>
    </template>
    <template v-else>
      <span class="quota-indicator__item">
        今日 {{ quota.dailyUsed }}/{{
          quota.dailyLimit > 0 ? quota.dailyLimit : "不限"
        }}
        积分
      </span>
      <el-progress
        :percentage="dailyPercent"
        :stroke-width="6"
        class="quota-indicator__bar"
      />
      <span class="quota-indicator__item">
        本月 {{ quota.monthlyUsed }}/{{
          quota.monthlyLimit > 0 ? quota.monthlyLimit : "不限"
        }}
        积分
      </span>
      <el-progress
        :percentage="monthlyPercent"
        :stroke-width="6"
        class="quota-indicator__bar"
      />
      <span class="quota-indicator__item"
        >余额 {{ quota.creditsBalance }} 积分</span
      >
    </template>
  </div>
</template>

<style scoped lang="scss">
.quota-indicator {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 6px 16px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  border-bottom: 1px solid var(--el-border-color-lighter);

  &--alert {
    padding: 6px 16px;
  }

  &__item {
    flex-shrink: 0;
  }

  &__bar {
    width: 120px;
  }
}

.flex-1 {
  flex: 1;
}
</style>
