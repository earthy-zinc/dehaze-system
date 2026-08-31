<!-- 余额不足/欠费/配额达限时的加购引导卡片 -->
<script lang="ts" setup>
import { Wallet } from "@element-plus/icons-vue";
import { computed } from "vue";
import { useRouter } from "vue-router";
import { RechargeAlert } from "@/store/modules/billing";

defineOptions({ name: "RechargeGuide" });

const props = defineProps<{ alert: RechargeAlert }>();

const router = useRouter();

const tipText = computed(() => {
  const { reason, gap } = props.alert;
  if (reason === "arrears") {
    return `账户已欠费 ${gap} 积分，补足后即可恢复 AI 能力调用`;
  }
  if (reason === "quota") {
    return "当前配额已用尽，加购积分或升级会员可继续畅享 AI 能力";
  }
  return `积分余额偏低，建议至少补充 ${gap} 积分，避免服务中断`;
});

function goShop() {
  router.push("/package/shop");
}
</script>

<template>
  <div class="recharge-guide">
    <div class="guide-content">
      <el-icon class="guide-icon"><Wallet /></el-icon>
      <span class="guide-text">{{ tipText }}</span>
    </div>
    <el-button type="primary" size="small" round @click="goShop">
      去加购
    </el-button>
  </div>
</template>

<style lang="scss" scoped>
.recharge-guide {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  background: linear-gradient(135deg, #fff7e6 0%, #fff1e0 100%);
  border: 1px solid #ffe0b2;
  border-radius: 12px;

  .guide-content {
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .guide-icon {
    font-size: 20px;
    color: var(--el-color-warning);
  }

  .guide-text {
    font-size: 14px;
    color: #874d00;
  }
}
</style>
