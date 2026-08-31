<!-- 私有库配额进度：已建 X/Y，达限引导升级会员 -->
<script lang="ts" setup>
import { computed } from "vue";
import { useUserKbStore } from "@/store/modules/userKb";

defineOptions({ name: "QuotaProgressCard" });

const userKbStore = useUserKbStore();

const reached = computed(
  () => userKbStore.quota.created >= userKbStore.quota.limit
);
const percent = computed(() =>
  Math.min(
    100,
    Math.round((userKbStore.quota.created / userKbStore.quota.limit) * 100)
  )
);
</script>

<template>
  <el-card shadow="never" class="!border-none mb-4">
    <div class="flex items-center justify-between mb-2">
      <span class="font-bold">私有知识库配额</span>
      <span :class="reached ? 'text-red-500' : 'text-gray-500'">
        已建 {{ userKbStore.quota.created }} / {{ userKbStore.quota.limit }}
      </span>
    </div>
    <el-progress
      :percentage="percent"
      :status="reached ? 'exception' : undefined"
    />
    <div v-if="reached" class="mt-2 text-xs text-red-500">
      私有库配额已用完，升级会员可提升配额上限
    </div>
  </el-card>
</template>
