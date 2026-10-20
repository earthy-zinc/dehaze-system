<!-- AI 计费管理端页：概览看板 → 消耗与成本下钻 → 治理与对账 三层垂直排布 -->
<template>
  <div class="app-container">
    <div ref="overviewRef">
      <admin-overview-board
        class="mb-[12px]"
        @drill-detail="scrollToDrilldown"
      />
    </div>
    <div ref="drilldownRef">
      <admin-drilldown-panel class="mb-[12px]" />
    </div>
    <el-collapse v-model="governanceOpen">
      <el-collapse-item name="governance" title="治理与对账">
        <admin-governance-panel />
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "AiBilling" });

const governanceOpen = ref<string[]>([]);
const drilldownRef = ref<HTMLElement>();

function scrollToDrilldown() {
  drilldownRef.value?.scrollIntoView({ behavior: "smooth" });
}
</script>
