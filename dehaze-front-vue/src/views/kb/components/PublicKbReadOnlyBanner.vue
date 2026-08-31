<!-- 公共库只读提示条：可见性边界显性化 -->
<script lang="ts" setup>
defineOptions({ name: "PublicKbReadOnlyBanner" });
</script>

<template>
  <el-alert
    type="info"
    show-icon
    :closable="false"
    class="mb-4"
    title="平台公共知识库为只读，仅支持浏览文档与检索"
  />
</template>
