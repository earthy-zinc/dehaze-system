<!-- 知识库列表（用户端）：配额进度 + 分组 + 卡片列表 + 创建向导 -->
<script lang="ts" setup>
import { AiKnowledgeBaseAPI, type KnowledgeBaseVO } from "dehaze-sdk-js";
import { Plus } from "@element-plus/icons-vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { computed, onMounted } from "vue";
import { useRouter } from "vue-router";
import { useKbDataStore } from "@/store/modules/kbData";
import { useUserKbStore } from "@/store/modules/userKb";

defineOptions({
  name: "KbList",
  inheritAttrs: false,
});

const router = useRouter();
const kbDataStore = useKbDataStore();
const userKbStore = useUserKbStore();

// 列表接口已按可见性过滤（本人私有库 + 公共库），前端按分组与关键字过滤展示
const filteredList = computed(() => {
  const keyword = kbDataStore.kbListQuery.keyword?.trim() ?? "";
  return kbDataStore.kbList.filter((kb) => {
    if (userKbStore.activeGroup === "mine" && kb.visibility !== "private") {
      return false;
    }
    if (userKbStore.activeGroup === "public" && kb.visibility !== "public") {
      return false;
    }
    if (
      keyword &&
      !kb.name.includes(keyword) &&
      !(kb.description ?? "").includes(keyword)
    ) {
      return false;
    }
    return true;
  });
});

const quotaReached = computed(
  () => userKbStore.quota.created >= userKbStore.quota.limit
);

function goDetail(kb: KnowledgeBaseVO) {
  router.push(`/kb/${kb.id}`);
}

function handleCreateClick() {
  if (quotaReached.value) {
    ElMessage.warning("私有库配额已用完，升级会员可提升配额上限");
    return;
  }
  userKbStore.openCreateGuide();
}

async function handleDelete(kb: KnowledgeBaseVO) {
  try {
    await ElMessageBox.confirm(
      `确认删除知识库 "${kb.name}" ？删除后文档分块与 ES 索引同步清除`,
      "删除确认",
      { type: "warning" }
    );
  } catch {
    return;
  }
  await AiKnowledgeBaseAPI.delete(kb.id);
  ElMessage.success("知识库已删除");
  await kbDataStore.fetchKbList();
  userKbStore.refreshQuota();
}

onMounted(async () => {
  await kbDataStore.fetchKbList();
  userKbStore.refreshQuota();
});
</script>

<template>
  <div class="app-container">
    <QuotaProgressCard />

    <div
      class="search-container flex flex-wrap items-center justify-between gap-2"
    >
      <KbGroupTabs v-model="userKbStore.activeGroup" />
      <div class="flex items-center gap-2">
        <el-input
          v-model="kbDataStore.kbListQuery.keyword"
          placeholder="搜索知识库名称/描述"
          clearable
          style="width: 220px"
        />
        <el-button
          type="primary"
          :disabled="quotaReached"
          @click="handleCreateClick"
        >
          <el-icon class="mr-1"><Plus /></el-icon>
          新建知识库
        </el-button>
      </div>
    </div>

    <el-row
      v-if="filteredList.length > 0"
      v-loading="kbDataStore.loading"
      :gutter="16"
    >
      <el-col
        v-for="kb in filteredList"
        :key="kb.id"
        :xs="24"
        :sm="12"
        :md="8"
        :lg="6"
        class="mb-4"
      >
        <KbListCard
          :kb="kb"
          scope="self"
          @click="goDetail(kb)"
          @edit="goDetail(kb)"
          @delete="handleDelete(kb)"
        />
      </el-col>
    </el-row>
    <el-card
      v-else
      v-loading="kbDataStore.loading"
      shadow="never"
      class="!border-none"
    >
      <EmptyStateGuide
        :is-public="userKbStore.activeGroup === 'public'"
        @create="handleCreateClick"
      />
    </el-card>

    <CreateKbGuide v-model:visible="userKbStore.createDialogVisible" />
  </div>
</template>
