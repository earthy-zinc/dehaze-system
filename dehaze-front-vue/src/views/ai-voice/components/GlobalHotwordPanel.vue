<!-- 全局热词配置：对所有用户的语音识别生效，增删需 voice:hotword:edit 权限 -->
<script lang="ts" setup>
import { HotwordVO } from "dehaze-sdk-js";
import { onMounted, ref } from "vue";
import { Plus } from "@element-plus/icons-vue";
import { useAdminVoiceStore } from "@/store/modules/adminVoice";

const adminVoiceStore = useAdminVoiceStore();

const newWord = ref("");
const submitting = ref(false);

async function handleAdd() {
  const word = newWord.value.trim();
  if (!word) {
    ElMessage.warning("请输入热词内容");
    return;
  }
  submitting.value = true;
  try {
    await adminVoiceStore.addGlobalHotword(word);
    newWord.value = "";
  } finally {
    submitting.value = false;
  }
}

async function handleDelete(hotword: HotwordVO) {
  try {
    await ElMessageBox.confirm(
      `确认删除全局热词「${hotword.word}」？删除后所有用户不再生效。`,
      "删除确认",
      { type: "warning", confirmButtonText: "确定", cancelButtonText: "取消" }
    );
  } catch {
    return;
  }
  await adminVoiceStore.deleteGlobalHotword(hotword.id);
}

onMounted(() => {
  adminVoiceStore.fetchGlobalHotwords();
});
</script>

<template>
  <div>
    <div class="flex justify-between mb-4">
      <div class="flex items-center gap-2">
        <el-input
          v-model="newWord"
          v-has-perm="['voice:hotword:edit']"
          placeholder="输入热词（专业术语、品牌名等）"
          maxlength="50"
          class="!w-64"
          @keyup.enter="handleAdd"
        />
        <el-button
          v-has-perm="['voice:hotword:edit']"
          type="primary"
          :loading="submitting"
          @click="handleAdd"
        >
          <el-icon><Plus /></el-icon>
          新增热词
        </el-button>
      </div>
      <el-button @click="adminVoiceStore.fetchGlobalHotwords()">刷新</el-button>
    </div>

    <el-table
      v-loading="adminVoiceStore.hotwordLoading"
      :data="adminVoiceStore.globalHotwords"
      border
    >
      <el-table-column prop="id" label="ID" width="100" />
      <el-table-column prop="word" label="热词内容" min-width="200" />
      <el-table-column prop="createTime" label="创建时间" width="180" />
      <el-table-column label="操作" width="120" fixed="right">
        <template #default="{ row }">
          <el-button
            v-has-perm="['voice:hotword:edit']"
            type="danger"
            link
            @click="handleDelete(row as HotwordVO)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>
