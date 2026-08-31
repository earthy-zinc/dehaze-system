// 用户端知识库 Store：分组选择、配额进度、创建向导
import { AiKnowledgeBaseAPI, KnowledgeBaseCreateForm } from "dehaze-sdk-js";
import { ElMessage } from "element-plus";
import { defineStore } from "pinia";
import { ref } from "vue";
import { useKbDataStore } from "./kbData";

// 配额上限需后端在会员权益/字典中补充，当前为前端默认值
export const PRIVATE_KB_QUOTA_DEFAULT = 3;

export const useUserKbStore = defineStore("userKb", () => {
  const activeGroup = ref<"mine" | "public">("mine");
  const quota = ref({ created: 0, limit: PRIVATE_KB_QUOTA_DEFAULT });
  const createDialogVisible = ref(false);

  function switchGroup(group: "mine" | "public") {
    activeGroup.value = group;
  }

  function openCreateGuide() {
    createDialogVisible.value = true;
  }

  /** 已建私有库数量从 scope=self 列表统计，配额上限待后端会员权益提供 */
  function refreshQuota() {
    const kbDataStore = useKbDataStore();
    quota.value.created = kbDataStore.kbList.filter(
      (kb) => kb.visibility === "private"
    ).length;
  }

  function setQuota(created: number, limit?: number) {
    quota.value.created = created;
    if (limit !== undefined) {
      quota.value.limit = limit;
    }
  }

  async function submitCreate(form: KnowledgeBaseCreateForm) {
    const kbDataStore = useKbDataStore();
    await AiKnowledgeBaseAPI.create(form);
    ElMessage.success("知识库创建成功");
    createDialogVisible.value = false;
    await kbDataStore.fetchKbList();
    refreshQuota();
  }

  return {
    activeGroup,
    quota,
    createDialogVisible,
    switchGroup,
    openCreateGuide,
    submitCreate,
    refreshQuota,
    setQuota,
  };
});
