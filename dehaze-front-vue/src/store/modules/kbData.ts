// 知识库数据 Store：用户端（scope=self）与管理端（scope=admin）共用的数据视图层
import {
  AiKnowledgeBaseAPI,
  DocumentQuery,
  DocumentVO,
  KnowledgeBaseQuery,
  KnowledgeBaseVO,
} from "dehaze-sdk-js";
import { defineStore } from "pinia";
import { reactive, ref } from "vue";

export type KbDataScope = "self" | "admin";

export const useKbDataStore = defineStore("kbData", () => {
  const scope = ref<KbDataScope>("self");
  const loading = ref(false);

  // pageSize 取 100：用户端列表页需在前端按"我的/公共"分组过滤，需拉取全量可见库
  const kbListQuery = reactive<KnowledgeBaseQuery>({
    keyword: "",
    pageNum: 1,
    pageSize: 100,
  });
  const kbList = ref<KnowledgeBaseVO[]>([]);
  const kbListTotal = ref(0);

  const kbDetail = ref<KnowledgeBaseVO | null>(null);

  const documentQuery = reactive<DocumentQuery>({ pageNum: 1, pageSize: 10 });
  const documents = ref<DocumentVO[]>([]);
  const total = ref(0);

  /** 注入数据范围，由宿主页面在挂载时调用 */
  function initScope(next: KbDataScope) {
    scope.value = next;
  }

  async function fetchKbList() {
    loading.value = true;
    try {
      const query: KnowledgeBaseQuery = { ...kbListQuery };
      if (scope.value === "admin") {
        query.view = "admin";
      }
      const result = await AiKnowledgeBaseAPI.getList(query);
      kbList.value = result.list ?? [];
      kbListTotal.value = result.total ?? 0;
    } finally {
      loading.value = false;
    }
  }

  async function fetchKbDetail(id: number) {
    kbDetail.value = await AiKnowledgeBaseAPI.getDetail(id);
    return kbDetail.value;
  }

  async function fetchDocuments(kbId: number) {
    loading.value = true;
    try {
      const result = await AiKnowledgeBaseAPI.getDocuments(kbId, {
        ...documentQuery,
      });
      documents.value = result.list ?? [];
      total.value = result.total ?? 0;
    } finally {
      loading.value = false;
    }
  }

  return {
    scope,
    loading,
    kbListQuery,
    kbList,
    kbListTotal,
    kbDetail,
    documentQuery,
    documents,
    total,
    initScope,
    fetchKbList,
    fetchKbDetail,
    fetchDocuments,
  };
});
