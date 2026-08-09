import { ref } from "vue";
import type { Ref } from "vue";

export interface FetchParams {
  pageNum: number;
  pageSize: number;
  keyword?: string;
}

export interface UsePagedListOptions<T> {
  /** 分页请求，返回当页记录数组（调用方负责从响应中取出记录） */
  fetcher: (params: FetchParams) => Promise<T[]>;
  /** 单页大小，默认 20 */
  pageSize?: number;
}

export function usePagedList<T>(options: UsePagedListOptions<T>) {
  const { fetcher, pageSize = 20 } = options;

  const list = ref<T[]>([]) as Ref<T[]>;
  const keyword = ref("");
  const pageNum = ref(1);
  const hasMore = ref(false);
  const loading = ref(false);

  async function fetchList(reset = false) {
    if (reset) {
      pageNum.value = 1;
      list.value = [];
    }
    loading.value = true;
    try {
      const records = await fetcher({
        pageNum: pageNum.value,
        pageSize,
        keyword: keyword.value || undefined,
      });
      if (reset) list.value = records;
      else list.value.push(...records);
      hasMore.value = records.length === pageSize;
      pageNum.value++;
    } finally {
      loading.value = false;
    }
  }

  const handleSearch = () => fetchList(true);
  const loadMore = () => fetchList();

  return {
    list,
    keyword,
    pageNum,
    hasMore,
    loading,
    fetchList,
    handleSearch,
    loadMore,
  };
}
