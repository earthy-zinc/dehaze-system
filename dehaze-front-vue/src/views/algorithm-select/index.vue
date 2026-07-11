<script lang="ts" setup>
import { Algorithm, AlgorithmAPI } from "dehaze-sdk-js";

defineOptions({
  name: "AlgorithmSelect",
  inheritAttrs: false,
});

const emit = defineEmits<{
  (e: "select", algorithm: Algorithm): void;
}>();

const router = useRouter();
const loading = ref(false);
const keyword = ref("");
// 所有算法（扁平化后的叶子节点）
const allAlgorithms = ref<Algorithm[]>([]);
// 当前选中的分类
const activeCategory = ref<string>("全部");
// 当前选中的算法
const selectedAlgorithm = ref<Algorithm>();

// 算法分类体系：传统/深度学习/混合
const categories = ref([
  { label: "全部", value: "全部" },
  { label: "传统算法", value: "传统算法" },
  { label: "深度学习算法", value: "深度学习算法" },
  { label: "混合算法", value: "混合算法" },
]);

// 根据算法类型归入分类
function classifyAlgorithm(type: string): string {
  if (type.includes("深度学习") || type.includes("deep")) {
    return "深度学习算法";
  }
  if (type.includes("混合")) {
    return "混合算法";
  }
  return "传统算法";
}

// 递归扁平化算法树，提取叶子算法
function flattenAlgorithms(nodes: Algorithm[]): Algorithm[] {
  const result: Algorithm[] = [];
  nodes.forEach((node) => {
    if (node.children && node.children.length > 0) {
      result.push(...flattenAlgorithms(node.children));
    } else {
      result.push(node);
    }
  });
  return result;
}

// 树形分类数据
const treeData = computed(() => {
  const tree: any[] = categories.value
    .filter((c) => c.value !== "全部")
    .map((c) => ({
      label: c.label,
      value: c.value,
      children: allAlgorithms.value
        .filter((a) => classifyAlgorithm(a.type) === c.value)
        .map((a) => ({ label: a.name, value: a.name, algorithm: a })),
    }));
  return tree;
});

// 各分类算法数量统计
const categoryCount = computed(() => {
  const counts: Record<string, number> = {
    全部: allAlgorithms.value.length,
    传统算法: 0,
    深度学习算法: 0,
    混合算法: 0,
  };
  allAlgorithms.value.forEach((a) => {
    const cat = classifyAlgorithm(a.type);
    counts[cat]++;
  });
  return counts;
});

// 过滤后的算法卡片列表
const filteredAlgorithms = computed(() => {
  let result = allAlgorithms.value;
  // 分类过滤
  if (activeCategory.value !== "全部") {
    result = result.filter(
      (a) => classifyAlgorithm(a.type) === activeCategory.value
    );
  }
  // 关键词过滤
  const kw = keyword.value.trim().toLowerCase();
  if (kw) {
    result = result.filter(
      (a) =>
        a.name.toLowerCase().includes(kw) ||
        a.type.toLowerCase().includes(kw) ||
        (a.description || "").toLowerCase().includes(kw)
    );
  }
  return result;
});

// 智能推荐：基于算法类型推荐 Top 3（每个分类取第一个启用的算法）
const recommendAlgorithms = computed(() => {
  const recommended: Algorithm[] = [];
  const categoryList = ["传统算法", "深度学习算法", "混合算法"];
  categoryList.forEach((cat) => {
    const found = allAlgorithms.value.find(
      (a) => classifyAlgorithm(a.type) === cat && a.status === 1
    );
    if (found) {
      recommended.push(found);
    }
  });
  return recommended.slice(0, 3);
});

// 加载算法列表
function loadAlgorithms() {
  loading.value = true;
  AlgorithmAPI.getList()
    .then((data) => {
      allAlgorithms.value = flattenAlgorithms(data);
    })
    .finally(() => {
      loading.value = false;
    });
}

// 重置过滤
function resetFilter() {
  keyword.value = "";
  activeCategory.value = "全部";
}

// 点击树节点
function handleNodeClick(data: any) {
  if (data.algorithm) {
    // 点击叶子节点：选中算法
    selectedAlgorithm.value = data.algorithm;
  } else if (data.value) {
    // 点击分类节点：切换分类
    activeCategory.value = data.value;
  }
}

// 选择算法卡片
function handleSelectCard(algorithm: Algorithm) {
  selectedAlgorithm.value = algorithm;
}

// 确认选择
function handleConfirm() {
  if (!selectedAlgorithm.value) {
    ElMessage.warning("请先选择一个算法");
    return;
  }
  emit("select", selectedAlgorithm.value);
  ElMessage.success(`已选择算法：${selectedAlgorithm.value.name}`);
  // 返回上一页
  router.back();
}

onMounted(() => {
  loadAlgorithms();
});
</script>

<template>
  <div class="algorithm-select-container">
    <!-- 顶部搜索区 -->
    <el-card class="search-card" shadow="never">
      <div class="search-header">
        <el-input
          v-model="keyword"
          placeholder="搜索算法名称、类型或描述"
          clearable
          class="search-input"
        >
          <template #prefix>
            <i-ep-search />
          </template>
        </el-input>
        <el-button @click="resetFilter">
          <template #icon>
            <i-ep-refresh />
          </template>
          重置
        </el-button>
      </div>
    </el-card>

    <!-- 智能推荐区域 -->
    <el-card class="recommend-card" shadow="never">
      <template #header>
        <div class="card-title">
          <i-ep-magic-stick />
          <span>智能推荐</span>
          <el-tag size="small" type="success">Top 3</el-tag>
        </div>
      </template>
      <div v-loading="loading" class="recommend-list">
        <div
          v-for="algo in recommendAlgorithms"
          :key="algo.id"
          class="recommend-item"
          :class="{ active: selectedAlgorithm?.id === algo.id }"
          @click="handleSelectCard(algo)"
        >
          <div class="recommend-info">
            <div class="recommend-name">{{ algo.name }}</div>
            <el-tag size="small" type="primary">
              {{ classifyAlgorithm(algo.type) }}
            </el-tag>
          </div>
          <div class="recommend-desc">{{ algo.description || "暂无描述" }}</div>
        </div>
        <el-empty
          v-if="!loading && recommendAlgorithms.length === 0"
          description="暂无推荐算法"
          :image-size="60"
        />
      </div>
    </el-card>

    <!-- 主体区域：左侧树 + 右侧卡片 -->
    <div class="main-content">
      <!-- 左侧树形分类 -->
      <el-card class="tree-card" shadow="never">
        <template #header>
          <div class="card-title">
            <i-ep-folder />
            <span>算法分类</span>
          </div>
        </template>
        <div class="category-tabs">
          <div
            v-for="cat in categories"
            :key="cat.value"
            class="category-tab"
            :class="{ active: activeCategory === cat.value }"
            @click="activeCategory = cat.value"
          >
            {{ cat.label }}
            <span class="category-count">{{ categoryCount[cat.value] || 0 }}</span>
          </div>
        </div>
        <el-tree
          :data="treeData"
          :props="{ label: 'label', children: 'children' }"
          node-key="value"
          default-expand-all
          highlight-current
          @node-click="handleNodeClick"
        >
          <template #default="{ data }">
            <span class="tree-node">
              <span>{{ data.label }}</span>
              <span v-if="data.algorithm" class="tree-node-type">
                {{ data.algorithm.type }}
              </span>
            </span>
          </template>
        </el-tree>
      </el-card>

      <!-- 右侧算法卡片列表 -->
      <el-card class="cards-card" shadow="never">
        <template #header>
          <div class="card-title">
            <i-ep-grid />
            <span>算法列表</span>
            <el-tag size="small">共 {{ filteredAlgorithms.length }} 个</el-tag>
          </div>
        </template>
        <div v-loading="loading" class="card-list">
          <div
            v-for="algo in filteredAlgorithms"
            :key="algo.id"
            class="algo-card"
            :class="{ selected: selectedAlgorithm?.id === algo.id }"
            @click="handleSelectCard(algo)"
          >
            <div class="algo-card-header">
              <span class="algo-name">{{ algo.name }}</span>
              <el-tag
                size="small"
                :type="algo.status === 1 ? 'success' : 'info'"
              >
                {{ algo.status === 1 ? "启用" : "禁用" }}
              </el-tag>
            </div>
            <div class="algo-card-type">
              <el-tag size="small" type="primary">
                {{ classifyAlgorithm(algo.type) }}
              </el-tag>
              <span class="algo-type-raw">{{ algo.type }}</span>
            </div>
            <div class="algo-card-desc">{{ algo.description || "暂无描述" }}</div>
            <div v-if="selectedAlgorithm?.id === algo.id" class="algo-card-check">
              <i-ep-circle-check-filled />
            </div>
          </div>
          <el-empty
            v-if="!loading && filteredAlgorithms.length === 0"
            description="暂无匹配算法"
          />
        </div>
      </el-card>
    </div>

    <!-- 底部确认选择 -->
    <div class="footer-bar">
      <div class="selected-info">
        已选择：
        <span v-if="selectedAlgorithm" class="selected-name">
          {{ selectedAlgorithm.name }}
        </span>
        <span v-else class="selected-empty">未选择</span>
      </div>
      <el-button
        type="primary"
        size="large"
        :disabled="!selectedAlgorithm"
        @click="handleConfirm"
      >
        确认选择
      </el-button>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.algorithm-select-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
}

.search-header {
  display: flex;
  gap: 12px;
  align-items: center;

  .search-input {
    width: 360px;
  }
}

.card-title {
  display: flex;
  gap: 8px;
  align-items: center;
  font-size: 16px;
  font-weight: bold;
}

.recommend-list {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.recommend-item {
  position: relative;
  flex: 1;
  min-width: 220px;
  padding: 12px 16px;
  cursor: pointer;
  background: var(--el-fill-color-light);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  transition: all 0.2s;

  &:hover {
    border-color: var(--el-color-primary);
    box-shadow: 0 2px 12px rgb(0 0 0 / 10%);
  }

  &.active {
    background: var(--el-color-primary-light-9);
    border-color: var(--el-color-primary);
  }

  .recommend-info {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;

    .recommend-name {
      font-size: 15px;
      font-weight: 600;
    }
  }

  .recommend-desc {
    color: var(--el-text-color-secondary);
    font-size: 13px;
    line-height: 1.5;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }
}

.main-content {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}

.tree-card {
  flex: 0 0 260px;

  .category-tabs {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 12px;
  }

  .category-tab {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    cursor: pointer;
    border-radius: 6px;
    transition: all 0.2s;

    &:hover {
      background: var(--el-fill-color-light);
    }

    &.active {
      color: var(--el-color-primary);
      background: var(--el-color-primary-light-9);
    }

    .category-count {
      padding: 0 8px;
      color: var(--el-text-color-secondary);
      font-size: 12px;
      background: var(--el-fill-color);
      border-radius: 10px;
    }
  }
}

.tree-node {
  display: flex;
  gap: 8px;
  align-items: center;

  .tree-node-type {
    color: var(--el-text-color-secondary);
    font-size: 12px;
  }
}

.cards-card {
  flex: 1;
  min-width: 0;
}

.card-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}

.algo-card {
  position: relative;
  padding: 16px;
  cursor: pointer;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  transition: all 0.2s;

  &:hover {
    border-color: var(--el-color-primary);
    box-shadow: 0 2px 12px rgb(0 0 0 / 10%);
  }

  &.selected {
    background: var(--el-color-primary-light-9);
    border-color: var(--el-color-primary);
  }

  .algo-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;

    .algo-name {
      font-size: 15px;
      font-weight: 600;
    }
  }

  .algo-card-type {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;

    .algo-type-raw {
      color: var(--el-text-color-secondary);
      font-size: 12px;
    }
  }

  .algo-card-desc {
    color: var(--el-text-color-secondary);
    font-size: 13px;
    line-height: 1.5;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }

  .algo-card-check {
    position: absolute;
    top: 12px;
    right: 12px;
    color: var(--el-color-primary);
    font-size: 20px;
  }
}

.footer-bar {
  position: sticky;
  bottom: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  background: var(--el-bg-color);
  border-top: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  box-shadow: 0 -2px 12px rgb(0 0 0 / 5%);

  .selected-info {
    font-size: 14px;

    .selected-name {
      color: var(--el-color-primary);
      font-weight: 600;
    }

    .selected-empty {
      color: var(--el-text-color-secondary);
    }
  }
}
</style>
