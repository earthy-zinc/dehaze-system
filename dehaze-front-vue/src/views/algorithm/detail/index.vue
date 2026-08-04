<template>
  <div class="app-container">
    <el-page-header @back="$router.back()">
      <template #content>
        <span class="page-title">{{ algorithm?.name || "算法详情" }}</span>
      </template>
    </el-page-header>

    <el-card v-loading="loading" shadow="never" class="!border-none mt-4">
      <el-empty
        v-if="!loading && !algorithm"
        description="算法不存在或已被删除"
      />

      <template v-else-if="algorithm">
        <el-tabs v-model="activeTab">
          <el-tab-pane label="基本信息" name="basic">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="算法名称" :span="2">
                {{ algorithm.name }}
              </el-descriptions-item>
              <el-descriptions-item label="算法类型">
                <el-tag>{{ algorithm.type }}</el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="状态">
                <el-tag :type="statusType">
                  {{ statusLabel }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="版本">{{
                algorithm.version || "-"
              }}</el-descriptions-item>
              <el-descriptions-item label="审核人">{{
                algorithm.auditBy ?? "-"
              }}</el-descriptions-item>
              <el-descriptions-item label="算法描述" :span="2">
                {{ algorithm.description || "-" }}
              </el-descriptions-item>
              <el-descriptions-item label="创建时间">{{
                algorithm.createTime || "-"
              }}</el-descriptions-item>
              <el-descriptions-item label="审核时间">{{
                algorithm.auditTime || "-"
              }}</el-descriptions-item>
            </el-descriptions>
          </el-tab-pane>

          <el-tab-pane label="技术信息" name="tech">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="模型大小">
                {{ algorithm.size || "-" }}
              </el-descriptions-item>
              <el-descriptions-item label="浮点运算量(FLOPs)">
                {{ algorithm.flops || "-" }}
              </el-descriptions-item>
              <el-descriptions-item label="参数" :span="2">
                {{ algorithm.params || "-" }}
              </el-descriptions-item>
              <el-descriptions-item label="导入路径" :span="2">
                {{ algorithm.importPath || "-" }}
              </el-descriptions-item>
              <el-descriptions-item label="网络架构图" :span="2">
                <el-image
                  v-if="algorithm.img"
                  :src="algorithm.img"
                  fit="contain"
                  style="max-height: 300px"
                  preview-teleported
                >
                  <template #error>
                    <div class="arch-placeholder">
                      <el-icon :size="40"><Picture /></el-icon>
                      <span>网络架构图加载失败</span>
                    </div>
                  </template>
                </el-image>
                <div v-else class="arch-placeholder">
                  <el-icon :size="40"><Picture /></el-icon>
                  <span>暂无网络架构图</span>
                </div>
              </el-descriptions-item>
            </el-descriptions>
          </el-tab-pane>

          <el-tab-pane label="运营信息" name="ops">
            <el-descriptions :column="2" border v-loading="monitorLoading">
              <el-descriptions-item label="总调用次数">
                {{ monitorData?.callCount ?? "-" }}
              </el-descriptions-item>
              <el-descriptions-item label="今日调用">
                {{ monitorData?.todayCallCount ?? "-" }}
              </el-descriptions-item>
              <el-descriptions-item label="平均耗时">
                {{ monitorData?.avgTime ? `${monitorData.avgTime}ms` : "-" }}
              </el-descriptions-item>
              <el-descriptions-item label="成功率">
                {{
                  monitorData?.successRate
                    ? `${(monitorData.successRate * 100).toFixed(1)}%`
                    : "-"
                }}
              </el-descriptions-item>
            </el-descriptions>
          </el-tab-pane>
        </el-tabs>
      </template>
    </el-card>
  </div>
</template>

<script lang="ts" setup>
import { AlgorithmAPI, Algorithm, AlgorithmMonitorVO } from "dehaze-sdk-js";
import { Picture } from "@element-plus/icons-vue";

defineOptions({ name: "AlgorithmDetail" });

const route = useRoute();
const router = useRouter();
const loading = ref(true);
const monitorLoading = ref(false);
const algorithm = ref<Algorithm | null>(null);
const monitorData = ref<AlgorithmMonitorVO | null>(null);
const activeTab = ref("basic");

const statusLabel = computed(() => {
  const map: Record<number, string> = {
    1: "草稿",
    2: "测试中",
    3: "待审核",
    4: "已发布",
    5: "已停用",
    6: "已归档",
  };
  return map[algorithm.value?.status ?? 0] || "未知";
});

const statusType = computed(
  (): "primary" | "success" | "warning" | "info" | "danger" => {
    const map: Record<
      number,
      "primary" | "success" | "warning" | "info" | "danger"
    > = {
      1: "info",
      2: "warning",
      3: "warning",
      4: "success",
      5: "danger",
      6: "info",
    };
    return map[algorithm.value?.status ?? 0] || "info";
  }
);

async function loadMonitor(id: number) {
  monitorLoading.value = true;
  try {
    monitorData.value = await AlgorithmAPI.getMonitorData(id);
  } catch {
    monitorData.value = null;
  } finally {
    monitorLoading.value = false;
  }
}

onMounted(async () => {
  const id = Number(route.query.id);
  if (!id || isNaN(id)) {
    ElMessage.error("算法ID无效");
    router.back();
    return;
  }
  try {
    loading.value = true;
    algorithm.value = await AlgorithmAPI.getAlgorithmInfoById(id);
    loadMonitor(id);
  } catch (e: any) {
    ElMessage.error("获取算法详情失败：" + (e.message || "未知错误"));
  } finally {
    loading.value = false;
  }
});
</script>

<style scoped>
.page-title {
  font-size: 18px;
  font-weight: 600;
}

.arch-placeholder {
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: center;
  justify-content: center;
  min-height: 120px;
  font-size: 14px;
  color: #909399;
}
</style>
