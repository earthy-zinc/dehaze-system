<template>
  <el-card class="recommendation-panel" shadow="never">
    <template #header>
      <div class="panel-header">
        <span class="panel-title">算法推荐</span>
        <el-tag v-if="imageReady" type="success" size="small">分析完成</el-tag>
        <el-tag v-else-if="imageId || imageUrl" type="warning" size="small"
          >等待分析</el-tag
        >
      </div>
    </template>

    <!-- No Image Provided -->
    <div
      v-if="!imageId && !imageUrl && !analyzing && !imageReady"
      class="no-image-state"
    >
      <el-empty description="请通过 imageId 或 imageUrl prop 传入图片" />
      <div class="prop-hint">
        示例：&lt;RecommendationPanel image-id="123" /&gt; 或
        &lt;RecommendationPanel image-url="'https://...'"/&gt;
      </div>
    </div>

    <!-- Analyzing State -->
    <div v-else-if="analyzing" class="analyzing-state">
      <el-progress type="circle" :percentage="50" color="#409eff" />
      <div class="analyzing-text">正在分析图像特征...</div>
    </div>

    <!-- Result State -->
    <template v-else>
      <!-- Feature Analysis Section -->
      <el-collapse accordion v-model="activeCollapse">
        <el-collapse-item name="features" title="图像特征分析">
          <div class="feature-grid">
            <div class="feature-item">
              <span class="feature-label">雾霾浓度</span>
              <div class="feature-value">
                {{ hazeLevelLabel }}
                <el-tag :type="getHazeLevelType()" size="small" effect="dark">
                  {{ hazeConfidence.toFixed(2) }}
                </el-tag>
              </div>
            </div>
            <div class="feature-item">
              <span class="feature-label">场景类型</span>
              <span class="feature-value"
                >{{ sceneTypeLabel }}
                <el-tag :type="getSceneTypeType()" size="small" effect="dark">
                  {{ sceneConfidence.toFixed(2) }}
                </el-tag>
              </span>
            </div>
            <div class="feature-item">
              <span class="feature-label">光照条件</span>
              <span class="feature-value">{{ lightingLabel }}</span>
            </div>
            <div class="feature-item">
              <span class="feature-label">复杂度</span>
              <el-progress
                :percentage="Math.round(complexity * 100)"
                :color="complexityColor"
                :stroke-width="6"
                style="width: 120px"
              />
            </div>
            <div class="feature-item">
              <span class="feature-label">分辨率</span>
              <span class="feature-value">{{ resolutionLabel }}</span>
            </div>
            <div class="feature-item">
              <span class="feature-label">噪声水平</span>
              <span class="feature-value">{{ noiseLevelLabel }}</span>
            </div>
          </div>
        </el-collapse-item>
      </el-collapse>

      <!-- Recommended Algorithms Section -->
      <div class="algo-section">
        <div class="section-title">Top 推荐算法</div>
        <div
          v-for="(algo, idx) in recommendations"
          :key="algo.algorithmId"
          class="algo-card"
        >
          <div class="algo-header">
            <span class="algo-rank">{{ idx + 1 }}</span>
            <div class="algo-info">
              <div class="algo-name">{{ algo.algorithmName }}</div>
              <el-rate :model-value="algo.rating" disabled :show-text="false" />
            </div>
            <el-button link type="primary" @click="handleSelect(algo)">
              选择
            </el-button>
          </div>
          <div class="algo-score-bar">
            <el-progress
              :percentage="algo.matchScore"
              :color="scoreColor(algo.matchScore)"
              :stroke-width="8"
            />
          </div>
          <div class="algo-reason">{{ algo.reason }}</div>
          <div class="algo-meta">
            <span>匹配度 {{ algo.matchScore }}%</span>
            <span v-if="algo.estimatedTime"
              >预计耗时 {{ formatTime(algo.estimatedTime) }}</span
            >
          </div>
          <div class="algo-feedback">
            <el-button
              link
              type="success"
              size="small"
              :loading="feedbackLoading === algo.recommendationId"
              @click="handleFeedback(algo, true)"
            >
              <el-icon><ThumbUp /></el-icon>有用
            </el-button>
            <el-button
              link
              type="danger"
              size="small"
              :loading="feedbackLoading === algo.recommendationId"
              @click="handleFeedback(algo, false)"
            >
              <el-icon><CircleClose /></el-icon>无用
            </el-button>
          </div>
        </div>
      </div>
    </template>

    <!-- Error State -->
    <el-alert
      v-if="error"
      title="分析失败"
      :description="error"
      type="error"
      closable
      style="margin-top: 12px"
      @close="error = ''"
    />
  </el-card>
</template>

<script lang="ts" setup>
import { ref, computed, watch } from "vue";
import {
  RecommendationAPI,
  ImageFeatureAnalysis,
  RecommendedAlgorithm,
} from "dehaze-sdk-js";
import { CircleCheck, CircleClose } from "@element-plus/icons-vue";

defineOptions({ name: "RecommendationPanel" });

const props = defineProps<{
  imageId?: number;
  imageUrl?: string;
}>();

const emit = defineEmits<{
  (e: "select", algorithm: RecommendedAlgorithm): void;
}>();

const analysis = ref<ImageFeatureAnalysis | null>(null);
const recommendations = ref<RecommendedAlgorithm[]>([]);
const analyzing = ref(false);
const error = ref("");
const feedbackLoading = ref<number | null>(null);
const imageReady = ref(false);

function reset() {
  analysis.value = null;
  recommendations.value = [];
  analyzing.value = false;
  error.value = "";
  imageReady.value = false;
}

async function doAnalyze() {
  if (!props.imageId && !props.imageUrl) return;
  reset();
  analyzing.value = true;
  error.value = "";

  try {
    const result = await RecommendationAPI.analyze({
      imageId: props.imageId,
      imageUrl: props.imageUrl,
    });
    analysis.value = result;
    imageReady.value = true;
    fetchRecommendations(result.imageMd5);
  } catch (err: any) {
    error.value = err?.message || "图像分析失败";
    ElMessage.error(error.value);
  } finally {
    analyzing.value = false;
  }
}

async function fetchRecommendations(imageMd5?: string) {
  try {
    const list = await RecommendationAPI.getAlgorithmRecommendations({
      imageMd5,
    });
    recommendations.value = list.slice(0, 3);
  } catch {
    ElMessage.warning("未能获取推荐算法");
  }
}

function handleSelect(algo: RecommendedAlgorithm) {
  emit("select", algo);
  ElMessage.success(`已选择 ${algo.algorithmName}`);
}

async function handleFeedback(algo: RecommendedAlgorithm, useful: boolean) {
  if (!algo.recommendationId) {
    ElMessage.warning("暂无反馈入口");
    return;
  }
  feedbackLoading.value = algo.recommendationId;
  try {
    await RecommendationAPI.submitFeedback({
      recommendationId: algo.recommendationId,
      useful,
    });
    ElMessage.success(useful ? "反馈已提交（有用）" : "反馈已提交（无用）");
  } catch {
    ElMessage.error("反馈提交失败");
  } finally {
    feedbackLoading.value = null;
  }
}

// Feature display helpers
const activeCollapse = ref<string | undefined>("features");

const hazeLevelLabel = computed(() => {
  const map: Record<string, string> = {
    light: "轻度",
    moderate: "中度",
    heavy: "重度",
  };
  return map[analysis.value?.hazeLevel || ""] || "-";
});
const hazeConfidence = computed(() => analysis.value?.hazeConfidence || 0);
const sceneTypeLabel = computed(() => {
  const map: Record<string, string> = {
    urban: "城市",
    landscape: "风景",
    building: "建筑",
    night: "夜景",
    backlight: "逆光",
    indoor: "室内",
  };
  return map[analysis.value?.sceneType || ""] || "-";
});
const sceneConfidence = computed(() => analysis.value?.sceneConfidence || 0);
const lightingLabel = computed(() => {
  const map: Record<string, string> = {
    bright: "明亮",
    normal: "正常",
    dark: "昏暗",
    veryDark: "黑暗",
    backlight: "逆光",
  };
  return map[analysis.value?.lighting || ""] || "-";
});
const complexity = computed(() => analysis.value?.complexity || 0);
const resolutionLabel = computed(() => {
  const map: Record<string, string> = { sd: "SD", hd: "HD", uhd: "UHD" };
  return map[analysis.value?.resolution || ""] || "-";
});
const noiseLevelLabel = computed(() => {
  const map: Record<string, string> = { low: "低", medium: "中", high: "高" };
  return map[analysis.value?.noiseLevel || ""] || "-";
});

function getHazeLevelType():
  "primary" | "success" | "warning" | "info" | "danger" {
  const h = analysis.value?.hazeLevel;
  if (h === "heavy") return "danger";
  if (h === "moderate") return "warning";
  return "success";
}
function getSceneTypeType():
  "primary" | "success" | "warning" | "info" | "danger" {
  return "info";
}
function complexityColor(): string {
  const c = complexity.value;
  if (c > 0.7) return "#f56c6c";
  if (c > 0.4) return "#e6a23c";
  return "#67c23a";
}
function scoreColor(score: number): string {
  if (score >= 80) return "#67c23a";
  if (score >= 60) return "#e6a23c";
  return "#909399";
}
function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m${s % 60}s`;
}

watch(
  () => [props.imageId, props.imageUrl],
  () => {
    if (props.imageId || props.imageUrl) {
      doAnalyze();
    }
  }
);
</script>

<style scoped>
.panel-header {
  display: flex;
  gap: 8px;
  align-items: center;
}

.panel-title {
  font-size: 16px;
  font-weight: 600;
}

.no-image-state {
  padding: 40px 0;
  text-align: center;
}

.prop-hint {
  margin-top: 12px;
  font-size: 12px;
  color: #909399;
}

.analyzing-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px 0;
}

.analyzing-text {
  margin-top: 16px;
  font-size: 14px;
  color: #606266;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  padding: 8px 0;
}

.feature-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.feature-label {
  font-size: 12px;
  color: #909399;
}

.feature-value {
  display: flex;
  gap: 6px;
  align-items: center;
  font-size: 14px;
  font-weight: 500;
  color: #303133;
}

.algo-section {
  margin-top: 16px;
}

.section-title {
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 600;
}

.algo-card {
  padding: 12px;
  margin-bottom: 12px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  transition: box-shadow 0.2s;
}

.algo-card:hover {
  box-shadow: 0 2px 8px rgb(0 0 0 / 8%);
}

.algo-header {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 8px;
}

.algo-rank {
  display: flex;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  font-size: 14px;
  font-weight: 600;
  color: #fff;
  background: #409eff;
  border-radius: 50%;
}

.algo-info {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 2px;
}

.algo-name {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.algo-score-bar {
  margin-bottom: 6px;
}

.algo-reason {
  margin-bottom: 6px;
  font-size: 12px;
  line-height: 1.5;
  color: #606266;
}

.algo-meta {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 12px;
  color: #909399;
}

.algo-feedback {
  display: flex;
  gap: 12px;
}
</style>
