<!-- 调价影响测算：输入新价格版本，模拟重算成本与毛利 -->
<template>
  <div>
    <el-form :inline="true">
      <el-form-item label="模型">
        <el-select
          v-model="modelId"
          filterable
          style="width: 240px"
          placeholder="选择模型"
        >
          <el-option
            v-for="m in modelOptions"
            :key="m.modelId"
            :label="`${m.displayName} (${m.modelId})`"
            :value="m.modelId"
          />
        </el-select>
      </el-form-item>
      <el-form-item label="供应商">
        <el-select
          v-model="providerId"
          style="width: 200px"
          placeholder="选择供应商"
        >
          <el-option
            v-for="p in providerOptions"
            :key="p.id"
            :label="p.displayName"
            :value="p.id"
          />
        </el-select>
      </el-form-item>
      <el-form-item>
        <el-button
          type="primary"
          :loading="simulating"
          :disabled="!modelId"
          @click="simulate"
        >
          测算影响
        </el-button>
      </el-form-item>
    </el-form>

    <div class="text-xs text-gray-400 mb-1">
      新版本档位单价（元/百万token），按新档位均价与现有成本估算
    </div>
    <el-table :data="detailRows" size="small" class="max-w-[760px]">
      <el-table-column label="Token类型" width="130">
        <template #default="{ row }">
          <el-select v-model="row.tokenType" size="small">
            <el-option label="input 输入" value="input" />
            <el-option label="cached 缓存命中" value="cached" />
            <el-option label="output 输出" value="output" />
          </el-select>
        </template>
      </el-table-column>
      <el-table-column label="时段" width="110">
        <template #default="{ row }">
          <el-select v-model="row.timeSlot" size="small">
            <el-option label="peak 高峰" value="peak" />
            <el-option label="idle 空闲" value="idle" />
          </el-select>
        </template>
      </el-table-column>
      <el-table-column label="单价" width="160">
        <template #default="{ row }">
          <el-input-number
            v-model="row.unitPrice"
            :min="0"
            :precision="4"
            size="small"
            controls-position="right"
          />
        </template>
      </el-table-column>
      <el-table-column label="操作" width="70" align="center">
        <template #default="{ $index }">
          <el-button
            link
            type="danger"
            size="small"
            @click="detailRows.splice($index, 1)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-button class="mt-2" size="small" type="primary" plain @click="addRow">
      <el-icon><Plus /></el-icon>添加档位
    </el-button>

    <!-- 模拟结果：毛利对比卡片 -->
    <template v-if="result">
      <el-divider content-position="left">模拟结果</el-divider>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-[12px] max-w-[860px]">
        <el-card shadow="never">
          <div class="text-xs text-gray-400">当前成本（元）</div>
          <div class="text-xl mt-1">{{ result.currentCost.toFixed(2) }}</div>
        </el-card>
        <el-card shadow="never">
          <div class="text-xs text-gray-400">模拟成本（元）</div>
          <div class="text-xl mt-1">{{ result.simulatedCost.toFixed(2) }}</div>
        </el-card>
        <el-card shadow="never">
          <div class="text-xs text-gray-400">当前毛利（元）</div>
          <div
            class="text-xl mt-1"
            :class="result.currentProfit < 0 ? 'text-red-500' : ''"
          >
            {{ result.currentProfit.toFixed(2) }}
          </div>
        </el-card>
        <el-card shadow="never">
          <div class="text-xs text-gray-400">模拟毛利（元）</div>
          <div
            class="text-xl mt-1"
            :class="result.simulatedProfit < 0 ? 'text-red-500' : ''"
          >
            {{ result.simulatedProfit.toFixed(2) }}
          </div>
        </el-card>
      </div>
      <el-alert
        class="mt-3 max-w-[860px]"
        :type="result.simulatedProfit < 0 ? 'error' : 'info'"
        :closable="false"
        :title="`调价后毛利变化：${(result.simulatedProfit - result.currentProfit).toFixed(2)} 元${
          result.simulatedProfit < 0
            ? '，将出现负毛利，建议同步调整用户积分售价或更换供应商'
            : ''
        }`"
      />
    </template>
  </div>
</template>

<script lang="ts" setup>
import { Plus } from "@element-plus/icons-vue";
import {
  AiModelAPI,
  AiModelVO,
  AiProviderAPI,
  ModelCostDetailForm,
  ModelCostForm,
  ProviderVO,
} from "dehaze-sdk-js";
import {
  useAdminBillingStore,
  PriceImpactResult,
} from "@/store/modules/adminBilling";

defineOptions({ name: "PriceImpactSimulator" });

const billingStore = useAdminBillingStore();

const modelId = ref("");
const providerId = ref<number>();
const simulating = ref(false);
const result = ref<PriceImpactResult | null>(null);
const modelOptions = ref<AiModelVO[]>([]);
const providerOptions = ref<ProviderVO[]>([]);
const detailRows = reactive<ModelCostDetailForm[]>([]);

function addRow() {
  detailRows.push({
    tokenType: "input",
    timeSlot: "peak",
    minTokens: 0,
    unitPrice: 0,
  });
}

async function simulate() {
  const form: ModelCostForm = {
    modelId: modelId.value,
    providerId: providerId.value,
    details: detailRows.filter((row) => row.unitPrice >= 0),
  };
  simulating.value = true;
  try {
    result.value = await billingStore.simulatePriceImpact(form);
    if (!result.value) {
      ElMessage.warning("当前周期无成本-利润统计数据，无法测算");
    }
  } finally {
    simulating.value = false;
  }
}

onMounted(async () => {
  const [modelPage, providerPage] = await Promise.all([
    AiModelAPI.listModels({ pageNum: 1, pageSize: 100 }),
    AiProviderAPI.listProviders({ pageNum: 1, pageSize: 100 }),
  ]);
  modelOptions.value = modelPage.list ?? [];
  providerOptions.value = providerPage.list ?? [];
});
</script>
