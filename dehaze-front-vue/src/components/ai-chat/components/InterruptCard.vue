<!-- 中断交互卡片：confirm 按 confirmKind 路由（算法推荐/工具授权/危险操作/写入冲突）/
     quota 配额不足 / async_wait 异步等待 / plan_approve 计划确认（可编辑：remove/reorder/add）。
     纯展示，决策经单一 resume 事件上抛完整 ChatResumeFormVM 载荷。 -->
<script lang="ts" setup>
import { computed, ref, watch } from "vue";
import type {
  ChatConfirmKindVM,
  ChatInterruptVM,
  ChatPlanEditVM,
  ChatResumeFormVM,
} from "../types";

defineOptions({ name: "InterruptCard" });

const props = defineProps<{
  interrupt: ChatInterruptVM;
}>();

const emit = defineEmits<{
  resume: [payload: ChatResumeFormVM];
}>();

const hasRecommendation = computed(
  () =>
    !!props.interrupt.recommendation || !!props.interrupt.alternatives?.length
);

// confirmKind 缺省时按载荷特征回退（算法推荐 / 写冲突），否则归"通用确认"
const kind = computed<ChatConfirmKindVM | undefined>(() => {
  if (props.interrupt.confirmKind) return props.interrupt.confirmKind;
  if (props.interrupt.recommendation) return "algorithm_recommend";
  if (props.interrupt.action === "write_conflict") return "write_conflict";
  return undefined;
});

const title = computed(() => {
  switch (props.interrupt.type) {
    case "plan_approve":
      return "计划待确认";
    case "quota":
      return "配额不足";
    case "async_wait":
      return "异步任务处理中";
    case "confirm":
      switch (kind.value) {
        case "algorithm_recommend":
          return "需要你的确认";
        case "tool_permission":
          return "工具授权确认";
        case "dangerous_op":
          return "危险操作确认";
        case "write_conflict":
          return "写入冲突";
        default:
          return "需要你的确认";
      }
    default:
      return "推理已暂停";
  }
});

// ===== confirm：算法推荐 / 工具授权 / 危险操作 / 写冲突 / 通用 =====
function acceptRecommendation(algorithmId: number) {
  emit("resume", { confirm: true, params: { algorithmId } });
}

function accept() {
  emit("resume", { confirm: true });
}

function reject() {
  emit("resume", { confirm: false });
}

function acceptWriteConflict() {
  emit("resume", { confirm: true, params: { action: "write_conflict" } });
}

// ===== quota / 其他：直接续流 =====
function resume() {
  emit("resume", {});
}

// ===== plan_approve：可编辑计划（remove / reorder / add） =====
interface EditableTask {
  id?: string;
  description: string;
  isNew: boolean;
}

const editable = ref<EditableTask[]>([]);
const originalIds = ref<string[]>([]);

watch(
  () => props.interrupt,
  (interrupt) => {
    const plan = interrupt.plan ?? [];
    editable.value = plan.map((task) => ({
      id: task.id,
      description: task.description ?? "",
      isNew: false,
    }));
    originalIds.value = plan
      .map((task) => task.id)
      .filter((id): id is string => !!id);
  },
  { immediate: true }
);

function removeTask(index: number) {
  editable.value.splice(index, 1);
}

function moveTask(index: number, delta: number) {
  const target = index + delta;
  if (target < 0 || target >= editable.value.length) return;
  const [task] = editable.value.splice(index, 1);
  editable.value.splice(target, 0, task);
}

// 后端 wire plan_edit.add 仅接受单个对象（数组会被静默忽略导致新增丢失），
// 故新增任务收敛为单条：已存在待新增项时禁用再次添加。
const hasNewTask = computed(() => editable.value.some((task) => task.isNew));

function addTask() {
  if (hasNewTask.value) return;
  editable.value.push({ description: "", isNew: true });
}

function buildPlanEdit(): ChatPlanEditVM {
  const currentIds = editable.value
    .filter((task) => task.id && !task.isNew)
    .map((task) => task.id as string);
  const remove = originalIds.value.filter((id) => !currentIds.includes(id));
  const newTask = editable.value.find(
    (task) => task.isNew && task.description.trim()
  );
  const orderChanged =
    currentIds.length === originalIds.value.length &&
    currentIds.some((id, i) => id !== originalIds.value[i]);

  const planEdit: ChatPlanEditVM = {};
  if (remove.length) planEdit.remove = remove;
  if (orderChanged) planEdit.reorder = currentIds;
  if (newTask)
    planEdit.add = { description: newTask.description.trim(), dependsOn: [] };
  return planEdit;
}

function approvePlan() {
  const planEdit = buildPlanEdit();
  const hasEdit = !!(planEdit.remove || planEdit.reorder || planEdit.add);
  emit("resume", hasEdit ? { confirm: true, planEdit } : { confirm: true });
}

const canApprove = computed(() =>
  editable.value.some((task) => task.description.trim() || task.id)
);
</script>

<template>
  <div class="interrupt-card">
    <div class="interrupt-card__header">
      <span class="interrupt-card__title">{{ title }}</span>
    </div>

    <!-- plan_approve：计划确认（可编辑） -->
    <template v-if="interrupt.type === 'plan_approve'">
      <div class="interrupt-card__body">
        <div class="interrupt-card__desc">
          执行计划共 {{ editable.length }} 个任务：
        </div>
        <ol class="interrupt-card__plan">
          <li
            v-for="(task, index) in editable"
            :key="task.id ?? `new-${index}`"
            class="interrupt-card__task"
          >
            <el-input
              v-model="task.description"
              size="small"
              class="interrupt-card__task-input"
              placeholder="任务描述"
            />
            <el-button
              link
              size="small"
              :disabled="index === 0"
              @click="moveTask(index, -1)"
              >上移</el-button
            >
            <el-button
              link
              size="small"
              :disabled="index === editable.length - 1"
              @click="moveTask(index, 1)"
              >下移</el-button
            >
            <el-button link size="small" @click="removeTask(index)"
              >移除</el-button
            >
          </li>
        </ol>
        <el-button link size="small" :disabled="hasNewTask" @click="addTask"
          >+ 添加任务</el-button
        >
      </div>
      <div class="interrupt-card__actions">
        <el-button
          type="primary"
          size="small"
          :disabled="!canApprove"
          @click="approvePlan"
        >
          批准执行
        </el-button>
        <el-button size="small" @click="reject">取消</el-button>
      </div>
    </template>

    <!-- confirm：算法推荐 -->
    <template
      v-else-if="interrupt.type === 'confirm' && kind === 'algorithm_recommend'"
    >
      <div class="interrupt-card__body">
        <template v-if="interrupt.recommendation">
          <div>
            推荐算法：<strong>{{
              interrupt.recommendation.algorithmName
            }}</strong>
          </div>
          <div class="interrupt-card__desc">
            {{ interrupt.recommendation.reason }}
          </div>
        </template>
        <div v-if="interrupt.alternatives?.length" class="interrupt-card__desc">
          备选算法：
          <el-button
            v-for="alt in interrupt.alternatives"
            :key="alt.algorithmId"
            link
            size="small"
            @click="acceptRecommendation(alt.algorithmId)"
          >
            {{ alt.algorithmName }}
          </el-button>
        </div>
      </div>
      <div class="interrupt-card__actions">
        <el-button
          v-if="interrupt.recommendation"
          type="primary"
          size="small"
          @click="acceptRecommendation(interrupt.recommendation.algorithmId)"
        >
          采纳推荐
        </el-button>
        <el-button size="small" @click="reject">拒绝</el-button>
      </div>
    </template>

    <!-- confirm：工具授权 -->
    <template
      v-else-if="interrupt.type === 'confirm' && kind === 'tool_permission'"
    >
      <div class="interrupt-card__body">
        <div>
          {{ interrupt.detail ?? interrupt.reason ?? "需要授权调用工具" }}
        </div>
        <div v-if="hasRecommendation" class="interrupt-card__desc">
          {{ interrupt.recommendation?.algorithmName }}
        </div>
      </div>
      <div class="interrupt-card__actions">
        <el-button type="primary" size="small" @click="accept">允许</el-button>
        <el-button size="small" @click="reject">拒绝</el-button>
      </div>
    </template>

    <!-- confirm：危险操作 -->
    <template
      v-else-if="interrupt.type === 'confirm' && kind === 'dangerous_op'"
    >
      <div class="interrupt-card__body">
        <div>
          {{ interrupt.detail ?? interrupt.reason ?? "该操作存在风险" }}
        </div>
      </div>
      <div class="interrupt-card__actions">
        <el-button type="danger" size="small" @click="accept"
          >确认执行</el-button
        >
        <el-button size="small" @click="reject">取消</el-button>
      </div>
    </template>

    <!-- confirm：写入冲突 -->
    <template
      v-else-if="interrupt.type === 'confirm' && kind === 'write_conflict'"
    >
      <div class="interrupt-card__body">
        <div>{{ interrupt.detail ?? "检测到写入冲突，需选择处理方式" }}</div>
      </div>
      <div class="interrupt-card__actions">
        <el-button type="primary" size="small" @click="acceptWriteConflict"
          >覆盖写入</el-button
        >
        <el-button size="small" @click="reject">取消</el-button>
      </div>
    </template>

    <!-- confirm：通用确认 -->
    <template v-else-if="interrupt.type === 'confirm'">
      <div class="interrupt-card__body">
        <div>{{ interrupt.detail ?? interrupt.reason ?? "需要你的确认" }}</div>
      </div>
      <div class="interrupt-card__actions">
        <el-button type="primary" size="small" @click="accept">继续</el-button>
        <el-button size="small" @click="reject">拒绝</el-button>
      </div>
    </template>

    <!-- quota：配额不足 -->
    <template v-else-if="interrupt.type === 'quota'">
      <div class="interrupt-card__body">
        <div>
          {{ interrupt.upgradeTip ?? "积分配额已用尽，推理已暂停" }}
        </div>
        <div
          v-if="interrupt.usedDaily != null && interrupt.dailyLimit"
          class="interrupt-card__desc"
        >
          今日已用 {{ interrupt.usedDaily }} / {{ interrupt.dailyLimit }}
        </div>
        <div
          v-if="interrupt.usedMonthly != null && interrupt.monthlyLimit"
          class="interrupt-card__desc"
        >
          本月已用 {{ interrupt.usedMonthly }} / {{ interrupt.monthlyLimit }}
        </div>
      </div>
      <div class="interrupt-card__actions">
        <el-button size="small" type="primary" @click="resume">重试</el-button>
      </div>
    </template>

    <!-- async_wait：异步等待 -->
    <template v-else-if="interrupt.type === 'async_wait'">
      <div class="interrupt-card__body">
        <div>后台任务执行中，完成后将自动继续推理</div>
        <div v-if="interrupt.estDuration" class="interrupt-card__desc">
          预计耗时 {{ interrupt.estDuration }}
        </div>
        <div v-if="interrupt.imageCount" class="interrupt-card__desc">
          共 {{ interrupt.imageCount }} 张图片
        </div>
      </div>
    </template>

    <!-- 其他中断 -->
    <template v-else>
      <div class="interrupt-card__body">
        {{ interrupt.reason ?? interrupt.detail ?? "推理已暂停" }}
      </div>
      <div class="interrupt-card__actions">
        <el-button type="primary" size="small" @click="resume">继续</el-button>
      </div>
    </template>
  </div>
</template>

<style scoped lang="scss">
.interrupt-card {
  max-width: 92%;
  padding: 12px 14px;
  margin: 0 auto 16px;
  background-color: var(--el-color-warning-light-9);
  border: 1px solid var(--el-color-warning-light-5);
  border-radius: 8px;

  &__header {
    margin-bottom: 6px;
  }

  &__title {
    font-size: 14px;
    font-weight: 600;
  }

  &__body {
    font-size: 13px;
    line-height: 1.6;
  }

  &__desc {
    margin-top: 4px;
    color: var(--el-text-color-secondary);
  }

  &__plan {
    padding-left: 0;
    margin: 4px 0 0;
    font-size: 13px;
    list-style: none;
  }

  &__task {
    display: flex;
    gap: 6px;
    align-items: center;
    margin-top: 4px;
  }

  &__task-input {
    flex: 1;
  }

  &__deps {
    color: var(--el-text-color-secondary);
  }

  &__actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }
}
</style>
