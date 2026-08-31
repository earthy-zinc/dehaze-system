<!-- Agent 配置表单：基本信息/系统提示词/模型与推理范式/推理参数/Skills/MCP/子 Agent/文件权限/安全护栏 -->
<template>
  <el-form
    ref="formRef"
    v-loading="agentStore.detailLoading"
    :model="form"
    :rules="rules"
    label-width="130px"
  >
    <el-alert
      v-if="props.agentId != null"
      class="mb-4"
      type="info"
      :closable="false"
      title="保存将生成新的草稿快照，需在「版本」页签发布并通过回归门禁后，才对新会话生效；进行中会话沿用创建时版本。"
    />

    <el-divider content-position="left">基本信息</el-divider>
    <el-form-item label="名称" prop="name">
      <el-input v-model="form.name" class="!w-[360px]" />
    </el-form-item>
    <el-form-item label="唯一编码" prop="agentCode">
      <el-input
        v-model="form.agentCode"
        class="!w-[360px]"
        :disabled="props.agentId != null"
        placeholder="如 image-analyst，创建后不可修改"
      />
    </el-form-item>
    <el-form-item label="描述">
      <el-input
        v-model="form.description"
        type="textarea"
        :rows="2"
        class="!w-[560px]"
      />
    </el-form-item>
    <el-form-item label="分类标签">
      <el-input
        v-model="form.tags"
        class="!w-[360px]"
        placeholder="逗号分隔，如 图像处理,数据分析"
      />
    </el-form-item>
    <el-form-item label="类型">
      <el-checkbox v-model="form.isSubagent"
        >可作为子 Agent（不可被会话直接选择）</el-checkbox
      >
      <el-checkbox v-model="form.isTeam"
        >Team 团队（由 Team Lead 编排成员协作）</el-checkbox
      >
    </el-form-item>
    <el-form-item label="对外暴露">
      <el-switch v-model="form.isExposed" />
      <span class="ml-2 text-xs text-gray-400">
        暴露为 A2A 子 Agent，供外部 Agent 通过 A2A 协议调用
      </span>
    </el-form-item>
    <el-form-item label="排序">
      <el-input-number
        v-model="form.sortOrder"
        :min="0"
        controls-position="right"
      />
    </el-form-item>
    <el-form-item v-if="props.agentId == null" label="状态">
      <el-switch v-model="form.status" :active-value="1" :inactive-value="0" />
    </el-form-item>

    <el-divider content-position="left">系统提示词（Markdown）</el-divider>
    <el-form-item label-width="0">
      <el-input
        v-model="form.systemPrompt"
        type="textarea"
        :rows="10"
        placeholder="Agent 的指令（System Prompt），支持 Markdown 格式"
      />
    </el-form-item>

    <el-divider content-position="left">模型与推理范式</el-divider>
    <el-form-item label="对话模型" prop="modelId">
      <el-select
        v-model="form.modelId"
        class="!w-[360px]"
        filterable
        placeholder="仅启用中的 chat 模型"
      >
        <el-option
          v-for="m in modelOptions"
          :key="m.modelId"
          :label="`${m.displayName} (${m.modelId})`"
          :value="m.modelId"
        />
      </el-select>
    </el-form-item>
    <el-form-item label="推理范式">
      <el-select v-model="form.reasoningMode" class="!w-[240px]">
        <el-option label="自动（按复杂度评估选择）" value="auto" />
        <el-option label="直接回复" value="direct" />
        <el-option label="ReAct" value="react" />
        <el-option label="Plan-and-Execute" value="plan_execute" />
        <el-option label="Reflexion 反思" value="reflexion" />
      </el-select>
    </el-form-item>

    <el-divider content-position="left"
      >推理参数（留空继承系统默认）</el-divider
    >
    <el-form-item label="最大推理步数">
      <el-input-number
        v-model="form.maxSteps"
        :min="1"
        controls-position="right"
        placeholder="继承默认"
      />
    </el-form-item>
    <el-form-item label="Token 预算">
      <el-input-number
        v-model="form.tokenBudget"
        :min="1"
        controls-position="right"
        placeholder="单会话上限"
      />
    </el-form-item>
    <el-form-item label="并行度">
      <el-input-number
        v-model="form.maxParallel"
        :min="1"
        controls-position="right"
        placeholder="并行子任务最大数"
      />
    </el-form-item>
    <el-form-item label="工具超时（秒）">
      <el-input-number
        v-model="form.toolTimeout"
        :min="1"
        controls-position="right"
      />
    </el-form-item>
    <el-form-item label="重试次数">
      <el-input-number
        v-model="form.retryMax"
        :min="0"
        controls-position="right"
        placeholder="工具失败最大重试"
      />
    </el-form-item>
    <el-form-item label="Reflexion 阈值">
      <el-input-number
        v-model="form.reflexionThreshold"
        :min="0"
        :max="1"
        :step="0.05"
        controls-position="right"
        placeholder="质量达标阈值 0-1"
      />
    </el-form-item>
    <el-form-item label="温度">
      <el-input-number
        v-model="form.temperature"
        :min="0"
        :max="2"
        :step="0.1"
        controls-position="right"
      />
    </el-form-item>

    <el-divider content-position="left"
      >Skills / MCP 命名空间（覆盖式保存）</el-divider
    >
    <el-form-item label="Skills">
      <el-select
        v-model="form.skills"
        class="!w-[560px]"
        multiple
        filterable
        clearable
        placeholder="该 Agent 可加载的 Skills 清单"
      >
        <el-option
          v-for="s in skillOptions"
          :key="s.name"
          :label="s.name"
          :value="s.name"
        />
      </el-select>
    </el-form-item>
    <el-form-item label="MCP 命名空间">
      <el-select
        v-model="form.mcpNamespaces"
        class="!w-[560px]"
        multiple
        filterable
        clearable
        placeholder="该 Agent 可访问的 MCP 工具分组"
      >
        <el-option
          v-for="n in mcpNamespaceOptions"
          :key="n"
          :label="n"
          :value="n"
        />
      </el-select>
    </el-form-item>

    <el-divider content-position="left">子 Agent（覆盖式保存）</el-divider>
    <template v-if="form.subagents.length">
      <el-form-item
        v-for="(item, index) in form.subagents"
        :key="index"
        label-width="0"
      >
        <div class="flex items-center gap-2 w-[560px]">
          <el-select
            v-model="item.kind"
            class="!w-[130px]"
            @change="onSubAgentKindChange(item)"
          >
            <el-option label="本地 Agent" value="local" />
            <el-option label="远程 A2A" value="remote" />
          </el-select>
          <el-select
            v-if="item.kind === 'local'"
            v-model="item.agentId"
            class="flex-1"
            filterable
            placeholder="选择本地子 Agent"
          >
            <el-option
              v-for="a in localAgentOptions"
              :key="a.id"
              :label="`${a.name} (${a.agentCode})`"
              :value="a.id"
            />
          </el-select>
          <el-select
            v-else
            v-model="item.endpointId"
            class="flex-1"
            filterable
            placeholder="选择外部 A2A 端点"
          >
            <el-option
              v-for="e in endpointOptions"
              :key="e.id"
              :label="e.name"
              :value="e.id"
            />
          </el-select>
          <el-input-number
            v-model="item.priority"
            :min="0"
            controls-position="right"
            placeholder="优先级"
          />
          <span class="text-xs text-gray-400">优先级</span>
          <el-button
            link
            type="danger"
            @click="form.subagents.splice(index, 1)"
          >
            移除
          </el-button>
        </div>
      </el-form-item>
    </template>
    <el-form-item label-width="0">
      <el-button type="primary" plain @click="addSubAgent">
        <el-icon><Plus /></el-icon>添加子 Agent
      </el-button>
      <span class="ml-3 text-xs text-gray-400">
        本地子 Agent 拥有独立上下文且不可再委派；远程 A2A 子 Agent 不计平台配额
      </span>
    </el-form-item>

    <el-divider content-position="left">文件系统权限</el-divider>
    <template v-if="form.permissions.length">
      <el-form-item
        v-for="(perm, index) in form.permissions"
        :key="index"
        label-width="0"
      >
        <div class="flex items-center gap-2 w-[560px]">
          <el-select v-model="perm.mode" class="!w-[120px]">
            <el-option label="只读" value="read" />
            <el-option label="读写" value="write" />
          </el-select>
          <el-input
            v-model="perm.path"
            placeholder="目录或文件 glob，如 /workspace/**"
          />
          <el-button
            link
            type="danger"
            @click="form.permissions.splice(index, 1)"
          >
            移除
          </el-button>
        </div>
      </el-form-item>
    </template>
    <el-form-item label-width="0">
      <el-button
        type="primary"
        plain
        @click="form.permissions.push({ mode: 'read', path: '' })"
      >
        <el-icon><Plus /></el-icon>添加权限规则
      </el-button>
    </el-form-item>

    <el-divider content-position="left">安全护栏</el-divider>
    <el-form-item label="覆盖系统默认">
      <el-switch v-model="overrideGuardrails" />
      <span class="ml-2 text-xs text-gray-400">
        关闭时继承系统默认护栏；开启后按下方规则覆盖
      </span>
    </el-form-item>
    <template v-if="overrideGuardrails">
      <el-form-item label="输入护栏">
        <el-checkbox v-model="form.guardrails.promptInjection!.enabled"
          >Prompt 注入防护</el-checkbox
        >
        <el-checkbox v-model="form.guardrails.unauthorizedAccess!.enabled"
          >越权查询检测</el-checkbox
        >
        <el-checkbox v-model="form.guardrails.sensitiveTopic!.enabled"
          >敏感话题过滤</el-checkbox
        >
      </el-form-item>
      <el-form-item label="输出护栏">
        <el-checkbox v-model="form.guardrails.piiMask!.enabled"
          >PII 脱敏</el-checkbox
        >
        <el-checkbox v-model="form.guardrails.factCheck!.enabled"
          >事实性校验</el-checkbox
        >
        <el-checkbox v-model="form.guardrails.formatCheck!.enabled"
          >输出格式合规校验</el-checkbox
        >
      </el-form-item>
    </template>

    <el-form-item label-width="0">
      <el-button
        v-hasPerm="['ai:agent:manage']"
        type="primary"
        :loading="submitting"
        @click="submit"
      >
        保 存
      </el-button>
    </el-form-item>
  </el-form>
</template>

<script lang="ts" setup>
import { Plus } from "@element-plus/icons-vue";
import {
  AiAgentAPI,
  AiModelAPI,
  AiMCPAPI,
  AiSkillAPI,
  AgentDetail,
  AgentListItem,
  AgentSubAgentItem,
  EndpointResult,
  GuardrailConfig,
  ReasoningMode,
  SkillVO,
} from "dehaze-sdk-js";
import {
  AgentFormPayload,
  useAdminAgentStore,
} from "@/store/modules/adminAgent";

defineOptions({ name: "AgentConfigForm" });

const props = defineProps<{ agentId: number | null }>();
const emit = defineEmits<{ saved: [] }>();

const agentStore = useAdminAgentStore();

const formRef = ref(ElForm);
const submitting = ref(false);

/** 表单内子 Agent 行：kind 仅用于本地/远程切换，提交时拆为 agentId/endpointId */
interface SubAgentRow {
  kind: "local" | "remote";
  agentId?: number;
  endpointId?: number;
  priority: number;
}

const emptyForm = () => ({
  name: "",
  agentCode: "",
  description: "",
  /** 逗号分隔的输入态，提交时拆为 string[] */
  tags: "",
  systemPrompt: "",
  modelId: "",
  reasoningMode: "auto" as ReasoningMode,
  maxSteps: null as number | null,
  tokenBudget: null as number | null,
  maxParallel: null as number | null,
  toolTimeout: null as number | null,
  retryMax: null as number | null,
  reflexionThreshold: null as number | null,
  temperature: null as number | null,
  guardrails: {
    promptInjection: { enabled: true },
    unauthorizedAccess: { enabled: true },
    sensitiveTopic: { enabled: true },
    piiMask: { enabled: true },
    factCheck: { enabled: false },
    formatCheck: { enabled: false },
  } as Required<GuardrailConfig>,
  isSubagent: false,
  isTeam: false,
  isExposed: false,
  sortOrder: 0,
  status: 1 as 0 | 1,
  skills: [] as string[],
  mcpNamespaces: [] as string[],
  subagents: [] as SubAgentRow[],
  permissions: [] as Array<{ mode: string; path: string }>,
});

const form = reactive(emptyForm());
/** 护栏覆盖开关：关闭时提交 config.guardrails = null（继承系统默认） */
const overrideGuardrails = ref(false);

const rules = {
  name: [{ required: true, message: "名称不能为空", trigger: "blur" }],
  agentCode: [
    { required: true, message: "唯一编码不能为空", trigger: "blur" },
    {
      pattern: /^[a-z0-9][a-z0-9-]*$/,
      message: "编码仅支持小写字母/数字/中划线",
      trigger: "blur",
    },
  ],
  modelId: [{ required: true, message: "对话模型不能为空", trigger: "change" }],
};

// ==================== 表单选项 ====================
const modelOptions = ref<
  Awaited<ReturnType<typeof AiModelAPI.listEnabledModels>>
>([]);
const skillOptions = ref<SkillVO[]>([]);
const mcpNamespaceOptions = ref<string[]>([]);
const localAgentOptions = ref<AgentListItem[]>([]);
const endpointOptions = ref<EndpointResult[]>([]);

async function loadOptions() {
  const [models, skills] = await Promise.all([
    AiModelAPI.listEnabledModels("chat"),
    AiSkillAPI.listSkills({ pageNum: 1, pageSize: 200, status: 1 }),
  ]);
  modelOptions.value = models ?? [];
  skillOptions.value = skills.list ?? [];

  // 命名空间按启用 Server 逐个拉取后合并去重
  const serversPage = await AiMCPAPI.listServers({
    pageNum: 1,
    pageSize: 200,
    status: 1,
  });
  const namespaceLists = await Promise.all(
    (serversPage.list ?? []).map((server) => AiMCPAPI.getNamespaces(server.id))
  );
  mcpNamespaceOptions.value = [
    ...new Set(namespaceLists.flatMap((ns) => (ns ?? []).map((n) => n.name))),
  ];

  const [enabledAgents, endpointsPage] = await Promise.all([
    AiAgentAPI.listEnabled(),
    AiAgentAPI.listEndpoints({ pageNum: 1, pageSize: 100 }),
  ]);
  // 子 Agent 候选排除自身，避免自引用
  localAgentOptions.value = (enabledAgents ?? []).filter(
    (a) => a.id !== props.agentId
  );
  endpointOptions.value = endpointsPage.list ?? [];
}

// ==================== 详情回填 ====================
watch(
  () => props.agentId,
  async (agentId) => {
    Object.assign(form, emptyForm());
    overrideGuardrails.value = false;
    loadOptions();
    if (agentId == null) return;
    const detail: AgentDetail = await agentStore.fetchAgentDetail(agentId);
    const config = detail.config ?? {};
    Object.assign(form, {
      name: detail.name,
      agentCode: detail.agentCode,
      description: detail.description,
      tags: (detail.tags ?? []).join(","),
      systemPrompt: detail.systemPrompt ?? "",
      modelId: detail.modelId,
      reasoningMode: (detail.reasoningMode as ReasoningMode) ?? "auto",
      maxSteps: config.maxSteps ?? null,
      tokenBudget: config.tokenBudget ?? null,
      maxParallel: config.maxParallel ?? null,
      toolTimeout: config.toolTimeout ?? null,
      retryMax: config.retryMax ?? null,
      reflexionThreshold: config.reflexionThreshold ?? null,
      temperature: config.temperature ?? null,
      guardrails: {
        promptInjection: config.guardrails?.promptInjection ?? {
          enabled: true,
        },
        unauthorizedAccess: config.guardrails?.unauthorizedAccess ?? {
          enabled: true,
        },
        sensitiveTopic: config.guardrails?.sensitiveTopic ?? { enabled: true },
        piiMask: config.guardrails?.piiMask ?? { enabled: true },
        factCheck: config.guardrails?.factCheck ?? { enabled: false },
        formatCheck: config.guardrails?.formatCheck ?? { enabled: false },
      },
      isSubagent: detail.isSubagent === 1,
      isTeam: detail.isTeam === 1,
      isExposed: detail.isExposed === 1,
      sortOrder: detail.sortOrder,
      skills: detail.skills ?? [],
      mcpNamespaces: detail.mcpNamespaces ?? [],
      subagents: (detail.subagents ?? []).map((s) => ({
        kind: s.endpointId != null ? ("remote" as const) : ("local" as const),
        agentId: s.endpointId != null ? undefined : s.agentId,
        endpointId: s.endpointId ?? undefined,
        priority: s.priority,
      })),
      permissions: (detail.permissions ?? []).map((p) => ({
        mode: String(p.mode ?? "read"),
        path: String(p.path ?? ""),
      })),
    });
    overrideGuardrails.value = !!config.guardrails;
    if (agentId != null) {
      form.status = detail.status;
    }
  },
  { immediate: true }
);

// ==================== 子 Agent 行操作 ====================
function addSubAgent() {
  form.subagents.push({ kind: "local", priority: 0 });
}

/** 本地/远程切换时清空另一侧的引用 */
function onSubAgentKindChange(item: SubAgentRow) {
  item.agentId = undefined;
  item.endpointId = undefined;
}

// ==================== 提交 ====================
function buildGuardrails(): GuardrailConfig {
  return {
    promptInjection: { enabled: form.guardrails.promptInjection!.enabled },
    unauthorizedAccess: {
      enabled: form.guardrails.unauthorizedAccess!.enabled,
    },
    sensitiveTopic: { enabled: form.guardrails.sensitiveTopic!.enabled },
    piiMask: { enabled: form.guardrails.piiMask!.enabled },
    factCheck: { enabled: form.guardrails.factCheck!.enabled },
    formatCheck: { enabled: form.guardrails.formatCheck!.enabled },
  };
}

function buildSubAgents(): AgentSubAgentItem[] {
  return form.subagents
    .filter((item) =>
      item.kind === "local" ? item.agentId != null : item.endpointId != null
    )
    .map((item) => ({
      agentId: item.agentId ?? 0,
      endpointId: item.kind === "remote" ? (item.endpointId ?? null) : null,
      priority: item.priority,
    }));
}

async function submit() {
  await formRef.value.validate();
  const invalidSubAgent = form.subagents.some((item) =>
    item.kind === "local" ? item.agentId == null : item.endpointId == null
  );
  if (invalidSubAgent) {
    ElMessage.error("存在未选择目标子 Agent 的行，请补全或移除");
    return;
  }

  submitting.value = true;
  try {
    const payload: AgentFormPayload = {
      name: form.name,
      agentCode: form.agentCode,
      description: form.description,
      tags: form.tags
        .split(/[,，]/)
        .map((tag) => tag.trim())
        .filter(Boolean),
      systemPrompt: form.systemPrompt || null,
      modelId: form.modelId,
      reasoningMode: form.reasoningMode,
      config: {
        maxSteps: form.maxSteps,
        tokenBudget: form.tokenBudget,
        maxParallel: form.maxParallel,
        toolTimeout: form.toolTimeout,
        retryMax: form.retryMax,
        reflexionThreshold: form.reflexionThreshold,
        temperature: form.temperature,
        guardrails: overrideGuardrails.value ? buildGuardrails() : null,
      },
      isSubagent: form.isSubagent,
      isTeam: form.isTeam,
      isExposed: form.isExposed,
      permissions: form.permissions.filter((p) => p.path),
      sortOrder: form.sortOrder,
      status: form.status,
      skills: form.skills,
      mcpNamespaces: form.mcpNamespaces,
      subagents: buildSubAgents(),
    };
    await agentStore.saveAgent(payload, props.agentId);
    ElMessage.success("已保存，生成草稿快照");
    emit("saved");
  } finally {
    submitting.value = false;
  }
}
</script>
