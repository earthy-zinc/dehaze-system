import type {
  AsrResultVO,
  HotwordVO,
  ServiceStatusVO,
  TtsResultVO,
  VoiceModelVO,
  VoiceProviderKeyVO,
  VoiceProviderVO,
  VoiceVO,
} from "dehaze-sdk-js";
import { defineMock } from "./base";

let nextHotwordId = 4;
let nextGlobalHotwordId = 6;
let nextSessionSeq = 1;
let asrCorpusIndex = 0;

const hotwords: HotwordVO[] = [
  { id: 1, word: "去雾", createTime: "2026-08-12 10:24:31" },
  { id: 2, word: "暗通道", createTime: "2026-08-12 10:25:07" },
  { id: 3, word: "RIDCP", createTime: "2026-08-19 16:40:52" },
];

const globalHotwords: HotwordVO[] = [
  { id: 1, word: "去雾算法", createTime: "2026-07-01 09:00:00" },
  { id: 2, word: "图像增强", createTime: "2026-07-01 09:00:00" },
  { id: 3, word: "暗通道先验", createTime: "2026-07-01 09:00:00" },
  { id: 4, word: "大气散射模型", createTime: "2026-07-15 14:32:18" },
  { id: 5, word: "透射率估计", createTime: "2026-07-15 14:33:05" },
];

// 后端 VOICE_CATALOG 当前仅 huayan，后续扩展后 mock 同步
const voices: VoiceVO[] = [
  {
    id: "huayan",
    name: "华燕",
    description: "中文女声，清晰自然",
    tags: ["女声", "中文"],
  },
];

/** 去雾/增强领域语料，模拟流式 ASR 的最终识别文本 */
const ASR_CORPUS = [
  "帮我把这张雾天图片做一次去雾处理，用暗通道先验算法。",
  "调整一下透射率估计的窗口半径，看看对细节保留有什么影响。",
  "对比一下 Retinex 和直方图均衡在低照度图像上的增强效果。",
  "查询这个月语音合成和语音识别的调用量统计。",
  "把当前去雾结果的 PSNR 和 SSIM 指标导出成报告。",
];

const asrResults = new Map<string, AsrResultVO>();

function formatNow() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(
    d.getHours()
  )}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/** mock 无真实 TTS 引擎，返回可播放的静音 WAV，保证前端播放链路可验证 */
function buildSilentWavDataUrl(seconds: number) {
  const sampleRate = 16000;
  const dataSize = Math.floor(sampleRate * seconds) * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);
  return `data:audio/wav;base64,${buffer.toString("base64")}`;
}

export default defineMock([
  // 全局热词列表
  {
    url: "voice/hotwords/global",
    method: ["GET"],
    body() {
      return { code: "00000", data: globalHotwords, msg: "一切ok" };
    },
  },

  // 新增全局热词
  {
    url: "voice/hotwords/global",
    method: ["POST"],
    body({ body }) {
      const hotword: HotwordVO = {
        id: nextGlobalHotwordId++,
        word: body.word,
        createTime: formatNow(),
      };
      globalHotwords.push(hotword);
      return {
        code: "00000",
        data: hotword,
        msg: "新增全局热词" + hotword.word + "成功",
      };
    },
  },

  // 删除全局热词
  {
    url: "voice/hotwords/global/:id",
    method: ["DELETE"],
    body({ params }) {
      const index = globalHotwords.findIndex(
        (item) => item.id === Number(params.id)
      );
      if (index === -1) {
        return { code: "A0401", data: null, msg: "全局热词不存在" };
      }
      const [hotword] = globalHotwords.splice(index, 1);
      return {
        code: "00000",
        data: null,
        msg: "删除全局热词" + hotword.word + "成功",
      };
    },
  },

  // 用户热词列表
  {
    url: "voice/hotwords",
    method: ["GET"],
    body() {
      return { code: "00000", data: hotwords, msg: "一切ok" };
    },
  },

  // 新增用户热词
  {
    url: "voice/hotwords",
    method: ["POST"],
    body({ body }) {
      const hotword: HotwordVO = {
        id: nextHotwordId++,
        word: body.word,
        createTime: formatNow(),
      };
      hotwords.push(hotword);
      return {
        code: "00000",
        data: hotword,
        msg: "新增热词" + hotword.word + "成功",
      };
    },
  },

  // 删除用户热词
  {
    url: "voice/hotwords/:id",
    method: ["DELETE"],
    body({ params }) {
      const index = hotwords.findIndex((item) => item.id === Number(params.id));
      if (index === -1) {
        return { code: "A0401", data: null, msg: "热词不存在" };
      }
      const [hotword] = hotwords.splice(index, 1);
      return {
        code: "00000",
        data: null,
        msg: "删除热词" + hotword.word + "成功",
      };
    },
  },

  // 音色列表
  {
    url: "voice/tts/voices",
    method: ["GET"],
    body() {
      return { code: "00000", data: voices, msg: "一切ok" };
    },
  },

  // 文本转语音
  {
    url: "voice/tts",
    method: ["POST"],
    body({ body }) {
      const result: TtsResultVO = {
        audioUrl: buildSilentWavDataUrl(0.2),
        format: body.format ?? "mp3",
      };
      return { code: "00000", data: result, msg: "一切ok" };
    },
  },

  // 创建流式 ASR 会话
  {
    url: "voice/asr/stream-session",
    method: ["POST"],
    body() {
      const sessionId = `asr-${Date.now()}-${nextSessionSeq++}`;
      const result: AsrResultVO = { sessionId, text: "", status: "processing" };
      asrResults.set(sessionId, result);
      // 真实链路经 WebSocket 推送音频后出结果，mock 无 WebSocket，延时置为完成以便查询结果
      setTimeout(() => {
        result.text = ASR_CORPUS[asrCorpusIndex++ % ASR_CORPUS.length];
        result.status = "completed";
      }, 2000);
      return {
        code: "00000",
        data: {
          sessionId,
          wsUrl: `ws://127.0.0.1:8991/ws/asr?sessionId=${sessionId}`,
        },
        msg: "一切ok",
      };
    },
  },

  // 查询流式 ASR 识别结果
  {
    url: "voice/asr/result/:sessionId",
    method: ["GET"],
    body({ params }) {
      const result = asrResults.get(params.sessionId);
      if (!result) {
        return { code: "A0401", data: null, msg: "识别会话不存在" };
      }
      return { code: "00000", data: result, msg: "一切ok" };
    },
  },

  // 语音服务状态
  {
    url: "voice/service/status",
    method: ["GET"],
    body() {
      const status: ServiceStatusVO = {
        asr: {
          engineStatus: "online",
          concurrentSessions: [...asrResults.values()].filter(
            (item) => item.status === "processing"
          ).length,
          maxConcurrentSessions: 50,
          streamModelLoaded: true,
          offlineModelLoaded: true,
        },
        tts: {
          engineStatus: "online",
          voiceModelLoaded: true,
        },
      };
      return { code: "00000", data: status, msg: "一切ok" };
    },
  },

  // ===== 语音引擎注册表管理（管理端）=====

  // 指定能力维度启用引擎列表（置于 :id 路由前，避免被路径参数匹配）
  {
    url: "voice/providers/enabled",
    method: ["GET"],
    body({ query }) {
      const list = providers.filter(
        (item) => item.status === 1 && item.engineType === query.engineType
      );
      return { code: "00000", data: list, msg: "一切ok" };
    },
  },

  // 引擎分页列表
  {
    url: "voice/providers",
    method: ["GET"],
    body({ query }) {
      let list = [...providers];
      if (query.keyword) {
        const kw = query.keyword.toLowerCase();
        list = list.filter(
          (item) =>
            item.displayName.toLowerCase().includes(kw) ||
            item.providerCode.toLowerCase().includes(kw)
        );
      }
      if (query.engineType) {
        list = list.filter((item) => item.engineType === query.engineType);
      }
      const pageNum = Number(query.pageNum ?? 1);
      const pageSize = Number(query.pageSize ?? 10);
      const start = (pageNum - 1) * pageSize;
      return {
        code: "00000",
        data: { list: list.slice(start, start + pageSize), total: list.length },
        msg: "一切ok",
      };
    },
  },

  // 新增引擎
  {
    url: "voice/providers",
    method: ["POST"],
    body({ body }) {
      const exists = providers.some(
        (item) =>
          item.providerCode === body.providerCode &&
          item.engineType === body.engineType
      );
      if (exists) {
        return { code: "A0500", data: null, msg: "引擎编码已存在" };
      }
      const provider: VoiceProviderVO = {
        id: nextProviderId++,
        providerCode: body.providerCode,
        engineType: body.engineType,
        displayName: body.displayName,
        apiBaseUrl: body.apiBaseUrl ?? null,
        authType: body.authType ?? "bearer",
        defaultHeaders: body.defaultHeaders ?? null,
        isDefault: body.isDefault ?? 0,
        sortOrder: body.sortOrder ?? 0,
        healthCheckEnabled: body.healthCheckEnabled ?? 1,
        remark: body.remark ?? null,
        status: body.status ?? 1,
        createTime: formatNow(),
        updateTime: formatNow(),
      };
      applyProviderDefault(provider);
      providers.push(provider);
      return { code: "00000", data: provider, msg: "新增引擎成功" };
    },
  },

  // 更新引擎
  {
    url: "voice/providers/:id",
    method: ["PUT"],
    body({ body, params }) {
      const provider = providers.find((item) => item.id === Number(params.id));
      if (!provider) {
        return { code: "A0401", data: null, msg: "引擎不存在" };
      }
      const updatable = [
        "displayName",
        "apiBaseUrl",
        "authType",
        "defaultHeaders",
        "isDefault",
        "sortOrder",
        "healthCheckEnabled",
        "remark",
        "status",
      ] as const;
      for (const key of updatable) {
        if (body[key] !== undefined) {
          (provider as unknown as Record<string, unknown>)[key] = body[key];
        }
      }
      provider.updateTime = formatNow();
      applyProviderDefault(provider);
      return { code: "00000", data: provider, msg: "修改引擎成功" };
    },
  },

  // 删除引擎
  {
    url: "voice/providers/:id",
    method: ["DELETE"],
    body({ params }) {
      const index = providers.findIndex(
        (item) => item.id === Number(params.id)
      );
      if (index === -1) {
        return { code: "A0401", data: null, msg: "引擎不存在" };
      }
      const provider = providers[index];
      const hasEnabledModel = models.some(
        (item) => item.providerId === provider.id && item.status === 1
      );
      if (hasEnabledModel) {
        return {
          code: "A0500",
          data: null,
          msg: "存在启用模型引用该引擎，请先禁用或删除关联模型",
        };
      }
      providers.splice(index, 1);
      return { code: "00000", data: null, msg: "删除引擎成功" };
    },
  },

  // 引擎连通性测试
  {
    url: "voice/providers/:id/test-connection",
    method: ["POST"],
    body({ params }) {
      const provider = providers.find((item) => item.id === Number(params.id));
      if (!provider) {
        return { code: "A0401", data: null, msg: "引擎不存在" };
      }
      // local 引擎进程内可用即连通；云端测试能力后端待接入
      const connected = provider.providerCode === "local" ? true : null;
      const result = provider.providerCode === "local" ? "本地引擎" : "待实现";
      return { code: "00000", data: { result, connected }, msg: "一切ok" };
    },
  },

  // 引擎 API Key 列表
  {
    url: "voice/providers/:id/keys",
    method: ["GET"],
    body({ params }) {
      const provider = providers.find((item) => item.id === Number(params.id));
      if (!provider) {
        return { code: "A0401", data: null, msg: "引擎不存在" };
      }
      const list = keys
        .filter((item) => item.providerId === provider.id)
        .sort((a, b) => a.priority - b.priority);
      return { code: "00000", data: list, msg: "一切ok" };
    },
  },

  // 新增引擎 API Key
  {
    url: "voice/providers/:id/keys",
    method: ["POST"],
    body({ body, params }) {
      const provider = providers.find((item) => item.id === Number(params.id));
      if (!provider) {
        return { code: "A0401", data: null, msg: "引擎不存在" };
      }
      const key: VoiceProviderKeyVO = {
        id: nextKeyId++,
        providerId: provider.id,
        name: body.name,
        keyPrefix: maskKey(body.key),
        status: body.status ?? 1,
        priority: body.priority ?? 0,
        weight: body.weight ?? 1,
        dailyQuota: body.dailyQuota ?? null,
        rpmLimit: body.rpmLimit ?? null,
        expiresAt: body.expiresAt ?? null,
        lastUsedAt: null,
        lastUsedBy: null,
        createTime: formatNow(),
        updateTime: formatNow(),
      };
      keys.push(key);
      return { code: "00000", data: key, msg: "新增API Key成功" };
    },
  },

  // 更新引擎 API Key
  {
    url: "voice/providers/:id/keys/:keyId",
    method: ["PUT"],
    body({ body, params }) {
      const key = keys.find(
        (item) =>
          item.id === Number(params.keyId) &&
          item.providerId === Number(params.id)
      );
      if (!key) {
        return { code: "A0401", data: null, msg: "API Key 不存在" };
      }
      const updatable = [
        "name",
        "status",
        "priority",
        "weight",
        "dailyQuota",
        "rpmLimit",
        "expiresAt",
      ] as const;
      for (const field of updatable) {
        if (body[field] !== undefined) {
          (key as unknown as Record<string, unknown>)[field] = body[field];
        }
      }
      key.updateTime = formatNow();
      return { code: "00000", data: key, msg: "修改API Key成功" };
    },
  },

  // 删除引擎 API Key（物理删除）
  {
    url: "voice/providers/:id/keys/:keyId",
    method: ["DELETE"],
    body({ params }) {
      const index = keys.findIndex(
        (item) =>
          item.id === Number(params.keyId) &&
          item.providerId === Number(params.id)
      );
      if (index === -1) {
        return { code: "A0401", data: null, msg: "API Key 不存在" };
      }
      keys.splice(index, 1);
      return { code: "00000", data: null, msg: "删除API Key成功" };
    },
  },

  // 模型/音色列表
  {
    url: "voice/models",
    method: ["GET"],
    body({ query }) {
      let list = [...models];
      if (query.engineType) {
        list = list.filter((item) => item.engineType === query.engineType);
      }
      return { code: "00000", data: list, msg: "一切ok" };
    },
  },

  // 新增模型/音色
  {
    url: "voice/models",
    method: ["POST"],
    body({ body }) {
      const provider = providers.find((item) => item.id === body.providerId);
      if (!provider) {
        return { code: "A0401", data: null, msg: "引擎不存在" };
      }
      const exists = models.some(
        (item) =>
          item.modelId === body.modelId && item.providerId === body.providerId
      );
      if (exists) {
        return { code: "A0500", data: null, msg: "该引擎+模型组合已存在" };
      }
      const model: VoiceModelVO = {
        id: nextModelId++,
        providerId: body.providerId,
        modelId: body.modelId,
        engineType: body.engineType,
        modelType: body.modelType,
        displayName: body.displayName,
        params: body.params ?? null,
        status: body.status ?? 1,
        createTime: formatNow(),
        updateTime: formatNow(),
      };
      models.push(model);
      return { code: "00000", data: model, msg: "新增模型成功" };
    },
  },

  // 更新模型/音色
  {
    url: "voice/models/:id",
    method: ["PUT"],
    body({ body, params }) {
      const model = models.find((item) => item.id === Number(params.id));
      if (!model) {
        return { code: "A0401", data: null, msg: "模型不存在" };
      }
      for (const field of ["displayName", "params", "status"] as const) {
        if (body[field] !== undefined) {
          (model as unknown as Record<string, unknown>)[field] = body[field];
        }
      }
      model.updateTime = formatNow();
      return { code: "00000", data: model, msg: "修改模型成功" };
    },
  },

  // 删除模型/音色
  {
    url: "voice/models/:id",
    method: ["DELETE"],
    body({ params }) {
      const index = models.findIndex((item) => item.id === Number(params.id));
      if (index === -1) {
        return { code: "A0401", data: null, msg: "模型不存在" };
      }
      models.splice(index, 1);
      return { code: "00000", data: null, msg: "删除模型成功" };
    },
  },
]);

// ===== 语音引擎注册表 mock 数据（对齐后端引擎注册表架构 F-VS-005）=====

let nextProviderId = 5;
let nextKeyId = 2;
let nextModelId = 4;

const providers: VoiceProviderVO[] = [
  {
    id: 1,
    providerCode: "local",
    engineType: "asr",
    displayName: "本地 FunASR 引擎",
    apiBaseUrl: null,
    authType: "bearer",
    defaultHeaders: null,
    isDefault: 1,
    sortOrder: 0,
    healthCheckEnabled: 1,
    remark: "内置本地流式/离线识别引擎",
    status: 1,
    createTime: "2026-08-01 09:00:00",
    updateTime: "2026-08-01 09:00:00",
  },
  {
    id: 2,
    providerCode: "local",
    engineType: "tts",
    displayName: "本地 Piper 引擎",
    apiBaseUrl: null,
    authType: "bearer",
    defaultHeaders: null,
    isDefault: 1,
    sortOrder: 0,
    healthCheckEnabled: 1,
    remark: "内置本地语音合成引擎",
    status: 1,
    createTime: "2026-08-01 09:00:00",
    updateTime: "2026-08-01 09:00:00",
  },
  {
    id: 3,
    providerCode: "aliyun",
    engineType: "asr",
    displayName: "阿里云语音识别",
    apiBaseUrl: "https://nls-gateway.cn-shanghai.aliyuncs.com",
    authType: "bearer",
    defaultHeaders: null,
    isDefault: 0,
    sortOrder: 10,
    healthCheckEnabled: 1,
    remark: null,
    status: 1,
    createTime: "2026-08-01 09:00:00",
    updateTime: "2026-08-01 09:00:00",
  },
  {
    id: 4,
    providerCode: "aliyun",
    engineType: "tts",
    displayName: "阿里云语音合成",
    apiBaseUrl: "https://nls-gateway.cn-shanghai.aliyuncs.com",
    authType: "x-api-key",
    defaultHeaders: { "X-Nls-Region": "cn-shanghai" },
    isDefault: 0,
    sortOrder: 10,
    healthCheckEnabled: 0,
    remark: null,
    status: 0,
    createTime: "2026-08-01 09:00:00",
    updateTime: "2026-08-01 09:00:00",
  },
];

const keys: VoiceProviderKeyVO[] = [
  {
    id: 1,
    providerId: 3,
    name: "主 Key",
    keyPrefix: "sk-1a2b****",
    status: 1,
    priority: 0,
    weight: 1,
    dailyQuota: 1000,
    rpmLimit: 60,
    expiresAt: null,
    lastUsedAt: "2026-08-28 18:30:12",
    lastUsedBy: 2,
    createTime: "2026-08-01 09:00:00",
    updateTime: "2026-08-01 09:00:00",
  },
];

const models: VoiceModelVO[] = [
  {
    id: 1,
    providerId: 1,
    modelId: "sensevoice",
    engineType: "asr",
    modelType: "stream",
    displayName: "SenseVoice 流式识别",
    params: { sampleRate: 16000 },
    status: 1,
    createTime: "2026-08-01 09:00:00",
    updateTime: "2026-08-01 09:00:00",
  },
  {
    id: 2,
    providerId: 1,
    modelId: "paraformer",
    engineType: "asr",
    modelType: "offline",
    displayName: "Paraformer 离线识别",
    params: { sampleRate: 16000 },
    status: 1,
    createTime: "2026-08-01 09:00:00",
    updateTime: "2026-08-01 09:00:00",
  },
  {
    id: 3,
    providerId: 2,
    modelId: "huayan",
    engineType: "tts",
    modelType: "voice",
    displayName: "华燕（中文女声）",
    params: null,
    status: 1,
    createTime: "2026-08-01 09:00:00",
    updateTime: "2026-08-01 09:00:00",
  },
];

/** 设为默认引擎时清除同能力类型下其他默认，对齐后端行为 */
function applyProviderDefault(provider: VoiceProviderVO) {
  if (provider.isDefault !== 1) return;
  for (const other of providers) {
    if (other.engineType === provider.engineType && other.id !== provider.id) {
      other.isDefault = 0;
    }
  }
}

/** 密钥掩码前缀，对齐后端 mask_key 展示效果 */
function maskKey(key: string) {
  if (key.length <= 4) return "****";
  return `${key.slice(0, 4)}****`;
}
