import type {
  AsrResultVO,
  HotwordVO,
  ServiceStatusVO,
  TtsResultVO,
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

// mock 固定返回一个内置音色用于离线 UI 联调；真实音色列表由后端按默认 tts 模型所属协议族返回
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
          wsUrl: `/ws/asr?sessionId=${sessionId}`,
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
]);
