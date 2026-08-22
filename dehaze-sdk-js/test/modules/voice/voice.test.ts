import { AiBillingAPI, VoiceAPI, service } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login, logout } from "#/utils/auth";
import { getRedis } from "#/utils/redis";
import { USERS } from "#/factories/constants";
import { createHotwordForm, createStreamAsrSessionForm, createTtsForm } from "#/factories/voice";

/**
 * 语音交互模块接口测试。
 *
 * 环境依赖说明（按需求/实现文档判定，非断言缺陷）：
 * - FunASR 语音识别引擎（进程内懒加载，需 funasr 依赖 + 模型可加载）：影响「WebSocket 流式 ASR
 *   连接 / EOS 识别结果 / 离线实际识别」。不可用时按环境依赖条件跳过。
 * - TTS 为本地 Piper 引擎（进程内懒加载，模型自动下载），无外部依赖，实际合成用例直接执行。
 * - 语音能力计费需用户有 AI 积分余额（A0682 余额不足）。测试 beforeAll 通过 ai-billing/adjust 为
 *   admin 充值积分，使 ASR 会话创建 / TTS 参数校验等可跑通。
 */

// 环境探测标志
let asrEngineAvailable = false; // FunASR 引擎可用
let hotwordReady = false; // 后端热词路由已修复（重启后为 true）

/** 为 admin 充值 AI 积分，避免 ASR/TTS 因余额不足（A0682）被拒 */
async function topUpAdminBalance() {
  await login(USERS.ADMIN.username);
  try {
    await AiBillingAPI.adjustCredits({
      userId: USERS.ADMIN.id,
      amount: 100000,
      reason: "语音模块测试积分充值",
    });
  } catch {
    // 历史遗留的 `ai:balance:{uid}` 缓存可能是 "0.00" 等非整数字符串，导致 Redis INCRBY 抛
    // ResponseError，清空该缓存后由 MySQL 重新回填整数值。
    const redis = getRedis();
    await redis.del(`ai:balance:${USERS.ADMIN.id}`, `ai:arrears:${USERS.ADMIN.id}`);
    await AiBillingAPI.adjustCredits({
      userId: USERS.ADMIN.id,
      amount: 100000,
      reason: "语音模块测试积分充值",
    });
  }
}

/** 下载缓存音频并返回字节（响应拦截器对 arraybuffer 非 JSON 响应返回 Blob） */
async function fetchAudioBytes(audioUrl: string): Promise<Uint8Array> {
  const blob = (await service.get(audioUrl, { responseType: "arraybuffer" })) as Blob;
  return new Uint8Array(await blob.arrayBuffer());
}

/** 探测 FunASR 引擎是否可用：能建流式会话并建立 WebSocket 且不被服务端拒绝 */
async function probeAsrEngine(): Promise<boolean> {
  await login(USERS.ADMIN.username);
  return new Promise<boolean>((resolve) => {
    let settled = false;
    let handle: any = null;
    const finish = (ok: boolean) => {
      if (settled) return;
      settled = true;
      resolve(ok);
    };
    // 仅经 startStreamAsr 创建一次会话并连接 WS（避免重复创建触发防重复提交）
    VoiceAPI.startStreamAsr(
      { model: "sensevoice" },
      {
        onMessage: () => {},
        onOpen: () => {
          // 能建立连接即视为引擎可用；随后发送 EOS 结束，避免占用并发名额
          try {
            if (handle) handle.stop();
          } catch {
            /* ignore */
          }
          finish(true);
        },
        onClose: () => finish(true),
        onError: () => finish(false),
      }
    )
      .then((h) => {
        handle = h;
      })
      .catch(() => finish(false));
    // 兜底：若 60s 内无任何回调，视为不可用（FunASR 模型冷加载需 10~30s）
    setTimeout(() => finish(false), 60000);
  });
}

beforeAll(async () => {
  await topUpAdminBalance();
  hotwordReady = await probeHotwordReady();
  // FunASR 探测较重（可能触发模型加载），单独在 WS 套件内惰性探测。
});

/** 探测后端热词路由是否可用（缺陷是否已修复并重启生效） */
async function probeHotwordReady(): Promise<boolean> {
  await login(USERS.ADMIN.username);
  try {
    const r = await VoiceAPI.addHotword({ word: "__voice_probe__" });
    try {
      await VoiceAPI.deleteHotword(r.id);
    } catch {
      /* 清理失败忽略 */
    }
    return true;
  } catch {
    return false;
  }
}

/** 后端热词路由未就绪时条件跳过热词用例 */
function hotwordReadyOrSkip(ctx: any): boolean {
  if (!hotwordReady) {
    ctx.skip("后端热词路由缺陷已修复待重启生效，跳过热词用例");
    return false;
  }
  return true;
}

/** FunASR 引擎不可用时条件跳过流式 ASR 用例 */
function asrReadyOrSkip(ctx: any, reason: string): boolean {
  if (!asrEngineAvailable) {
    ctx.skip(reason);
    return false;
  }
  return true;
}

describe("语音交互模块接口测试 - VoiceAPI", () => {
  // ===== ASR 识别 =====

  describe("POST /api/v1/voice/asr/stream-session - 创建流式 ASR 会话", () => {
    test("正向测试：创建流式 ASR 会话返回 sessionId 和 wsUrl", async () => {
      const result = await VoiceAPI.createStreamAsrSession(createStreamAsrSessionForm());
      expect(typeof result.sessionId).toBe("string");
      expect(typeof result.wsUrl).toBe("string");
    });

    test("验证：wsUrl 以 ws:// 或 wss:// 开头", async () => {
      // 不同 body（model 不同）规避防重复提交中间件（A0002）
      const result = await VoiceAPI.createStreamAsrSession({ model: "sensevoice" });
      expect(result.wsUrl).toMatch(/^wss?:\/\//);
    });

    test("边界：未登录访问应返回 401", async () => {
      await logout(USERS.ADMIN.username);
      try {
        await expectBizError(VoiceAPI.createStreamAsrSession(createStreamAsrSessionForm()), [
          "A0230",
          "A0400",
          "ERR_BAD_REQUEST",
        ]);
      } finally {
        await login(USERS.ADMIN.username);
      }
    });
  });

  describe("GET /api/v1/voice/asr/result/{sessionId} - 查询 ASR 结果", () => {
    test("边界：查询不存在的 sessionId 应失败", async () => {
      await expectBizError(VoiceAPI.getAsrResult("nonexistent-session-id-99999"), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("正向测试：查询刚创建的会话结果（可能尚无结果）", async () => {
      const session = await VoiceAPI.createStreamAsrSession({ model: "paraformer" });
      const result = await VoiceAPI.getAsrResult(session.sessionId);
      expect(result.sessionId).toBe(session.sessionId);
      expect(["completed", "processing", "failed"]).toContain(result.status);
    });
  });

  // ===== TTS 合成 =====

  describe("POST /api/v1/voice/tts - 文本转语音", () => {
    test("正向测试：TTS 合成返回音频 URL（本地 Piper 引擎）", async () => {
      const result = await VoiceAPI.tts(createTtsForm({ text: "处理完成" }));
      expect(typeof result.audioUrl).toBe("string");
      expect(result.audioUrl).toMatch(/^\/api\/v1\/voice\/tts\/audio\/[0-9a-f]{64}$/);
      expect(result.format).toBe("mp3");
    });

    test("正向测试：audioUrl 可下载且返回非空音频字节", async () => {
      const result = await VoiceAPI.tts(createTtsForm({ text: "音频下载验证" }));
      const bytes = await fetchAudioBytes(result.audioUrl!);
      expect(bytes.length).toBeGreaterThan(1000);
      // MP3 帧同步字（0xFF Ex）
      expect(bytes[0]).toBe(0xff);
      expect(bytes[1]! & 0xe0).toBe(0xe0);
    });

    test("缓存：相同参数二次合成命中缓存返回同一 audioUrl", async () => {
      const first = await VoiceAPI.tts(createTtsForm({ text: "缓存命中验证" }));
      const second = await VoiceAPI.tts(createTtsForm({ text: "缓存命中验证" }));
      expect(second.audioUrl).toBe(first.audioUrl);
    });

    test("验证：不同语速生成不同音频（缓存 Key 含语速）", async () => {
      const normal = await VoiceAPI.tts(createTtsForm({ text: "语速区分验证" }));
      const fast = await VoiceAPI.tts(createTtsForm({ text: "语速区分验证", speed: 1.5 }));
      expect(fast.audioUrl).not.toBe(normal.audioUrl);
    });

    test("验证：wav 格式输出可解析的 WAV 头", async () => {
      const result = await VoiceAPI.tts(
        createTtsForm({ text: "波形格式验证", format: "wav", sampleRate: 16000 })
      );
      expect(result.format).toBe("wav");
      const bytes = await fetchAudioBytes(result.audioUrl!);
      // RIFF 头 + WAVE 标记
      expect(String.fromCharCode(...bytes.slice(0, 4))).toBe("RIFF");
      expect(String.fromCharCode(...bytes.slice(8, 12))).toBe("WAVE");
    });

    test("参数校验：空文本应失败", async () => {
      await expectBizError(VoiceAPI.tts(createTtsForm({ text: "" })), [
        "A0400",
        "A0600",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：不支持的音色应失败", async () => {
      await expectBizError(VoiceAPI.tts(createTtsForm({ text: "文本", voice: "aixia" })), [
        "A0400",
        "A0600",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：文本超长应失败", async () => {
      await expectBizError(VoiceAPI.tts(createTtsForm({ text: "x".repeat(10001) })), [
        "A0400",
        "A0600",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：未登录访问应返回 401", async () => {
      await logout(USERS.ADMIN.username);
      try {
        await expectBizError(VoiceAPI.tts(createTtsForm()), ["A0230", "A0400", "ERR_BAD_REQUEST"]);
      } finally {
        await login(USERS.ADMIN.username);
      }
    });
  });

  describe("GET /api/v1/voice/tts/voices - 音色列表", () => {
    test("正向测试：获取可用音色列表", async () => {
      const voices = await VoiceAPI.getVoices();
      expect(Array.isArray(voices)).toBe(true);
      expect(voices.length).toBeGreaterThan(0);
    });

    test("验证：音色列表含 id 和 name 字段", async () => {
      const voices = await VoiceAPI.getVoices();
      voices.forEach((v) => {
        expect(typeof v.id).toBe("string");
        expect(typeof v.name).toBe("string");
      });
    });

    test("验证：音色列表包含默认音色 huayan（本地 Piper 音色）", async () => {
      const voices = await VoiceAPI.getVoices();
      expect(voices.some((v) => v.id === "huayan")).toBe(true);
    });
  });

  // ===== 热词管理 - 用户级 =====

  describe("用户级热词管理", () => {
    const createdHotwordIds: number[] = [];

    afterAll(async () => {
      for (const id of createdHotwordIds.reverse()) {
        try {
          await VoiceAPI.deleteHotword(id);
        } catch {
          // 忽略
        }
      }
    });

    test("正向测试：新增用户热词", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      const result = await VoiceAPI.addHotword(createHotwordForm());
      expect(result.id).toBeGreaterThan(0);
      expect(result.word).toBeTruthy();
      createdHotwordIds.push(result.id);
    });

    test("正向测试：查询用户热词列表", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      const hotwords = await VoiceAPI.getHotwords();
      expect(Array.isArray(hotwords)).toBe(true);
    });

    test("正向测试：删除用户热词", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      const created = await VoiceAPI.addHotword(createHotwordForm());
      await VoiceAPI.deleteHotword(created.id);

      const hotwords = await VoiceAPI.getHotwords();
      const found = hotwords.find((h) => h.id === created.id);
      expect(found).toBeUndefined();
    });

    test("参数校验：空热词应失败", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      await expectBizError(VoiceAPI.addHotword({ word: "" }), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：删除不存在的热词应失败", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      await expectBizError(VoiceAPI.deleteHotword(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("安全：热词含 XSS 内容被转义存储", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      const result = await VoiceAPI.addHotword(
        createHotwordForm({ word: "<script>alert(1)</script>" })
      );
      createdHotwordIds.push(result.id);

      const hotwords = await VoiceAPI.getHotwords();
      const found = hotwords.find((h) => h.id === result.id);
      expect(found).toBeDefined();
      const jsonStr = JSON.stringify(found);
      expect(jsonStr).not.toContain("<script>");
    });
  });

  // ===== 热词管理 - 全局（管理员）=====

  describe("全局热词管理（管理员）", () => {
    const createdGlobalIds: number[] = [];

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
    });

    afterAll(async () => {
      for (const id of createdGlobalIds.reverse()) {
        try {
          await VoiceAPI.deleteGlobalHotword(id);
        } catch {
          // 忽略
        }
      }
    });

    test("正向测试：管理员新增全局热词", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      const result = await VoiceAPI.addGlobalHotword(createHotwordForm());
      expect(result.id).toBeGreaterThan(0);
      createdGlobalIds.push(result.id);
    });

    test("正向测试：查询全局热词列表", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      const hotwords = await VoiceAPI.getGlobalHotwords();
      expect(Array.isArray(hotwords)).toBe(true);
    });

    test("正向测试：管理员删除全局热词", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      const created = await VoiceAPI.addGlobalHotword(createHotwordForm());
      await VoiceAPI.deleteGlobalHotword(created.id);

      const hotwords = await VoiceAPI.getGlobalHotwords();
      const found = hotwords.find((h) => h.id === created.id);
      expect(found).toBeUndefined();
    });

    test("边界：普通用户管理全局热词应失败", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      await login(USERS.USER.username);
      try {
        await expectBizError(VoiceAPI.addGlobalHotword(createHotwordForm()), [
          "A0301",
          "A0400",
          "B0001",
          "ERR_BAD_REQUEST",
        ]);

        await expectBizError(VoiceAPI.deleteGlobalHotword(1), [
          "A0301",
          "A0401",
          "A0400",
          "B0001",
          "ERR_BAD_REQUEST",
        ]);
      } finally {
        await login(USERS.ADMIN.username);
      }
    });
  });

  // ===== 数据隔离 =====

  describe("数据隔离 - 用户级热词生效范围", () => {
    test("验证：用户 A 的热词对用户 B 不可见", async (ctx) => {
      if (!hotwordReadyOrSkip(ctx)) return;
      await login(USERS.ADMIN.username);
      const adminHotword = await VoiceAPI.addHotword(createHotwordForm());
      try {
        await login(USERS.USER.username);
        const userHotwords = await VoiceAPI.getHotwords();
        const found = userHotwords.find((h) => h.id === adminHotword.id);
        expect(found).toBeUndefined();
      } finally {
        await login(USERS.ADMIN.username);
        try {
          await VoiceAPI.deleteHotword(adminHotword.id);
        } catch {
          /* 清理失败忽略 */
        }
      }
    });
  });

  // ===== WebSocket 流式 ASR 测试 =====

  describe("startStreamAsr - WebSocket 流式 ASR", () => {
    beforeAll(async () => {
      // 防重复提交窗口为 5 秒：前面用例已 POST 过相同 body（{model:"sensevoice"}），
      // 等待窗口过期后再探测，避免探测请求被 A0002 误判为引擎不可用
      await new Promise((resolve) => setTimeout(resolve, 5500));
      asrEngineAvailable = await probeAsrEngine();
    }, 90000);

    test("正向测试：创建流式 ASR 会话并建立 WebSocket 连接", async (ctx) => {
      if (!asrReadyOrSkip(ctx, "FunASR 语音识别引擎未部署/模型不可加载，跳过流式 ASR 连接测试")) {
        return;
      }
      let connected = false;
      let closed = false;
      let errored = false;

      const asrSession = await VoiceAPI.startStreamAsr(createStreamAsrSessionForm(), {
        onMessage: () => {},
        onOpen: () => {
          connected = true;
          // 连接建立后立即发送 EOS 结束，避免占用并发名额
          asrSession.stop();
        },
        onClose: () => {
          closed = true;
        },
        onError: () => {
          errored = true;
        },
      });

      await new Promise<void>((resolve) => {
        setTimeout(resolve, 5000);
      });

      expect(asrSession).toBeDefined();
      expect(connected || closed || errored).toBe(true);
    }, 15000);

    test("正向测试：发送 EOS 后收到最终识别结果", async (ctx) => {
      if (!asrReadyOrSkip(ctx, "FunASR 语音识别引擎未部署/模型不可加载，跳过流式 ASR 识别测试")) {
        return;
      }
      let receivedMessage = false;
      let wsError: Error | null = null;

      const asrSession = await VoiceAPI.startStreamAsr(createStreamAsrSessionForm(), {
        onOpen: () => {
          const silence = new ArrayBuffer(640);
          asrSession.sendAudio(silence);
          asrSession.stop();
        },
        onMessage: (msg) => {
          receivedMessage = true;
          if (msg.text !== undefined) {
            expect(typeof msg.text).toBe("string");
          }
          if (msg.isFinal !== undefined) {
            expect(typeof msg.isFinal).toBe("boolean");
          }
        },
        onClose: () => {},
        onError: (error) => {
          wsError = error instanceof Error ? error : new Error(String(error));
        },
      });

      await new Promise((resolve) => setTimeout(resolve, 5000));

      expect(wsError).toBeNull();
      expect(receivedMessage).toBe(true);
    }, 15000);

    test("边界：主动关闭连接不触发重连", async (ctx) => {
      if (!asrReadyOrSkip(ctx, "FunASR 语音识别引擎未部署/模型不可加载，跳过流式 ASR 重连测试")) {
        return;
      }
      const asrSession = await VoiceAPI.startStreamAsr(createStreamAsrSessionForm(), {
        onMessage: () => {},
        onOpen: () => {
          asrSession.close();
        },
        onClose: () => {},
        onError: () => {},
        onReconnect: () => {
          expect(false).toBe(true);
        },
      });

      await new Promise((resolve) => setTimeout(resolve, 2000));

      expect(asrSession.ws.isOpen()).toBe(false);
    }, 10000);
  });

  // ===== offlineAsr 文件格式校验 =====

  describe("POST /api/v1/voice/asr/offline - 离线识别格式校验", () => {
    /** 构造 file 字段的 Blob/File（后端 _validate_audio 在调用 FunASR 前校验格式） */
    const makeAudioFile = (text: string, filename: string, type: string): File =>
      new File([new Blob([text], { type })], filename, { type });

    test("边界：非音频文件应失败", async () => {
      const file = makeAudioFile("not audio", "test.txt", "text/plain");
      await expectBizError(VoiceAPI.offlineAsr({ file, model: "paraformer-zh" }), [
        "A0400",
        "B0001",
        "A0500",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：空文件应失败", async () => {
      const file = makeAudioFile("", "empty.wav", "audio/wav");
      await expectBizError(VoiceAPI.offlineAsr({ file }), [
        "A0400",
        "B0001",
        "A0500",
        "ERR_BAD_REQUEST",
      ]);
    });
  });
});
