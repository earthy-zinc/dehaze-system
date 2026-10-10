import type { HotwordForm, TtsForm, StreamAsrSessionForm } from "../../src/api/voice/model";
import { uniqueName } from "./common";

/** 流式 ASR 会话创建表单工厂 */
export const createStreamAsrSessionForm = (
  overrides?: Partial<StreamAsrSessionForm>
): StreamAsrSessionForm => ({
  ...overrides,
});

/** TTS 合成请求工厂（不指定音色 → 由后端取当前默认 tts 模型的音色，避免与具体默认模型耦合） */
export const createTtsForm = (overrides?: Partial<TtsForm>): TtsForm => ({
  text: "处理完成",
  speed: 1.0,
  ...overrides,
});

/** 热词新增表单工厂 */
export const createHotwordForm = (overrides?: Partial<HotwordForm>): HotwordForm => ({
  word: uniqueName("RIDCP"),
  ...overrides,
});
