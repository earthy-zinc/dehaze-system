import type { HotwordForm, TtsForm, StreamAsrSessionForm } from "../../src/api/voice/model";
import { uniqueName } from "./common";

/** 流式 ASR 会话创建表单工厂 */
export const createStreamAsrSessionForm = (
  overrides?: Partial<StreamAsrSessionForm>
): StreamAsrSessionForm => ({
  ...overrides,
});

/** TTS 合成请求工厂 */
export const createTtsForm = (overrides?: Partial<TtsForm>): TtsForm => ({
  text: "处理完成",
  voice: "huayan",
  speed: 1.0,
  ...overrides,
});

/** 热词新增表单工厂 */
export const createHotwordForm = (overrides?: Partial<HotwordForm>): HotwordForm => ({
  word: uniqueName("RIDCP"),
  ...overrides,
});
