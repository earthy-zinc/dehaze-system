/** 轮询默认间隔：2s */
export const DEFAULT_POLL_INTERVAL_MS = 2000;

/** 轮询默认超时：120s */
export const DEFAULT_POLL_TIMEOUT_MS = 120000;

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
