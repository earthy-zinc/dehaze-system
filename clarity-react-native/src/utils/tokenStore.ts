let sessionId: string | null = null;
let onSessionInvalid: (() => void) | null = null;
let isInvalidating = false;

export const sessionStore = {
  get(): string | null {
    return sessionId;
  },
  set(id: string) {
    sessionId = id;
    isInvalidating = false;
  },
  clear() {
    sessionId = null;
  },
};

export function setOnSessionInvalid(cb: (() => void) | null) {
  onSessionInvalid = cb;
}

export function triggerSessionInvalid() {
  if (isInvalidating) {
    return;
  }
  isInvalidating = true;
  onSessionInvalid?.();
}
