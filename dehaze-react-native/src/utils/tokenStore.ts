let sessionId: string | null = null;
let onSessionInvalid: (() => void) | null = null;

export const sessionStore = {
  get(): string | null {
    return sessionId;
  },
  set(id: string) {
    sessionId = id;
  },
  clear() {
    sessionId = null;
  },
};

export function setOnSessionInvalid(cb: (() => void) | null) {
  onSessionInvalid = cb;
}

export function triggerSessionInvalid() {
  onSessionInvalid?.();
}
