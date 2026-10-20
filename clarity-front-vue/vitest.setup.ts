import "vitest-canvas-mock";

import { createPinia, setActivePinia } from "pinia";
import { beforeEach, vi } from "vitest";

// Mock browser APIs that are not available in jsdom
Object.defineProperty(global, "requestAnimationFrame", {
  writable: true,
  value: vi.fn((cb) => setTimeout(cb, 0)),
});

Object.defineProperty(global, "cancelAnimationFrame", {
  writable: true,
  value: vi.fn((id) => clearTimeout(id)),
});

// 在每个测试之前创建新的 Pinia 实例
beforeEach(() => {
  const pinia = createPinia();
  setActivePinia(pinia);

  // Clear all mocks before each test
  vi.clearAllMocks();
});
