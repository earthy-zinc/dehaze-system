import "vitest-canvas-mock";

import { createPinia, setActivePinia } from "pinia";
import { beforeEach } from "vitest";

// 在每个测试之前创建新的 Pinia 实例
beforeEach(() => {
  const pinia = createPinia();
  setActivePinia(pinia);
});
