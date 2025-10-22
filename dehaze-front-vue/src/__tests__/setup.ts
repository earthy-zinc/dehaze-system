import { afterEach, vi } from "vitest";
import { config } from "@vue/test-utils";

// Mock Element Plus 组件
config.global.stubs = {
  ElButton: true,
  ElForm: true,
  ElFormItem: true,
  ElInput: true,
  ElMessage: true,
  ElMessageBox: true,
  ElLoading: true,
  ElDialog: true,
  ElTable: true,
  ElTableColumn: true,
  ElPagination: true,
  ElUpload: true,
  ElSelect: true,
  ElOption: true,
  ElIcon: true,
};

// Mock window.matchMedia
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn((key) => {
    return localStorageMock.storage[key] || null;
  }),
  setItem: vi.fn((key, value) => {
    localStorageMock.storage[key] = value;
  }),
  removeItem: vi.fn((key) => {
    delete localStorageMock.storage[key];
  }),
  clear: vi.fn(() => {
    localStorageMock.storage = {};
  }),
  storage: {} as Record<string, string>,
};
global.localStorage = localStorageMock as any;

// Mock sessionStorage
const sessionStorageMock = {
  getItem: vi.fn((key) => {
    return sessionStorageMock.storage[key] || null;
  }),
  setItem: vi.fn((key, value) => {
    sessionStorageMock.storage[key] = value;
  }),
  removeItem: vi.fn((key) => {
    delete sessionStorageMock.storage[key];
  }),
  clear: vi.fn(() => {
    sessionStorageMock.storage = {};
  }),
  storage: {} as Record<string, string>,
};
global.sessionStorage = sessionStorageMock as any;

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  takeRecords() {
    return [];
  }
  unobserve() {}
} as any;

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
} as any;

// Mock HTMLCanvasElement.getContext
HTMLCanvasElement.prototype.getContext = vi.fn(() => {
  return {
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineCap: "butt",
    lineJoin: "miter",
    miterLimit: 10,
    lineDashOffset: 0,
    shadowOffsetX: 0,
    shadowOffsetY: 0,
    shadowBlur: 0,
    shadowColor: "transparent",
    globalAlpha: 1,
    globalCompositeOperation: "source-over",
    font: "10px sans-serif",
    textAlign: "start",
    textBaseline: "alphabetic",
    direction: "ltr",
    imageSmoothingEnabled: true,
    filter: "none",
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    clearRect: vi.fn(),
    fillText: vi.fn(),
    strokeText: vi.fn(),
    measureText: vi.fn(() => ({ width: 0 })),
    drawImage: vi.fn(),
    createImageData: vi.fn(),
    getImageData: vi.fn(),
    putImageData: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    scale: vi.fn(),
    rotate: vi.fn(),
    translate: vi.fn(),
    transform: vi.fn(),
    setTransform: vi.fn(),
    resetTransform: vi.fn(),
    beginPath: vi.fn(),
    closePath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    bezierCurveTo: vi.fn(),
    quadraticCurveTo: vi.fn(),
    arc: vi.fn(),
    arcTo: vi.fn(),
    ellipse: vi.fn(),
    rect: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    clip: vi.fn(),
    isPointInPath: vi.fn(),
    isPointInStroke: vi.fn(),
    createLinearGradient: vi.fn(),
    createRadialGradient: vi.fn(),
    createPattern: vi.fn(),
    canvas: {} as HTMLCanvasElement,
    getLineDash: vi.fn(() => []),
    setLineDash: vi.fn(),
  } as any;
}) as any;

// 清理函数，在每个测试后执行
afterEach(() => {
  vi.clearAllMocks();
  localStorageMock.getItem.mockClear();
  localStorageMock.setItem.mockClear();
  localStorageMock.removeItem.mockClear();
  localStorageMock.clear.mockClear();
  localStorageMock.storage = {};
});
