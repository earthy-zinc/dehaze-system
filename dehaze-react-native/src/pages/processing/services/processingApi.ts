/**
 * 去雾处理服务
 *
 * 封装 dehaze-sdk-js 的 ModelAPI.predictAndWait：
 * - POST /prediction 立即返回 logId + status=processing
 * - 内部自动轮询 GET /prediction/{taskId} 直到 completed/failed
 *
 * 不模拟进度百分比，仅推进真实状态与已用时间。
 */
import { ModelAPI } from 'dehaze-sdk-js';
import type { PredictionResultVO } from 'dehaze-sdk-js';
import type {
  CommonAlgorithmParams,
  ProcessingResult,
  TaskProgress,
} from '@/types/processing';

/** 默认参数（推荐配置） */
export const DEFAULT_PARAMS: CommonAlgorithmParams = {
  strength: 50,
  saturation: 100,
  contrast: 100,
  sharpen: 30,
};

/** 参数 schema（用于动态渲染调节面板） */
export const PARAM_SCHEMAS = [
  {
    key: 'strength' as const,
    label: '去雾强度',
    type: 'slider' as const,
    min: 0,
    max: 100,
    step: 1,
    default: 50,
    description: '控制去雾程度，值越大去雾效果越强',
  },
  {
    key: 'saturation' as const,
    label: '色彩饱和度',
    type: 'slider' as const,
    min: 0,
    max: 200,
    step: 1,
    default: 100,
    description: '调整色彩鲜艳度，100 为原始值',
  },
  {
    key: 'contrast' as const,
    label: '对比度',
    type: 'slider' as const,
    min: 0,
    max: 200,
    step: 1,
    default: 100,
    description: '调整明暗对比，100 为原始值',
  },
  {
    key: 'sharpen' as const,
    label: '锐化程度',
    type: 'slider' as const,
    min: 0,
    max: 100,
    step: 1,
    default: 30,
    description: '增强细节清晰度',
  },
];

/** 预设方案 */
export const PARAM_PRESETS = [
  {
    key: 'recommended',
    name: '推荐配置',
    description: '适用于大多数场景的均衡配置',
    params: { strength: 50, saturation: 100, contrast: 100, sharpen: 30 },
  },
  {
    key: 'landscape',
    name: '风景优化',
    description: '增强色彩与对比，适合户外风景',
    params: { strength: 70, saturation: 130, contrast: 120, sharpen: 40 },
  },
  {
    key: 'night',
    name: '夜景增强',
    description: '降低强度，保留夜景氛围',
    params: { strength: 35, saturation: 90, contrast: 110, sharpen: 50 },
  },
];

/** 轮询间隔（ms） */
const POLL_INTERVAL = 1500;
/** 最大轮询时长（ms） */
const MAX_POLL_DURATION = 5 * 60 * 1000;

/** 将参数对象序列化为后端可识别的 JSON 字符串 */
function serializeParams(params?: CommonAlgorithmParams): string | undefined {
  if (!params) return undefined;
  const filtered: Record<string, number> = {};
  (Object.keys(params) as (keyof CommonAlgorithmParams)[]).forEach(k => {
    const v = params[k];
    if (v !== undefined && v !== null) {
      filtered[k] = v;
    }
  });
  return Object.keys(filtered).length ? JSON.stringify(filtered) : undefined;
}

function toProcessingResult(vo: PredictionResultVO): ProcessingResult {
  return {
    logId: vo.logId,
    resultUrl: vo.resultUrl,
    resultThumbnailUrl: vo.resultThumbnailUrl,
    time: vo.time,
    fromCache: vo.fromCache,
  };
}

export interface PredictOptions {
  algorithmId: number;
  imageUrl: string;
  params?: CommonAlgorithmParams;
  /** 进度回调（用于真实状态更新 + 轮询状态） */
  onProgress?: (progress: TaskProgress) => void;
  /** 取消信号 */
  cancelSignal?: { canceled: boolean };
}

/**
 * 单张去雾处理
 *
 * 调用 SDK predictAndWait（POST + 自动轮询），失败时通过 status=failed 抛出错误。
 * 取消信号触发时立即抛出错误。
 */
export async function predictSingle(opts: PredictOptions): Promise<ProcessingResult> {
  const { algorithmId, imageUrl, params, onProgress, cancelSignal } = opts;
  const startTime = Date.now();

  const emit = (status: TaskProgress['status'], error?: string) => {
    onProgress?.({
      status,
      elapsed: Date.now() - startTime,
      error,
    });
  };

  emit(1);

  try {
    const result = await ModelAPI.predictAndWait(
      {
        algorithmId,
        imageUrl,
        params: serializeParams(params),
      },
      {
        intervalMs: POLL_INTERVAL,
        timeoutMs: MAX_POLL_DURATION,
        onPoll: () => {
          if (cancelSignal?.canceled) {
            throw new Error('用户已取消处理');
          }
          emit(1);
        },
      },
    );

    if (cancelSignal?.canceled) throw new Error('用户已取消处理');

    if (result.status === 3) {
      throw new Error(result.errorMessage || '处理失败');
    }

    emit(2);
    return toProcessingResult(result);
  } catch (err) {
    emit(3, err instanceof Error ? err.message : '预测请求失败');
    throw err;
  }
}

