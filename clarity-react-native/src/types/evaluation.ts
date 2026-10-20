/**
 * 效果评估相关类型（RN 业务层补充）
 *
 * 基础评估类型复用 SDK 导出，
 * 此处仅定义路由参数所需的简化结构。
 */
export interface EvaluationMetrics {
  psnr?: number;
  ssim?: number;
  mse?: number;
  entropy?: number;
  lpips?: number;
  niqe?: number;
  contrastGain?: number;
  saturationGain?: number;
  sharpnessGain?: number;
  duration?: number;
}
