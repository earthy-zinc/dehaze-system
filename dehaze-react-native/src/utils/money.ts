/**
 * 金额换算：接口金额统一以分为单位，页面展示与录入以元为单位
 */

export function fenToYuan(fen?: number): string {
  return ((fen ?? 0) / 100).toFixed(2);
}

export function yuanToFen(yuan: number | string): number {
  return Math.round(Number(yuan) * 100);
}
