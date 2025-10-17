/**
 * 通用响应结果包装类
 */
export class Result<T> {
  code: number = 0;
  data: T = undefined;
  message: string = '';

  isSuccess(): boolean {
    return this.code === 0;
  }
}