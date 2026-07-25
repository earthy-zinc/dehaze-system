/**
 * 测试数据清理注册表
 *
 * 统一管理测试数据的创建与清理，替代各测试文件中重复的 afterAll 清理逻辑。
 * - 支持按 LIFO 顺序执行清理（后创建的先清理，保证子资源先于父资源删除）
 * - 清理失败时静默忽略，确保所有注册的清理操作都被尝试执行
 * - 提供便捷的 ID 注册方法，配合各模块 API 批量清理
 */
export class TestCleanupRegistry {
  private tasks: Array<() => Promise<void>> = [];

  /**
   * 注册一个清理回调（在 afterAll 中按 LIFO 顺序执行）
   * @param fn 清理函数
   */
  register(fn: () => Promise<void>): void {
    this.tasks.push(fn);
  }

  /**
   * 批量注册 ID 的清理（从后往前执行，保证子资源先清理）
   * @param ids ID 数组引用（会在执行时读取最新值）
   * @param deleteFn 删除函数，接收单个 ID 字符串
   */
  registerIds(ids: () => number[], deleteFn: (id: string) => Promise<unknown>): void {
    this.tasks.push(async () => {
      const currentIds = ids();
      // 从后往前删除，确保子资源先于父资源被清理
      for (const id of [...currentIds].reverse()) {
        try {
          await deleteFn(id.toString());
        } catch {
          // 静默忽略清理失败（资源可能已被测试本身删除）
        }
      }
    });
  }

  /**
   * 执行所有注册的清理任务（按 LIFO 顺序）
   * 任何单个任务失败不会阻止后续任务执行
   */
  async executeAll(): Promise<void> {
    const errors: Array<{ index: number; error: unknown }> = [];
    for (let i = this.tasks.length - 1; i >= 0; i--) {
      try {
        await this.tasks[i]!();
      } catch (e) {
        errors.push({ index: i, error: e });
      }
    }
    this.tasks = [];
    if (errors.length > 0) {
      console.warn(`[TestCleanupRegistry] ${errors.length} 个清理任务失败（已忽略）`);
    }
  }
}
