/**
 * 穷尽检查：在 switch / 处理表的兜底分支调用。
 *
 * 参数类型为 `never`——判别联合的所有成员都被处理时，残留值才是 never；
 * 联合新增成员而未接处理时实参不再收敛为 never，编译失败。
 * 运行期抛错只是兜底（类型被 any / 断言绕过时才会走到）。
 */
export function assertNever(value: never): never {
  throw new Error(`未处理的分支: ${String(value)}`);
}
