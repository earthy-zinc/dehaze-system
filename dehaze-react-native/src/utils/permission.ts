/**
 * 权限判定工具
 *
 * 后端下发的权限码全部是具体串（如 `sys:user:add`），不存在通配符。
 * 入口显隐需要按「模块前缀」判定（命中该模块任一具体权限即可见），
 * 因此这里支持以 `:*` 结尾的模块前缀模式，与 Flutter `matchModulePerm` /
 * Android `hasModulePerm` 口径一致；具体码仍按精确匹配。
 */
export function matchPerm(
  userPerms: string[] | null | undefined,
  required: string
): boolean {
  if (!userPerms || userPerms.length === 0 || !required) return false;
  if (required.endsWith(':*')) {
    const prefix = required.slice(0, -1); // "sys:user:"
    return userPerms.some(
      p => p === prefix.slice(0, -1) || p.startsWith(prefix)
    );
  }
  return userPerms.includes(required);
}
