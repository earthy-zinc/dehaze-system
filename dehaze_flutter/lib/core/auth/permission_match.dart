/// 模块级权限判定
///
/// 后端权限来自 `sys_menu.perm`，是按钮级细粒度串（如 `sys:user:add`、`order:list`），
/// 且多数模块没有"列表/查看"权限、对应列表接口也不做权限校验（仅要求登录）。
/// 因此前端"管理入口显隐 / 路由守卫"按**模块前缀**判定：命中该模块任一具体权限即视为可见。
library;

/// 命中 [modulePrefix] 前缀（如 `sys:user`、`order`）的任一权限
///
/// 传入时带不带结尾冒号均可。
bool matchModulePerm(Iterable<String> perms, String modulePrefix) {
  final prefix = modulePrefix.endsWith(':') ? modulePrefix : '$modulePrefix:';
  for (final perm in perms) {
    if (perm == modulePrefix || perm.startsWith(prefix)) return true;
  }
  return false;
}
