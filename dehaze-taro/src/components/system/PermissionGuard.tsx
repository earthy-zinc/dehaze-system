import React from "react";
import { View } from "@tarojs/components";
import { useAuth } from "@/hooks/useAuth";
import { usePermission } from "@/hooks/usePermission";
import Loading from "@/components/common/Loading";

const NoAuth = (
  <View className="permission-fallback">
    <View className="no-permission">抱歉，您没有权限访问此页面</View>
  </View>
);
interface PermissionGuardProps {
  children: React.ReactNode;
  // 权限要求
  permissions?: string | string[];
  roles?: string | string[];
  // 是否需要认证
  requireAuth?: boolean;
  // 无权限时显示的组件
  fallback?: React.ReactNode;
  // 加载时显示的组件
  loadingComponent?: React.ReactNode;
}

const PermissionGuard: React.FC<PermissionGuardProps> = (props) => {
  const {
    children,
    permissions,
    roles,
    requireAuth = true,
    fallback = NoAuth,
    loadingComponent = <Loading>加载中...</Loading>,
  } = props;
  const { isAuthenticated, loading: authLoading } = useAuth();
  const {
    hasPermission: hasPermissionFn,
    hasRole: hasRoleFn,
    isSuperAdmin,
  } = usePermission();

  // 如果正在加载认证状态，显示加载组件
  if (authLoading) {
    return <>{loadingComponent}</>;
  }

  // 检查是否需要认证
  if (requireAuth && !isAuthenticated) {
    // 跳转到登录页面的逻辑需要在路由层面处理
    // 这里只返回空组件，由路由守卫处理跳转
    return <></>;
  }

  // 超级管理员拥有所有权限
  if (isSuperAdmin()) {
    return <>{children}</>;
  }

  // 检查权限
  if (permissions && !hasPermissionFn(permissions)) {
    return <>{fallback}</>;
  }

  // 检查角色
  if (roles && !hasRoleFn(roles)) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
};

export default PermissionGuard;
