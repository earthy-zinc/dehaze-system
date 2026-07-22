import React, { useState, useCallback, useMemo } from "react";
import { View } from "@tarojs/components";
import Taro, { useRouter, useLoad } from "@tarojs/taro";
import {
  Navbar,
  Button,
  Checkbox,
  Loading,
  ConfigProvider,
} from "@taroify/core";
import { ArrowLeft } from "@taroify/icons";
import { useRoleManagement } from "@/hooks/useRoleManagement";
import "./permission.scss";

const RolePermissionPage: React.FC = () => {
  const router = useRouter();
  const { id } = router.params;

  const {
    permissions,
    fetchPermissions,
    fetchRolePermissions,
    assignPermissions,
  } = useRoleManagement();

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [checkedPermissions, setCheckedPermissions] = useState<string[]>([]);
  const [roleName, setRoleName] = useState("");

  // 缓存权限ID字符串数组，避免重复计算
  const allPermissionIds = useMemo(
    () => permissions.map((item) => String(item.value)),
    [permissions]
  );

  // 优化全选按钮逻辑
  const isAllSelected = useMemo(
    () =>
      checkedPermissions.length === allPermissionIds.length &&
      allPermissionIds.length > 0,
    [checkedPermissions.length, allPermissionIds.length]
  );

  // 加载数据
  const loadData = useCallback(async () => {
    if (!id) return;
    try {
      setLoading(true);

      // 并行加载权限列表和角色已有权限
      const [_, rolePermissionIds] = await Promise.all([
        fetchPermissions(),
        fetchRolePermissions(Number(id)),
      ]);

      // 获取角色名称
      const { RoleAPI } = await import("dehaze-sdk-js");
      const roleInfo = await RoleAPI.getFormData(Number(id));
      setRoleName(roleInfo.name || "");

      // 设置已选中的权限（转换为字符串数组）
      setCheckedPermissions(
        (rolePermissionIds || []).map((permId) => String(permId))
      );
    } catch (error) {
      console.error("加载数据失败:", error);
      Taro.showToast({
        title: "加载数据失败",
        icon: "none",
        duration: 2000,
      });
    } finally {
      setLoading(false);
    }
  }, [id, fetchPermissions, fetchRolePermissions]);

  // 页面加载时初始化数据
  useLoad(async () => {
    await loadData();
  });

  // 处理权限选择变化
  const handlePermissionChange = (permissionIds: string[]) => {
    setCheckedPermissions(permissionIds);
  };

  // 全选/取消全选
  const handleSelectAll = () => {
    if (isAllSelected) {
      // 取消全选
      setCheckedPermissions([]);
    } else {
      // 全选
      setCheckedPermissions(allPermissionIds);
    }
  };

  // 保存权限分配
  const handleSave = useCallback(async () => {
    if (!id) return;
    try {
      setSaving(true);
      // 转换为数字数组
      const permissionIds = checkedPermissions.map((permId) => Number(permId));
      await assignPermissions(Number(id), permissionIds);

      Taro.showToast({
        title: "权限分配成功",
        icon: "success",
        duration: 1500,
      });

      // 返回上一页
      setTimeout(() => {
        Taro.navigateBack();
      }, 1000);
    } catch (error) {
      // 错误已在 hook 中处理
      console.error("保存权限分配失败:", error);
    } finally {
      setSaving(false);
    }
  }, [id, checkedPermissions, assignPermissions]);

  if (!id) {
    Taro.showToast({
      title: "角色ID不能为空",
      icon: "none",
      duration: 2000,
    });
    Taro.navigateBack();
    return null;
  }

  if (loading) {
    return (
      <ConfigProvider>
        <View className="role-permission-page">
          <Navbar title="权限配置">
            <Navbar.NavLeft>
              <ArrowLeft onClick={() => Taro.navigateBack()} />
            </Navbar.NavLeft>
          </Navbar>
          <View className="loading-container">
            <Loading size="24px">加载中...</Loading>
          </View>
        </View>
      </ConfigProvider>
    );
  }

  return (
    <ConfigProvider>
      <View className="role-permission-page">
        <Navbar title={roleName ? `权限配置 - ${roleName}` : "权限配置"}>
          <Navbar.NavLeft>
            <ArrowLeft onClick={() => Taro.navigateBack()} />
          </Navbar.NavLeft>
        </Navbar>

        {/* 操作栏 */}
        <View className="permission-actions">
          <Button size="small" variant="outlined" onClick={handleSelectAll}>
            {isAllSelected ? "取消全选" : "全选"}
          </Button>
          <View className="selected-count">
            已选择 {checkedPermissions.length} / {allPermissionIds.length} 项
          </View>
        </View>

        {/* 权限树 */}
        <View className="permission-tree">
          {permissions.length === 0 ? (
            <View className="empty-permissions">暂无权限数据</View>
          ) : (
            <Checkbox.Group
              value={checkedPermissions}
              onChange={handlePermissionChange}
            ></Checkbox.Group>
          )}
        </View>

        {/* 底部操作按钮 */}
        <View className="permission-footer">
          <Button block color="primary" onClick={handleSave} loading={saving}>
            保存配置
          </Button>
        </View>
      </View>
    </ConfigProvider>
  );
};

export default RolePermissionPage;
