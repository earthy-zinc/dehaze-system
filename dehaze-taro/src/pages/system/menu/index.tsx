import React, { useState } from "react";
import { View, ScrollView } from "@tarojs/components";
import type { BaseEventOrig } from "@tarojs/components";
import Taro, { useLoad, usePullDownRefresh } from "@tarojs/taro";
import { confirmDialog } from "@/utils/dialog";
import { Navbar, Search, Button, Loading, Empty } from "@taroify/core";
import { ArrowLeft, Add } from "@taroify/icons";
import { useMenuManagement } from "@/hooks/useMenuManagement";
import { usePermission } from "@/hooks/usePermission";
import type { MenuVO, MenuForm } from "dehaze-sdk-js";
import { MenuTypeEnum } from "dehaze-sdk-js";
import { DEFAULT_FORM } from "./constants";
import TreeNode from "./components/TreeNode";
import MenuFormDialog from "./components/MenuFormDialog";
import "./index.less";

// 递归展开第一层菜单
const expandFirstLevel = (list: MenuVO[]): number[] => {
  return list
    .filter((item) => item.children && item.children.length > 0)
    .map((item) => item.id!)
    .filter((id): id is number => typeof id === "number");
};

const MenuPage: React.FC = () => {
  const {
    menuList,
    loading,
    menuOptions,
    fetchMenus,
    fetchMenuOptions,
    fetchMenuForm,
    createMenu,
    updateMenu,
    deleteMenu,
    searchMenus,
    resetQuery,
  } = useMenuManagement();

  const { hasPermission } = usePermission();

  const [searchKeyword, setSearchKeyword] = useState("");
  const [expandedKeys, setExpandedKeys] = useState<number[]>([]);

  // 表单弹窗状态
  const [showFormDialog, setShowFormDialog] = useState(false);
  const [editingId, setEditingId] = useState<string | undefined>();
  const [formData, setFormData] = useState<MenuForm>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);

  useLoad(async () => {
    const list = await fetchMenus();
    if (list) {
      setExpandedKeys(expandFirstLevel(list));
    }
    await fetchMenuOptions();
  });

  usePullDownRefresh(async () => {
    try {
      const list = await fetchMenus();
      if (list) {
        setExpandedKeys(expandFirstLevel(list));
      }
      Taro.stopPullDownRefresh();
    } catch {
      Taro.stopPullDownRefresh();
    }
  });

  // 搜索处理
  const performSearch = async (value: string) => {
    setSearchKeyword(value);
    if (value.trim()) {
      await searchMenus(value.trim());
    } else {
      await resetQuery();
    }
  };

  const handleSearch = async (event: BaseEventOrig<{ value: string }>) => {
    await performSearch(event.detail?.value || "");
  };

  // 展开/收起节点
  const handleToggle = (id: number) => {
    setExpandedKeys((prev) =>
      prev.includes(id) ? prev.filter((key) => key !== id) : [...prev, id]
    );
  };

  // 新增菜单（顶级）
  const handleAdd = () => {
    setEditingId(undefined);
    setFormData({ ...DEFAULT_FORM, parentId: 0 });
    setShowFormDialog(true);
  };

  // 新增子菜单
  const handleAddChild = (parentId: number) => {
    setEditingId(undefined);
    setFormData({ ...DEFAULT_FORM, parentId });
    setShowFormDialog(true);
  };

  // 编辑菜单
  const handleEdit = async (node: MenuVO) => {
    if (!node.id) return;
    try {
      const form = await fetchMenuForm(node.id);
      setEditingId(String(node.id));
      setFormData(form);
      setShowFormDialog(true);
    } catch {
      // 错误已在 hook 中处理
    }
  };

  // 删除菜单
  const handleDelete = async (node: MenuVO) => {
    const confirmed = await confirmDialog({
      title: "确认删除",
      content: `确定要删除菜单 "${node.name}" 吗？如有子菜单需先删除子菜单，此操作不可恢复。`,
      confirmText: "删除",
      cancelText: "取消",
    });
    if (!confirmed) return;
    await confirmDelete(node);
  };

  const confirmDelete = async (node: MenuVO) => {
    if (!node?.id) return;
    try {
      await deleteMenu(node.id);
    } catch {
      // 错误已在 hook 中处理
    }
  };

  // 表单字段更新
  const handleFieldChange = (field: keyof MenuForm, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  // 菜单类型切换时重置条件字段
  const handleTypeChange = (type: MenuTypeEnum) => {
    setFormData((prev) => ({
      ...prev,
      type,
      path: type === MenuTypeEnum.BUTTON ? prev.path : "",
      component: type === MenuTypeEnum.MENU ? prev.component : "",
      perm: type === MenuTypeEnum.BUTTON ? prev.perm : "",
      redirect: type === MenuTypeEnum.CATALOG ? prev.redirect : "",
    }));
  };

  // 表单校验
  const validateForm = (): boolean => {
    if (!formData.name?.trim()) {
      Taro.showToast({ title: "菜单名称不能为空", icon: "none" });
      return false;
    }
    if (formData.type === MenuTypeEnum.MENU) {
      if (!formData.path?.trim()) {
        Taro.showToast({ title: "路由地址不能为空", icon: "none" });
        return false;
      }
      if (!formData.path.startsWith("/")) {
        Taro.showToast({ title: "路由地址必须以 / 开头", icon: "none" });
        return false;
      }
      if (!formData.component?.trim()) {
        Taro.showToast({ title: "组件路径不能为空", icon: "none" });
        return false;
      }
    } else if (formData.type === MenuTypeEnum.BUTTON) {
      if (!formData.perm?.trim()) {
        Taro.showToast({ title: "权限标识不能为空", icon: "none" });
        return false;
      }
      if (!/^[a-z]+:[a-z]+:[a-z]+$/.test(formData.perm)) {
        Taro.showToast({
          title: "权限标识格式应为 模块:功能:操作",
          icon: "none",
        });
        return false;
      }
    } else if (formData.type === MenuTypeEnum.EXTLINK) {
      if (!formData.path?.trim()) {
        Taro.showToast({ title: "外链地址不能为空", icon: "none" });
        return false;
      }
      if (!/^https?:\/\/.+/.test(formData.path)) {
        Taro.showToast({ title: "请输入正确的外链地址", icon: "none" });
        return false;
      }
    }
    return true;
  };

  // 提交表单
  const submitForm = async () => {
    if (!validateForm()) return;

    setSubmitting(true);
    try {
      if (editingId) {
        await updateMenu(editingId, formData);
      } else {
        await createMenu(formData);
      }
      setShowFormDialog(false);
    } catch {
      // 错误已在 hook 中处理
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <View className="menu-page">
      <Navbar title="菜单管理">
        <Navbar.NavLeft>
          <ArrowLeft onClick={() => Taro.navigateBack()} />
        </Navbar.NavLeft>
        <Navbar.NavRight>
          {hasPermission("sys:menu:add") && <Add onClick={handleAdd} />}
        </Navbar.NavRight>
      </Navbar>

      {/* 搜索栏 */}
      <View className="search-bar">
        <Search
          placeholder="请输入菜单名称"
          value={searchKeyword}
          onChange={(e) => setSearchKeyword(e.detail.value)}
          onSearch={handleSearch}
          onClear={() => performSearch("")}
        />
      </View>

      {/* 菜单树形列表 */}
      <ScrollView scrollY className="menu-list">
        {loading && menuList.length === 0 ? (
          <Loading>加载中...</Loading>
        ) : menuList.length === 0 ? (
          <Empty>
            <Empty.Image />
            <Empty.Description>暂无菜单数据</Empty.Description>
            {hasPermission("sys:menu:add") && (
              <Button color="primary" size="small" onClick={handleAdd}>
                新增菜单
              </Button>
            )}
          </Empty>
        ) : (
          menuList.map((menu) => (
            <TreeNode
              key={menu.id}
              node={menu}
              depth={0}
              expandedKeys={expandedKeys}
              onToggle={handleToggle}
              onAddChild={handleAddChild}
              onEdit={handleEdit}
              onDelete={handleDelete}
              hasPermission={hasPermission}
            />
          ))
        )}
      </ScrollView>

      {/* 菜单表单弹窗 */}
      <MenuFormDialog
        open={showFormDialog}
        editingId={editingId}
        form={formData}
        submitting={submitting}
        menuOptions={menuOptions}
        onClose={() => setShowFormDialog(false)}
        onFieldChange={handleFieldChange}
        onTypeChange={handleTypeChange}
        onSubmit={submitForm}
      />
    </View>
  );
};

export default MenuPage;
