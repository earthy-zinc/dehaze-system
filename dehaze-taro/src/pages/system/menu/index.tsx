import React, { useState } from 'react';
import { View, Text, Input, ScrollView } from '@tarojs/components';
import Taro, { useLoad, usePullDownRefresh } from '@tarojs/taro';
import {
  Navbar,
  Search,
  Button,
  Loading,
  Empty,
  SwipeCell,
  Dialog,
  Tag,
  Cell,
  Popup,
  Switch,
} from '@taroify/core';
import { ArrowLeft, Add, Edit, Delete, Arrow } from '@taroify/icons';
import { useMenuManagement } from '@/hooks/useMenuManagement';
import { usePermission } from '@/hooks/usePermission';
import type { MenuVO, MenuForm } from 'dehaze-sdk-js';
import { MenuTypeEnum } from 'dehaze-sdk-js';
import './index.scss';

// 菜单类型配置
const MENU_TYPE_CONFIG: Record<string, { label: string; color: 'primary' | 'success' | 'warning' | 'info' }> = {
  [MenuTypeEnum.CATALOG]: { label: '目录', color: 'primary' },
  [MenuTypeEnum.MENU]: { label: '菜单', color: 'success' },
  [MenuTypeEnum.BUTTON]: { label: '按钮', color: 'warning' },
  [MenuTypeEnum.EXTLINK]: { label: '外链', color: 'info' },
};

// 菜单类型选项
const MENU_TYPE_OPTIONS = [
  { value: MenuTypeEnum.CATALOG, label: '目录' },
  { value: MenuTypeEnum.MENU, label: '菜单' },
  { value: MenuTypeEnum.BUTTON, label: '按钮' },
  { value: MenuTypeEnum.EXTLINK, label: '外链' },
];

// 默认表单
const DEFAULT_FORM: MenuForm = {
  type: MenuTypeEnum.CATALOG,
  parentId: 0,
  name: '',
  path: '',
  component: '',
  perm: '',
  icon: '',
  redirect: '',
  visible: 1,
  sort: 1,
};

// 树节点渲染组件
interface TreeNodeProps {
  node: MenuVO;
  depth: number;
  expandedKeys: number[];
  onToggle: (id: number) => void;
  onAddChild: (parentId: number) => void;
  onEdit: (node: MenuVO) => void;
  onDelete: (node: MenuVO) => void;
  hasPermission: (perm: string) => boolean;
}

const TreeNode: React.FC<TreeNodeProps> = ({
  node,
  depth,
  expandedKeys,
  onToggle,
  onAddChild,
  onEdit,
  onDelete,
  hasPermission,
}) => {
  const hasChildren = node.children && node.children.length > 0;
  const isExpanded = expandedKeys.includes(node.id!);
  const typeConfig = node.type ? MENU_TYPE_CONFIG[node.type] : null;

  return (
    <View className="menu-tree-node">
      <SwipeCell className="menu-swipe-cell">
        <SwipeCell.Actions side="right">
          {hasPermission('sys:menu:add') && (
            <Button
              className="action-btn add-btn"
              size="small"
              onClick={() => onAddChild(node.id!)}
            >
              <Add />
              子级
            </Button>
          )}
          {hasPermission('sys:menu:edit') && (
            <Button
              className="action-btn edit-btn"
              size="small"
              onClick={() => onEdit(node)}
            >
              <Edit />
              编辑
            </Button>
          )}
          {hasPermission('sys:menu:delete') && (
            <Button
              className="action-btn delete-btn"
              size="small"
              onClick={() => onDelete(node)}
            >
              <Delete />
              删除
            </Button>
          )}
        </SwipeCell.Actions>
        <Cell
          className="menu-cell"
          style={{ paddingLeft: `${16 + depth * 20}px` }}
          onClick={() => hasChildren && onToggle(node.id!)}
        >
          <View className="menu-row">
            {hasChildren ? (
              <View className="menu-toggle">
                <Arrow className={isExpanded ? 'arrow-expanded' : 'arrow-collapsed'} />
              </View>
            ) : (
              <View className="menu-toggle-placeholder" />
            )}
            <View className="menu-info">
              <View className="menu-name-row">
                <Text className="menu-name">{node.name}</Text>
                {typeConfig && (
                  <Tag color={typeConfig.color} size="small">
                    {typeConfig.label}
                  </Tag>
                )}
                {node.visible === 0 && (
                  <Tag color="default" size="small">
                    隐藏
                  </Tag>
                )}
              </View>
              <View className="menu-meta">
                {node.routePath && (
                  <Text className="meta-text">路由: {node.routePath}</Text>
                )}
                {node.perm && (
                  <Text className="meta-text">权限: {node.perm}</Text>
                )}
                <Text className="meta-text">排序: {node.sort ?? 0}</Text>
              </View>
            </View>
          </View>
        </Cell>
      </SwipeCell>
      {hasChildren && isExpanded && (
        <View className="menu-children">
          {node.children!.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              expandedKeys={expandedKeys}
              onToggle={onToggle}
              onAddChild={onAddChild}
              onEdit={onEdit}
              onDelete={onDelete}
              hasPermission={hasPermission}
            />
          ))}
        </View>
      )}
    </View>
  );
};

// 递归展开第一层菜单
const expandFirstLevel = (list: MenuVO[]): number[] => {
  return list
    .filter((item) => item.children && item.children.length > 0)
    .map((item) => item.id!)
    .filter((id): id is number => typeof id === 'number');
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

  const [searchKeyword, setSearchKeyword] = useState('');
  const [expandedKeys, setExpandedKeys] = useState<number[]>([]);

  // 表单弹窗状态
  const [showFormDialog, setShowFormDialog] = useState(false);
  const [editingId, setEditingId] = useState<string | undefined>();
  const [formData, setFormData] = useState<MenuForm>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);

  // 删除确认弹窗
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deletingMenu, setDeletingMenu] = useState<MenuVO | null>(null);

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
  const handleSearch = async (event: any) => {
    const value = event.detail?.value || '';
    setSearchKeyword(value);
    if (value.trim()) {
      await searchMenus(value.trim());
    } else {
      await resetQuery();
    }
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
  const handleDelete = (node: MenuVO) => {
    setDeletingMenu(node);
    setShowDeleteDialog(true);
  };

  const confirmDelete = async () => {
    if (!deletingMenu?.id) return;
    try {
      await deleteMenu(deletingMenu.id);
      setShowDeleteDialog(false);
      setDeletingMenu(null);
    } catch {
      // 错误已在 hook 中处理
    }
  };

  // 表单字段更新
  const handleFieldChange = (field: keyof MenuForm, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  // 菜单类型切换时重置条件字段
  const handleTypeChange = (type: MenuTypeEnum) => {
    setFormData((prev) => ({
      ...prev,
      type,
      path: type === MenuTypeEnum.BUTTON ? prev.path : '',
      component: type === MenuTypeEnum.MENU ? prev.component : '',
      perm: type === MenuTypeEnum.BUTTON ? prev.perm : '',
      redirect: type === MenuTypeEnum.CATALOG ? prev.redirect : '',
    }));
  };

  // 表单校验
  const validateForm = (): boolean => {
    if (!formData.name?.trim()) {
      Taro.showToast({ title: '菜单名称不能为空', icon: 'none' });
      return false;
    }
    if (formData.type === MenuTypeEnum.MENU) {
      if (!formData.path?.trim()) {
        Taro.showToast({ title: '路由地址不能为空', icon: 'none' });
        return false;
      }
      if (!formData.path.startsWith('/')) {
        Taro.showToast({ title: '路由地址必须以 / 开头', icon: 'none' });
        return false;
      }
      if (!formData.component?.trim()) {
        Taro.showToast({ title: '组件路径不能为空', icon: 'none' });
        return false;
      }
    } else if (formData.type === MenuTypeEnum.BUTTON) {
      if (!formData.perm?.trim()) {
        Taro.showToast({ title: '权限标识不能为空', icon: 'none' });
        return false;
      }
      if (!/^[a-z]+:[a-z]+:[a-z]+$/.test(formData.perm)) {
        Taro.showToast({ title: '权限标识格式应为 模块:功能:操作', icon: 'none' });
        return false;
      }
    } else if (formData.type === MenuTypeEnum.EXTLINK) {
      if (!formData.path?.trim()) {
        Taro.showToast({ title: '外链地址不能为空', icon: 'none' });
        return false;
      }
      if (!/^https?:\/\/.+/.test(formData.path)) {
        Taro.showToast({ title: '请输入正确的外链地址', icon: 'none' });
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

  // 根据菜单类型动态显示字段
  const showPath = formData.type === MenuTypeEnum.MENU || formData.type === MenuTypeEnum.EXTLINK;
  const showComponent = formData.type === MenuTypeEnum.MENU;
  const showPerm = formData.type === MenuTypeEnum.BUTTON;
  const showIcon = formData.type === MenuTypeEnum.CATALOG || formData.type === MenuTypeEnum.MENU;
  const showRedirect = formData.type === MenuTypeEnum.CATALOG;

  return (
    <View className="menu-page">
      <Navbar title="菜单管理">
        <Navbar.NavLeft>
          <ArrowLeft onClick={() => Taro.navigateBack()} />
        </Navbar.NavLeft>
        <Navbar.NavRight>
          {hasPermission('sys:menu:add') && <Add onClick={handleAdd} />}
        </Navbar.NavRight>
      </Navbar>

      {/* 搜索栏 */}
      <View className="search-bar">
        <Search
          placeholder="请输入菜单名称"
          value={searchKeyword}
          onChange={(e) => setSearchKeyword(e.detail.value)}
          onSearch={handleSearch}
          onClear={() => handleSearch('')}
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
            {hasPermission('sys:menu:add') && (
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

      {/* 删除确认弹窗 */}
      <Dialog
        open={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        title="确认删除"
      >
        <Dialog.Content>
          确定要删除菜单 "{deletingMenu?.name}" 吗？如有子菜单需先删除子菜单，此操作不可恢复。
        </Dialog.Content>
        <Dialog.Actions>
          <Button onClick={() => setShowDeleteDialog(false)}>取消</Button>
          <Button color="danger" onClick={confirmDelete}>删除</Button>
        </Dialog.Actions>
      </Dialog>

      {/* 菜单表单弹窗 */}
      <Popup
        open={showFormDialog}
        onClose={() => setShowFormDialog(false)}
        placement="bottom"
        style={{ height: '85%' }}
      >
        <View className="form-popup">
          <View className="form-header">
            <Text className="form-title">
              {editingId ? '编辑菜单' : '新增菜单'}
            </Text>
          </View>
          <ScrollView scrollY className="form-body">
            {/* 菜单类型 */}
            <View className="form-item">
              <Text className="form-label">菜单类型 *</Text>
              <View className="type-options">
                {MENU_TYPE_OPTIONS.map((option) => (
                  <View
                    key={option.value}
                    className={`type-option ${formData.type === option.value ? 'active' : ''}`}
                    onClick={() => handleTypeChange(option.value)}
                  >
                    <Text>{option.label}</Text>
                  </View>
                ))}
              </View>
            </View>

            {/* 上级菜单 */}
            <View className="form-item">
              <Text className="form-label">上级菜单</Text>
              <View className="parent-menu-display">
                {formData.parentId === 0
                  ? '顶级菜单'
                  : menuOptions.find((opt) => Number(opt.value) === formData.parentId)?.label || '未知菜单'}
              </View>
            </View>

            {/* 菜单名称 */}
            <View className="form-item">
              <Text className="form-label">菜单名称 *</Text>
              <Input
                className="form-input"
                placeholder="请输入菜单名称"
                value={formData.name || ''}
                onInput={(e) => handleFieldChange('name', e.detail.value)}
              />
            </View>

            {/* 路由地址（菜单/外链） */}
            {showPath && (
              <View className="form-item">
                <Text className="form-label">
                  {formData.type === MenuTypeEnum.EXTLINK ? '外链地址 *' : '路由地址 *'}
                </Text>
                <Input
                  className="form-input"
                  placeholder={
                    formData.type === MenuTypeEnum.EXTLINK
                      ? '请输入外链地址（https://）'
                      : '请输入路由地址（/开头）'
                  }
                  value={formData.path || ''}
                  onInput={(e) => handleFieldChange('path', e.detail.value)}
                />
              </View>
            )}

            {/* 组件路径（菜单） */}
            {showComponent && (
              <View className="form-item">
                <Text className="form-label">组件路径 *</Text>
                <Input
                  className="form-input"
                  placeholder="请输入组件路径"
                  value={formData.component || ''}
                  onInput={(e) => handleFieldChange('component', e.detail.value)}
                />
              </View>
            )}

            {/* 权限标识（按钮） */}
            {showPerm && (
              <View className="form-item">
                <Text className="form-label">权限标识 *</Text>
                <Input
                  className="form-input"
                  placeholder="格式：模块:功能:操作（如 sys:menu:add）"
                  value={formData.perm || ''}
                  onInput={(e) => handleFieldChange('perm', e.detail.value)}
                />
              </View>
            )}

            {/* 图标（目录/菜单） */}
            {showIcon && (
              <View className="form-item">
                <Text className="form-label">图标</Text>
                <Input
                  className="form-input"
                  placeholder="请输入图标名称"
                  value={formData.icon || ''}
                  onInput={(e) => handleFieldChange('icon', e.detail.value)}
                />
              </View>
            )}

            {/* 路由重定向（目录） */}
            {showRedirect && (
              <View className="form-item">
                <Text className="form-label">路由重定向</Text>
                <Input
                  className="form-input"
                  placeholder="请输入路由重定向地址"
                  value={formData.redirect || ''}
                  onInput={(e) => handleFieldChange('redirect', e.detail.value)}
                />
              </View>
            )}

            {/* 排序 */}
            <View className="form-item">
              <Text className="form-label">排序</Text>
              <Input
                className="form-input"
                type="number"
                placeholder="请输入排序值"
                value={String(formData.sort ?? 1)}
                onInput={(e) => handleFieldChange('sort', Number(e.detail.value) || 1)}
              />
            </View>

            {/* 显示状态 */}
            <View className="form-item">
              <Text className="form-label">显示状态</Text>
              <View className="form-switch">
                <Switch
                  checked={formData.visible === 1}
                  onChange={(checked) => handleFieldChange('visible', checked ? 1 : 0)}
                />
                <Text>{formData.visible === 1 ? '显示' : '隐藏'}</Text>
              </View>
            </View>
          </ScrollView>
          <View className="form-footer">
            <Button onClick={() => setShowFormDialog(false)}>取消</Button>
            <Button color="primary" loading={submitting} onClick={submitForm}>
              确定
            </Button>
          </View>
        </View>
      </Popup>
    </View>
  );
};

export default MenuPage;
