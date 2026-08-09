import React, { useState } from "react";
import { View, Text, Input, ScrollView } from "@tarojs/components";
import type { BaseEventOrig } from "@tarojs/components";
import Taro, { useLoad, usePullDownRefresh } from "@tarojs/taro";
import { confirmDialog } from "@/utils/dialog";
import {
  Navbar,
  Search,
  Button,
  Loading,
  Empty,
  SwipeCell,
  Tag,
  Cell,
  Popup,
  Switch,
} from "@taroify/core";
import { ArrowLeft, Add, Edit, Delete, Arrow } from "@taroify/icons";
import { useDeptManagement } from "@/hooks/useDeptManagement";
import { usePermission } from "@/hooks/usePermission";
import type { DeptVO, DeptForm } from "dehaze-sdk-js";
import "./index.less";

// 默认表单
const DEFAULT_FORM: DeptForm = {
  parentId: 0,
  name: "",
  sort: 1,
  status: 1,
};

// 树节点渲染组件
interface TreeNodeProps {
  node: DeptVO;
  depth: number;
  expandedKeys: number[];
  onToggle: (id: number) => void;
  onAddChild: (parentId: number) => void;
  onEdit: (node: DeptVO) => void;
  onDelete: (node: DeptVO) => void;
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

  return (
    <View className="dept-tree-node">
      <SwipeCell className="dept-swipe-cell">
        <SwipeCell.Actions side="right">
          {hasPermission("sys:dept:add") && (
            <Button
              className="action-btn add-btn"
              size="small"
              onClick={() => onAddChild(node.id!)}
            >
              <Add />
              子级
            </Button>
          )}
          {hasPermission("sys:dept:edit") && (
            <Button
              className="action-btn edit-btn"
              size="small"
              onClick={() => onEdit(node)}
            >
              <Edit />
              编辑
            </Button>
          )}
          {hasPermission("sys:dept:delete") && (
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
          className="dept-cell"
          style={{ paddingLeft: `${16 + depth * 20}px` }}
          onClick={() => hasChildren && onToggle(node.id!)}
        >
          <View className="dept-row">
            {hasChildren ? (
              <View className="dept-toggle">
                <Arrow
                  className={isExpanded ? "arrow-expanded" : "arrow-collapsed"}
                />
              </View>
            ) : (
              <View className="dept-toggle-placeholder" />
            )}
            <View className="dept-info">
              <View className="dept-name-row">
                <Text className="dept-name">{node.name}</Text>
                {node.status === 1 ? (
                  <Tag color="success" size="small">
                    启用
                  </Tag>
                ) : (
                  <Tag color="default" size="small">
                    禁用
                  </Tag>
                )}
              </View>
              <View className="dept-meta">
                <Text className="meta-text">排序: {node.sort ?? 0}</Text>
              </View>
            </View>
          </View>
        </Cell>
      </SwipeCell>
      {hasChildren && isExpanded && (
        <View className="dept-children">
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

// 递归展开第一层部门
const expandFirstLevel = (list: DeptVO[]): number[] => {
  return list
    .filter((item) => item.children && item.children.length > 0)
    .map((item) => item.id!)
    .filter((id): id is number => typeof id === "number");
};

const DeptPage: React.FC = () => {
  const {
    deptList,
    loading,
    deptOptions,
    fetchDeptList,
    fetchDeptOptions,
    fetchDeptForm,
    createDept,
    updateDept,
    deleteDept,
    searchDepts,
    resetQuery,
  } = useDeptManagement();

  const { hasPermission } = usePermission();

  const [searchKeyword, setSearchKeyword] = useState("");
  const [expandedKeys, setExpandedKeys] = useState<number[]>([]);

  // 表单弹窗状态
  const [showFormDialog, setShowFormDialog] = useState(false);
  const [editingId, setEditingId] = useState<number | undefined>();
  const [formData, setFormData] = useState<DeptForm>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);

  useLoad(async () => {
    const list = await fetchDeptList();
    if (list) {
      setExpandedKeys(expandFirstLevel(list));
    }
    await fetchDeptOptions();
  });

  usePullDownRefresh(async () => {
    try {
      const list = await fetchDeptList();
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
      await searchDepts(value.trim());
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

  // 新增部门（顶级）
  const handleAdd = () => {
    setEditingId(undefined);
    setFormData({ ...DEFAULT_FORM, parentId: 0 });
    setShowFormDialog(true);
  };

  // 新增子部门
  const handleAddChild = (parentId: number) => {
    setEditingId(undefined);
    setFormData({ ...DEFAULT_FORM, parentId });
    setShowFormDialog(true);
  };

  // 编辑部门
  const handleEdit = async (node: DeptVO) => {
    if (!node.id) return;
    try {
      const form = await fetchDeptForm(node.id);
      setEditingId(node.id);
      setFormData(form);
      setShowFormDialog(true);
    } catch {
      // 错误已在 hook 中处理
    }
  };

  // 删除部门
  const handleDelete = async (node: DeptVO) => {
    const confirmed = await confirmDialog({
      title: "确认删除",
      content: `确定要删除部门 "${node.name}" 吗？如有子部门需先删除子部门，此操作不可恢复。`,
      confirmText: "删除",
      cancelText: "取消",
    });
    if (!confirmed) return;
    await confirmDelete(node);
  };

  const confirmDelete = async (node: DeptVO) => {
    if (!node?.id) return;
    try {
      await deleteDept(node.id);
    } catch {
      // 错误已在 hook 中处理
    }
  };

  // 表单字段更新
  const handleFieldChange = (field: keyof DeptForm, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value } as DeptForm));
  };

  // 表单校验
  const validateForm = (): boolean => {
    if (!formData.name?.trim()) {
      Taro.showToast({ title: "部门名称不能为空", icon: "none" });
      return false;
    }
    return true;
  };

  // 提交表单
  const submitForm = async () => {
    if (!validateForm()) return;

    setSubmitting(true);
    try {
      if (editingId) {
        await updateDept(editingId, formData);
      } else {
        await createDept(formData);
      }
      setShowFormDialog(false);
    } catch {
      // 错误已在 hook 中处理
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <View className="dept-page">
      <Navbar title="部门管理">
        <Navbar.NavLeft>
          <ArrowLeft onClick={() => Taro.navigateBack()} />
        </Navbar.NavLeft>
        <Navbar.NavRight>
          {hasPermission("sys:dept:add") && <Add onClick={handleAdd} />}
        </Navbar.NavRight>
      </Navbar>

      {/* 搜索栏 */}
      <View className="search-bar">
        <Search
          placeholder="请输入部门名称"
          value={searchKeyword}
          onChange={(e) => setSearchKeyword(e.detail.value)}
          onSearch={handleSearch}
          onClear={() => performSearch("")}
        />
      </View>

      {/* 部门树形列表 */}
      <ScrollView scrollY className="dept-list">
        {loading && deptList.length === 0 ? (
          <Loading>加载中...</Loading>
        ) : deptList.length === 0 ? (
          <Empty>
            <Empty.Image />
            <Empty.Description>暂无部门数据</Empty.Description>
            {hasPermission("sys:dept:add") && (
              <Button color="primary" size="small" onClick={handleAdd}>
                新增部门
              </Button>
            )}
          </Empty>
        ) : (
          deptList.map((dept) => (
            <TreeNode
              key={dept.id}
              node={dept}
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

      {/* 部门表单弹窗 */}
      <Popup
        open={showFormDialog}
        onClose={() => setShowFormDialog(false)}
        placement="bottom"
        style={{ height: "60%" }}
      >
        <View className="form-popup">
          <View className="form-header">
            <Text className="form-title">
              {editingId ? "编辑部门" : "新增部门"}
            </Text>
          </View>
          <View className="form-body">
            {/* 上级部门 */}
            <View className="form-item">
              <Text className="form-label">上级部门</Text>
              <View className="parent-dept-display">
                {formData.parentId === 0
                  ? "顶级部门"
                  : deptOptions.find(
                      (opt) => Number(opt.value) === formData.parentId
                    )?.label || "未知部门"}
              </View>
            </View>

            {/* 部门名称 */}
            <View className="form-item">
              <Text className="form-label">部门名称 *</Text>
              <Input
                className="form-input"
                placeholder="请输入部门名称"
                value={formData.name || ""}
                onInput={(e) => handleFieldChange("name", e.detail.value)}
              />
            </View>

            {/* 排序 */}
            <View className="form-item">
              <Text className="form-label">排序</Text>
              <Input
                className="form-input"
                type="number"
                placeholder="请输入排序值"
                value={String(formData.sort ?? 1)}
                onInput={(e) =>
                  handleFieldChange("sort", Number(e.detail.value) || 1)
                }
              />
            </View>

            {/* 状态 */}
            <View className="form-item">
              <Text className="form-label">状态</Text>
              <View className="form-switch">
                <Switch
                  checked={formData.status === 1}
                  onChange={(checked) =>
                    handleFieldChange("status", checked ? 1 : 0)
                  }
                />
                <Text>{formData.status === 1 ? "启用" : "禁用"}</Text>
              </View>
            </View>
          </View>
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

export default DeptPage;
