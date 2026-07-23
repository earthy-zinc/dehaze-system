import React, { useState } from "react";
import { View } from "@tarojs/components";
import type { BaseEventOrig } from "@tarojs/components";
import Taro, {
  useLoad,
  usePullDownRefresh,
  useReachBottom,
} from "@tarojs/taro";
import {
  Navbar,
  Search,
  Button,
  Loading,
  Empty,
  SwipeCell,
  Cell,
} from "@taroify/core";
import { confirmDialog } from "@/utils/dialog";
import { ArrowLeft, Add, Edit, Delete, SettingOutlined } from "@taroify/icons";
import { useDictManagement } from "@/hooks/useDictManagement";
import { usePermission } from "@/hooks/usePermission";
import type {
  DictTypeForm,
  DictForm,
  DictTypePageVO,
  DictPageVO,
} from "dehaze-sdk-js";
import StatusTag from "@/components/common/StatusTag";
import DictTypeFormDialog from "./components/DictTypeFormDialog";
import DictItemDialog from "./components/DictItemDialog";
import DictItemFormDialog from "./components/DictItemFormDialog";
import "./index.scss";

const DictPage: React.FC = () => {
  const {
    dictTypes,
    dictTypeLoading,
    dictTypeTotal,
    dictTypeQueryParams,
    fetchDictTypes,
    createDictType,
    fetchDictTypeForm,
    updateDictType,
    deleteDictTypes,
    searchDictTypes,
    resetDictTypeQuery,
    // 字典数据
    dictItems,
    dictItemLoading,
    fetchDictItems,
    createDictItem,
    fetchDictItemForm,
    updateDictItem,
    deleteDictItems,
  } = useDictManagement();

  const { hasPermission } = usePermission();

  const [searchKeyword, setSearchKeyword] = useState("");

  // 字典类型表单弹窗
  const [showTypeDialog, setShowTypeDialog] = useState(false);
  const [editingTypeId, setEditingTypeId] = useState<number | undefined>();
  const [typeForm, setTypeForm] = useState<DictTypeForm>({
    name: "",
    code: "",
    status: 1,
    remark: "",
  });
  const [submittingType, setSubmittingType] = useState(false);

  // 字典数据管理弹窗
  const [showItemDialog, setShowItemDialog] = useState(false);
  const [currentTypeCode, setCurrentTypeCode] = useState("");
  const [currentTypeName, setCurrentTypeName] = useState("");

  // 字典数据表单弹窗
  const [showItemFormDialog, setShowItemFormDialog] = useState(false);
  const [editingItemId, setEditingItemId] = useState<number | undefined>();
  const [itemForm, setItemForm] = useState<DictForm>({
    name: "",
    value: "",
    typeCode: "",
    sort: 1,
    status: 1,
    defaulted: 0,
    remark: "",
  });
  const [submittingItem, setSubmittingItem] = useState(false);

  useLoad(async () => {
    await fetchDictTypes();
  });

  usePullDownRefresh(async () => {
    try {
      await fetchDictTypes({ pageNum: 1 });
      Taro.stopPullDownRefresh();
    } catch {
      Taro.stopPullDownRefresh();
    }
  });

  useReachBottom(async () => {
    if (dictTypes.length < dictTypeTotal) {
      await fetchDictTypes({ pageNum: dictTypeQueryParams.pageNum + 1 });
    }
  });

  // 搜索
  const performSearch = async (value: string) => {
    setSearchKeyword(value);
    if (value.trim()) {
      await searchDictTypes(value.trim());
    } else {
      await resetDictTypeQuery();
    }
  };

  const handleSearch = async (event: BaseEventOrig<{ value: string }>) => {
    await performSearch(event.detail?.value || "");
  };

  // 新增字典类型
  const handleAddType = () => {
    setEditingTypeId(undefined);
    setTypeForm({ name: "", code: "", status: 1, remark: "" });
    setShowTypeDialog(true);
  };

  // 编辑字典类型
  const handleEditType = async (id: number) => {
    try {
      const formData = await fetchDictTypeForm(id);
      setEditingTypeId(id);
      setTypeForm(formData);
      setShowTypeDialog(true);
    } catch {
      // 错误已在 hook 中处理
    }
  };

  // 提交字典类型表单
  const submitTypeForm = async () => {
    if (!typeForm.name?.trim()) {
      Taro.showToast({ title: "字典名称不能为空", icon: "none" });
      return;
    }
    if (!typeForm.code?.trim()) {
      Taro.showToast({ title: "字典编码不能为空", icon: "none" });
      return;
    }

    setSubmittingType(true);
    try {
      if (editingTypeId) {
        await updateDictType(editingTypeId, typeForm);
      } else {
        await createDictType(typeForm);
      }
      setShowTypeDialog(false);
    } catch {
      // 错误已在 hook 中处理
    } finally {
      setSubmittingType(false);
    }
  };

  // 删除字典类型
  const handleDeleteType = async (dictType: DictTypePageVO) => {
    const confirmed = await confirmDialog({
      title: "确认删除",
      content: `确定要删除字典类型 "${dictType.name}" 吗？关联的字典数据也将被删除，此操作不可恢复。`,
      confirmText: "删除",
      cancelText: "取消",
    });
    if (!confirmed) return;
    await confirmDeleteType(dictType);
  };

  const confirmDeleteType = async (dictType: DictTypePageVO) => {
    if (!dictType) return;
    try {
      await deleteDictTypes(String(dictType.id));
    } catch {
      // 错误已在 hook 中处理
    }
  };

  // 管理字典数据
  const handleManageItems = (dictType: DictTypePageVO) => {
    setCurrentTypeCode(dictType.code);
    setCurrentTypeName(dictType.name);
    setShowItemDialog(true);
    fetchDictItems({ typeCode: dictType.code, pageNum: 1 });
  };

  // 新增字典数据
  const handleAddItem = () => {
    setEditingItemId(undefined);
    setItemForm({
      name: "",
      value: "",
      typeCode: currentTypeCode,
      sort: 1,
      status: 1,
      defaulted: 0,
      remark: "",
    });
    setShowItemFormDialog(true);
  };

  // 编辑字典数据
  const handleEditItem = async (id: number) => {
    try {
      const formData = await fetchDictItemForm(id);
      setEditingItemId(id);
      setItemForm(formData);
      setShowItemFormDialog(true);
    } catch {
      // 错误已在 hook 中处理
    }
  };

  // 提交字典数据表单
  const submitItemForm = async () => {
    if (!itemForm.name?.trim()) {
      Taro.showToast({ title: "字典标签不能为空", icon: "none" });
      return;
    }
    if (!itemForm.value?.trim()) {
      Taro.showToast({ title: "字典键值不能为空", icon: "none" });
      return;
    }

    setSubmittingItem(true);
    try {
      if (editingItemId) {
        await updateDictItem(editingItemId, itemForm);
      } else {
        await createDictItem(itemForm);
      }
      setShowItemFormDialog(false);
      await fetchDictItems({ typeCode: currentTypeCode });
    } catch {
      // 错误已在 hook 中处理
    } finally {
      setSubmittingItem(false);
    }
  };

  // 删除字典数据
  const handleDeleteItem = async (item: DictPageVO) => {
    const confirmed = await confirmDialog({
      title: "确认删除",
      content: `确定要删除字典数据 "${item.name}" 吗？`,
      confirmText: "删除",
      cancelText: "取消",
    });
    if (!confirmed) return;
    await confirmDeleteItem(item);
  };

  const confirmDeleteItem = async (item: DictPageVO) => {
    if (!item) return;
    try {
      await deleteDictItems(String(item.id));
      await fetchDictItems({ typeCode: currentTypeCode });
    } catch {
      // 错误已在 hook 中处理
    }
  };

  return (
    <View className="dict-page">
      <Navbar title="字典管理">
        <Navbar.NavLeft>
          <ArrowLeft onClick={() => Taro.navigateBack()} />
        </Navbar.NavLeft>
        <Navbar.NavRight>
          {hasPermission("sys:dict:type:add") && (
            <Add onClick={handleAddType} />
          )}
        </Navbar.NavRight>
      </Navbar>

      {/* 搜索栏 */}
      <View className="search-bar">
        <Search
          placeholder="请输入字典名称或编码"
          value={searchKeyword}
          onChange={(e) => setSearchKeyword(e.detail.value)}
          onSearch={handleSearch}
          onClear={() => performSearch("")}
        />
      </View>

      {/* 字典类型列表 */}
      <View className="dict-list">
        {dictTypeLoading && dictTypes.length === 0 ? (
          <Loading>加载中...</Loading>
        ) : dictTypes.length === 0 ? (
          <Empty>
            <Empty.Image />
            <Empty.Description>暂无字典类型</Empty.Description>
            {hasPermission("sys:dict:type:add") && (
              <Button color="primary" size="small" onClick={handleAddType}>
                新增字典类型
              </Button>
            )}
          </Empty>
        ) : (
          dictTypes.map((dictType) => (
            <SwipeCell key={dictType.id} className="dict-swipe-cell">
              <SwipeCell.Actions side="right">
                {hasPermission("sys:dict:data:list") && (
                  <Button
                    className="action-btn data-btn"
                    size="small"
                    onClick={() => handleManageItems(dictType)}
                  >
                    <SettingOutlined />
                    数据
                  </Button>
                )}
                {hasPermission("sys:dict:type:edit") && (
                  <Button
                    className="action-btn edit-btn"
                    size="small"
                    onClick={() => handleEditType(dictType.id)}
                  >
                    <Edit />
                    编辑
                  </Button>
                )}
                {hasPermission("sys:dict:type:delete") && (
                  <Button
                    className="action-btn delete-btn"
                    size="small"
                    onClick={() => handleDeleteType(dictType)}
                  >
                    <Delete />
                    删除
                  </Button>
                )}
              </SwipeCell.Actions>
              <Cell className="dict-cell">
                <View className="dict-info">
                  <View className="dict-name">{dictType.name}</View>
                  <View className="dict-code">编码: {dictType.code}</View>
                </View>
                <View className="dict-status">
                  <StatusTag status={dictType.status} />
                </View>
                {dictType.remark && (
                  <View className="dict-remark">备注: {dictType.remark}</View>
                )}
              </Cell>
            </SwipeCell>
          ))
        )}
      </View>

      {/* 加载更多 */}
      {dictTypeLoading && dictTypes.length > 0 && (
        <View className="loading-more">
          <Loading size="small">加载中...</Loading>
        </View>
      )}

      {/* 字典类型表单弹窗 */}
      <DictTypeFormDialog
        open={showTypeDialog}
        editingId={editingTypeId}
        form={typeForm}
        submitting={submittingType}
        onClose={() => setShowTypeDialog(false)}
        onFormChange={setTypeForm}
        onSubmit={submitTypeForm}
      />

      {/* 字典数据管理弹窗 */}
      <DictItemDialog
        open={showItemDialog}
        typeName={currentTypeName}
        items={dictItems}
        loading={dictItemLoading}
        canAdd={hasPermission("sys:dict:data:add")}
        canEdit={hasPermission("sys:dict:data:edit")}
        canDelete={hasPermission("sys:dict:data:delete")}
        onClose={() => setShowItemDialog(false)}
        onAdd={handleAddItem}
        onEdit={handleEditItem}
        onDelete={handleDeleteItem}
      />

      {/* 字典数据表单弹窗 */}
      <DictItemFormDialog
        open={showItemFormDialog}
        editingId={editingItemId}
        form={itemForm}
        submitting={submittingItem}
        onClose={() => setShowItemFormDialog(false)}
        onFormChange={setItemForm}
        onSubmit={submitItemForm}
      />
    </View>
  );
};

export default DictPage;
