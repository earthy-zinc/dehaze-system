import React, { useState } from 'react';
import { View, Text, Input, Textarea } from '@tarojs/components';
import Taro, { useLoad, usePullDownRefresh, useReachBottom } from '@tarojs/taro';
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
import { ArrowLeft, Add, Edit, Delete, SettingOutlined } from '@taroify/icons';
import { useDictManagement } from '@/hooks/useDictManagement';
import { usePermission } from '@/hooks/usePermission';
import type { DictTypeForm, DictForm } from 'dehaze-sdk-js';
import './index.scss';

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

  const [searchKeyword, setSearchKeyword] = useState('');
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deletingType, setDeletingType] = useState<any>(null);

  // 字典类型表单弹窗
  const [showTypeDialog, setShowTypeDialog] = useState(false);
  const [editingTypeId, setEditingTypeId] = useState<number | undefined>();
  const [typeForm, setTypeForm] = useState<DictTypeForm>({
    name: '',
    code: '',
    status: 1,
    remark: '',
  });
  const [submittingType, setSubmittingType] = useState(false);

  // 字典数据管理弹窗
  const [showItemDialog, setShowItemDialog] = useState(false);
  const [currentTypeCode, setCurrentTypeCode] = useState('');
  const [currentTypeName, setCurrentTypeName] = useState('');

  // 字典数据表单弹窗
  const [showItemFormDialog, setShowItemFormDialog] = useState(false);
  const [editingItemId, setEditingItemId] = useState<number | undefined>();
  const [itemForm, setItemForm] = useState<DictForm>({
    name: '',
    value: '',
    typeCode: '',
    sort: 1,
    status: 1,
    defaulted: 0,
    remark: '',
  });
  const [submittingItem, setSubmittingItem] = useState(false);

  // 字典数据删除确认
  const [showItemDeleteDialog, setShowItemDeleteDialog] = useState(false);
  const [deletingItem, setDeletingItem] = useState<any>(null);

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
  const handleSearch = async (event: any) => {
    const value = event.detail?.value || '';
    setSearchKeyword(value);
    if (value.trim()) {
      await searchDictTypes(value.trim());
    } else {
      await resetDictTypeQuery();
    }
  };

  // 新增字典类型
  const handleAddType = () => {
    setEditingTypeId(undefined);
    setTypeForm({ name: '', code: '', status: 1, remark: '' });
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
      Taro.showToast({ title: '字典名称不能为空', icon: 'none' });
      return;
    }
    if (!typeForm.code?.trim()) {
      Taro.showToast({ title: '字典编码不能为空', icon: 'none' });
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
  const handleDeleteType = (dictType: any) => {
    setDeletingType(dictType);
    setShowDeleteDialog(true);
  };

  const confirmDeleteType = async () => {
    if (!deletingType) return;
    try {
      await deleteDictTypes(String(deletingType.id));
      setShowDeleteDialog(false);
      setDeletingType(null);
    } catch {
      // 错误已在 hook 中处理
    }
  };

  // 管理字典数据
  const handleManageItems = (dictType: any) => {
    setCurrentTypeCode(dictType.code);
    setCurrentTypeName(dictType.name);
    setShowItemDialog(true);
    fetchDictItems({ typeCode: dictType.code, pageNum: 1 });
  };

  // 新增字典数据
  const handleAddItem = () => {
    setEditingItemId(undefined);
    setItemForm({
      name: '',
      value: '',
      typeCode: currentTypeCode,
      sort: 1,
      status: 1,
      defaulted: 0,
      remark: '',
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
      Taro.showToast({ title: '字典标签不能为空', icon: 'none' });
      return;
    }
    if (!itemForm.value?.trim()) {
      Taro.showToast({ title: '字典键值不能为空', icon: 'none' });
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
  const handleDeleteItem = (item: any) => {
    setDeletingItem(item);
    setShowItemDeleteDialog(true);
  };

  const confirmDeleteItem = async () => {
    if (!deletingItem) return;
    try {
      await deleteDictItems(String(deletingItem.id));
      setShowItemDeleteDialog(false);
      setDeletingItem(null);
      await fetchDictItems({ typeCode: currentTypeCode });
    } catch {
      // 错误已在 hook 中处理
    }
  };

  const getStatusTag = (status?: number) => {
    return status === 1
      ? <Tag color="success" size="small">启用</Tag>
      : <Tag color="danger" size="small">禁用</Tag>;
  };

  return (
    <View className="dict-page">
      <Navbar title="字典管理">
        <Navbar.NavLeft>
          <ArrowLeft onClick={() => Taro.navigateBack()} />
        </Navbar.NavLeft>
        <Navbar.NavRight>
          {hasPermission('sys:dict:type:add') && (
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
          onClear={() => handleSearch('')}
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
            {hasPermission('sys:dict:type:add') && (
              <Button color="primary" size="small" onClick={handleAddType}>
                新增字典类型
              </Button>
            )}
          </Empty>
        ) : (
          dictTypes.map((dictType) => (
            <SwipeCell key={dictType.id} className="dict-swipe-cell">
              <SwipeCell.Actions side="right">
                {hasPermission('sys:dict:data:list') && (
                  <Button
                    className="action-btn data-btn"
                    size="small"
                    onClick={() => handleManageItems(dictType)}
                  >
                    <SettingOutlined />
                    数据
                  </Button>
                )}
                {hasPermission('sys:dict:type:edit') && (
                  <Button
                    className="action-btn edit-btn"
                    size="small"
                    onClick={() => handleEditType(dictType.id)}
                  >
                    <Edit />
                    编辑
                  </Button>
                )}
                {hasPermission('sys:dict:type:delete') && (
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
                  {getStatusTag(dictType.status)}
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

      {/* 字典类型删除确认弹窗 */}
      <Dialog
        open={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        title="确认删除"
      >
        <Dialog.Content>
          确定要删除字典类型 "{deletingType?.name}" 吗？关联的字典数据也将被删除，此操作不可恢复。
        </Dialog.Content>
        <Dialog.Actions>
          <Button onClick={() => setShowDeleteDialog(false)}>取消</Button>
          <Button color="danger" onClick={confirmDeleteType}>删除</Button>
        </Dialog.Actions>
      </Dialog>

      {/* 字典类型表单弹窗 */}
      <Popup
        open={showTypeDialog}
        onClose={() => setShowTypeDialog(false)}
        placement="bottom"
        style={{ height: '60%' }}
      >
        <View className="form-popup">
          <View className="form-header">
            <Text className="form-title">
              {editingTypeId ? '编辑字典类型' : '新增字典类型'}
            </Text>
          </View>
          <View className="form-body">
            <View className="form-item">
              <Text className="form-label">字典名称 *</Text>
              <Input
                className="form-input"
                placeholder="请输入字典名称"
                value={typeForm.name || ''}
                onInput={(e) => setTypeForm({ ...typeForm, name: e.detail.value })}
              />
            </View>
            <View className="form-item">
              <Text className="form-label">字典编码 *</Text>
              <Input
                className="form-input"
                placeholder="请输入字典编码"
                value={typeForm.code || ''}
                disabled={!!editingTypeId}
                onInput={(e) => setTypeForm({ ...typeForm, code: e.detail.value })}
              />
            </View>
            <View className="form-item">
              <Text className="form-label">状态</Text>
              <View className="form-switch">
                <Switch
                  checked={typeForm.status === 1}
                  onChange={(checked) => setTypeForm({ ...typeForm, status: checked ? 1 : 0 })}
                />
                <Text>{typeForm.status === 1 ? '启用' : '禁用'}</Text>
              </View>
            </View>
            <View className="form-item">
              <Text className="form-label">备注</Text>
              <Textarea
                className="form-textarea"
                placeholder="请输入备注信息（最多200字符）"
                maxlength={200}
                value={typeForm.remark || ''}
                onInput={(e) => setTypeForm({ ...typeForm, remark: e.detail.value })}
              />
            </View>
          </View>
          <View className="form-footer">
            <Button onClick={() => setShowTypeDialog(false)}>取消</Button>
            <Button color="primary" loading={submittingType} onClick={submitTypeForm}>
              确定
            </Button>
          </View>
        </View>
      </Popup>

      {/* 字典数据管理弹窗 */}
      <Popup
        open={showItemDialog}
        onClose={() => setShowItemDialog(false)}
        placement="bottom"
        style={{ height: '80%' }}
      >
        <View className="item-popup">
          <View className="item-header">
            <Text className="item-title">字典数据 - {currentTypeName}</Text>
            {hasPermission('sys:dict:data:add') && (
              <Button size="small" color="primary" onClick={handleAddItem}>
                <Add /> 新增
              </Button>
            )}
          </View>

          <View className="item-list">
            {dictItemLoading && dictItems.length === 0 ? (
              <Loading>加载中...</Loading>
            ) : dictItems.length === 0 ? (
              <Empty>
                <Empty.Description>暂无字典数据</Empty.Description>
              </Empty>
            ) : (
              dictItems.map((item) => (
                <SwipeCell key={item.id} className="item-swipe-cell">
                  <SwipeCell.Actions side="right">
                    {hasPermission('sys:dict:data:edit') && (
                      <Button
                        size="small"
                        onClick={() => handleEditItem(item.id!)}
                      >
                        <Edit /> 编辑
                      </Button>
                    )}
                    {hasPermission('sys:dict:data:delete') && (
                      <Button
                        color="danger"
                        size="small"
                        onClick={() => handleDeleteItem(item)}
                      >
                        <Delete /> 删除
                      </Button>
                    )}
                  </SwipeCell.Actions>
                  <Cell className="item-cell">
                    <View className="item-info">
                      <View className="item-name">{item.name}</View>
                      <View className="item-value">值: {item.value}</View>
                    </View>
                    <View className="item-status">
                      {getStatusTag(item.status)}
                    </View>
                  </Cell>
                </SwipeCell>
              ))
            )}
          </View>

          {dictItemLoading && dictItems.length > 0 && (
            <View className="loading-more">
              <Loading size="small">加载中...</Loading>
            </View>
          )}
        </View>
      </Popup>

      {/* 字典数据表单弹窗 */}
      <Popup
        open={showItemFormDialog}
        onClose={() => setShowItemFormDialog(false)}
        placement="bottom"
        style={{ height: '70%' }}
      >
        <View className="form-popup">
          <View className="form-header">
            <Text className="form-title">
              {editingItemId ? '编辑字典数据' : '新增字典数据'}
            </Text>
          </View>
          <View className="form-body">
            <View className="form-item">
              <Text className="form-label">字典标签 *</Text>
              <Input
                className="form-input"
                placeholder="请输入字典标签"
                value={itemForm.name || ''}
                onInput={(e) => setItemForm({ ...itemForm, name: e.detail.value })}
              />
            </View>
            <View className="form-item">
              <Text className="form-label">字典键值 *</Text>
              <Input
                className="form-input"
                placeholder="请输入字典键值"
                value={itemForm.value || ''}
                onInput={(e) => setItemForm({ ...itemForm, value: e.detail.value })}
              />
            </View>
            <View className="form-item">
              <Text className="form-label">排序</Text>
              <Input
                className="form-input"
                type="number"
                placeholder="请输入排序值"
                value={String(itemForm.sort || 1)}
                onInput={(e) => setItemForm({ ...itemForm, sort: Number(e.detail.value) || 1 })}
              />
            </View>
            <View className="form-item">
              <Text className="form-label">是否默认</Text>
              <View className="form-switch">
                <Switch
                  checked={itemForm.defaulted === 1}
                  onChange={(checked) => setItemForm({ ...itemForm, defaulted: checked ? 1 : 0 })}
                />
                <Text>{itemForm.defaulted === 1 ? '是' : '否'}</Text>
              </View>
            </View>
            <View className="form-item">
              <Text className="form-label">状态</Text>
              <View className="form-switch">
                <Switch
                  checked={itemForm.status === 1}
                  onChange={(checked) => setItemForm({ ...itemForm, status: checked ? 1 : 0 })}
                />
                <Text>{itemForm.status === 1 ? '启用' : '禁用'}</Text>
              </View>
            </View>
            <View className="form-item">
              <Text className="form-label">备注</Text>
              <Textarea
                className="form-textarea"
                placeholder="请输入备注信息（最多200字符）"
                maxlength={200}
                value={itemForm.remark || ''}
                onInput={(e) => setItemForm({ ...itemForm, remark: e.detail.value })}
              />
            </View>
          </View>
          <View className="form-footer">
            <Button onClick={() => setShowItemFormDialog(false)}>取消</Button>
            <Button color="primary" loading={submittingItem} onClick={submitItemForm}>
              确定
            </Button>
          </View>
        </View>
      </Popup>

      {/* 字典数据删除确认弹窗 */}
      <Dialog
        open={showItemDeleteDialog}
        onClose={() => setShowItemDeleteDialog(false)}
        title="确认删除"
      >
        <Dialog.Content>
          确定要删除字典数据 "{deletingItem?.name}" 吗？
        </Dialog.Content>
        <Dialog.Actions>
          <Button onClick={() => setShowItemDeleteDialog(false)}>取消</Button>
          <Button color="danger" onClick={confirmDeleteItem}>删除</Button>
        </Dialog.Actions>
      </Dialog>
    </View>
  );
};

export default DictPage;
