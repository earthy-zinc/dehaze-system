import React, { useState, useEffect, useCallback } from "react";
import { View, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Loading, Empty } from "@taroify/core";
import { Add } from "@taroify/icons";
import { DatasetAPI } from "dehaze-sdk-js";
import type { Dataset } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import ErrorState from "@/components/common/ErrorState";
import { getErrorMessage } from "@/utils/error";
import { usePermission } from "@/hooks/usePermission";
import { confirmDialog } from "@/utils/dialog";
import SearchBar from "@/components/common/SearchBar";
import DatasetList from "@/pages/dataset/components/DatasetList";
import DatasetFormDialog, {
  DatasetFormData,
} from "@/pages/dataset/components/DatasetFormDialog";
import "./index.less";

interface DialogState {
  visible: boolean;
  mode: "create" | "edit";
  dataset: Dataset | null;
  defaultParentId: number;
}

const DatasetManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canAdd = hasPermission("sys:dataset:add");
  const canEdit = hasPermission("sys:dataset:edit");
  const canDelete = hasPermission("sys:dataset:delete");

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [searchKeyword, setSearchKeyword] = useState("");
  const [expandedIds, setExpandedIds] = useState<number[]>([]);
  const [childrenMap, setChildrenMap] = useState<Record<number, Dataset[]>>({});
  const [childrenLoading, setChildrenLoading] = useState<
    Record<number, boolean>
  >({});

  const [dialog, setDialog] = useState<DialogState>({
    visible: false,
    mode: "create",
    dataset: null,
    defaultParentId: 0,
  });
  // 表单提交时本地 loading 态（UI 不需要单独展示，只用于防重）
  const [, setSubmitting] = useState(false);

  const fetchDatasets = useCallback(async (page: number, keyword: string) => {
    setLoading(true);
    setLoadError(null);
    try {
      const res = await DatasetAPI.getList({
        pageNum: page,
        pageSize: 10,
        keyword: keyword || undefined,
      });
      setDatasets(res.list);
      setTotal(res.total);
      setPageNum(page);
    } catch (err: unknown) {
      setLoadError(getErrorMessage(err, "加载数据集失败"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDatasets(1, "");
  }, [fetchDatasets]);

  const handleSearch = useCallback(
    (keyword: string) => {
      setSearchKeyword(keyword);
      fetchDatasets(1, keyword);
    },
    [fetchDatasets]
  );

  const toggleExpand = useCallback(
    async (id: number) => {
      setExpandedIds((prev) => {
        if (prev.includes(id)) {
          return prev.filter((i) => i !== id);
        } else {
          return [...prev, id];
        }
      });

      if (!childrenMap[id]) {
        setChildrenLoading((prev) => ({ ...prev, [id]: true }));
        try {
          const children = await DatasetAPI.getChildren(id);
          setChildrenMap((prev) => ({ ...prev, [id]: children || [] }));
        } catch {
          // 静默处理
        } finally {
          setChildrenLoading((prev) => ({ ...prev, [id]: false }));
        }
      }
    },
    [childrenMap]
  );

  const handleAddRoot = () => {
    if (!canAdd) return;
    setDialog({
      visible: true,
      mode: "create",
      dataset: null,
      defaultParentId: 0,
    });
  };

  const handleAddChild = (parent: Dataset) => {
    if (!canAdd) return;
    setDialog({
      visible: true,
      mode: "create",
      dataset: null,
      defaultParentId: parent.id,
    });
  };

  const handleEdit = (dataset: Dataset) => {
    if (!canEdit) return;
    setDialog({
      visible: true,
      mode: "edit",
      dataset,
      defaultParentId: dataset.parentId ?? 0,
    });
  };

  const handleDelete = async (dataset: Dataset) => {
    if (!canDelete) return;
    const confirmed = await confirmDialog({
      title: "确认删除",
      content: `确定要删除数据集「${dataset.name}」吗？此操作不可恢复。`,
      confirmText: "删除",
      confirmColor: "#ef4444",
    });
    if (!confirmed) return;
    try {
      await DatasetAPI.deleteById(dataset.id);
      Taro.showToast({ title: "删除成功", icon: "success" });
      fetchDatasets(pageNum, searchKeyword);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "删除失败"), icon: "none" });
    }
  };

  const handleSubmit = async (data: DatasetFormData): Promise<boolean> => {
    setSubmitting(true);
    try {
      if (dialog.mode === "create") {
        await DatasetAPI.add({
          parentId: data.parentId,
          type: data.type,
          name: data.name,
          description: data.description,
          status: data.status,
        });
        Taro.showToast({ title: "创建成功", icon: "success" });
      } else {
        if (!dialog.dataset) return false;
        await DatasetAPI.update(dialog.dataset.id, {
          type: data.type,
          name: data.name,
          description: data.description,
          status: data.status,
        });
        Taro.showToast({ title: "更新成功", icon: "success" });
      }
      setDialog((prev) => ({ ...prev, visible: false }));
      fetchDatasets(pageNum, searchKeyword);
      return true;
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "操作失败"), icon: "none" });
      return false;
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <PageLayout level="L2" title="数据集管理">
      <View className="dataset-manage-page">
        <View className="search-section">
          <SearchBar
            placeholder="搜索数据集..."
            value={searchKeyword}
            onSearch={handleSearch}
            onClear={() => handleSearch("")}
          />
        </View>

        <View className="action-bar">
          {canAdd && (
            <View className="add-btn" onClick={handleAddRoot}>
              <Add size="16" color="#ffffff" />
              <View className="add-btn-text">新增数据集</View>
            </View>
          )}
        </View>

        {loading && datasets.length === 0 ? (
          <View className="loading-wrapper">
            <Loading>加载中...</Loading>
          </View>
        ) : loadError && datasets.length === 0 ? (
          <ErrorState
            message={loadError}
            onRetry={() => fetchDatasets(1, searchKeyword)}
          />
        ) : datasets.length === 0 ? (
          <Empty>
            <Empty.Description>暂无数据集</Empty.Description>
          </Empty>
        ) : (
          <ScrollView scrollY className="dataset-list-scroll">
            <DatasetList
              datasets={datasets}
              loading={loading}
              error={loadError}
              onRetry={() => fetchDatasets(1, searchKeyword)}
              hasMore={datasets.length < total}
              onLoadMore={() => fetchDatasets(pageNum + 1, searchKeyword)}
              onDatasetClick={(dataset) => {
                Taro.navigateTo({
                  url: `/pages/dataset/index?datasetId=${dataset.id}`,
                });
              }}
              expandedIds={expandedIds}
              childrenMap={childrenMap}
              childrenLoading={childrenLoading}
              onToggleExpand={toggleExpand}
              onAddChild={canAdd ? handleAddChild : undefined}
              onEdit={canEdit ? handleEdit : undefined}
              onDelete={canDelete ? handleDelete : undefined}
            />
          </ScrollView>
        )}

        <DatasetFormDialog
          visible={dialog.visible}
          mode={dialog.mode}
          dataset={dialog.dataset}
          options={[]}
          defaultParentId={dialog.defaultParentId}
          onSubmit={handleSubmit}
          onClose={() => setDialog((prev) => ({ ...prev, visible: false }))}
        />
      </View>
    </PageLayout>
  );
};

export default DatasetManagePage;
