import React, { useState, useCallback, useMemo } from "react";
import { View, Text, Input, ScrollView } from "@tarojs/components";
import Taro, { useLoad, usePullDownRefresh } from "@tarojs/taro";
import { Loading, Tag, Button, Popup, Input as TInput, Textarea } from "@taroify/core";
import { Search } from "@taroify/icons";
import { AlgorithmAPI } from "dehaze-sdk-js";
import type { Algorithm, AlgorithmAuditForm } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import ErrorState from "@/components/common/ErrorState";
import { getErrorMessage } from "@/utils/error";
import { usePermission } from "@/hooks/usePermission";
import {
  STATUS_INFO,
  STATUS_FILTERS,
  flattenTree,
  filterTree,
  updateAlgorithmInTree,
  removeAlgorithmFromTree,
} from "@/pages/algorithm/utils";
import type { FlatNode } from "@/pages/algorithm/utils";
import AlgorithmDetailPopup from "@/pages/algorithm/components/AlgorithmDetailPopup";
import AlgorithmAuditPopup from "@/pages/algorithm/components/AlgorithmAuditPopup";
import "./index.less";

const AlgorithmManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canAdd = hasPermission("sys:algorithm:add");
  const canAudit = hasPermission("sys:algorithm:audit");
  const canEdit = hasPermission("sys:algorithm:edit");
  const canDelete = hasPermission("sys:algorithm:delete");

  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [searchKeyword, setSearchKeyword] = useState("");
  const [statusFilter, setStatusFilter] = useState<number | "">("");

  const [detailAlgo, setDetailAlgo] = useState<Algorithm | null>(null);
  const [detailVisible, setDetailVisible] = useState(false);

  const [auditAlgo, setAuditAlgo] = useState<Algorithm | null>(null);
  const [auditVisible, setAuditVisible] = useState(false);
  const [auditApproved, setAuditApproved] = useState(true);
  const [auditRemark, setAuditRemark] = useState("");
  const [auditSubmitting, setAuditSubmitting] = useState(false);

  const [actionLoadingId, setActionLoadingId] = useState<number | null>(null);

  // ==================== 新增算法 ====================
  const [addVisible, setAddVisible] = useState(false);
  const [addForm, setAddForm] = useState({ name: "", type: "", version: "", description: "", path: "", importPath: "" });
  const [addSubmitting, setAddSubmitting] = useState(false);

  const resetAddForm = () => setAddForm({ name: "", type: "", version: "", description: "", path: "", importPath: "" });

  const fetchAlgorithms = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const data = await AlgorithmAPI.getList();
      setAlgorithms(data || []);
    } catch (err: unknown) {
      setLoadError(getErrorMessage(err, "加载失败，请重试"));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleAdd = useCallback(async () => {
    const { name, type, version } = addForm;
    if (!name.trim() || !type.trim() || !version.trim()) {
      Taro.showToast({ title: "名称/类型/版本为必填", icon: "none" });
      return;
    }
    if (!/^v?\d+\.\d+\.\d+$/.test(version.trim())) {
      Taro.showToast({ title: "版本号格式: vX.Y.Z", icon: "none" });
      return;
    }
    setAddSubmitting(true);
    try {
      await AlgorithmAPI.add({
        name: name.trim(),
        type: type.trim(),
        version: version.trim(),
        description: addForm.description.trim() || undefined,
        path: addForm.path.trim() || undefined,
        importPath: addForm.importPath.trim() || undefined,
      } as Partial<Algorithm>);
      setAddVisible(false);
      resetAddForm();
      Taro.showToast({ title: "新增成功", icon: "success" });
      fetchAlgorithms();
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "新增失败"), icon: "none" });
    } finally {
      setAddSubmitting(false);
    }
  }, [addForm, fetchAlgorithms]);

  useLoad(() => {
    fetchAlgorithms();
  });

  usePullDownRefresh(() => {
    fetchAlgorithms().finally(() => Taro.stopPullDownRefresh());
  });

  const flatList = useMemo(() => {
    const filtered = filterTree(algorithms, searchKeyword, statusFilter);
    return flattenTree(filtered);
  }, [algorithms, searchKeyword, statusFilter]);

  const handleDetail = useCallback(async (algo: Algorithm) => {
    setDetailAlgo(algo);
    setDetailVisible(true);
    try {
      const detail = await AlgorithmAPI.getAlgorithmInfoById(algo.id);
      setDetailAlgo(detail);
    } catch {
      // 使用列表中的数据
    }
  }, []);

  const handleToggleStatus = useCallback(async (algo: Algorithm) => {
    // 已发布(4) → 停用(5)，已停用(5) → 启用恢复为已发布(4)
    const isPublished = algo.status === 4;
    const newStatus = isPublished ? 5 : 4;
    const actionText = isPublished ? "停用" : "启用";
    Taro.showModal({
      title: `确认${actionText}`,
      content: `确认${actionText}算法"${algo.name}"吗？`,
      success: async (res) => {
        if (!res.confirm) return;
        setActionLoadingId(algo.id);
        try {
          await AlgorithmAPI.updateStatus(algo.id, newStatus);
          setAlgorithms((prev) =>
            updateAlgorithmInTree(prev, algo.id, { status: newStatus })
          );
          setDetailAlgo((prev) =>
            prev?.id === algo.id ? { ...prev, status: newStatus } : prev
          );
          Taro.showToast({ title: `${actionText}成功`, icon: "success" });
        } catch (err: unknown) {
          Taro.showToast({
            title: getErrorMessage(err, `${actionText}失败`),
            icon: "none",
          });
        } finally {
          setActionLoadingId(null);
        }
      },
    });
  }, []);

  const handleOpenAudit = useCallback((algo: Algorithm, approved: boolean) => {
    setAuditAlgo(algo);
    setAuditApproved(approved);
    setAuditRemark("");
    setAuditVisible(true);
  }, []);

  const handleAuditSubmit = useCallback(async () => {
    if (!auditAlgo) return;
    if (!auditApproved && !auditRemark.trim()) {
      Taro.showToast({ title: "驳回需填写原因", icon: "none" });
      return;
    }
    setAuditSubmitting(true);
    try {
      const form: AlgorithmAuditForm = {
        approved: auditApproved,
        remark: auditRemark.trim() || undefined,
      };
      await AlgorithmAPI.auditAlgorithm(auditAlgo.id, form);
      // 审核通过：待审核(3)→已发布(4)；驳回：待审核(3)→测试中(2)
      const newStatus = auditApproved ? 4 : 2;
      setAlgorithms((prev) =>
        updateAlgorithmInTree(prev, auditAlgo.id, { status: newStatus })
      );
      setDetailAlgo((prev) =>
        prev?.id === auditAlgo.id ? { ...prev, status: newStatus } : prev
      );
      setAuditVisible(false);
      Taro.showToast({
        title: auditApproved ? "审核通过" : "已驳回",
        icon: "success",
      });
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "审核失败"), icon: "none" });
    } finally {
      setAuditSubmitting(false);
    }
  }, [auditAlgo, auditApproved, auditRemark]);

  const handleDelete = useCallback((algo: Algorithm) => {
    Taro.showModal({
      title: "确认删除",
      content: `确认删除算法"${algo.name}"吗？此操作不可恢复。`,
      confirmColor: "#ff4d4f",
      success: async (res) => {
        if (!res.confirm) return;
        setActionLoadingId(algo.id);
        try {
          await AlgorithmAPI.deleteByIds([String(algo.id)]);
          setAlgorithms((prev) => removeAlgorithmFromTree(prev, algo.id));
          setDetailAlgo((prev) => (prev?.id === algo.id ? null : prev));
          Taro.showToast({ title: "删除成功", icon: "success" });
        } catch (err: unknown) {
          Taro.showToast({
            title: getErrorMessage(err, "删除失败"),
            icon: "none",
          });
        } finally {
          setActionLoadingId(null);
        }
      },
    });
  }, []);

  const renderNode = (item: FlatNode) => {
    const { algorithm: algo, level, hasChildren } = item;
    const statusInfo = STATUS_INFO[algo.status ?? 0] || STATUS_INFO[1];
    const isPendingAudit = algo.status === 3; // 待审核
    const isPublished = algo.status === 4;    // 已发布
    const isDisabled = algo.status === 5;     // 已停用
    const isDeletableStatus =
      algo.status === 1 || algo.status === 5 || algo.status === 6;

    return (
      <View
        key={algo.id}
        className="algo-node"
        style={{ marginLeft: `${level * 24}px` }}
        onClick={() => handleDetail(algo)}
      >
        <View className="node-main">
          <View className="node-info">
            {hasChildren && <Text className="node-icon">📁</Text>}
            {!hasChildren && <Text className="node-icon">📄</Text>}
            <Text className="node-name">{algo.name}</Text>
          </View>
          <View className="node-meta">
            <Tag color={statusInfo.color} size="small">
              {statusInfo.label}
            </Tag>
            {algo.type && <Text className="node-type">{algo.type}</Text>}
            {algo.version && (
              <Text className="node-version">v{algo.version}</Text>
            )}
          </View>
        </View>

        {!hasChildren && (canAudit || canEdit || canDelete) && (
          <View className="node-actions" onClick={(e) => e.stopPropagation()}>
            {isPendingAudit && canAudit && (
              <>
                <Button
                  size="mini"
                  color="success"
                  onClick={() => handleOpenAudit(algo, true)}
                >
                  通过
                </Button>
                <Button
                  size="mini"
                  color="danger"
                  onClick={() => handleOpenAudit(algo, false)}
                >
                  驳回
                </Button>
              </>
            )}
            {(isPublished || isDisabled) && canEdit && (
              <Button
                size="mini"
                color={isPublished ? "warning" : "primary"}
                loading={actionLoadingId === algo.id}
                onClick={() => handleToggleStatus(algo)}
              >
                {isPublished ? "停用" : "启用"}
              </Button>
            )}
            {isDeletableStatus && canDelete && (
              <Button
                size="mini"
                color="danger"
                loading={actionLoadingId === algo.id}
                onClick={() => handleDelete(algo)}
              >
                删除
              </Button>
            )}
          </View>
        )}
      </View>
    );
  };

  return (
    <PageLayout level="L2" title="算法管理">
      <View className="algo-manage-page">
        {/* 搜索栏 + 新增按钮 */}
        <View className="search-bar">
          <Search className="search-icon" />
          <Input
            className="search-input"
            type="text"
            placeholder="搜索算法名称或类型"
            value={searchKeyword}
            onInput={(e) => setSearchKeyword(e.detail.value)}
          />
          {canAdd && (
            <Button
              size="mini"
              color="primary"
              className="add-btn"
              onClick={() => {
                resetAddForm();
                setAddVisible(true);
              }}
            >
              新增
            </Button>
          )}
        </View>

        {/* 状态筛选 */}
        <ScrollView
          scrollX
          className="filter-bar"
          enhanced
          showScrollbar={false}
        >
          {STATUS_FILTERS.map((filter) => (
            <View
              key={String(filter.value)}
              className={`filter-item ${statusFilter === filter.value ? "active" : ""}`}
              onClick={() => setStatusFilter(filter.value)}
            >
              <Text>{filter.label}</Text>
            </View>
          ))}
        </ScrollView>

        {/* 算法列表 */}
        <ScrollView scrollY className="algo-list">
          {loading ? (
            <View className="loading-wrapper">
              <Loading>加载中...</Loading>
            </View>
          ) : loadError ? (
            <ErrorState message={loadError} onRetry={fetchAlgorithms} />
          ) : flatList.length === 0 ? (
            <View className="empty-wrapper">
              <Text className="empty-text">暂无算法数据</Text>
            </View>
          ) : (
            flatList.map(renderNode)
          )}
        </ScrollView>

        <AlgorithmDetailPopup
          open={detailVisible}
          algorithm={detailAlgo}
          actionLoadingId={actionLoadingId}
          canAudit={canAudit}
          canEdit={canEdit}
          canDelete={canDelete}
          browseMode={false}
          onClose={() => setDetailVisible(false)}
          onToggleStatus={handleToggleStatus}
          onDelete={handleDelete}
          onOpenAudit={handleOpenAudit}
        />

        <AlgorithmAuditPopup
          open={auditVisible}
          algorithm={auditAlgo}
          approved={auditApproved}
          remark={auditRemark}
          submitting={auditSubmitting}
          onClose={() => setAuditVisible(false)}
          onRemarkChange={setAuditRemark}
          onSubmit={handleAuditSubmit}
        />

        {/* 新增算法弹窗 */}
        <Popup
          open={addVisible}
          placement="bottom"
          rounded
          onClose={() => setAddVisible(false)}
          style={{ maxHeight: "85vh", overflowY: "auto" }}
        >
          <View className="detail-content">
            <View className="detail-header">
              <Text className="detail-title">新增算法</Text>
              <Text className="detail-close" onClick={() => setAddVisible(false)}>关闭</Text>
            </View>
            <View className="detail-section">
              <Text className="section-title">基本信息</Text>
              <View className="form-item">
                <Text className="form-label">名称 *</Text>
                <TInput
                  placeholder="算法名称"
                  value={addForm.name}
                  onInput={(e) => setAddForm((f) => ({ ...f, name: e.detail.value }))}
                />
              </View>
              <View className="form-item">
                <Text className="form-label">类型 *</Text>
                <TInput
                  placeholder="算法类型"
                  value={addForm.type}
                  onInput={(e) => setAddForm((f) => ({ ...f, type: e.detail.value }))}
                />
              </View>
              <View className="form-item">
                <Text className="form-label">版本号 *</Text>
                <TInput
                  placeholder="v1.0.0"
                  value={addForm.version}
                  onInput={(e) => setAddForm((f) => ({ ...f, version: e.detail.value }))}
                />
              </View>
              <View className="form-item">
                <Text className="form-label">描述</Text>
                <Textarea
                  placeholder="算法描述"
                  value={addForm.description}
                  onInput={(e) => setAddForm((f) => ({ ...f, description: e.detail.value }))}
                  autoHeight
                  maxlength={500}
                />
              </View>
              <View className="form-item">
                <Text className="form-label">模型路径</Text>
                <TInput
                  placeholder="模型文件路径"
                  value={addForm.path}
                  onInput={(e) => setAddForm((f) => ({ ...f, path: e.detail.value }))}
                />
              </View>
              <View className="form-item">
                <Text className="form-label">导入路径</Text>
                <TInput
                  placeholder="模型导入路径"
                  value={addForm.importPath}
                  onInput={(e) => setAddForm((f) => ({ ...f, importPath: e.detail.value }))}
                />
              </View>
            </View>
            <View className="detail-footer">
              <Button block onClick={() => setAddVisible(false)}>取消</Button>
              <Button block color="primary" loading={addSubmitting} onClick={handleAdd}>提交</Button>
            </View>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default AlgorithmManagePage;
