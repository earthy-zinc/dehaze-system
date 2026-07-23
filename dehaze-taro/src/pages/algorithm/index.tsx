import React, { useState, useCallback, useMemo } from "react";
import { View, Text, Input, ScrollView } from "@tarojs/components";
import Taro, { useLoad, usePullDownRefresh } from "@tarojs/taro";
import { Navbar, Loading, Tag, Button } from "@taroify/core";
import { ArrowLeft, Search } from "@taroify/icons";
import { AlgorithmAPI } from "dehaze-sdk-js";
import type { Algorithm, AlgorithmAuditForm } from "dehaze-sdk-js";
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
} from "./utils";
import type { FlatNode } from "./utils";
import AlgorithmDetailPopup from "./components/AlgorithmDetailPopup";
import AlgorithmAuditPopup from "./components/AlgorithmAuditPopup";
import "./index.less";

// ==================== 页面组件 ====================

const AlgorithmManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canAudit = hasPermission("sys:algorithm:audit");
  const canEdit = hasPermission("sys:algorithm:edit");
  const canDelete = hasPermission("sys:algorithm:delete");

  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [searchKeyword, setSearchKeyword] = useState("");
  const [statusFilter, setStatusFilter] = useState<number | "">("");

  // 详情弹窗
  const [detailAlgo, setDetailAlgo] = useState<Algorithm | null>(null);
  const [detailVisible, setDetailVisible] = useState(false);

  // 审核弹窗
  const [auditAlgo, setAuditAlgo] = useState<Algorithm | null>(null);
  const [auditVisible, setAuditVisible] = useState(false);
  const [auditApproved, setAuditApproved] = useState(true);
  const [auditRemark, setAuditRemark] = useState("");
  const [auditSubmitting, setAuditSubmitting] = useState(false);

  // 操作加载状态
  const [actionLoadingId, setActionLoadingId] = useState<number | null>(null);

  // ==================== 数据加载 ====================

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

  useLoad(() => {
    fetchAlgorithms();
  });

  usePullDownRefresh(() => {
    fetchAlgorithms().finally(() => Taro.stopPullDownRefresh());
  });

  // ==================== 过滤后的平铺列表 ====================

  const flatList = useMemo(() => {
    const filtered = filterTree(algorithms, searchKeyword, statusFilter);
    return flattenTree(filtered);
  }, [algorithms, searchKeyword, statusFilter]);

  // ==================== 事件处理 ====================

  /** 查看详情 */
  const handleDetail = useCallback(async (algo: Algorithm) => {
    setDetailAlgo(algo);
    setDetailVisible(true);
    // 拉取最新详情
    try {
      const detail = await AlgorithmAPI.getAlgorithmInfoById(algo.id);
      setDetailAlgo(detail);
    } catch {
      // 使用列表中的数据
    }
  }, []);

  /** 修改状态（启用/停用） */
  const handleToggleStatus = useCallback(async (algo: Algorithm) => {
    const isPublished = algo.status === 3;
    const newStatus = isPublished ? 4 : 3;
    const actionText = isPublished ? "停用" : "启用";
    Taro.showModal({
      title: `确认${actionText}`,
      content: `确认${actionText}算法"${algo.name}"吗？`,
      success: async (res) => {
        if (!res.confirm) return;
        setActionLoadingId(algo.id);
        try {
          await AlgorithmAPI.updateStatus(algo.id, newStatus);
          // 本地更新
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

  /** 打开审核弹窗 */
  const handleOpenAudit = useCallback((algo: Algorithm, approved: boolean) => {
    setAuditAlgo(algo);
    setAuditApproved(approved);
    setAuditRemark("");
    setAuditVisible(true);
  }, []);

  /** 提交审核 */
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
      const newStatus = auditApproved ? 3 : 1;
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

  /** 删除算法 */
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
          Taro.showToast({ title: getErrorMessage(err, "删除失败"), icon: "none" });
        } finally {
          setActionLoadingId(null);
        }
      },
    });
  }, []);

  // ==================== 渲染 ====================

  /** 渲染算法节点 */
  const renderNode = (item: FlatNode) => {
    const { algorithm: algo, level, hasChildren } = item;
    const statusInfo = STATUS_INFO[algo.status ?? 0] || STATUS_INFO[0];
    const isPending = algo.status === 2;
    const isPublished = algo.status === 3;
    const isDisabled = algo.status === 4;
    const isDeletableStatus = algo.status === 0 || algo.status === 4;

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
            {isPending && canAudit && (
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
    <View className="algo-manage-page">
      <Navbar title="算法管理">
        <Navbar.NavLeft>
          <ArrowLeft />
        </Navbar.NavLeft>
      </Navbar>

      {/* 搜索栏 */}
      <View className="search-bar">
        <Search className="search-icon" />
        <Input
          className="search-input"
          type="text"
          placeholder="搜索算法名称或类型"
          value={searchKeyword}
          onInput={(e) => setSearchKeyword(e.detail.value)}
        />
      </View>

      {/* 状态筛选 */}
      <ScrollView scrollX className="filter-bar" enhanced showScrollbar={false}>
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

      {/* 算法详情弹窗 */}
      <AlgorithmDetailPopup
        open={detailVisible}
        algorithm={detailAlgo}
        actionLoadingId={actionLoadingId}
        canAudit={canAudit}
        canEdit={canEdit}
        canDelete={canDelete}
        onClose={() => setDetailVisible(false)}
        onToggleStatus={handleToggleStatus}
        onDelete={handleDelete}
        onOpenAudit={handleOpenAudit}
      />

      {/* 审核弹窗 */}
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
    </View>
  );
};

export default AlgorithmManagePage;
