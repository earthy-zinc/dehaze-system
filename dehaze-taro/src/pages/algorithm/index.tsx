import React, { useState, useCallback, useMemo } from "react";
import { View, Text, Input, ScrollView } from "@tarojs/components";
import Taro, { useLoad, usePullDownRefresh } from "@tarojs/taro";
import { Navbar, Loading, Tag, Button, Popup, Textarea } from "@taroify/core";
import { ArrowLeft, Search } from "@taroify/icons";
import { AlgorithmAPI } from "dehaze-sdk-js";
import type { Algorithm, AlgorithmAuditForm } from "dehaze-sdk-js";
import ErrorState from "@/components/common/ErrorState";
import "./index.less";

// ==================== 状态定义 ====================

/** 状态信息映射 */
const STATUS_INFO: Record<
  number,
  {
    label: string;
    color: "default" | "primary" | "success" | "warning" | "danger";
  }
> = {
  0: { label: "草稿", color: "default" },
  1: { label: "测试中", color: "warning" },
  2: { label: "待审核", color: "primary" },
  3: { label: "已发布", color: "success" },
  4: { label: "已停用", color: "default" },
  5: { label: "已归档", color: "default" },
};

/** 状态筛选选项 */
const STATUS_FILTERS: { label: string; value: number | "" }[] = [
  { label: "全部", value: "" },
  { label: "草稿", value: 0 },
  { label: "测试中", value: 1 },
  { label: "待审核", value: 2 },
  { label: "已发布", value: 3 },
  { label: "已停用", value: 4 },
];

// ==================== 工具函数 ====================

/** 递归展开算法树为平铺列表（含层级缩进信息） */
interface FlatNode {
  algorithm: Algorithm;
  level: number;
  hasChildren: boolean;
}

function flattenTree(nodes: Algorithm[], level = 0): FlatNode[] {
  const result: FlatNode[] = [];
  for (const node of nodes) {
    const hasChildren = !!(node.children && node.children.length > 0);
    result.push({ algorithm: node, level, hasChildren });
    if (hasChildren) {
      result.push(...flattenTree(node.children!, level + 1));
    }
  }
  return result;
}

/** 递归过滤算法树（按关键词和状态） */
function filterTree(
  nodes: Algorithm[],
  keyword: string,
  statusFilter: number | ""
): Algorithm[] {
  const lowerKeyword = keyword.toLowerCase();
  const match = (algo: Algorithm): boolean => {
    const nameMatch =
      !keyword || (algo.name || "").toLowerCase().includes(lowerKeyword);
    const typeMatch =
      !keyword || (algo.type || "").toLowerCase().includes(lowerKeyword);
    const statusMatch = statusFilter === "" || algo.status === statusFilter;
    return (nameMatch || typeMatch) && statusMatch;
  };

  const walk = (list: Algorithm[]): Algorithm[] => {
    const result: Algorithm[] = [];
    for (const node of list) {
      const children = node.children ? walk(node.children) : [];
      if (match(node) || children.length > 0) {
        result.push({
          ...node,
          children: children.length > 0 ? children : undefined,
        });
      }
    }
    return result;
  };
  return walk(nodes);
}

// ==================== 页面组件 ====================

const AlgorithmManagePage: React.FC = () => {
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
    } catch (err: any) {
      setLoadError(err?.message || "加载失败，请重试");
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
        } catch (err: any) {
          Taro.showToast({
            title: err?.message || `${actionText}失败`,
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
    } catch (err: any) {
      Taro.showToast({ title: err?.message || "审核失败", icon: "none" });
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
        } catch (err: any) {
          Taro.showToast({ title: err?.message || "删除失败", icon: "none" });
        } finally {
          setActionLoadingId(null);
        }
      },
    });
  }, []);

  // ==================== 树操作工具函数 ====================

  function updateAlgorithmInTree(
    nodes: Algorithm[],
    id: number,
    patch: Partial<Algorithm>
  ): Algorithm[] {
    return nodes.map((node) => {
      if (node.id === id) return { ...node, ...patch };
      if (node.children)
        return {
          ...node,
          children: updateAlgorithmInTree(node.children, id, patch),
        };
      return node;
    });
  }

  function removeAlgorithmFromTree(
    nodes: Algorithm[],
    id: number
  ): Algorithm[] {
    return nodes
      .filter((node) => node.id !== id)
      .map((node) =>
        node.children
          ? { ...node, children: removeAlgorithmFromTree(node.children, id) }
          : node
      );
  }

  // ==================== 渲染 ====================

  /** 渲染算法节点 */
  const renderNode = (item: FlatNode) => {
    const { algorithm: algo, level, hasChildren } = item;
    const statusInfo = STATUS_INFO[algo.status ?? 0] || STATUS_INFO[0];
    const isPending = algo.status === 2;
    const isPublished = algo.status === 3;
    const isDisabled = algo.status === 4;
    const canDelete = algo.status === 0 || algo.status === 4;

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

        {!hasChildren && (
          <View className="node-actions" onClick={(e) => e.stopPropagation()}>
            {isPending && (
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
            {(isPublished || isDisabled) && (
              <Button
                size="mini"
                color={isPublished ? "warning" : "primary"}
                loading={actionLoadingId === algo.id}
                onClick={() => handleToggleStatus(algo)}
              >
                {isPublished ? "停用" : "启用"}
              </Button>
            )}
            {canDelete && (
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

  /** 渲染详情项 */
  const renderDetailItem = (label: string, value: React.ReactNode) => (
    <View className="detail-item">
      <Text className="detail-label">{label}</Text>
      <View className="detail-value">{value || "-"}</View>
    </View>
  );

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
      <Popup
        open={detailVisible}
        placement="bottom"
        rounded
        onClose={() => setDetailVisible(false)}
        className="detail-popup"
      >
        {detailAlgo && (
          <View className="detail-content">
            <View className="detail-header">
              <Text className="detail-title">{detailAlgo.name}</Text>
              <Text
                className="detail-close"
                onClick={() => setDetailVisible(false)}
              >
                关闭
              </Text>
            </View>

            <View className="detail-section">
              <Text className="section-title">基本信息</Text>
              {renderDetailItem("算法名称", detailAlgo.name)}
              {renderDetailItem("算法类型", detailAlgo.type)}
              {renderDetailItem("描述", detailAlgo.description)}
              {renderDetailItem(
                "状态",
                <Tag
                  color={
                    STATUS_INFO[detailAlgo.status ?? 0]?.color || "default"
                  }
                  size="small"
                >
                  {STATUS_INFO[detailAlgo.status ?? 0]?.label || "未知"}
                </Tag>
              )}
              {renderDetailItem("版本", detailAlgo.version)}
              {renderDetailItem("大小", detailAlgo.size)}
            </View>

            <View className="detail-section">
              <Text className="section-title">技术信息</Text>
              {renderDetailItem("路径", detailAlgo.path)}
              {renderDetailItem("导入路径", detailAlgo.importPath)}
              {renderDetailItem("参数", detailAlgo.params)}
              {renderDetailItem("计算量(FLOPs)", detailAlgo.flops)}
            </View>

            {(detailAlgo.status === 2 || detailAlgo.auditBy != null) && (
              <View className="detail-section">
                <Text className="section-title">审核信息</Text>
                {renderDetailItem("审核人", detailAlgo.auditBy)}
                {renderDetailItem("审核时间", detailAlgo.auditTime)}
                {renderDetailItem("审核备注", detailAlgo.auditRemark)}
              </View>
            )}

            {renderDetailItem("创建时间", detailAlgo.createTime)}

            {/* 操作按钮 */}
            <View className="detail-footer">
              {detailAlgo.status === 2 && (
                <>
                  <Button
                    block
                    color="success"
                    onClick={() => {
                      setDetailVisible(false);
                      handleOpenAudit(detailAlgo, true);
                    }}
                  >
                    审核通过
                  </Button>
                  <Button
                    block
                    color="danger"
                    onClick={() => {
                      setDetailVisible(false);
                      handleOpenAudit(detailAlgo, false);
                    }}
                  >
                    审核驳回
                  </Button>
                </>
              )}
              {detailAlgo.status === 3 && (
                <Button
                  block
                  color="warning"
                  loading={actionLoadingId === detailAlgo.id}
                  onClick={() => handleToggleStatus(detailAlgo)}
                >
                  停用算法
                </Button>
              )}
              {detailAlgo.status === 4 && (
                <>
                  <Button
                    block
                    color="primary"
                    loading={actionLoadingId === detailAlgo.id}
                    onClick={() => handleToggleStatus(detailAlgo)}
                  >
                    启用算法
                  </Button>
                  <Button
                    block
                    color="danger"
                    loading={actionLoadingId === detailAlgo.id}
                    onClick={() => {
                      setDetailVisible(false);
                      handleDelete(detailAlgo);
                    }}
                  >
                    删除算法
                  </Button>
                </>
              )}
              {detailAlgo.status === 0 && (
                <Button
                  block
                  color="danger"
                  loading={actionLoadingId === detailAlgo.id}
                  onClick={() => {
                    setDetailVisible(false);
                    handleDelete(detailAlgo);
                  }}
                >
                  删除算法
                </Button>
              )}
            </View>
          </View>
        )}
      </Popup>

      {/* 审核弹窗 */}
      <Popup
        open={auditVisible}
        placement="center"
        rounded
        onClose={() => setAuditVisible(false)}
        className="audit-popup"
      >
        <View className="audit-content">
          <Text className="audit-title">
            {auditApproved ? "审核通过" : "审核驳回"}
          </Text>
          {auditAlgo && (
            <Text className="audit-name">算法：{auditAlgo.name}</Text>
          )}
          {!auditApproved && (
            <View className="audit-remark">
              <Text className="remark-label">驳回原因（必填）</Text>
              <Textarea
                className="remark-input"
                placeholder="请输入驳回原因"
                value={auditRemark}
                onInput={(e) => setAuditRemark(e.detail.value)}
                maxlength={200}
              />
            </View>
          )}
          <View className="audit-footer">
            <Button block onClick={() => setAuditVisible(false)}>
              取消
            </Button>
            <Button
              block
              color={auditApproved ? "success" : "danger"}
              loading={auditSubmitting}
              onClick={handleAuditSubmit}
            >
              确认
            </Button>
          </View>
        </View>
      </Popup>
    </View>
  );
};

export default AlgorithmManagePage;
