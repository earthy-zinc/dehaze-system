/**
 * 算法库浏览版（L2）
 * 个人浏览视角：算法列表、智能推荐、详情查看、使用该算法带入去雾流程
 * 无审计/上下架/删除等管理操作
 */
import React, { useState, useCallback, useMemo } from "react";
import { View, Text, Input, ScrollView } from "@tarojs/components";
import Taro, { useLoad, usePullDownRefresh } from "@tarojs/taro";
import { Navbar, Loading, Tag, Button } from "@taroify/core";
import { ArrowLeft, Search } from "@taroify/icons";
import { AlgorithmAPI, RecommendationAPI } from "dehaze-sdk-js";
import type { Algorithm } from "dehaze-sdk-js";
import ErrorState from "@/components/common/ErrorState";
import { getErrorMessage } from "@/utils/error";
import { useProcessStore } from "@/stores/process";
import { STATUS_INFO, flattenTree, filterTree } from "./utils";
import type { FlatNode } from "./utils";
import AlgorithmDetailPopup from "./components/AlgorithmDetailPopup";
import "./index.less";

const AlgorithmBrowsePage: React.FC = () => {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [searchKeyword, setSearchKeyword] = useState("");
  const [statusFilter, setStatusFilter] = useState<number | "">(4); // 默认显示已发布

  // 推荐
  const [recommendLoading, setRecommendLoading] = useState(false);
  const [recommendedIds, setRecommendedIds] = useState<Set<number>>(new Set());

  // 详情弹窗
  const [detailAlgo, setDetailAlgo] = useState<Algorithm | null>(null);
  const [detailVisible, setDetailVisible] = useState(false);

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

  // ==================== 智能推荐 ====================

  const fetchRecommendations = useCallback(async () => {
    setRecommendLoading(true);
    try {
      const recs = await RecommendationAPI.getAlgorithmRecommendations({});
      const ids = new Set<number>();
      (recs || []).forEach((r) => ids.add(r.algorithmId));
      setRecommendedIds(ids);
    } catch {
      // 推荐加载失败不影响主列表
    } finally {
      setRecommendLoading(false);
    }
  }, []);

  useLoad(() => {
    fetchRecommendations();
  });

  // ==================== 过滤后的平铺列表 ====================

  const flatList = useMemo(() => {
    const filtered = filterTree(algorithms, searchKeyword, statusFilter);
    return flattenTree(filtered);
  }, [algorithms, searchKeyword, statusFilter]);

  // ==================== 事件处理 ====================

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

  const handleUseAlgorithm = useCallback((algo: Algorithm) => {
    // 将算法带入去雾流程
    useProcessStore.getState().setAlgorithm(algo);
    // 跳转到 processing 页面
    Taro.navigateTo({ url: "/pages/processing/index" });
  }, []);

  // ==================== 渲染 ====================

  const renderNode = (item: FlatNode) => {
    const { algorithm: algo, level, hasChildren } = item;
    const statusInfo = STATUS_INFO[algo.status ?? 0] || STATUS_INFO[1];
    const isRecommended = recommendedIds.has(algo.id);

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
            {isRecommended && (
              <Tag color="warning" size="small">
                推荐
              </Tag>
            )}
            <Tag color={statusInfo.color} size="small">
              {statusInfo.label}
            </Tag>
            {algo.type && <Text className="node-type">{algo.type}</Text>}
          </View>
        </View>

        {!hasChildren && (
          <View className="node-actions" onClick={(e) => e.stopPropagation()}>
            <Button
              size="mini"
              variant="contained"
              color="primary"
              onClick={() => handleUseAlgorithm(algo)}
            >
              使用该算法
            </Button>
          </View>
        )}
      </View>
    );
  };

  const statusFilters = [
    { label: "全部", value: "" },
    { label: "已发布", value: 4 },
  ];

  return (
    <View className="algo-browse-page">
      <Navbar title="算法库">
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
        {statusFilters.map((filter) => (
          <View
            key={String(filter.value)}
            className={`filter-item ${statusFilter === filter.value ? "active" : ""}`}
            onClick={() => setStatusFilter(filter.value as number | "")}
          >
            <Text>{filter.label}</Text>
          </View>
        ))}
        {recommendLoading && (
          <View className="filter-item">
            <Text className="text-muted">加载推荐中...</Text>
          </View>
        )}
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
        actionLoadingId={null}
        canAudit={false}
        canEdit={false}
        canDelete={false}
        onClose={() => setDetailVisible(false)}
        onToggleStatus={() => {}}
        onDelete={() => {}}
        onOpenAudit={() => {}}
      />
    </View>
  );
};

export default AlgorithmBrowsePage;
