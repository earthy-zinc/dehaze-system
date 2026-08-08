/**
 * 指标管理页面（L2）
 * 评估指标历史列表 + 筛选 + 多选对比
 * 与 L3 compare/metrics 的区别：这里是管理视角（列表+筛选+对比表格）
 */
import React, { useState, useCallback, useMemo } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro, { useLoad } from "@tarojs/taro";
import { Navbar, Button, Loading, Tag } from "@taroify/core";
import { ArrowLeft } from "@taroify/icons";
import { ModelAPI } from "dehaze-sdk-js";
import type { EvalMetricsVO, PredEvalTaskStatus } from "dehaze-sdk-js";
import ErrorState from "@/components/common/ErrorState";
import EmptyState from "@/components/common/EmptyState";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

const METRIC_LABELS: Record<string, string> = {
  psnr: "PSNR",
  ssim: "SSIM",
  lpips: "LPIPS",
  niqe: "NIQE",
  entropy: "信息熵",
  mse: "MSE",
};

const MetricsManagePage: React.FC = () => {
  const [records, setRecords] = useState<EvalMetricsVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [compareMode, setCompareMode] = useState(false);

  const fetchRecords = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await ModelAPI.getEvalMetrics({ pageNum: 1, pageSize: 50 });
      setRecords((res.list || []) as EvalMetricsVO[]);
    } catch (err: unknown) {
      setError(getErrorMessage(err, "加载评估记录失败"));
    } finally {
      setLoading(false);
    }
  }, []);

  useLoad(() => {
    fetchRecords();
  });

  // 切换选中
  const toggleSelect = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        if (next.size >= 3) {
          Taro.showToast({ title: "最多选择3条记录对比", icon: "none" });
          return prev;
        }
        next.add(id);
      }
      return next;
    });
  }, []);

  // 开启对比
  const startCompare = useCallback(() => {
    if (selectedIds.size < 2) {
      Taro.showToast({ title: "请至少选择2条记录", icon: "none" });
      return;
    }
    setCompareMode(true);
  }, [selectedIds.size]);

  // 退出对比
  const exitCompare = useCallback(() => {
    setCompareMode(false);
    setSelectedIds(new Set());
  }, []);

  // 筛选出的对比记录
  const compareRecords = useMemo(() => {
    if (!compareMode) return [];
    return records.filter((r) => selectedIds.has(r.id));
  }, [compareMode, records, selectedIds]);

  // 收集所有指标 key
  const metricKeys = useMemo(() => {
    const keys = new Set<string>();
    compareRecords.forEach((r) => {
      if (r.metrics) {
        Object.keys(r.metrics).forEach((k) => keys.add(k));
      }
    });
    return Array.from(keys);
  }, [compareRecords]);

  const statusLabel = (status?: PredEvalTaskStatus): string => {
    if (status === 2) return "已完成";
    if (status === 3) return "失败";
    if (status === 1) return "处理中";
    return "未知";
  };

  const formatValue = (key: string, value: number): string => {
    if (key === "psnr") return value.toFixed(2) + " dB";
    if (key === "ssim" || key === "lpips") return value.toFixed(4);
    return value.toFixed(2);
  };

  return (
    <View className="metrics-manage-page">
      <Navbar title="指标管理">
        <Navbar.NavLeft>
          <ArrowLeft />
        </Navbar.NavLeft>
      </Navbar>

      <ScrollView className="mm-content" scrollY>
        {/* 操作区 */}
        <View className="mm-toolbar">
          {!compareMode ? (
            <View className="toolbar-left">
              <Text className="toolbar-hint">
                选择记录进行指标对比（最多3条）
              </Text>
              {selectedIds.size >= 2 && (
                <Button size="mini" color="primary" onClick={startCompare}>
                  对比 ({selectedIds.size})
                </Button>
              )}
            </View>
          ) : (
            <View className="toolbar-left">
              <Text className="toolbar-hint">
                对比 {compareRecords.length} 条记录
              </Text>
              <Button size="mini" onClick={exitCompare}>
                退出对比
              </Button>
            </View>
          )}
        </View>

        {/* 对比表格 */}
        {compareMode && compareRecords.length > 0 && (
          <View className="compare-table-section">
            <ScrollView scrollX className="compare-table-wrapper">
              <View className="compare-table">
                <View className="table-row header">
                  <View className="table-cell metric-label-cell">
                    <Text>指标</Text>
                  </View>
                  {compareRecords.map((r) => (
                    <View key={r.id} className="table-cell algo-cell">
                      <Text className="algo-name">
                        {r.algorithmName || `算法${r.algorithmId}`}
                      </Text>
                    </View>
                  ))}
                </View>
                {metricKeys.map((key) => (
                  <View key={key} className="table-row">
                    <View className="table-cell metric-label-cell">
                      <Text>{METRIC_LABELS[key] || key}</Text>
                    </View>
                    {compareRecords.map((r) => (
                      <View key={r.id} className="table-cell value-cell">
                        <Text>
                          {r.metrics?.[key] != null
                            ? formatValue(key, r.metrics[key])
                            : "-"}
                        </Text>
                      </View>
                    ))}
                  </View>
                ))}
              </View>
            </ScrollView>
          </View>
        )}

        {/* 记录列表 */}
        <View className="mm-list">
          {loading ? (
            <View className="loading-wrapper">
              <Loading>加载中...</Loading>
            </View>
          ) : error ? (
            <ErrorState message={error} onRetry={fetchRecords} />
          ) : records.length === 0 ? (
            <EmptyState
              type="search"
              title="暂无评估记录"
              description="完成去雾处理后可在对比页生成评估指标"
            />
          ) : (
            records.map((record) => {
              const isSelected = selectedIds.has(record.id);
              return (
                <View
                  key={record.id}
                  className={`mm-record-card ${isSelected ? "selected" : ""}`}
                  onClick={() => !compareMode && toggleSelect(record.id)}
                >
                  <View className="card-header">
                    <Text className="card-algo">
                      {record.algorithmName || `算法${record.algorithmId}`}
                    </Text>
                    <View className="card-meta">
                      <Tag
                        color={
                          record.status === 2
                            ? "success"
                            : record.status === 3
                              ? "danger"
                              : "primary"
                        }
                        size="small"
                      >
                        {statusLabel(record.status)}
                      </Tag>
                      {!compareMode && isSelected && (
                        <View className="select-mark">✓</View>
                      )}
                    </View>
                  </View>

                  {record.metrics && Object.keys(record.metrics).length > 0 && (
                    <View className="card-metrics">
                      {Object.entries(record.metrics).map(([key, value]) => (
                        <View key={key} className="mini-metric">
                          <Text className="mini-label">
                            {METRIC_LABELS[key] || key}
                          </Text>
                          <Text className="mini-value">
                            {formatValue(key, value)}
                          </Text>
                        </View>
                      ))}
                    </View>
                  )}

                  <View className="card-footer">
                    {record.time != null && (
                      <Text className="card-time">
                        耗时 {(record.time / 1000).toFixed(1)}s
                      </Text>
                    )}
                    {record.createTime && (
                      <Text className="card-date">{record.createTime}</Text>
                    )}
                  </View>
                </View>
              );
            })
          )}
        </View>
      </ScrollView>
    </View>
  );
};

export default MetricsManagePage;
