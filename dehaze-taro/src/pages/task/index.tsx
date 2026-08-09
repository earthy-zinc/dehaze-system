/**
 * 处理历史页面
 *
 * 展示去雾处理历史记录，支持查看对比和重新处理。
 * 对接 ModelAPI.getPredLogs（分页查询处理日志）
 */
import React, { useState, useCallback } from "react";
import { View, Text, Image, ScrollView } from "@tarojs/components";
import Taro, {
  useLoad,
  usePullDownRefresh,
  useReachBottom,
} from "@tarojs/taro";
import { Navbar, Loading, Empty } from "@taroify/core";
import { ArrowLeft } from "@taroify/icons";
import { ModelAPI, AlgorithmAPI } from "dehaze-sdk-js";
import type { PredLogVO } from "dehaze-sdk-js";
import ErrorState from "@/components/common/ErrorState";
import { getErrorMessage } from "@/utils/error";
import { formatDateTime } from "@/utils/format";
import { useProcessStore } from "@/stores/process";
import "./index.less";

const PAGE_SIZE = 15;

const TaskPage: React.FC = () => {
  const [records, setRecords] = useState<PredLogVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pageNum, setPageNum] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  /** 加载处理历史 */
  const loadData = useCallback(
    async (page: number) => {
      if (loading) return;
      setLoading(true);
      setLoadError(null);
      try {
        const res = await ModelAPI.getPredLogs({
          pageNum: page,
          pageSize: PAGE_SIZE,
        });
        const list = res.list || [];
        if (page === 1) {
          setRecords(list);
        } else {
          setRecords((prev) => [...prev, ...list]);
        }
        setPageNum(page);
        setHasMore(list.length >= PAGE_SIZE);
      } catch (err: unknown) {
        setLoadError(getErrorMessage(err, "加载失败"));
      } finally {
        setLoading(false);
      }
    },
    [loading]
  );

  useLoad(() => {
    loadData(1);
  });

  usePullDownRefresh(() => {
    loadData(1).finally(() => Taro.stopPullDownRefresh());
  });

  useReachBottom(() => {
    if (hasMore && !loading) {
      loadData(pageNum + 1);
    }
  });

  /** 查看对比：跳转对比页面 */
  const handleCompare = useCallback((record: PredLogVO) => {
    if (!record.predUrl || !record.originUrl) {
      Taro.showToast({ title: "缺少原图或结果图", icon: "none" });
      return;
    }
    useProcessStore.getState().setResult({
      status: 2,
      resultUrl: record.predUrl,
      time: record.time || 0,
    });
    Taro.navigateTo({ url: "/pages/side-by-side/index" });
  }, []);

  /** 重新处理：加载算法详情后跳转处理页 */
  const handleReprocess = useCallback(async (record: PredLogVO) => {
    if (!record.originUrl) {
      Taro.showToast({ title: "缺少原图", icon: "none" });
      return;
    }
    if (!record.algorithmId) {
      Taro.showToast({ title: "缺少算法信息", icon: "none" });
      return;
    }
    Taro.showLoading({ title: "准备中..." });
    try {
      const algorithm = await AlgorithmAPI.getAlgorithmInfoById(record.algorithmId);
      Taro.hideLoading();
      useProcessStore.getState().setImage({
        url: record.originUrl,
        name: record.originUrl.split("/").pop() || "历史图片",
      });
      useProcessStore.getState().setAlgorithm(algorithm);
      Taro.navigateTo({ url: "/pages/processing/index" });
    } catch (e) {
      Taro.hideLoading();
      Taro.showToast({ title: getErrorMessage(e, "加载失败"), icon: "none" });
    }
  }, []);

  /** 预览结果图 */
  const handlePreview = useCallback((url: string) => {
    Taro.previewImage({ urls: [url], current: url });
  }, []);

  return (
    <View className="task-page">
      <Navbar title="处理历史">
        <Navbar.NavLeft>
          <ArrowLeft />
        </Navbar.NavLeft>
      </Navbar>

      <ScrollView scrollY className="task-content">
        {loading && records.length === 0 ? (
          <View className="loading-wrapper">
            <Loading>加载中...</Loading>
          </View>
        ) : loadError && records.length === 0 ? (
          <ErrorState message={loadError} onRetry={() => loadData(1)} />
        ) : records.length === 0 ? (
          <Empty>
            <Empty.Description>暂无处理记录</Empty.Description>
          </Empty>
        ) : (
          <>
            {records.map((record) => (
              <View key={record.id} className="record-card">
                {record.predUrl && (
                  <View
                    className="record-thumb"
                    onClick={() => handlePreview(record.predUrl!)}
                  >
                    <Image
                      src={record.predUrl}
                      className="thumb-img"
                      mode="aspectFill"
                    />
                  </View>
                )}
                <View className="record-body">
                  <Text className="record-algo ellipsis">
                    {record.algorithmName || "未知算法"}
                  </Text>
                  <Text className="record-time">
                    耗时 {record.time != null ? `${record.time}s` : "-"}
                  </Text>
                  <Text className="record-date">
                    {formatDateTime(record.createTime)}
                  </Text>
                  <View className="record-actions">
                    <View
                      className={`action-btn compare-btn ${!record.predUrl || !record.originUrl ? "disabled" : ""}`}
                      onClick={() => handleCompare(record)}
                    >
                      <Text>对比</Text>
                    </View>
                    <View
                      className={`action-btn reprocess-btn ${!record.originUrl ? "disabled" : ""}`}
                      onClick={() => handleReprocess(record)}
                    >
                      <Text>重新处理</Text>
                    </View>
                  </View>
                </View>
              </View>
            ))}
            {hasMore ? (
              <View className="load-more" onClick={() => loadData(pageNum + 1)}>
                <Text>加载更多</Text>
              </View>
            ) : (
              <View className="no-more">
                <Text>— 没有更多了 —</Text>
              </View>
            )}
          </>
        )}
      </ScrollView>
    </View>
  );
};

export default TaskPage;
