import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { FileAPI } from "dehaze-sdk-js";
import type { FileInfo } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import "./index.less";

const FilesPage: React.FC = () => {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [pageNum, setPageNum] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  const loadData = useCallback(
    async (page = 1) => {
      if (loading) return;
      setLoading(true);
      try {
        const result = await FileAPI.getPage({ pageNum: page, pageSize: 20 });
        const list = result.list || [];
        if (page === 1) {
          setFiles(list);
        } else {
          setFiles((prev) => [...prev, ...list]);
        }
        setHasMore(list.length < (result.total || 0));
        setPageNum(page);
      } catch {
        Taro.showToast({ title: "加载失败", icon: "none" });
      } finally {
        setLoading(false);
      }
    },
    [loading]
  );

  useEffect(() => {
    loadData(1);
  }, []);

  const handleClick = (file: FileInfo) => {
    if (file.url) {
      Taro.setClipboardData({
        data: file.url,
        success: () => Taro.showToast({ title: "URL 已复制", icon: "success" }),
      });
    }
  };

  const getIcon = (type?: string): string => {
    if (!type) return "📄";
    const t = type.toLowerCase();
    if (t.includes("image")) return "🖼️";
    if (t.includes("video")) return "🎬";
    if (t.includes("audio")) return "🎵";
    if (t.includes("pdf") || t.includes("doc")) return "📝";
    return "📄";
  };

  const formatFileSize = (size?: string): string => {
    return size || "-";
  };

  const formatTime = (time?: string): string => {
    if (!time) return "";
    try {
      const d = new Date(time);
      const now = new Date();
      const diff = now.getTime() - d.getTime();
      if (diff < 60000) return "刚刚";
      if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`;
      if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`;
      if (diff < 604800000) return `${Math.floor(diff / 86400000)} 天前`;
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    } catch {
      return time;
    }
  };

  return (
    <PageLayout level="L2" title="我的文件">
      <View className="personal-files-page">
        {loading && files.length === 0 ? (
          <View className="loading-wrapper">
            <Text className="loading-text">加载中...</Text>
          </View>
        ) : files.length > 0 ? (
          <ScrollView scrollY className="files-scroll" enhanced showScrollbar={false}>
            {files.map((file) => (
              <View key={file.id} className="file-card" onClick={() => handleClick(file)}>
                <View className="file-icon">
                  <Text>{getIcon(file.type)}</Text>
                </View>
                <View className="file-info">
                  <Text className="file-name">{file.name}</Text>
                  <Text className="file-meta">
                    {formatFileSize(file.size)} · {formatTime(file.createTime)}
                  </Text>
                </View>
                <View className="file-arrow">
                  <Text>›</Text>
                </View>
              </View>
            ))}
            {!hasMore && files.length > 0 && (
              <View className="no-more">
                <Text>没有更多了</Text>
              </View>
            )}
            {hasMore && (
              <View className="load-more" onClick={() => loadData(pageNum + 1)}>
                <Text>加载更多</Text>
              </View>
            )}
          </ScrollView>
        ) : (
          <View className="empty-wrapper">
            <Text className="empty-icon">📄</Text>
            <Text className="empty-title">暂无文件</Text>
            <Text className="empty-desc">
              您上传和处理的图像文件将显示在这里
            </Text>
          </View>
        )}
      </View>
    </PageLayout>
  );
};

export default FilesPage;
