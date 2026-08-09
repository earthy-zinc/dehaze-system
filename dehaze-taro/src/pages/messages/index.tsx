/**
 * 消息 Tab 根页面
 *
 * 消息列表（分页/分类 Tab/未读红点）+ 搜索 + 删除 + 设置入口
 */
import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView, Input } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { MessageAPI } from "dehaze-sdk-js";
import type { MessageVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import EmptyState from "@/components/common/EmptyState";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

const TABS = [
  { key: "", label: "全部" },
  { key: "announcement", label: "系统公告" },
  { key: "business", label: "业务通知" },
  { key: "member", label: "会员通知" },
  { key: "alert", label: "告警" },
];

function formatTime(dateStr: string): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - d.getTime()) / 86400000);
  const pad = (n: number) => String(n).padStart(2, "0");
  const hhmm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  if (diffDays === 0) return hhmm;
  if (diffDays === 1) return "昨天";
  if (diffDays === 2) return "前天";
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const MessagesPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState("");
  const [messages, setMessages] = useState<MessageVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [unreadCount, setUnreadCount] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [showSearch, setShowSearch] = useState(false);
  const [keyword, setKeyword] = useState("");
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [deleteMode, setDeleteMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

  const fetchUnreadCount = useCallback(async () => {
    try {
      const res = await MessageAPI.getUnreadCount();
      setUnreadCount(res.count || 0);
      if (res.count > 0) {
        Taro.setTabBarBadge({
          index: 3,
          text: String(res.count > 99 ? "99+" : res.count),
        });
      } else {
        Taro.removeTabBarBadge({ index: 3 });
      }
    } catch {
      // 不阻塞主流程
    }
  }, []);

  const fetchMessages = useCallback(
    async (page: number, type: string, append: boolean = false) => {
      try {
        setLoading(true);
        const queryParams: Record<string, unknown> = {
          pageNum: page,
          pageSize: 20,
        };
        if (type) queryParams.type = type;

        const res = await MessageAPI.getPage(queryParams);
        const list = res.list || [];
        if (append) {
          setMessages((prev) => [...prev, ...list]);
        } else {
          setMessages(list);
        }
        setHasMore(list.length >= 20);
      } catch (error) {
        Taro.showToast({
          title: getErrorMessage(error, "加载消息失败"),
          icon: "none",
        });
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const doSearch = useCallback(
    async (page: number, kw: string, append = false) => {
      if (!kw.trim()) return;
      try {
        setLoading(true);
        const res = await MessageAPI.search({ keyword: kw.trim(), pageNum: page, pageSize: 20 });
        const list = res.list || [];
        if (append) {
          setMessages((prev) => [...prev, ...list]);
        } else {
          setMessages(list);
        }
        setHasMore(list.length >= 20);
        // 保存搜索历史（去重，最多5条）
        setSearchHistory((prev) => {
          const next = [kw.trim(), ...prev.filter((h) => h !== kw.trim())].slice(0, 5);
          return next;
        });
      } catch (error) {
        Taro.showToast({
          title: getErrorMessage(error, "搜索失败"),
          icon: "none",
        });
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    fetchUnreadCount();
  }, [fetchUnreadCount]);

  useEffect(() => {
    setPageNum(1);
    if (showSearch && keyword.trim()) {
      doSearch(1, keyword, false);
    } else {
      fetchMessages(1, activeTab, false);
    }
  }, [activeTab, showSearch]);

  const handleTabChange = useCallback((key: string) => {
    setActiveTab(key);
    setMessages([]);
    setPageNum(1);
    setHasMore(true);
    setShowSearch(false);
    setKeyword("");
    setDeleteMode(false);
    setSelectedIds(new Set());
  }, []);

  const handleMessageClick = useCallback(async (message: MessageVO) => {
    if (deleteMode) {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        if (next.has(message.id)) next.delete(message.id);
        else next.add(message.id);
        return next;
      });
      return;
    }
    if (message.readStatus === 0) {
      try {
        await MessageAPI.markRead(message.id);
        setMessages((prev) =>
          prev.map((m) => (m.id === message.id ? { ...m, readStatus: 1 } : m))
        );
        setUnreadCount((prev) => Math.max(0, prev - 1));
      } catch {
        // 标记失败也继续跳转
      }
    }
    Taro.navigateTo({ url: `/pages/messages/detail/index?id=${message.id}` });
  }, [deleteMode]);

  const handleMarkAllRead = useCallback(async () => {
    try {
      await MessageAPI.markAllRead(activeTab || undefined);
      setMessages((prev) => prev.map((m) => ({ ...m, readStatus: 1 })));
      setUnreadCount(0);
      Taro.removeTabBarBadge({ index: 3 });
      Taro.showToast({ title: "已全部标记为已读", icon: "success" });
    } catch (error) {
      Taro.showToast({
        title: getErrorMessage(error, "操作失败"),
        icon: "none",
      });
    }
  }, [activeTab]);

  const handleLoadMore = useCallback(() => {
    if (loading || !hasMore) return;
    const nextPage = pageNum + 1;
    setPageNum(nextPage);
    if (showSearch && keyword.trim()) {
      doSearch(nextPage, keyword, true);
    } else {
      fetchMessages(nextPage, activeTab, true);
    }
  }, [loading, hasMore, pageNum, activeTab, showSearch, keyword, fetchMessages, doSearch]);

  const handleGoSettings = useCallback(() => {
    Taro.navigateTo({ url: "/pages/notify/index" });
  }, []);

  const handleSearchConfirm = useCallback(() => {
    if (!keyword.trim()) {
      setShowSearch(false);
      setPageNum(1);
      fetchMessages(1, activeTab, false);
      return;
    }
    setMessages([]);
    setPageNum(1);
    setHasMore(true);
    doSearch(1, keyword, false);
  }, [keyword, activeTab, fetchMessages, doSearch]);

  const handleDeleteSingle = useCallback(async (id: number) => {
    const res = await Taro.showModal({
      title: "确认删除",
      content: "确定删除这条消息吗？",
    });
    if (!res.confirm) return;
    try {
      await MessageAPI.deleteByIds(String(id));
      setMessages((prev) => prev.filter((m) => m.id !== id));
      Taro.showToast({ title: "已删除", icon: "success" });
      fetchUnreadCount();
    } catch (error) {
      Taro.showToast({
        title: getErrorMessage(error, "删除失败"),
        icon: "none",
      });
    }
  }, [fetchUnreadCount]);

  const handleBatchDelete = useCallback(async () => {
    if (selectedIds.size === 0) return;
    const res = await Taro.showModal({
      title: "批量删除",
      content: `确定删除选中的 ${selectedIds.size} 条消息吗？`,
    });
    if (!res.confirm) return;
    try {
      await MessageAPI.deleteByIds(Array.from(selectedIds).join(","));
      setMessages((prev) => prev.filter((m) => !selectedIds.has(m.id)));
      setSelectedIds(new Set());
      setDeleteMode(false);
      Taro.showToast({ title: `已删除 ${selectedIds.size} 条`, icon: "success" });
      fetchUnreadCount();
    } catch (error) {
      Taro.showToast({
        title: getErrorMessage(error, "批量删除失败"),
        icon: "none",
      });
    }
  }, [selectedIds, fetchUnreadCount]);

  const getTypeLabel = (type: string): string => {
    const tab = TABS.find((t) => t.key === type);
    return tab ? tab.label : type;
  };

  return (
    <PageLayout level="L1" title="消息">
      <View className="messages-page">
        {/* 搜索栏 */}
        {showSearch && (
          <View className="messages-search-bar">
            <View className="messages-search-input-wrap">
              <Input
                className="messages-search-input"
                placeholder="搜索消息"
                value={keyword}
                onInput={(e) => setKeyword(e.detail.value)}
                onConfirm={handleSearchConfirm}
                focus
              />
              <View
                className="messages-search-cancel"
                onClick={() => {
                  setShowSearch(false);
                  setKeyword("");
                  setPageNum(1);
                  fetchMessages(1, activeTab, false);
                }}
              >
                <Text>取消</Text>
              </View>
            </View>
            {!keyword && searchHistory.length > 0 && (
              <View className="messages-search-history">
                <Text className="search-history-title">搜索历史</Text>
                <View className="search-history-tags">
                  {searchHistory.map((h) => (
                    <View
                      key={h}
                      className="search-history-tag"
                      onClick={() => {
                        setKeyword(h);
                        doSearch(1, h, false);
                      }}
                    >
                      <Text>{h}</Text>
                    </View>
                  ))}
                </View>
              </View>
            )}
          </View>
        )}

        {/* 顶部操作区 */}
        <View className="messages-header">
          <View className="messages-tabs-scroll">
            <ScrollView scrollX className="messages-tabs">
              <View className="messages-tabs-row">
                {TABS.map((tab) => (
                  <View
                    key={tab.key}
                    className={`messages-tab ${activeTab === tab.key ? "active" : ""}`}
                    onClick={() => handleTabChange(tab.key)}
                  >
                    <Text>{tab.label}</Text>
                  </View>
                ))}
              </View>
            </ScrollView>
          </View>
          <View className="messages-header-actions">
            <View className="messages-header-icon" onClick={() => setShowSearch(true)}>
              <Text className="icon-text">🔍</Text>
            </View>
            {deleteMode ? (
              <>
                <View className="messages-action-btn" onClick={() => { setDeleteMode(false); setSelectedIds(new Set()); }}>
                  <Text>取消</Text>
                </View>
                {selectedIds.size > 0 && (
                  <View className="messages-action-btn danger" onClick={handleBatchDelete}>
                    <Text>删除({selectedIds.size})</Text>
                  </View>
                )}
              </>
            ) : (
              <>
                {unreadCount > 0 && (
                  <View className="messages-mark-all" onClick={handleMarkAllRead}>
                    <Text>全部已读</Text>
                  </View>
                )}
                <View className="messages-header-icon" onClick={() => setDeleteMode(true)}>
                  <Text className="icon-text">🗑</Text>
                </View>
                <View className="messages-header-icon" onClick={handleGoSettings}>
                  <Text className="icon-text">⚙</Text>
                </View>
              </>
            )}
          </View>
        </View>

        {/* 消息列表 */}
        <ScrollView
          className="messages-list"
          scrollY
          onScrollToLower={handleLoadMore}
        >
          {loading && messages.length === 0 ? (
            <View className="messages-loading">
              <Text>加载中...</Text>
            </View>
          ) : messages.length === 0 ? (
            <EmptyState
              type="empty"
              title="暂无消息"
              description="处理完成、系统通知等将在这里展示"
            />
          ) : (
            <View className="messages-items">
              {messages.map((msg) => (
                <View
                  key={msg.id}
                  className={`message-item ${msg.readStatus === 0 ? "unread" : ""} ${deleteMode && selectedIds.has(msg.id) ? "selected" : ""}`}
                  onClick={() => handleMessageClick(msg)}
                  onLongPress={() => handleDeleteSingle(msg.id)}
                >
                  <View className="message-item-left">
                    <View className="message-item-header">
                      <Text className="message-item-type">
                        {msg.typeLabel || getTypeLabel(msg.type)}
                      </Text>
                      {msg.readStatus === 0 && (
                        <View className="message-unread-dot" />
                      )}
                      {deleteMode && (
                        <View className={`message-checkbox ${selectedIds.has(msg.id) ? "checked" : ""}`}>
                          {selectedIds.has(msg.id) && <Text>✓</Text>}
                        </View>
                      )}
                    </View>
                    <Text className="message-item-title">{msg.title}</Text>
                    <Text className="message-item-summary">
                      {msg.summary || ""}
                    </Text>
                  </View>
                  <Text className="message-item-time">{formatTime(msg.createTime)}</Text>
                </View>
              ))}
            </View>
          )}
          {loading && messages.length > 0 && (
            <View className="messages-loading-more">
              <Text>加载更多...</Text>
            </View>
          )}
          {!hasMore && messages.length > 0 && (
            <View className="messages-no-more">
              <Text>没有更多了</Text>
            </View>
          )}
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default MessagesPage;
