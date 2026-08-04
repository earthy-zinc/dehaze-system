import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Bell } from "@taroify/icons";
import PageLayout from "@/layout";
import EmptyState from "@/components/common/EmptyState";
import { tabBarItems } from "@/config/menu";
import { MessageAPI } from "dehaze-sdk-js";
import type { MessageVO } from "dehaze-sdk-js";
import { formatDateTime } from "@/utils/format";
import "./index.less";

const PRIORITY_COLOR: Record<number, string> = {
  1: "#ef4444",
  2: "#f59e0b",
  3: "#6b7280",
};

const NotifyPage: React.FC = () => {
  const [messages, setMessages] = useState<MessageVO[]>([]);
  const [loading, setLoading] = useState(false);

  const loadMessages = useCallback(async () => {
    setLoading(true);
    try {
      const res = await MessageAPI.getPage({ pageNum: 1, pageSize: 50 });
      setMessages(res.list || []);
    } catch {
      Taro.showToast({ title: "加载失败", icon: "none" });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMessages();
  }, [loadMessages]);

  const markAsRead = useCallback(async (id: number) => {
    try {
      await MessageAPI.markRead(id);
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, readStatus: 1 } : m))
      );
    } catch {
      /* silent */
    }
  }, []);

  const handleTapMessage = useCallback(
    (item: MessageVO) => {
      if (item.readStatus === 0) {
        markAsRead(item.id);
      }
      if (item.jumpUrl) {
        // 目标为 Tab 根页面时用 switchTab（禁止压栈进入 Tab 页），其余用 navigateTo
        const target = item.jumpUrl.split("?")[0];
        if (tabBarItems.some((t) => t.route === target)) {
          Taro.switchTab({ url: target });
        } else {
          Taro.navigateTo({ url: item.jumpUrl });
        }
      }
    },
    [markAsRead]
  );

  const handleMarkAllRead = useCallback(async () => {
    try {
      await MessageAPI.markAllRead();
      setMessages((prev) => prev.map((m) => ({ ...m, readStatus: 1 })));
      Taro.showToast({ title: "已全部标记为已读", icon: "success" });
    } catch {
      Taro.showToast({ title: "操作失败", icon: "none" });
    }
  }, []);

  const handleDelete = useCallback(async (id: number) => {
    try {
      await MessageAPI.deleteByIds(String(id));
      setMessages((prev) => prev.filter((m) => m.id !== id));
      Taro.showToast({ title: "已删除", icon: "success" });
    } catch {
      Taro.showToast({ title: "删除失败", icon: "none" });
    }
  }, []);

  const handleLongPressDelete = useCallback(
    async (item: MessageVO) => {
      const confirmed = await Taro.showModal({
        title: "删除消息",
        content: `确定要删除「${item.title}」吗？`,
        confirmText: "删除",
        confirmColor: "#ef4444",
      });
      if (confirmed.confirm) {
        handleDelete(item.id);
      }
    },
    [handleDelete]
  );

  const hasUnread = messages.some((m) => m.readStatus === 0);

  return (
    <PageLayout level="L2" title="消息通知">
      <View className="notify-page">
        <View className="notify-header">
          <View className="header-title">
            <Text className="title-text">消息中心</Text>
            {hasUnread && <View className="unread-dot" />}
          </View>
          <View className="header-actions">
            <View className="action-btn" onClick={handleMarkAllRead}>
              <Text className="action-text">全部已读</Text>
            </View>
          </View>
        </View>

        <ScrollView
          scrollY
          className="notify-list"
          enhanced
          showScrollbar={false}
        >
          {loading ? (
            <View className="loading-wrapper">
              <Text className="loading-text">加载中...</Text>
            </View>
          ) : messages.length === 0 ? (
            <View className="empty-wrapper">
              <EmptyState
                type="search"
                title="暂无消息"
                description="目前没有新的通知"
              />
            </View>
          ) : (
            <>
              {messages.map((item) => (
                <View
                  key={item.id}
                  className={`notify-item ${item.readStatus === 1 ? "read" : "unread"}`}
                  onClick={() => handleTapMessage(item)}
                  onLongPress={() => handleLongPressDelete(item)}
                >
                  <View className="notify-icon">
                    <Bell
                      size="20"
                      color={item.readStatus === 1 ? "#9ca3af" : "#3b82f6"}
                    />
                  </View>
                  <View className="notify-content">
                    <View className="notify-top">
                      <Text
                        className={`notify-title ${item.readStatus === 1 ? "" : "unread-title"}`}
                      >
                        {item.title}
                      </Text>
                      <Text className="notify-time">
                        {formatDateTime(item.createTime)}
                      </Text>
                    </View>
                    <Text className="notify-body" numberOfLines={2}>
                      {item.summary || item.content || ""}
                    </Text>
                  </View>
                  {item.typeLabel && (
                    <View className="notify-type-tag">
                      <Text
                        className="type-tag-text"
                        style={{
                          color: PRIORITY_COLOR[item.priority] || "#6b7280",
                        }}
                      >
                        {item.typeLabel}
                      </Text>
                    </View>
                  )}
                </View>
              ))}
            </>
          )}
        </ScrollView>
      </View>
    </PageLayout>
  );
};

export default NotifyPage;
