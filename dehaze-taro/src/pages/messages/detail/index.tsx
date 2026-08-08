/**
 * 消息详情页面 (L2)
 */
import React, { useState, useEffect, useCallback } from "react";
import { View, Text, RichText } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { MessageAPI } from "dehaze-sdk-js";
import type { MessageVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

function formatDateTime(dateStr: string): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

const MessageDetailPage: React.FC = () => {
  const [message, setMessage] = useState<MessageVO | null>(null);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const instance = Taro.getCurrentInstance();
    const id = instance.router?.params?.id;
    if (!id) {
      Taro.showToast({ title: "参数错误", icon: "none" });
      setTimeout(() => Taro.navigateBack(), 1000);
      return;
    }

    MessageAPI.getDetail(Number(id))
      .then((res) => {
        setMessage(res);
        if (res.readStatus === 0) {
          MessageAPI.markRead(res.id).catch(() => {});
        }
      })
      .catch((error) => {
        Taro.showToast({
          title: getErrorMessage(error, "加载失败"),
          icon: "none",
        });
      })
      .finally(() => setLoading(false));
  }, []);

  const handleJump = useCallback(() => {
    if (message?.jumpUrl) {
      Taro.navigateTo({ url: message.jumpUrl });
    }
  }, [message]);

  const handleDelete = useCallback(async () => {
    if (!message || deleting) return;
    const res = await Taro.showModal({
      title: "确认删除",
      content: "确定删除这条消息吗？",
    });
    if (!res.confirm) return;
    setDeleting(true);
    try {
      await MessageAPI.deleteByIds(String(message.id));
      Taro.showToast({ title: "已删除", icon: "success" });
      setTimeout(() => Taro.navigateBack(), 800);
    } catch (error) {
      Taro.showToast({
        title: getErrorMessage(error, "删除失败"),
        icon: "none",
      });
    } finally {
      setDeleting(false);
    }
  }, [message, deleting]);

  if (loading) {
    return (
      <PageLayout level="L2" title="消息详情">
        <View className="message-detail-loading">
          <Text>加载中...</Text>
        </View>
      </PageLayout>
    );
  }

  if (!message) {
    return (
      <PageLayout level="L2" title="消息详情">
        <View className="message-detail-empty">
          <Text>消息不存在</Text>
        </View>
      </PageLayout>
    );
  }

  return (
    <PageLayout level="L2" title="消息详情">
      <View className="message-detail-page">
        <View className="message-detail-header">
          <View className="message-detail-type">
            <Text>{message.typeLabel}</Text>
          </View>
          <Text className="message-detail-title">{message.title}</Text>
          <Text className="message-detail-time">{formatDateTime(message.createTime)}</Text>
        </View>
        <View className="message-detail-body">
          {message.content ? (
            <RichText
              nodes={message.content}
              className="message-detail-content"
            />
          ) : (
            <Text className="message-detail-summary">
              {message.summary || "暂无内容"}
            </Text>
          )}
        </View>
        <View className="message-detail-footer">
          {message.jumpUrl && (
            <View className="message-detail-action" onClick={handleJump}>
              <Text>查看详情 →</Text>
            </View>
          )}
          <View className="message-detail-delete" onClick={handleDelete}>
            <Text>{deleting ? "删除中..." : "删除"}</Text>
          </View>
        </View>
      </View>
    </PageLayout>
  );
};

export default MessageDetailPage;
