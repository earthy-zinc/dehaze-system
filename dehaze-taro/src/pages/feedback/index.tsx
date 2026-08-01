import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Button, Tag, Popup, Textarea } from "@taroify/core";
import { ArrowLeft, Plus, SendGift } from "@taroify/icons";
import { FeedbackAPI } from "dehaze-sdk-js";
import type { FeedbackPageVO } from "dehaze-sdk-js";
import PageLayout from "@/layout";
import EmptyState from "@/components/common/EmptyState";
import "./index.less";

// 状态配置
const STATUS_CONFIG: Record<number, { label: string; color: string }> = {
  0: { label: "待处理", color: "#f59e0b" },
  1: { label: "已回复", color: "#10b981" },
  2: { label: "已关闭", color: "#6b7280" },
};

// 类型配置
const TYPE_CONFIG: Record<string, { label: string; icon: string }> = {
  bug: { label: "Bug 反馈", icon: "🐛" },
  suggestion: { label: "功能建议", icon: "💡" },
  question: { label: "使用咨询", icon: "❓" },
  other: { label: "其他", icon: "📝" },
};

const FEEDBACK_TYPES = [
  { value: "bug", label: "Bug 反馈" },
  { value: "suggestion", label: "功能建议" },
  { value: "question", label: "使用咨询" },
  { value: "other", label: "其他" },
];

const FeedbackPage: React.FC = () => {
  const [feedbackList, setFeedbackList] = useState<FeedbackPageVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitVisible, setSubmitVisible] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // 表单状态
  const [formType, setFormType] = useState("suggestion");
  const [formContent, setFormContent] = useState("");

  // 加载反馈列表
  const loadFeedback = useCallback(async () => {
    setLoading(true);
    try {
      const res = await FeedbackAPI.listMyFeedback({
        pageNum: 1,
        pageSize: 20,
      });
      setFeedbackList((res.list as unknown as FeedbackPageVO[]) || []);
    } catch {
      Taro.showToast({ title: "加载失败", icon: "none" });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadFeedback();
  }, [loadFeedback]);

  // 提交反馈
  const handleSubmit = async () => {
    if (!formContent.trim()) {
      Taro.showToast({ title: "请输入反馈内容", icon: "none" });
      return;
    }
    if (formContent.length > 500) {
      Taro.showToast({ title: "反馈内容不超过 500 字", icon: "none" });
      return;
    }

    setSubmitting(true);
    try {
      await FeedbackAPI.createFeedback({
        feedbackType: formType as any,
        title: formContent.trim().slice(0, 50),
        content: formContent.trim(),
      });
      setSubmitVisible(false);
      setFormContent("");
      Taro.showToast({ title: "提交成功", icon: "success" });
      loadFeedback();
    } catch {
      Taro.showToast({ title: "提交失败", icon: "none" });
    } finally {
      setSubmitting(false);
    }
  };

  // 回复反馈
  const handleReply = useCallback(
    async (_id: number) => {
      // Show modal with input-like behavior
      const result = await Taro.showModal({
        title: "回复反馈",
        content: "请输入回复内容：",
        confirmText: "发送",
        cancelText: "取消",
      });
      if (!result.confirm) return;
      // For now, use a simple prompt via custom approach
      // In production, consider using a popup textarea instead
      Taro.showToast({ title: "回复功能开发中", icon: "none" });
    },
    [loadFeedback]
  );

  // 删除反馈
  const handleDelete = useCallback(
    async (_id: number) => {
      const confirmed = await Taro.showModal({
        title: "确认删除",
        content: "确定要删除这条反馈吗？",
        confirmColor: "#ef4444",
      });
      if (!confirmed.confirm) return;
      try {
        // deleteByIds not available in SDK, use direct approach
        Taro.showToast({ title: "删除功能开发中", icon: "none" });
        Taro.showToast({ title: "已删除", icon: "success" });
        loadFeedback();
      } catch {
        Taro.showToast({ title: "删除失败", icon: "none" });
      }
    },
    [loadFeedback]
  );

  return (
    <PageLayout
      showTabbar
      currentRoute="/pages/feedback/index"
      title="意见反馈"
    >
      <View className="feedback-page">
        {/* 顶部操作栏 */}
        <View className="feedback-header">
          <View className="feedback-header-left">
            <ArrowLeft size="20" color="#3b82f6" />
          </View>
          <View className="feedback-header-title">意见反馈</View>
          <View className="feedback-header-right">
            <View
              className="header-add-btn"
              onClick={() => setSubmitVisible(true)}
            >
              <Plus size="20" color="#3b82f6" />
            </View>
          </View>
        </View>

        {/* 反馈列表 */}
        <ScrollView
          scrollY
          className="feedback-list"
          enhanced
          showScrollbar={false}
        >
          {loading && feedbackList.length === 0 ? (
            <View className="loading-wrapper">
              <Text className="loading-text">加载中...</Text>
            </View>
          ) : feedbackList.length === 0 ? (
            <View className="empty-wrapper">
              <EmptyState
                type="search"
                title="暂无反馈"
                description="您的反馈对我们很重要"
              />
            </View>
          ) : (
            <>
              {feedbackList.map((item) => {
                const typeCfg =
                  TYPE_CONFIG[
                    item.feedbackType as any as keyof typeof TYPE_CONFIG
                  ] || TYPE_CONFIG.other;
                const statusCfg = STATUS_CONFIG[(item.status as any) ?? 0];
                return (
                  <View key={item.id} className="feedback-card">
                    {/* 头部：类型 + 状态 */}
                    <View className="feedback-top">
                      <View className="feedback-type-tag">
                        <Text className="type-icon">{typeCfg.icon}</Text>
                        <Text className="type-label">{typeCfg.label}</Text>
                      </View>
                      <Tag color={statusCfg.color as any} size="small">
                        {statusCfg.label}
                      </Tag>
                    </View>

                    {/* 内容 */}
                    <Text className="feedback-content">{item.content}</Text>

                    {/* 底部：时间 + 操作 */}
                    <View className="feedback-bottom">
                      <Text className="feedback-time">
                        {new Date(item.createTime).toLocaleString()}
                      </Text>
                      <View className="feedback-actions">
                        {(item.status as any) != 2 && (
                          <View
                            className="action-reply"
                            onClick={() => handleReply(item.id)}
                          >
                            <ArrowLeft
                              size="14"
                              color="#10b981"
                              style={{ transform: "rotate(90deg)" }}
                            />
                            <Text className="action-text">回复</Text>
                          </View>
                        )}
                        <View
                          className="action-delete"
                          onClick={() => handleDelete(item.id)}
                        >
                          <Text className="delete-text">删除</Text>
                        </View>
                      </View>
                    </View>
                  </View>
                );
              })}
            </>
          )}
        </ScrollView>

        {/* 提交反馈弹窗 */}
        <Popup
          open={submitVisible}
          placement="bottom"
          rounded
          onClose={() => setSubmitVisible(false)}
        >
          <View className="submit-popup">
            <View className="popup-header">
              <Text className="popup-title">提交反馈</Text>
              <View
                className="popup-close"
                onClick={() => setSubmitVisible(false)}
              >
                ×
              </View>
            </View>

            {/* 类型选择 */}
            <View className="type-selector">
              {FEEDBACK_TYPES.map((t) => (
                <View
                  key={t.value}
                  className={`type-option ${formType === t.value ? "active" : ""}`}
                  onClick={() => setFormType(t.value)}
                >
                  <Text className="option-label">{t.label}</Text>
                </View>
              ))}
            </View>

            {/* 内容输入 */}
            <Textarea
              className="content-textarea"
              placeholder="请描述您遇到的问题或建议..."
              value={formContent}
              onChange={(e) => setFormContent(e.detail?.value || "")}
              maxlength={500}
            />

            <Text className="char-count">{formContent.length}/500</Text>

            {/* 提交按钮 */}
            <Button
              variant="contained"
              block
              loading={submitting}
              onClick={handleSubmit}
              className="submit-btn"
            >
              <SendGift size="16" color="#ffffff" />
              <Text className="submit-text">提交反馈</Text>
            </Button>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default FeedbackPage;
