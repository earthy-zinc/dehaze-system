import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Button, Tag, Popup, Textarea, Input, Tabs } from "@taroify/core";
import { Plus } from "@taroify/icons";
import { FeedbackAPI } from "dehaze-sdk-js";
import type {
  FeedbackPageVO,
  FeedbackDetailVO,
  MyRatingVO,
  FeedbackType,
  FeedbackStatus,
} from "dehaze-sdk-js";
import PageLayout from "@/layout";
import EmptyState from "@/components/common/EmptyState";
import "./index.less";

type TagColor =
  "default" | "primary" | "info" | "success" | "warning" | "danger";

const STATUS_CONFIG: Record<FeedbackStatus, { label: string; color: string }> =
  {
    pending: { label: "待处理", color: "#f59e0b" },
    processing: { label: "处理中", color: "#3b82f6" },
    replied: { label: "已回复", color: "#10b981" },
    closed: { label: "已关闭", color: "#6b7280" },
  };

const TYPE_CONFIG: Record<FeedbackType, { label: string }> = {
  suggestion: { label: "建议" },
  bug: { label: "缺陷" },
  experience: { label: "体验" },
  complaint: { label: "投诉" },
};

const FEEDBACK_TYPES: { value: FeedbackType; label: string }[] = [
  { value: "suggestion", label: "建议" },
  { value: "bug", label: "缺陷" },
  { value: "experience", label: "体验" },
  { value: "complaint", label: "投诉" },
];

const FeedbackPage: React.FC = () => {
  const [tab, setTab] = useState(0);

  // 反馈
  const [feedbackList, setFeedbackList] = useState<FeedbackPageVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitVisible, setSubmitVisible] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [formType, setFormType] = useState<FeedbackType>("suggestion");
  const [formTitle, setFormTitle] = useState("");
  const [formContent, setFormContent] = useState("");
  const [formContact, setFormContact] = useState("");

  // 反馈详情
  const [detailVisible, setDetailVisible] = useState(false);
  const [feedbackDetail, setFeedbackDetail] = useState<FeedbackDetailVO | null>(
    null
  );
  const [supplementVisible, setSupplementVisible] = useState(false);
  const [supplementContent, setSupplementContent] = useState("");
  const [supplementing, setSupplementing] = useState(false);

  // 评价
  const [ratings, setRatings] = useState<MyRatingVO[]>([]);
  const [ratingLoading, setRatingLoading] = useState(false);

  const loadFeedback = useCallback(async () => {
    setLoading(true);
    try {
      const res = await FeedbackAPI.listMyFeedback({
        pageNum: 1,
        pageSize: 20,
      });
      setFeedbackList(res.list || []);
    } catch {
      Taro.showToast({ title: "加载失败", icon: "none" });
    } finally {
      setLoading(false);
    }
  }, []);

  const loadRatings = useCallback(async () => {
    setRatingLoading(true);
    try {
      const res = await FeedbackAPI.listMyRatings({
        pageNum: 1,
        pageSize: 20,
      });
      setRatings(res.list || []);
    } catch {
      Taro.showToast({ title: "加载评价失败", icon: "none" });
    } finally {
      setRatingLoading(false);
    }
  }, []);

  useEffect(() => {
    loadFeedback();
    loadRatings();
  }, [loadFeedback, loadRatings]);

  const handleSubmit = async () => {
    if (!formTitle.trim()) {
      Taro.showToast({ title: "请输入标题", icon: "none" });
      return;
    }
    if (!formContent.trim()) {
      Taro.showToast({ title: "请输入反馈内容", icon: "none" });
      return;
    }
    setSubmitting(true);
    try {
      await FeedbackAPI.createFeedback({
        feedbackType: formType,
        title: formTitle.trim(),
        content: formContent.trim(),
        contact: formContact.trim() || undefined,
      });
      setSubmitVisible(false);
      setFormTitle("");
      setFormContent("");
      setFormContact("");
      Taro.showToast({ title: "提交成功", icon: "success" });
      loadFeedback();
    } catch {
      Taro.showToast({ title: "提交失败", icon: "none" });
    } finally {
      setSubmitting(false);
    }
  };

  const handleViewDetail = async (id: number) => {
    try {
      const detail = await FeedbackAPI.getFeedbackDetail(id);
      setFeedbackDetail(detail);
      setDetailVisible(true);
    } catch {
      Taro.showToast({ title: "获取详情失败", icon: "none" });
    }
  };

  const handleSupplement = async () => {
    if (!supplementContent.trim() || !feedbackDetail) return;
    setSupplementing(true);
    try {
      await FeedbackAPI.supplementFeedback(feedbackDetail.id, {
        content: supplementContent.trim(),
      });
      Taro.showToast({ title: "补充成功", icon: "success" });
      setSupplementVisible(false);
      setSupplementContent("");
      const detail = await FeedbackAPI.getFeedbackDetail(feedbackDetail.id);
      setFeedbackDetail(detail);
    } catch {
      Taro.showToast({ title: "补充失败", icon: "none" });
    } finally {
      setSupplementing(false);
    }
  };

  return (
    <PageLayout level="L2" title="反馈评价">
      <View className="personal-feedback-page">
        <Tabs value={tab} onChange={setTab}>
          <Tabs.TabPane title="我的反馈" />
          <Tabs.TabPane title="我的评价" />
        </Tabs>

        {tab === 0 && (
          <>
            <View className="feedback-toolbar">
              <View
                className="submit-btn"
                onClick={() => setSubmitVisible(true)}
              >
                <Plus size="16" color="#fff" />
                <Text className="submit-text">提交反馈</Text>
              </View>
            </View>

            <ScrollView
              scrollY
              className="feedback-list"
              enhanced
              showScrollbar={false}
            >
              {loading && feedbackList.length === 0 ? (
                <View className="loading-wrapper">
                  <Text>加载中...</Text>
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
                feedbackList.map((item) => {
                  const typeCfg =
                    TYPE_CONFIG[item.feedbackType as FeedbackType] ||
                    TYPE_CONFIG.suggestion;
                  const status = item.status as FeedbackStatus;
                  const statusCfg =
                    STATUS_CONFIG[status] || STATUS_CONFIG.pending;
                  return (
                    <View
                      key={item.id}
                      className="feedback-card"
                      onClick={() => handleViewDetail(item.id)}
                    >
                      <View className="feedback-top">
                        <View className="feedback-title-row">
                          <Text className="feedback-title">{item.title}</Text>
                          <Tag color={statusCfg.color as TagColor} size="small">
                            {statusCfg.label}
                          </Tag>
                        </View>
                      </View>
                      <Text className="feedback-content" numberOfLines={2}>
                        {item.content}
                      </Text>
                      <View className="feedback-bottom">
                        <Tag size="small" color="default">
                          {typeCfg.label}
                        </Tag>
                        <Text className="feedback-time">
                          {new Date(item.createTime).toLocaleString()}
                        </Text>
                      </View>
                    </View>
                  );
                })
              )}
            </ScrollView>
          </>
        )}

        {tab === 1 && (
          <ScrollView
            scrollY
            className="feedback-list"
            enhanced
            showScrollbar={false}
          >
            {ratingLoading && ratings.length === 0 ? (
              <View className="loading-wrapper">
                <Text>加载中...</Text>
              </View>
            ) : ratings.length === 0 ? (
              <View className="empty-wrapper">
                <EmptyState
                  type="search"
                  title="暂无评价"
                  description="去雾处理后可以评价效果哦"
                />
              </View>
            ) : (
              ratings.map((r) => (
                <View key={r.id} className="feedback-card">
                  <View className="feedback-top">
                    <View className="feedback-title-row">
                      <Text className="feedback-title">{r.algorithmName}</Text>
                      <Tag color="warning" size="small">
                        {"★".repeat(r.rating)}
                      </Tag>
                    </View>
                  </View>
                  {r.comment && (
                    <Text className="feedback-content">{r.comment}</Text>
                  )}
                  {r.adminReply && (
                    <View className="admin-reply">
                      <Text className="reply-label">管理员回复: </Text>
                      <Text>{r.adminReply}</Text>
                    </View>
                  )}
                  <View className="feedback-bottom">
                    <Text className="feedback-time">
                      {new Date(r.createTime).toLocaleString()}
                    </Text>
                  </View>
                </View>
              ))
            )}
          </ScrollView>
        )}

        {/* 提交弹窗 */}
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
            <Input
              className="form-input-field"
              placeholder="标题（必填）"
              value={formTitle}
              onInput={(e) => setFormTitle(e.detail?.value || "")}
              maxlength={100}
            />
            <Textarea
              className="content-textarea"
              placeholder="请描述您遇到的问题或建议..."
              value={formContent}
              onChange={(e) => setFormContent(e.detail?.value || "")}
              maxlength={2000}
            />
            <Input
              className="form-input-field"
              placeholder="联系方式（选填，如邮箱/手机号）"
              value={formContact}
              onInput={(e) => setFormContact(e.detail?.value || "")}
              maxlength={100}
            />
            <Button
              variant="contained"
              block
              loading={submitting}
              onClick={handleSubmit}
              className="submit-action-btn"
            >
              <Text className="submit-btn-text">提交反馈</Text>
            </Button>
          </View>
        </Popup>

        {/* 反馈详情弹窗 */}
        <Popup
          open={detailVisible}
          placement="bottom"
          rounded
          onClose={() => {
            setDetailVisible(false);
            setFeedbackDetail(null);
          }}
        >
          <View className="submit-popup">
            {feedbackDetail && (
              <>
                <View className="popup-header">
                  <Text className="popup-title">反馈详情</Text>
                  <View
                    className="popup-close"
                    onClick={() => {
                      setDetailVisible(false);
                      setFeedbackDetail(null);
                    }}
                  >
                    ×
                  </View>
                </View>
                <View className="detail-section">
                  <View className="detail-row">
                    <Text className="detail-label">类型</Text>
                    <Tag size="small" color="default">
                      {TYPE_CONFIG[feedbackDetail.feedbackType as FeedbackType]
                        ?.label || feedbackDetail.feedbackType}
                    </Tag>
                  </View>
                  <View className="detail-row">
                    <Text className="detail-label">状态</Text>
                    <Tag
                      size="small"
                      color={
                        STATUS_CONFIG[feedbackDetail.status as FeedbackStatus]
                          ?.color as TagColor
                      }
                    >
                      {STATUS_CONFIG[feedbackDetail.status as FeedbackStatus]
                        ?.label || feedbackDetail.status}
                    </Tag>
                  </View>
                  <View className="detail-row">
                    <Text className="detail-label">标题</Text>
                    <Text className="detail-value">{feedbackDetail.title}</Text>
                  </View>
                  <View className="detail-row">
                    <Text className="detail-label">内容</Text>
                    <Text className="detail-value">
                      {feedbackDetail.content}
                    </Text>
                  </View>
                  {feedbackDetail.contact && (
                    <View className="detail-row">
                      <Text className="detail-label">联系方式</Text>
                      <Text className="detail-value">
                        {feedbackDetail.contact}
                      </Text>
                    </View>
                  )}
                  <View className="detail-row">
                    <Text className="detail-label">时间</Text>
                    <Text className="detail-value">
                      {new Date(feedbackDetail.createTime).toLocaleString()}
                    </Text>
                  </View>
                </View>

                {feedbackDetail.replies &&
                  feedbackDetail.replies.length > 0 && (
                    <View className="replies-section">
                      <Text className="section-title">回复记录</Text>
                      {feedbackDetail.replies.map((reply) => (
                        <View key={reply.id} className="reply-item">
                          <View className="reply-header">
                            <Text className="reply-author">
                              {reply.replierName}
                            </Text>
                            <Text className="reply-time">
                              {new Date(reply.createTime).toLocaleString()}
                            </Text>
                          </View>
                          <Text className="reply-content">{reply.content}</Text>
                        </View>
                      ))}
                    </View>
                  )}

                {feedbackDetail.status !== "closed" && (
                  <View className="detail-actions">
                    <Button
                      variant="outlined"
                      onClick={() => setSupplementVisible(true)}
                    >
                      补充说明
                    </Button>
                  </View>
                )}
              </>
            )}
          </View>
        </Popup>

        {/* 补充说明弹窗 */}
        <Popup
          open={supplementVisible}
          placement="bottom"
          rounded
          onClose={() => setSupplementVisible(false)}
        >
          <View className="submit-popup">
            <View className="popup-header">
              <Text className="popup-title">补充说明</Text>
              <View
                className="popup-close"
                onClick={() => setSupplementVisible(false)}
              >
                ×
              </View>
            </View>
            <Textarea
              className="content-textarea"
              placeholder="请输入补充内容..."
              value={supplementContent}
              onChange={(e) => setSupplementContent(e.detail?.value || "")}
              maxlength={2000}
            />
            <Button
              variant="contained"
              block
              loading={supplementing}
              onClick={handleSupplement}
              className="submit-action-btn"
            >
              提交补充
            </Button>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default FeedbackPage;
