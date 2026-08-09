import React, { useState, useEffect, useCallback } from "react";
import { View, Text, ScrollView } from "@tarojs/components";
import Taro from "@tarojs/taro";
import { Tag, Loading, Empty, Tabs, Popup, Input, Textarea } from "@taroify/core";
import { FeedbackAPI } from "dehaze-sdk-js";
import type {
  FeedbackPageVO,
  RatingPageVO,
  FeedbackQuery,
  FeedbackStatus,
  FeedbackType,
} from "dehaze-sdk-js";
import PageLayout from "@/layout";
import { usePermission } from "@/hooks/usePermission";
import { getErrorMessage } from "@/utils/error";
import "./index.less";

const FEEDBACK_STATUS_LABELS: Record<string, string> = {
  pending: "待处理",
  processing: "处理中",
  replied: "已回复",
  closed: "已关闭",
};

const FEEDBACK_TYPE_LABELS: Record<string, string> = {
  suggestion: "建议",
  bug: "缺陷",
  experience: "体验",
  complaint: "投诉",
};

const FeedbackManagePage: React.FC = () => {
  const { hasPermission } = usePermission();
  const canManage = hasPermission("sys:feedback:*");

  const [tab, setTab] = useState(0);
  const [feedbacks, setFeedbacks] = useState<FeedbackPageVO[]>([]);
  const [ratings, setRatings] = useState<RatingPageVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [totalFeedback, setTotalFeedback] = useState(0);
  const [totalRating, setTotalRating] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [ratingPageNum, setRatingPageNum] = useState(1);
  const [keyword, setKeyword] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [typeFilter, setTypeFilter] = useState("");

  const [replyPopupVisible, setReplyPopupVisible] = useState(false);
  const [replyTarget, setReplyTarget] = useState<{
    id: number;
    type: "feedback" | "rating";
  } | null>(null);
  const [replyContent, setReplyContent] = useState("");

  const [closePopupVisible, setClosePopupVisible] = useState(false);
  const [closeTargetId, setCloseTargetId] = useState<number | null>(null);
  const [closeReason, setCloseReason] = useState("");

  const fetchFeedbacks = useCallback(
    async (page: number, kw: string, status: string, type: string) => {
      setLoading(true);
      try {
        const params: FeedbackQuery = { pageNum: page, pageSize: 15 };
        if (kw) params.keywords = kw;
        if (status) params.status = status as FeedbackStatus;
        if (type) params.feedbackType = type as FeedbackType;
        const res = await FeedbackAPI.listFeedback(params);
        setFeedbacks(res.list);
        setTotalFeedback(res.total);
        setPageNum(page);
      } catch (err: unknown) {
        Taro.showToast({
          title: getErrorMessage(err, "加载反馈失败"),
          icon: "none",
        });
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const fetchRatings = useCallback(async (page: number) => {
    try {
      const res = await FeedbackAPI.listRatings({
        pageNum: page,
        pageSize: 15,
      });
      setRatings(res.list);
      setTotalRating(res.total);
      setRatingPageNum(page);
    } catch {
      // 静默
    }
  }, []);

  useEffect(() => {
    fetchFeedbacks(1, "", "", "");
    fetchRatings(1);
  }, [fetchFeedbacks, fetchRatings]);

  const handleSearch = () => {
    fetchFeedbacks(1, keyword, statusFilter, typeFilter);
  };

  const handleLoadMoreFeedback = () => {
    if (feedbacks.length < totalFeedback) {
      fetchFeedbacks(pageNum + 1, keyword, statusFilter, typeFilter);
    }
  };

  const handleLoadMoreRating = () => {
    if (ratings.length < totalRating) {
      fetchRatings(ratingPageNum + 1);
    }
  };

  const openClose = (id: number) => {
    setCloseTargetId(id);
    setCloseReason("");
    setClosePopupVisible(true);
  };

  const handleCloseFeedback = async () => {
    if (!closeTargetId || !closeReason.trim()) {
      Taro.showToast({ title: "请输入关闭原因", icon: "none" });
      return;
    }
    try {
      await FeedbackAPI.closeFeedback(closeTargetId, {
        closeReason: closeReason.trim(),
      });
      Taro.showToast({ title: "已关闭", icon: "success" });
      setClosePopupVisible(false);
      fetchFeedbacks(pageNum, keyword, statusFilter, typeFilter);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "操作失败"), icon: "none" });
    }
  };

  const openReply = (id: number, type: "feedback" | "rating") => {
    setReplyTarget({ id, type });
    setReplyContent("");
    setReplyPopupVisible(true);
  };

  const handleReply = async () => {
    if (!replyTarget || !replyContent.trim()) {
      Taro.showToast({ title: "请输入回复内容", icon: "none" });
      return;
    }
    try {
      if (replyTarget.type === "feedback") {
        await FeedbackAPI.replyFeedback(replyTarget.id, {
          content: replyContent,
        });
      } else {
        await FeedbackAPI.replyRating(replyTarget.id, replyContent);
      }
      Taro.showToast({ title: "回复成功", icon: "success" });
      setReplyPopupVisible(false);
      if (replyTarget.type === "feedback") {
        fetchFeedbacks(pageNum, keyword, statusFilter, typeFilter);
      } else {
        fetchRatings(ratingPageNum);
      }
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "回复失败"), icon: "none" });
    }
  };

  const handleHideRating = async (id: number) => {
    const res = await Taro.showModal({
      title: "确认隐藏",
      content: "确定要隐藏这条评价吗？",
    });
    if (!res.confirm) return;
    try {
      await FeedbackAPI.hideRating(id);
      Taro.showToast({ title: "已隐藏", icon: "success" });
      fetchRatings(ratingPageNum);
    } catch (err: unknown) {
      Taro.showToast({ title: getErrorMessage(err, "操作失败"), icon: "none" });
    }
  };

  return (
    <PageLayout level="L2" title="反馈评价管理">
      <View className="system-manage-page">
        <Tabs value={tab} onChange={setTab}>
          <Tabs.TabPane title="反馈列表" />
          <Tabs.TabPane title="评价列表" />
        </Tabs>

        {tab === 0 && (
          <>
            <View className="search-bar">
              <Input
                className="search-input"
                placeholder="搜索反馈内容"
                value={keyword}
                onInput={(e) => setKeyword(e.detail.value)}
                onConfirm={handleSearch}
              />
              <View className="filter-row">
                {["", "pending", "processing", "replied", "closed"].map((s) => (
                  <Tag
                    key={s}
                    color={statusFilter === s ? "primary" : "default"}
                    size="small"
                    onClick={() => {
                      setStatusFilter(s);
                      fetchFeedbacks(1, keyword, s, typeFilter);
                    }}
                  >
                    {s ? FEEDBACK_STATUS_LABELS[s] || s : "全部"}
                  </Tag>
                ))}
              </View>
              <View className="filter-row" style={{ marginTop: "12rpx" }}>
                {["", "suggestion", "bug", "experience", "complaint"].map(
                  (t) => (
                    <Tag
                      key={t}
                      color={typeFilter === t ? "primary" : "default"}
                      size="small"
                      onClick={() => {
                        setTypeFilter(t);
                        fetchFeedbacks(1, keyword, statusFilter, t);
                      }}
                    >
                      {t ? FEEDBACK_TYPE_LABELS[t] || t : "全部类型"}
                    </Tag>
                  )
                )}
              </View>
            </View>

            <ScrollView
              scrollY
              className="list-scroll"
              onScrollToLower={handleLoadMoreFeedback}
            >
              {loading && feedbacks.length === 0 ? (
                <View className="loading-wrapper">
                  <Loading>加载中...</Loading>
                </View>
              ) : feedbacks.length === 0 ? (
                <Empty>
                  <Empty.Description>暂无反馈数据</Empty.Description>
                </Empty>
              ) : (
                feedbacks.map((f) => (
                  <View key={f.id} className="list-card">
                    <View className="card-header">
                      <View className="card-title-row">
                        <Text className="card-name">{f.title}</Text>
                        <Tag
                          size="small"
                          color={
                            f.status === "closed"
                              ? "default"
                              : f.status === "replied"
                                ? "success"
                                : f.status === "processing"
                                  ? "warning"
                                  : "primary"
                          }
                        >
                          {FEEDBACK_STATUS_LABELS[f.status] || f.status}
                        </Tag>
                      </View>
                      <Tag size="small" color="default">
                        {FEEDBACK_TYPE_LABELS[f.feedbackType] || f.feedbackType}
                      </Tag>
                    </View>
                    <View className="card-meta">
                      <Text className="meta-item">
                        用户: {f.username} (ID:{f.userId})
                      </Text>
                      {f.assigneeName && (
                        <Text className="meta-item">
                          处理人: {f.assigneeName}
                        </Text>
                      )}
                      <Text className="meta-item">优先级: {f.priority}</Text>
                    </View>
                    <Text className="card-content" numberOfLines={2}>
                      {f.content}
                    </Text>
                    <View className="card-meta">
                      <Text className="meta-item">
                        {new Date(f.createTime).toLocaleString("zh-CN")}
                      </Text>
                    </View>
                    {canManage && (
                      <View className="card-actions">
                        {f.status !== "closed" && (
                          <>
                            <View
                              className="action-btn"
                              onClick={() => openReply(f.id, "feedback")}
                            >
                              回复
                            </View>
                            <View
                              className="action-btn danger"
                              onClick={() => openClose(f.id)}
                            >
                              关闭
                            </View>
                          </>
                        )}
                      </View>
                    )}
                  </View>
                ))
              )}
              {feedbacks.length > 0 && feedbacks.length < totalFeedback && (
                <View className="load-more" onClick={handleLoadMoreFeedback}>
                  <Text>加载更多</Text>
                </View>
              )}
            </ScrollView>
          </>
        )}

        {tab === 1 && (
          <ScrollView
            scrollY
            className="list-scroll"
            onScrollToLower={handleLoadMoreRating}
          >
            {ratings.length === 0 ? (
              <Empty>
                <Empty.Description>暂无评价数据</Empty.Description>
              </Empty>
            ) : (
              ratings.map((r) => (
                <View key={r.id} className="list-card">
                  <View className="card-header">
                    <View className="card-title-row">
                      <Text className="card-name">{r.algorithmName}</Text>
                      <Tag size="small" color="warning">
                        {"★".repeat(r.rating)}
                      </Tag>
                    </View>
                    {r.isHidden === 1 && (
                      <Tag size="small" color="default">
                        已隐藏
                      </Tag>
                    )}
                  </View>
                  {r.comment && (
                    <Text className="card-content">{r.comment}</Text>
                  )}
                  <View className="card-meta">
                    <Text className="meta-item">
                      用户: {r.username || `ID:${r.userId}`}
                    </Text>
                    <Text className="meta-item">
                      {new Date(r.createTime).toLocaleString("zh-CN")}
                    </Text>
                  </View>
                  {r.adminReply && (
                    <View className="admin-reply">
                      <Text className="reply-label">管理员回复: </Text>
                      <Text>{r.adminReply}</Text>
                    </View>
                  )}
                  {canManage && (
                    <View className="card-actions">
                      <View
                        className="action-btn"
                        onClick={() => openReply(r.id, "rating")}
                      >
                        回复
                      </View>
                      {r.isHidden !== 1 && (
                        <View
                          className="action-btn danger"
                          onClick={() => handleHideRating(r.id)}
                        >
                          隐藏
                        </View>
                      )}
                    </View>
                  )}
                </View>
              ))
            )}
            {ratings.length > 0 && ratings.length < totalRating && (
              <View className="load-more" onClick={handleLoadMoreRating}>
                <Text>加载更多</Text>
              </View>
            )}
          </ScrollView>
        )}

        {/* 回复弹窗 */}
        <Popup
          open={replyPopupVisible}
          placement="bottom"
          rounded
          onClose={() => setReplyPopupVisible(false)}
        >
          <View className="popup-content">
            <View className="popup-header">
              <Text className="popup-title">回复</Text>
              <Text
                className="popup-close"
                onClick={() => setReplyPopupVisible(false)}
              >
                ×
              </Text>
            </View>
            <View className="popup-body">
              <View className="form-item">
                <Text className="form-label">回复内容</Text>
                <Textarea
                  className="form-textarea"
                  placeholder="请输入回复内容"
                  value={replyContent}
                  onChange={(e) =>
                    setReplyContent(e.detail?.value || "")
                  }
                  maxlength={2000}
                />
              </View>
              <View className="popup-confirm-btn" onClick={handleReply}>
                <Text>提交回复</Text>
              </View>
            </View>
          </View>
        </Popup>

        {/* 关闭反馈弹窗 */}
        <Popup
          open={closePopupVisible}
          placement="bottom"
          rounded
          onClose={() => setClosePopupVisible(false)}
        >
          <View className="popup-content">
            <View className="popup-header">
              <Text className="popup-title">关闭反馈</Text>
              <Text
                className="popup-close"
                onClick={() => setClosePopupVisible(false)}
              >
                ×
              </Text>
            </View>
            <View className="popup-body">
              <View className="form-item">
                <Text className="form-label">关闭原因 *</Text>
                <Textarea
                  className="form-textarea"
                  placeholder="请填写关闭原因（必填）"
                  value={closeReason}
                  onChange={(e) =>
                    setCloseReason(e.detail?.value || "")
                  }
                  maxlength={500}
                />
              </View>
              <View className="popup-confirm-btn" onClick={handleCloseFeedback}>
                <Text>确认关闭</Text>
              </View>
            </View>
          </View>
        </Popup>
      </View>
    </PageLayout>
  );
};

export default FeedbackManagePage;
