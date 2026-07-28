import {
  FeedbackAPI,
  type FeedbackDetailVO,
  type FeedbackReplyType,
  type FeedbackStatus,
  type FeedbackType,
  type ReplierType,
} from "dehaze-sdk-js";
import {
  Button,
  Card,
  Empty,
  Image,
  Input,
  Spin,
  Tag,
  Timeline,
  Typography,
  message,
} from "antd";
import {
  ArrowLeftOutlined,
  CloseCircleOutlined,
  MessageOutlined,
  PictureOutlined,
} from "@ant-design/icons";
import React, { useCallback, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import "./index.scss";

const { Paragraph, Title } = Typography;

const TYPE_LABEL: Record<FeedbackType, string> = {
  suggestion: "功能建议",
  bug: "问题报告",
  experience: "体验反馈",
  complaint: "投诉",
};
const STATUS_LABEL: Record<FeedbackStatus, string> = {
  pending: "待处理",
  processing: "处理中",
  replied: "已回复",
  closed: "已关闭",
};
const STATUS_COLOR: Record<FeedbackStatus, string> = {
  pending: "#fa8c16",
  processing: "#409eff",
  replied: "#67c23a",
  closed: "#909399",
};
const STATUS_BG: Record<FeedbackStatus, string> = {
  pending: "#fff7e6",
  processing: "#ecf5ff",
  replied: "#f0f9eb",
  closed: "#f4f4f5",
};
const STRIPE_COLOR: Record<FeedbackStatus, string> = {
  pending: "linear-gradient(180deg, #fa8c16, #ffc069)",
  processing: "linear-gradient(180deg, #409eff, #79bbff)",
  replied: "linear-gradient(180deg, #67c23a, #95d475)",
  closed: "linear-gradient(180deg, #909399, #b1b3b8)",
};
const REPLY_TYPE_LABEL: Record<FeedbackReplyType, string> = {
  info: "通知",
  resolved: "已解决",
  unsupported: "不支持",
  dev_transfer: "转开发",
};
const REPLY_TYPE_COLOR: Record<FeedbackReplyType, string> = {
  info: "default",
  resolved: "success",
  unsupported: "default",
  dev_transfer: "orange",
};

const FeedbackDetail: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [detail, setDetail] = useState<FeedbackDetailVO | null>(null);
  const [supplementContent, setSupplementContent] = useState("");
  const [supplementLoading, setSupplementLoading] = useState(false);

  const feedbackId = Number(searchParams.get("id"));

  const loadDetail = useCallback(async () => {
    if (!feedbackId) {
      setDetail(null);
      return;
    }
    setLoading(true);
    try {
      const data = await FeedbackAPI.getFeedbackDetail(feedbackId);
      setDetail(data);
    } catch {
      setDetail(null);
    } finally {
      setLoading(false);
    }
  }, [feedbackId]);

  useEffect(() => {
    loadDetail();
  }, [loadDetail]);

  const goBack = () => {
    navigate("/feedback/my");
  };

  const handleSupplement = async () => {
    if (!detail) return;
    const content = supplementContent.trim();
    if (!content) return;
    setSupplementLoading(true);
    try {
      await FeedbackAPI.supplementFeedback(detail.id, { content });
      message.success("补充说明已提交");
      setSupplementContent("");
      loadDetail();
    } catch (error: any) {
      message.error(error?.message || "提交失败");
    } finally {
      setSupplementLoading(false);
    }
  };

  return (
    <div className="feedback-detail-container">
      <Spin spinning={loading}>
        <div className="detail-wrapper">
          {detail ? (
            <>
              <div className="detail-header">
                <Button
                  type="link"
                  icon={<ArrowLeftOutlined />}
                  onClick={goBack}
                >
                  返回列表
                </Button>
              </div>

              <div className={`status-card status-${detail.status}`}>
                <div
                  className="card-stripe"
                  style={{ background: STRIPE_COLOR[detail.status] }}
                />
                <div className="card-content">
                  <div className="meta-row">
                    <span
                      className="status-tag"
                      style={{
                        color: STATUS_COLOR[detail.status],
                        background: STATUS_BG[detail.status],
                      }}
                    >
                      {STATUS_LABEL[detail.status]}
                    </span>
                    <span className="type-tag">
                      {TYPE_LABEL[detail.feedbackType]}
                    </span>
                    {detail.relatedModule && (
                      <span className="module-tag">{detail.relatedModule}</span>
                    )}
                    <span className="time-text">{detail.createTime}</span>
                  </div>

                  <Title level={3} className="detail-title">
                    {detail.title}
                  </Title>

                  {detail.assigneeName && (
                    <div className="assignee-info">
                      <MessageOutlined />
                      处理人：{detail.assigneeName}
                      {detail.assignedTime && (
                        <span className="assigned-time">
                          分配于 {detail.assignedTime}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>

              <Card size="small" className="content-card">
                <div className="card-header-title">
                  <PictureOutlined />
                  <span>反馈内容</span>
                </div>
                <Paragraph
                  className="content-text"
                  style={{ whiteSpace: "pre-wrap" }}
                >
                  {detail.content}
                </Paragraph>

                {detail.images?.length ? (
                  <div className="content-images">
                    {detail.images.map((url, idx) => (
                      <Image
                        key={idx}
                        src={url}
                        width={100}
                        height={100}
                        style={{ objectFit: "cover", borderRadius: 6 }}
                      />
                    ))}
                  </div>
                ) : null}

                {detail.contact && (
                  <div className="content-contact">
                    <Tag>联系方式：{detail.contact}</Tag>
                  </div>
                )}

                {detail.tags?.length ? (
                  <div className="content-tags">
                    {detail.tags.map((tag) => (
                      <Tag key={tag} color="blue">
                        {tag}
                      </Tag>
                    ))}
                  </div>
                ) : null}
              </Card>

              {detail.replies?.length ? (
                <Card size="small" className="timeline-card">
                  <div className="card-header-title">
                    <MessageOutlined />
                    <span>处理时间线</span>
                  </div>
                  <Timeline
                    items={detail.replies.map((reply) => ({
                      key: reply.id,
                      label: reply.createTime,
                      color: reply.replierType === 2 ? "#409eff" : "#67c23a",
                      children: (
                        <div>
                          <div className="reply-meta">
                            <span className="replier-name">
                              {reply.replierName}
                            </span>
                            {reply.replyType && (
                              <Tag color={REPLY_TYPE_COLOR[reply.replyType]}>
                                {REPLY_TYPE_LABEL[reply.replyType]}
                              </Tag>
                            )}
                            <span className="replier-type-text">
                              {replierTypeLabel(reply.replierType)}
                            </span>
                          </div>
                          <Paragraph
                            className="reply-content"
                            style={{ whiteSpace: "pre-wrap" }}
                          >
                            {reply.content}
                          </Paragraph>
                          {reply.attachments?.length ? (
                            <div className="reply-attachments">
                              {reply.attachments.map((att, idx) => (
                                <Image
                                  key={idx}
                                  src={att}
                                  width={80}
                                  height={80}
                                  style={{
                                    objectFit: "cover",
                                    borderRadius: 6,
                                  }}
                                />
                              ))}
                            </div>
                          ) : null}
                        </div>
                      ),
                    }))}
                  />
                </Card>
              ) : null}

              {detail.closeReason ? (
                <Card size="small" className="close-card">
                  <div className="card-header-title close-header">
                    <CloseCircleOutlined />
                    <span>关闭原因</span>
                  </div>
                  <Paragraph
                    className="close-content"
                    style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}
                  >
                    {detail.closeReason}
                  </Paragraph>
                </Card>
              ) : null}

              {detail.status !== "closed" && (
                <Card size="small" className="supplement-card">
                  <div className="card-header-title">
                    <MessageOutlined />
                    <span>补充说明</span>
                  </div>
                  <Input.TextArea
                    value={supplementContent}
                    onChange={(e) => setSupplementContent(e.target.value)}
                    rows={4}
                    maxLength={1000}
                    showCount
                    placeholder="请输入补充说明内容"
                  />
                  <div className="supplement-footer">
                    <Button
                      type="primary"
                      loading={supplementLoading}
                      disabled={!supplementContent.trim()}
                      onClick={handleSupplement}
                    >
                      提交补充
                    </Button>
                  </div>
                </Card>
              )}
            </>
          ) : (
            !loading && (
              <Empty description="反馈不存在或已被删除">
                <Button type="primary" onClick={goBack}>
                  返回反馈列表
                </Button>
              </Empty>
            )
          )}
        </div>
      </Spin>
    </div>
  );
};

function replierTypeLabel(type: ReplierType): string {
  return type === 2 ? "管理员" : "用户";
}

export default FeedbackDetail;
