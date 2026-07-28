import {
  type FeedbackDetailVO,
  type FeedbackPageVO,
  type FeedbackReplyType,
  type FeedbackStatus,
  type FeedbackType,
  type ReplierType,
} from "dehaze-sdk-js";
import {
  Card,
  Descriptions,
  Drawer,
  Empty,
  Image,
  Space,
  Spin,
  Tabs,
  Tag,
  Timeline,
  Typography,
} from "antd";
import React, {
  forwardRef,
  useCallback,
  useImperativeHandle,
  useState,
} from "react";
import { FeedbackAPI } from "dehaze-sdk-js";

const { Paragraph } = Typography;

const TYPE_LABEL: Record<FeedbackType, string> = {
  suggestion: "功能建议",
  bug: "问题报告",
  experience: "体验反馈",
  complaint: "投诉",
};
const TYPE_COLOR: Record<FeedbackType, string> = {
  suggestion: "blue",
  bug: "red",
  experience: "green",
  complaint: "orange",
};
const STATUS_LABEL: Record<FeedbackStatus, string> = {
  pending: "待处理",
  processing: "处理中",
  replied: "已回复",
  closed: "已关闭",
};
const STATUS_COLOR: Record<FeedbackStatus, string> = {
  pending: "orange",
  processing: "blue",
  replied: "green",
  closed: "default",
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

export interface FeedbackDetailDrawerRef {
  open: (record: FeedbackPageVO) => void;
}

const FeedbackDetailDrawer = forwardRef<FeedbackDetailDrawerRef>((_, ref) => {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [detail, setDetail] = useState<FeedbackDetailVO | null>(null);

  const openDrawer = useCallback(async (record: FeedbackPageVO) => {
    setOpen(true);
    setDetail(null);
    setLoading(true);
    try {
      const data = await FeedbackAPI.getFeedbackDetail(record.id);
      setDetail(data);
    } catch {
      setDetail(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useImperativeHandle(ref, () => ({ open: openDrawer }), [openDrawer]);

  const handleClose = () => {
    setOpen(false);
    setDetail(null);
  };

  return (
    <Drawer
      title="反馈详情"
      open={open}
      onClose={handleClose}
      width={780}
      destroyOnClose
    >
      <Spin spinning={loading}>
        {!detail ? (
          <Empty description="暂无数据" />
        ) : (
          <Tabs
            items={[
              {
                key: "basic",
                label: "基本信息",
                children: (
                  <Descriptions column={2} size="small" bordered>
                    <Descriptions.Item label="标题">
                      {detail.title}
                    </Descriptions.Item>
                    <Descriptions.Item label="类型">
                      <Tag color={TYPE_COLOR[detail.feedbackType]}>
                        {TYPE_LABEL[detail.feedbackType]}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="状态">
                      <Tag color={STATUS_COLOR[detail.status]}>
                        {STATUS_LABEL[detail.status]}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="优先级">
                      <Tag>{priorityLabel(detail.priority)}</Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="模块">
                      {detail.relatedModule || "-"}
                    </Descriptions.Item>
                    <Descriptions.Item label="处理人">
                      {detail.assigneeName || "未分配"}
                    </Descriptions.Item>
                    <Descriptions.Item label="提交时间">
                      {detail.createTime}
                    </Descriptions.Item>
                    <Descriptions.Item label="联系方式">
                      {detail.contact ? <Tag>{detail.contact}</Tag> : "-"}
                    </Descriptions.Item>
                    <Descriptions.Item label="标签" span={2}>
                      {detail.tags?.length ? (
                        <Space wrap>
                          {detail.tags.map((tag) => (
                            <Tag key={tag}>{tag}</Tag>
                          ))}
                        </Space>
                      ) : (
                        "-"
                      )}
                    </Descriptions.Item>
                  </Descriptions>
                ),
              },
              {
                key: "content",
                label: "内容与回复",
                children: (
                  <>
                    <Card
                      size="small"
                      title="反馈内容"
                      style={{ marginBottom: 16 }}
                    >
                      <Paragraph
                        style={{ whiteSpace: "pre-wrap", marginBottom: 12 }}
                      >
                        {detail.content}
                      </Paragraph>
                      {detail.images?.length ? (
                        <Space wrap>
                          {detail.images.map((img, idx) => (
                            <Image
                              key={idx}
                              src={img}
                              width={100}
                              height={100}
                              style={{
                                objectFit: "cover",
                                borderRadius: 6,
                              }}
                            />
                          ))}
                        </Space>
                      ) : null}
                    </Card>

                    {detail.replies?.length ? (
                      <Timeline
                        style={{ marginTop: 16 }}
                        items={detail.replies.map((reply) => ({
                          key: reply.id,
                          label: reply.createTime,
                          color:
                            reply.replierType === 2 ? "#409eff" : "#67c23a",
                          children: (
                            <div>
                              <div
                                style={{
                                  display: "flex",
                                  gap: 8,
                                  alignItems: "center",
                                  marginBottom: 6,
                                }}
                              >
                                <span style={{ fontWeight: 600 }}>
                                  {reply.replierName}
                                </span>
                                {reply.replyType && (
                                  <Tag
                                    color={REPLY_TYPE_COLOR[reply.replyType]}
                                  >
                                    {REPLY_TYPE_LABEL[reply.replyType]}
                                  </Tag>
                                )}
                                <span
                                  style={{
                                    fontSize: 12,
                                    color: "var(--ant-color-text-secondary)",
                                  }}
                                >
                                  {replierTypeLabel(reply.replierType)}
                                </span>
                              </div>
                              <Paragraph
                                style={{
                                  whiteSpace: "pre-wrap",
                                  marginBottom: 8,
                                }}
                              >
                                {reply.content}
                              </Paragraph>
                              {reply.attachments?.length ? (
                                <Space wrap>
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
                                </Space>
                              ) : null}
                            </div>
                          ),
                        }))}
                      />
                    ) : null}

                    {detail.closeReason ? (
                      <Card
                        size="small"
                        title="关闭原因"
                        style={{ marginTop: 16 }}
                      >
                        <Paragraph
                          style={{
                            whiteSpace: "pre-wrap",
                            marginBottom: 0,
                          }}
                        >
                          {detail.closeReason}
                        </Paragraph>
                      </Card>
                    ) : null}
                  </>
                ),
              },
            ]}
          />
        )}
      </Spin>
    </Drawer>
  );
});

FeedbackDetailDrawer.displayName = "FeedbackDetailDrawer";

function priorityLabel(priority: number): string {
  const map: Record<number, string> = {
    1: "低",
    2: "中",
    3: "高",
    4: "紧急",
  };
  return map[priority] || String(priority);
}

function replierTypeLabel(type: ReplierType): string {
  return type === 2 ? "管理员" : "用户";
}

export default FeedbackDetailDrawer;
