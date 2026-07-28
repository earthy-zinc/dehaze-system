import { type RatingDetailVO, type RatingPageVO } from "dehaze-sdk-js";
import {
  Card,
  Descriptions,
  Drawer,
  Empty,
  Image,
  Rate,
  Space,
  Tag,
  Typography,
} from "antd";
import React, { forwardRef, useImperativeHandle, useState } from "react";

const { Paragraph } = Typography;

export interface RatingDetailDrawerRef {
  open: (record: RatingPageVO) => void;
}

const RatingDetailDrawer = forwardRef<RatingDetailDrawerRef>((_, ref) => {
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<RatingDetailVO | null>(null);

  const openDrawer = (record: RatingPageVO) => {
    setDetail(record as RatingDetailVO);
    setOpen(true);
  };

  useImperativeHandle(ref, () => ({ open: openDrawer }), [openDrawer]);

  const handleClose = () => {
    setOpen(false);
    setDetail(null);
  };

  return (
    <Drawer
      title="评价详情"
      open={open}
      onClose={handleClose}
      width={640}
      destroyOnClose
    >
      {!detail ? (
        <Empty description="暂无数据" />
      ) : (
        <>
          <Card size="small" title="基本信息" className="detail-section">
            <Descriptions column={2} size="small" bordered>
              <Descriptions.Item label="用户名">
                {detail.isAnonymous === 1 ? "匿名用户" : detail.username || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="算法">
                {detail.algorithmName}
              </Descriptions.Item>
              <Descriptions.Item label="评分">
                <Rate disabled value={detail.rating} />
              </Descriptions.Item>
              <Descriptions.Item label="评价时间">
                {detail.createTime}
              </Descriptions.Item>
              <Descriptions.Item label="是否匿名">
                {detail.isAnonymous === 1 ? "是" : "否"}
              </Descriptions.Item>
              <Descriptions.Item label="是否隐藏">
                {detail.isHidden === 1 ? "是" : "否"}
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
          </Card>

          <Card size="small" title="评价内容" className="detail-section">
            <Paragraph style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}>
              {detail.comment || "无"}
            </Paragraph>
          </Card>

          {detail.imageUrls?.length ? (
            <Card size="small" title="图片预览" className="detail-section">
              <Space wrap>
                {detail.imageUrls.map((url, idx) => (
                  <Image
                    key={idx}
                    src={url}
                    width={100}
                    height={100}
                    style={{ objectFit: "cover", borderRadius: 6 }}
                  />
                ))}
              </Space>
            </Card>
          ) : null}

          {detail.adminReply ? (
            <Card size="small" title="管理员回复" className="detail-section">
              <Paragraph style={{ whiteSpace: "pre-wrap", marginBottom: 4 }}>
                {detail.adminReply}
              </Paragraph>
              {detail.replyTime && (
                <div className="reply-time">回复时间：{detail.replyTime}</div>
              )}
            </Card>
          ) : null}
        </>
      )}
    </Drawer>
  );
});

RatingDetailDrawer.displayName = "RatingDetailDrawer";

export default RatingDetailDrawer;
