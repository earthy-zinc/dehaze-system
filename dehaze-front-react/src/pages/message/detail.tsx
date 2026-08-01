import { MessageAPI, type MessageVO } from "dehaze-sdk-js";
import { ArrowLeftOutlined, DeleteOutlined } from "@ant-design/icons";
import { Button, Empty, Modal, Spin, message } from "antd";
import React, { useCallback, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import "./detail.scss";

const MessageDetail: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const id = searchParams.get("id");

  const [loading, setLoading] = useState(false);
  const [messageData, setMessageData] = useState<MessageVO | null>(null);

  const loadDetail = useCallback(() => {
    const numId = Number(id);
    if (!id || isNaN(numId)) {
      setMessageData(null);
      return;
    }
    setLoading(true);
    MessageAPI.getDetail(numId)
      .then((data) => {
        setMessageData(data);
        if (data.readStatus === 0) {
          MessageAPI.markRead(numId).then(() => {
            setMessageData((prev) =>
              prev ? { ...prev, readStatus: 1 } : prev
            );
          });
        }
      })
      .catch(() => setMessageData(null))
      .finally(() => setLoading(false));
  }, [id]);

  useEffect(() => {
    loadDetail();
  }, [loadDetail]);

  const goBack = useCallback(() => {
    navigate("/message");
  }, [navigate]);

  const handleJump = useCallback(() => {
    if (!messageData?.jumpUrl) return;
    navigate(messageData.jumpUrl);
  }, [messageData, navigate]);

  const handleDelete = useCallback(() => {
    if (!messageData) return;
    Modal.confirm({
      title: "提示",
      content: `确定删除消息「${messageData.title}」吗？`,
      okText: "确定",
      cancelText: "取消",
      okType: "danger",
      onOk: () =>
        MessageAPI.deleteByIds(String(messageData.id))
          .then(() => {
            message.success("删除成功");
            goBack();
          })
          .catch((err) => {
            message.error(err?.message || "删除失败");
            return Promise.reject(err);
          }),
    });
  }, [messageData, goBack]);

  return (
    <div className="app-container message-detail">
      <Spin spinning={loading}>
        {messageData ? (
          <div className="detail-wrapper">
            <div className="detail-header">
              <Button type="link" icon={<ArrowLeftOutlined />} onClick={goBack}>
                返回列表
              </Button>
            </div>

            <div className={`detail-card type-${messageData.type}`}>
              <div className="card-stripe" />
              <div className="card-content">
                <div className="meta-row">
                  <span className={`type-tag tag-${messageData.type}`}>
                    {messageData.typeLabel}
                  </span>
                  {messageData.priority >= 3 && (
                    <span className="priority-flag">
                      {messageData.priority === 4 ? "紧急" : "高优"}
                    </span>
                  )}
                  {messageData.senderType && (
                    <span className="sender-text">来自：消息发送方</span>
                  )}
                  <span className="time-text">{messageData.createTime}</span>
                </div>

                <h1 className="detail-title">{messageData.title}</h1>

                {messageData.readStatus === 1 && messageData.readTime && (
                  <div className="read-info">已读于 {messageData.readTime}</div>
                )}

                <div className="detail-divider" />

                <div className="detail-content">{messageData.content}</div>

                {messageData.extra && (
                  <div className="extra-block">
                    <div className="extra-title">附加信息</div>
                    <pre className="extra-content">
                      {JSON.stringify(messageData.extra, null, 2)}
                    </pre>
                  </div>
                )}

                <div className="detail-footer">
                  {messageData.jumpUrl && (
                    <Button type="primary" onClick={handleJump}>
                      查看详情
                    </Button>
                  )}
                  <Button
                    danger
                    ghost
                    icon={<DeleteOutlined />}
                    onClick={handleDelete}
                  >
                    删除消息
                  </Button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          !loading && (
            <Empty description="消息不存在或已被删除">
              <Button type="primary" onClick={goBack}>
                返回消息列表
              </Button>
            </Empty>
          )
        )}
      </Spin>
    </div>
  );
};

export default MessageDetail;
