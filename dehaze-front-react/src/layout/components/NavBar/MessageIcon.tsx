import { MessageAPI, type MessageVO } from "dehaze-sdk-js";
import {
  BellOutlined,
  CheckOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import { Badge, Button, Empty, Popover, Spin, message } from "antd";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./MessageIcon.scss";

const POLL_INTERVAL = 60_000;

const MessageIcon: React.FC = () => {
  const navigate = useNavigate();
  const [unreadCount, setUnreadCount] = useState(0);
  const [recentList, setRecentList] = useState<MessageVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchUnreadCount = useCallback(() => {
    MessageAPI.getUnreadCount()
      .then((res) => setUnreadCount(res.count ?? 0))
      .catch(() => {});
  }, []);

  const fetchRecentList = useCallback(() => {
    setLoading(true);
    MessageAPI.getPage({ pageNum: 1, pageSize: 5, readStatus: 0 })
      .then((data) => setRecentList(data.list || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchUnreadCount();
    pollRef.current = setInterval(fetchUnreadCount, POLL_INTERVAL);

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        fetchUnreadCount();
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [fetchUnreadCount]);

  const handleOpenChange = useCallback(
    (visible: boolean) => {
      setOpen(visible);
      if (visible) {
        fetchRecentList();
      }
    },
    [fetchRecentList]
  );

  const handleMarkAllRead = useCallback(() => {
    MessageAPI.markAllRead()
      .then((res) => {
        message.success(`已标记 ${res.affectedCount} 条消息为已读`);
        setUnreadCount(0);
        setRecentList([]);
        setOpen(false);
      })
      .catch((err) => message.error(err?.message || "操作失败"));
  }, []);

  const handleViewDetail = useCallback(
    (msg: MessageVO) => {
      setOpen(false);
      navigate(`/message/detail?id=${msg.id}`);
    },
    [navigate]
  );

  const content = (
    <div className="message-popover">
      <Spin spinning={loading}>
        {recentList.length > 0 ? (
          <div className="recent-list">
            {recentList.map((msg) => (
              <div
                key={msg.id}
                className="recent-item"
                onClick={() => handleViewDetail(msg)}
              >
                <div className="item-meta">
                  <span className={`type-tag tag-${msg.type}`}>
                    {msg.typeLabel}
                  </span>
                  <span className="item-time">{msg.createTime}</span>
                </div>
                <div className="item-title">{msg.title}</div>
                {msg.summary && (
                  <div className="item-summary">{msg.summary}</div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <Empty
            description="暂无未读消息"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        )}
      </Spin>
      <div className="popover-footer">
        <Button
          type="link"
          size="small"
          icon={<SettingOutlined />}
          onClick={() => {
            setOpen(false);
            navigate("/message/settings");
          }}
        >
          通知设置
        </Button>
        <Button
          type="link"
          size="small"
          icon={<CheckOutlined />}
          disabled={unreadCount === 0}
          onClick={handleMarkAllRead}
        >
          全部已读
        </Button>
        <Button
          type="link"
          size="small"
          onClick={() => {
            setOpen(false);
            navigate("/message");
          }}
        >
          查看全部
        </Button>
      </div>
    </div>
  );

  return (
    <Popover
      content={content}
      trigger="click"
      placement="bottomRight"
      open={open}
      onOpenChange={handleOpenChange}
      overlayClassName="message-popover-wrapper"
    >
      <Badge count={unreadCount} overflowCount={99} size="small">
        <button className="menu-status-icon message-icon-btn">
          <BellOutlined />
        </button>
      </Badge>
    </Popover>
  );
};

export { MessageIcon };
