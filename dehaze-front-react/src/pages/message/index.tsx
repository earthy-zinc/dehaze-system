import { MessageAPI, type MessageVO, type MessageQuery } from "dehaze-sdk-js";
import {
  ArrowRightOutlined,
  CheckOutlined,
  DeleteOutlined,
  SearchOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import { Button, Empty, Input, Modal, Spin, message } from "antd";
import React, { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./index.scss";

type TabValue = string | "unread" | null;

const TYPE_TABS: { label: string; value: TabValue }[] = [
  { label: "全部", value: null },
  { label: "系统公告", value: "announcement" },
  { label: "业务通知", value: "business" },
  { label: "会员通知", value: "member" },
  { label: "未读", value: "unread" },
];

function formatTime(time: string) {
  if (!time) return "";
  const date = new Date(time.replace(/-/g, "/"));
  const now = new Date();
  const isSameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  const hh = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  if (isSameDay) return `${hh}:${mm}`;
  const M = String(date.getMonth() + 1).padStart(2, "0");
  const D = String(date.getDate()).padStart(2, "0");
  return `${M}-${D}`;
}

const MessageCenter: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [messageList, setMessageList] = useState<MessageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [unreadCount, setUnreadCount] = useState(0);
  const [activeTab, setActiveTab] = useState<TabValue>(null);
  const [searchKeyword, setSearchKeyword] = useState("");
  const [pageNum, setPageNum] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [refreshFlag, setRefreshFlag] = useState(0);

  const loadUnreadCount = useCallback(() => {
    MessageAPI.getUnreadCount()
      .then((res) => setUnreadCount(res.count ?? 0))
      .catch(() => {});
  }, []);

  const loadList = useCallback(() => {
    setLoading(true);
    if (searchKeyword.trim()) {
      MessageAPI.search({
        keyword: searchKeyword.trim(),
        pageNum,
        pageSize,
      })
        .then((data) => {
          setMessageList(data.list || []);
          setTotal(data.total || 0);
        })
        .finally(() => setLoading(false));
      return;
    }
    const query: MessageQuery = { pageNum, pageSize };
    if (activeTab === "unread") {
      query.readStatus = 0;
    } else if (activeTab) {
      query.type = activeTab;
    }
    MessageAPI.getPage(query)
      .then((data) => {
        setMessageList(data.list || []);
        setTotal(data.total || 0);
      })
      .finally(() => setLoading(false));
  }, [searchKeyword, activeTab, pageNum, pageSize]);

  useEffect(() => {
    loadList();
  }, [loadList, refreshFlag]);

  useEffect(() => {
    loadUnreadCount();
  }, [loadUnreadCount, refreshFlag]);

  const handleTabChange = useCallback((value: TabValue) => {
    setActiveTab(value);
    setSearchKeyword("");
    setPageNum(1);
  }, []);

  const handleSearch = useCallback(() => {
    setPageNum(1);
    setRefreshFlag((prev) => prev + 1);
  }, []);

  const handlePageChange = useCallback((page: number, size: number) => {
    setPageNum(page);
    setPageSize(size);
  }, []);

  const goDetail = useCallback(
    (msg: MessageVO) => {
      navigate(`/message/detail?id=${msg.id}`);
    },
    [navigate]
  );

  const handleDelete = useCallback((e: React.MouseEvent, msg: MessageVO) => {
    e.stopPropagation();
    Modal.confirm({
      title: "提示",
      content: `确定删除消息「${msg.title}」吗？`,
      okText: "确定",
      cancelText: "取消",
      okType: "danger",
      onOk: () =>
        MessageAPI.deleteByIds(String(msg.id))
          .then(() => {
            message.success("删除成功");
            setRefreshFlag((prev) => prev + 1);
          })
          .catch((err) => {
            message.error(err?.message || "删除失败");
            return Promise.reject(err);
          }),
    });
  }, []);

  const handleMarkAllRead = useCallback(() => {
    if (unreadCount === 0) return;
    Modal.confirm({
      title: "提示",
      content: "确定将所有未读消息标记为已读吗？",
      okText: "确定",
      cancelText: "取消",
      onOk: () =>
        MessageAPI.markAllRead()
          .then((res) => {
            message.success(`已标记 ${res.affectedCount} 条消息为已读`);
            setRefreshFlag((prev) => prev + 1);
          })
          .catch((err) => {
            message.error(err?.message || "操作失败");
            return Promise.reject(err);
          }),
    });
  }, [unreadCount]);

  return (
    <div className="app-container message-center">
      <div className="page-header">
        <div className="header-title">
          <span className="title-text">消息中心</span>
          {unreadCount > 0 && (
            <span className="unread-pill">{unreadCount} 条未读</span>
          )}
        </div>
        <div className="header-actions">
          <Button
            type="primary"
            ghost
            disabled={unreadCount === 0}
            icon={<CheckOutlined />}
            onClick={handleMarkAllRead}
          >
            全部已读
          </Button>
          <Button
            type="link"
            icon={<SettingOutlined />}
            onClick={() => navigate("/message/settings")}
          >
            通知设置
          </Button>
        </div>
      </div>

      <div className="filter-bar">
        <div className="type-tabs">
          {TYPE_TABS.map((tab) => (
            <button
              key={tab.value ?? "all"}
              className={
                "type-tab" + (activeTab === tab.value ? " active" : "")
              }
              onClick={() => handleTabChange(tab.value)}
            >
              <span className="tab-label">{tab.label}</span>
              {tab.value === "unread" && unreadCount > 0 && (
                <span className="tab-count">{unreadCount}</span>
              )}
            </button>
          ))}
        </div>
        <Input
          className="search-input"
          allowClear
          placeholder="搜索消息标题或正文"
          prefix={<SearchOutlined />}
          value={searchKeyword}
          onChange={(e) => setSearchKeyword(e.target.value)}
          onPressEnter={handleSearch}
        />
      </div>

      <Spin spinning={loading}>
        <div className="message-list">
          {messageList.length > 0
            ? messageList.map((msg) => (
                <div
                  key={msg.id}
                  className={
                    "message-card" + (msg.readStatus === 0 ? " unread" : "")
                  }
                  onClick={() => goDetail(msg)}
                >
                  <div className={`type-stripe type-${msg.type}`} />
                  <div className="card-body">
                    <div className="card-meta">
                      <span className={`type-tag tag-${msg.type}`}>
                        {msg.typeLabel}
                      </span>
                      {msg.priority >= 3 && (
                        <span className="priority-flag">
                          {msg.priority === 4 ? "紧急" : "高优"}
                        </span>
                      )}
                      <span className="time-text">
                        {formatTime(msg.createTime)}
                      </span>
                    </div>
                    <div className="card-title">
                      {msg.readStatus === 0 && <span className="unread-dot" />}
                      <span className="title-text">{msg.title}</span>
                    </div>
                    {msg.summary && (
                      <div className="card-summary">{msg.summary}</div>
                    )}
                    <div className="card-footer">
                      {msg.jumpUrl && (
                        <span className="jump-link">
                          点击查看详情
                          <ArrowRightOutlined />
                        </span>
                      )}
                      <Button
                        className="delete-btn"
                        type="link"
                        danger
                        size="small"
                        icon={<DeleteOutlined />}
                        onClick={(e) => handleDelete(e, msg)}
                      >
                        删除
                      </Button>
                    </div>
                  </div>
                </div>
              ))
            : !loading && (
                <Empty
                  description={
                    searchKeyword
                      ? "没有找到匹配的消息"
                      : "所有消息都已处理完毕"
                  }
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                />
              )}
        </div>
      </Spin>

      {total > 0 && (
        <div className="pagination-wrap">
          <Button
            type="link"
            disabled={pageNum <= 1}
            onClick={() => handlePageChange(pageNum - 1, pageSize)}
          >
            上一页
          </Button>
          <span className="page-info">
            第 {pageNum} 页 / 共 {Math.max(1, Math.ceil(total / pageSize))} 页
            （{total} 条）
          </span>
          <Button
            type="link"
            disabled={pageNum >= Math.ceil(total / pageSize)}
            onClick={() => handlePageChange(pageNum + 1, pageSize)}
          >
            下一页
          </Button>
        </div>
      )}
    </div>
  );
};

export default MessageCenter;
