import {
  FeedbackAPI,
  type FeedbackCreateForm,
  type FeedbackPageVO,
  type FeedbackStatus,
  type FeedbackType,
} from "dehaze-sdk-js";
import { Button, Empty, Pagination, Spin } from "antd";
import { PlusOutlined, RightOutlined } from "@ant-design/icons";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import FeedbackFormDialog, {
  type FeedbackFormDialogRef,
} from "./components/FeedbackFormDialog";
import "./index.scss";

const PAGE_SIZE = 10;

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

const MyFeedback: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [list, setList] = useState<FeedbackPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);

  const formDialogRef = useRef<FeedbackFormDialogRef>(null);
  const navigate = useNavigate();

  const loadData = useCallback(async (page: number) => {
    setLoading(true);
    try {
      const result = await FeedbackAPI.listMyFeedback({
        pageNum: page,
        pageSize: PAGE_SIZE,
      });
      setList(result.list || []);
      setTotal(result.total || 0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(pageNum);
  }, [pageNum, loadData]);

  const handlePageChange = (page: number) => {
    setPageNum(page);
  };

  const handleOpenCreate = () => {
    formDialogRef.current?.open();
  };

  const handleCreateSuccess = () => {
    setPageNum(1);
    loadData(1);
  };

  const handleGoDetail = (record: FeedbackPageVO) => {
    navigate(`/feedback/detail?id=${record.id}`);
  };

  return (
    <div className="my-feedback-container">
      <div className="page-header">
        <span className="title-text">我的反馈</span>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={handleOpenCreate}
        >
          新建反馈
        </Button>
      </div>

      <Spin spinning={loading}>
        <div className="feedback-list">
          {list.length > 0
            ? list.map((feedback) => (
                <div
                  key={feedback.id}
                  className={`feedback-card status-${feedback.status}`}
                  onClick={() => handleGoDetail(feedback)}
                >
                  <div
                    className="status-stripe"
                    style={{ background: STRIPE_COLOR[feedback.status] }}
                  />
                  <div className="card-body">
                    <div className="card-meta">
                      <span
                        className="status-tag"
                        style={{
                          color: STATUS_COLOR[feedback.status],
                          background: STATUS_BG[feedback.status],
                        }}
                      >
                        {STATUS_LABEL[feedback.status]}
                      </span>
                      <span className="type-tag">
                        {TYPE_LABEL[feedback.feedbackType]}
                      </span>
                      {feedback.relatedModule && (
                        <span className="module-tag">
                          {feedback.relatedModule}
                        </span>
                      )}
                      <span className="time-text">{feedback.createTime}</span>
                    </div>

                    <div className="card-title">{feedback.title}</div>
                    <div className="card-summary">{feedback.content}</div>

                    <div className="card-footer">
                      <span
                        className={`assignee-text ${
                          feedback.assigneeName ? "" : "unassigned"
                        }`}
                      >
                        {feedback.assigneeName
                          ? `处理人：${feedback.assigneeName}`
                          : "暂未分配"}
                      </span>
                      <Button
                        type="link"
                        size="small"
                        className="detail-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleGoDetail(feedback);
                        }}
                      >
                        查看详情
                        <RightOutlined />
                      </Button>
                    </div>
                  </div>
                </div>
              ))
            : !loading && (
                <Empty
                  description={
                    <span className="empty-text">
                      有任何建议或问题，欢迎提交反馈
                    </span>
                  }
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                />
              )}
        </div>

        {total > 0 && (
          <div className="pagination-wrapper">
            <Pagination
              current={pageNum}
              pageSize={PAGE_SIZE}
              total={total}
              onChange={handlePageChange}
              showTotal={(t) => `共 ${t} 条`}
              showSizeChanger={false}
            />
          </div>
        )}
      </Spin>

      <FeedbackFormDialog ref={formDialogRef} onSuccess={handleCreateSuccess} />
    </div>
  );
};

export default MyFeedback;
