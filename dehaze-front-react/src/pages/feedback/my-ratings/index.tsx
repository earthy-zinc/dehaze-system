import { FeedbackAPI, type MyRatingVO } from "dehaze-sdk-js";
import {
  Card,
  Empty,
  Image,
  Pagination,
  Rate,
  Spin,
  Tag,
  Typography,
} from "antd";
import { MessageOutlined, StarOutlined } from "@ant-design/icons";
import React, { useCallback, useEffect, useState } from "react";
import "./index.scss";

const { Paragraph } = Typography;

const PAGE_SIZE = 10;

const MyRatings: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [list, setList] = useState<MyRatingVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);

  const loadData = useCallback(async (page: number) => {
    setLoading(true);
    try {
      const result = await FeedbackAPI.listMyRatings({
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

  return (
    <div className="my-ratings-container">
      <div className="page-header">
        <span className="title-text">我的评价</span>
      </div>

      <Spin spinning={loading}>
        <div className="rating-list">
          {list.length > 0
            ? list.map((rating) => (
                <div key={rating.id} className="rating-card">
                  <div className="card-header">
                    <div className="header-left">
                      <StarOutlined className="algo-icon" />
                      <span className="algo-name">{rating.algorithmName}</span>
                    </div>
                    <div className="header-right">
                      <Rate disabled value={rating.rating} />
                      <span className="time-text">{rating.createTime}</span>
                    </div>
                  </div>

                  {rating.comment ? (
                    <Paragraph
                      className="card-comment"
                      style={{ whiteSpace: "pre-wrap" }}
                    >
                      {rating.comment}
                    </Paragraph>
                  ) : null}

                  {rating.tags?.length ? (
                    <div className="card-tags">
                      {rating.tags.map((tag) => (
                        <Tag key={tag} color="blue">
                          {tag}
                        </Tag>
                      ))}
                    </div>
                  ) : null}

                  {rating.imageUrls?.length ? (
                    <div className="card-images">
                      {rating.imageUrls.map((url, idx) => (
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

                  {rating.adminReply ? (
                    <Card size="small" className="reply-card">
                      <div className="reply-header">
                        <MessageOutlined />
                        <span>管理员回复</span>
                        {rating.replyTime && (
                          <span className="reply-time">{rating.replyTime}</span>
                        )}
                      </div>
                      <Paragraph
                        className="reply-content"
                        style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}
                      >
                        {rating.adminReply}
                      </Paragraph>
                    </Card>
                  ) : null}
                </div>
              ))
            : !loading && (
                <Empty
                  description="暂无评价"
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
    </div>
  );
};

export default MyRatings;
