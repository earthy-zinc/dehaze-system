import {
  FavoriteAPI,
  type FavoriteTargetType,
  type FavoriteVO,
} from "dehaze-sdk-js";
import {
  CalendarOutlined,
  DeleteOutlined,
  EyeOutlined,
  HeartOutlined,
  SearchOutlined,
  SortAscendingOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Col,
  Empty,
  Input,
  message,
  Pagination as AntPagination,
  Popconfirm,
  Row,
  Select,
  Space,
  Tag,
  Tabs,
  Typography,
} from "antd";
import React, { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./index.module.scss";

const { Text } = Typography;

interface FavoriteItem extends FavoriteVO {
  isInvalid?: boolean;
}

const TARGET_TYPE_MAP: Record<
  FavoriteTargetType,
  { label: string; color: string }
> = {
  algorithm: { label: "算法", color: "blue" },
  result: { label: "结果", color: "green" },
  dataset: { label: "数据集", color: "orange" },
  image: { label: "图片", color: "purple" },
  preset: { label: "预设", color: "pink" },
};

const TABS: { key: string; label: string; targetType?: FavoriteTargetType }[] =
  [
    { key: "all", label: "全部" },
    { key: "algorithm", label: "算法", targetType: "algorithm" },
    { key: "result", label: "结果", targetType: "result" },
    { key: "dataset", label: "数据集", targetType: "dataset" },
    { key: "image", label: "图片", targetType: "image" },
    { key: "preset", label: "预设", targetType: "preset" },
  ];

export default function MyFavorites() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [favorites, setFavorites] = useState<FavoriteItem[]>([]);
  const [total, setTotal] = useState(0);
  const [currentTab, setCurrentTab] = useState("all");
  const [keywords, setKeywords] = useState("");
  const [sortBy, setSortBy] = useState("createTime");
  const [page, setPage] = useState(1);
  const pageSize = 12;

  const loadData = useCallback(
    async (params: {
      page: number;
      pageSize: number;
      sortBy?: "createTime" | "rating" | "usageCount";
      sortOrder?: "asc" | "desc";
      targetType?: FavoriteTargetType;
      keywords?: string;
    }) => {
      setLoading(true);
      try {
        const result = await FavoriteAPI.getPage(params);
        const items: FavoriteItem[] = (result.list || []).map((item) => ({
          ...item,
          isInvalid: !item.targetId,
        }));
        setFavorites(items);
        setTotal(result.total || 0);
      } catch (err: unknown) {
        const messageErr = err as { message?: string };
        message.error(messageErr.message || "加载收藏列表失败");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const loadFavorites = useCallback(
    (p = 1) => {
      const params: {
        page: number;
        pageSize: number;
        sortBy: "createTime" | "rating" | "usageCount";
        sortOrder: "asc" | "desc";
        targetType?: FavoriteTargetType;
        keywords?: string;
      } = {
        page: p,
        pageSize,
        sortBy: sortBy as "createTime" | "rating" | "usageCount",
        sortOrder: "desc" as const,
      };
      const targetType = TABS.find((t) => t.key === currentTab)?.targetType;
      if (targetType) {
        params.targetType = targetType;
      }
      if (keywords.trim()) {
        params.keywords = keywords.trim();
      }
      loadData(params);
      setPage(p);
    },
    [currentTab, keywords, sortBy, loadData]
  );

  useEffect(() => {
    loadFavorites(1);
  }, [loadFavorites]);

  const handleDelete = useCallback(
    (id: number) => {
      FavoriteAPI.deleteByIds([id])
        .then(() => {
          message.success("已取消收藏");
          loadFavorites(page);
        })
        .catch((err: unknown) => {
          const messageErr = err as { message?: string };
          message.error(messageErr.message || "取消收藏失败");
        });
    },
    [page, loadFavorites]
  );

  const handleCardClick = (item: FavoriteItem) => {
    if (item.isInvalid) return;
    const routeMap: Record<FavoriteTargetType, string> = {
      algorithm: `/algorithm/${item.targetId}`,
      result: `/task/result/${item.targetId}`,
      dataset: `/dataset/${item.targetId}`,
      image: `/image/${item.targetId}`,
      preset: `/presentation/preset/${item.targetId}`,
    };
    const targetType = item.targetType as FavoriteTargetType;
    const path = routeMap[targetType];
    if (path) navigate(path);
  };

  return (
    <div className="my-favorites-page">
      <Card className="filter-card">
        <Row gutter={[16, 16]} align="middle">
          <Col span={8}>
            <Input
              placeholder="搜索收藏名称"
              prefix={<SearchOutlined />}
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              allowClear
            />
          </Col>
          <Col span={6}>
            <Select
              value={sortBy}
              onChange={setSortBy}
              options={[
                { label: "最新收藏", value: "createTime" },
                { label: "评分最高", value: "rating" },
                { label: "使用最多", value: "usageCount" },
              ]}
              style={{ width: "100%" }}
              prefix={<SortAscendingOutlined />}
            />
          </Col>
        </Row>
      </Card>

      <Tabs
        activeKey={currentTab}
        onChange={setCurrentTab}
        className="favorite-tabs"
        items={TABS.map((tab) => ({
          key: tab.key,
          label: (
            <Space>
              <HeartOutlined />
              {tab.label}
            </Space>
          ),
        }))}
      />

      <div className="favorite-grid">
        {favorites.length === 0 && !loading ? (
          <Empty description="暂无收藏内容" style={{ padding: "60px 0" }}>
            <Text type="secondary">
              点击右上角&ldquo;+&rdquo;按钮收藏您喜欢的内容
            </Text>
          </Empty>
        ) : (
          <>
            {favorites.map((item) => (
              <Card
                key={item.id}
                className={`favorite-card ${item.isInvalid ? "invalid" : ""}`}
                hoverable
                onClick={() => handleCardClick(item)}
                cover={
                  item.targetThumbnail ? (
                    <div className="card-cover">
                      <img src={item.targetThumbnail} alt={item.targetName} />
                    </div>
                  ) : null
                }
                actions={[
                  <EyeOutlined key="view" />,
                  <Popconfirm
                    key="delete"
                    title="确认取消收藏？"
                    onConfirm={() => handleDelete(item.id)}
                    okText="确定"
                    cancelText="取消"
                  >
                    <DeleteOutlined style={{ color: "#ff4d4f" }} />
                  </Popconfirm>,
                ]}
              >
                <Card.Meta
                  title={
                    <Space>
                      <Text ellipsis>{item.targetName}</Text>
                      {item.isInvalid && <Tag color="red">已失效</Tag>}
                    </Space>
                  }
                  description={
                    <div>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {TARGET_TYPE_MAP[item.targetType]?.label ||
                          item.targetType}
                      </Text>
                      <div className="meta-info">
                        <CalendarOutlined />
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {new Date(item.createTime).toLocaleDateString()}
                        </Text>
                      </div>
                      {item.targetSummary && (
                        <Text type="secondary" ellipsis>
                          {item.targetSummary}
                        </Text>
                      )}
                    </div>
                  }
                />
              </Card>
            ))}
          </>
        )}
      </div>

      <div className="pagination-wrapper">
        <AntPagination
          current={page}
          total={total}
          pageSize={pageSize}
          onChange={(p) => loadFavorites(p)}
          showSizeChanger
          showTotal={(t) => `共 ${t} 条`}
        />
      </div>
    </div>
  );
}
