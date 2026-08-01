import { AlgorithmAPI, RecommendationAPI, type Algorithm } from "dehaze-sdk-js";
import { useDebounceFn } from "ahooks";
import {
  CheckCircleFilled,
  FireOutlined,
  ReloadOutlined,
  SearchOutlined,
  ThunderboltFilled,
  PictureOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Col,
  Divider,
  Empty,
  Form,
  Input,
  Progress,
  Rate,
  Row,
  Space,
  Spin,
  Statistic,
  Tabs,
  Tag,
  Tree,
  Typography,
  message,
} from "antd";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

const { Title, Paragraph, Text } = Typography;

/** 算法类型分类配置 */
const TYPE_CATEGORIES: { key: string; label: string }[] = [
  { key: "traditional", label: "传统算法" },
  { key: "deep_learning", label: "深度学习算法" },
  { key: "hybrid", label: "混合算法" },
];

/** 推荐优先级：深度学习 > 混合 > 传统 */
const RECOMMEND_ORDER = ["deep_learning", "hybrid", "traditional"];

/** 类型标签颜色映射 */
const TYPE_COLOR_MAP: Record<string, string> = {
  traditional: "blue",
  deep_learning: "volcano",
  hybrid: "cyan",
};

/** 递归提取叶子算法（无 children 的节点） */
function flattenAlgorithms(list: Algorithm[]): Algorithm[] {
  const result: Algorithm[] = [];
  const walk = (nodes: Algorithm[]) => {
    for (const node of nodes) {
      if (node.children?.length) {
        walk(node.children);
      } else {
        result.push(node);
      }
    }
  };
  walk(list);
  return result;
}

export default function AlgorithmSelect(): React.JSX.Element {
  const navigate = useNavigate();
  const [algorithmTree, setAlgorithmTree] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedType, setSelectedType] = useState<string>("all");
  const [keywords, setKeywords] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);

  // ==================== 智能推荐 - 图像分析 ====================

  /** 图像分析结果 */
  interface ImageAnalysisResult {
    hazeLevel: string;
    sceneType: string;
    lighting: string;
    complexity: string;
  }

  const [analysisResult, setAnalysisResult] =
    useState<ImageAnalysisResult | null>(null);
  const [imageUrlInput, setImageUrlInput] = useState("");
  const [imageMd5, setImageMd5] = useState<string>("");
  const [analyzing, setAnalyzing] = useState(false);
  const [recommendations, setRecommendations] = useState<
    Array<{
      algorithm: Algorithm;
      matchScore: number;
      reason: string;
      effectDescription: string;
    }>
  >([]);
  const [analyzingRecs, setAnalyzingRecs] = useState(false);

  /** 分析图像特征 */
  const handleAnalyzeImage = useCallback(async (imageUrl: string) => {
    if (!imageUrl.trim()) {
      message.warning("请输入图片地址");
      return;
    }
    setAnalyzing(true);
    try {
      const analysis = await RecommendationAPI.analyze({
        imageUrl: imageUrl.trim(),
      });
      setAnalysisResult({
        hazeLevel: String(analysis.hazeLevel) || "未知",
        sceneType: String(analysis.sceneType) || "未知",
        lighting: String(analysis.lighting) || "未知",
        complexity: String(analysis.complexity) || "未知",
      });
      setImageMd5(analysis.imageMd5 || "");
      // 获取推荐算法
      if (analysis.imageMd5) {
        setAnalyzingRecs(true);
        const recs = await RecommendationAPI.getAlgorithmRecommendations({
          imageMd5: analysis.imageMd5,
        });
        const mapped = (recs || []).slice(0, 3).map((r) => ({
          algorithm: {
            id: r.algorithmId,
            name: r.algorithmName,
            type: "deep_learning" as const,
            description: r.effectDescription || "",
            status: 1,
          } as Algorithm,
          matchScore: r.matchScore || 0,
          reason: r.reason || "",
          effectDescription: r.effectDescription || "",
        }));
        setRecommendations(mapped);
      }
    } catch (error: any) {
      message.error(error?.message || "图像分析失败");
    } finally {
      setAnalyzing(false);
      setAnalyzingRecs(false);
    }
  }, []);

  /** 选择推荐算法 */
  const handleSelectRecommendation = useCallback((algo: Algorithm) => {
    setSelectedId(algo.id);
    message.success(`已选择 ${algo.name}`);
  }, []);

  // ==================== 数据加载 ====================

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const data = await AlgorithmAPI.getList();
      setAlgorithmTree(Array.isArray(data) ? data : []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // ==================== 派生数据 ====================

  /** 所有叶子算法 */
  const allAlgorithms = useMemo(
    () => flattenAlgorithms(algorithmTree),
    [algorithmTree]
  );

  /** 按类型统计数量 */
  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const a of allAlgorithms) {
      counts[a.type] = (counts[a.type] || 0) + 1;
    }
    return counts;
  }, [allAlgorithms]);

  /** 分类树数据 */
  const treeData = useMemo(() => {
    const nodes = TYPE_CATEGORIES.map((c) => ({
      key: c.key,
      title: `${c.label}（${typeCounts[c.key] || 0}）`,
    }));
    return [
      { key: "all", title: `全部算法（${allAlgorithms.length}）` },
      ...nodes,
    ];
  }, [allAlgorithms.length, typeCounts]);

  /** 智能推荐 Top 3（基于算法类型推荐，取已启用算法） */
  const recommendList = useMemo(() => {
    const enabled = allAlgorithms.filter((a) => a.status === 1);
    const ordered: Algorithm[] = [];
    for (const type of RECOMMEND_ORDER) {
      for (const a of enabled) {
        if (ordered.length >= 3) break;
        if (a.type === type) ordered.push(a);
      }
      if (ordered.length >= 3) break;
    }
    return ordered.slice(0, 3);
  }, [allAlgorithms]);

  /** 当前展示的算法列表（按分类 + 关键词过滤） */
  const displayList = useMemo(() => {
    let list = allAlgorithms;
    if (selectedType !== "all") {
      list = list.filter((a) => a.type === selectedType);
    }
    const kw = keywords.trim().toLowerCase();
    if (kw) {
      list = list.filter(
        (a) =>
          a.name.toLowerCase().includes(kw) ||
          (a.description || "").toLowerCase().includes(kw) ||
          (a.type || "").toLowerCase().includes(kw)
      );
    }
    return list;
  }, [allAlgorithms, selectedType, keywords]);

  // ==================== 搜索防抖（300ms） ====================

  const { run: debouncedSearch } = useDebounceFn(
    (value: string) => setKeywords(value),
    { wait: 300 }
  );

  // ==================== 事件处理 ====================

  const handleTreeSelect = useCallback((keys: React.Key[]) => {
    setSelectedType((keys[0] as string) || "all");
    setSelectedId(null);
  }, []);

  const handleReset = useCallback(() => {
    setKeywords("");
    setSelectedType("all");
    setSelectedId(null);
  }, []);

  /** 确认选择，跳转回图像去雾页并传递算法参数 */
  const handleConfirm = useCallback(() => {
    const selected = allAlgorithms.find((a) => a.id === selectedId);
    if (!selected) {
      message.warning("请先选择一个算法");
      return;
    }
    navigate("/presentation/dehaze", {
      state: {
        algorithmId: selected.id,
        algorithmName: selected.name,
        algorithmType: selected.type,
        importPath: selected.importPath,
      },
    });
  }, [allAlgorithms, selectedId, navigate]);

  // ==================== 渲染 ====================

  return (
    <div className="app-container">
      {/* 智能推荐区域 */}
      <Card size="small" style={{ marginBottom: 12 }} loading={loading}>
        <Title level={5} style={{ marginTop: 0, marginBottom: 12 }}>
          <ThunderboltFilled style={{ color: "#faad14", marginRight: 8 }} />
          智能推荐
        </Title>
        <Tabs
          defaultActiveKey="quick"
          items={[
            {
              key: "quick",
              label: (
                <Space>
                  <ThunderboltFilled style={{ color: "#faad14" }} />
                  快速推荐
                </Space>
              ),
              children:
                recommendList.length === 0 ? (
                  <Empty description="暂无推荐算法" style={{ padding: 24 }} />
                ) : (
                  <Row gutter={[12, 12]}>
                    {recommendList.map((algo) => (
                      <Col key={algo.id} span={8}>
                        <Card
                          size="small"
                          hoverable
                          style={{
                            borderColor:
                              selectedId === algo.id ? "#3B82F6" : undefined,
                            background:
                              selectedId === algo.id ? "#eff6ff" : undefined,
                          }}
                          onClick={() => setSelectedId(algo.id)}
                        >
                          <Space
                            direction="vertical"
                            size={4}
                            style={{ width: "100%" }}
                          >
                            <Space>
                              {selectedId === algo.id && (
                                <CheckCircleFilled
                                  style={{ color: "#3B82F6" }}
                                />
                              )}
                              <Text strong>{algo.name}</Text>
                              <Tag color="orange">推荐</Tag>
                            </Space>
                            <Space size={4}>
                              <Tag>{algo.type}</Tag>
                              <Tag
                                color={algo.status === 1 ? "green" : "default"}
                              >
                                {algo.status === 1 ? "启用" : "禁用"}
                              </Tag>
                            </Space>
                            <Paragraph
                              type="secondary"
                              ellipsis={{ rows: 2 }}
                              style={{ marginBottom: 0 }}
                            >
                              {algo.description || "暂无描述"}
                            </Paragraph>
                          </Space>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                ),
            },
            {
              key: "image",
              label: (
                <Space>
                  <PictureOutlined />
                  图像分析推荐
                </Space>
              ),
              children: (
                <Card size="small">
                  {/* 图像分析输入区 */}
                  <Form layout="inline" style={{ marginBottom: 16 }}>
                    <Form.Item label="图片地址">
                      <Input
                        placeholder="请输入图片 URL"
                        prefix={<FireOutlined />}
                        value={imageUrlInput}
                        disabled={!!analysisResult}
                        onChange={(e) => setImageUrlInput(e.target.value)}
                        onPressEnter={() => handleAnalyzeImage(imageUrlInput)}
                        style={{ width: 320 }}
                      />
                    </Form.Item>
                    <Form.Item>
                      <Button
                        type="primary"
                        icon={<SearchOutlined />}
                        onClick={() => handleAnalyzeImage(imageUrlInput)}
                        loading={analyzing}
                      >
                        分析
                      </Button>
                    </Form.Item>
                  </Form>

                  {/* 分析结果 */}
                  {analyzing && (
                    <Spin
                      tip="正在分析图像特征..."
                      style={{ display: "block", marginBottom: 16 }}
                    />
                  )}

                  {analysisResult && !analyzing && (
                    <>
                      <Card size="small" style={{ marginBottom: 16 }}>
                        <Title
                          level={5}
                          style={{ marginTop: 0, marginBottom: 8 }}
                        >
                          图像特征分析
                        </Title>
                        <Row gutter={[16, 8]}>
                          <Col span={6}>
                            <Statistic
                              title="雾霾程度"
                              value={analysisResult.hazeLevel}
                              suffix=""
                            />
                          </Col>
                          <Col span={6}>
                            <Statistic
                              title="场景类型"
                              value={analysisResult.sceneType}
                              suffix=""
                            />
                          </Col>
                          <Col span={6}>
                            <Statistic
                              title="光照条件"
                              value={analysisResult.lighting}
                              suffix=""
                            />
                          </Col>
                          <Col span={6}>
                            <Statistic
                              title="复杂度"
                              value={analysisResult.complexity}
                              suffix=""
                            />
                          </Col>
                        </Row>
                      </Card>

                      {/* 推荐算法列表 */}
                      <Title level={5} style={{ marginBottom: 8 }}>
                        推荐算法
                      </Title>
                      {analyzingRecs ? (
                        <Spin
                          tip="正在获取推荐..."
                          style={{ display: "block", padding: 16 }}
                        />
                      ) : recommendations.length === 0 ? (
                        <Empty
                          description="暂无推荐结果"
                          image={Empty.PRESENTED_IMAGE_SIMPLE}
                        />
                      ) : (
                        <Row gutter={[12, 12]}>
                          {recommendations.map((rec, idx) => (
                            <Col key={idx} span={24}>
                              <Card
                                size="small"
                                style={{
                                  borderLeft: `3px solid ${rec.matchScore >= 80 ? "#52c41a" : rec.matchScore >= 60 ? "#faad14" : "#f5222d"}`,
                                }}
                              >
                                <Row gutter={12} align="middle">
                                  <Col span={16}>
                                    <Space direction="vertical" size={4}>
                                      <Space>
                                        <Text strong>{rec.algorithm.name}</Text>
                                        <Tag
                                          color={
                                            TYPE_COLOR_MAP[
                                              rec.algorithm.type
                                            ] || "blue"
                                          }
                                        >
                                          {rec.algorithm.type}
                                        </Tag>
                                      </Space>
                                      <Progress
                                        percent={rec.matchScore}
                                        strokeColor={{
                                          "60%": "#52c41a",
                                          "40%": "#faad14",
                                          "0%": "#f5222d",
                                        }}
                                        format={() => `${rec.matchScore}% 匹配`}
                                      />
                                      <Paragraph
                                        type="secondary"
                                        style={{ marginBottom: 0 }}
                                      >
                                        {rec.effectDescription ||
                                          rec.reason ||
                                          "暂无描述"}
                                      </Paragraph>
                                    </Space>
                                  </Col>
                                  <Col span={8} style={{ textAlign: "right" }}>
                                    <Button
                                      type="primary"
                                      size="small"
                                      onClick={() =>
                                        handleSelectRecommendation(
                                          rec.algorithm
                                        )
                                      }
                                    >
                                      选择
                                    </Button>
                                  </Col>
                                </Row>
                              </Card>
                            </Col>
                          ))}
                        </Row>
                      )}
                    </>
                  )}
                </Card>
              ),
            },
          ]}
        />
      </Card>

      {/* 主体：左侧分类树 + 右侧算法卡片 */}
      <Row gutter={12}>
        <Col span={5}>
          <Card size="small" title="算法分类" loading={loading}>
            <Tree
              treeData={treeData}
              defaultSelectedKeys={["all"]}
              onSelect={handleTreeSelect}
              blockNode
            />
          </Card>
        </Col>
        <Col span={19}>
          <Card
            size="small"
            title={
              <Space>
                <Input
                  placeholder="搜索算法名称、类型、描述"
                  allowClear
                  prefix={<SearchOutlined />}
                  style={{ width: 260 }}
                  value={keywords}
                  onChange={(e) => debouncedSearch(e.target.value)}
                />
                <Button icon={<ReloadOutlined />} onClick={handleReset}>
                  重置
                </Button>
              </Space>
            }
            extra={
              <Button
                type="primary"
                disabled={!selectedId}
                onClick={handleConfirm}
              >
                确认选择
              </Button>
            }
          >
            {displayList.length === 0 ? (
              <Empty description="暂无匹配算法" />
            ) : (
              <Row gutter={[12, 12]}>
                {displayList.map((algo) => (
                  <Col key={algo.id} span={8}>
                    <Card
                      size="small"
                      hoverable
                      style={{
                        borderColor:
                          selectedId === algo.id ? "#3B82F6" : undefined,
                        background:
                          selectedId === algo.id ? "#eff6ff" : undefined,
                      }}
                      onClick={() => setSelectedId(algo.id)}
                    >
                      <Space
                        direction="vertical"
                        size={6}
                        style={{ width: "100%" }}
                      >
                        <Space>
                          {selectedId === algo.id && (
                            <CheckCircleFilled style={{ color: "#3B82F6" }} />
                          )}
                          <Text strong>{algo.name}</Text>
                        </Space>
                        <Space size={4}>
                          <Tag>{algo.type}</Tag>
                          <Tag color={algo.status === 1 ? "green" : "default"}>
                            {algo.status === 1 ? "启用" : "禁用"}
                          </Tag>
                        </Space>
                        <Paragraph
                          type="secondary"
                          ellipsis={{ rows: 2 }}
                          style={{ marginBottom: 0 }}
                        >
                          {algo.description || "暂无描述"}
                        </Paragraph>
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
