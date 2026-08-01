import {
  LoadingOutlined,
  PictureOutlined,
  DislikeOutlined,
  LikeOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Col,
  Empty,
  Input,
  message,
  Progress,
  Row,
  Space,
  Spin,
  Tag,
  Typography,
  Rate,
  Alert,
} from "antd";
import React, { useState } from "react";
import { RecommendationAPI } from "dehaze-sdk-js";
import type {
  AnalyzeRequest,
  ImageFeatureAnalysis,
  RecommendedAlgorithm,
  RecommendationFeedback,
} from "dehaze-sdk-js";
import "./index.module.scss";

const { Text, Title } = Typography;

interface RecommendationWidgetProps {
  imageId?: number;
  imageUrl?: string;
  onSelect?: (algorithm: RecommendedAlgorithm) => void;
  className?: string;
}

export default function RecommendationWidget({
  imageId,
  imageUrl: propImageUrl,
  onSelect,
  className = "",
}: RecommendationWidgetProps) {
  const [imageUrl, setImageUrl] = useState(propImageUrl || "");
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<ImageFeatureAnalysis | null>(null);
  const [recommendations, setRecommendations] = useState<
    RecommendedAlgorithm[]
  >([]);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!imageId && !imageUrl.trim()) {
      message.warning("请输入图片 URL 或选择图片");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const params: AnalyzeRequest = imageId
        ? { imageId }
        : { imageUrl: imageUrl.trim() };

      const result = await RecommendationAPI.analyze(params);
      setAnalysis(result);

      if (result.imageMd5) {
        const recs = await RecommendationAPI.getAlgorithmRecommendations({
          imageMd5: result.imageMd5,
        });
        setRecommendations(recs);
      }
    } catch (err: any) {
      setError(err?.message || "分析失败，请检查图片地址后重试");
      message.error(err?.message || "分析失败");
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (rec: RecommendedAlgorithm, useful: boolean) => {
    if (!rec.recommendationId) return;
    try {
      await RecommendationAPI.submitFeedback({
        recommendationId: rec.recommendationId,
        useful,
      });
      message.success(useful ? "反馈已提交（有用）" : "反馈已提交（无用）");
    } catch (err: any) {
      message.error(err?.message || "提交反馈失败");
    }
  };

  const reset = () => {
    setAnalysis(null);
    setRecommendations([]);
    setError(null);
    setImageUrl("");
  };

  // === 无结果状态：显示图片输入 ===
  if (!analysis && !propImageUrl) {
    return (
      <Card
        className={`recommendation-widget ${className}`}
        title="智能算法推荐"
      >
        <div className="rec-upload-section">
          <PictureOutlined style={{ fontSize: 40, color: "#1677ff" }} />
          <Text type="secondary" style={{ display: "block", marginBottom: 16 }}>
            上传图片以获取智能算法推荐
          </Text>
          <Input
            placeholder="输入图片 URL（可选）"
            value={imageUrl}
            onChange={(e) => setImageUrl(e.target.value)}
            onPressEnter={handleAnalyze}
            suffix={
              <Button
                type="primary"
                icon={<UploadOutlined />}
                onClick={handleAnalyze}
                loading={loading}
              >
                分析
              </Button>
            }
          />
        </div>
      </Card>
    );
  }

  // === 加载中 ===
  if (loading) {
    return (
      <Card
        className={`recommendation-widget ${className}`}
        title="智能算法推荐"
      >
        <Spin tip="正在分析图像特征..." indicator={<LoadingOutlined spin />} />
      </Card>
    );
  }

  // === 错误状态 ===
  if (error) {
    return (
      <Card
        className={`recommendation-widget ${className}`}
        title="智能算法推荐"
      >
        <Alert
          message={error}
          type="error"
          showIcon
          action={<Button onClick={reset}>重新上传</Button>}
        />
      </Card>
    );
  }

  if (!analysis) return null;

  // === 分析完成，展示结果 ===
  return (
    <Card
      className={`recommendation-widget loaded ${className}`}
      title="智能算法推荐"
    >
      {/* 图像特征分析 */}
      <div className="feature-analysis">
        <Title level={5}>图像特征分析</Title>
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <div className="feature-item">
              <Text type="secondary">雾霾浓度:</Text>
              <Tag color="blue">{analysis.hazeLevel}</Tag>
              <Progress
                percent={Math.round(analysis.hazeConfidence * 100)}
                size="small"
                style={{ marginTop: 4 }}
              />
            </div>
          </Col>
          <Col span={12}>
            <div className="feature-item">
              <Text type="secondary">场景类型:</Text>
              <Tag color="green">{analysis.sceneType}</Tag>
              <Progress
                percent={Math.round(analysis.sceneConfidence * 100)}
                size="small"
                style={{ marginTop: 4 }}
              />
            </div>
          </Col>
          <Col span={12}>
            <div className="feature-item">
              <Text type="secondary">光照条件:</Text>
              <Tag color="orange">{analysis.lighting}</Tag>
            </div>
          </Col>
          <Col span={12}>
            <div className="feature-item">
              <Text type="secondary">图像复杂度:</Text>
              <Progress
                percent={Math.round(analysis.complexity * 100)}
                size="small"
              />
            </div>
          </Col>
          <Col span={12}>
            <div className="feature-item">
              <Text type="secondary">分辨率:</Text>
              <Tag color="purple">{analysis.resolution.toUpperCase()}</Tag>
            </div>
          </Col>
          <Col span={12}>
            <div className="feature-item">
              <Text type="secondary">噪声水平:</Text>
              <Tag color="red">{analysis.noiseLevel}</Tag>
            </div>
          </Col>
        </Row>
      </div>

      {/* Top 3 推荐算法 */}
      <div className="algorithm-recommendations">
        <Title level={5}>推荐算法</Title>
        {recommendations.length === 0 ? (
          <Empty
            description="暂无推荐算法"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <div className="rec-list">
            {recommendations.slice(0, 3).map((rec, idx) => (
              <div key={rec.algorithmId} className="rec-item">
                <div className="rec-header">
                  <Text strong>
                    {idx + 1}. {rec.algorithmName}
                  </Text>
                  <Rate value={rec.rating} disabled style={{ marginLeft: 8 }} />
                </div>
                <Text
                  type="secondary"
                  style={{ fontSize: 12, display: "block", marginBottom: 4 }}
                >
                  {rec.reason}
                </Text>
                <Progress
                  percent={rec.matchScore}
                  strokeColor={{
                    "0%": "#108ee9",
                    "100%": "#87d068",
                  }}
                  format={() => `${rec.matchScore}%`}
                  style={{ marginBottom: 8 }}
                />
                <div className="rec-actions">
                  <Button
                    type="primary"
                    size="small"
                    onClick={() => {
                      onSelect?.(rec);
                    }}
                  >
                    选用
                  </Button>
                  <Space>
                    <Button
                      size="small"
                      icon={<LikeOutlined />}
                      onClick={() => handleFeedback(rec, true)}
                      disabled={!rec.recommendationId}
                    >
                      有用
                    </Button>
                    <Button
                      size="small"
                      icon={<DislikeOutlined />}
                      onClick={() => handleFeedback(rec, false)}
                      disabled={!rec.recommendationId}
                    >
                      无用
                    </Button>
                  </Space>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}
