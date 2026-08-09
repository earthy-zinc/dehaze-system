package algorithm

import (
	"context"
)

// AnalyzeImageRequest 图像特征分析请求（对应 Python /api/v1/recommendations/analyze）
type AnalyzeImageRequest struct {
	ImageID  *int64 `json:"imageId,omitempty"`
	ImageURL string `json:"imageUrl,omitempty"`
}

// AnalyzeImageResponse 图像特征分析响应
type AnalyzeImageResponse struct {
	ImageMd5          string          `json:"imageMd5"`
	HazeLevel         string          `json:"hazeLevel"`
	HazeConfidence    float64         `json:"hazeConfidence"`
	SceneType         string          `json:"sceneType"`
	SceneConfidence   float64         `json:"sceneConfidence"`
	Lighting          string          `json:"lighting"`
	Complexity        float64         `json:"complexity"`
	ColorDistribution ColorFeature   `json:"colorDistribution"`
	Resolution        string          `json:"resolution"`
	NoiseLevel        string          `json:"noiseLevel"`
}

// ColorFeature 颜色分布特征
type ColorFeature struct {
	Temperature float64 `json:"temperature"`
	Saturation  float64 `json:"saturation"`
}

// AnalyzeImage 调用 Python 图像特征分析服务
// Python 服务不可用时返回 error，不降级为伪特征，避免误导用户
func (c *Client) AnalyzeImage(ctx context.Context, imageURL string) (*AnalyzeImageResponse, error) {
	req := &AnalyzeImageRequest{ImageURL: imageURL}
	var resp AnalyzeImageResponse
	if err := c.doPost(ctx, "/api/v1/recommendations/analyze", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}
