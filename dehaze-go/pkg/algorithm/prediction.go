package algorithm

import (
	"context"
	"fmt"
)

type PredictionRequest struct {
	Model         string `json:"model"`
	ImageURL      string `json:"imageUrl"`
	Params        string `json:"params,omitempty"`
	RecommendedBy *int64 `json:"recommendedBy,omitempty"`
}

type PredictionResponse struct {
	TaskID             int64  `json:"taskId"`
	Status             int8   `json:"status"`
	ResultURL          string `json:"resultUrl,omitempty"`
	ResultMd5          string `json:"resultMd5,omitempty"`
	OriginMd5          string `json:"originMd5,omitempty"`
	ResultThumbnailURL string `json:"resultThumbnailUrl,omitempty"`
	Time               int    `json:"time,omitempty"`
	ErrorMessage       string `json:"errorMessage,omitempty"`
}

// Predict 提交图像编辑任务（POST /api/v1/images/edits）
func (c *Client) Predict(ctx context.Context, req *PredictionRequest) (*PredictionResponse, error) {
	var resp PredictionResponse
	if err := c.doPost(ctx, "/api/v1/images/edits", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// GetPredTaskStatus 查询图像任务状态（GET /api/v1/images/tasks/{taskId}）
func (c *Client) GetPredTaskStatus(ctx context.Context, taskID int64) (*PredictionResponse, error) {
	var resp PredictionResponse
	path := fmt.Sprintf("/api/v1/images/tasks/%d", taskID)
	if err := c.doGet(ctx, path, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}
