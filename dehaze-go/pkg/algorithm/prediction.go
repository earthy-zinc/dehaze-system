package algorithm

import (
	"context"
	"fmt"
)

type PredictionRequest struct {
	AlgorithmID int64  `json:"algorithmId"`
	ImageURL    string `json:"imageUrl"`
	Params      string `json:"params,omitempty"`
}

type PredictionResponse struct {
	LogID              int64  `json:"logId"`
	Status             string `json:"status"`
	ResultURL          string `json:"resultUrl,omitempty"`
	ResultThumbnailURL string `json:"resultThumbnailUrl,omitempty"`
	Time               int    `json:"time,omitempty"`
	ErrorMessage       string `json:"errorMessage,omitempty"`
}

func (c *Client) Predict(ctx context.Context, req *PredictionRequest) (*PredictionResponse, error) {
	var resp PredictionResponse
	if err := c.doPost(ctx, "/api/v1/prediction", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (c *Client) GetPredTaskStatus(ctx context.Context, taskID int64) (*PredictionResponse, error) {
	var resp PredictionResponse
	path := fmt.Sprintf("/api/v1/prediction/%d", taskID)
	if err := c.doGet(ctx, path, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}
