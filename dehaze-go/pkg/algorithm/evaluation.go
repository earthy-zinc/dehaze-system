package algorithm

import (
	"context"
	"fmt"
)

type EvaluationRequest struct {
	AlgorithmID int64  `json:"algorithmId"`
	PredURL     string `json:"predUrl"`
	GtURL       string `json:"gtUrl"`
}

type EvaluationResponse struct {
	LogID        int64              `json:"logId"`
	Status       string             `json:"status"`
	Metrics      map[string]float64 `json:"metrics,omitempty"`
	Time         int                `json:"time,omitempty"`
	ErrorMessage string             `json:"errorMessage,omitempty"`
}

func (c *Client) Evaluate(ctx context.Context, req *EvaluationRequest) (*EvaluationResponse, error) {
	var resp EvaluationResponse
	if err := c.doPost(ctx, "/api/v1/evaluation", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (c *Client) GetEvalTaskStatus(ctx context.Context, taskID int64) (*EvaluationResponse, error) {
	var resp EvaluationResponse
	path := fmt.Sprintf("/api/v1/evaluation/%d", taskID)
	if err := c.doGet(ctx, path, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}
