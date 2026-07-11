package algorithm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// Client Python 算法服务 HTTP 客户端
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient 创建算法服务客户端
func NewClient(cfg options.Algorithm) *Client {
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60
	}
	return &Client{
		baseURL: cfg.ServiceURL,
		httpClient: &http.Client{
			Timeout: time.Duration(timeout) * time.Second,
		},
	}
}

// PredictionRequest 预测请求
type PredictionRequest struct {
	AlgorithmID int64  `json:"algorithmId"`
	ImageURL    string `json:"imageUrl"`
	Params      string `json:"params,omitempty"`
}

// PredictionResponse 预测响应
type PredictionResponse struct {
	LogID              int64  `json:"logId"`
	ResultURL          string `json:"resultUrl"`
	ResultThumbnailURL string `json:"resultThumbnailUrl"`
	Time               int    `json:"time"`
	FromCache          bool   `json:"fromCache"`
}

// EvaluationRequest 评估请求
type EvaluationRequest struct {
	AlgorithmID int64  `json:"algorithmId"`
	PredURL     string `json:"predUrl"`
	GtURL       string `json:"gtUrl"`
}

// EvaluationResponse 评估响应
type EvaluationResponse struct {
	LogID     int64              `json:"logId"`
	Metrics   map[string]float64 `json:"metrics"`
	Qualified bool               `json:"qualified"`
	Time      int                `json:"time"`
}

// Predict 调用 Python 预测服务
func (c *Client) Predict(ctx context.Context, req *PredictionRequest) (*PredictionResponse, error) {
	var resp PredictionResponse
	if err := c.doPost(ctx, "/api/v1/prediction", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Evaluate 调用 Python 评估服务
func (c *Client) Evaluate(ctx context.Context, req *EvaluationRequest) (*EvaluationResponse, error) {
	var resp EvaluationResponse
	if err := c.doPost(ctx, "/api/v1/evaluation", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// doPost 通用 POST 请求
func (c *Client) doPost(ctx context.Context, path string, reqBody interface{}, respBody interface{}) error {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("序列化请求失败: %w", err)
	}

	url := c.baseURL + path
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("创建请求失败: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		logger.Error("算法服务请求失败", zap.String("url", url), zap.Duration("elapsed", time.Since(start)), zap.Error(err))
		return fmt.Errorf("算法服务请求失败: %w", err)
	}
	defer httpResp.Body.Close()

	respBytes, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return fmt.Errorf("读取响应失败: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		logger.Error("算法服务返回错误", zap.Int("status", httpResp.StatusCode), zap.String("body", string(respBytes)))
		return fmt.Errorf("算法服务返回状态码 %d: %s", httpResp.StatusCode, string(respBytes))
	}

	// Python 端返回格式: {"code":"00000","msg":"...","data":{...}}
	var apiResp struct {
		Code string          `json:"code"`
		Msg  string          `json:"msg"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(respBytes, &apiResp); err != nil {
		return fmt.Errorf("解析响应失败: %w", err)
	}
	if apiResp.Code != "00000" {
		return fmt.Errorf("算法服务返回错误: [%s] %s", apiResp.Code, apiResp.Msg)
	}

	if err := json.Unmarshal(apiResp.Data, respBody); err != nil {
		return fmt.Errorf("解析响应数据失败: %w", err)
	}

	logger.Info("算法服务调用成功", zap.String("path", path), zap.Duration("elapsed", time.Since(start)))
	return nil
}
