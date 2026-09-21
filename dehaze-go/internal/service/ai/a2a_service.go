package ai

import (
	"context"
	"fmt"
	"strconv"
	"time"

	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// 已发布版本号缓存（与 python ai_agent_service 同键，TTL 1800s）
const (
	agentPublishedKeyFmt = "ai:agent:%d:published"
	agentPublishedTTL    = 1800 * time.Second
)

// A2AService A2A 协议 Agent Card 生成（对外暴露的已发布 Agent）
type A2AService struct {
	repo *airepo.A2ARepository
}

func NewA2AService(repo *airepo.A2ARepository) *A2AService {
	return &A2AService{repo: repo}
}

// AgentCard 动态生成 Agent Card：须为启用、已对外暴露、非子 Agent 且存在已发布版本
func (s *A2AService) AgentCard(ctx context.Context, agentID int64, baseURL string) (map[string]any, error) {
	agent, err := s.repo.GetAgent(ctx, agentID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 Agent 失败", err)
	}
	if agent == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "Agent 不存在")
	}
	if agent.Status != 1 || agent.IsExposed != 1 || agent.IsSubagent == 1 {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "Agent 不可对外服务")
	}

	versionNo, err := s.publishedVersionNo(ctx, agentID)
	if err != nil {
		return nil, err
	}
	name := agent.Name
	if name == "" {
		name = agent.AgentCode
	}
	return map[string]any{
		"name":        name,
		"description": agent.Description,
		"version":     strconv.Itoa(versionNo),
		"url":         baseURL + "/a2a",
		"capabilities": map[string]any{
			"streaming":         true,
			"pushNotifications": false,
		},
		"defaultInputModes":  []string{"text", "file"},
		"defaultOutputModes": []string{"text", "file"},
		"skills": []map[string]any{
			{"name": agent.AgentCode, "description": agent.Description},
		},
		"securitySchemes": map[string]any{"http": map[string]any{"scheme": "bearer"}},
		"security":        []map[string]any{{"http": []string{}}},
	}, nil
}

// publishedVersionNo 读已发布版本号（Redis 缓存 → sys_ai_agent_version）
func (s *A2AService) publishedVersionNo(ctx context.Context, agentID int64) (int, error) {
	cacheKey := fmt.Sprintf(agentPublishedKeyFmt, agentID)
	if client := redisClient(); client != nil {
		if raw, err := client.Get(ctx, cacheKey).Result(); err == nil && raw != "" {
			if cached, convErr := strconv.Atoi(raw); convErr == nil {
				return cached, nil
			}
		}
	}
	versionNo, found, err := s.repo.LatestPublishedVersionNo(ctx, agentID)
	if err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "查询已发布版本失败", err)
	}
	if !found {
		return 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "该 Agent 暂无已发布版本")
	}
	if client := redisClient(); client != nil {
		client.Set(ctx, cacheKey, strconv.Itoa(versionNo), agentPublishedTTL)
	}
	return versionNo, nil
}
