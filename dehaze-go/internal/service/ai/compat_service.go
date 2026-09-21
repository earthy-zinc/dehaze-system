package ai

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// CompatAuditService AI 兼容端点调用审计查询（登录用户仅可查本人）
type CompatAuditService struct {
	repo *airepo.CompatAuditRepository
}

func NewCompatAuditService(repo *airepo.CompatAuditRepository) *CompatAuditService {
	return &CompatAuditService{repo: repo}
}

// ListCalls 分页查询当前用户的兼容调用日志；时间格式非法按无过滤处理（与 python 一致）
func (s *CompatAuditService) ListCalls(
	ctx context.Context,
	userID int64,
	keyID *int64,
	model string,
	startTime, endTime string,
	page, size int,
) (*vo.PageResult[vo.CompatCallVO], error) {
	if s.repo == nil {
		return nil, common.NewBizError(common.MIDDLEWARE_SERVICE_ERROR, "审计存储不可用")
	}
	if page < 1 {
		page = 1
	}
	if size < 1 {
		size = 20
	}
	if size > 100 {
		size = 100
	}
	records, total, err := s.repo.Query(ctx, userID, keyID, model, parseAuditTime(startTime), parseAuditTime(endTime), page, size)
	if err != nil {
		return nil, common.WrapBizError(common.MIDDLEWARE_SERVICE_ERROR, "查询兼容调用审计失败", err)
	}
	items := make([]vo.CompatCallVO, 0, len(records))
	for i := range records {
		items = append(items, vo.CompatCallVO{
			ID:             records[i].ID,
			KeyID:          records[i].KeyID,
			KeyPrefix:      records[i].KeyPrefix,
			ConversationID: records[i].ConversationID,
			Model:          records[i].Model,
			Endpoint:       records[i].Endpoint,
			Protocol:       records[i].Protocol,
			IsStream:       records[i].IsStream,
			InputTokens:    records[i].InputTokens,
			OutputTokens:   records[i].OutputTokens,
			Credits:        records[i].Credits,
			StatusCode:     records[i].StatusCode,
			DurationMs:     records[i].DurationMs,
			ClientIp:       records[i].ClientIp,
			RequestID:      records[i].RequestID,
			ErrorMsg:       records[i].ErrorMsg,
			CreateTime:     records[i].CreateTime,
		})
	}
	return &vo.PageResult[vo.CompatCallVO]{List: items, Total: total}, nil
}

// parseAuditTime 支持 "%Y-%m-%d %H:%M:%S" / ISO / "%Y-%m-%d" 三种格式，非法返回 nil
func parseAuditTime(value string) *time.Time {
	if value == "" {
		return nil
	}
	for _, layout := range []string{"2006-01-02 15:04:05", "2006-01-02T15:04:05", "2006-01-02"} {
		if parsed, err := time.ParseInLocation(layout, value, time.Local); err == nil {
			return &parsed
		}
	}
	return nil
}
