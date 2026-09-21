package input_history

import (
	"context"
	"encoding/json"
	"strconv"
	"unicode/utf8"

	"github.com/earthyzinc/dehaze-go/internal/model"
	ihrepo "github.com/earthyzinc/dehaze-go/internal/repository/input_history"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// defaultHistoryRetention 无会员档案/权益缺失时的兜底保留条数（对齐 sys_member_benefit level_0 种子值）
const defaultHistoryRetention = 100

// InputHistoryService 图像输入历史记录服务
type InputHistoryService struct {
	repo        ihrepo.IInputHistoryRepository
	memberRepo  memberrepo.IMemberRepository
	benefitRepo memberrepo.IMemberBenefitRepository
}

func NewInputHistoryService(repo ihrepo.IInputHistoryRepository, memberRepo memberrepo.IMemberRepository, benefitRepo memberrepo.IMemberBenefitRepository) *InputHistoryService {
	return &InputHistoryService{repo: repo, memberRepo: memberRepo, benefitRepo: benefitRepo}
}

// GetPage 分页查询历史记录
func (s *InputHistoryService) GetPage(ctx context.Context, userID int64, pageNum, pageSize int, inputSource, keyword string, status int) (*common.PageResult, error) {
	list, total, err := s.repo.FindPage(ctx, userID, pageNum, pageSize, inputSource, keyword, status)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询历史记录失败", err)
	}
	return &common.PageResult{
		List:     list,
		Total:    total,
		Page:     pageNum,
		PageSize: pageSize,
	}, nil
}

// GetByID 查询历史记录详情（校验归属，非本人记录与不存在同样返回 RESOURCE_NOT_FOUND，不泄露存在性）
func (s *InputHistoryService) GetByID(ctx context.Context, id, userID int64) (*model.SysInputHistory, error) {
	history, err := s.repo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询历史记录失败", err)
	}
	if history == nil || history.UserID != userID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "历史记录不存在")
	}
	return history, nil
}

// Create 创建历史记录（入参校验 + 配额自动清理）
func (s *InputHistoryService) Create(ctx context.Context, history *model.SysInputHistory) error {
	if err := validateHistory(history); err != nil {
		return err
	}

	retention, err := s.historyRetention(ctx, history.UserID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询会员权益失败", err)
	}
	count, err := s.repo.CountByUserID(ctx, history.UserID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询历史记录失败", err)
	}
	// 配额已满：自动清理最旧记录后写入（对齐 Java autoCleanup 与文档 §4.3）
	if count >= int64(retention) {
		if err := s.repo.DeleteOldest(ctx, history.UserID); err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "清理历史记录失败", err)
		}
	}
	return s.repo.Create(ctx, history)
}

// Delete 删除单条历史记录（校验归属，幂等：不存在或非本人记录均静默成功）
func (s *InputHistoryService) Delete(ctx context.Context, id, userID int64) error {
	history, err := s.repo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询历史记录失败", err)
	}
	if history == nil || history.UserID != userID {
		return nil
	}
	_, err = s.repo.DeleteByUserAndIDs(ctx, userID, []int64{id})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除历史记录失败", err)
	}
	return nil
}

// BatchDelete 批量删除（user_id 过滤确保只删除当前用户的记录），返回实际删除数量
func (s *InputHistoryService) BatchDelete(ctx context.Context, ids []int64, userID int64) (int64, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	affected, err := s.repo.DeleteByUserAndIDs(ctx, userID, ids)
	if err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "批量删除历史记录失败", err)
	}
	return affected, nil
}

// ClearAll 清空用户所有历史记录
func (s *InputHistoryService) ClearAll(ctx context.Context, userID int64) (int64, error) {
	return s.repo.DeleteByUserID(ctx, userID)
}

// validateHistory 创建入参校验（长度对齐 sys_input_history 列定义，超长入库会 MySQL 报错）
func validateHistory(h *model.SysInputHistory) error {
	for _, c := range []struct {
		name string
		v    *string
		max  int
	}{
		{"originalImageUrl", h.OriginalImageURL, 500},
		{"originalThumbnailUrl", h.OriginalThumbnailURL, 500},
		{"resultImageUrl", h.ResultImageURL, 500},
		{"resultThumbnailUrl", h.ResultThumbnailURL, 500},
		{"algorithmName", h.AlgorithmName, 100},
	} {
		if c.v != nil && utf8.RuneCountInString(*c.v) > c.max {
			return common.NewBizError(common.PARAM_ERROR, c.name+"长度不能超过"+strconv.Itoa(c.max))
		}
	}

	if h.Status != nil && (*h.Status < 1 || *h.Status > 3) {
		return common.NewBizError(common.PARAM_ERROR, "status 取值无效（1=成功，2=失败，3=处理中）")
	}
	if h.InputSource != nil && *h.InputSource != "" &&
		*h.InputSource != "upload" && *h.InputSource != "camera" && *h.InputSource != "sample" {
		return common.NewBizError(common.PARAM_ERROR, "inputSource 取值无效（upload/camera/sample）")
	}
	// algorithmParams 落库为 JSON 字符串，非 JSON 内容会在回读解析时炸（A0400 前置拦截）
	if h.AlgorithmParams != nil && *h.AlgorithmParams != "" && !json.Valid([]byte(*h.AlgorithmParams)) {
		return common.NewBizError(common.PARAM_ERROR, "algorithmParams 必须是合法的 JSON 字符串")
	}
	if h.ProcessingTime != nil && *h.ProcessingTime < 0 {
		return common.NewBizError(common.PARAM_ERROR, "processingTime 不能为负数")
	}
	return nil
}

// historyRetention 按会员等级取历史保留条数（sys_member_benefit.history_retention）
func (s *InputHistoryService) historyRetention(ctx context.Context, userID int64) (int, error) {
	member, err := s.memberRepo.FindByUserID(ctx, userID)
	if err != nil {
		return 0, err
	}
	if member == nil {
		return defaultHistoryRetention, nil
	}
	benefit, err := s.benefitRepo.FindByLevelCode(ctx, member.LevelCode)
	if err != nil {
		return 0, err
	}
	if benefit == nil || benefit.HistoryRetention <= 0 {
		return defaultHistoryRetention, nil
	}
	return benefit.HistoryRetention, nil
}
