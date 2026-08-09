package input_history

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	ihrepo "github.com/earthyzinc/dehaze-go/internal/repository/input_history"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// InputHistoryService 图像输入历史记录服务
type InputHistoryService struct {
	repo ihrepo.IInputHistoryRepository
}

func NewInputHistoryService(repo ihrepo.IInputHistoryRepository) *InputHistoryService {
	return &InputHistoryService{repo: repo}
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

// GetByID 查询历史记录详情（校验归属）
func (s *InputHistoryService) GetByID(ctx context.Context, id, userID int64) (*model.SysInputHistory, error) {
	history, err := s.repo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询历史记录失败", err)
	}
	if history == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "历史记录不存在")
	}
	if history.UserID != userID {
		return nil, common.NewBizError(common.OPERATION_NOT_ALLOW, "无权查看他人的历史记录")
	}
	return history, nil
}

// Create 创建历史记录
func (s *InputHistoryService) Create(ctx context.Context, history *model.SysInputHistory) error {
	return s.repo.Create(ctx, history)
}

// Delete 删除单条历史记录（校验归属，幂等：记录不存在时静默成功）
func (s *InputHistoryService) Delete(ctx context.Context, id, userID int64) error {
	history, err := s.repo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询历史记录失败", err)
	}
	if history == nil {
		// 幂等：记录不存在视为已删除，静默成功
		return nil
	}
	if history.UserID != userID {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "无权删除他人的历史记录")
	}
	_, err = s.repo.DeleteByUserAndIDs(ctx, userID, []int64{id})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除历史记录失败", err)
	}
	return nil
}

// BatchDelete 批量删除（通过 user_id 过滤确保只删除当前用户的记录）
func (s *InputHistoryService) BatchDelete(ctx context.Context, ids []int64, userID int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "删除列表不能为空")
	}
	affected, err := s.repo.DeleteByUserAndIDs(ctx, userID, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "批量删除历史记录失败", err)
	}
	if affected == 0 {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "未找到可删除的历史记录")
	}
	return nil
}

// ClearAll 清空用户所有历史记录
func (s *InputHistoryService) ClearAll(ctx context.Context, userID int64) (int64, error) {
	return s.repo.DeleteByUserID(ctx, userID)
}

