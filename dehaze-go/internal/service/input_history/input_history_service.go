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
func (s *InputHistoryService) GetPage(ctx context.Context, userID int64, pageNum, pageSize int, inputSource, keyword string, favoriteOnly bool) (*common.PageResult, error) {
	list, total, err := s.repo.FindPage(ctx, userID, pageNum, pageSize, inputSource, keyword, favoriteOnly)
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

// GetByID 查询历史记录详情
func (s *InputHistoryService) GetByID(ctx context.Context, id int64) (*model.SysInputHistory, error) {
	history, err := s.repo.FindByID(ctx, id)
	if err != nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "历史记录不存在")
	}
	return history, nil
}

// Create 创建历史记录
func (s *InputHistoryService) Create(ctx context.Context, history *model.SysInputHistory) error {
	return s.repo.Create(ctx, history)
}

// Update 更新历史记录（如收藏标记）
func (s *InputHistoryService) Update(ctx context.Context, id int64, userID int64, updates map[string]interface{}) error {
	history, err := s.repo.FindByID(ctx, id)
	if err != nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "历史记录不存在")
	}
	if history.UserID != userID {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "无权操作他人的历史记录")
	}
	if v, ok := updates["is_favorite"]; ok {
		history.IsFavorite = ptrInt8(v)
	}
	return s.repo.Update(ctx, history)
}

// Delete 删除单条历史记录
func (s *InputHistoryService) Delete(ctx context.Context, id int64, userID int64) error {
	history, err := s.repo.FindByID(ctx, id)
	if err != nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "历史记录不存在")
	}
	if history.UserID != userID {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "无权操作他人的历史记录")
	}
	return s.repo.Delete(ctx, []int64{id})
}

// BatchDelete 批量删除
func (s *InputHistoryService) BatchDelete(ctx context.Context, ids []int64, userID int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "删除列表不能为空")
	}
	return s.repo.Delete(ctx, ids)
}

// ClearAll 清空用户所有历史记录
func (s *InputHistoryService) ClearAll(ctx context.Context, userID int64) (int64, error) {
	return s.repo.DeleteByUserID(ctx, userID)
}

func ptrInt8(v interface{}) *int8 {
	switch val := v.(type) {
	case float64:
		i := int8(val)
		return &i
	case int:
		i := int8(val)
		return &i
	default:
		return nil
	}
}
