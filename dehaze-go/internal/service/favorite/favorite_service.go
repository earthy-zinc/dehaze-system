package favorite

import (
	"context"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	favrepo "github.com/earthyzinc/dehaze-go/internal/repository/favorite"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

const (
	// favoriteCapacityDictType 收藏容量字典类型（sys_dict: favorite_capacity）
	favoriteCapacityDictType = "favorite_capacity"

	targetTypeAlgorithm = "algorithm"
	targetTypeResult    = "result"
	targetTypeDataset   = "dataset"
	targetTypeImage     = "image"
	targetTypePreset    = "preset"
)

var validTargetTypes = []string{targetTypeAlgorithm, targetTypeResult, targetTypeDataset, targetTypeImage, targetTypePreset}

type FavoriteService struct {
	favRepo       favrepo.IFavoriteRepository
	memberRepo    memberrepo.IMemberRepository
	algorithmRepo algorepo.IAlgorithmRepository
	predLogRepo   predrepo.IPredLogRepository
	datasetRepo   datasetrepo.IDatasetRepository
	dictSvc       dictservice.IDictService
}

func NewFavoriteService(
	favRepo favrepo.IFavoriteRepository,
	memberRepo memberrepo.IMemberRepository,
	algorithmRepo algorepo.IAlgorithmRepository,
	predLogRepo predrepo.IPredLogRepository,
	datasetRepo datasetrepo.IDatasetRepository,
	dictSvc dictservice.IDictService,
) *FavoriteService {
	return &FavoriteService{
		favRepo:       favRepo,
		memberRepo:    memberRepo,
		algorithmRepo: algorithmRepo,
		predLogRepo:   predLogRepo,
		datasetRepo:   datasetRepo,
		dictSvc:       dictSvc,
	}
}

func (s *FavoriteService) Add(ctx context.Context, userID int64, form *bo.FavoriteForm) (int64, error) {
	// 校验目标对象是否存在
	if err := s.checkTargetExists(ctx, form.TargetType, form.TargetID); err != nil {
		return 0, err
	}

	// 容量校验
	currentCount, err := s.favRepo.CountByUserID(ctx, userID)
	if err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "查询收藏数量失败", err)
	}
	capacity := s.getCapacity(ctx, userID)
	if currentCount >= int64(capacity) {
		return 0, common.NewBizError(common.BUSINESS_ERROR,
			fmt.Sprintf("收藏已达上限（%d条），请清理后重试", capacity))
	}

	favorite := &model.SysFavorite{
		UserID:     userID,
		TargetType: form.TargetType,
		TargetID:   form.TargetID,
		IsInvalid:  0,
		Deleted:    0,
	}
	if err := s.favRepo.Upsert(ctx, favorite); err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "添加收藏失败", err)
	}
	return favorite.ID, nil
}

func (s *FavoriteService) checkTargetExists(ctx context.Context, targetType string, targetID int64) error {
	switch targetType {
	case targetTypeAlgorithm:
		exists, err := s.algorithmRepo.ExistsByID(ctx, targetID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
		}
		if !exists {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
		}
	case targetTypeDataset:
		exists, err := s.datasetRepo.ExistsByID(ctx, targetID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询数据集失败", err)
		}
		if !exists {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "数据集不存在")
		}
	case targetTypeResult:
		exists, err := s.predLogRepo.ExistsByID(ctx, targetID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询处理记录失败", err)
		}
		if !exists {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "处理记录不存在")
		}
	}
	return nil
}

func (s *FavoriteService) DeleteByIDs(ctx context.Context, userID int64, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	return s.favRepo.DeleteByIDs(ctx, userID, ids)
}

func (s *FavoriteService) GetPage(ctx context.Context, userID int64, q *query.FavoritePageQuery) (*vo.PageResult[vo.FavoriteVO], error) {
	rows, total, err := s.favRepo.FindPage(ctx, userID, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询收藏列表失败", err)
	}

	list := make([]vo.FavoriteVO, 0, len(rows))
	for _, row := range rows {
		createTime := ""
		if !row.CreatedAt.IsZero() {
			createTime = row.CreatedAt.Format(time.DateTime)
		}
		list = append(list, vo.FavoriteVO{
			ID:         row.ID,
			UserID:     row.UserID,
			TargetType: row.TargetType,
			TargetID:   row.TargetID,
			TargetName: row.AlgorithmName,
			IsInvalid:  row.IsInvalid != 0,
			CreateTime: createTime,
		})
	}

	return &vo.PageResult[vo.FavoriteVO]{List: list, Total: total}, nil
}

func (s *FavoriteService) GetStatus(ctx context.Context, userID int64, targetType string, targetID int64) (*vo.FavoriteStatusVO, error) {
	count, err := s.favRepo.CountByUserAndType(ctx, userID, targetType)
	_ = count // count for debugging if needed
	fav, err := s.favRepo.FindByUserAndTarget(ctx, userID, targetType, targetID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询收藏状态失败", err)
	}
	return &vo.FavoriteStatusVO{
		TargetType: targetType,
		TargetID:   targetID,
		Favorited:  fav != nil,
	}, nil
}

func (s *FavoriteService) GetCount(ctx context.Context, userID int64, targetType string) ([]vo.FavoriteCountVO, error) {
	rows, err := s.favRepo.CountGroupByType(ctx, userID, targetType)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询收藏数量失败", err)
	}

	countMap := make(map[string]int64)
	for _, row := range rows {
		countMap[row.TargetType] = row.Count
	}

	result := make([]vo.FavoriteCountVO, 0, len(validTargetTypes))
	for _, t := range validTargetTypes {
		if targetType != "" && targetType != t {
			continue
		}
		result = append(result, vo.FavoriteCountVO{
			TargetType: t,
			Count:      countMap[t],
		})
	}
	return result, nil
}

func (s *FavoriteService) MarkInvalid(ctx context.Context, targetType string, targetIDs []int64) error {
	if len(targetIDs) == 0 {
		return nil
	}
	return s.favRepo.MarkInvalid(ctx, targetType, targetIDs)
}

// levelCapacityKeys 会员等级 → 收藏容量字典键映射（sys_dict: favorite_capacity）
var levelCapacityKeys = map[string]string{
	"level_0": "default",
	"level_1": "vip1",
	"level_2": "vip2",
	"level_3": "svip",
}

// getCapacity 按用户当前会员等级从字典读取收藏容量上限。
// 缺键/读取失败回退默认容量 200，升级后容量立即扩展（实时读字典）。
func (s *FavoriteService) getCapacity(ctx context.Context, userID int64) int {
	key := "default"
	if member, err := s.memberRepo.FindByUserID(ctx, userID); err == nil && member != nil {
		if k, ok := levelCapacityKeys[member.LevelCode]; ok {
			key = k
		}
	}
	return int(dictservice.GetIntValue(ctx, s.dictSvc, favoriteCapacityDictType, key, 200))
}

var _ IFavoriteService = (*FavoriteService)(nil)
