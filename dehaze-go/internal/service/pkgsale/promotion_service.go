package pkgsale

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	pkgsalerepo "github.com/earthyzinc/dehaze-go/internal/repository/pkgsale"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

type PromotionService struct {
	db            *gorm.DB
	promotionRepo pkgsalerepo.IPromotionRepository
	cache         types.ICache
}

func NewPromotionService(
	db *gorm.DB,
	promotionRepo pkgsalerepo.IPromotionRepository,
	cache types.ICache,
) *PromotionService {
	return &PromotionService{
		db:            db,
		promotionRepo: promotionRepo,
		cache:         cache,
	}
}

func (s *PromotionService) GetPage(ctx context.Context, q *query.PromotionPageQuery) (*vo.PageResult[vo.PromotionVO], error) {
	list, total, err := s.promotionRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询促销活动列表失败", err)
	}
	vos := make([]vo.PromotionVO, 0, len(list))
	for i := range list {
		vos = append(vos, toPromotionVO(&list[i]))
	}
	return &vo.PageResult[vo.PromotionVO]{List: vos, Total: total}, nil
}

func (s *PromotionService) Create(ctx context.Context, form *bo.PromotionForm) (*vo.PromotionVO, error) {
	startTime, endTime, err := parsePromotionTimeRange(form)
	if err != nil {
		return nil, err
	}

	p := &model.SysPromotion{
		Name:        form.Name,
		Type:        form.Type,
		Description: form.Description,
		StartTime:   startTime,
		EndTime:     endTime,
		NewUserOnly: normalizeNewUserOnly(form.NewUserOnly),
		Status:      0,
	}
	p.ActivityRules = marshalNullableJSON(form.ActivityRules)
	if form.Status != nil {
		p.Status = int8(*form.Status)
	}

	if err := s.promotionRepo.Create(ctx, p); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建促销活动失败", err)
	}
	result := toPromotionVO(p)
	return &result, nil
}

func (s *PromotionService) Update(ctx context.Context, id int64, form *bo.PromotionForm) (*vo.PromotionVO, error) {
	existing, err := s.promotionRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询促销活动失败", err)
	}
	if existing == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "促销活动不存在")
	}

	startTime, endTime, err := parsePromotionTimeRange(form)
	if err != nil {
		return nil, err
	}

	updates := map[string]interface{}{
		"name":           form.Name,
		"type":           form.Type,
		"description":    form.Description,
		"start_time":     startTime,
		"end_time":       endTime,
		"activity_rules": marshalNullableJSON(form.ActivityRules),
		"new_user_only":  normalizeNewUserOnly(form.NewUserOnly),
	}
	// 状态只能通过 /status 端点切换，修改请求不更新 status
	if err := s.promotionRepo.Update(ctx, id, updates); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新促销活动失败", err)
	}

	updated, err := s.promotionRepo.FindByID(ctx, id)
	if err != nil || updated == nil {
		updated = existing
	}
	result := toPromotionVO(updated)
	return &result, nil
}

func (s *PromotionService) UpdateStatus(ctx context.Context, id int64, status int) (*vo.PromotionVO, error) {
	existing, err := s.promotionRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询促销活动失败", err)
	}
	if existing == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "促销活动不存在")
	}

	if err := s.promotionRepo.UpdateStatus(ctx, id, int8(status)); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新促销活动状态失败", err)
	}
	// 活动上下架影响商品详情的 activePromotions，联动失效关联商品缓存
	s.invalidatePackageCacheByPromotion(ctx, id)

	existing.Status = int8(status)
	result := toPromotionVO(existing)
	return &result, nil
}

func (s *PromotionService) Delete(ctx context.Context, id int64) error {
	existing, err := s.promotionRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询促销活动失败", err)
	}
	if existing == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "促销活动不存在")
	}

	packageIDs, err := s.promotionRepo.ListPackageIDs(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询关联套餐失败", err)
	}
	if err := s.promotionRepo.DeleteByID(ctx, id); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除促销活动失败", err)
	}
	// 软删活动后同步清理关联关系
	if err := s.promotionRepo.RebindPackages(ctx, id, "percent", 0, nil); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "清理关联套餐失败", err)
	}
	s.invalidatePackageCacheByIDs(ctx, packageIDs)
	return nil
}

func (s *PromotionService) BindPackages(ctx context.Context, id int64, form *bo.PromotionPackageForm) error {
	existing, err := s.promotionRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询促销活动失败", err)
	}
	if existing == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "促销活动不存在")
	}

	// 折扣方式/折扣值取自活动规则，与价格计算口径一致
	rules := unmarshalNullableRules(existing.ActivityRules)
	discountType := "percent"
	if existing.Type == "full_reduction" {
		discountType = "full_reduction"
	}
	if raw, ok := rules["discount_type"].(string); ok {
		if raw == "percent" || raw == "fixed" || raw == "full_reduction" {
			discountType = raw
		}
	}
	discountValue := int64(0)
	if raw, ok := rules["discount_value"].(float64); ok {
		discountValue = int64(raw)
	}

	if err := s.promotionRepo.RebindPackages(ctx, id, discountType, discountValue, form.PackageIDs); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "关联套餐失败", err)
	}
	s.invalidatePackageCacheByIDs(ctx, form.PackageIDs)
	return nil
}

func (s *PromotionService) invalidatePackageCacheByPromotion(ctx context.Context, promotionID int64) {
	packageIDs, err := s.promotionRepo.ListPackageIDs(ctx, promotionID)
	if err != nil {
		return
	}
	s.invalidatePackageCacheByIDs(ctx, packageIDs)
}

func (s *PromotionService) invalidatePackageCacheByIDs(ctx context.Context, packageIDs []int64) {
	if s.cache == nil {
		return
	}
	for _, id := range packageIDs {
		_ = s.cache.Delete(ctx, fmt.Sprintf("package:detail:%d", id))
	}
}

func toPromotionVO(p *model.SysPromotion) vo.PromotionVO {
	return vo.PromotionVO{
		ID:            p.ID,
		Name:          p.Name,
		Type:          p.Type,
		Description:   p.Description,
		StartTime:     p.StartTime.Format(timeFormat),
		EndTime:       p.EndTime.Format(timeFormat),
		ActivityRules: unmarshalNullableRules(p.ActivityRules),
		NewUserOnly:   int(p.NewUserOnly),
		Status:        int(p.Status),
		CreateTime:    p.CreatedAt.Format(timeFormat),
	}
}

func parsePromotionTimeRange(form *bo.PromotionForm) (time.Time, time.Time, error) {
	startTime, err := time.ParseInLocation(timeFormat, form.StartTime, time.Local)
	if err != nil {
		return time.Time{}, time.Time{}, common.NewBizError(common.PARAM_ERROR, "开始时间格式不正确")
	}
	endTime, err := time.ParseInLocation(timeFormat, form.EndTime, time.Local)
	if err != nil {
		return time.Time{}, time.Time{}, common.NewBizError(common.PARAM_ERROR, "结束时间格式不正确")
	}
	return startTime, endTime, nil
}

func normalizeNewUserOnly(v *int) int8 {
	if v != nil && *v == 1 {
		return 1
	}
	return 0
}

func marshalNullableJSON(v interface{}) sql.NullString {
	if v == nil {
		return sql.NullString{}
	}
	data, err := json.Marshal(v)
	if err != nil {
		return sql.NullString{}
	}
	return sql.NullString{String: string(data), Valid: true}
}

func unmarshalNullableRules(ns sql.NullString) map[string]interface{} {
	if !ns.Valid || ns.String == "" {
		return nil
	}
	var rules map[string]interface{}
	if err := json.Unmarshal([]byte(ns.String), &rules); err != nil {
		return nil
	}
	return rules
}

var _ IPromotionService = (*PromotionService)(nil)
