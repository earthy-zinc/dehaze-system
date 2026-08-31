package member

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/lifecycle"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

const (
	// memberGrowthRulesDictType 会员成长值规则字典类型（sys_dict: member_growth_rules）
	memberGrowthRulesDictType = "member_growth_rules"
	// signInBonusCycle 连续签到奖励周期（连续 7 天额外奖励，字典未覆盖，属代码常量）
	signInBonusCycle = 7

	timeFormat        = "2006-01-02 15:04:05"
	dateFormat        = "2006-01-02"

	memberProfileCacheTTL = 10 * time.Minute
	memberBenefitCacheTTL = 30 * time.Minute
	quotaCounterCacheTTL  = 35 * 24 * time.Hour
)

var levelNames = map[string]string{
	"level_0": "普通用户",
	"level_1": "VIP1",
	"level_2": "VIP2",
	"level_3": "SVIP",
}

type MemberService struct {
	db            *gorm.DB
	memberRepo    memberrepo.IMemberRepository
	benefitRepo   memberrepo.IMemberBenefitRepository
	growthLogRepo memberrepo.IMemberGrowthLogRepository
	signInRepo    memberrepo.IMemberSignInRepository
	cache         types.ICache
	auditLogSvc   *auditlogservice.AuditLogService
	messageSender MessageSender
	lifecycle     *lifecycle.Manager
	dictSvc       dictservice.IDictService
}

func NewMemberService(
	db *gorm.DB,
	memberRepo memberrepo.IMemberRepository,
	benefitRepo memberrepo.IMemberBenefitRepository,
	growthLogRepo memberrepo.IMemberGrowthLogRepository,
	signInRepo memberrepo.IMemberSignInRepository,
	cache types.ICache,
	auditLogSvc *auditlogservice.AuditLogService,
	messageSender MessageSender,
	lm *lifecycle.Manager,
	dictSvc dictservice.IDictService,
) *MemberService {
	return &MemberService{
		db:            db,
		memberRepo:    memberRepo,
		benefitRepo:   benefitRepo,
		growthLogRepo: growthLogRepo,
		signInRepo:    signInRepo,
		cache:         cache,
		auditLogSvc:   auditLogSvc,
		messageSender: messageSender,
		lifecycle:     lm,
		dictSvc:       dictSvc,
	}
}

func (s *MemberService) GetProfile(ctx context.Context, userID int64) (*vo.MemberProfileVO, error) {
	cacheKey := MemberProfileKey(userID)
	if s.cache != nil {
		if cached, err := s.cache.Get(ctx, cacheKey); err == nil && cached != "" {
			var profile vo.MemberProfileVO
			if err := json.Unmarshal([]byte(cached), &profile); err == nil && profile.UserID > 0 {
				return &profile, nil
			}
		}
	}

	mu, err := s.memberRepo.FindWithUserByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if mu == nil {
		if initErr := s.initDefaultMember(ctx, userID); initErr != nil {
			return nil, initErr
		}
		mu, err = s.memberRepo.FindWithUserByUserID(ctx, userID)
		if err != nil || mu == nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "初始化会员记录失败", err)
		}
	}

	benefit, err := s.findBenefitByLevelCode(ctx, mu.LevelCode)
	if err != nil {
		return nil, err
	}

	benefits, err := s.findAllBenefits(ctx)
	if err != nil {
		return nil, err
	}

	nextLevelGrowth, progressPercent := calcGrowthProgress(mu.LevelCode, mu.GrowthValue, benefit, benefits)

	profile := &vo.MemberProfileVO{
		UserID:               mu.UserID,
		Username:             mu.Username,
		Nickname:             mu.Nickname,
		Avatar:               mu.Avatar,
		LevelCode:            mu.LevelCode,
		LevelName:            getLevelName(mu.LevelCode),
		GrowthValue:          mu.GrowthValue,
		NextLevelGrowth:      nextLevelGrowth,
		ProgressPercent:      progressPercent,
		ExpireTime:           formatTime(mu.ExpireTime),
		MonthlyDehazeQuota:   mu.MonthlyDehazeQuota,
		MonthlyDehazeUsed:    mu.MonthlyDehazeUsed,
		MonthlyEvaluateQuota: mu.MonthlyEvaluateQuota,
		MonthlyEvaluateUsed:  mu.MonthlyEvaluateUsed,
		Benefits:             toBenefitVO(benefit),
		Status:               int(mu.Status),
	}

	if s.cache != nil {
		if data, err := json.Marshal(profile); err == nil {
			_ = s.cache.Set(ctx, cacheKey, string(data), memberProfileCacheTTL)
		}
		_ = s.cache.Set(ctx, MemberLevelKey(userID), mu.LevelCode, memberProfileCacheTTL)
	}
	return profile, nil
}

func (s *MemberService) findBenefitByLevelCode(ctx context.Context, levelCode string) (*model.SysMemberBenefit, error) {
	cacheKey := MemberBenefitKey(levelCode)
	if s.cache != nil {
		if cached, err := s.cache.Get(ctx, cacheKey); err == nil && cached != "" {
			var b model.SysMemberBenefit
			if err := json.Unmarshal([]byte(cached), &b); err == nil && b.LevelCode != "" {
				return &b, nil
			}
		}
	}
	b, err := s.benefitRepo.FindByLevelCode(ctx, levelCode)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}
	if b != nil && s.cache != nil {
		if data, err := json.Marshal(b); err == nil {
			_ = s.cache.Set(ctx, cacheKey, string(data), memberBenefitCacheTTL)
		}
	}
	return b, nil
}

func (s *MemberService) findAllBenefits(ctx context.Context) ([]model.SysMemberBenefit, error) {
	if s.cache != nil {
		if cached, err := s.cache.Get(ctx, MemberBenefitAllKey()); err == nil && cached != "" {
			var list []model.SysMemberBenefit
			if err := json.Unmarshal([]byte(cached), &list); err == nil && len(list) > 0 {
				return list, nil
			}
		}
	}
	list, err := s.benefitRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}
	if s.cache != nil && len(list) > 0 {
		if data, err := json.Marshal(list); err == nil {
			_ = s.cache.Set(ctx, MemberBenefitAllKey(), string(data), memberBenefitCacheTTL)
		}
	}
	return list, nil
}

// InvalidateMemberCache 失效指定用户的会员相关缓存（profile/level/quota/benefit）
// 供 OrderService 支付完成后调用，避免重复实现
func (s *MemberService) InvalidateMemberCache(ctx context.Context, userID int64, levelCode string) {
	if s.cache == nil {
		return
	}
	_ = s.cache.Delete(ctx, MemberProfileKey(userID))
	_ = s.cache.Delete(ctx, MemberLevelKey(userID))
	_ = s.cache.Delete(ctx, MemberQuotaKey(userID, QuotaTypeDehaze))
	_ = s.cache.Delete(ctx, MemberQuotaKey(userID, QuotaTypeEvaluate))
	if levelCode != "" {
		_ = s.cache.Delete(ctx, MemberBenefitKey(levelCode))
	}
	_ = s.cache.Delete(ctx, MemberBenefitAllKey())
}

func (s *MemberService) ListGrowthLogs(ctx context.Context, userID int64, q *query.GrowthLogQuery) (*vo.PageResult[vo.GrowthLogVO], error) {
	logs, total, err := s.growthLogRepo.FindPageByUserID(ctx, userID, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询成长值流水失败", err)
	}
	list := make([]vo.GrowthLogVO, 0, len(logs))
	for _, l := range logs {
		list = append(list, vo.GrowthLogVO{
			ID:          l.ID,
			ChangeType:  l.ChangeType,
			ChangeValue: l.ChangeValue,
			Balance:     l.Balance,
			RelatedID:   l.RelatedID,
			Reason:      l.Reason,
			OperatorID:  l.OperatorID,
			CreateTime:  l.CreateTime.Format(timeFormat),
		})
	}
	return &vo.PageResult[vo.GrowthLogVO]{List: list, Total: total}, nil
}

func (s *MemberService) SignIn(ctx context.Context, userID int64) (*vo.SignInResultVO, error) {
	now := time.Now()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())

	existing, err := s.signInRepo.FindByUserIDAndDate(ctx, userID, today)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询签到记录失败", err)
	}
	if existing != nil {
		return nil, common.NewBizError(common.SIGN_IN_ALREADY, "今日已签到")
	}

	yesterday := today.AddDate(0, 0, -1)
	yesterdaySign, err := s.signInRepo.FindByUserIDAndDate(ctx, userID, yesterday)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询昨日签到记录失败", err)
	}

	continuousDays := 1
	if yesterdaySign != nil {
		continuousDays = yesterdaySign.ContinuousDays + 1
	}

	// 营销激励参数来自字典：每日签到成长值 + 连续签到额外奖励（缺键回退默认值）
	signInBaseGrowth := dictservice.GetIntValue(ctx, s.dictSvc, memberGrowthRulesDictType, "sign_in_value", 3)
	signInBonusGrowth := dictservice.GetIntValue(ctx, s.dictSvc, memberGrowthRulesDictType, "sign_in_streak_bonus", 20)

	bonusGrowth := int64(0)
	if continuousDays%signInBonusCycle == 0 {
		bonusGrowth = signInBonusGrowth
	}
	totalGrowth := int(signInBaseGrowth + bonusGrowth)

	benefits, err := s.findAllBenefits(ctx)
	if err != nil {
		return nil, err
	}

	member, err := s.memberRepo.FindByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if member == nil {
		if initErr := s.initDefaultMember(ctx, userID); initErr != nil {
			return nil, initErr
		}
		member, err = s.memberRepo.FindByUserID(ctx, userID)
		if err != nil || member == nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "初始化会员记录失败", err)
		}
	}

	newGrowthValue := member.GrowthValue + int64(totalGrowth)

	var newLevel string
	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txMemberRepo := memberrepo.NewMemberRepository(tx)
		txGrowthLogRepo := memberrepo.NewMemberGrowthLogRepository(tx)
		txSignInRepo := memberrepo.NewMemberSignInRepository(tx)

		signRecord := &model.SysMemberSignIn{
			UserID:         userID,
			SignDate:       today,
			ContinuousDays: continuousDays,
			GrowthValue:    totalGrowth,
		}
		if err := txSignInRepo.Create(ctx, signRecord); err != nil {
			return err
		}

		balanceAfterBase := member.GrowthValue + signInBaseGrowth
		if err := txGrowthLogRepo.Create(ctx, &model.SysMemberGrowthLog{
			UserID:      userID,
			ChangeType:  "sign_in",
			ChangeValue: int(signInBaseGrowth),
			Balance:     balanceAfterBase,
			RelatedID:   strconv.FormatInt(signRecord.ID, 10),
		}); err != nil {
			return err
		}

		if bonusGrowth > 0 {
			if err := txGrowthLogRepo.Create(ctx, &model.SysMemberGrowthLog{
				UserID:      userID,
				ChangeType:  "sign_in_bonus",
				ChangeValue: int(bonusGrowth),
				Balance:     newGrowthValue,
				RelatedID:   strconv.FormatInt(signRecord.ID, 10),
			}); err != nil {
				return err
			}
		}

		if err := txMemberRepo.UpdateGrowth(ctx, userID, newGrowthValue); err != nil {
			return err
		}

		newLevel = determineLevelByGrowth(benefits, newGrowthValue)
		if newLevel != member.LevelCode && member.LevelSource == "growth" {
			updates := map[string]interface{}{
				"level_code":   newLevel,
				"level_source": "growth",
			}
			for _, b := range benefits {
				if b.LevelCode == newLevel {
					updates["monthly_dehaze_quota"] = b.MonthlyDehazeQuota
					updates["monthly_evaluate_quota"] = b.MonthlyEvaluateQuota
					break
				}
			}
			if err := txMemberRepo.Update(ctx, userID, updates); err != nil {
				return err
			}
		}

		return nil
	})
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "签到失败", err)
	}

	s.InvalidateMemberCache(ctx, userID, newLevel)

	return &vo.SignInResultVO{
		SignDate:       today.Format(dateFormat),
		ContinuousDays: continuousDays,
		GrowthValue:    int(signInBaseGrowth),
		BonusGrowth:    int(bonusGrowth),
	}, nil
}

func (s *MemberService) GetSignInCalendar(ctx context.Context, userID int64, year, month int) (*vo.SignInCalendarVO, error) {
	start := time.Date(year, time.Month(month), 1, 0, 0, 0, 0, time.Local)
	end := start.AddDate(0, 1, -1)

	records, err := s.signInRepo.FindByUserIDAndDateRange(ctx, userID, start, end)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询签到记录失败", err)
	}

	signDates := make([]string, 0, len(records))
	for _, r := range records {
		signDates = append(signDates, r.SignDate.Format(dateFormat))
	}

	continuousDays := 0
	latest, err := s.signInRepo.FindLatestByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询最新签到记录失败", err)
	}
	if latest != nil {
		today := time.Now().Format(dateFormat)
		if latest.SignDate.Format(dateFormat) == today {
			continuousDays = latest.ContinuousDays
		}
	}

	return &vo.SignInCalendarVO{
		SignDates:      signDates,
		ContinuousDays: continuousDays,
		TotalDays:      len(records),
	}, nil
}

func (s *MemberService) ListPagedMembers(ctx context.Context, q *query.MemberPageQuery) (*vo.PageResult[vo.MemberPageVO], error) {
	list, total, err := s.memberRepo.FindPageWithUser(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询会员列表失败", err)
	}
	vos := make([]vo.MemberPageVO, 0, len(list))
	for _, m := range list {
		vos = append(vos, vo.MemberPageVO{
			UserID:           m.UserID,
			Username:         m.Username,
			Nickname:         m.Nickname,
			LevelCode:        m.LevelCode,
			LevelName:        getLevelName(m.LevelCode),
			GrowthValue:      m.GrowthValue,
			MonthlyUsed:      m.MonthlyDehazeUsed + m.MonthlyEvaluateUsed,
			ExpireTime:       formatTime(m.ExpireTime),
			Status:           int(m.Status),
			BecomeMemberTime: formatTime(m.BecomeMemberTime),
		})
	}
	return &vo.PageResult[vo.MemberPageVO]{List: vos, Total: total}, nil
}

func (s *MemberService) GetMemberDetail(ctx context.Context, userID int64) (*vo.MemberDetailVO, error) {
	mu, err := s.memberRepo.FindWithUserByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if mu == nil {
		return nil, common.NewBizError(common.MEMBER_NOT_FOUND, "会员不存在")
	}

	benefit, err := s.findBenefitByLevelCode(ctx, mu.LevelCode)
	if err != nil {
		return nil, err
	}

	benefits, err := s.findAllBenefits(ctx)
	if err != nil {
		return nil, err
	}

	nextLevelGrowth, progressPercent := calcGrowthProgress(mu.LevelCode, mu.GrowthValue, benefit, benefits)

	quotaResetMonth := 0
	if mu.QuotaResetMonth != nil {
		quotaResetMonth = *mu.QuotaResetMonth
	}

	return &vo.MemberDetailVO{
		MemberProfileVO: vo.MemberProfileVO{
			UserID:               mu.UserID,
			Username:             mu.Username,
			Nickname:             mu.Nickname,
			Avatar:               mu.Avatar,
			LevelCode:            mu.LevelCode,
			LevelName:            getLevelName(mu.LevelCode),
			GrowthValue:          mu.GrowthValue,
			NextLevelGrowth:      nextLevelGrowth,
			ProgressPercent:      progressPercent,
			ExpireTime:           formatTime(mu.ExpireTime),
			MonthlyDehazeQuota:   mu.MonthlyDehazeQuota,
			MonthlyDehazeUsed:    mu.MonthlyDehazeUsed,
			MonthlyEvaluateQuota: mu.MonthlyEvaluateQuota,
			MonthlyEvaluateUsed:  mu.MonthlyEvaluateUsed,
			Benefits:             toBenefitVO(benefit),
			Status:               int(mu.Status),
		},
		LevelSource:      mu.LevelSource,
		TotalConsumption: mu.TotalConsumption,
		BecomeMemberTime: formatTime(mu.BecomeMemberTime),
		FrozenReason:     mu.FrozenReason,
		FrozenTime:       formatTime(mu.FrozenTime),
		QuotaResetMonth:  quotaResetMonth,
	}, nil
}

func (s *MemberService) AdjustLevel(ctx context.Context, userID, operatorID int64, form *bo.MemberLevelAdjustForm) error {
	if form.Reason == "" {
		return common.NewBizError(common.PARAM_ERROR, "调整原因必填")
	}

	member, err := s.memberRepo.FindByUserID(ctx, userID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if member == nil {
		return common.NewBizError(common.MEMBER_NOT_FOUND, "会员不存在")
	}

	updates := map[string]interface{}{
		"level_code":   form.LevelCode,
		"level_source": "admin",
		"update_by":    operatorID,
	}

	if form.ExpireTime != nil && *form.ExpireTime != "" {
		t, err := time.ParseInLocation(timeFormat, *form.ExpireTime, time.Local)
		if err == nil {
			updates["expire_time"] = t
		} else {
			updates["expire_time"] = nil
		}
	} else {
		updates["expire_time"] = nil
	}

	if member.BecomeMemberTime == nil {
		updates["become_member_time"] = time.Now()
	}

	benefit, err := s.findBenefitByLevelCode(ctx, form.LevelCode)
	if err != nil {
		return err
	}
	if benefit != nil {
		updates["monthly_dehaze_quota"] = benefit.MonthlyDehazeQuota
		updates["monthly_evaluate_quota"] = benefit.MonthlyEvaluateQuota
	}

	if err := s.memberRepo.UpdateLevel(ctx, userID, updates); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "等级调整失败", err)
	}
	s.InvalidateMemberCache(ctx, userID, form.LevelCode)
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, operatorID, "member", userID, "level_change", "member", member.LevelCode, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return nil
}

func (s *MemberService) AdjustGrowth(ctx context.Context, userID, operatorID int64, form *bo.MemberGrowthAdjustForm) error {
	if form.Reason == "" {
		return common.NewBizError(common.PARAM_ERROR, "调整原因必填")
	}
	if form.ChangeValue == 0 {
		return common.NewBizError(common.PARAM_ERROR, "变动值不能为0")
	}

	member, err := s.memberRepo.FindByUserID(ctx, userID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if member == nil {
		return common.NewBizError(common.MEMBER_NOT_FOUND, "会员不存在")
	}

	newGrowth := member.GrowthValue + int64(form.ChangeValue)
	if newGrowth < 0 {
		newGrowth = 0
	}

	benefits, err := s.findAllBenefits(ctx)
	if err != nil {
		return err
	}

	actualChange := int(newGrowth - member.GrowthValue)

	var newLevel string
	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txMemberRepo := memberrepo.NewMemberRepository(tx)
		txGrowthLogRepo := memberrepo.NewMemberGrowthLogRepository(tx)

		if err := txMemberRepo.UpdateGrowth(ctx, userID, newGrowth); err != nil {
			return err
		}

		if err := txGrowthLogRepo.Create(ctx, &model.SysMemberGrowthLog{
			UserID:      userID,
			ChangeType:  "admin_adjust",
			ChangeValue: actualChange,
			Balance:     newGrowth,
			Reason:      form.Reason,
			OperatorID:  &operatorID,
		}); err != nil {
			return err
		}

		newLevel = determineLevelByGrowth(benefits, newGrowth)
		if newLevel != member.LevelCode && member.LevelSource == "growth" {
			updates := map[string]interface{}{
				"level_code":   newLevel,
				"level_source": "growth",
			}
			for _, b := range benefits {
				if b.LevelCode == newLevel {
					updates["monthly_dehaze_quota"] = b.MonthlyDehazeQuota
					updates["monthly_evaluate_quota"] = b.MonthlyEvaluateQuota
					break
				}
			}
			if err := txMemberRepo.Update(ctx, userID, updates); err != nil {
				return err
			}
		}

		return nil
	})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "成长值调整失败", err)
	}
	s.InvalidateMemberCache(ctx, userID, newLevel)
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, operatorID, "member", userID, "growth_change", "member", member.GrowthValue, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return nil
}

func (s *MemberService) UpdateStatus(ctx context.Context, userID int64, form *bo.MemberStatusForm) error {
	if form.Status == 0 && form.Reason == "" {
		return common.NewBizError(common.PARAM_ERROR, "冻结原因必填")
	}

	member, err := s.memberRepo.FindByUserID(ctx, userID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if member == nil {
		return common.NewBizError(common.MEMBER_NOT_FOUND, "会员不存在")
	}

	if form.Status == 0 {
		now := time.Now()
		if err := s.memberRepo.Update(ctx, userID, map[string]interface{}{
			"status":        0,
			"frozen_reason": form.Reason,
			"frozen_time":   now,
		}); err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "状态更新失败", err)
		}
		s.InvalidateMemberCache(ctx, userID, member.LevelCode)
		if s.auditLogSvc != nil {
			s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "member", userID, "status_change", "member", member.Status, form, database.GetIP(ctx), database.GetUserAgent(ctx))
		}
		return nil
	}

	if err := s.memberRepo.Update(ctx, userID, map[string]interface{}{
		"status": 1,
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "状态更新失败", err)
	}
	s.InvalidateMemberCache(ctx, userID, member.LevelCode)
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "member", userID, "status_change", "member", member.Status, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return nil
}

func (s *MemberService) ListBenefits(ctx context.Context) ([]vo.BenefitVO, error) {
	list, err := s.findAllBenefits(ctx)
	if err != nil {
		return nil, err
	}
	vos := make([]vo.BenefitVO, 0, len(list))
	for i := range list {
		vos = append(vos, toBenefitVO(&list[i]))
	}
	return vos, nil
}

func (s *MemberService) UpdateBenefit(ctx context.Context, levelCode string, form *bo.BenefitForm) error {
	updates := make(map[string]interface{})
	if form.LevelName != nil {
		updates["level_name"] = *form.LevelName
	}
	if form.GrowthMin != nil {
		updates["growth_min"] = *form.GrowthMin
	}
	if form.GrowthMax != nil {
		updates["growth_max"] = *form.GrowthMax
	}
	if form.MonthlyDehazeQuota != nil {
		updates["monthly_dehaze_quota"] = *form.MonthlyDehazeQuota
	}
	if form.MonthlyEvaluateQuota != nil {
		updates["monthly_evaluate_quota"] = *form.MonthlyEvaluateQuota
	}
	if form.HistoryRetention != nil {
		updates["history_retention"] = *form.HistoryRetention
	}
	if form.BatchLimit != nil {
		updates["batch_limit"] = *form.BatchLimit
	}
	if form.Priority != nil {
		updates["priority"] = *form.Priority
	}
	if form.AdvancedParams != nil {
		updates["advanced_params"] = *form.AdvancedParams
	}
	if form.HdExport != nil {
		updates["hd_export"] = *form.HdExport
	}
	if form.ReportExport != nil {
		updates["report_export"] = *form.ReportExport
	}
	if form.BatchDownload != nil {
		updates["batch_download"] = *form.BatchDownload
	}
	if form.Sort != nil {
		updates["sort"] = *form.Sort
	}
	if form.Status != nil {
		updates["status"] = *form.Status
	}

	if len(updates) == 0 {
		return nil
	}

	current, err := s.benefitRepo.FindByLevelCode(ctx, levelCode)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}
	if current == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "权益配置不存在")
	}
	effectiveGrowthMin := current.GrowthMin
	if form.GrowthMin != nil {
		effectiveGrowthMin = *form.GrowthMin
	}
	effectiveGrowthMax := current.GrowthMax
	if form.GrowthMax != nil {
		effectiveGrowthMax = *form.GrowthMax
	}
	if effectiveGrowthMax > 0 && effectiveGrowthMin > effectiveGrowthMax {
		return common.NewBizError(common.BENEFIT_CONFIG_INVALID, "成长值下限不能大于上限")
	}

	if err := s.benefitRepo.Update(ctx, levelCode, updates); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "权益配置更新失败", err)
	}
	if s.cache != nil {
		_ = s.cache.Delete(ctx, MemberBenefitKey(levelCode))
		_ = s.cache.Delete(ctx, MemberBenefitAllKey())
		if userIDs, err := s.memberRepo.FindUserIDsByLevelCodes(ctx, []string{levelCode}); err == nil {
			for _, uid := range userIDs {
				_ = s.cache.Delete(ctx, MemberQuotaKey(uid, QuotaTypeDehaze))
				_ = s.cache.Delete(ctx, MemberQuotaKey(uid, QuotaTypeEvaluate))
			}
		}
	}
	return nil
}

func (s *MemberService) AwardGrowth(ctx context.Context, userID int64, changeType string, changeValue int, reason, relatedID string) error {
	if changeValue == 0 {
		return nil
	}

	member, err := s.memberRepo.FindByUserID(ctx, userID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if member == nil {
		return common.NewBizError(common.MEMBER_NOT_FOUND, "会员不存在")
	}

	newGrowth := member.GrowthValue + int64(changeValue)
	if newGrowth < 0 {
		newGrowth = 0
	}

	benefits, err := s.findAllBenefits(ctx)
	if err != nil {
		return err
	}

	var newLevel string
	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txMemberRepo := memberrepo.NewMemberRepository(tx)
		txGrowthLogRepo := memberrepo.NewMemberGrowthLogRepository(tx)

		if err := txMemberRepo.UpdateGrowth(ctx, userID, newGrowth); err != nil {
			return err
		}

		if err := txGrowthLogRepo.Create(ctx, &model.SysMemberGrowthLog{
			UserID:      userID,
			ChangeType:  changeType,
			ChangeValue: changeValue,
			Balance:     newGrowth,
			RelatedID:   relatedID,
			Reason:      reason,
		}); err != nil {
			return err
		}

		newLevel = determineLevelByGrowth(benefits, newGrowth)
		if newLevel != member.LevelCode && member.LevelSource == "growth" {
			updates := map[string]interface{}{
				"level_code":   newLevel,
				"level_source": "growth",
			}
			for _, b := range benefits {
				if b.LevelCode == newLevel {
					updates["monthly_dehaze_quota"] = b.MonthlyDehazeQuota
					updates["monthly_evaluate_quota"] = b.MonthlyEvaluateQuota
					break
				}
			}
			if err := txMemberRepo.Update(ctx, userID, updates); err != nil {
				return err
			}
		}

		return nil
	})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "成长值奖励失败", err)
	}
	s.InvalidateMemberCache(ctx, userID, newLevel)
	return nil
}

// CheckAndDeductQuota 校验并扣减会员配额（使用 Redis DECR 原子操作，异步落库）
func (s *MemberService) CheckAndDeductQuota(ctx context.Context, userID int64, quotaType QuotaType) error {
	member, err := s.memberRepo.FindByUserID(ctx, userID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if member == nil {
		return common.NewBizError(common.MEMBER_NOT_FOUND, "会员不存在")
	}
	if member.Status != 1 {
		return common.NewBizError(common.MEMBER_FROZEN, "会员已冻结")
	}

	var quota, used int
	if quotaType == QuotaTypeDehaze {
		quota = member.MonthlyDehazeQuota
		used = member.MonthlyDehazeUsed
	} else {
		quota = member.MonthlyEvaluateQuota
		used = member.MonthlyEvaluateUsed
	}

	if quota <= 0 {
		return common.NewBizError(common.QUOTA_EXCEEDED, "配额已用尽")
	}

	if s.cache != nil {
		counterKey := MemberQuotaKey(userID, quotaType)
		remainingStr, err := s.cache.Get(ctx, counterKey)
		if err != nil || remainingStr == "" {
			remaining := int64(quota - used)
			if remaining <= 0 {
				return common.NewBizError(common.QUOTA_EXCEEDED, "配额已用尽")
			}
			_ = s.cache.Set(ctx, counterKey, remaining, quotaCounterCacheTTL)
		}

		newVal, decErr := s.cache.Decr(ctx, counterKey)
		if decErr != nil {
			logger.Warn("Redis DECR 配额扣减失败，降级为数据库校验", zap.Int64("userID", userID), zap.String("quotaType", string(quotaType)), zap.Error(decErr))
		} else {
			if newVal < 0 {
				_, _ = s.cache.Incr(ctx, counterKey)
				return common.NewBizError(common.QUOTA_EXCEEDED, "配额已用尽")
			}
			s.lifecycle.Go(func(ctx context.Context) {
			s.asyncPersistQuotaUsed(ctx, userID, quotaType)
		})
			if s.auditLogSvc != nil {
				s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "member", userID, "quota_deduct", "member", nil, map[string]interface{}{"quotaType": string(quotaType), "amount": 1}, database.GetIP(ctx), database.GetUserAgent(ctx))
			}
			return nil
		}
	}

	// DB 同步路径：应用层预校验仅作快速失败，权威扣减依赖行级条件更新
	// （used < quota 原子判定），高并发下预校验读到的快照可能过期，超扣在此被拦截。
	if used >= quota {
		return common.NewBizError(common.QUOTA_EXCEEDED, "配额已用尽")
	}
	newUsed, deducted, err := s.memberRepo.DeductQuotaIfAvailable(ctx, userID, string(quotaType))
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "扣减配额失败", err)
	}
	if !deducted {
		return common.NewBizError(common.QUOTA_EXCEEDED, "配额已用尽")
	}
	if s.cache != nil {
		// DB 权威扣减成功后，缓存计数器写入 DB 精确剩余值，避免 Delete 造成缓存击穿/重建风暴；
		// 缓存写失败仅告警，账目以 DB 为准。
		if err := s.cache.Set(ctx, MemberQuotaKey(userID, quotaType), int64(quota-newUsed), quotaCounterCacheTTL); err != nil {
			logger.Warn("DB 扣减后缓存计数器对齐失败", zap.Int64("userID", userID), zap.String("quotaType", string(quotaType)), zap.Error(err))
		}
	}
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "member", userID, "quota_deduct", "member", nil, map[string]interface{}{"quotaType": string(quotaType), "amount": 1}, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return nil
}

func (s *MemberService) asyncPersistQuotaUsed(ctx context.Context, userID int64, quotaType QuotaType) {
	if err := s.memberRepo.IncrementQuotaUsed(ctx, userID, string(quotaType), 1); err != nil {
		logger.Error("异步落库配额扣减失败", zap.Int64("userID", userID), zap.String("quotaType", string(quotaType)), zap.Error(err))
	}
}

// RefundQuota 回补会员配额（预测/评估失败时调用，回补缓存计数器并异步回补数据库）
func (s *MemberService) RefundQuota(ctx context.Context, userID int64, quotaType QuotaType) error {
	if s.cache != nil {
		counterKey := MemberQuotaKey(userID, quotaType)
		if _, err := s.cache.Incr(ctx, counterKey); err != nil {
			logger.Warn("回补配额 Incr 失败", zap.Int64("userID", userID), zap.String("quotaType", string(quotaType)), zap.Error(err))
		}
	}
	if err := s.memberRepo.IncrementQuotaUsed(ctx, userID, string(quotaType), -1); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "回补配额失败", err)
	}
	return nil
}

// ResetMonthlyQuota 重置所有活跃会员的月度配额（每月初执行）
func (s *MemberService) ResetMonthlyQuota(ctx context.Context) error {
	now := time.Now()
	quotaMonth := now.Year()*100 + int(now.Month())

	benefits, err := s.findAllBenefits(ctx)
	if err != nil {
		return err
	}
	benefitMap := make(map[string]*model.SysMemberBenefit, len(benefits))
	for i := range benefits {
		benefitMap[benefits[i].LevelCode] = &benefits[i]
	}

	batchSize := 500
	totalCount := 0
	successCount := 0

	for {
		members, err := s.memberRepo.FindAllActive(ctx, &quotaMonth, batchSize)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询活跃会员失败", err)
		}
		if len(members) == 0 {
			break
		}

		for _, m := range members {
			totalCount++
			dehazeQuota := 0
			evaluateQuota := 0
			if b, ok := benefitMap[m.LevelCode]; ok {
				dehazeQuota = b.MonthlyDehazeQuota
				evaluateQuota = b.MonthlyEvaluateQuota
			}

			if m.QuotaResetMonth != nil {
				archive := &model.SysMemberQuota{
					UserID:        m.UserID,
					QuotaMonth:    *m.QuotaResetMonth,
					LevelCode:     m.LevelCode,
					DehazeQuota:   m.MonthlyDehazeQuota,
					DehazeUsed:    m.MonthlyDehazeUsed,
					EvaluateQuota: m.MonthlyEvaluateQuota,
					EvaluateUsed:  m.MonthlyEvaluateUsed,
					ResetTime:     now,
				}
				_ = s.memberRepo.CreateQuotaArchive(ctx, archive)
			}

			if err := s.memberRepo.ResetMonthlyQuota(ctx, m.UserID, dehazeQuota, evaluateQuota, quotaMonth); err != nil {
				logger.Error("重置会员月度配额失败", zap.Int64("userID", m.UserID), zap.Error(err))
				continue
			}

			if s.cache != nil {
				_ = s.cache.Delete(ctx, MemberQuotaKey(m.UserID, QuotaTypeDehaze))
				_ = s.cache.Delete(ctx, MemberQuotaKey(m.UserID, QuotaTypeEvaluate))
				s.InvalidateMemberCache(ctx, m.UserID, m.LevelCode)
			}
			successCount++
		}
	}

	logger.Debug("月度配额重置完成", zap.Int("total", totalCount), zap.Int("success", successCount))
	return nil
}

// ProcessExpiredMembers 处理已过期会员降级
// 扫描 expire_time < NOW() AND level_source != 'growth' 的会员，
// 按成长值重算等级、置 level_source=growth、清空 expire_time、刷新权益。
func (s *MemberService) ProcessExpiredMembers(ctx context.Context) error {
	now := time.Now()
	members, err := s.memberRepo.FindExpiredNonGrowth(ctx, now)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询过期会员失败", err)
	}
	if len(members) == 0 {
		return nil
	}

	benefits, err := s.findAllBenefits(ctx)
	if err != nil {
		return err
	}
	benefitMap := make(map[string]*model.SysMemberBenefit, len(benefits))
	for i := range benefits {
		benefitMap[benefits[i].LevelCode] = &benefits[i]
	}

	successCount := 0
	for _, m := range members {
		oldLevel := m.LevelCode
		newLevel := determineLevelByGrowth(benefits, m.GrowthValue)
		updates := map[string]interface{}{
			"level_code":   newLevel,
			"level_source": "growth",
			"expire_time":  nil,
		}
		if b, ok := benefitMap[newLevel]; ok {
			updates["monthly_dehaze_quota"] = b.MonthlyDehazeQuota
			updates["monthly_evaluate_quota"] = b.MonthlyEvaluateQuota
		}
		if err := s.memberRepo.Update(ctx, m.UserID, updates); err != nil {
			logger.Error("会员过期降级失败", zap.Int64("userID", m.UserID), zap.Error(err))
			continue
		}
		s.InvalidateMemberCache(ctx, m.UserID, oldLevel)
		s.InvalidateMemberCache(ctx, m.UserID, newLevel)
		successCount++
	}

	logger.Debug("会员过期降级处理完成", zap.Int("total", len(members)), zap.Int("success", successCount))
	return nil
}

func (s *MemberService) SendExpireReminders(ctx context.Context) error {
	if s.messageSender == nil {
		logger.Warn("消息发送服务未注入，跳过会员到期预警")
		return nil
	}

	now := time.Now()
	benefits, err := s.findAllBenefits(ctx)
	if err != nil {
		return err
	}
	benefitMap := make(map[string]*model.SysMemberBenefit, len(benefits))
	for i := range benefits {
		benefitMap[benefits[i].LevelCode] = &benefits[i]
	}

	dayTemplateMap := map[int]struct {
		bizPrefix    string
		templateCode string
	}{
		7: {"expire_reminder_7d", "member_expire_reminder_7"},
		3: {"expire_reminder_3d", "member_expire_reminder_3"},
		1: {"expire_reminder_1d", "member_expire_reminder_1"},
	}

	sentCount := 0
	for days, cfg := range dayTemplateMap {
		windowStart := time.Date(now.Year(), now.Month(), now.Day()+days, 0, 0, 0, 0, now.Location())
		windowEnd := windowStart.AddDate(0, 0, 1)

		members, err := s.memberRepo.FindExpiringBetween(ctx, windowStart, windowEnd)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询到期预警会员失败", err)
		}
		if len(members) == 0 {
			continue
		}

		for _, m := range members {
			currentBenefit := benefitMap[m.LevelCode]
			currentLevelName := m.LevelCode
			if currentBenefit != nil {
				currentLevelName = currentBenefit.LevelName
			}

			variables := map[string]string{
				"currentLevel": currentLevelName,
				"days":         fmt.Sprintf("%d", days),
				"expireDate":   "",
			}
			if m.ExpireTime != nil {
				variables["expireDate"] = m.ExpireTime.Format(dateFormat)
			}

			if days == 3 {
				targetLevel := determineLevelByGrowth(benefits, m.GrowthValue)
				downgradeBenefit := benefitMap[targetLevel]
				downgradeName := targetLevel
				if downgradeBenefit != nil {
					downgradeName = downgradeBenefit.LevelName
				}
				variables["downgradeLevel"] = downgradeName
				if currentBenefit != nil && downgradeBenefit != nil {
					variables["benefitCompare"] = fmt.Sprintf(
						"去雾:%d→%d次/月，评估:%d→%d次/月",
						currentBenefit.MonthlyDehazeQuota, downgradeBenefit.MonthlyDehazeQuota,
						currentBenefit.MonthlyEvaluateQuota, downgradeBenefit.MonthlyEvaluateQuota,
					)
				} else {
					variables["benefitCompare"] = ""
				}
			}

			form := &bo.MessageSendForm{
				Type:         "member",
				RecipientIDs: []int64{m.UserID},
				BizModule:    "member",
				BizID:        fmt.Sprintf("%s:%d:%s", cfg.bizPrefix, m.UserID, now.Format(dateFormat)),
				TemplateCode: cfg.templateCode,
				Variables:    variables,
			}
			if _, err := s.messageSender.Send(ctx, form); err != nil {
				logger.Warn("到期提醒发送失败", zap.Int64("userID", m.UserID), zap.Int("days", days), zap.Error(err))
				continue
			}
			sentCount++
		}
	}

	logger.Debug("会员到期预警完成", zap.Int("sent", sentCount))
	return nil
}

func getLevelName(levelCode string) string {
	if name, ok := levelNames[levelCode]; ok {
		return name
	}
	return levelCode
}

func formatTime(t *time.Time) string {
	if t == nil || t.IsZero() {
		return ""
	}
	return t.Format(timeFormat)
}

func toBenefitVO(b *model.SysMemberBenefit) vo.BenefitVO {
	if b == nil {
		return vo.BenefitVO{}
	}
	return vo.BenefitVO{
		LevelCode:            b.LevelCode,
		LevelName:            b.LevelName,
		GrowthMin:            b.GrowthMin,
		GrowthMax:            b.GrowthMax,
		MonthlyDehazeQuota:   b.MonthlyDehazeQuota,
		MonthlyEvaluateQuota: b.MonthlyEvaluateQuota,
		HistoryRetention:     b.HistoryRetention,
		BatchLimit:           b.BatchLimit,
		Priority:             int(b.Priority),
		AdvancedParams:       int(b.AdvancedParams),
		HdExport:             int(b.HdExport),
		ReportExport:         int(b.ReportExport),
		BatchDownload:        int(b.BatchDownload),
		Sort:                 b.Sort,
		Status:               int(b.Status),
	}
}

func calcGrowthProgress(levelCode string, growthValue int64, benefit *model.SysMemberBenefit, allBenefits []model.SysMemberBenefit) (nextLevelGrowth int64, progressPercent int) {
	if benefit == nil {
		return 0, 0
	}

	if benefit.GrowthMax == 0 {
		return 0, 100
	}

	rangeSize := benefit.GrowthMax - benefit.GrowthMin
	if rangeSize <= 0 {
		return 0, 100
	}

	percent := (growthValue - benefit.GrowthMin) * 100 / rangeSize
	if percent < 0 {
		percent = 0
	}
	if percent > 100 {
		percent = 100
	}

	for i, b := range allBenefits {
		if b.LevelCode == levelCode && i+1 < len(allBenefits) {
			next := allBenefits[i+1]
			nextLevelGrowth = next.GrowthMin - growthValue
			if nextLevelGrowth < 0 {
				nextLevelGrowth = 0
			}
			break
		}
	}

	return nextLevelGrowth, int(percent)
}

func determineLevelByGrowth(benefits []model.SysMemberBenefit, growthValue int64) string {
	result := "level_0"
	for _, b := range benefits {
		if growthValue >= b.GrowthMin {
			if b.GrowthMax == 0 || growthValue <= b.GrowthMax {
				result = b.LevelCode
			}
		}
	}
	return result
}

// GetLevelCode 获取用户会员等级代码
func (s *MemberService) GetLevelCode(ctx context.Context, userID int64) (string, error) {
	member, err := s.memberRepo.FindByUserID(ctx, userID)
	if err != nil {
		return "", common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if member == nil {
		return "level_0", nil
	}
	return member.LevelCode, nil
}

// GetBatchLimit 根据等级代码获取批量处理上限
func (s *MemberService) GetBatchLimit(ctx context.Context, levelCode string) (int, error) {
	benefit, err := s.findBenefitByLevelCode(ctx, levelCode)
	if err != nil {
		return 0, err
	}
	if benefit == nil {
		return 0, nil
	}
	return benefit.BatchLimit, nil
}

// InitDefaultMember 为新用户初始化默认会员记录（level_0）
func (s *MemberService) InitDefaultMember(ctx context.Context, userID int64) error {
	return s.initDefaultMember(ctx, userID)
}

func (s *MemberService) initDefaultMember(ctx context.Context, userID int64) error {
	benefit, err := s.benefitRepo.FindByLevelCode(ctx, "level_0")
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询默认权益失败", err)
	}
	now := time.Now()
	quotaMonth := now.Year()*100 + int(now.Month())
	member := &model.SysMember{
		UserID:               userID,
		LevelCode:            "level_0",
		LevelSource:          "growth",
		GrowthValue:          0,
		TotalConsumption:     0,
		Status:               1,
		MonthlyDehazeQuota:   0,
		MonthlyEvaluateQuota: 0,
		MonthlyDehazeUsed:    0,
		MonthlyEvaluateUsed:  0,
		QuotaResetMonth:      &quotaMonth,
	}
	if benefit != nil {
		member.MonthlyDehazeQuota = benefit.MonthlyDehazeQuota
		member.MonthlyEvaluateQuota = benefit.MonthlyEvaluateQuota
	}
	if err := s.memberRepo.Upsert(ctx, member); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建会员记录失败", err)
	}
	return nil
}

var _ IMemberService = (*MemberService)(nil)
