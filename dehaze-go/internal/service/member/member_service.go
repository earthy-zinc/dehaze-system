package member

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

const (
	signInBaseGrowth   = 3
	signInBonusGrowth  = 20
	signInBonusCycle   = 7
	timeFormat         = "2006-01-02 15:04:05"
	dateFormat         = "2006-01-02"
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
}

func NewMemberService(
	db *gorm.DB,
	memberRepo memberrepo.IMemberRepository,
	benefitRepo memberrepo.IMemberBenefitRepository,
	growthLogRepo memberrepo.IMemberGrowthLogRepository,
	signInRepo memberrepo.IMemberSignInRepository,
) *MemberService {
	return &MemberService{
		db:            db,
		memberRepo:    memberRepo,
		benefitRepo:   benefitRepo,
		growthLogRepo: growthLogRepo,
		signInRepo:    signInRepo,
	}
}

func (s *MemberService) GetProfile(ctx context.Context, userID int64) (*vo.MemberProfileVO, error) {
	mu, err := s.memberRepo.FindWithUserByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if mu == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会员不存在")
	}

	benefit, err := s.benefitRepo.FindByLevelCode(ctx, mu.LevelCode)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}

	benefits, err := s.benefitRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}

	nextLevelGrowth, progressPercent := calcGrowthProgress(mu.LevelCode, mu.GrowthValue, benefit, benefits)

	return &vo.MemberProfileVO{
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
	}, nil
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

	bonusGrowth := 0
	if continuousDays%signInBonusCycle == 0 {
		bonusGrowth = signInBonusGrowth
	}
	totalGrowth := signInBaseGrowth + bonusGrowth

	benefits, err := s.benefitRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}

	member, err := s.memberRepo.FindByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询会员信息失败", err)
	}
	if member == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会员不存在")
	}

	newGrowthValue := member.GrowthValue + int64(totalGrowth)

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

		balanceAfterBase := member.GrowthValue + int64(signInBaseGrowth)
		if err := txGrowthLogRepo.Create(ctx, &model.SysMemberGrowthLog{
			UserID:      userID,
			ChangeType:  "sign_in",
			ChangeValue: signInBaseGrowth,
			Balance:     balanceAfterBase,
			RelatedID:   formatInt64(signRecord.ID),
		}); err != nil {
			return err
		}

		if bonusGrowth > 0 {
			if err := txGrowthLogRepo.Create(ctx, &model.SysMemberGrowthLog{
				UserID:      userID,
				ChangeType:  "sign_in_bonus",
				ChangeValue: bonusGrowth,
				Balance:     newGrowthValue,
				RelatedID:   formatInt64(signRecord.ID),
			}); err != nil {
				return err
			}
		}

		if err := txMemberRepo.UpdateGrowth(ctx, userID, newGrowthValue); err != nil {
			return err
		}

		newLevel := determineLevelByGrowth(benefits, newGrowthValue)
		if newLevel != member.LevelCode {
			if err := txMemberRepo.Update(ctx, userID, map[string]interface{}{
				"level_code":   newLevel,
				"level_source": "growth",
			}); err != nil {
				return err
			}
		}

		return nil
	})
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "签到失败", err)
	}

	return &vo.SignInResultVO{
		SignDate:       today.Format(dateFormat),
		ContinuousDays: continuousDays,
		GrowthValue:    totalGrowth,
		BonusGrowth:    bonusGrowth,
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
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会员不存在")
	}

	benefit, err := s.benefitRepo.FindByLevelCode(ctx, mu.LevelCode)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}

	benefits, err := s.benefitRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
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
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "会员不存在")
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
		}
	}

	if member.BecomeMemberTime == nil {
		updates["become_member_time"] = time.Now()
	}

	return s.memberRepo.UpdateLevel(ctx, userID, updates)
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
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "会员不存在")
	}

	newGrowth := member.GrowthValue + int64(form.ChangeValue)
	if newGrowth < 0 {
		newGrowth = 0
	}

	benefits, err := s.benefitRepo.FindAll(ctx)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}

	actualChange := int(newGrowth - member.GrowthValue)

	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
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

		newLevel := determineLevelByGrowth(benefits, newGrowth)
		if newLevel != member.LevelCode {
			if err := txMemberRepo.Update(ctx, userID, map[string]interface{}{
				"level_code":   newLevel,
				"level_source": "growth",
			}); err != nil {
				return err
			}
		}

		return nil
	})
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
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "会员不存在")
	}

	if form.Status == 0 {
		now := time.Now()
		return s.memberRepo.UpdateStatus(ctx, userID, map[string]interface{}{
			"status":        0,
			"frozen_reason": form.Reason,
			"frozen_time":   now,
		})
	}

	return s.memberRepo.UpdateStatus(ctx, userID, map[string]interface{}{
		"status": 1,
	})
}

func (s *MemberService) ListBenefits(ctx context.Context) ([]vo.BenefitVO, error) {
	list, err := s.benefitRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}
	vos := make([]vo.BenefitVO, 0, len(list))
	for _, b := range list {
		vos = append(vos, toBenefitVO(&b))
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

	return s.benefitRepo.Update(ctx, levelCode, updates)
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

func formatInt64(n int64) string {
	return time.Now().Format("") + intToStr(n)
}

func intToStr(n int64) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
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

var _ IMemberService = (*MemberService)(nil)
