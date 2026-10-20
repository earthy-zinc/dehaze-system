package ai

import (
	"context"
	"time"

	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	memberrepo "github.com/earthyzinc/clarity-go/internal/repository/member"
	"github.com/earthyzinc/clarity-go/pkg/logger"
	"go.uber.org/zap"
)

// vipGiftBatchSize VIP 月度赠送分页扫描批量大小（对齐 python `list_active_by_level` limit=500）
const vipGiftBatchSize = 500

// VipGiftService 会员月度积分赠送：按等级权益 vip_gift_credits 发放，月末清零未用赠送积分。
// 承接原会员月度配额权益（设计 §9.1：会员权益改为月度积分赠送，复用 vip_gift 积分来源）。
type VipGiftService struct {
	billing  *BillingService
	credits  *airepo.BillingRepository
	members  *memberrepo.MemberRepository
	benefits *memberrepo.MemberBenefitRepository
}

func NewVipGiftService(
	billing *BillingService,
	credits *airepo.BillingRepository,
	members *memberrepo.MemberRepository,
	benefits *memberrepo.MemberBenefitRepository,
) *VipGiftService {
	return &VipGiftService{billing: billing, credits: credits, members: members, benefits: benefits}
}

// vipGiftMonthRange 当前自然月时间边界（月初 00:00:00 ~ 月末 23:59:59）
func vipGiftMonthRange(now time.Time) (time.Time, time.Time) {
	monthStart := time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, now.Location())
	monthEnd := monthStart.AddDate(0, 1, 0).Add(-time.Second)
	return monthStart, monthEnd
}

// GrantVipMonthlyGift 每月 1 日扫描配置了 vip_gift_credits(>0) 的启用等级，逐等级分页发放赠送积分。
func (s *VipGiftService) GrantVipMonthlyGift(ctx context.Context) error {
	benefits, err := s.benefits.FindAll(ctx)
	if err != nil {
		return err
	}

	granted := 0
	giftLevelCount := 0
	for _, benefit := range benefits {
		if benefit.Status != 1 || benefit.VipGiftCredits <= 0 {
			continue
		}
		giftLevelCount++
		amount := benefit.VipGiftCredits
		offset := 0
		for {
			userIDs, err := s.members.FindActiveUserIDsByLevelCode(ctx, benefit.LevelCode, offset, vipGiftBatchSize)
			if err != nil {
				logger.Error("查询活跃会员失败", zap.String("levelCode", benefit.LevelCode), zap.Error(err))
				break
			}
			if len(userIDs) == 0 {
				break
			}
			for _, userID := range userIDs {
				if err := s.billing.GrantVipMonthlyGiftCredits(ctx, userID, amount); err != nil {
					logger.Warn("VIP月度赠送失败",
						zap.Int64("userID", userID),
						zap.String("levelCode", benefit.LevelCode),
						zap.Error(err))
					continue
				}
				granted++
			}
			offset += len(userIDs)
		}
	}

	logger.Info("VIP月度赠送完成", zap.Int("levels", giftLevelCount), zap.Int("users", granted))
	return nil
}

// ClearVipGiftExpire 每月最后一天 23:59 清零当月 VIP 赠送（source=vip_gift）未用部分。
func (s *VipGiftService) ClearVipGiftExpire(ctx context.Context) error {
	now := time.Now()
	monthStart, monthEnd := vipGiftMonthRange(now)

	userIDs, err := s.credits.DistinctUserIDsBySource(ctx, "vip_gift", monthStart, monthEnd)
	if err != nil {
		return err
	}

	cleared := 0
	for _, userID := range userIDs {
		sums, err := s.credits.SumCreditLogBySource(ctx, userID, &monthStart, &monthEnd)
		if err != nil {
			logger.Warn("VIP赠送清零查询流水失败", zap.Int64("userID", userID), zap.Error(err))
			continue
		}
		giftTotal := sums["vip_gift"]
		balance, err := s.credits.GetUserBalance(ctx, userID)
		if err != nil || balance == nil {
			logger.Warn("VIP赠送清零查询余额失败", zap.Int64("userID", userID), zap.Error(err))
			continue
		}
		expireAmount := giftTotal
		if int64(balance.Balance) < expireAmount {
			expireAmount = int64(balance.Balance)
		}
		if expireAmount <= 0 {
			continue
		}
		if err := s.billing.ExpireVipGiftCredits(ctx, userID, expireAmount); err != nil {
			logger.Warn("VIP赠送清零失败", zap.Int64("userID", userID), zap.Error(err))
			continue
		}
		cleared++
	}

	logger.Info("VIP赠送积分月末清零完成", zap.Int("users", len(userIDs)), zap.Int("cleared", cleared))
	return nil
}
