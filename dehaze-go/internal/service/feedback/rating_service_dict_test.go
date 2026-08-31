package feedback

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	servicemocks "github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// ratingDictOptions 构造 member_growth_rules 字典选项：Label 为键，Value 为整型值字符串。
func ratingDictOptions(pairs ...struct{ key, val string }) []vo.Option {
	out := make([]vo.Option, 0, len(pairs))
	for _, p := range pairs {
		out = append(out, vo.Option{Label: p.key, Value: p.val})
	}
	return out
}

// awardGrowthForRating 从字典读取 rating_growth_value / rating_growth_daily_limit，
// 此处验证字典值正确传导至 AwardGrowth 的 changeValue。
func TestAwardGrowthForRating_DictValueApplied(t *testing.T) {
	userID := int64(1001)
	ratingID := int64(2001)
	counterKey := fmt.Sprintf("rating:daily:%d:%s", userID, time.Now().Format(dateFormat))

	memberMock := servicemocks.NewMockIMemberService(t)
	memberMock.EXPECT().AwardGrowth(mock.Anything, userID, "rating", 8, "评价奖励", fmt.Sprintf("%d", ratingID)).
		Return(nil)

	cache := servicemocks.NewMockICache(t)
	cache.EXPECT().Get(mock.Anything, counterKey).Return("", nil).Once()
	cache.EXPECT().Incr(mock.Anything, counterKey).Return(int64(1), nil).Once()
	cache.EXPECT().Expire(mock.Anything, counterKey, ratingDailyCounterTTL).Return(true, nil).Once()

	dictMock := servicemocks.NewMockIDictService(t)
	dictMock.EXPECT().GetByTypeCode(mock.Anything, memberGrowthRulesDictType).
		Return(ratingDictOptions(
			struct{ key, val string }{"rating_growth_value", "8"},
			struct{ key, val string }{"rating_growth_daily_limit", "5"},
		), nil)

	svc := &RatingService{memberSvc: memberMock, cache: cache, dictSvc: dictMock}
	assert.NoError(t, svc.awardGrowthForRating(context.Background(), userID, ratingID))
}

// 每日上限来自字典：达到上限后不再发放成长值。
func TestAwardGrowthForRating_DailyLimitFromDict_Stops(t *testing.T) {
	userID := int64(1002)
	ratingID := int64(2002)
	counterKey := fmt.Sprintf("rating:daily:%d:%s", userID, time.Now().Format(dateFormat))

	cache := servicemocks.NewMockICache(t)
	// 字典配置上限 2，今日已累计 2，应直接停止发放，不再调用 AwardGrowth。
	cache.EXPECT().Get(mock.Anything, counterKey).Return("2", nil).Once()

	dictMock := servicemocks.NewMockIDictService(t)
	dictMock.EXPECT().GetByTypeCode(mock.Anything, memberGrowthRulesDictType).
		Return(ratingDictOptions(
			struct{ key, val string }{"rating_growth_value", "5"},
			struct{ key, val string }{"rating_growth_daily_limit", "2"},
		), nil)

	// memberSvc 不设置 AwardGrowth 期望：若被调用会因无匹配而失败。
	memberMock := servicemocks.NewMockIMemberService(t)
	svc := &RatingService{memberSvc: memberMock, cache: cache, dictSvc: dictMock}
	assert.NoError(t, svc.awardGrowthForRating(context.Background(), userID, ratingID))
}
