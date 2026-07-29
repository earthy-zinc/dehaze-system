package feedback

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	fbrepo "github.com/earthyzinc/dehaze-go/internal/repository/feedback"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

const (
	ratingTimeLimitDays = 30
	ratingMaxImages    = 3
	ratingMaxCommentLen = 500
	timeFormat          = "2006-01-02 15:04:05"

	ratingGrowthValue      = 5
	ratingDailyGrowthLimit  = 5
	ratingStatsCacheTTL     = 10 * time.Minute
	ratingDailyCounterTTL   = 25 * time.Hour
	dateFormat              = "2006-01-02"
)

const (
	cacheKeyRatingStatsGlobal         = "rating:stats:global"
	cacheKeyRatingStatsGlobalVersion = "rating:stats:global:version"
)

var positiveTagSet = map[string]bool{
	"去雾彻底": true, "色彩自然": true, "细节清晰": true, "处理速度快": true, "整体提升明显": true,
}

var negativeTagSet = map[string]bool{
	"残留雾气": true, "色彩失真": true, "细节丢失": true, "处理速度慢": true, "无明显改善": true,
}

type RatingService struct {
	db           *gorm.DB
	ratingRepo   fbrepo.IRatingRepository
	predLogRepo  predrepo.IPredLogRepository
	memberSvc    memberservice.IMemberService
	cache        types.ICache
	alertSvc     ILowRatingAlertService
	logger       *zap.Logger
}

func NewRatingService(
	db *gorm.DB,
	ratingRepo fbrepo.IRatingRepository,
	predLogRepo predrepo.IPredLogRepository,
	memberSvc memberservice.IMemberService,
	cache types.ICache,
	alertSvc ILowRatingAlertService,
	logger *zap.Logger,
) *RatingService {
	return &RatingService{
		db:          db,
		ratingRepo:  ratingRepo,
		predLogRepo: predLogRepo,
		memberSvc:   memberSvc,
		cache:       cache,
		alertSvc:    alertSvc,
		logger:      logger,
	}
}

func (s *RatingService) CreateRating(ctx context.Context, userID int64, form *bo.RatingCreateForm) (int64, error) {
	if form.PredLogID <= 0 {
		return 0, common.NewBizError(common.PARAM_ERROR, "处理记录ID不能为空")
	}
	if form.Rating < 1 || form.Rating > 5 {
		return 0, common.NewBizError(common.PARAM_ERROR, "评分必须在1-5之间")
	}
	if len(form.Comment) > ratingMaxCommentLen {
		return 0, common.NewBizError(common.PARAM_ERROR, "评价文字不能超过500字符")
	}
	if err := validateImageUrls(form.ImageUrls, ratingMaxImages); err != nil {
		return 0, err
	}

	predLog, err := s.predLogRepo.FindByID(ctx, form.PredLogID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return 0, common.NewBizError(common.PREDICTION_LOG_NOT_FOUND, "处理记录不存在")
		}
		return 0, common.WrapBizError(common.DATABASE_ERROR, "查询处理记录失败", err)
	}
	if predLog.Status != model.LogStatusCompleted {
		return 0, common.NewBizError(common.OPERATION_NOT_ALLOW, "处理记录未完成")
	}
	if predLog.CreateBy != userID {
		return 0, common.NewBizError(common.OPERATION_NOT_ALLOW, "无权评价他人的处理记录")
	}
	if time.Since(predLog.UpdatedAt) > ratingTimeLimitDays*24*time.Hour {
		return 0, common.NewBizError(common.RATING_EXPIRED, "已超过评价时限")
	}

	existing, err := s.ratingRepo.FindByPredLogID(ctx, form.PredLogID)
	if err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "查询评价失败", err)
	}
	if existing != nil {
		return 0, common.NewBizError(common.RATING_ALREADY_EXISTS, "该处理记录已评价")
	}

	rating := &model.SysRating{
		UserID:      userID,
		PredLogID:   form.PredLogID,
		AlgorithmID: predLog.AlgorithmID,
		Rating:      int8(form.Rating),
		Comment:     form.Comment,
		Tags:        toJSONString(form.Tags),
		ImageUrls:   toJSONString(form.ImageUrls),
		IsAnonymous: int8(form.IsAnonymous),
	}

	if err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txRatingRepo := fbrepo.NewRatingRepository(tx)
		if err := txRatingRepo.Create(ctx, rating); err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "创建评价失败", err)
		}
		if err := s.awardGrowthForRating(ctx, userID, rating.ID); err != nil {
			return err
		}
		return nil
	}); err != nil {
		return 0, err
	}

	s.invalidateRatingStatsCache(ctx)

	if rating.Rating <= 2 && s.alertSvc != nil {
		if err := s.alertSvc.PublishRatingEvent(ctx, rating); err != nil {
			s.logger.Warn("低分告警事件发布失败",
				zap.Int64("ratingId", rating.ID),
				zap.Error(err))
		}
	}

	return rating.ID, nil
}

func (s *RatingService) UpdateRating(ctx context.Context, userID, ratingID int64, form *bo.RatingCreateForm) error {
	if form.Rating < 1 || form.Rating > 5 {
		return common.NewBizError(common.PARAM_ERROR, "评分必须在1-5之间")
	}
	if len(form.Comment) > ratingMaxCommentLen {
		return common.NewBizError(common.PARAM_ERROR, "评价文字不能超过500字符")
	}
	if err := validateImageUrls(form.ImageUrls, ratingMaxImages); err != nil {
		return err
	}

	rating, err := s.ratingRepo.FindByID(ctx, ratingID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询评价失败", err)
	}
	if rating == nil {
		return common.NewBizError(common.RATING_NOT_FOUND, "评价不存在")
	}
	if rating.UserID != userID {
		return common.NewBizError(common.RATING_NOT_FOUND, "评价不存在")
	}

	if err := s.ratingRepo.Update(ctx, ratingID, map[string]interface{}{
		"rating":       int8(form.Rating),
		"comment":      form.Comment,
		"tags":         toJSONString(form.Tags),
		"image_urls":   toJSONString(form.ImageUrls),
		"is_anonymous": int8(form.IsAnonymous),
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新评价失败", err)
	}

	s.invalidateRatingStatsCache(ctx)
	return nil
}

func (s *RatingService) ListMyRatings(ctx context.Context, userID int64, pageNum, pageSize int) (*vo.PageResult[vo.MyRatingVO], error) {
	list, total, err := s.ratingRepo.FindPageMy(ctx, userID, pageNum, pageSize)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评价列表失败", err)
	}
	vos := make([]vo.MyRatingVO, 0, len(list))
	for _, r := range list {
		vos = append(vos, toMyRatingVO(&r.SysRating, r.AlgorithmName))
	}
	return &vo.PageResult[vo.MyRatingVO]{List: vos, Total: total}, nil
}

func (s *RatingService) GetRatingByPrediction(ctx context.Context, userID, predLogID int64) (*vo.RatingDetailVO, error) {
	predLog, err := s.predLogRepo.FindByID(ctx, predLogID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.PREDICTION_LOG_NOT_FOUND, "处理记录不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询处理记录失败", err)
	}
	if predLog.CreateBy != userID {
		return nil, common.NewBizError(common.OPERATION_NOT_ALLOW, "无权查询他人处理记录的评价")
	}

	rating, err := s.ratingRepo.FindByPredLogID(ctx, predLogID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评价失败", err)
	}
	if rating == nil {
		return nil, nil
	}

	return s.buildRatingDetailVO(ctx, rating)
}

func (s *RatingService) ListPagedRatings(ctx context.Context, q *query.RatingPageQuery) (*vo.PageResult[vo.RatingPageVO], error) {
	list, total, err := s.ratingRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评价列表失败", err)
	}
	vos := make([]vo.RatingPageVO, 0, len(list))
	for _, r := range list {
		myVO := toMyRatingVO(&r.SysRating, r.AlgorithmName)
		v := vo.RatingPageVO{
			MyRatingVO: myVO,
			UserID:     r.UserID,
			IsHidden:   int(r.IsHidden),
		}
		if r.IsAnonymous == 0 {
			v.Username = r.Username
			v.UserAvatar = r.UserAvatar
		}
		vos = append(vos, v)
	}
	return &vo.PageResult[vo.RatingPageVO]{List: vos, Total: total}, nil
}

func (s *RatingService) HideRating(ctx context.Context, id int64) error {
	rating, err := s.ratingRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询评价失败", err)
	}
	if rating == nil {
		return common.NewBizError(common.RATING_NOT_FOUND, "评价不存在")
	}
	if err := s.ratingRepo.Update(ctx, id, map[string]interface{}{
		"is_hidden": 1,
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "隐藏评价失败", err)
	}
	return nil
}

func (s *RatingService) ReplyRating(ctx context.Context, id int64, content string) error {
	if content == "" {
		return common.NewBizError(common.PARAM_ERROR, "回复内容不能为空")
	}
	rating, err := s.ratingRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询评价失败", err)
	}
	if rating == nil {
		return common.NewBizError(common.RATING_NOT_FOUND, "评价不存在")
	}
	now := time.Now()
	if err := s.ratingRepo.Update(ctx, id, map[string]interface{}{
		"admin_reply": content,
		"reply_time":  now,
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "回复评价失败", err)
	}
	return nil
}

func (s *RatingService) GetRatingStats(ctx context.Context, startTime, endTime string) (*vo.RatingStatsVO, error) {
	cacheKey := fmt.Sprintf("%s:v%s:%s:%s", cacheKeyRatingStatsGlobal, s.getRatingStatsVersion(ctx), startTime, endTime)
	if s.cache != nil {
		if cached, err := s.cache.Get(ctx, cacheKey); err == nil && cached != "" {
			var stats vo.RatingStatsVO
			if err := json.Unmarshal([]byte(cached), &stats); err == nil {
				return &stats, nil
			}
		}
	}

	total, avgRating, distribution, err := s.ratingRepo.GetStats(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询评价统计失败", err)
	}

	tagRows, err := s.ratingRepo.GetTagRanking(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询标签排名失败", err)
	}
	var positiveRanking, negativeRanking []vo.TagCount
	for _, row := range tagRows {
		if positiveTagSet[row.Tag] {
			positiveRanking = append(positiveRanking, vo.TagCount{Tag: row.Tag, Count: row.Count})
		} else if negativeTagSet[row.Tag] {
			negativeRanking = append(negativeRanking, vo.TagCount{Tag: row.Tag, Count: row.Count})
		}
	}
	sort.Slice(positiveRanking, func(i, j int) bool { return positiveRanking[i].Count > positiveRanking[j].Count })
	sort.Slice(negativeRanking, func(i, j int) bool { return negativeRanking[i].Count > negativeRanking[j].Count })

	algoRows, err := s.ratingRepo.GetAlgorithmStats(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法统计失败", err)
	}
	algoStats := make([]vo.AlgorithmRatingStat, 0, len(algoRows))
	for _, row := range algoRows {
		lowRate := float64(0)
		if row.TotalRatings > 0 {
			lowRate = float64(row.LowRatingCount) / float64(row.TotalRatings) * 100
		}
		algoStats = append(algoStats, vo.AlgorithmRatingStat{
			AlgorithmID:   row.AlgorithmID,
			AlgorithmName: row.AlgorithmName,
			AverageRating: row.AverageRating,
			TotalRatings:  row.TotalRatings,
			LowRatingRate: lowRate,
		})
	}

	result := &vo.RatingStatsVO{
		TotalRatings:       total,
		AverageRating:     avgRating,
		RatingDistribution: distribution,
		PositiveTagRanking: positiveRanking,
		NegativeTagRanking: negativeRanking,
		AlgorithmStats:     algoStats,
	}

	if s.cache != nil {
		if data, err := json.Marshal(result); err == nil {
			_ = s.cache.Set(ctx, cacheKey, string(data), ratingStatsCacheTTL)
		}
	}

	return result, nil
}

func (s *RatingService) awardGrowthForRating(ctx context.Context, userID, ratingID int64) error {
	if s.memberSvc == nil {
		return nil
	}
	today := time.Now().Format(dateFormat)
	counterKey := fmt.Sprintf("rating:daily:%d:%s", userID, today)

	if s.cache != nil {
		countStr, err := s.cache.Get(ctx, counterKey)
		if err == nil && countStr != "" {
			if count, parseErr := strconv.ParseInt(countStr, 10, 64); parseErr == nil && count >= int64(ratingDailyGrowthLimit) {
				return nil
			}
		}
	}

	if err := s.memberSvc.AwardGrowth(ctx, userID, "rating", ratingGrowthValue, "评价奖励", fmt.Sprintf("%d", ratingID)); err != nil {
		return err
	}

	if s.cache != nil {
		if count, err := s.cache.Incr(ctx, counterKey); err == nil {
			if count == 1 {
				_, _ = s.cache.Expire(ctx, counterKey, ratingDailyCounterTTL)
			}
		}
	}
	return nil
}

func (s *RatingService) getRatingStatsVersion(ctx context.Context) string {
	if s.cache == nil {
		return "0"
	}
	v, err := s.cache.Get(ctx, cacheKeyRatingStatsGlobalVersion)
	if err != nil || v == "" {
		return "0"
	}
	return v
}

func (s *RatingService) invalidateRatingStatsCache(ctx context.Context) {
	if s.cache == nil {
		return
	}
	_, _ = s.cache.Incr(ctx, cacheKeyRatingStatsGlobalVersion)
}

func (s *RatingService) buildRatingDetailVO(ctx context.Context, r *model.SysRating) (*vo.RatingDetailVO, error) {
	myVO := toMyRatingVO(r, s.findAlgorithmName(ctx, r.AlgorithmID))
	v := &vo.RatingDetailVO{
		RatingPageVO: vo.RatingPageVO{
			MyRatingVO: myVO,
			IsHidden:   int(r.IsHidden),
		},
		AlgorithmID: r.AlgorithmID,
	}
	if r.IsAnonymous == 0 {
		v.UserID = r.UserID
		username, avatar := s.findUserinfo(ctx, r.UserID)
		v.Username = username
		v.UserAvatar = avatar
	}
	return v, nil
}

func (s *RatingService) findAlgorithmName(ctx context.Context, algorithmID int64) string {
	var name string
	s.db.WithContext(ctx).
		Table("sys_algorithm").
		Where("id = ? AND deleted = 0", algorithmID).
		Select("name").
		Scan(&name)
	return name
}

func (s *RatingService) findUserinfo(ctx context.Context, userID int64) (string, string) {
	type userRow struct {
		Username string `gorm:"column:username"`
		Avatar   string `gorm:"column:avatar"`
	}
	var row userRow
	s.db.WithContext(ctx).
		Table("sys_user").
		Where("id = ? AND deleted = 0", userID).
		Select("username, avatar").
		Scan(&row)
	return row.Username, row.Avatar
}

func toMyRatingVO(r *model.SysRating, algorithmName string) vo.MyRatingVO {
	v := vo.MyRatingVO{
		ID:            r.ID,
		PredLogID:     r.PredLogID,
		AlgorithmName: algorithmName,
		Rating:        int(r.Rating),
		IsAnonymous:   int(r.IsAnonymous),
		Comment:       r.Comment,
		Tags:          fromJSONString(r.Tags),
		ImageUrls:     fromJSONString(r.ImageUrls),
		AdminReply:    r.AdminReply,
		CreateTime:    r.CreatedAt.Format(timeFormat),
	}
	if r.ReplyTime != nil {
		v.ReplyTime = r.ReplyTime.Format(timeFormat)
	}
	return v
}

func toJSONString(arr []string) string {
	if arr == nil {
		return "[]"
	}
	b, _ := json.Marshal(arr)
	return string(b)
}

func fromJSONString(s string) []string {
	if s == "" {
		return []string{}
	}
	var arr []string
	if err := json.Unmarshal([]byte(s), &arr); err != nil {
		return []string{}
	}
	if arr == nil {
		return []string{}
	}
	return arr
}

var _ IRatingService = (*RatingService)(nil)
