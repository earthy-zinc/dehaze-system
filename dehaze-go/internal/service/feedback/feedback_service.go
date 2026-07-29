package feedback

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
	fbrepo "github.com/earthyzinc/dehaze-go/internal/repository/feedback"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

const (
	feedbackDailyLimit       = 5
	feedbackTitleMinLen      = 5
	feedbackTitleMaxLen      = 50
	feedbackContentMinLen    = 10
	feedbackContentMaxLen    = 1000
	feedbackMaxImages        = 5
	topKeywordsLimit         = 10
	feedbackStatsCacheTTL    = 10 * time.Minute
	feedbackDailyCounterTTL  = 25 * time.Hour
)

const (
	cacheKeyFeedbackStats         = "feedback:stats"
	cacheKeyFeedbackStatsVersion = "feedback:stats:version"
)

var feedbackTypes = []string{"suggestion", "bug", "experience", "complaint"}
var feedbackStatuses = []string{"pending", "processing", "replied", "closed"}

type FeedbackService struct {
	db                *gorm.DB
	feedbackRepo     fbrepo.IFeedbackRepository
	feedbackReplyRepo fbrepo.IFeedbackReplyRepository
	cache            types.ICache
}

func NewFeedbackService(
	db *gorm.DB,
	feedbackRepo fbrepo.IFeedbackRepository,
	feedbackReplyRepo fbrepo.IFeedbackReplyRepository,
	cache types.ICache,
) *FeedbackService {
	return &FeedbackService{
		db:                db,
		feedbackRepo:      feedbackRepo,
		feedbackReplyRepo: feedbackReplyRepo,
		cache:             cache,
	}
}

func (s *FeedbackService) CreateFeedback(ctx context.Context, userID int64, form *bo.FeedbackCreateForm) (int64, error) {
	if form.FeedbackType == "" {
		return 0, common.NewBizError(common.PARAM_ERROR, "反馈类型不能为空")
	}
	if len(form.Title) < feedbackTitleMinLen || len(form.Title) > feedbackTitleMaxLen {
		return 0, common.NewBizError(common.PARAM_ERROR, "标题长度必须在5-50字符之间")
	}
	if len(form.Content) < feedbackContentMinLen || len(form.Content) > feedbackContentMaxLen {
		return 0, common.NewBizError(common.PARAM_ERROR, "内容长度必须在10-1000字符之间")
	}
	if err := validateImageUrls(form.Images, feedbackMaxImages); err != nil {
		return 0, err
	}

	if s.cache != nil {
		today := time.Now().Format(dateFormat)
		counterKey := fmt.Sprintf("feedback:daily:%d:%s", userID, today)
		countStr, err := s.cache.Get(ctx, counterKey)
		if err == nil && countStr != "" {
			todayCount, parseErr := strconv.ParseInt(countStr, 10, 64)
			if parseErr == nil && todayCount >= int64(feedbackDailyLimit) {
				return 0, common.NewBizError(common.FEEDBACK_LIMIT_EXCEEDED, "今日反馈次数已达上限")
			}
		}
	} else {
		todayCount, err := s.feedbackRepo.CountTodayByUserID(ctx, userID)
		if err != nil {
			return 0, common.WrapBizError(common.DATABASE_ERROR, "查询今日反馈次数失败", err)
		}
		if todayCount >= int64(feedbackDailyLimit) {
			return 0, common.NewBizError(common.FEEDBACK_LIMIT_EXCEEDED, "今日反馈次数已达上限")
		}
	}

	fb := &model.SysFeedback{
		UserID:        userID,
		FeedbackType:  form.FeedbackType,
		Title:         form.Title,
		Content:       form.Content,
		Contact:       form.Contact,
		Images:        toJSONString(form.Images),
		RelatedModule: form.RelatedModule,
		Status:        1,
		Priority:      1,
		Tags:          "[]",
	}

	if err := s.feedbackRepo.Create(ctx, fb); err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "创建反馈失败", err)
	}

	if s.cache != nil {
		today := time.Now().Format(dateFormat)
		counterKey := fmt.Sprintf("feedback:daily:%d:%s", userID, today)
		newCount, err := s.cache.Incr(ctx, counterKey)
		if err == nil && newCount == 1 {
			_, _ = s.cache.Expire(ctx, counterKey, feedbackDailyCounterTTL)
		}
	}

	s.invalidateFeedbackStatsCache(ctx)
	return fb.ID, nil
}

func (s *FeedbackService) ListMyFeedback(ctx context.Context, userID int64, pageNum, pageSize int) (*vo.PageResult[vo.FeedbackPageVO], error) {
	list, total, err := s.feedbackRepo.FindPageMy(ctx, userID, pageNum, pageSize)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询反馈列表失败", err)
	}
	vos := make([]vo.FeedbackPageVO, 0, len(list))
	for _, f := range list {
		vos = append(vos, toFeedbackPageVO(&f.SysFeedback, f.Username, f.AssigneeName))
	}
	return &vo.PageResult[vo.FeedbackPageVO]{List: vos, Total: total}, nil
}

func (s *FeedbackService) GetFeedbackDetail(ctx context.Context, id, userID int64, isAdmin bool) (*vo.FeedbackDetailVO, error) {
	fb, err := s.feedbackRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询反馈失败", err)
	}
	if fb == nil {
		return nil, common.NewBizError(common.FEEDBACK_NOT_FOUND, "反馈不存在")
	}
	if !isAdmin && fb.UserID != userID {
		return nil, common.NewBizError(common.FEEDBACK_NOT_FOUND, "反馈不存在")
	}

	replies, err := s.feedbackReplyRepo.FindByFeedbackID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询回复失败", err)
	}

	replierNames := s.findReplierNames(ctx, replies)

	replyVOs := make([]vo.FeedbackReplyVO, 0, len(replies))
	for i, r := range replies {
		rv := vo.FeedbackReplyVO{
			ID:          r.ID,
			FeedbackID:  r.FeedbackID,
			ReplierID:   r.ReplierID,
			ReplierName: replierNames[i],
			ReplierType: int(r.ReplierType),
			Content:     r.Content,
			ReplyType:   r.ReplyType,
			Attachments: fromJSONString(r.Attachments),
			CreateTime:  r.CreateTime.Format(timeFormat),
		}
		replyVOs = append(replyVOs, rv)
	}

	username, assigneeName := s.findFeedbackUserinfo(ctx, fb)
	pageVO := toFeedbackPageVO(fb, username, assigneeName)
	detail := &vo.FeedbackDetailVO{
		FeedbackPageVO: pageVO,
		Images:         fromJSONString(fb.Images),
		Replies:        replyVOs,
	}
	if isAdmin {
		detail.Contact = fb.Contact
	}
	if fb.AssignedTime != nil {
		detail.AssignedTime = fb.AssignedTime.Format(timeFormat)
	}
	detail.CloseReason = fb.CloseReason
	return detail, nil
}

func (s *FeedbackService) SupplementFeedback(ctx context.Context, userID, feedbackID int64, form *bo.FeedbackSupplementForm) error {
	if form.Content == "" {
		return common.NewBizError(common.PARAM_ERROR, "补充内容不能为空")
	}
	fb, err := s.feedbackRepo.FindByID(ctx, feedbackID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询反馈失败", err)
	}
	if fb == nil {
		return common.NewBizError(common.FEEDBACK_NOT_FOUND, "反馈不存在")
	}
	if fb.UserID != userID {
		return common.NewBizError(common.FEEDBACK_NOT_FOUND, "反馈不存在")
	}
	if fb.Status == 4 {
		return common.NewBizError(common.FEEDBACK_CLOSED, "反馈已关闭")
	}

	if err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txFeedbackRepo := fbrepo.NewFeedbackRepository(tx)
		txReplyRepo := fbrepo.NewFeedbackReplyRepository(tx)

		reply := &model.SysFeedbackReply{
			FeedbackID:  feedbackID,
			ReplierID:   userID,
			ReplierType: 1,
			Content:     form.Content,
			ReplyType:   "info",
			Attachments: toJSONString(form.Attachments),
		}
		if err := txReplyRepo.Create(ctx, reply); err != nil {
			return err
		}

		if fb.Status == 3 {
			return txFeedbackRepo.Update(ctx, feedbackID, map[string]interface{}{
				"status": 2,
			})
		}
		return nil
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "补充反馈失败", err)
	}
	s.invalidateFeedbackStatsCache(ctx)
	return nil
}

func (s *FeedbackService) ListPagedFeedback(ctx context.Context, q *query.FeedbackPageQuery) (*vo.PageResult[vo.FeedbackPageVO], error) {
	list, total, err := s.feedbackRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询反馈列表失败", err)
	}
	vos := make([]vo.FeedbackPageVO, 0, len(list))
	for _, f := range list {
		vos = append(vos, toFeedbackPageVO(&f.SysFeedback, f.Username, f.AssigneeName))
	}
	return &vo.PageResult[vo.FeedbackPageVO]{List: vos, Total: total}, nil
}

func (s *FeedbackService) AssignFeedback(ctx context.Context, id, assigneeID int64) error {
	fb, err := s.feedbackRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询反馈失败", err)
	}
	if fb == nil {
		return common.NewBizError(common.FEEDBACK_NOT_FOUND, "反馈不存在")
	}
	if fb.Status == 4 {
		return common.NewBizError(common.FEEDBACK_CLOSED, "反馈已关闭")
	}
	now := time.Now()
	updates := map[string]interface{}{
		"assignee_id":   assigneeID,
		"assigned_time": now,
	}
	if fb.Status == 1 {
		updates["status"] = int8(2)
	}
	if err := s.feedbackRepo.Update(ctx, id, updates); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "分配反馈失败", err)
	}
	s.invalidateFeedbackStatsCache(ctx)
	return nil
}

func (s *FeedbackService) ReplyFeedback(ctx context.Context, adminID, feedbackID int64, form *bo.FeedbackReplyForm) error {
	if form.Content == "" {
		return common.NewBizError(common.PARAM_ERROR, "回复内容不能为空")
	}
	fb, err := s.feedbackRepo.FindByID(ctx, feedbackID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询反馈失败", err)
	}
	if fb == nil {
		return common.NewBizError(common.FEEDBACK_NOT_FOUND, "反馈不存在")
	}
	if fb.Status == 4 {
		return common.NewBizError(common.FEEDBACK_CLOSED, "反馈已关闭")
	}

	if err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txFbRepo := fbrepo.NewFeedbackRepository(tx)
		txReplyRepo := fbrepo.NewFeedbackReplyRepository(tx)

		reply := &model.SysFeedbackReply{
			FeedbackID:  feedbackID,
			ReplierID:   adminID,
			ReplierType: 2,
			Content:     form.Content,
			ReplyType:   form.ReplyType,
			Attachments: toJSONString(form.Attachments),
		}
		if err := txReplyRepo.Create(ctx, reply); err != nil {
			return err
		}
		return txFbRepo.Update(ctx, feedbackID, map[string]interface{}{
			"status": int8(3),
		})
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "回复反馈失败", err)
	}
	s.invalidateFeedbackStatsCache(ctx)
	return nil
}

func (s *FeedbackService) CloseFeedback(ctx context.Context, id int64, reason string) error {
	fb, err := s.feedbackRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询反馈失败", err)
	}
	if fb == nil {
		return common.NewBizError(common.FEEDBACK_NOT_FOUND, "反馈不存在")
	}
	if fb.Status == 4 {
		return common.NewBizError(common.FEEDBACK_CLOSED, "反馈已关闭")
	}
	if err := s.feedbackRepo.Update(ctx, id, map[string]interface{}{
		"status":       int8(4),
		"close_reason": reason,
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "关闭反馈失败", err)
	}
	s.invalidateFeedbackStatsCache(ctx)
	return nil
}

func (s *FeedbackService) UpdateFeedbackTags(ctx context.Context, id int64, tags []string) error {
	fb, err := s.feedbackRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询反馈失败", err)
	}
	if fb == nil {
		return common.NewBizError(common.FEEDBACK_NOT_FOUND, "反馈不存在")
	}
	if err := s.feedbackRepo.Update(ctx, id, map[string]interface{}{
		"tags": toJSONString(tags),
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新标签失败", err)
	}
	s.invalidateFeedbackStatsCache(ctx)
	return nil
}

func (s *FeedbackService) GetFeedbackStats(ctx context.Context, startTime, endTime string) (*vo.FeedbackStatsVO, error) {
	cacheKey := fmt.Sprintf("%s:v%s:%s:%s", cacheKeyFeedbackStats, s.getFeedbackStatsVersion(ctx), startTime, endTime)
	if s.cache != nil {
		if cached, err := s.cache.Get(ctx, cacheKey); err == nil && cached != "" {
			var stats vo.FeedbackStatsVO
			if err := json.Unmarshal([]byte(cached), &stats); err == nil {
				return &stats, nil
			}
		}
	}

	total, err := s.feedbackRepo.GetTotalCount(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询反馈总数失败", err)
	}

	typeDist := make(map[string]int64)
	for _, t := range feedbackTypes {
		typeDist[t] = 0
	}
	typeRows, err := s.feedbackRepo.GetTypeDistribution(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询类型分布失败", err)
	}
	for _, row := range typeRows {
		typeDist[row.Type] = row.Count
	}

	statusDist := make(map[string]int64)
	for _, st := range feedbackStatuses {
		statusDist[st] = 0
	}
	statusRows, err := s.feedbackRepo.GetStatusDistribution(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询状态分布失败", err)
	}
	for _, row := range statusRows {
		statusDist[fbrepo.FeedbackStatusToString(row.Status)] = row.Count
	}

	moduleRows, err := s.feedbackRepo.GetModuleDistribution(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询模块分布失败", err)
	}
	moduleDist := make([]vo.ModuleCount, 0, len(moduleRows))
	for _, row := range moduleRows {
		moduleDist = append(moduleDist, vo.ModuleCount{Module: row.Module, Count: row.Count})
	}

	avgResponse, err := s.feedbackRepo.GetAvgResponseTime(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询平均响应时间失败", err)
	}

	avgClose, err := s.feedbackRepo.GetAvgCloseTime(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询平均关闭时间失败", err)
	}

	keywordRows, err := s.feedbackRepo.GetTopKeywords(ctx, startTime, endTime, topKeywordsLimit)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询热门关键词失败", err)
	}
	keywords := make([]vo.KeywordCount, 0, len(keywordRows))
	for _, row := range keywordRows {
		keywords = append(keywords, vo.KeywordCount{Keyword: row.Keyword, Count: row.Count})
	}

	result := &vo.FeedbackStatsVO{
		TotalFeedback:       total,
		TypeDistribution:    typeDist,
		ModuleDistribution:  moduleDist,
		StatusDistribution:  statusDist,
		AverageResponseTime: avgResponse,
		AverageCloseTime:    avgClose,
		TopKeywords:         keywords,
	}

	if s.cache != nil {
		if data, err := json.Marshal(result); err == nil {
			_ = s.cache.Set(ctx, cacheKey, string(data), feedbackStatsCacheTTL)
		}
	}

	return result, nil
}

func (s *FeedbackService) getFeedbackStatsVersion(ctx context.Context) string {
	if s.cache == nil {
		return "0"
	}
	v, err := s.cache.Get(ctx, cacheKeyFeedbackStatsVersion)
	if err != nil || v == "" {
		return "0"
	}
	return v
}

func (s *FeedbackService) invalidateFeedbackStatsCache(ctx context.Context) {
	if s.cache == nil {
		return
	}
	_, _ = s.cache.Incr(ctx, cacheKeyFeedbackStatsVersion)
}

func (s *FeedbackService) findReplierNames(ctx context.Context, replies []model.SysFeedbackReply) []string {
	names := make([]string, len(replies))
	if len(replies) == 0 {
		return names
	}
	idSet := make(map[int64]bool)
	for _, r := range replies {
		idSet[r.ReplierID] = true
	}
	ids := make([]int64, 0, len(idSet))
	for id := range idSet {
		ids = append(ids, id)
	}
	nameMap := make(map[int64]string)
	type userRow struct {
		ID       int64  `gorm:"column:id"`
		Username string `gorm:"column:username"`
	}
	var rows []userRow
	s.db.WithContext(ctx).
		Table("sys_user").
		Where("id IN ? AND deleted = 0", ids).
		Select("id, username").
		Scan(&rows)
	for _, row := range rows {
		nameMap[row.ID] = row.Username
	}
	for i, r := range replies {
		names[i] = nameMap[r.ReplierID]
	}
	return names
}

func (s *FeedbackService) findFeedbackUserinfo(ctx context.Context, fb *model.SysFeedback) (string, string) {
	type userRow struct {
		Username string `gorm:"column:username"`
	}
	type assigneeRow struct {
		Username string `gorm:"column:assignee_name"`
	}
	var uRow userRow
	s.db.WithContext(ctx).
		Table("sys_user").
		Where("id = ? AND deleted = 0", fb.UserID).
		Select("username").
		Scan(&uRow)

	assigneeName := ""
	if fb.AssigneeID != nil && *fb.AssigneeID > 0 {
		var aRow userRow
		s.db.WithContext(ctx).
			Table("sys_user").
			Where("id = ? AND deleted = 0", *fb.AssigneeID).
			Select("username").
			Scan(&aRow)
		assigneeName = aRow.Username
	}
	return uRow.Username, assigneeName
}

func toFeedbackPageVO(f *model.SysFeedback, username, assigneeName string) vo.FeedbackPageVO {
	v := vo.FeedbackPageVO{
		ID:            f.ID,
		UserID:        f.UserID,
		Username:      username,
		FeedbackType:  f.FeedbackType,
		Title:         f.Title,
		Content:       f.Content,
		Status:        fbrepo.FeedbackStatusToString(f.Status),
		Priority:      int(f.Priority),
		AssigneeID:    f.AssigneeID,
		AssigneeName:  assigneeName,
		RelatedModule: f.RelatedModule,
		Tags:          fromJSONString(f.Tags),
		CreateTime:    f.CreatedAt.Format(timeFormat),
	}
	if !f.UpdatedAt.IsZero() {
		v.UpdateTime = f.UpdatedAt.Format(timeFormat)
	}
	return v
}

var _ IFeedbackService = (*FeedbackService)(nil)
