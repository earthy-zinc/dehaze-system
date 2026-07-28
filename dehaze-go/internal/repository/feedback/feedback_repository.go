package feedback

import (
	"context"
	"errors"
	"sort"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type FeedbackRepository struct {
	db *gorm.DB
}

func NewFeedbackRepository(db *gorm.DB) *FeedbackRepository {
	return &FeedbackRepository{db: db}
}

func (r *FeedbackRepository) Create(ctx context.Context, f *model.SysFeedback) error {
	return r.db.WithContext(ctx).Create(f).Error
}

func (r *FeedbackRepository) FindByID(ctx context.Context, id int64) (*model.SysFeedback, error) {
	var f model.SysFeedback
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&f).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &f, err
}

func (r *FeedbackRepository) FindPageMy(ctx context.Context, userID int64, pageNum, pageSize int) ([]FeedbackWithUser, int64, error) {
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).
		Table("sys_feedback f").
		Select("f.*, u.username as username").
		Joins("LEFT JOIN sys_user u ON f.user_id = u.id").
		Where("f.user_id = ? AND f.deleted = 0", userID)

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []FeedbackWithUser
	err := db.Order("f.id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Scan(&list).Error
	return list, total, err
}

func (r *FeedbackRepository) FindPage(ctx context.Context, q *query.FeedbackPageQuery) ([]FeedbackWithUser, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).
		Table("sys_feedback f").
		Select("f.*, u.username as username, a.username as assignee_name").
		Joins("LEFT JOIN sys_user u ON f.user_id = u.id").
		Joins("LEFT JOIN sys_user a ON f.assignee_id = a.id").
		Where("f.deleted = 0")

	if q.Keywords != "" {
		kw := "%" + q.Keywords + "%"
		db = db.Where("f.title LIKE ? OR f.content LIKE ?", kw, kw)
	}
	if q.FeedbackType != "" {
		db = db.Where("f.feedback_type = ?", q.FeedbackType)
	}
	if q.Status != "" {
		db = db.Where("f.status = ?", FeedbackStatusToInt(q.Status))
	}
	if q.RelatedModule != "" {
		db = db.Where("f.related_module = ?", q.RelatedModule)
	}
	if q.Priority != nil {
		db = db.Where("f.priority = ?", *q.Priority)
	}
	if q.AssigneeID != nil {
		db = db.Where("f.assignee_id = ?", *q.AssigneeID)
	}
	if q.StartTime != "" {
		db = db.Where("f.create_time >= ?", q.StartTime)
	}
	if q.EndTime != "" {
		db = db.Where("f.create_time <= ?", q.EndTime)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []FeedbackWithUser
	err := db.Order("f.id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Scan(&list).Error
	return list, total, err
}

func (r *FeedbackRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysFeedback{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *FeedbackRepository) CountTodayByUserID(ctx context.Context, userID int64) (int64, error) {
	now := time.Now()
	todayStart := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysFeedback{}).
		Where("user_id = ? AND deleted = 0 AND create_time >= ?", userID, todayStart).
		Count(&count).Error
	return count, err
}

func (r *FeedbackRepository) GetTypeDistribution(ctx context.Context, startTime, endTime string) ([]TypeCountRow, error) {
	db := r.db.WithContext(ctx).
		Table("sys_feedback").
		Select("feedback_type as type, COUNT(*) as count").
		Where("deleted = 0")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	var rows []TypeCountRow
	err := db.Group("feedback_type").Scan(&rows).Error
	return rows, err
}

func (r *FeedbackRepository) GetStatusDistribution(ctx context.Context, startTime, endTime string) ([]StatusCountRow, error) {
	db := r.db.WithContext(ctx).
		Table("sys_feedback").
		Select("status, COUNT(*) as count").
		Where("deleted = 0")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	var rows []StatusCountRow
	err := db.Group("status").Scan(&rows).Error
	return rows, err
}

func (r *FeedbackRepository) GetModuleDistribution(ctx context.Context, startTime, endTime string) ([]ModuleCountRow, error) {
	db := r.db.WithContext(ctx).
		Table("sys_feedback").
		Select("related_module as module, COUNT(*) as count").
		Where("deleted = 0 AND related_module IS NOT NULL AND related_module <> ''")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	var rows []ModuleCountRow
	err := db.Group("related_module").Scan(&rows).Error
	return rows, err
}

func (r *FeedbackRepository) GetTotalCount(ctx context.Context, startTime, endTime string) (int64, error) {
	db := r.db.WithContext(ctx).
		Model(&model.SysFeedback{}).
		Where("deleted = 0")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	var count int64
	err := db.Count(&count).Error
	return count, err
}

type feedbackTimeRow struct {
	FeedbackID         int64      `gorm:"column:feedback_id"`
	FeedbackCreateTime time.Time  `gorm:"column:feedback_create_time"`
	ReplyCreateTime    *time.Time `gorm:"column:reply_create_time"`
}

func (r *FeedbackRepository) GetAvgResponseTime(ctx context.Context, startTime, endTime string) (float64, error) {
	db := r.db.WithContext(ctx).
		Table("sys_feedback f").
		Select("f.id as feedback_id, f.create_time as feedback_create_time, fr.create_time as reply_create_time").
		Joins("INNER JOIN sys_feedback_reply fr ON fr.feedback_id = f.id AND fr.replier_type = 2").
		Where("f.deleted = 0")
	if startTime != "" {
		db = db.Where("f.create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("f.create_time <= ?", endTime)
	}

	var rows []feedbackTimeRow
	if err := db.Scan(&rows).Error; err != nil {
		return 0, err
	}

	type fbTime struct {
		CreateTime time.Time
		FirstReply time.Time
	}
	fbMap := make(map[int64]*fbTime)
	for _, row := range rows {
		if row.ReplyCreateTime == nil {
			continue
		}
		entry, ok := fbMap[row.FeedbackID]
		if !ok {
			fbMap[row.FeedbackID] = &fbTime{CreateTime: row.FeedbackCreateTime, FirstReply: *row.ReplyCreateTime}
			continue
		}
		if row.ReplyCreateTime.Before(entry.FirstReply) {
			entry.FirstReply = *row.ReplyCreateTime
		}
	}

	if len(fbMap) == 0 {
		return 0, nil
	}

	var totalDiff float64
	var count int64
	for _, entry := range fbMap {
		diff := entry.FirstReply.Sub(entry.CreateTime).Hours()
		if diff > 0 {
			totalDiff += diff
			count++
		}
	}

	if count == 0 {
		return 0, nil
	}
	return totalDiff / float64(count), nil
}

type closeTimeRow struct {
	CreateTime time.Time `gorm:"column:create_time"`
	UpdateTime time.Time `gorm:"column:update_time"`
}

func (r *FeedbackRepository) GetAvgCloseTime(ctx context.Context, startTime, endTime string) (float64, error) {
	db := r.db.WithContext(ctx).
		Table("sys_feedback").
		Select("create_time, update_time").
		Where("deleted = 0 AND status = 4")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}

	var rows []closeTimeRow
	if err := db.Scan(&rows).Error; err != nil {
		return 0, err
	}

	if len(rows) == 0 {
		return 0, nil
	}

	var totalDiff float64
	for _, row := range rows {
		diff := row.UpdateTime.Sub(row.CreateTime).Hours()
		if diff > 0 {
			totalDiff += diff
		}
	}
	return totalDiff / float64(len(rows)), nil
}

func (r *FeedbackRepository) GetTopKeywords(ctx context.Context, startTime, endTime string, limit int) ([]KeywordCountRow, error) {
	db := r.db.WithContext(ctx).
		Table("sys_feedback").
		Select("title, content").
		Where("deleted = 0")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}

	type textRow struct {
		Title   string `gorm:"column:title"`
		Content string `gorm:"column:content"`
	}
	var rows []textRow
	if err := db.Scan(&rows).Error; err != nil {
		return nil, err
	}

	counts := make(map[string]int64)
	for _, row := range rows {
		text := row.Title + " " + row.Content
		words := splitKeywords(text)
		for _, w := range words {
			if len(w) >= 2 {
				counts[w]++
			}
		}
	}

	type kv struct {
		k string
		v int64
	}
	var sorted []kv
	for k, v := range counts {
		sorted = append(sorted, kv{k, v})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].v > sorted[j].v
	})

	if limit > 0 && len(sorted) > limit {
		sorted = sorted[:limit]
	}

	result := make([]KeywordCountRow, 0, len(sorted))
	for _, s := range sorted {
		result = append(result, KeywordCountRow{Keyword: s.k, Count: s.v})
	}
	return result, nil
}

func splitKeywords(text string) []string {
	separators := " \t\n\r,，。.!！?？;；:：、\"'()（）[]【】{}/\\|"
	fields := strings.FieldsFunc(text, func(r rune) bool {
		return strings.ContainsRune(separators, r)
	})
	return fields
}

func FeedbackStatusToInt(status string) int8 {
	switch status {
	case "pending":
		return 1
	case "processing":
		return 2
	case "replied":
		return 3
	case "closed":
		return 4
	}
	return 0
}

func FeedbackStatusToString(status int8) string {
	switch status {
	case 1:
		return "pending"
	case 2:
		return "processing"
	case 3:
		return "replied"
	case 4:
		return "closed"
	}
	return ""
}

var _ IFeedbackRepository = (*FeedbackRepository)(nil)
