package feedback

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type RatingRepository struct {
	db *gorm.DB
}

func NewRatingRepository(db *gorm.DB) *RatingRepository {
	return &RatingRepository{db: db}
}

func (r *RatingRepository) Create(ctx context.Context, rating *model.SysRating) error {
	return r.db.WithContext(ctx).Create(rating).Error
}

func (r *RatingRepository) FindByID(ctx context.Context, id int64) (*model.SysRating, error) {
	var rating model.SysRating
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&rating).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &rating, err
}

func (r *RatingRepository) FindByPredLogID(ctx context.Context, predLogID int64) (*model.SysRating, error) {
	var rating model.SysRating
	err := r.db.WithContext(ctx).
		Where("pred_log_id = ? AND deleted = 0", predLogID).
		First(&rating).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &rating, err
}

func (r *RatingRepository) FindPageMy(ctx context.Context, userID int64, pageNum, pageSize int) ([]RatingWithAlgorithm, int64, error) {
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).
		Table("sys_rating r").
		Select("r.*, a.name as algorithm_name").
		Joins("LEFT JOIN sys_algorithm a ON r.algorithm_id = a.id").
		Where("r.user_id = ? AND r.deleted = 0", userID)

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []RatingWithAlgorithm
	err := db.Order("r.id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Scan(&list).Error
	return list, total, err
}

func (r *RatingRepository) FindPage(ctx context.Context, q *query.RatingPageQuery) ([]RatingWithUserAndAlgorithm, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).
		Table("sys_rating r").
		Select("r.*, u.username, u.avatar as user_avatar, a.name as algorithm_name").
		Joins("LEFT JOIN sys_user u ON r.user_id = u.id").
		Joins("LEFT JOIN sys_algorithm a ON r.algorithm_id = a.id").
		Where("r.deleted = 0")

	if q.Keywords != "" {
		kw := "%" + q.Keywords + "%"
		db = db.Where("u.username LIKE ?", kw)
	}
	if q.AlgorithmID != nil {
		db = db.Where("r.algorithm_id = ?", *q.AlgorithmID)
	}
	if q.RatingMin != nil {
		db = db.Where("r.rating >= ?", *q.RatingMin)
	}
	if q.RatingMax != nil {
		db = db.Where("r.rating <= ?", *q.RatingMax)
	}
	if q.HasComment != nil {
		if *q.HasComment {
			db = db.Where("r.comment <> ''")
		} else {
			db = db.Where("r.comment = ''")
		}
	}
	if q.StartTime != "" {
		db = db.Where("r.create_time >= ?", q.StartTime)
	}
	if q.EndTime != "" {
		db = db.Where("r.create_time <= ?", q.EndTime)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []RatingWithUserAndAlgorithm
	err := db.Order("r.id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Scan(&list).Error
	return list, total, err
}

func (r *RatingRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysRating{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *RatingRepository) GetStats(ctx context.Context, startTime, endTime string) (int64, float64, map[int]int64, error) {
	db := r.db.WithContext(ctx).
		Table("sys_rating").
		Where("deleted = 0")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}

	type row struct {
		Rating int   `gorm:"column:rating"`
		Count  int64 `gorm:"column:count"`
	}
	var rows []row
	if err := db.Select("rating, COUNT(*) as count").Group("rating").Scan(&rows).Error; err != nil {
		return 0, 0, nil, err
	}

	var total int64
	var sum int64
	distribution := make(map[int]int64)
	for _, r := range rows {
		total += r.Count
		sum += int64(r.Rating) * r.Count
		distribution[r.Rating] = r.Count
	}

	avg := float64(0)
	if total > 0 {
		avg = float64(sum) / float64(total)
	}
	return total, avg, distribution, nil
}

func (r *RatingRepository) GetAlgorithmStats(ctx context.Context, startTime, endTime string) ([]AlgorithmRatingStatRow, error) {
	db := r.db.WithContext(ctx).
		Table("sys_rating r").
		Select(`r.algorithm_id, a.name as algorithm_name,
			AVG(r.rating) as average_rating,
			COUNT(*) as total_ratings,
			SUM(CASE WHEN r.rating <= 2 THEN 1 ELSE 0 END) as low_rating_count`).
		Joins("LEFT JOIN sys_algorithm a ON r.algorithm_id = a.id").
		Where("r.deleted = 0")
	if startTime != "" {
		db = db.Where("r.create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("r.create_time <= ?", endTime)
	}

	var rows []AlgorithmRatingStatRow
	err := db.Group("r.algorithm_id, a.name").Scan(&rows).Error
	return rows, err
}

func (r *RatingRepository) GetTagRanking(ctx context.Context, startTime, endTime string) ([]TagCountRow, error) {
	db := r.db.WithContext(ctx).
		Table("sys_rating").
		Select("tags").
		Where("deleted = 0 AND tags IS NOT NULL AND tags <> '' AND tags <> '[]'")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}

	var tagStrings []string
	if err := db.Pluck("tags", &tagStrings).Error; err != nil {
		return nil, err
	}

	counts := make(map[string]int64)
	for _, ts := range tagStrings {
		var tags []string
		if err := json.Unmarshal([]byte(ts), &tags); err != nil {
			continue
		}
		for _, t := range tags {
			counts[t]++
		}
	}

	result := make([]TagCountRow, 0, len(counts))
	for tag, count := range counts {
		result = append(result, TagCountRow{Tag: tag, Count: count})
	}
	return result, nil
}

func (r *RatingRepository) GetRatingCountByValue(ctx context.Context, rating int, startTime, endTime string) (int64, error) {
	db := r.db.WithContext(ctx).
		Model(&model.SysRating{}).
		Where("rating = ? AND deleted = 0", rating)
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

func (r *RatingRepository) CountLowRatingsByAlgorithmSince(ctx context.Context, algorithmID int64, since time.Time) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysRating{}).
		Where("algorithm_id = ? AND rating <= 2 AND deleted = 0 AND create_time >= ?", algorithmID, since).
		Count(&count).Error
	return count, err
}

func (r *RatingRepository) GetTodayLowRatingCounts(ctx context.Context) (int64, int64, error) {
	now := time.Now()
	todayStart := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
	var lowCount, totalCount int64
	if err := r.db.WithContext(ctx).
		Model(&model.SysRating{}).
		Where("rating <= 2 AND deleted = 0 AND create_time >= ?", todayStart).
		Count(&lowCount).Error; err != nil {
		return 0, 0, err
	}
	if err := r.db.WithContext(ctx).
		Model(&model.SysRating{}).
		Where("deleted = 0 AND create_time >= ?", todayStart).
		Count(&totalCount).Error; err != nil {
		return 0, 0, err
	}
	return lowCount, totalCount, nil
}

func (r *RatingRepository) GetStatsByAlgorithmID(ctx context.Context, algorithmID int64) (int64, float64, map[int8]int64, error) {
	type row struct {
		Rating int8  `gorm:"column:rating"`
		Count  int64 `gorm:"column:count"`
	}
	var rows []row
	err := r.db.WithContext(ctx).
		Model(&model.SysRating{}).
		Select("rating, COUNT(*) as count").
		Where("algorithm_id = ? AND is_hidden = 0 AND deleted = 0", algorithmID).
		Group("rating").
		Scan(&rows).Error
	if err != nil {
		return 0, 0, nil, err
	}
	var totalCount int64
	var totalScore float64
	dist := make(map[int8]int64)
	for _, r := range rows {
		dist[r.Rating] = r.Count
		totalCount += r.Count
		totalScore += float64(r.Rating) * float64(r.Count)
	}
	if totalCount == 0 {
		return 0, 0, dist, nil
	}
	avg := totalScore / float64(totalCount)
	return totalCount, avg, dist, nil
}

var _ IRatingRepository = (*RatingRepository)(nil)
