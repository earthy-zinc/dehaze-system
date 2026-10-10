package feedback

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	fbrepo "github.com/earthyzinc/dehaze-go/internal/repository/feedback"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	msgservice "github.com/earthyzinc/dehaze-go/internal/service/message"
	"github.com/earthyzinc/dehaze-go/pkg/mq"
	"go.uber.org/zap"
)

const (
	lowRatingAlertQueue  = "feedback.low_rating"
	alertUrgentThreshold = int64(3)
	alertSevereThreshold = 0.2
	alertWindow          = 24 * time.Hour
)

type ratingEvent struct {
	RatingID  int64     `json:"ratingId"`
	UserID    int64     `json:"userId"`
	ModelID   string    `json:"modelId"`
	Rating    int8      `json:"rating"`
	Comment   string    `json:"comment"`
	CreatedAt time.Time `json:"createdAt"`
}

type LowRatingAlertService struct {
	ratingRepo     fbrepo.IRatingRepository
	userRepo       userrepo.IUserRepository
	modelRepo      *airepo.ModelRepository
	messageService msgservice.IMessageService
	publisher      *mq.Publisher
	logger         *zap.Logger
}

func NewLowRatingAlertService(
	ratingRepo fbrepo.IRatingRepository,
	userRepo userrepo.IUserRepository,
	modelRepo *airepo.ModelRepository,
	messageService msgservice.IMessageService,
	publisher *mq.Publisher,
	logger *zap.Logger,
) *LowRatingAlertService {
	return &LowRatingAlertService{
		ratingRepo:     ratingRepo,
		userRepo:       userRepo,
		modelRepo:      modelRepo,
		messageService: messageService,
		publisher:      publisher,
		logger:         logger,
	}
}

func (s *LowRatingAlertService) PublishRatingEvent(ctx context.Context, rating *model.SysRating) error {
	if rating.Rating > 2 {
		return nil
	}
	if s.publisher == nil || !s.publisher.IsConnected() {
		s.logger.Warn("MQ Publisher 不可用，跳过低分告警事件发布",
			zap.Int64("ratingId", rating.ID))
		return nil
	}

	event := ratingEvent{
		RatingID:  rating.ID,
		UserID:    rating.UserID,
		ModelID:   rating.ModelID,
		Rating:    rating.Rating,
		Comment:   rating.Comment,
		CreatedAt: rating.CreatedAt,
	}
	body, err := json.Marshal(event)
	if err != nil {
		return err
	}

	routingKey := s.buildRoutingKey()
	if err := s.publisher.Publish(ctx, routingKey, body); err != nil {
		s.logger.Error("发布低分告警事件失败",
			zap.Int64("ratingId", rating.ID),
			zap.Error(err))
		return err
	}
	return nil
}

func (s *LowRatingAlertService) HandleMessage(ctx context.Context, body []byte) error {
	var event ratingEvent
	if err := json.Unmarshal(body, &event); err != nil {
		return err
	}
	return s.CheckAndAlert(ctx, event.RatingID)
}

func (s *LowRatingAlertService) CheckAndAlert(ctx context.Context, ratingID int64) error {
	rating, err := s.ratingRepo.FindByID(ctx, ratingID)
	if err != nil {
		return err
	}
	if rating == nil {
		return nil
	}
	if rating.Rating > 2 {
		return nil
	}

	adminIDs, err := s.findAdminUserIDs(ctx)
	if err != nil {
		s.logger.Error("查询管理员用户失败", zap.Error(err))
		return err
	}
	if len(adminIDs) == 0 {
		return nil
	}

	s.sendNormalAlert(ctx, rating, adminIDs)

	if rating.Rating == 1 {
		s.checkUrgentAlert(ctx, rating, adminIDs)
	}

	s.checkSevereAlert(ctx, rating, adminIDs)
	return nil
}

func (s *LowRatingAlertService) sendNormalAlert(ctx context.Context, rating *model.SysRating, adminIDs []int64) {
	modelName := s.findModelName(ctx, rating.ModelID)
	title := fmt.Sprintf("低分评价告警（%d星）", rating.Rating)
	content := fmt.Sprintf("用户ID:%d 对模型「%s」提交了 %d 星评价。评价内容：%s",
		rating.UserID, modelName, rating.Rating, rating.Comment)

	form := &bo.MessageSendForm{
		Type:         "alert",
		Title:        title,
		Content:      content,
		RecipientIDs: adminIDs,
		BizModule:    "feedback",
		BizID:        fmt.Sprintf("rating_alert_%d", rating.ID),
		Priority:     3,
	}
	if _, err := s.messageService.Send(ctx, form); err != nil {
		s.logger.Error("发送普通低分告警失败",
			zap.Int64("ratingId", rating.ID),
			zap.Error(err))
	}
}

func (s *LowRatingAlertService) checkUrgentAlert(ctx context.Context, rating *model.SysRating, adminIDs []int64) {
	since := time.Now().Add(-alertWindow)
	lowCount, err := s.ratingRepo.CountLowRatingsByModelSince(ctx, rating.ModelID, since)
	if err != nil {
		s.logger.Error("查询同模型24小时低分评价数失败", zap.Error(err))
		return
	}
	if lowCount < alertUrgentThreshold {
		return
	}

	modelName := s.findModelName(ctx, rating.ModelID)
	title := "紧急低分告警：同模型24小时内低分评价达标"
	content := fmt.Sprintf("模型「%s」24小时内收到 %d 条低分评价（阈值 %d），请及时排查模型质量问题。",
		modelName, lowCount, alertUrgentThreshold)

	form := &bo.MessageSendForm{
		Type:         "critical_alert",
		Title:        title,
		Content:      content,
		RecipientIDs: adminIDs,
		BizModule:    "feedback",
		BizID:        fmt.Sprintf("rating_urgent_%s_%d", rating.ModelID, time.Now().Unix()),
		Priority:     1,
	}
	if _, err := s.messageService.Send(ctx, form); err != nil {
		s.logger.Error("发送紧急低分告警失败",
			zap.String("modelId", rating.ModelID),
			zap.Error(err))
	}
}

func (s *LowRatingAlertService) checkSevereAlert(ctx context.Context, rating *model.SysRating, adminIDs []int64) {
	lowCount, totalCount, err := s.ratingRepo.GetTodayLowRatingCounts(ctx)
	if err != nil {
		s.logger.Error("查询当日全局低分率失败", zap.Error(err))
		return
	}
	if totalCount == 0 {
		return
	}
	lowRate := float64(lowCount) / float64(totalCount)
	if lowRate <= alertSevereThreshold {
		return
	}

	title := "严重低分告警：全局低分率超标"
	content := fmt.Sprintf("当日全局低分率 %.2f%%（阈值 %.0f%%），低分评价 %d/%d，请立即排查。",
		lowRate*100, alertSevereThreshold*100, lowCount, totalCount)

	form := &bo.MessageSendForm{
		Type:         "critical_alert",
		Title:        title,
		Content:      content,
		RecipientIDs: adminIDs,
		BizModule:    "feedback",
		BizID:        fmt.Sprintf("rating_severe_%d", time.Now().Unix()),
		Priority:     2,
	}
	if _, err := s.messageService.Send(ctx, form); err != nil {
		s.logger.Error("发送严重低分告警失败", zap.Error(err))
	}
}

func (s *LowRatingAlertService) findAdminUserIDs(ctx context.Context) ([]int64, error) {
	return s.userRepo.FindAdminUserIDs(ctx)
}

func (s *LowRatingAlertService) findModelName(ctx context.Context, modelID string) string {
	model, err := s.modelRepo.GetByModelID(ctx, modelID)
	if err != nil || model == nil {
		s.logger.Warn("查询模型名称失败",
			zap.String("modelID", modelID),
			zap.Error(err))
		return ""
	}
	return model.DisplayName
}

func (s *LowRatingAlertService) buildRoutingKey() string {
	return lowRatingAlertQueue
}

var _ ILowRatingAlertService = (*LowRatingAlertService)(nil)
