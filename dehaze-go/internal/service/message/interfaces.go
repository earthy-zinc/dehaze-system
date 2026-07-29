package message

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// IMessageService 消息服务接口
type IMessageService interface {
	Send(ctx context.Context, form *bo.MessageSendForm) (*vo.MessageSendResultVO, error)
	GetPage(ctx context.Context, userID int64, q *query.MessageQuery) (*vo.PageResult[vo.MessageVO], error)
	GetDetail(ctx context.Context, id, userID int64) (*vo.MessageDetailVO, error)
	GetUnreadCount(ctx context.Context, userID int64) (*vo.UnreadCountVO, error)
	MarkRead(ctx context.Context, id, userID int64) error
	MarkAllRead(ctx context.Context, userID int64, msgType string) (*vo.ReadAllResultVO, error)
	Delete(ctx context.Context, ids []int64, userID int64) error
	Search(ctx context.Context, userID int64, q *query.MessageSearchQuery) (*vo.PageResult[vo.MessageVO], error)
	RefreshUnreadCountCache(ctx context.Context) error
}

// IAnnouncementService 公告服务接口
type IAnnouncementService interface {
	Create(ctx context.Context, userID int64, form *bo.AnnouncementForm) (*vo.AnnouncementCreateResultVO, error)
	Update(ctx context.Context, id int64, userID int64, form *bo.AnnouncementForm) error
	Delete(ctx context.Context, id int64) error
	GetDetail(ctx context.Context, id int64) (*vo.AnnouncementDetailVO, error)
	GetPage(ctx context.Context, q *query.AnnouncementQuery) (*vo.PageResult[vo.AnnouncementVO], error)
	Send(ctx context.Context, id int64) (*vo.AnnouncementSendResultVO, error)
	Cancel(ctx context.Context, id int64) error
}

// IMessageTemplateService 消息模板服务接口
type IMessageTemplateService interface {
	GetPage(ctx context.Context, q *query.MessageTemplateQuery) (*vo.PageResult[vo.MessageTemplateVO], error)
	GetDetail(ctx context.Context, id int64) (*vo.MessageTemplateDetailVO, error)
	Update(ctx context.Context, id int64, userID int64, form *bo.MessageTemplateForm) error
}

// INotificationSettingService 通知设置服务接口
type INotificationSettingService interface {
	Get(ctx context.Context, userID int64) (*vo.NotificationSettingsVO, error)
	Update(ctx context.Context, userID int64, form *bo.NotificationSettingForm) error
}
