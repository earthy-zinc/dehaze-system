package message

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
)

// IMessageRepository 消息仓储接口
type IMessageRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysMessage, error)
	FindByBizModuleAndBizID(ctx context.Context, bizModule, bizID string) ([]model.SysMessage, error)
	FindByBizModuleAndBizIDAndRecipientIDs(ctx context.Context, bizModule, bizID string, recipientIDs []int64) ([]model.SysMessage, error)
	FindPage(ctx context.Context, userID int64, q *query.MessageQuery) ([]model.SysMessage, int64, error)
	SearchPage(ctx context.Context, userID int64, q *query.MessageSearchQuery) ([]model.SysMessage, int64, error)
	CountUnread(ctx context.Context, userID int64) (int64, error)
	Create(ctx context.Context, msg *model.SysMessage) error
	CreateBatch(ctx context.Context, msgs []model.SysMessage) ([]int64, error)
	MarkRead(ctx context.Context, id, userID int64) (int64, error)
	MarkAllRead(ctx context.Context, userID int64, msgType string) (int64, error)
	SoftDelete(ctx context.Context, ids []int64, userID int64) error
	DeleteExpiredBatch(ctx context.Context, before time.Time, batchSize int) (int64, error)
}

// IMessageTemplateRepository 消息模板仓储接口
type IMessageTemplateRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysMessageTemplate, error)
	FindByCode(ctx context.Context, code string) (*model.SysMessageTemplate, error)
	FindPage(ctx context.Context, q *query.MessageTemplateQuery) ([]model.SysMessageTemplate, int64, error)
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
}

// INotificationSettingRepository 通知设置仓储接口
type INotificationSettingRepository interface {
	FindByUserID(ctx context.Context, userID int64) (*model.SysNotificationSetting, error)
	Create(ctx context.Context, setting *model.SysNotificationSetting) error
	Update(ctx context.Context, setting *model.SysNotificationSetting) error
}

// IAnnouncementRepository 公告仓储接口
type IAnnouncementRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysAnnouncement, error)
	FindPage(ctx context.Context, q *query.AnnouncementQuery) ([]model.SysAnnouncement, int64, error)
	FindPendingScheduled(ctx context.Context, before time.Time) ([]model.SysAnnouncement, error)
	Create(ctx context.Context, ann *model.SysAnnouncement) (int64, error)
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	SoftDelete(ctx context.Context, id int64) error
}

// IUserLookupRepository 用户查询接口（用于公告发送时查询目标用户）
type IUserLookupRepository interface {
	FindAllUserIDs(ctx context.Context) ([]int64, error)
	FindUserIDsByLevel(ctx context.Context, level int) ([]int64, error)
}
