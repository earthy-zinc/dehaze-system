package message

import (
	"context"
	"regexp"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	msgrepo "github.com/earthyzinc/dehaze-go/internal/repository/message"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

type MessageService struct {
	msgRepo     msgrepo.IMessageRepository
	tplRepo     msgrepo.IMessageTemplateRepository
}

func NewMessageService(msgRepo msgrepo.IMessageRepository, tplRepo msgrepo.IMessageTemplateRepository) *MessageService {
	return &MessageService{msgRepo: msgRepo, tplRepo: tplRepo}
}

var varPattern = regexp.MustCompile(`\{(\w+)\}`)

func (s *MessageService) Send(ctx context.Context, form *bo.MessageSendForm) (*vo.MessageSendResultVO, error) {
	if form.Type == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "消息类型不能为空")
	}
	if len(form.RecipientIDs) == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "接收人列表不能为空")
	}

	if form.BizModule != "" && form.BizID != "" {
		existing, err := s.msgRepo.FindByBizModuleAndBizID(ctx, form.BizModule, form.BizID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "幂等检查失败", err)
		}
		if len(existing) > 0 {
			ids := make([]int64, 0, len(existing))
			for _, m := range existing {
				ids = append(ids, m.ID)
			}
			return &vo.MessageSendResultVO{MessageIDs: ids}, nil
		}
	}

	title := form.Title
	content := form.Content
	prioritySet := form.Priority > 0
	priority := int8(form.Priority)
	senderType := int8(1)

	if form.TemplateCode != "" {
		tpl, err := s.tplRepo.FindByCode(ctx, form.TemplateCode)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询消息模板失败", err)
		}
		if tpl == nil {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "消息模板不存在")
		}
		if tpl.Status == 0 {
			return nil, common.NewBizError(common.BUSINESS_ERROR, "消息模板已禁用")
		}

		missing := checkTemplateVars(tpl.TitleTemplate+" "+tpl.ContentTemplate, form.Variables)
		if len(missing) > 0 {
			return nil, common.NewBizError(common.PARAM_ERROR, "模板变量缺失: "+missing[0])
		}

		title = renderTemplate(tpl.TitleTemplate, form.Variables)
		content = renderTemplate(tpl.ContentTemplate, form.Variables)
		if !prioritySet {
			priority = tpl.Priority
		}
	}

	if priority == 0 {
		priority = 2
	}

	if title == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "消息标题不能为空")
	}
	if content == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "消息正文不能为空")
	}

	expiresAt := calcExpiresAt(form.Type)
	extraStr := ""
	if len(form.Extra) > 0 {
		extraStr = toJSONString(form.Extra)
	}

	msgs := make([]model.SysMessage, 0, len(form.RecipientIDs))
	for _, rid := range form.RecipientIDs {
		msgs = append(msgs, model.SysMessage{
			Type:        form.Type,
			Title:       title,
			Content:     content,
			SenderType:  senderType,
			RecipientID: rid,
			BizModule:   form.BizModule,
			BizID:       form.BizID,
			Priority:    priority,
			JumpURL:     form.JumpURL,
			Extra:       extraStr,
			ReadStatus:  0,
			Deleted:     0,
			ExpiresAt:   expiresAt,
		})
	}

	ids, err := s.msgRepo.CreateBatch(ctx, msgs)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建消息失败", err)
	}

	return &vo.MessageSendResultVO{MessageIDs: ids}, nil
}

func (s *MessageService) GetPage(ctx context.Context, userID int64, q *query.MessageQuery) (*vo.PageResult[vo.MessageVO], error) {
	msgs, total, err := s.msgRepo.FindPage(ctx, userID, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询消息列表失败", err)
	}
	list := make([]vo.MessageVO, 0, len(msgs))
	for _, m := range msgs {
		list = append(list, vo.MessageVO{
			ID:         m.ID,
			Type:       m.Type,
			TypeLabel:  messageTypeLabels[m.Type],
			Title:      m.Title,
			Summary:    summary(m.Content),
			Priority:   int(m.Priority),
			ReadStatus: int(m.ReadStatus),
			SenderType: int(m.SenderType),
			JumpURL:    m.JumpURL,
			CreateTime: formatTimeVal(m.CreatedAt),
		})
	}
	return &vo.PageResult[vo.MessageVO]{List: list, Total: total}, nil
}

func (s *MessageService) GetDetail(ctx context.Context, id, userID int64) (*vo.MessageDetailVO, error) {
	msg, err := s.msgRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询消息失败", err)
	}
	if msg == nil || msg.Deleted == 1 || msg.RecipientID != userID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "消息不存在")
	}

	if msg.ReadStatus == 0 {
		_, _ = s.msgRepo.MarkRead(ctx, id, userID)
		msg.ReadStatus = 1
		now := time.Now()
		msg.ReadTime = &now
	}

	return &vo.MessageDetailVO{
		ID:              msg.ID,
		Type:            msg.Type,
		TypeLabel:       messageTypeLabels[msg.Type],
		Title:           msg.Title,
		Content:         msg.Content,
		Priority:        int(msg.Priority),
		SenderType:      int(msg.SenderType),
		SenderTypeLabel: senderTypeLabels[int(msg.SenderType)],
		ReadStatus:      int(msg.ReadStatus),
		ReadTime:        formatTime(msg.ReadTime),
		JumpURL:         msg.JumpURL,
		Extra:           parseJSONToInterface(msg.Extra),
		CreateTime:      formatTimeVal(msg.CreatedAt),
	}, nil
}

func (s *MessageService) GetUnreadCount(ctx context.Context, userID int64) (*vo.UnreadCountVO, error) {
	count, err := s.msgRepo.CountUnread(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询未读消息数失败", err)
	}
	return &vo.UnreadCountVO{Count: count}, nil
}

func (s *MessageService) MarkRead(ctx context.Context, id, userID int64) error {
	_, err := s.msgRepo.MarkRead(ctx, id, userID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "标记已读失败", err)
	}
	return nil
}

func (s *MessageService) MarkAllRead(ctx context.Context, userID int64, msgType string) (*vo.ReadAllResultVO, error) {
	affected, err := s.msgRepo.MarkAllRead(ctx, userID, msgType)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "全部标记已读失败", err)
	}
	return &vo.ReadAllResultVO{AffectedCount: affected}, nil
}

func (s *MessageService) Delete(ctx context.Context, ids []int64, userID int64) error {
	if err := s.msgRepo.SoftDelete(ctx, ids, userID); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除消息失败", err)
	}
	return nil
}

func (s *MessageService) CleanupExpired(ctx context.Context) error {
	total, err := s.msgRepo.DeleteExpiredBatch(ctx, time.Now(), 500)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "清理过期消息失败", err)
	}
	if total > 0 {
		logger.Info("过期消息清理完成", zap.Int64("count", total))
	}
	return nil
}

func (s *MessageService) Search(ctx context.Context, userID int64, q *query.MessageSearchQuery) (*vo.PageResult[vo.MessageVO], error) {
	if q.Keyword == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "搜索关键字不能为空")
	}
	msgs, total, err := s.msgRepo.SearchPage(ctx, userID, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "搜索消息失败", err)
	}
	list := make([]vo.MessageVO, 0, len(msgs))
	for _, m := range msgs {
		list = append(list, vo.MessageVO{
			ID:         m.ID,
			Type:       m.Type,
			TypeLabel:  messageTypeLabels[m.Type],
			Title:      m.Title,
			Summary:    summary(m.Content),
			Priority:   int(m.Priority),
			ReadStatus: int(m.ReadStatus),
			SenderType: int(m.SenderType),
			JumpURL:    m.JumpURL,
			CreateTime: formatTimeVal(m.CreatedAt),
		})
	}
	return &vo.PageResult[vo.MessageVO]{List: list, Total: total}, nil
}

func checkTemplateVars(template string, variables map[string]string) []string {
	matches := varPattern.FindAllStringSubmatch(template, -1)
	var missing []string
	seen := make(map[string]bool)
	for _, m := range matches {
		name := m[1]
		if seen[name] {
			continue
		}
		seen[name] = true
		if _, ok := variables[name]; !ok {
			missing = append(missing, name)
		}
	}
	return missing
}

func renderTemplate(template string, variables map[string]string) string {
	return varPattern.ReplaceAllStringFunc(template, func(match string) string {
		name := match[1 : len(match)-1]
		if v, ok := variables[name]; ok {
			return v
		}
		return ""
	})
}

func calcExpiresAt(msgType string) *time.Time {
	var days int
	switch msgType {
	case "alert":
		days = 7
	case "critical_alert":
		days = 90
	default:
		days = 30
	}
	t := time.Now().AddDate(0, 0, days)
	return &t
}

var _ IMessageService = (*MessageService)(nil)
