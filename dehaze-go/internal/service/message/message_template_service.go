package message

import (
	"context"
	"encoding/json"

	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	msgrepo "github.com/earthyzinc/dehaze-go/internal/repository/message"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

type MessageTemplateService struct {
	tplRepo msgrepo.IMessageTemplateRepository
}

func NewMessageTemplateService(tplRepo msgrepo.IMessageTemplateRepository) *MessageTemplateService {
	return &MessageTemplateService{tplRepo: tplRepo}
}

func (s *MessageTemplateService) GetPage(ctx context.Context, q *query.MessageTemplateQuery) (*vo.PageResult[vo.MessageTemplateVO], error) {
	tpls, total, err := s.tplRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询模板列表失败", err)
	}
	list := make([]vo.MessageTemplateVO, 0, len(tpls))
	for _, tpl := range tpls {
		list = append(list, vo.MessageTemplateVO{
			ID:            tpl.ID,
			Code:          tpl.Code,
			Name:          tpl.Name,
			Type:          tpl.Type,
			TitleTemplate: tpl.TitleTemplate,
			Priority:      int(tpl.Priority),
			Status:        int(tpl.Status),
			CreateTime:    formatTimeVal(tpl.CreatedAt),
		})
	}
	return &vo.PageResult[vo.MessageTemplateVO]{List: list, Total: total}, nil
}

func (s *MessageTemplateService) GetDetail(ctx context.Context, id int64) (*vo.MessageTemplateDetailVO, error) {
	tpl, err := s.tplRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询模板失败", err)
	}
	if tpl == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "模板不存在")
	}

	var channels interface{}
	if tpl.Channels != "" {
		_ = json.Unmarshal([]byte(tpl.Channels), &channels)
	}

	var variables []vo.TemplateVarVO
	if tpl.Variables != "" {
		_ = json.Unmarshal([]byte(tpl.Variables), &variables)
	}

	return &vo.MessageTemplateDetailVO{
		ID:              tpl.ID,
		Code:            tpl.Code,
		Name:            tpl.Name,
		Type:            tpl.Type,
		TitleTemplate:   tpl.TitleTemplate,
		ContentTemplate: tpl.ContentTemplate,
		Priority:        int(tpl.Priority),
		Channels:        channels,
		Variables:       variables,
		Status:          int(tpl.Status),
		CreateTime:      formatTimeVal(tpl.CreatedAt),
		UpdateTime:      formatTimeVal(tpl.UpdatedAt),
	}, nil
}

func (s *MessageTemplateService) Update(ctx context.Context, id int64, userID int64, form *bo.MessageTemplateForm) error {
	tpl, err := s.tplRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询模板失败", err)
	}
	if tpl == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "模板不存在")
	}

	updates := make(map[string]interface{})
	if form.Name != "" {
		updates["name"] = form.Name
	}
	if form.TitleTemplate != "" {
		updates["title_template"] = form.TitleTemplate
	}
	if form.ContentTemplate != "" {
		updates["content_template"] = form.ContentTemplate
	}
	if form.Priority != nil {
		updates["priority"] = *form.Priority
	}
	if form.Channels != nil {
		updates["channels"] = toJSONString(form.Channels)
	}
	if form.Status != nil {
		updates["status"] = *form.Status
	}
	updates["update_by"] = userID

	if len(updates) > 1 {
		if err := s.tplRepo.Update(ctx, id, updates); err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "更新模板失败", err)
		}
	}
	return nil
}

var _ IMessageTemplateService = (*MessageTemplateService)(nil)
