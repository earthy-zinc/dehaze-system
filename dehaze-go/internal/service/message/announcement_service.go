package message

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	msgrepo "github.com/earthyzinc/dehaze-go/internal/repository/message"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

type AnnouncementService struct {
	annRepo    msgrepo.IAnnouncementRepository
	userRepo   msgrepo.IUserLookupRepository
	msgService IMessageService
}

func NewAnnouncementService(
	annRepo msgrepo.IAnnouncementRepository,
	userRepo msgrepo.IUserLookupRepository,
	msgService IMessageService,
) *AnnouncementService {
	return &AnnouncementService{
		annRepo:    annRepo,
		userRepo:   userRepo,
		msgService: msgService,
	}
}

func (s *AnnouncementService) Create(ctx context.Context, userID int64, form *bo.AnnouncementForm) (*vo.AnnouncementCreateResultVO, error) {
	if err := validateAnnouncementForm(form); err != nil {
		return nil, err
	}

	ann := &model.SysAnnouncement{
		Title:       form.Title,
		Content:     form.Content,
		Type:        form.Type,
		Importance:  int8(form.Importance),
		TargetScope: form.TargetScope,
		Status:      1,
	}
	ann.CreateBy = userID

	if form.TargetParams != nil {
		ann.TargetParams = toJSONString(form.TargetParams)
	}

	if form.SendTime != nil && *form.SendTime != "" {
		t, err := time.ParseInLocation(timeFormat, *form.SendTime, time.Local)
		if err != nil {
			return nil, common.NewBizError(common.PARAM_ERROR, "定时发送时间格式不正确")
		}
		ann.SendTime = &t
		ann.Status = 2
	}

	if form.ExpireTime != nil && *form.ExpireTime != "" {
		t, err := time.ParseInLocation(timeFormat, *form.ExpireTime, time.Local)
		if err != nil {
			return nil, common.NewBizError(common.PARAM_ERROR, "过期时间格式不正确")
		}
		ann.ExpireTime = &t
	}

	id, err := s.annRepo.Create(ctx, ann)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建公告失败", err)
	}

	return &vo.AnnouncementCreateResultVO{ID: id}, nil
}

func (s *AnnouncementService) Update(ctx context.Context, id int64, userID int64, form *bo.AnnouncementForm) error {
	ann, err := s.annRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询公告失败", err)
	}
	if ann == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "公告不存在")
	}
	if ann.Status != 1 && ann.Status != 2 {
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW, "公告状态不允许编辑")
	}

	updates := make(map[string]interface{})
	if form.Title != "" {
		updates["title"] = form.Title
	}
	if form.Content != "" {
		updates["content"] = form.Content
	}
	if form.Type != "" {
		updates["type"] = form.Type
	}
	if form.Importance > 0 {
		updates["importance"] = form.Importance
	}
	if form.TargetScope != "" {
		updates["target_scope"] = form.TargetScope
	}
	if form.TargetParams != nil {
		updates["target_params"] = toJSONString(form.TargetParams)
	}
	if form.SendTime != nil {
		if *form.SendTime == "" {
			updates["send_time"] = nil
		} else {
			t, err := time.ParseInLocation(timeFormat, *form.SendTime, time.Local)
			if err != nil {
				return common.NewBizError(common.PARAM_ERROR, "定时发送时间格式不正确")
			}
			updates["send_time"] = t
		}
	}
	if form.ExpireTime != nil {
		if *form.ExpireTime == "" {
			updates["expire_time"] = nil
		} else {
			t, err := time.ParseInLocation(timeFormat, *form.ExpireTime, time.Local)
			if err != nil {
				return common.NewBizError(common.PARAM_ERROR, "过期时间格式不正确")
			}
			updates["expire_time"] = t
		}
	}
	updates["update_by"] = userID

	if len(updates) > 1 {
		if err := s.annRepo.Update(ctx, id, updates); err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "更新公告失败", err)
		}
	}
	return nil
}

func (s *AnnouncementService) Delete(ctx context.Context, id int64) error {
	ann, err := s.annRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询公告失败", err)
	}
	if ann == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "公告不存在")
	}
	if err := s.annRepo.SoftDelete(ctx, id); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除公告失败", err)
	}
	return nil
}

func (s *AnnouncementService) GetDetail(ctx context.Context, id int64) (*vo.AnnouncementDetailVO, error) {
	ann, err := s.annRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询公告失败", err)
	}
	if ann == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "公告不存在")
	}
	return s.toDetailVO(ann), nil
}

func (s *AnnouncementService) GetPage(ctx context.Context, q *query.AnnouncementQuery) (*vo.PageResult[vo.AnnouncementVO], error) {
	anns, total, err := s.annRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询公告列表失败", err)
	}
	list := make([]vo.AnnouncementVO, 0, len(anns))
	for _, ann := range anns {
		list = append(list, s.toListVO(&ann))
	}
	return &vo.PageResult[vo.AnnouncementVO]{List: list, Total: total}, nil
}

func (s *AnnouncementService) Send(ctx context.Context, id int64) (*vo.AnnouncementSendResultVO, error) {
	ann, err := s.annRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询公告失败", err)
	}
	if ann == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "公告不存在")
	}
	if ann.Status != 1 && ann.Status != 2 {
		return nil, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "公告状态不允许发送")
	}

	recipientIDs, err := s.resolveTargetUserIDs(ctx, ann)
	if err != nil {
		return nil, err
	}
	if len(recipientIDs) == 0 {
		return nil, common.NewBizError(common.BUSINESS_ERROR, "发送范围为空")
	}

	msgForm := &bo.MessageSendForm{
		Type:         "announcement",
		Title:        ann.Title,
		Content:      ann.Content,
		RecipientIDs: recipientIDs,
		BizModule:    "system",
		BizID:        fmt.Sprintf("announcement_%d", ann.ID),
		Priority:     int(ann.Importance) + 1,
	}
	if _, err := s.msgService.Send(ctx, msgForm); err != nil {
		return nil, err
	}

	now := time.Now()
	updates := map[string]interface{}{
		"status":     int8(3),
		"sent_count": len(recipientIDs),
		"send_time":  now,
		"update_by":  ann.CreateBy,
	}
	if err := s.annRepo.Update(ctx, id, updates); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新公告状态失败", err)
	}

	return &vo.AnnouncementSendResultVO{SentCount: len(recipientIDs)}, nil
}

func (s *AnnouncementService) Cancel(ctx context.Context, id int64) error {
	ann, err := s.annRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询公告失败", err)
	}
	if ann == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "公告不存在")
	}
	if ann.Status != 2 {
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW, "仅待发送状态的公告可取消")
	}

	updates := map[string]interface{}{
		"status":    int8(4),
		"update_by": ann.CreateBy,
	}
	if err := s.annRepo.Update(ctx, id, updates); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "取消公告失败", err)
	}
	return nil
}

func (s *AnnouncementService) resolveTargetUserIDs(ctx context.Context, ann *model.SysAnnouncement) ([]int64, error) {
	switch ann.TargetScope {
	case "all":
		return s.userRepo.FindAllUserIDs(ctx)
	case "specified":
		if ann.TargetParams == "" {
			return nil, nil
		}
		var params struct {
			UserIDs []int64 `json:"userIds"`
		}
		if err := json.Unmarshal([]byte(ann.TargetParams), &params); err != nil {
			return nil, common.WrapBizError(common.PARAM_ERROR, "解析目标用户参数失败", err)
		}
		return params.UserIDs, nil
	case "level":
		return s.userRepo.FindAllUserIDs(ctx)
	default:
		return s.userRepo.FindAllUserIDs(ctx)
	}
}

func (s *AnnouncementService) toListVO(ann *model.SysAnnouncement) vo.AnnouncementVO {
	return vo.AnnouncementVO{
		ID:               ann.ID,
		Title:            ann.Title,
		Type:             ann.Type,
		TypeLabel:        announcementTypeLabels[ann.Type],
		Importance:       int(ann.Importance),
		TargetScope:      ann.TargetScope,
		TargetScopeLabel: targetScopeLabels[ann.TargetScope],
		Status:           int(ann.Status),
		StatusLabel:      announcementStatusLabels[int(ann.Status)],
		SendTime:         formatTime(ann.SendTime),
		ExpireTime:       formatTime(ann.ExpireTime),
		SentCount:        ann.SentCount,
		CreateTime:       formatTimeVal(ann.CreatedAt),
		CreateBy:         ann.CreateBy,
	}
}

func (s *AnnouncementService) toDetailVO(ann *model.SysAnnouncement) *vo.AnnouncementDetailVO {
	return &vo.AnnouncementDetailVO{
		ID:               ann.ID,
		Title:            ann.Title,
		Content:          ann.Content,
		Type:             ann.Type,
		TypeLabel:        announcementTypeLabels[ann.Type],
		Importance:       int(ann.Importance),
		ImportanceLabel:  importanceLabels[int(ann.Importance)],
		TargetScope:      ann.TargetScope,
		TargetScopeLabel: targetScopeLabels[ann.TargetScope],
		TargetParams:     parseJSONToInterface(ann.TargetParams),
		Status:           int(ann.Status),
		StatusLabel:      announcementStatusLabels[int(ann.Status)],
		SendTime:         formatTime(ann.SendTime),
		ExpireTime:       formatTime(ann.ExpireTime),
		SentCount:        ann.SentCount,
		CreateTime:       formatTimeVal(ann.CreatedAt),
		UpdateTime:       formatTimeVal(ann.UpdatedAt),
	}
}

func validateAnnouncementForm(form *bo.AnnouncementForm) error {
	if len([]rune(form.Title)) < 2 || len([]rune(form.Title)) > 50 {
		return common.NewBizError(common.PARAM_ERROR, "公告标题长度需为2-50字符")
	}
	if form.Content == "" {
		return common.NewBizError(common.PARAM_ERROR, "公告内容不能为空")
	}
	if form.Type == "" {
		return common.NewBizError(common.PARAM_ERROR, "公告类型不能为空")
	}
	if form.Importance != 1 && form.Importance != 2 {
		return common.NewBizError(common.PARAM_ERROR, "重要级别只能为1或2")
	}
	if form.TargetScope == "" {
		return common.NewBizError(common.PARAM_ERROR, "发送范围不能为空")
	}
	return nil
}

var _ IAnnouncementService = (*AnnouncementService)(nil)
