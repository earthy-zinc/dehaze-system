package api

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	taskservice "github.com/earthyzinc/dehaze-go/internal/service/task"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// SysTaskApi 任务管理 API
type SysTaskApi struct {
	taskService *taskservice.TaskService
}

func NewSysTaskApi(taskService *taskservice.TaskService) *SysTaskApi {
	return &SysTaskApi{taskService: taskService}
}

// CreateTask 创建任务
// 统一任务接口：同步创建任务记录（PENDING），异步执行具体策略
// 支持 Idempotency-Key 请求头进行幂等去重
// @Summary 创建任务
// @Tags 任务接口
// @Accept application/json
// @Produce application/json
// @Param form body bo.TaskCreateForm true "任务创建表单"
// @Param Idempotency-Key header string false "客户端幂等键"
// @Success 200 {object} common.Response{data=vo.TaskVO}
// @Router /api/v1/tasks [post]
func (api *SysTaskApi) CreateTask(c *gin.Context) {
	var form bo.TaskCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	idempotencyKey := c.GetHeader("Idempotency-Key")
	task, err := api.taskService.CreateTask(c.Request.Context(), form.Type, form.Params, userID, idempotencyKey)
	if err != nil {
		_ = c.Error(err)
		return
	}

	taskVO := api.taskService.ConvertToTaskVO(c.Request.Context(), task)
	common.OkWithDetailed(taskVO, "任务创建成功", c)
}

// GetTaskPage 任务分页列表
func (api *SysTaskApi) GetTaskPage(c *gin.Context) {
	ctx := c.Request.Context()
	pageNum, pageSize := getPageParams(c)
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	q := &query.TaskPageQuery{
		PageNum:      pageNum,
		PageSize:     pageSize,
		TaskType:     c.Query("taskType"),
		TaskCategory: c.Query("taskCategory"),
		UserID:       userID,
	}
	if statusStr := c.Query("status"); statusStr != "" {
		if statusVal, err := strconv.ParseInt(statusStr, 10, 8); err == nil {
			s := int8(statusVal)
			q.Status = &s
		}
	}
	result, err := api.taskService.GetPage(ctx, q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// GetTaskById 任务详情
func (api *SysTaskApi) GetTaskById(c *gin.Context) {
	idStr := c.Param("id")
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	// 权限校验：仅任务创建者可查询
	if _, err := api.taskService.CheckTaskOwnership(c.Request.Context(), idStr, userID); err != nil {
		_ = c.Error(err)
		return
	}
	task, err := api.taskService.GetTaskStatus(c.Request.Context(), idStr)
	if err != nil {
		_ = c.Error(err)
		return
	}

	if task == nil {
		_ = c.Error(common.NewBizError(common.TASK_NOT_FOUND, "任务不存在: "+idStr))
		return
	}

	taskVO := api.taskService.ConvertToTaskVO(c.Request.Context(), task)
	common.OkWithData(taskVO, c)
}

// DownloadExportFile 下载导出文件（302重定向到文件存储URL）
// @Summary 下载任务结果
// @Tags 任务接口
// @Produce json
// @Param id path string true "任务ID"
// @Router /api/v1/tasks/{id}/download [get]
func (api *SysTaskApi) DownloadExportFile(c *gin.Context) {
	idStr := c.Param("id")
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	// 权限校验：仅任务创建者可下载
	if _, err := api.taskService.CheckTaskOwnership(c.Request.Context(), idStr, userID); err != nil {
		_ = c.Error(err)
		return
	}
	downloadURL, err := api.taskService.DownloadExportFile(c.Request.Context(), idStr)
	if err != nil {
		_ = c.Error(err)
		return
	}
	c.Redirect(http.StatusFound, downloadURL)
}

// CancelTask 取消任务
func (api *SysTaskApi) CancelTask(c *gin.Context) {
	idStr := c.Param("id")
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := api.taskService.CancelTask(c.Request.Context(), idStr, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("任务已取消", c)
}

// RetryTask 重试失败的任务
// @Summary 重试任务
// @Tags 任务接口
// @Produce application/json
// @Param id path string true "任务ID"
// @Success 200 {object} common.Response{data=vo.TaskVO}
// @Router /api/v1/tasks/{id}/retry [post]
func (api *SysTaskApi) RetryTask(c *gin.Context) {
	idStr := c.Param("id")
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	task, err := api.taskService.RetryTask(c.Request.Context(), idStr, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	taskVO := api.taskService.ConvertToTaskVO(c.Request.Context(), task)
	common.OkWithDetailed(taskVO, "任务已重新提交", c)
}

// getPageParams 从请求中提取分页参数
func getPageParams(c *gin.Context) (int, int) {
	pageNum := 1
	pageSize := 10
	if v := c.Query("pageNum"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			pageNum = n
		}
	}
	if v := c.Query("pageSize"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			pageSize = n
		}
	}
	return pageNum, pageSize
}

// parseIDsFromCSV 将逗号分隔的 ID 字符串解析为 []int64
// 行为约定：
//   - 自动 TrimSpace 每个元素
//   - 跳过空字符串
//   - 遇到第一个无法解析的元素立即返回 PARAM_ERROR，避免静默丢弃非法 ID 造成
//     "用户以为查询/删除 3 个但实际只处理了 2 个" 的数据一致性问题
func parseIDsFromCSV(csvStr string) ([]int64, error) {
	if csvStr == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "ID列表不能为空")
	}
	parts := strings.Split(csvStr, ",")
	ids := make([]int64, 0, len(parts))
	for _, s := range parts {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		id, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, common.NewBizError(common.PARAM_ERROR, "ID格式不正确: "+s)
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "ID列表不能为空")
	}
	return ids, nil
}
