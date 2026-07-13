package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	taskrepo "github.com/earthyzinc/dehaze-go/internal/repository/task"
	taskservice "github.com/earthyzinc/dehaze-go/internal/service/task"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

// SysTaskApi 任务管理 API
type SysTaskApi struct {
	taskService *taskservice.TaskService
	taskRepo    taskrepo.ITaskRepository
}

func NewSysTaskApi(taskService *taskservice.TaskService, taskRepo taskrepo.ITaskRepository) *SysTaskApi {
	return &SysTaskApi{taskService: taskService, taskRepo: taskRepo}
}

// CreateTask 创建任务
// 统一任务接口：同步创建任务记录（PENDING），异步执行具体策略
// @Summary 创建任务
// @Tags 任务接口
// @Accept application/json
// @Produce application/json
// @Param form body bo.ExportTaskCreateForm true "任务创建表单"
// @Success 200 {object} common.Response{data=vo.TaskVO}
// @Router /api/v1/tasks [post]
func (api *SysTaskApi) CreateTask(c *gin.Context) {
	var form bo.ExportTaskCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	userID := getCurrentUserID(c)
	task, err := api.taskService.CreateExportTask(form, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}

	taskVO := api.taskService.ConvertToTaskVO(task)
	common.OkWithDetailed(taskVO, "任务创建成功", c)
}

// GetTaskPage 任务分页列表
func (api *SysTaskApi) GetTaskPage(c *gin.Context) {
	ctx := c.Request.Context()
	pageNum, pageSize := getPageParams(c)
	result, err := api.taskRepo.FindPage(ctx, map[string]interface{}{
		"pageNum":  pageNum,
		"pageSize": pageSize,
	})
	if err != nil {
		_ = c.Error(common.WrapBizError(common.DATABASE_ERROR, "查询任务列表失败", err))
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// GetTaskById 任务详情
func (api *SysTaskApi) GetTaskById(c *gin.Context) {
	idStr := c.Param("id")
	task, err := api.taskService.GetTaskStatus(idStr)
	if err != nil {
		_ = c.Error(err)
		return
	}

	if task == nil {
		common.OkWithData(nil, c)
		return
	}

	taskVO := api.taskService.ConvertToTaskVO(task)
	common.OkWithData(taskVO, c)
}

// CancelTask 取消任务
func (api *SysTaskApi) CancelTask(c *gin.Context) {
	idStr := c.Param("id")
	userID := getCurrentUserID(c)
	if err := api.taskService.CancelTask(idStr, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("任务已取消", c)
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

// getCurrentUserID 从 JWT claims 获取当前用户 ID
func getCurrentUserID(c *gin.Context) int64 {
	if claims, exists := c.Get("claims"); exists {
		type userIDGetter interface {
			GetUserID() int64
		}
		if u, ok := claims.(userIDGetter); ok {
			return u.GetUserID()
		}
	}
	return 0
}
