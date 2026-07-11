package api

import (
	"strconv"

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
	common.OkWithData(task, c)
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
