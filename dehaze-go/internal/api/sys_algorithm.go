package api

import (
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	favrepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm_favorite"
	algoservice "github.com/earthyzinc/dehaze-go/internal/service/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type AlgorithmApi struct {
	algorithmService algoservice.IAlgorithmService
	favRepo          favrepo.IAlgorithmFavoriteRepository
}

func NewAlgorithmApi(algorithmService algoservice.IAlgorithmService, favRepo favrepo.IAlgorithmFavoriteRepository) *AlgorithmApi {
	return &AlgorithmApi{algorithmService: algorithmService, favRepo: favRepo}
}

// GetList 获取算法树形表格
func (api *AlgorithmApi) GetList(c *gin.Context) {
	ctx := c.Request.Context()
	var q query.AlgorithmQuery
	if err := c.ShouldBindQuery(&q); err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "参数绑定失败"))
		return
	}
	result, err := api.algorithmService.GetTree(ctx, &q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// Compare 算法对比
func (api *AlgorithmApi) Compare(c *gin.Context) {
	ctx := c.Request.Context()
	ids, err := parseIDsFromCSV(c.Query("ids"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	algorithms, err := api.algorithmService.Compare(ctx, ids)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(algorithms, c)
}

// GetOptions 获取模型下拉选项列表
func (api *AlgorithmApi) GetOptions(c *gin.Context) {
	ctx := c.Request.Context()
	options, err := api.algorithmService.GetOptions(ctx)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(options, c)
}

// GetById 根据ID获取算法信息
func (api *AlgorithmApi) GetById(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "参数错误"))
		return
	}
	form, err := api.algorithmService.GetFormData(ctx, id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(form, c)
}

// Add 新增算法
func (api *AlgorithmApi) Add(c *gin.Context) {
	ctx := c.Request.Context()
	var form bo.AlgorithmFormBO
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	id, err := api.algorithmService.Create(ctx, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(id, c)
}

// Update 修改算法
func (api *AlgorithmApi) Update(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "参数错误"))
		return
	}
	var form bo.AlgorithmFormBO
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	form.ID = id
	if err := api.algorithmService.Update(ctx, id, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("修改成功", c)
}

// Delete 删除算法
func (api *AlgorithmApi) Delete(c *gin.Context) {
	ctx := c.Request.Context()
	ids, err := parseIDsFromCSV(c.Query("ids"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := api.algorithmService.Delete(ctx, ids); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("删除成功", c)
}

// UpdateStatus 更新算法状态
func (api *AlgorithmApi) UpdateStatus(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "参数错误"))
		return
	}
	var req struct {
		Status int8 `json:"status"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}
	if err := api.algorithmService.UpdateStatus(ctx, id, req.Status); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("状态更新成功", c)
}

// ToggleFavorite 切换算法收藏状态
func (api *AlgorithmApi) ToggleFavorite(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	idStr := c.Param("id")
	algorithmID, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}
	favorited, err := api.favRepo.IsFavorited(ctx, userID, algorithmID)
	if err != nil {
		_ = c.Error(common.WrapBizError(common.DATABASE_ERROR, "查询收藏状态失败", err))
		return
	}
	if favorited {
		if err := api.favRepo.Delete(ctx, userID, algorithmID); err != nil {
			_ = c.Error(common.WrapBizError(common.DATABASE_ERROR, "取消收藏失败", err))
			return
		}
		common.OkWithMessage("已取消收藏", c)
	} else {
		fav := &model.SysAlgorithmFavorite{UserID: userID, AlgorithmID: algorithmID}
		if err := api.favRepo.Create(ctx, fav); err != nil {
			_ = c.Error(common.WrapBizError(common.DATABASE_ERROR, "收藏失败", err))
			return
		}
		common.OkWithMessage("收藏成功", c)
	}
}

// ListFavorites 获取用户收藏列表
// favRepo.FindByUserID 使用 Find()，无记录时返回空切片而非 ErrRecordNotFound，
// 因此无需在 handler 层做 not-found 兜底
func (api *AlgorithmApi) ListFavorites(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	favorites, err := api.favRepo.FindByUserID(ctx, userID)
	if err != nil {
		_ = c.Error(common.WrapBizError(common.DATABASE_ERROR, "查询收藏列表失败", err))
		return
	}
	common.OkWithData(favorites, c)
}

// CheckFavorite 检查是否已收藏
func (api *AlgorithmApi) CheckFavorite(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	idStr := c.Query("algorithmId")
	algorithmID, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "参数错误"))
		return
	}
	favorited, err := api.favRepo.IsFavorited(ctx, userID, algorithmID)
	if err != nil {
		_ = c.Error(common.WrapBizError(common.DATABASE_ERROR, "查询收藏状态失败", err))
		return
	}
	common.OkWithData(favorited, c)
}

// GetVersions 获取算法版本历史
func (api *AlgorithmApi) GetVersions(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "参数错误"))
		return
	}
	versions, err := api.algorithmService.GetVersionHistory(ctx, id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(versions, c)
}

// GetMonitorData 获取算法监控数据
func (api *AlgorithmApi) GetMonitorData(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "参数错误"))
		return
	}
	monitor, err := api.algorithmService.GetMonitorData(ctx, id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(monitor, c)
}

// ExportAlgorithm 导出单个算法（返回 JSON 文件下载）
func (api *AlgorithmApi) ExportAlgorithm(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "参数错误"))
		return
	}
	jsonStr, err := api.algorithmService.ExportAlgorithmJson(ctx, id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=algorithm_%d.json", id))
	c.Data(http.StatusOK, "application/json", []byte(jsonStr))
}

// ValidateImport 校验导入包（multipart 文件上传，字段名 file）
func (api *AlgorithmApi) ValidateImport(c *gin.Context) {
	ctx := c.Request.Context()
	file, err := c.FormFile("file")
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "导入文件不能为空"))
		return
	}
	if file.Size == 0 {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "导入文件不能为空"))
		return
	}
	f, err := file.Open()
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "读取导入文件失败"))
		return
	}
	defer f.Close()
	content, err := io.ReadAll(f)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "读取导入文件失败"))
		return
	}
	result, err := api.algorithmService.ValidateImport(ctx, file.Filename, content)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}
