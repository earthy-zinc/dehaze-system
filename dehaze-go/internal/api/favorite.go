package api

import (
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	favoriteservice "github.com/earthyzinc/dehaze-go/internal/service/favorite"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type FavoriteApi struct {
	favService favoriteservice.IFavoriteService
}

func NewFavoriteApi(favService favoriteservice.IFavoriteService) *FavoriteApi {
	return &FavoriteApi{favService: favService}
}

func (api *FavoriteApi) GetPage(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	q := &query.FavoritePageQuery{
		TargetType: c.Query("targetType"),
		Keywords:   c.Query("keywords"),
		SortBy:     c.Query("sortBy"),
		SortOrder:  c.Query("sortOrder"),
		PageNum:    1,
		PageSize:   20,
	}
	if v := c.Query("pageNum"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageNum = n
		}
	}
	if v := c.Query("pageSize"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageSize = n
		}
	}

	result, err := api.favService.GetPage(c.Request.Context(), userID, q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *FavoriteApi) Add(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.FavoriteForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	id, err := api.favService.Add(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(id, c)
}

func (api *FavoriteApi) DeleteByIDs(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	idsStr := c.Param("ids")
	idList := parseIDList(idsStr)

	if err := api.favService.DeleteByIDs(c.Request.Context(), userID, idList); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("取消收藏成功", c)
}

func (api *FavoriteApi) GetStatus(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	targetID, err := strconv.ParseInt(c.Param("targetId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	targetType := c.Query("targetType")

	result, err := api.favService.GetStatus(c.Request.Context(), userID, targetType, targetID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *FavoriteApi) GetCount(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	targetType := c.Query("targetType")

	result, err := api.favService.GetCount(c.Request.Context(), userID, targetType)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func parseIDList(idsStr string) []int64 {
	if idsStr == "" {
		return nil
	}
	parts := strings.Split(idsStr, ",")
	ids := make([]int64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if n, err := strconv.ParseInt(p, 10, 64); err == nil {
			ids = append(ids, n)
		}
	}
	return ids
}
