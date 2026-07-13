package api

import (
	"context"
	"fmt"
	"mime/multipart"
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	datasetservice "github.com/earthyzinc/dehaze-go/internal/service/dataset"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type SysDatasetItemApi struct {
	datasetItemService *datasetservice.DatasetItemService
	operationService   *datasetservice.DatasetOperationService
	fileService        *fileservice.FileService
}

func NewSysDatasetItemApi(
	datasetItemService *datasetservice.DatasetItemService,
	operationService *datasetservice.DatasetOperationService,
	fileService *fileservice.FileService,
) *SysDatasetItemApi {
	return &SysDatasetItemApi{
		datasetItemService: datasetItemService,
		operationService:   operationService,
		fileService:        fileService,
	}
}

// GetDatasetItemById 获取数据项详情
// @Summary 获取数据项详情
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据项ID"
// @Success 200 {object} common.Response
// @Router /api/v1/dataset-items/{id} [get]
func (api *SysDatasetItemApi) GetDatasetItemById(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据项ID格式不正确"))
		return
	}

	itemVO, err := api.datasetItemService.GetDatasetItemVOByID(id)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(itemVO, "查询成功", c)
}

// GetDatasetItems 分页查询数据项列表
// @Summary 分页查询数据项列表
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param pageNum query int false "页码" default(1)
// @Param pageSize query int false "每页数量" default(10)
// @Param datasetId query int false "数据集ID"
// @Param sceneType query string false "场景类型"
// @Param keyword query string false "关键字（搜索文件名/描述）"
// @Param hazeLevel query string false "雾霾程度（light/medium/heavy）"
// @Success 200 {object} common.Response{data=[]vo.ImageItemVO}
// @Router /api/v1/dataset-items [get]
func (api *SysDatasetItemApi) GetDatasetItems(c *gin.Context) {
	pageNumStr := c.DefaultQuery("pageNum", "1")
	pageSizeStr := c.DefaultQuery("pageSize", "10")
	pageNum, _ := strconv.ParseInt(pageNumStr, 10, 64)
	pageSize, _ := strconv.ParseInt(pageSizeStr, 10, 64)

	datasetIdStr := c.Query("datasetId")
	sceneType := c.Query("sceneType")
	keyword := c.Query("keyword")
	hazeLevel := c.Query("hazeLevel")

	var datasetId int64
	var err error
	if datasetIdStr != "" {
		datasetId, err = strconv.ParseInt(datasetIdStr, 10, 64)
		if err != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据集ID格式不正确"))
			return
		}
	}

	items, total, err := api.datasetItemService.GetDatasetItemsByPage(int(pageNum), int(pageSize), datasetId, sceneType, keyword, hazeLevel)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result := common.PageResult{
		List:     items,
		Total:    total,
		Page:     int(pageNum),
		PageSize: int(pageSize),
	}

	common.OkWithDetailed(result, "查询成功", c)
}

// createDatasetItemRequest 创建空数据项请求
type createDatasetItemRequest struct {
	DatasetID  int64  `json:"datasetId" binding:"required"`
	Name       string `json:"name"`
	SceneType  string `json:"sceneType"`
	Description string `json:"description"`
}

// CreateDatasetItem 创建空数据项
// @Summary 创建空数据项
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param request body createDatasetItemRequest true "创建请求"
// @Success 200 {object} common.Response{data=vo.ImageItemVO}
// @Router /api/v1/dataset-items [post]
func (api *SysDatasetItemApi) CreateDatasetItem(c *gin.Context) {
	var req createDatasetItemRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	item, err := api.datasetItemService.CreateDatasetItemWithName(req.DatasetID, req.Name)
	if err != nil {
		_ = c.Error(err)
		return
	}

	// 返回完整 VO
	itemVO, err := api.datasetItemService.GetDatasetItemVOByID(item.ID)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(itemVO, "创建数据项成功", c)
}

// uploadDatasetItemRequest 创建数据项并上传配对图片的请求（转发到 operationService）
// 复用 datasetservice.CreateDatasetItemWithImagesRequest

// CreateDatasetItemWithImages 创建数据项并上传配对图片
// @Summary 创建数据项并上传配对图片
// @Tags 数据项接口
// @Accept multipart/form-data
// @Produce application/json
// @Param datasetId formData int true "数据集ID"
// @Param name formData string false "数据项名称"
// @Param sceneType formData string false "场景类型"
// @Param clearImage formData file false "清晰图"
// @Param hazyImages formData file false "有雾图（可多个）"
// @Param hazeLevels formData string false "雾霾程度（与hazyImages一一对应）"
// @Success 200 {object} common.Response{data=vo.ImageItemVO}
// @Router /api/v1/dataset-items/upload [post]
func (api *SysDatasetItemApi) CreateDatasetItemWithImages(c *gin.Context) {
	ctx := c.Request.Context()

	datasetIdStr := c.PostForm("datasetId")
	datasetId, err := strconv.ParseInt(datasetIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据集ID格式不正确"))
		return
	}

	itemName := c.PostForm("name")
	sceneType := c.PostForm("sceneType")

	req := datasetservice.CreateDatasetItemWithImagesRequest{
		DatasetID: datasetId,
		ItemName:  itemName,
		SceneType: sceneType,
		Options: datasetservice.CreateItemOptions{
			AsyncThumbnail: true,
		},
	}

	baseURL := fmt.Sprintf("http://%s/api/v1/files/download", c.Request.Host)

	// 上传清晰图（可选）
	if clearHeader, err := c.FormFile("clearImage"); err == nil {
		clearInfo, err := api.uploadFormFile(ctx, clearHeader, baseURL)
		if err != nil {
			_ = c.Error(err)
			return
		}
		req.ClearImage = clearInfo
	}

	// 上传有雾图（可选，可多个）
	hazeLevels := c.PostFormArray("hazeLevels")
	if form, err := c.MultipartForm(); err == nil {
		hazyHeaders := form.File["hazyImages"]
		for i, hazyHeader := range hazyHeaders {
			hazyInfo, err := api.uploadFormFile(ctx, hazyHeader, baseURL)
			if err != nil {
				_ = c.Error(err)
				return
			}
			if i < len(hazeLevels) {
				hazyInfo.HazeLevel = hazeLevels[i]
			}
			req.HazyImages = append(req.HazyImages, hazyInfo)
		}
	}

	result, err := api.operationService.CreateDatasetItemWithImages(ctx, req)
	if err != nil {
		logger.Error("创建数据项失败", zap.Error(err))
		_ = c.Error(err)
		return
	}

	common.OkWithData(result, c)
}

// uploadFormFile 上传单个表单文件，返回 ImageUploadInfo
func (api *SysDatasetItemApi) uploadFormFile(ctx context.Context, fileHeader *multipart.FileHeader, baseURL string) (datasetservice.ImageUploadInfo, error) {
	file, err := fileHeader.Open()
	if err != nil {
		return datasetservice.ImageUploadInfo{}, common.NewBizError(common.PARAM_ERROR, "无法读取文件")
	}
	defer file.Close()

	md5Hash, reader, err := fileservice.ComputeMD5(file)
	if err != nil {
		return datasetservice.ImageUploadInfo{}, common.WrapBizError(common.SYSTEM_RESOURCE_ACCESS_ERR, "计算文件MD5失败", err)
	}

	sysFile, err := api.fileService.UploadFile(ctx, fileHeader, reader, md5Hash, baseURL)
	if err != nil {
		return datasetservice.ImageUploadInfo{}, err
	}

	return datasetservice.ImageUploadInfo{
		Name: fileHeader.Filename,
		Path: sysFile.Path,
		URL:  utils.StringVal(sysFile.URL),
		Size: fileHeader.Size,
		MD5:  md5Hash,
	}, nil
}

// BatchCreateDatasetItemsWithImages 批量创建数据项并上传图片
// @Summary 批量创建数据项并上传图片
// @Tags 数据项接口
// @Accept multipart/form-data
// @Produce application/json
// @Param datasetId formData int true "数据集ID"
// @Param sceneType formData string false "场景类型"
// @Param files formData file true "图片文件（可多个，按文件名前缀分组配对）"
// @Success 200 {object} common.Response{data=batchUploadResult}
// @Router /api/v1/dataset-items/batch [post]
func (api *SysDatasetItemApi) BatchCreateDatasetItemsWithImages(c *gin.Context) {
	ctx := c.Request.Context()

	datasetIdStr := c.PostForm("datasetId")
	datasetId, err := strconv.ParseInt(datasetIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据集ID格式不正确"))
		return
	}

	sceneType := c.PostForm("sceneType")

	form, err := c.MultipartForm()
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "文件上传失败"))
		return
	}

	fileHeaders := form.File["files"]
	if len(fileHeaders) == 0 {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "文件列表不能为空"))
		return
	}

	baseURL := fmt.Sprintf("http://%s/api/v1/files/download", c.Request.Host)

	// 按文件名前缀（第一个 _ 之前的部分）分组
	groups := make(map[string][]*multipart.FileHeader)
	groupOrder := make([]string, 0)
	for _, fh := range fileHeaders {
		prefix := getFileGroupPrefix(fh.Filename)
		if _, exists := groups[prefix]; !exists {
			groupOrder = append(groupOrder, prefix)
		}
		groups[prefix] = append(groups[prefix], fh)
	}

	total := len(fileHeaders)
	succeeded := 0
	successItems := make([]batchUploadSuccessItem, 0)
	failedItems := make([]batchUploadFailedItem, 0)

	for _, prefix := range groupOrder {
		files := groups[prefix]

		var clearInfo datasetservice.ImageUploadInfo
		var hazyInfos []datasetservice.ImageUploadInfo
		var uploadedFileNames []string
		uploadedCount := 0

		for _, fh := range files {
			info, err := api.uploadFormFile(ctx, fh, baseURL)
			if err != nil {
				failedItems = append(failedItems, batchUploadFailedItem{
					FileName: fh.Filename,
					Reason:   err.Error(),
				})
				continue
			}
			uploadedCount++
			uploadedFileNames = append(uploadedFileNames, fh.Filename)

			lowerName := strings.ToLower(fh.Filename)
			if strings.Contains(lowerName, "hazy") {
				hazyInfos = append(hazyInfos, info)
			} else {
				clearInfo = info
			}
		}

		if uploadedCount == 0 {
			continue
		}

		req := datasetservice.CreateDatasetItemWithImagesRequest{
			DatasetID:  datasetId,
			ItemName:   prefix,
			SceneType:  sceneType,
			ClearImage: clearInfo,
			HazyImages: hazyInfos,
			Options: datasetservice.CreateItemOptions{
				AsyncThumbnail: true,
			},
		}

		itemVO, err := api.operationService.CreateDatasetItemWithImages(ctx, req)
		if err != nil {
			for _, fname := range uploadedFileNames {
				failedItems = append(failedItems, batchUploadFailedItem{
					FileName: fname,
					Reason:   err.Error(),
				})
			}
			continue
		}

		succeeded += uploadedCount
		successItems = append(successItems, batchUploadSuccessItem{
			ID:        itemVO.ID,
			Name:      itemVO.Name,
			FileCount: uploadedCount,
		})
	}

	result := batchUploadResult{
		Total:        total,
		Succeeded:    succeeded,
		Failed:       len(failedItems),
		SuccessItems: successItems,
		FailedItems:  failedItems,
	}

	common.OkWithDetailed(result, "批量上传完成", c)
}

// getFileGroupPrefix 提取文件名分组前缀（第一个 _ 之前的部分，若无 _ 则去掉扩展名）
func getFileGroupPrefix(filename string) string {
	idx := strings.Index(filename, "_")
	if idx == -1 {
		idx = strings.LastIndex(filename, ".")
		if idx == -1 {
			return filename
		}
		return filename[:idx]
	}
	return filename[:idx]
}

// batchUploadResult 批量上传结果
type batchUploadResult struct {
	Total        int                      `json:"total"`
	Succeeded    int                      `json:"succeeded"`
	Failed       int                      `json:"failed"`
	SuccessItems []batchUploadSuccessItem `json:"successItems,omitempty"`
	FailedItems  []batchUploadFailedItem  `json:"failedItems,omitempty"`
}

// batchUploadSuccessItem 批量上传成功项
type batchUploadSuccessItem struct {
	ID        int64  `json:"id"`
	Name      string `json:"name"`
	FileCount int    `json:"fileCount"`
}

// batchUploadFailedItem 批量上传失败项
type batchUploadFailedItem struct {
	FileName string `json:"fileName"`
	Reason   string `json:"reason"`
}

// updateDatasetItemRequest 修改数据项请求
type updateDatasetItemRequest struct {
	Name      string `json:"name"`
	SceneType string `json:"sceneType"`
}

// UpdateDatasetItem 修改数据项
// @Summary 修改数据项
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据项ID"
// @Param request body updateDatasetItemRequest true "修改请求"
// @Success 200 {object} common.Response{data=vo.ImageItemVO}
// @Router /api/v1/dataset-items/{id} [put]
func (api *SysDatasetItemApi) UpdateDatasetItem(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据项ID格式不正确"))
		return
	}

	var req updateDatasetItemRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	itemVO, err := api.datasetItemService.UpdateDatasetItem(id, req.Name)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(itemVO, "修改数据项成功", c)
}

// DeleteDatasetItem 删除数据项（级联删除关联文件）
// @Summary 删除数据项
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据项ID"
// @Success 200 {object} common.Response
// @Router /api/v1/dataset-items/{id} [delete]
func (api *SysDatasetItemApi) DeleteDatasetItem(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据项ID格式不正确"))
		return
	}

	err = api.operationService.DeleteDatasetItemCascade(c.Request.Context(), id)
	if err != nil {
		logger.Error("删除数据项失败", zap.Error(err))
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("删除成功", c)
}

// BatchDeleteDatasetItems 批量删除数据项
// @Summary 批量删除数据项
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param request body bo.BatchDeleteForm true "批量删除请求"
// @Success 200 {object} common.Response{data=vo.BatchOperationResultVO}
// @Router /api/v1/dataset-items/batch [delete]
func (api *SysDatasetItemApi) BatchDeleteDatasetItems(c *gin.Context) {
	var req bo.BatchDeleteForm
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	successIds := make([]int64, 0, len(req.IDs))
	failureDetails := make([]batchFailureDetail, 0)
	successCount := 0
	failedCount := 0

	for _, id := range req.IDs {
		if err := api.operationService.DeleteDatasetItemCascade(c.Request.Context(), id); err != nil {
			failedCount++
			failureDetails = append(failureDetails, batchFailureDetail{
				Identifier: strconv.FormatInt(id, 10),
				Reason:     err.Error(),
			})
			logger.Warn("批量删除数据项失败", zap.Int64("itemID", id), zap.Error(err))
			continue
		}
		successCount++
		successIds = append(successIds, id)
	}

	result := batchOperationResult{
		SuccessCount:  successCount,
		FailedCount:   failedCount,
		Message:       "批量删除完成",
		SuccessIds:    successIds,
		FailureDetails: failureDetails,
	}

	common.OkWithDetailed(result, "批量删除成功", c)
}

// batchOperationResult 批量操作结果
type batchOperationResult struct {
	SuccessCount  int                 `json:"successCount"`
	FailedCount   int                 `json:"failedCount"`
	Message       string              `json:"message"`
	SuccessIds    []int64             `json:"successIds,omitempty"`
	FailureDetails []batchFailureDetail `json:"failureDetails,omitempty"`
}

// batchFailureDetail 批量操作失败详情
type batchFailureDetail struct {
	Identifier string `json:"identifier,omitempty"`
	Reason     string `json:"reason"`
}
