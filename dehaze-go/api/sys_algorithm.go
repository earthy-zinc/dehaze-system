package api

import (
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/gin-gonic/gin"
)

type AlgorithmApi struct{}

// GetList 获取算法树形表格
func (algorithmApi *AlgorithmApi) GetList(c *gin.Context) {
	var queryParams query.AlgorithmQuery
	_ = c.ShouldBindQuery(&queryParams)

	algorithms, err := algorithmService.GetAlgorithmList(queryParams)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithData(algorithms, c)
}

// GetOptions 获取模型下拉选项列表
func (algorithmApi *AlgorithmApi) GetOptions(c *gin.Context) {
	options, err := algorithmService.GetAlgorithmOptions()
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithData(options, c)
}

// GetById 根据ID获取算法信息
func (algorithmApi *AlgorithmApi) GetById(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		common.FailWithMessage("参数错误", c)
		return
	}

	algorithm, err := algorithmService.GetAlgorithmById(id)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithData(algorithm, c)
}

// Add 新增算法
func (algorithmApi *AlgorithmApi) Add(c *gin.Context) {
	var algorithmForm bo.AlgorithmFormBO
	_ = c.ShouldBindJSON(&algorithmForm)

	if algorithmForm.ParentID < 0 {
		common.FailWithMessage("父算法ID不能为负数", c)
		return
	}

	err := algorithmService.AddAlgorithm(algorithmForm)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithMessage("添加成功", c)
}

// Update 修改算法
func (algorithmApi *AlgorithmApi) Update(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		common.FailWithMessage("参数错误", c)
		return
	}

	var algorithmForm bo.AlgorithmFormBO
	_ = c.ShouldBindJSON(&algorithmForm)
	algorithmForm.ID = id

	if algorithmForm.ParentID < 0 {
		common.FailWithMessage("父算法ID不能为负数", c)
		return
	}

	err = algorithmService.UpdateAlgorithm(id, algorithmForm)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithMessage("修改成功", c)
}

// Delete 删除算法
func (algorithmApi *AlgorithmApi) Delete(c *gin.Context) {
	idsStr := c.Param("ids")
	if idsStr == "" {
		common.FailWithMessage("参数错误", c)
		return
	}

	idsSlice := []int64{}
	// 解析ID列表
	for _, idStr := range strings.Split(idsStr, ",") {
		if id, err := strconv.ParseInt(idStr, 10, 64); err == nil {
			idsSlice = append(idsSlice, id)
		}
	}

	if len(idsSlice) == 0 {
		common.FailWithMessage("参数错误", c)
		return
	}

	err := algorithmService.DeleteAlgorithms(idsSlice)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithMessage("删除成功", c)
}
