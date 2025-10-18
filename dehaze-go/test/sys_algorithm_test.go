package test

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/assert"
)

func TestAlgorithmService_GetAlgorithmList(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	algorithmService := &service.AlgorithmService{}

	// 测试用例1: 正常获取算法列表
	t.Run("NormalGetAlgorithmList", func(t *testing.T) {
		// 准备测试数据
		testAlgorithm := model.SysAlgorithm{
			ParentID:   0,
			Type:       "test_type",
			Name:       "test_algorithm_list",
			Path:       "/test/path",
			Status:     1,
			CreateBy:   1,
			UpdateBy:   1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testAlgorithm.Name).Delete(&model.SysAlgorithm{})

		// 插入测试算法
		result := global.DB.Create(&testAlgorithm)
		assert.NoError(t, result.Error)

		// 执行查询
		queryParams := query.AlgorithmQuery{}
		algorithms, err := algorithmService.GetAlgorithmList(queryParams)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, algorithms)
		assert.NotEmpty(t, algorithms)

		// 清理测试数据
		global.DB.Where("name = ?", testAlgorithm.Name).Delete(&model.SysAlgorithm{})
	})

	// 测试用例2: 带关键字查询
	t.Run("GetAlgorithmListWithKeywords", func(t *testing.T) {
		// 准备测试数据
		testAlgorithm1 := model.SysAlgorithm{
			ParentID:   0,
			Type:       "test_type",
			Name:       "test_algorithm_keyword_1",
			Path:       "/test/path",
			Status:     1,
			CreateBy:   1,
			UpdateBy:   1,
		}

		testAlgorithm2 := model.SysAlgorithm{
			ParentID:   0,
			Type:       "test_type",
			Name:       "another_algorithm",
			Path:       "/test/path",
			Status:     1,
			CreateBy:   1,
			UpdateBy:   1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name IN ?", []string{testAlgorithm1.Name, testAlgorithm2.Name}).Delete(&model.SysAlgorithm{})

		// 插入测试算法
		result1 := global.DB.Create(&testAlgorithm1)
		assert.NoError(t, result1.Error)
		result2 := global.DB.Create(&testAlgorithm2)
		assert.NoError(t, result2.Error)

		// 执行查询
		queryParams := query.AlgorithmQuery{
			Keywords: "test_algorithm_keyword",
		}
		algorithms, err := algorithmService.GetAlgorithmList(queryParams)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, algorithms)
		assert.Equal(t, 1, len(algorithms))
		assert.Equal(t, testAlgorithm1.Name, algorithms[0].Name)

		// 清理测试数据
		global.DB.Where("name IN ?", []string{testAlgorithm1.Name, testAlgorithm2.Name}).Delete(&model.SysAlgorithm{})
	})
}

func TestAlgorithmService_AddAlgorithm(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	algorithmService := &service.AlgorithmService{}

	// 测试用例1: 正常添加算法
	t.Run("NormalAddAlgorithm", func(t *testing.T) {
		algorithmForm := bo.AlgorithmFormBO{
			ParentID:   0,
			Type:       "test_type",
			Name:       "test_add_algorithm",
			Path:       "/test/path",
			Status:     1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", algorithmForm.Name).Delete(&model.SysAlgorithm{})

		// 执行添加
		err := algorithmService.AddAlgorithm(algorithmForm)
		
		// 验证结果
		assert.NoError(t, err)

		// 验证数据是否插入
		var algorithm model.SysAlgorithm
		result := global.DB.Where("name = ?", algorithmForm.Name).First(&algorithm)
		assert.NoError(t, result.Error)
		assert.Equal(t, algorithmForm.Name, algorithm.Name)
		assert.Equal(t, algorithmForm.Type, algorithm.Type)
		assert.Equal(t, algorithmForm.Path, algorithm.Path)

		// 清理测试数据
		global.DB.Where("name = ?", algorithmForm.Name).Delete(&model.SysAlgorithm{})
	})

	// 测试用例2: 添加带父节点的算法
	t.Run("AddAlgorithmWithParent", func(t *testing.T) {
		// 先创建父算法
		parentAlgorithm := model.SysAlgorithm{
			ParentID:   0,
			Type:       "test_type",
			Name:       "test_parent_algorithm",
			Path:       "/test/parent/path",
			Status:     1,
			CreateBy:   1,
			UpdateBy:   1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", parentAlgorithm.Name).Delete(&model.SysAlgorithm{})

		// 插入父算法
		result := global.DB.Create(&parentAlgorithm)
		assert.NoError(t, result.Error)

		// 创建子算法
		algorithmForm := bo.AlgorithmFormBO{
			ParentID:   parentAlgorithm.ID,
			Type:       "test_type",
			Name:       "test_child_algorithm",
			Path:       "/test/child/path",
			Status:     1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", algorithmForm.Name).Delete(&model.SysAlgorithm{})

		// 执行添加
		err := algorithmService.AddAlgorithm(algorithmForm)
		
		// 验证结果
		assert.NoError(t, err)

		// 验证数据是否插入
		var algorithm model.SysAlgorithm
		result = global.DB.Where("name = ?", algorithmForm.Name).First(&algorithm)
		assert.NoError(t, result.Error)
		assert.Equal(t, algorithmForm.Name, algorithm.Name)
		assert.Equal(t, algorithmForm.ParentID, algorithm.ParentID)

		// 清理测试数据
		global.DB.Where("name IN ?", []string{parentAlgorithm.Name, algorithmForm.Name}).Delete(&model.SysAlgorithm{})
	})
}

func TestAlgorithmService_UpdateAlgorithm(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	algorithmService := &service.AlgorithmService{}

	// 测试用例1: 正常更新算法
	t.Run("NormalUpdateAlgorithm", func(t *testing.T) {
		// 准备测试数据
		testAlgorithm := model.SysAlgorithm{
			ParentID:   0,
			Type:       "test_type",
			Name:       "test_update_algorithm",
			Path:       "/test/path",
			Status:     1,
			CreateBy:   1,
			UpdateBy:   1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testAlgorithm.Name).Delete(&model.SysAlgorithm{})

		// 插入测试算法
		result := global.DB.Create(&testAlgorithm)
		assert.NoError(t, result.Error)

		// 更新算法
		algorithmForm := bo.AlgorithmFormBO{
			ParentID:   0,
			Type:       "updated_type",
			Name:       "updated_algorithm_name",
			Path:       "/updated/path",
			Status:     1,
		}

		err := algorithmService.UpdateAlgorithm(testAlgorithm.ID, algorithmForm)
		
		// 验证结果
		assert.NoError(t, err)

		// 验证数据是否更新
		var algorithm model.SysAlgorithm
		result = global.DB.Where("id = ?", testAlgorithm.ID).First(&algorithm)
		assert.NoError(t, result.Error)
		assert.Equal(t, algorithmForm.Name, algorithm.Name)
		assert.Equal(t, algorithmForm.Type, algorithm.Type)
		assert.Equal(t, algorithmForm.Path, algorithm.Path)

		// 清理测试数据
		global.DB.Where("id = ?", testAlgorithm.ID).Delete(&model.SysAlgorithm{})
	})
}

func TestAlgorithmService_DeleteAlgorithms(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	algorithmService := &service.AlgorithmService{}

	// 测试用例1: 正常删除算法
	t.Run("NormalDeleteAlgorithms", func(t *testing.T) {
		// 准备测试数据
		testAlgorithm := model.SysAlgorithm{
			ParentID:   0,
			Type:       "test_type",
			Name:       "test_delete_algorithm",
			Path:       "/test/path",
			Status:     1,
			CreateBy:   1,
			UpdateBy:   1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testAlgorithm.Name).Delete(&model.SysAlgorithm{})

		// 插入测试算法
		result := global.DB.Create(&testAlgorithm)
		assert.NoError(t, result.Error)

		// 执行删除
		err := algorithmService.DeleteAlgorithms([]int64{testAlgorithm.ID})
		
		// 验证结果
		assert.NoError(t, err)

		// 验证数据是否删除
		var algorithm model.SysAlgorithm
		result = global.DB.Where("id = ?", testAlgorithm.ID).First(&algorithm)
		assert.Error(t, result.Error) // 应该找不到记录
	})

	// 测试用例2: 删除有子算法的算法（应该失败）
	t.Run("DeleteAlgorithmWithChildren", func(t *testing.T) {
		// 准备测试数据
		parentAlgorithm := model.SysAlgorithm{
			ParentID:   0,
			Type:       "test_type",
			Name:       "test_parent_algorithm_for_delete",
			Path:       "/test/parent/path",
			Status:     1,
			CreateBy:   1,
			UpdateBy:   1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", parentAlgorithm.Name).Delete(&model.SysAlgorithm{})

		// 插入父算法
		result := global.DB.Create(&parentAlgorithm)
		assert.NoError(t, result.Error)

		// 创建子算法
		childAlgorithm := model.SysAlgorithm{
			ParentID:   parentAlgorithm.ID,
			Type:       "test_type",
			Name:       "test_child_algorithm_for_delete",
			Path:       "/test/child/path",
			Status:     1,
			CreateBy:   1,
			UpdateBy:   1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", childAlgorithm.Name).Delete(&model.SysAlgorithm{})

		// 插入子算法
		result = global.DB.Create(&childAlgorithm)
		assert.NoError(t, result.Error)

		// 执行删除（应该失败）
		err := algorithmService.DeleteAlgorithms([]int64{parentAlgorithm.ID})
		
		// 验证结果
		assert.Error(t, err)
		assert.Equal(t, "存在子算法，无法删除", err.Error())

		// 清理测试数据
		global.DB.Where("name IN ?", []string{parentAlgorithm.Name, childAlgorithm.Name}).Delete(&model.SysAlgorithm{})
	})
}