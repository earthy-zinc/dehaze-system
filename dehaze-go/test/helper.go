package test

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/stretchr/testify/suite"
	"gorm.io/gorm"
)

// BaseTestSuite 提供事务隔离的测试套件基类
// 这确保了测试之间的独立性，无需手动清理数据
type BaseTestSuite struct {
	suite.Suite
	DB *gorm.DB
}

// SetupTest 在每个测试方法运行前执行
func (s *BaseTestSuite) SetupTest() {
	// 确保数据库连接有效
	if global.DB == nil {
		s.T().Fatal("数据库连接未初始化")
	}
	if s.DB == nil {
		s.DB = global.DB
	}
	initialize.Migrate()
}

// TearDownTest 在每个测试方法运行后执行
func (s *BaseTestSuite) TearDownTest() {
	// 删除测试数据
	s.DB.Exec("DROP DATABASE IF EXISTS dehaze_test")
	s.DB.Exec("CREATE DATABASE IF NOT EXISTS dehaze_test")
	s.DB.Exec("USE dehaze_test")
}

// GetDB 获取当前事务的数据库连接
// 在测试代码中可以使用这个方法获取数据库连接
func (s *BaseTestSuite) GetDB() *gorm.DB {
	return s.DB
}

// CreateTestData 创建测试数据的辅助函数
func (s *BaseTestSuite) CreateTestData(model any) error {
	return s.GetDB().Create(model).Error
}

// UpdateTestData 更新测试数据的辅助函数
func (s *BaseTestSuite) UpdateTestData(model any) error {
	return s.GetDB().Save(model).Error
}

// DeleteTestData 删除测试数据的辅助函数
func (s *BaseTestSuite) DeleteTestData(model any, conditions ...any) error {
	return s.GetDB().Delete(model, conditions...).Error
}

// FindTestData 查找测试数据的辅助函数
func (s *BaseTestSuite) FindTestData(dest any, conditions ...any) error {
	return s.GetDB().Find(dest, conditions...).Error
}

// AssertNoError 断言没有错误的辅助函数
func (s *BaseTestSuite) AssertNoError(err error, msgAndArgs ...any) {
	s.Require().NoError(err, msgAndArgs...)
}

// AssertError 断言有错误的辅助函数
func (s *BaseTestSuite) AssertError(err error, msgAndArgs ...any) {
	s.Require().Error(err, msgAndArgs...)
}

// AssertEqual 断言相等的辅助函数
func (s *BaseTestSuite) AssertEqual(expected, actual any, msgAndArgs ...any) {
	s.Require().Equal(expected, actual, msgAndArgs...)
}

// AssertNotNil 断言不为nil的辅助函数
func (s *BaseTestSuite) AssertNotNil(object any, msgAndArgs ...any) {
	s.Require().NotNil(object, msgAndArgs...)
}

// AssertNil 断言为nil的辅助函数
func (s *BaseTestSuite) AssertNil(object any, msgAndArgs ...any) {
	s.Require().Nil(object, msgAndArgs...)
}

// RunTransactionTestSuite 运行测试套件的辅助函数
// 使用示例：
//
//	func TestMyServiceSuite(t *testing.T) {
//	    suite.Run(t, &MyServiceTestSuite{})
//	}
func RunTransactionTestSuite(t *testing.T, testSuite suite.TestingSuite) {
	suite.Run(t, testSuite)
}
