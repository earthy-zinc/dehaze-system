package test

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/stretchr/testify/suite"
	"gorm.io/gorm"
)

// TransactionTestSuite 提供事务隔离的测试套件基类
// 每个测试方法都会在独立的事务中运行，测试结束后自动回滚
// 这确保了测试之间的独立性，无需手动清理数据
type TransactionTestSuite struct {
	suite.Suite
	DB *gorm.DB // 原始数据库连接
	tx *gorm.DB // 当前测试的事务连接
}

// SetupTest 在每个测试方法运行前执行
// 开启一个新的数据库事务
func (s *TransactionTestSuite) SetupTest() {
	// 确保数据库连接有效
	if s.DB == nil {
		s.T().Fatal("数据库连接未初始化")
	}
	
	s.tx = s.DB.Begin()
	if s.tx.Error != nil {
		s.T().Fatal("开启事务失败: ", s.tx.Error)
	}

	// 临时替换全局数据库连接为事务连接
	// 这样测试代码中使用的 global.DB 就是事务连接了
	global.DB = s.tx
}

// TearDownTest 在每个测试方法运行后执行
// 回滚事务，确保测试数据不会保留在数据库中
func (s *TransactionTestSuite) TearDownTest() {
	if s.tx != nil {
		s.tx.Rollback()
		// 恢复全局数据库连接
		global.DB = s.DB
	}
}

// GetDB 获取当前事务的数据库连接
// 在测试代码中可以使用这个方法获取数据库连接
func (s *TransactionTestSuite) GetDB() *gorm.DB {
	if s.tx != nil {
		return s.tx
	}
	return s.DB
}

// RunInTransaction 在事务中运行一个函数
// 如果函数返回错误，事务会自动回滚
// 否则事务会提交（但在测试套件中，外层事务最终仍会回滚）
func (s *TransactionTestSuite) RunInTransaction(fn func(tx *gorm.DB) error) error {
	tx := s.GetDB().Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
			panic(r)
		}
	}()

	if err := fn(tx); err != nil {
		tx.Rollback()
		return err
	}

	return tx.Commit().Error
}

// CreateTestData 创建测试数据的辅助函数
// 自动使用当前事务，测试结束后会自动清理
func (s *TransactionTestSuite) CreateTestData(model interface{}) error {
	return s.GetDB().Create(model).Error
}

// UpdateTestData 更新测试数据的辅助函数
func (s *TransactionTestSuite) UpdateTestData(model interface{}) error {
	return s.GetDB().Save(model).Error
}

// DeleteTestData 删除测试数据的辅助函数
func (s *TransactionTestSuite) DeleteTestData(model interface{}, conditions ...interface{}) error {
	return s.GetDB().Delete(model, conditions...).Error
}

// FindTestData 查找测试数据的辅助函数
func (s *TransactionTestSuite) FindTestData(dest interface{}, conditions ...interface{}) error {
	return s.GetDB().Find(dest, conditions...).Error
}

// AssertNoError 断言没有错误的辅助函数
func (s *TransactionTestSuite) AssertNoError(err error, msgAndArgs ...interface{}) {
	s.Require().NoError(err, msgAndArgs...)
}

// AssertError 断言有错误的辅助函数
func (s *TransactionTestSuite) AssertError(err error, msgAndArgs ...interface{}) {
	s.Require().Error(err, msgAndArgs...)
}

// AssertEqual 断言相等的辅助函数
func (s *TransactionTestSuite) AssertEqual(expected, actual interface{}, msgAndArgs ...interface{}) {
	s.Require().Equal(expected, actual, msgAndArgs...)
}

// AssertNotNil 断言不为nil的辅助函数
func (s *TransactionTestSuite) AssertNotNil(object interface{}, msgAndArgs ...interface{}) {
	s.Require().NotNil(object, msgAndArgs...)
}

// AssertNil 断言为nil的辅助函数
func (s *TransactionTestSuite) AssertNil(object interface{}, msgAndArgs ...interface{}) {
	s.Require().Nil(object, msgAndArgs...)
}

// RunTransactionTestSuite 运行事务测试套件的辅助函数
// 使用示例：
//
//	func TestMyServiceSuite(t *testing.T) {
//	    suite.Run(t, &MyServiceTestSuite{})
//	}
func RunTransactionTestSuite(t *testing.T, testSuite suite.TestingSuite) {
	suite.Run(t, testSuite)
}