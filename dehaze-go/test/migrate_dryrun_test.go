package test

import (
	"fmt"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"

	"gorm.io/gorm"
)

// TestMigrateDryRun 使用DryRun模式查看AutoMigrate会生成什么SQL
func TestMigrateDryRun(t *testing.T) {
	// 初始化配置和数据库
	initialize.Viper()
	initialize.Gorm()

	if global.DB == nil {
		t.Fatal("数据库连接失败")
	}

	// 使用DryRun模式,只打印SQL不执行
	session := global.DB.Session(&gorm.Session{DryRun: true})

	// 定义所有需要迁移的表
	tables := []interface{}{
		&model.SysUser{},
		&model.SysRole{},
		&model.SysMenu{},
		&model.SysDept{},
		&model.SysDict{},
		&model.SysDictType{},
		&model.SysUserRole{},
		&model.SysRoleMenu{},
		&model.SysDataset{},
		&model.SysAlgorithm{},
		&model.SysFile{},
		&model.SysDatasetItem{},
		&model.SysItemFile{},
		&model.SysPredLog{},
		&model.SysEvalLog{},
		&model.SysWpxFile{},
		&model.SysOperationRecord{},
	}

	fmt.Println("=== GORM AutoMigrate DryRun 模式 - 查看将要执行的SQL ===")

	// 对每个表单独执行AutoMigrate以便查看具体SQL
	for _, table := range tables {
		fmt.Printf("--- 处理表: %T ---\n", table)
		session.AutoMigrate(table)
	}

	fmt.Println("=== 使用Migrator检查表是否存在 ===")

	// 检查每个表是否已存在
	for _, table := range tables {
		migrator := global.DB.Migrator()
		tableName := ""

		// 获取表名
		stmt := &gorm.Statement{DB: global.DB}
		stmt.Parse(table)
		tableName = stmt.Table

		hasTable := migrator.HasTable(table)
		fmt.Printf("表 %-30s 是否存在: %v\n", tableName, hasTable)

		if hasTable {
			// 检查表的列信息
			columns, err := migrator.ColumnTypes(table)
			if err != nil {
				fmt.Printf("  获取列信息失败: %v\n", err)
			} else {
				fmt.Printf("  当前列数: %d\n", len(columns))
			}
		}
	}

	fmt.Println("\n=== 检查完成 ===")
	fmt.Println("提示: DryRun模式只显示检查结果,不会实际执行任何SQL")
	fmt.Println("建议: 查看日志中的SQL语句,确认是否会修改现有表")
}
