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

	fmt.Println("=== GORM AutoMigrate DryRun 模式 - 查看将要执行的SQL ===\n")

	// 对每个表单独执行AutoMigrate以便查看具体SQL
	for _, table := range tables {
		fmt.Printf("--- 处理表: %T ---\n", table)

		// 捕获SQL语句
		result := session.AutoMigrate(table)
		if result.Error != nil {
			fmt.Printf("错误: %v\n\n", result.Error)
		} else {
			// DryRun模式下,SQL会被记录但不执行
			fmt.Printf("✓ 检查完成\n\n")
		}
	}

	fmt.Println("=== 使用Migrator检查表是否存在 ===\n")

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

// TestMigrateWithLogger 使用详细日志模式查看AutoMigrate过程
func TestMigrateWithLogger(t *testing.T) {
	initialize.Viper()
	initialize.Gorm()

	if global.DB == nil {
		t.Fatal("数据库连接失败")
	}

	fmt.Println("=== 开始AutoMigrate检查(带详细日志) ===\n")

	// 使用Debug模式显示详细SQL
	db := global.DB.Debug()

	// 执行AutoMigrate
	err := db.AutoMigrate(
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
	)

	if err != nil {
		t.Fatalf("AutoMigrate失败: %v", err)
	}

	fmt.Println("\n=== AutoMigrate完成 ===")
}

// TestCompareTables 比较Model定义和数据库实际结构
func TestCompareTables(t *testing.T) {
	initialize.Viper()
	initialize.Gorm()

	if global.DB == nil {
		t.Fatal("数据库连接失败")
	}

	tables := []interface{}{
		&model.SysUser{},
		&model.SysRole{},
		&model.SysMenu{},
	}

	fmt.Println("=== 比较Model定义和数据库实际结构 ===\n")

	for _, table := range tables {
		migrator := global.DB.Migrator()
		stmt := &gorm.Statement{DB: global.DB}
		stmt.Parse(table)
		tableName := stmt.Table

		fmt.Printf("表名: %s\n", tableName)

		if !migrator.HasTable(table) {
			fmt.Printf("  ⚠ 表不存在\n\n")
			continue
		}

		// 获取列信息
		columns, err := migrator.ColumnTypes(table)
		if err != nil {
			fmt.Printf("  错误: %v\n\n", err)
			continue
		}

		fmt.Printf("  列数: %d\n", len(columns))
		for _, col := range columns {
			colType, _ := col.ColumnType()
			nullable, _ := col.Nullable()
			fmt.Printf("    - %s: %s (可空: %v)\n", col.Name(), colType, nullable)
		}
		fmt.Println()
	}
}
