package main

import (
	"fmt"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

func main() {
	dsn := "root:12345678@tcp(127.0.0.1:3306)/dehaze?charset=utf8mb4&parseTime=True&loc=Local"
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Check current columns
	var cols []struct {
		Field string `gorm:"column:Field"`
		Type  string `gorm:"column:Type"`
	}
	db.Raw("SHOW COLUMNS FROM sys_task").Scan(&cols)
	fmt.Println("Current columns:")
	for _, c := range cols {
		fmt.Printf("  %s (%s)\n", c.Field, c.Type)
	}

	// Rename created_at to create_time if it exists
	var count int64
	db.Raw("SELECT COUNT(*) FROM information_schema.columns WHERE table_schema='dehaze' AND table_name='sys_task' AND column_name='created_at'").Scan(&count)
	if count > 0 {
		fmt.Println("Renaming created_at to create_time...")
		if err := db.Exec("ALTER TABLE sys_task CHANGE COLUMN created_at create_time datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'").Error; err != nil {
			fmt.Println("Rename error:", err)
		} else {
			fmt.Println("Renamed successfully")
		}
	}

	// Add update_time if it doesn't exist
	db.Raw("SELECT COUNT(*) FROM information_schema.columns WHERE table_schema='dehaze' AND table_name='sys_task' AND column_name='update_time'").Scan(&count)
	if count == 0 {
		fmt.Println("Adding update_time column...")
		if err := db.Exec("ALTER TABLE sys_task ADD COLUMN update_time datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'").Error; err != nil {
			fmt.Println("Add column error:", err)
		} else {
			fmt.Println("Added successfully")
		}
	}

	// Verify
	db.Raw("SHOW COLUMNS FROM sys_task").Scan(&cols)
	fmt.Println("\nFinal columns:")
	for _, c := range cols {
		fmt.Printf("  %s (%s)\n", c.Field, c.Type)
	}
}
