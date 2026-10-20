// Package aidomain 实现 AI 对话域 A 类端点的原生数据访问（共享 MySQL/Redis）。
// B 类（推理/流式）与 AI 基础模块（供应商/模型/Skill/MCP）不在本包范围。
package aidomain

import (
	"strings"

	"gorm.io/gorm"
)

// escapeLike 转义 LIKE 通配符，避免用户输入中的 % / _ 被当作通配符。
func escapeLike(s string) string {
	return strings.NewReplacer("\\", "\\\\", "%", "\\%", "_", "\\_").Replace(s)
}

// paginate 归一化分页参数（与 python BasePageQuery 的 ge=1 语义一致）。
func paginate(page, size int) (offset, limit int) {
	if page <= 0 {
		page = 1
	}
	if size <= 0 {
		size = 10
	}
	return (page - 1) * size, size
}

// countAndFind 先计数再分页取列表（Count 在 Session 克隆上执行，避免污染后续 Find）。
func countAndFind[T any](db *gorm.DB, page, size int) ([]T, int64, error) {
	var total int64
	if err := db.Session(&gorm.Session{}).Count(&total).Error; err != nil {
		return nil, 0, err
	}
	offset, limit := paginate(page, size)
	var items []T
	if err := db.Offset(offset).Limit(limit).Find(&items).Error; err != nil {
		return nil, 0, err
	}
	return items, total, nil
}
