package database

import (
	"reflect"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// softDeleteClauseKey 用于标记已添加软删除条件，避免重复添加
const softDeleteClauseKey = "soft_delete_enabled"

// RegisterSoftDeleteCallback 注册逻辑删除自动过滤回调
// 对所有包含 Deleted 字段的模型，在查询时自动追加 deleted = 0 条件，
// 防止遗漏手动过滤导致已删除数据泄露。
// 使用 clause.CurrentTable 确保在 JOIN 查询中正确引用主表别名。
func RegisterSoftDeleteCallback(db *gorm.DB) *gorm.DB {
	if db == nil {
		return db
	}

	// 查询回调：自动追加 deleted = 0
	db.Callback().Query().Before("gorm:query").Register("soft_delete:query", softDeleteQueryCallback)

	// 行查询回调（First/Take/Last）
	db.Callback().Row().Before("gorm:row").Register("soft_delete:row", softDeleteQueryCallback)

	return db
}

// softDeleteQueryCallback 逻辑删除查询回调
// 检查目标模型是否包含 Deleted 字段，若包含则自动追加 deleted = 0 条件。
// 通过 softDeleteClauseKey 标记避免重复添加。
// 支持 Unscoped() 跳过过滤（与 GORM 内置软删除行为一致）。
func softDeleteQueryCallback(db *gorm.DB) {
	stmt := db.Statement
	if stmt == nil || stmt.Unscoped {
		return
	}

	// 已添加过则跳过
	if _, ok := stmt.Clauses[softDeleteClauseKey]; ok {
		return
	}

	// 检查模型或目标结构体是否包含 Deleted 字段
	if !hasDeletedField(stmt) {
		return
	}

	// 使用 clause.CurrentTable 让 GORM 在生成 SQL 时自动解析为当前表名/别名
	stmt.AddClause(clause.Where{
		Exprs: []clause.Expression{
			clause.Eq{
				Column: clause.Column{Table: clause.CurrentTable, Name: "deleted"},
				Value:  0,
			},
		},
	})

	// 标记已添加，防止重复
	stmt.Clauses[softDeleteClauseKey] = clause.Clause{}
}

// hasDeletedField 检查 Statement 的 Model 或 Dest 是否包含名为 Deleted 的字段
func hasDeletedField(stmt *gorm.Statement) bool {
	// 优先检查 Model
	if stmt.Model != nil {
		if checkDeletedField(stmt.Model) {
			return true
		}
	}

	// 再检查 Dest（查询目标）
	if stmt.Dest != nil {
		return checkDeletedField(stmt.Dest)
	}

	return false
}

// checkDeletedField 通过反射检查值是否包含 Deleted 字段
func checkDeletedField(v interface{}) bool {
	val := reflect.ValueOf(v)
	if val.Kind() == reflect.Ptr {
		if val.IsNil() {
			return false
		}
		val = val.Elem()
	}

	// 处理切片（取元素类型）
	if val.Kind() == reflect.Slice || val.Kind() == reflect.Array {
		elemType := val.Type().Elem()
		if elemType.Kind() == reflect.Ptr {
			elemType = elemType.Elem()
		}
		if elemType.Kind() == reflect.Struct {
			_, found := elemType.FieldByName("Deleted")
			return found
		}
		return false
	}

	if val.Kind() != reflect.Struct {
		return false
	}

	// 直接查找 Deleted 字段（包括内嵌结构体中的字段）
	_, found := val.Type().FieldByName("Deleted")
	return found
}
