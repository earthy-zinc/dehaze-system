package database

import (
	"fmt"
	"regexp"
	"strings"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// DataScope 数据权限范围常量
const (
	DataScopeAll      int8 = 0 // 全部数据
	DataScopeDeptTree int8 = 1 // 部门及子部门数据
	DataScopeDept     int8 = 2 // 本部门数据
	DataScopeSelf     int8 = 3 // 本人数据
)

// DataScopeConfig 数据权限配置
type DataScopeConfig struct {
	// Enabled 是否启用数据权限
	Enabled bool
	// Tables 需要进行数据权限过滤的表配置
	// key: 表名, value: 表配置
	Tables map[string]TableScopeConfig
	// DefaultScopeField 默认的创建人字段名（用于"本人数据"场景）
	DefaultScopeField string
	// DefaultDeptField 默认的部门字段名
	DefaultDeptField string
}

// TableScopeConfig 表级数据权限配置
type TableScopeConfig struct {
	// Enabled 是否启用（默认 true）
	Enabled bool
	// ScopeField 创建人字段名（用于"本人数据"场景），为空则使用 DataScopeConfig.DefaultScopeField
	ScopeField string
	// DeptField 部门字段名（用于"部门数据"场景），为空则使用 DataScopeConfig.DefaultDeptField
	DeptField string
	// TreePathField 部门树路径字段名（用于"部门及子部门"场景），为空则不支持树查询
	TreePathField string
	// Alias 表别名（用于 JOIN 场景）
	Alias string
}

// DataScopePlugin 数据权限插件
type DataScopePlugin struct {
	config DataScopeConfig
}

// NewDataScopePlugin 创建数据权限插件
func NewDataScopePlugin(config DataScopeConfig) *DataScopePlugin {
	// 设置默认值
	if config.DefaultScopeField == "" {
		config.DefaultScopeField = "create_by"
	}
	if config.DefaultDeptField == "" {
		config.DefaultDeptField = "dept_id"
	}
	if config.Tables == nil {
		config.Tables = make(map[string]TableScopeConfig)
	}

	return &DataScopePlugin{config: config}
}

// Name 实现 gorm.Plugin 接口
func (p *DataScopePlugin) Name() string {
	return "data_scope"
}

// Initialize 实现 gorm.Plugin 接口
func (p *DataScopePlugin) Initialize(db *gorm.DB) error {
	// 注册查询回调
	if err := db.Callback().Query().Before("gorm:query").Register("data_scope:query", p.dataScopeCallback); err != nil {
		return fmt.Errorf("注册数据权限查询回调失败: %w", err)
	}

	// 注册行查询回调（用于 First/Take/Last）
	if err := db.Callback().Row().Before("gorm:row").Register("data_scope:row", p.dataScopeCallback); err != nil {
		return fmt.Errorf("注册数据权限行查询回调失败: %w", err)
	}

	return nil
}

// dataScopeCallback 数据权限回调函数
// 从 db.Statement.Context 读取用户身份（userID/deptID/dataScope），
// 该上下文由 UserContextMiddleware 在 HTTP 请求路径注入，
// 或由 MQ Consumer / CleanupJob 在异步路径通过 SetUserID/SetDataScope/SetDeptID 注入
func (p *DataScopePlugin) dataScopeCallback(db *gorm.DB) {
	// 未启用则跳过
	if !p.config.Enabled {
		return
	}

	// 支持通过 InstanceSet("skip_data_scope", true) 跳过数据权限过滤
	// 用于直接 ID 查询（FindByID/GetFormData 等），这些查询由 API 层控制权限
	if skip, ok := db.InstanceGet("skip_data_scope"); ok {
		if skipBool, ok := skip.(bool); ok && skipBool {
			return
		}
	}

	ctx := db.Statement.Context
	dataScope := GetDataScope(ctx)
	if dataScope == DataScopeAll {
		// 全部数据权限，无需过滤
		return
	}

	deptID := GetDeptID(ctx)
	userID := GetUserID(ctx)

	// 获取当前查询的表名
	tableName := getTableName(db)
	if tableName == "" {
		return
	}

	// 获取表配置（白名单模式：未配置的表不进行数据权限过滤）
	tableConfig, ok := p.config.Tables[tableName]
	if !ok {
		return
	}

	// 表未启用数据权限
	if !tableConfig.Enabled {
		return
	}

	// 生成数据权限 SQL 条件
	condition, args := p.buildDataScopeCondition(dataScope, deptID, userID, tableConfig, db)
	if condition == "" {
		return
	}

	// 添加 WHERE 条件
	db.Statement.AddClause(clause.Where{
		Exprs: []clause.Expression{clause.Expr{SQL: condition, Vars: args}},
	})
}

// buildDataScopeCondition 构建数据权限 SQL 条件
func (p *DataScopePlugin) buildDataScopeCondition(dataScope int8, deptID, userID int64, config TableScopeConfig, db *gorm.DB) (string, []interface{}) {
	var conditions []string
	var args []interface{}

	// 获取表别名
	alias := config.Alias
	if alias == "" {
		alias = getTableAlias(db)
	}

	// 字段前缀（考虑表别名）
	prefix := ""
	if alias != "" {
		prefix = alias + "."
	}

	switch dataScope {
	case DataScopeAll:
		// 全部数据，无需过滤
		return "", nil

	case DataScopeDeptTree:
		// 部门及子部门数据
		deptField := config.DeptField
		if deptField == "" {
			deptField = p.config.DefaultDeptField
		}

		if config.TreePathField != "" {
			// 使用树路径查询：匹配本部门 + 所有子部门
			// tree_path 格式为 "0,1,2"（逗号分隔的祖先 ID 链）
			// 需覆盖：本部门(id=deptID) + 路径末尾(%,deptID) + 路径中间(%,deptID,%) + 路径开头(deptID,%)
			treeField := prefix + config.TreePathField
			conditions = append(conditions,
				fmt.Sprintf("(id = ? OR %s LIKE ? OR %s LIKE ? OR %s LIKE ?)",
					treeField, treeField, treeField))
			args = append(args,
				deptID,
				fmt.Sprintf("%%,%d", deptID),   // 末尾：如 "0,1"
				fmt.Sprintf("%%,%d,%%", deptID), // 中间：如 "0,1,2"
				fmt.Sprintf("%d,%%", deptID),    // 开头：如 "1,2"
			)
		} else {
			// 仅本部门（无树路径字段时降级）
			conditions = append(conditions, fmt.Sprintf("%s%s = ?", prefix, deptField))
			args = append(args, deptID)
		}

	case DataScopeDept:
		// 本部门数据
		deptField := config.DeptField
		if deptField == "" {
			deptField = p.config.DefaultDeptField
		}
		conditions = append(conditions, fmt.Sprintf("%s%s = ?", prefix, deptField))
		args = append(args, deptID)

	case DataScopeSelf:
		// 本人数据
		scopeField := config.ScopeField
		if scopeField == "" {
			scopeField = p.config.DefaultScopeField
		}
		conditions = append(conditions, fmt.Sprintf("%s%s = ?", prefix, scopeField))
		args = append(args, userID)
	}

	if len(conditions) == 0 {
		return "", nil
	}

	return strings.Join(conditions, " AND "), args
}

// getTableName 从 GORM Statement 获取表名
func getTableName(db *gorm.DB) string {
	if db.Statement == nil || db.Statement.Table == "" {
		return ""
	}
	return db.Statement.Table
}

// getTableAlias 从 GORM Statement 获取表别名
func getTableAlias(db *gorm.DB) string {
	if db.Statement == nil {
		return ""
	}

	// 从 TableExpr 解析别名
	if db.Statement.TableExpr != nil {
		tableExpr := db.Statement.TableExpr.SQL
		// 解析 "table_name AS alias" 或 "table_name alias" 格式
		re := regexp.MustCompile(`(?i)\bAS\s+(\w+)\s*$`)
		matches := re.FindStringSubmatch(tableExpr)
		if len(matches) > 1 {
			return matches[1]
		}

		// 尝试匹配 "table_name alias" 格式
		parts := strings.Fields(tableExpr)
		if len(parts) >= 2 && !strings.EqualFold(parts[len(parts)-2], "AS") {
			return parts[len(parts)-1]
		}
	}

	return ""
}

// DefaultDataScopeConfig 默认数据权限配置
// 包含常见业务表的数据权限配置
// 白名单模式：仅此处显式配置的表会被 DataScopePlugin 过滤，
// 未配置的表（如 sys_role/sys_menu 等系统表）不受影响。
//
// 注意：配置表时必须确认该表拥有 ScopeField（默认 create_by）和 DeptField（默认 dept_id）列，
// 否则当用户拥有部门级数据权限时，查询会因 Unknown column 报错。
// 新增业务表前请先核实表结构。
func DefaultDataScopeConfig() DataScopeConfig {
	return DataScopeConfig{
		Enabled:           true,
		DefaultScopeField: "create_by",
		DefaultDeptField:  "dept_id",
		Tables: map[string]TableScopeConfig{
			"sys_user": {
				Enabled:    true,
				ScopeField: "id", // 用户表本人数据按 ID 过滤
				DeptField:  "dept_id",
			},
			"sys_dept": {
				Enabled:       true,
				ScopeField:    "create_by",
				DeptField:     "id",
				TreePathField: "tree_path",
			},
		},
	}
}

// RegisterDataScopePlugin 在数据库连接上注册数据权限插件
func RegisterDataScopePlugin(db *gorm.DB) error {
	plugin := NewDataScopePlugin(DefaultDataScopeConfig())
	if err := db.Use(plugin); err != nil {
		return fmt.Errorf("注册数据权限插件失败: %w", err)
	}
	return nil
}
