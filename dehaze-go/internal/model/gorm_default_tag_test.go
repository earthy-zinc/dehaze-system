package model

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// ============================================================================
// GORM `default:` 标签全仓守卫
//
// 背景（缺陷类别，非风格问题）：
//   GORM 在 Create/INSERT 时，对「带 default 标签且值为该类型的零值」的字段，
//   会把零值替换成标签里的默认值（callbacks/create.go:298-301，并回写结构体字段）。
//   于是 `default:1` 让显式 `status = 0` 永远落库为 1 —— 「禁用券 / 下架活动 /
//   关闭自动续费 / 隐藏菜单」全部静默失效。
//   注解 `default:` 只影响 Go 侧 INSERT 的取值，不改变 DB 列默认值。
//
// 规则：
//   状态 / 布尔类列（status、visible、enabled、deleted、delval、is_*、supports_* 等，
//   其零值本身是合法业务取值）禁止携带非零 `default:` 标签。
//   需要「缺省即某态」的写法应在创建路径显式赋值，不要依赖 GORM 的零值替换。
//
// 现存的合理例外见下方白名单，必须逐条写明业务理由。
//
// 实现说明：这里用 go/parser 扫描 internal/model 下的源码（而非反射注册表），
//   因为反射无法枚举包内类型，必须手工维护注册表 —— 新模型漏登记就等于守卫有洞。
//   AST 扫描对新增模型自动生效，且同样能给出 model 名 + 字段名 + default 值。
// ============================================================================

// gormZeroValueDefaultWhitelist 记录"状态/布尔类列携带非零 default 标签"的合理例外。
// 键为 "模型名.字段名"，值必须写明业务理由。条目一旦不再命中，测试会失败要求清理。
var gormZeroValueDefaultWhitelist = map[string]string{
	"SysInputHistory.Status": "字段类型为指针 *int8：GORM 仅在指针为 nil 时才套用 default，" +
		"显式 &0 仍会原样写入，因此不存在\"零值被吞\"的缺陷；该列合法取值为 1/2/3（成功/失败/处理中）、0 非法" +
		"（InputHistoryService.validateHistory 已显式拒绝），default:3 的语义是\"创建时未指定则记为处理中\"，属预期行为。",
}

// stateColumnExact 精确匹配的状态 / 布尔类列名。
var stateColumnExact = map[string]struct{}{
	"status":  {},
	"visible": {},
	"enabled": {},
	"deleted": {},
	"delval":  {},
}

// stateColumnPrefixes 前缀匹配的布尔类列名（is_invalid / supports_streaming ...）。
var stateColumnPrefixes = []string{"is_", "has_", "supports_", "allow_", "enable_"}

// stateColumnSuffixes 后缀匹配的状态 / 布尔类列名（read_status / push_enabled / ...）。
var stateColumnSuffixes = []string{"_status", "_visible", "_enabled", "_deleted", "_delval", "_flag"}

var (
	gormColumnTagRe  = regexp.MustCompile(`(?:^|;)column:([^;]*)`)
	gormDefaultTagRe = regexp.MustCompile(`(?:^|;)default:([^;]*)`)
	// upperSnakeRe 用于把 Go 字段名（无显式 column 标签时）转成 GORM 默认列名。
	camelBoundaryRe = regexp.MustCompile(`([a-z0-9])([A-Z])`)
)

// TestGormStateColumnsMustNotCarryNonZeroDefault 守卫：状态/布尔类列不得带非零 default 标签。
func TestGormStateColumnsMustNotCarryNonZeroDefault(t *testing.T) {
	files := modelPackageFiles(t)
	if len(files) == 0 {
		t.Fatal("未定位到 internal/model 下的任何 .go 源文件，扫描无效")
	}

	fset := token.NewFileSet()
	whitelistHit := map[string]bool{}
	var violations []string
	stateColumns := 0
	structCount := 0

	for _, path := range files {
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			t.Fatalf("解析 %s 失败: %v", path, err)
		}
		rel := filepath.Base(path)

		for _, decl := range file.Decls {
			genDecl, ok := decl.(*ast.GenDecl)
			if !ok || genDecl.Tok != token.TYPE {
				continue
			}
			for _, spec := range genDecl.Specs {
				typeSpec, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				structType, ok := typeSpec.Type.(*ast.StructType)
				if !ok {
					continue
				}
				structCount++
				modelName := typeSpec.Name.Name

				for _, field := range structType.Fields.List {
					if len(field.Names) == 0 || field.Tag == nil {
						continue // 内嵌结构体等无字段名/无标签者无关
					}
					rawTag, err := strconv.Unquote(field.Tag.Value)
					if err != nil {
						t.Fatalf("%s: 解析 %s 的标签失败: %v", rel, field.Names[0].Name, err)
					}
					gormTag := reflect.StructTag(rawTag).Get("gorm")
					if gormTag == "" {
						continue
					}

					fieldName := field.Names[0].Name
					column, def, hasDefault := parseGormColumnAndDefault(gormTag, fieldName)
					if !isStateColumn(column) {
						continue
					}
					stateColumns++
					if !hasDefault || isZeroDefaultValue(def) {
						continue
					}

					key := modelName + "." + fieldName
					if _, allowed := gormZeroValueDefaultWhitelist[key]; allowed {
						whitelistHit[key] = true
						continue
					}
					line := fset.Position(field.Pos()).Line
					violations = append(violations, fmt.Sprintf(
						"%s:%d 模型 %s 字段 %s（列 %s）携带非零 default:%s",
						rel, line, modelName, fieldName, column, def))
				}
			}
		}
	}

	// 检测器自证：扫描面为空说明测试失效（而不是"没有问题"）。
	if structCount == 0 || stateColumns == 0 {
		t.Fatalf("扫描未覆盖到模型结构体或状态类列（struct=%d, stateColumns=%d），守卫失效", structCount, stateColumns)
	}

	// 白名单防腐烂：不再命中的条目说明已被修复或列名已改，必须清理。
	for key, reason := range gormZeroValueDefaultWhitelist {
		if !whitelistHit[key] {
			t.Errorf("白名单条目 %s 未命中任何真实字段，请删除该条目（原理由：%s）", key, reason)
		}
	}

	if len(violations) > 0 {
		sort.Strings(violations)
		t.Errorf("发现 %d 处「状态/布尔类列携带非零 default 标签」（GORM 会在 Create 时用默认值吞掉显式零值）：\n  %s\n"+
			"修复方式：移除该字段的 default 标签（DB 列默认值不受影响），并在创建路径显式赋初值；"+
			"确属合理例外时登记 gormZeroValueDefaultWhitelist 并写明业务理由。",
			len(violations), strings.Join(violations, "\n  "))
	}
}

// TestGormDefaultDetectorSelfCheck 自证检测器本身有效（负向断言不能是假绿）。
func TestGormDefaultDetectorSelfCheck(t *testing.T) {
	cases := []struct {
		tag         string
		field       string
		wantColumn  string
		wantDefault string
		wantHas     bool
		wantState   bool
		wantNonZero bool
	}{
		{"column:status;type:tinyint;not null;default:1;comment:状态(1:启用;0:禁用)", "Status", "status", "1", true, true, true},
		{"column:status;type:tinyint;not null;default:0;comment:状态", "Status", "status", "0", true, true, false},
		{"column:is_invalid;type:tinyint;not null;default:0", "IsInvalid", "is_invalid", "0", true, true, false},
		{"column:deleted;type:bigint;not null;default:0", "Deleted", "deleted", "0", true, true, false},
		{"column:supports_streaming;type:tinyint;default:1", "SupportsStreaming", "supports_streaming", "1", true, true, true},
		{"column:name;type:varchar(128);default:''", "Name", "name", "''", true, false, false},
		{"column:model_type;type:varchar(16);default:chat", "ModelType", "model_type", "chat", true, false, true},
		{"type:tinyint", "PushEnabled", "push_enabled", "", false, true, false},
		{"column:payload;type:json;default:null", "Payload", "payload", "null", true, false, false},
	}
	for _, c := range cases {
		column, def, has := parseGormColumnAndDefault(c.tag, c.field)
		if column != c.wantColumn || def != c.wantDefault || has != c.wantHas {
			t.Errorf("parseGormColumnAndDefault(%q, %q) = (%q, %q, %v)，期望 (%q, %q, %v)",
				c.tag, c.field, column, def, has, c.wantColumn, c.wantDefault, c.wantHas)
		}
		if got := isStateColumn(column); got != c.wantState {
			t.Errorf("isStateColumn(%q) = %v，期望 %v", column, got, c.wantState)
		}
		if nonZero := c.wantHas && !isZeroDefaultValue(def); nonZero != c.wantNonZero {
			t.Errorf("tag %q 的 non-zero 判定 = %v，期望 %v", c.tag, nonZero, c.wantNonZero)
		}
	}
}

// modelPackageFiles 返回 internal/model 包下参与扫描的源文件（排除 _test.go）。
func modelPackageFiles(t *testing.T) []string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller 无法定位测试文件")
	}
	dir := filepath.Dir(thisFile)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("读取 %s 失败: %v", dir, err)
	}
	var files []string
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		files = append(files, filepath.Join(dir, name))
	}
	sort.Strings(files)
	return files
}

// parseGormColumnAndDefault 解析 gorm 标签里的列名与 default 值；无 column 标签时按 GORM 默认命名推导。
func parseGormColumnAndDefault(gormTag, fieldName string) (column, defaultValue string, hasDefault bool) {
	if m := gormColumnTagRe.FindStringSubmatch(gormTag); m != nil {
		column = strings.TrimSpace(m[1])
	}
	if column == "" {
		column = toDBColumnName(fieldName)
	}
	if m := gormDefaultTagRe.FindStringSubmatch(gormTag); m != nil {
		return column, strings.TrimSpace(m[1]), true
	}
	return column, "", false
}

// isStateColumn 判定列名是否属于"零值有业务语义"的状态 / 布尔类列。
func isStateColumn(column string) bool {
	if column == "" {
		return false
	}
	if _, ok := stateColumnExact[column]; ok {
		return true
	}
	for _, prefix := range stateColumnPrefixes {
		if strings.HasPrefix(column, prefix) {
			return true
		}
	}
	for _, suffix := range stateColumnSuffixes {
		if strings.HasSuffix(column, suffix) {
			return true
		}
	}
	return false
}

// isZeroDefaultValue 判定 default 标签是否等价于该列的零值（0 / false / null / 空串）。
func isZeroDefaultValue(value string) bool {
	v := strings.TrimSpace(value)
	if v == "" {
		return true
	}
	switch strings.ToLower(strings.Trim(v, `'"`)) {
	case "", "0", "0.0", "0.00", "false", "null":
		return true
	}
	return false
}

// toDBColumnName 复刻 GORM 默认命名策略（CamelCase → snake_case）的常用部分。
func toDBColumnName(fieldName string) string {
	snake := camelBoundaryRe.ReplaceAllString(fieldName, "${1}_${2}")
	return strings.ToLower(snake)
}
