package router

import (
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// seedPermRe 从 sys_menu.sql 中提取 perm 列取值（形如 'sys:user:add'）。
var seedPermRe = regexp.MustCompile(`'([a-z][a-z0-9_]*(?::[a-z0-9_]+)+)'`)

// routerPermCallRe 匹配 middleware.Permission("...") / middleware.PermissionWithWildcard("...")。
var routerPermCallRe = regexp.MustCompile(`middleware\.Permission(?:WithWildcard)?\(([^)]*)\)`)

// quotedPermRe 提取调用参数中的字面量权限码。
var quotedPermRe = regexp.MustCompile(`"([^"]+)"`)

// TestRouterPermissionCodesExistInSeed 路由中间件引用的每个权限码都必须存在于 sys_menu 种子。
//
// 幽灵权限码（路由要求、但 sys_menu/sys_role_menu 种子中无定义）会导致该权限永远无法被授予，
// 于是非 ROOT 管理员访问时一律被误判 A0301（HTTP 403）——线上曾因此无法创建/重试/取消任务
// （task 路由要求 sys:task:add / sys:task:delete，而任务菜单 perm=NULL）。本用例是"权限码必须
// 存在于种子 menu"这条规则的机器化护栏。
func TestRouterPermissionCodesExistInSeed(t *testing.T) {
	seedSQL, err := os.ReadFile(filepath.Join(config.RepoRoot(), "config", "sql", "data", "sys_menu.sql"))
	require.NoError(t, err, "无法读取 sys_menu 种子")
	seedCodes := map[string]bool{}
	for _, m := range seedPermRe.FindAllStringSubmatch(string(seedSQL), -1) {
		seedCodes[m[1]] = true
	}
	require.NotEmpty(t, seedCodes, "未能从 sys_menu.sql 提取任何权限码（提取逻辑失效，护栏会假绿）")

	routerDir := filepath.Join(config.GoRoot(), "internal", "router")
	entries, err := os.ReadDir(routerDir)
	require.NoError(t, err)

	// 同一权限码可能被多个文件引用，报错时列出全部来源文件。
	usedCodes := map[string][]string{}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(routerDir, name))
		require.NoError(t, err)
		for _, call := range routerPermCallRe.FindAllStringSubmatch(string(raw), -1) {
			for _, q := range quotedPermRe.FindAllStringSubmatch(call[1], -1) {
				usedCodes[q[1]] = append(usedCodes[q[1]], name)
			}
		}
	}
	require.NotEmpty(t, usedCodes, "未在路由中发现任何权限码引用（检索式失效，护栏会假绿）")

	var missing []string
	for code, files := range usedCodes {
		if !seedCodes[code] {
			missing = append(missing, code+"（引用于 "+strings.Join(files, ",")+"）")
		}
	}
	require.Empty(t, missing,
		"路由使用了未在 sys_menu 种子中定义的幽灵权限码（会导致 A0301/403）：%v", missing)
}

// TestTaskWriteRouteNotGatedByPermission 任务写操作不得要求接口级权限码。
//
// 事实源（python app/router/task.py、java SysTaskController）均无接口级权限标识，任务操作权限
// 由 service 层 create_by 归属校验保证。本用例以「非 ROOT、无任何权限」的用户发起创建任务请求：
// 请求必须穿过中间件到达 handler（落到 A0400 参数错误），不得出现 A0301。若有人给任务写路由
// 恢复 sys:task:add 之类的校验，本用例立即变红。
func TestTaskWriteRouteNotGatedByPermission(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.Use(middleware.ContextErrorHandler())
	engine.Use(func(c *gin.Context) {
		// ROLE_ADMIN 而非 ROLE_ROOT，且不带任何具体权限码
		c.Set("claims", &security.CustomClaims{UserID: 961001, Authorities: []string{"ROLE_ADMIN"}})
		c.Next()
	})
	RegisterTaskRoutes(engine.Group("/api/v1"), api.NewSysTaskApi(nil))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/tasks", strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, req)

	body := rec.Body.String()
	require.NotContains(t, body, "A0301", "任务写操作不得要求无法获得的幽灵权限码（会 403）")
	require.Contains(t, body, "A0400", "空 body 缺 type 应落到参数错误，证明请求已到达 handler")
}

// ==================== 权限码三处一致性（种子 / 前端 mock / 角色可授） ====================

// 权限码在仓库里有三处独立载体，任何一处漂移都不会被现有单端测试发现：
//
//	①配置源头：config/sql/data/sys_menu.sql（perm 列）+ sys_role_menu.sql（谁能授）
//	②前端 mock：dehaze-front-vue/mock/{menu,user}.mock.ts（开发态菜单树 / 用户 perms 数组）
//	③三端代码里的实际引用：java `@ss.hasPerm('x')`、python `require_permission/check_permission`、
//	                        go `middleware.Permission/CheckPermission`
//
// 三处之间的一致性规则（均为"会真红"的硬断言）：
//	A. 前端 mock 用到的权限码必须存在于 sys_menu 种子（幽灵码 ⇒ 开发态按钮/菜单与真实种子不一致）
//	B. 代码中实际要求的每个权限码，必须存在某个 sys_menu 行携带该 perm，且该行被**任一角色**授过
//	   （查 `sys_role_menu`）；否则该权限在任何角色下都不可授予 ⇒ 非 ROOT 用户永远 A0301/403
//	C. `sys_role_menu` 的每个 menu_id 必须指向存在的菜单（孤儿授权行＝种子改动后的静默残留）

var (
	fsPermRe = regexp.MustCompile(`@ss\.hasPerm\('([^']+)'\)`)
	pyPermRe = regexp.MustCompile(`require_permission\("([^"]+)"\)`)
	pyChkRe  = regexp.MustCompile(`check_permission\(\s*[^,]+,\s*"([^"]+)"\s*\)`)
	goPermRe = regexp.MustCompile(`middleware\.Permission(?:WithWildcard)?\(\s*"([^"]+)"`)
	goChkRe  = regexp.MustCompile(`middleware\.CheckPermission\([^,]+,\s*"([^"]+)"`)
	// 前端 mock 中的权限码字面量：`perm: "x:y:z"` 与用户 perms 数组项（形如 `"x:y"`）
	frontMockPermRe = regexp.MustCompile(`"([a-z][a-z0-9_]*:[a-z0-9_]+(?::[a-z0-9_]+)*)"`)
	// sys_menu.sql 中 VALUES(...) 的元组，以及 sys_role_menu.sql 的 (role_id, menu_id)
	valuesRowRe   = regexp.MustCompile(`VALUES\s*\((.*)\);\s*$`)
	roleMenuRowRe = regexp.MustCompile(`\((\d+)\s*,\s*(\d+)\)`)
)

// splitSQLFields 按 SQL 字面量切分元组字段（去引号，NULL 保留为 "NULL"）。
// 不能朴素按逗号切：字符列 `tree_path`/`name`/备注里含逗号与分号。
func splitSQLFields(tuple string) []string {
	var fields []string
	var buf strings.Builder
	inQuote := false
	for i := 0; i < len(tuple); i++ {
		c := tuple[i]
		switch {
		case inQuote:
			if c == '\\' && i+1 < len(tuple) {
				buf.WriteByte(c)
				buf.WriteByte(tuple[i+1])
				i++
				continue
			}
			if c == '\'' {
				inQuote = false
				continue
			}
			buf.WriteByte(c)
		case c == '\'':
			inQuote = true
		case c == ',':
			fields = append(fields, strings.TrimSpace(buf.String()))
			buf.Reset()
		default:
			buf.WriteByte(c)
		}
	}
	return append(fields, strings.TrimSpace(buf.String()))
}

// seedMenuPerms 解析 sys_menu.sql：perm 码 → 携带它的菜单 id 列表；并返回全部菜单 id。
func seedMenuPerms(t *testing.T) (map[string][]string, map[string]bool) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(config.RepoRoot(), "config", "sql", "data", "sys_menu.sql"))
	require.NoError(t, err, "无法读取 sys_menu 种子")

	permToIDs := map[string][]string{}
	menuIDs := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		match := valuesRowRe.FindStringSubmatch(line)
		if match == nil {
			continue
		}
		fields := splitSQLFields(match[1])
		// 列序（全表统一）：id,parent_id,tree_path,name,type,path,component,perm,...
		require.Greaterf(t, len(fields), 7, "sys_menu 行字段数异常：%s", line)
		menuIDs[fields[0]] = true
		if perm := fields[7]; perm != "NULL" && perm != "" {
			permToIDs[perm] = append(permToIDs[perm], fields[0])
		}
	}
	require.NotEmpty(t, menuIDs, "未能从 sys_menu.sql 解析出菜单行（解析式失效，护栏会假绿）")
	require.NotEmpty(t, permToIDs, "未能从 sys_menu.sql 解析出权限码（解析式失效，护栏会假绿）")
	return permToIDs, menuIDs
}

// grantedMenuIDs 解析 sys_role_menu.sql：被任一角色授过的菜单 id 集合。
func grantedMenuIDs(t *testing.T) map[string]bool {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(config.RepoRoot(), "config", "sql", "data", "sys_role_menu.sql"))
	require.NoError(t, err, "无法读取 sys_role_menu 种子")

	granted := map[string]bool{}
	for _, m := range roleMenuRowRe.FindAllStringSubmatch(string(raw), -1) {
		granted[m[2]] = true
	}
	require.NotEmpty(t, granted, "未能从 sys_role_menu.sql 解析出授权行（解析式失效，护栏会假绿）")
	return granted
}

// scanDirPerms 收集目录下 root 内所有源码中匹配式命中的权限码（相对 RepoRoot 的路径 + 命中来源）。
func scanDirPerms(t *testing.T, relDir string, skipTests bool, res ...*regexp.Regexp) map[string]string {
	t.Helper()
	found := map[string]string{}
	root := filepath.Join(config.RepoRoot(), filepath.FromSlash(relDir))
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		require.NoError(t, err)
		if d.IsDir() || (skipTests && strings.HasSuffix(path, "_test.go")) {
			return nil
		}
		raw, readErr := os.ReadFile(path)
		if readErr != nil {
			return nil
		}
		text := string(raw)
		for _, re := range res {
			for _, m := range re.FindAllStringSubmatch(text, -1) {
				if _, ok := found[m[1]]; !ok {
					found[m[1]] = path
				}
			}
		}
		return nil
	})
	require.NoError(t, err, "扫描目录失败：%s", relDir)
	return found
}

// usedPermissionCodes 收集三端代码中实际引用的权限码（跨 java / python / go 三端一致口径）。
func usedPermissionCodes(t *testing.T) map[string]string {
	t.Helper()
	used := map[string]string{}
	merge := func(src map[string]string) {
		for code, from := range src {
			if _, ok := used[code]; !ok {
				used[code] = from
			}
		}
	}
	merge(scanDirPerms(t, "dehaze-java/src/main", false, fsPermRe))
	merge(scanDirPerms(t, "dehaze-python/app", false, pyPermRe, pyChkRe))
	merge(scanDirPerms(t, "dehaze-go/internal", true, goPermRe, goChkRe))
	return used
}

// discoverFrontMockFiles 找出所有"存在 mock 菜单文件"的前端，返回其 mock 权限码载体文件：
// 每个前端的 `mock/menu.mock.ts`（必有）与同目录 `mock/user.mock.ts`（登录用户 perms 数组，若存在）。
// 新增前端 mock 目录后被自动纳入，无需改动本用例的清单。
func discoverFrontMockFiles(t *testing.T) []string {
	t.Helper()
	entries, err := os.ReadDir(config.RepoRoot())
	require.NoError(t, err, "无法列出仓库根目录")

	var files []string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		menuMock := filepath.Join(entry.Name(), "mock", "menu.mock.ts")
		if _, statErr := os.Stat(filepath.Join(config.RepoRoot(), menuMock)); statErr != nil {
			continue
		}
		files = append(files, menuMock)
		userMock := filepath.Join(entry.Name(), "mock", "user.mock.ts")
		if _, statErr := os.Stat(filepath.Join(config.RepoRoot(), userMock)); statErr == nil {
			files = append(files, userMock)
		}
	}
	require.NotEmpty(t, files, "未发现任何前端 mock 菜单文件（检索式失效，护栏会假绿）")
	return files
}

// TestFrontMockPermissionCodesExistInSeed 前端 mock 用到的权限码必须都在 sys_menu 种子中。
//
// 覆盖面：所有带 `mock/menu.mock.ts` 的前端（当前＝dehaze-front-vue、dehaze-front-react；taro / uniapp /
// react-native / flutter 无 mock 目录，故不在内）。发现方式为目录扫描，新增前端自动纳入。
//
// 会红场景：mock 里写了种子里不存在的码（曾实测 `sys:dict_type:add` / `sys:dict:add` 是历史命名，
// 种子早已改成 `sys:dict:type:add` / `sys:dict:data:add`）⇒ 开发态 mock 授权的按钮与真实种子脱节，
// 排查真实环境"按钮不显示"时被误导。
func TestFrontMockPermissionCodesExistInSeed(t *testing.T) {
	seedCodes, _ := seedMenuPerms(t)
	mockFiles := discoverFrontMockFiles(t)
	var missing []string
	total := 0
	for _, rel := range mockFiles {
		raw, err := os.ReadFile(filepath.Join(config.RepoRoot(), filepath.FromSlash(rel)))
		require.NoErrorf(t, err, "无法读取前端 mock：%s", rel)
		codes := map[string]bool{}
		for _, m := range frontMockPermRe.FindAllStringSubmatch(string(raw), -1) {
			codes[m[1]] = true
		}
		require.NotEmptyf(t, codes, "%s 中未提取到权限码（提取式失效，护栏会假绿）", rel)
		total += len(codes)
		for _, code := range sortedKeysFrom(codes) {
			if _, ok := seedCodes[code]; !ok {
				missing = append(missing, code+"（"+rel+"）")
			}
		}
	}
	require.Greaterf(t, total, 10, "前端 mock 权限码提取量骤减（%d 条）：提取式或 mock 结构已变", total)
	require.Emptyf(t, missing,
		"前端 mock 使用了 sys_menu 种子中不存在的权限码（开发态与真实种子脱节）：%v", missing)
}

// TestEveryUsedPermissionIsGrantableBySomeRole 代码要求的权限码必须"可授"。
//
// 判据＝存在某个 sys_menu 行携带该 perm，且该行出现在 sys_role_menu（任一角色）。
// 会红场景：①代码用了种子里没有的幽灵码（新权限只改了路由没改种子）；
// ②菜单加了 perm 但没配给任何角色 ⇒ 该能力对全部非 ROOT 用户永久 403/A0301。
func TestEveryUsedPermissionIsGrantableBySomeRole(t *testing.T) {
	permToIDs, _ := seedMenuPerms(t)
	granted := grantedMenuIDs(t)
	used := usedPermissionCodes(t)
	require.Greaterf(t, len(used), 50, "三端权限码提取量骤减（%d 条）：提取式或目录结构已变", len(used))

	var unknown, ungrantable []string
	for _, code := range sortedKeysFrom(used) {
		ids, inSeed := permToIDs[code]
		if !inSeed {
			unknown = append(unknown, code+"（"+used[code]+"）")
			continue
		}
		grantable := false
		for _, id := range ids {
			if granted[id] {
				grantable = true
				break
			}
		}
		if !grantable {
			ungrantable = append(ungrantable, code+"（menu id "+strings.Join(ids, ",")+" 未被任何角色授予，来源 "+used[code]+"）")
		}
	}
	require.Emptyf(t, unknown,
		"代码引用了 sys_menu 种子中不存在的幽灵权限码（非 ROOT 一律 A0301/403）：%v", unknown)
	require.Emptyf(t, ungrantable,
		"代码要求的权限码对任何角色都不可授予（menu 行未被 sys_role_menu 授予）：%v", ungrantable)
}

// TestRoleMenuGrantsPointToExistingMenus sys_role_menu 不得留下指向已删菜单的孤儿授权行。
func TestRoleMenuGrantsPointToExistingMenus(t *testing.T) {
	_, menuIDs := seedMenuPerms(t)
	granted := grantedMenuIDs(t)

	var orphans []string
	for _, id := range sortedKeysFrom(granted) {
		if !menuIDs[id] {
			orphans = append(orphans, id)
		}
	}
	require.Emptyf(t, orphans, "sys_role_menu 授权了 sys_menu 中不存在的菜单 id（种子残留）：%v", orphans)
}

func sortedKeysFrom[T any](set map[string]T) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
