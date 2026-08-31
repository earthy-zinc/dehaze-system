// Package testutil 提供 Go 测试的公共基建。
//
// 数据库策略（对齐 dehaze-python/tests/conftest.py，2026-08-23 三端决策）：
// 真实 MySQL 测试库 dehaze_test（与开发同实例），schema/种子数据同源于根目录
// config/sql/，规避 SQLite 方言漂移（DECIMAL 精度/外键/锁语义）。测试数据隔离
// 用外部事务 + SAVEPOINT：被测代码内部 db.Transaction() 自动降级为 SAVEPOINT，
// 测试结束整体 ROLLBACK，种子数据零污染。
package testutil

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	_ "github.com/go-sql-driver/mysql" // 注册 database/sql 的 mysql driver
	"github.com/joho/godotenv"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"gorm.io/gorm/schema"
)

// TestDBName 测试库名，与 config.test.yaml 的 database 保持一致。
const TestDBName = "dehaze_test"

// rebuildLockName 跨进程重建互斥锁：go test ./... 会并行跑多个测试二进制，
// 各自触发重建，须用 MySQL 命名锁串行化，避免同时 DROP/CREATE 互相踩踏。
const rebuildLockName = "dehaze_test_rebuild"

// rebuildLockTimeoutSec 获取重建锁的超时：需覆盖另一进程完成全量重建（导入
// schema+data 全部脚本）的时间。
const rebuildLockTimeoutSec = 300

var (
	schemaOnce sync.Once // 进程级只重建一次
	schemaErr  error

	envOnce sync.Once

	testDBOnce sync.Once
	testDB     *gorm.DB
	testDBErr  error
)

// silentLogger 静音 gorm 日志，避免测试输出被 SQL 刷屏。
var silentLogger = logger.Default.LogMode(logger.Silent)

type mySQLConfig struct {
	host     string
	port     int
	user     string
	password string
	database string
}

// NewTestDB 返回一个基于 dehaze_test 库、包在独立事务中的 *gorm.DB，供 repository
// 构造函数注入。测试结束整体回滚，写入不落库。首次调用触发进程级库重建（带
// schema 指纹跳过与跨进程锁），MySQL 不可达或脚本失败时 fail-fast。
func NewTestDB(t *testing.T) *gorm.DB {
	t.Helper()
	if err := ensureSchema(); err != nil {
		cfg := loadMySQLConfig()
		t.Fatalf("testutil: 测试库 `%s` 初始化失败: %v\n连接参数: %s@%s:%d（凭证来自根目录 .env 的 MYSQL_HOST/MYSQL_PASSWORD）",
			cfg.database, err, cfg.user, cfg.host, cfg.port)
	}
	db, err := openTestDB()
	if err != nil {
		cfg := loadMySQLConfig()
		t.Fatalf("testutil: 连接测试库 `%s` 失败: %v\n连接参数: %s@%s:%d（凭证来自根目录 .env 的 MYSQL_HOST/MYSQL_PASSWORD）",
			cfg.database, err, cfg.user, cfg.host, cfg.port)
	}
	tx := db.Begin()
	if tx.Error != nil {
		t.Fatalf("testutil: 开启测试事务失败: %v", tx.Error)
	}
	t.Cleanup(func() { _ = tx.Rollback() })
	return tx.Session(&gorm.Session{Logger: silentLogger})
}

// NewPoolTestDB 返回直连 dehaze_test 的池化 *gorm.DB（不包事务），用于并发/
// CAS 乐观锁类用例：多个 goroutine 需要各自独立连接与真实行锁竞争，单事务
// 回滚模式（NewTestDB）会把全部操作串在同一连接上，无法产生真实并发。用例
// 须自管数据清理（t.Cleanup 中 DELETE 自建数据），不得污染种子数据。
func NewPoolTestDB(t *testing.T) *gorm.DB {
	t.Helper()
	if err := ensureSchema(); err != nil {
		cfg := loadMySQLConfig()
		t.Fatalf("testutil: 测试库 `%s` 初始化失败: %v\n连接参数: %s@%s:%d（凭证来自根目录 .env 的 MYSQL_HOST/MYSQL_PASSWORD）",
			cfg.database, err, cfg.user, cfg.host, cfg.port)
	}
	db, err := openTestDB()
	if err != nil {
		cfg := loadMySQLConfig()
		t.Fatalf("testutil: 连接测试库 `%s` 失败: %v\n连接参数: %s@%s:%d（凭证来自根目录 .env 的 MYSQL_HOST/MYSQL_PASSWORD）",
			cfg.database, err, cfg.user, cfg.host, cfg.port)
	}
	return db.Session(&gorm.Session{Logger: silentLogger})
}

// ensureSchema 进程级一次执行库重建检查：hash 一致则跳过，不一致才 DROP+CREATE
// 并全量导入 config/sql。跨进程用 GET_LOCK 串行化，后到者等待后复查指纹。
func ensureSchema() error {
	schemaOnce.Do(func() { schemaErr = rebuildIfStale() })
	return schemaErr
}

func rebuildIfStale() error {
	cfg := loadMySQLConfig()
	raw, err := sql.Open("mysql", serverDSN(cfg))
	if err != nil {
		return fmt.Errorf("打开 MySQL 连接失败: %w", err)
	}
	raw.SetMaxOpenConns(1) // 全部重建语句与 GET_LOCK 需在同一连接上执行
	ctx := context.Background()
	conn, err := raw.Conn(ctx)
	if err != nil {
		return fmt.Errorf("获取 MySQL 连接失败: %w", err)
	}
	defer conn.Close()

	var got sql.NullInt64
	if err := conn.QueryRowContext(ctx, "SELECT GET_LOCK(?, ?)", rebuildLockName, rebuildLockTimeoutSec).Scan(&got); err != nil {
		return fmt.Errorf("获取重建锁失败: %w", err)
	}
	if !got.Valid || got.Int64 != 1 {
		return fmt.Errorf("获取重建锁超时（%s），另一测试进程可能在长时间重建", rebuildLockName)
	}
	defer conn.ExecContext(ctx, "SELECT RELEASE_LOCK(?)", rebuildLockName)

	hash, err := sqlFingerprint()
	if err != nil {
		return err
	}
	stale, err := isSchemaStale(ctx, conn, hash)
	if err != nil {
		return fmt.Errorf("检查测试库 `%s` 指纹失败: %w", cfg.database, err)
	}
	if !stale {
		return nil
	}
	if err := rebuildSchema(ctx, conn, hash); err != nil {
		return fmt.Errorf("重建测试库 `%s` 失败: %w", cfg.database, err)
	}
	return nil
}

// isSchemaStale 返回库是否需要重建：库不存在、meta 表缺失或指纹不一致时重建。
func isSchemaStale(ctx context.Context, conn *sql.Conn, hash string) (bool, error) {
	var schemaName sql.NullString
	err := conn.QueryRowContext(ctx,
		"SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = ?", TestDBName).Scan(&schemaName)
	if err == sql.ErrNoRows {
		return true, nil
	}
	if err != nil {
		return false, err
	}
	var stored string
	err = conn.QueryRowContext(ctx,
		"SELECT schema_hash FROM `"+TestDBName+"`._test_schema_meta WHERE id = 1").Scan(&stored)
	if err != nil {
		// 表不存在（首次建库后未写过指纹）按不一致处理
		return true, nil
	}
	return stored != hash, nil
}

func rebuildSchema(ctx context.Context, conn *sql.Conn, hash string) error {
	exec := func(stmt string) error {
		if _, err := conn.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("%s: %w\n语句: %.200s", stmt, err, stmt)
		}
		return nil
	}
	if err := exec("DROP DATABASE IF EXISTS `" + TestDBName + "`"); err != nil {
		return err
	}
	if err := exec("CREATE DATABASE `" + TestDBName + "` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"); err != nil {
		return err
	}
	if err := exec("USE `" + TestDBName + "`"); err != nil {
		return err
	}
	root := repoRoot()
	for _, dir := range []string{"schema", "data"} {
		path := filepath.Join(root, "config", "sql", dir)
		entries, err := os.ReadDir(path) // 按文件名排序，保证依赖顺序（schema 先于 data）
		if err != nil {
			return fmt.Errorf("读取 SQL 目录失败: %w", err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasPrefix(e.Name(), "sys_") || !strings.HasSuffix(e.Name(), ".sql") {
				continue
			}
			if err := execSQLFile(ctx, conn, filepath.Join(path, e.Name())); err != nil {
				return err
			}
		}
	}
	// meta 表记录本次指纹，供后续进程跳过重建
	if err := exec("CREATE TABLE IF NOT EXISTS _test_schema_meta (id INT NOT NULL PRIMARY KEY, schema_hash VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"); err != nil {
		return err
	}
	if _, err := conn.ExecContext(ctx,
		"INSERT INTO _test_schema_meta (id, schema_hash) VALUES (1, ?) ON DUPLICATE KEY UPDATE schema_hash = ?", hash, hash); err != nil {
		return err
	}
	return nil
}

// execSQLFile 执行单个 SQL 脚本：过滤 -- 行注释；语句按 ; 分割（跳过单引号字符串
// 与反引号标识符内的分号，schema 的 COMMENT 字符串含分号）。
func execSQLFile(ctx context.Context, conn *sql.Conn, path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var lines []string
	for _, line := range strings.Split(string(content), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "--") {
			continue
		}
		lines = append(lines, line)
	}
	for _, stmt := range splitStatements(strings.Join(lines, "\n")) {
		stmt = strings.TrimSpace(stmt)
		if stmt == "" {
			continue
		}
		if _, err := conn.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("执行 SQL 失败（%s）: %w\n语句: %.200s", filepath.Base(path), err, stmt)
		}
	}
	return nil
}

// splitStatements 移植自 dehaze-python/tests/conftest.py 的 _split_statements。
// COMMENT 字符串内的分号（如 sys_ai_agent 的 agent_code 注释）不能被当作语句
// 分隔符，否则会截断语句。
func splitStatements(content string) []string {
	var statements []string
	buf := make([]byte, 0, 1024)
	inString := false
	inBacktick := false
	i := 0
	for i < len(content) {
		ch := content[i]
		switch {
		case inString:
			buf = append(buf, ch)
			if ch == '\\' && i+1 < len(content) {
				buf = append(buf, content[i+1])
				i++
			} else if ch == '\'' {
				if i+1 < len(content) && content[i+1] == '\'' {
					buf = append(buf, '\'')
					i++
				} else {
					inString = false
				}
			}
		case inBacktick:
			buf = append(buf, ch)
			if ch == '`' {
				inBacktick = false
			}
		case ch == '\'':
			inString = true
			buf = append(buf, ch)
		case ch == '`':
			inBacktick = true
			buf = append(buf, ch)
		case ch == ';':
			statements = append(statements, string(buf))
			buf = buf[:0]
		default:
			buf = append(buf, ch)
		}
		i++
	}
	statements = append(statements, string(buf))
	return statements
}

// sqlFingerprint 计算 config/sql 下全部 sys_*.sql 文件（含文件名）的内容摘要，
// 作为 schema/种子脚本是否变更的依据。
func sqlFingerprint() (string, error) {
	h := sha256.New()
	root := repoRoot()
	for _, dir := range []string{"schema", "data"} {
		path := filepath.Join(root, "config", "sql", dir)
		entries, err := os.ReadDir(path)
		if err != nil {
			return "", fmt.Errorf("读取 SQL 目录失败: %w", err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasPrefix(e.Name(), "sys_") || !strings.HasSuffix(e.Name(), ".sql") {
				continue
			}
			b, err := os.ReadFile(filepath.Join(path, e.Name()))
			if err != nil {
				return "", err
			}
			h.Write([]byte(e.Name()))
			h.Write(b)
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func openTestDB() (*gorm.DB, error) {
	testDBOnce.Do(func() {
		cfg := loadMySQLConfig()
		// 与项目全局一致启用单数表名（pkg/database.GetGormConfig），否则 SysUser 会被映射成 sys_users
		db, err := gorm.Open(mysql.Open(testDBDSN(cfg)), &gorm.Config{
			Logger: silentLogger,
			NamingStrategy: schema.NamingStrategy{
				SingularTable: true,
			},
		})
		if err != nil {
			testDBErr = err
			return
		}
		sqlDB, err := db.DB()
		if err != nil {
			testDBErr = err
			return
		}
		// 连接池：并发/CAS 用例需要多个 goroutine 各持独立连接产生真实行锁竞争
		sqlDB.SetMaxOpenConns(8)
		sqlDB.SetMaxIdleConns(2)
		sqlDB.SetConnMaxLifetime(time.Hour)
		if err := sqlDB.Ping(); err != nil {
			testDBErr = err
			return
		}
		testDB = db
	})
	return testDB, testDBErr
}

func loadMySQLConfig() mySQLConfig {
	envOnce.Do(func() {
		// 凭证与 CWD 解耦：测试进程 CWD 是各包目录，必须显式加载仓库根 .env
		_ = godotenv.Load(filepath.Join(repoRoot(), ".env"))
	})
	host := os.Getenv("MYSQL_HOST")
	if host == "" {
		host = "127.0.0.1"
	}
	port := 3306
	if p, err := strconv.Atoi(os.Getenv("MYSQL_PORT")); err == nil && p > 0 {
		port = p
	}
	return mySQLConfig{
		host:     host,
		port:     port,
		user:     os.Getenv("MYSQL_USERNAME"), // 与 config.yaml/config.test.yaml 的 mysql.username 一致
		password: os.Getenv("MYSQL_PASSWORD"),
		database: TestDBName,
	}
}

func serverDSN(cfg mySQLConfig) string {
	return fmt.Sprintf("%s:%s@tcp(%s:%d)/?charset=utf8mb4&parseTime=True&loc=Local",
		cfg.user, cfg.password, cfg.host, cfg.port)
}

func testDBDSN(cfg mySQLConfig) string {
	return fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=utf8mb4&parseTime=True&loc=Local",
		cfg.user, cfg.password, cfg.host, cfg.port, cfg.database)
}

// repoRoot 定位仓库根（dehaze-system）：dehaze-go 的上级（config/sql、.env 位于仓库根）。
func repoRoot() string {
	return filepath.Dir(goRepoRoot())
}
