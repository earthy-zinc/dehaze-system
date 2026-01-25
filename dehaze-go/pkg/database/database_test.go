package database_test

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/database"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/mysql"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/postgres"
	_ "github.com/earthyzinc/dehaze-go/pkg/database/sqlite"
	"github.com/stretchr/testify/assert"
)

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name      string
		config    *database.Config
		wantError bool
	}{
		{
			name: "valid mysql config",
			config: &database.Config{
				Driver:       "mysql",
				MaxIdleConns: 10,
				MaxOpenConns: 100,
				MySQL: database.MySQLConfig{
					Master: database.MySQLInstanceConfig{
						Host:     "localhost",
						Port:     3306,
						Database: "test",
						Username: "root",
						Password: "password",
					},
				},
			},
			wantError: false,
		},
		{
			name: "invalid driver",
			config: &database.Config{
				Driver: "invalid",
			},
			wantError: true,
		},
		{
			name: "missing mysql host",
			config: &database.Config{
				Driver:       "mysql",
				MaxIdleConns: 10,
				MaxOpenConns: 100,
				MySQL: database.MySQLConfig{
					Master: database.MySQLInstanceConfig{
						Database: "test",
						Username: "root",
					},
				},
			},
			wantError: true,
		},
		{
			name: "invalid connection pool config",
			config: &database.Config{
				Driver:       "mysql",
				MaxIdleConns: 200,
				MaxOpenConns: 100,
				MySQL: database.MySQLConfig{
					Master: database.MySQLInstanceConfig{
						Host:     "localhost",
						Port:     3306,
						Database: "test",
						Username: "root",
					},
				},
			},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.wantError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestMaskDSN(t *testing.T) {
	tests := []struct {
		name string
		dsn  string
		want string
	}{
		{
			name: "mysql dsn",
			dsn:  "root:password123@tcp(localhost:3306)/test?charset=utf8mb4",
			want: "root:***@tcp(localhost:3306)/test?charset=utf8mb4",
		},
		{
			name: "postgres dsn",
			dsn:  "host=localhost port=5432 user=postgres password=secret dbname=test",
			want: "host=localhost port=5432 user=postgres password=*** dbname=test",
		},
		{
			name: "no password",
			dsn:  "root@tcp(localhost:3306)/test",
			want: "root@tcp(localhost:3306)/test",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := database.MaskDSN(tt.dsn)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestBuildMySQLDSN(t *testing.T) {
	instance := database.MySQLInstanceConfig{
		Host:     "localhost",
		Port:     3306,
		Database: "test",
		Username: "root",
		Password: "password",
		Config:   "timeout=10s",
	}

	dsn := database.BuildMySQLDSN(instance, "utf8mb4", "Local", true)
	
	assert.Contains(t, dsn, "root:password@tcp(localhost:3306)/test")
	assert.Contains(t, dsn, "charset=utf8mb4")
	assert.Contains(t, dsn, "loc=Local")
	assert.Contains(t, dsn, "parseTime=True")
	assert.Contains(t, dsn, "timeout=10s")
}

func TestBuildPostgresDSN(t *testing.T) {
	instance := database.PostgresInstanceConfig{
		Host:     "localhost",
		Port:     5432,
		Database: "test",
		Username: "postgres",
		Password: "password",
		Config:   "connect_timeout=10",
	}

	dsn := database.BuildPostgresDSN(instance, "disable")
	
	assert.Contains(t, dsn, "host=localhost")
	assert.Contains(t, dsn, "port=5432")
	assert.Contains(t, dsn, "user=postgres")
	assert.Contains(t, dsn, "password=password")
	assert.Contains(t, dsn, "dbname=test")
	assert.Contains(t, dsn, "sslmode=disable")
	assert.Contains(t, dsn, "connect_timeout=10")
}

func TestFactoryRegistration(t *testing.T) {
	// 测试工厂是否正确注册
	drivers := []string{"mysql", "postgres", "sqlite"}
	
	for _, driver := range drivers {
		factory, err := database.GetFactory(driver)
		assert.NoError(t, err)
		assert.NotNil(t, factory)
	}
	
	// 测试未知驱动
	_, err := database.GetFactory("unknown")
	assert.Error(t, err)
}

// 注意：以下测试需要实际数据库连接，在CI环境中应使用mock或跳过
func TestSQLiteIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	config := &database.Config{
		Driver:       "sqlite",
		MaxIdleConns: 1,
		MaxOpenConns: 1,
		SQLite: database.SQLiteConfig{
			Path: ":memory:", // 使用内存数据库
		},
	}

	// 重置全局实例（测试用）
	database.ResetGlobal()

	err := database.Init(config)
	assert.NoError(t, err)

	// 测试获取实例
	db := database.DB()
	assert.NotNil(t, db)

	// 测试Master/Slave方法
	master := database.Master()
	slave := database.Slave()
	assert.NotNil(t, master)
	assert.NotNil(t, slave)
	// SQLite的Slave应该返回Master
	assert.Equal(t, master, slave)

	// 清理
	err = database.Close()
	assert.NoError(t, err)
}
