package testutil

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/stretchr/testify/assert"
)

func TestLoadTestConfig(t *testing.T) {
	cfg := LoadTestConfig(t)
	assert.NotNil(t, cfg)
	assert.Equal(t, "mysql", cfg.DB.Driver, "测试配置应指向真实 MySQL")
	assert.Equal(t, "dehaze_test", cfg.DB.MySQL.Database)
	// 若 os.ExpandEnv 未生效，host 会保留 ${MYSQL_HOST} 字面值
	assert.NotEqual(t, "${MYSQL_HOST}", cfg.DB.MySQL.Host, "DB host 应经 os.ExpandEnv 展开")
	assert.Equal(t, 4, cfg.Cache.Redis.DB, "Redis 应使用 db=4 与开发环境（db=0）隔离")
	assert.False(t, cfg.XxlJob.Enabled, "测试环境应禁用 XXL-Job")
	assert.Equal(t, 8998, cfg.System.Port, "测试专用端口，与开发实例端口 8990 错开")
	assert.Same(t, cfg, config.Config, "加载结果应注入全局 config.Config")
}
