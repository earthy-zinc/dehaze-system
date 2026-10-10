package feedback

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stubFileStorageConfig 注入存储后端配置（各后端 baseUrl 决定允许的图片 URL 前缀）。
// 校验只依赖 config.GetConfig()，无需 DB。
func stubFileStorageConfig(t *testing.T) {
	t.Helper()
	original := config.Config
	t.Cleanup(func() { config.Config = original })

	cfg := &config.AppConfig{}
	cfg.File.Storage.MinIO.Endpoint = "cdn.test:9110"
	cfg.File.Storage.MinIO.BucketName = "dehaze"
	cfg.File.Storage.NginxStatic.BaseURL = "http://cdn.test:9000"
	cfg.File.Storage.Local.BaseURL = "http://cdn.test:8989/api/v1/files/download"
	config.Config = cfg
}

func bizCode(t *testing.T, err error) string {
	t.Helper()
	require.Error(t, err)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误: %v", err)
	return bizErr.Code().Code
}

func TestValidateImageUrls_DomainPrefixAndExtension(t *testing.T) {
	stubFileStorageConfig(t)

	t.Run("各存储后端 baseUrl 前缀均通过", func(t *testing.T) {
		urls := []string{
			"http://cdn.test:9000/datasets/NH-HAZE-2023/hazy/001.JPG",
			"http://cdn.test:9110/dehaze/predictions/20260922/DCP_1.png",
			"http://cdn.test:8989/api/v1/files/download/upload/20260922/abc.webp",
			"http://cdn.test:9000/img1.jpg?x=1#frag",
		}
		assert.NoError(t, validateImageUrls(urls, 5))
	})

	// 与 Java/Python 对齐：比完整前缀而非仅 host，同 host 但路径不属存储后端一律拒绝
	t.Run("同 host 但路径前缀不符被拒绝", func(t *testing.T) {
		err := validateImageUrls([]string{"http://cdn.test:9110/other/x.jpg"}, 5)
		assert.Equal(t, "A0400", bizCode(t, err))
		assert.Equal(t, "图片URL域名不合法", err.Error())
	})

	t.Run("未配置域名被拒绝", func(t *testing.T) {
		err := validateImageUrls([]string{"http://127.0.0.1:9000/img1.jpg"}, 5)
		assert.Equal(t, "A0400", bizCode(t, err))
	})

	t.Run("数量超限", func(t *testing.T) {
		err := validateImageUrls([]string{
			"http://cdn.test:9000/1.jpg",
			"http://cdn.test:9000/2.jpg",
			"http://cdn.test:9000/3.jpg",
			"http://cdn.test:9000/4.jpg",
		}, 3)
		assert.Equal(t, "A0400", bizCode(t, err))
		assert.Equal(t, "图片数量不能超过3张", err.Error())
	})

	t.Run("空 URL", func(t *testing.T) {
		err := validateImageUrls([]string{"  "}, 3)
		assert.Equal(t, "A0400", bizCode(t, err))
		assert.Equal(t, "图片URL不能为空", err.Error())
	})

	// 扩展名不合法报 A0701（与 Java/Python 及文档一致）
	t.Run("扩展名不合法报 A0701", func(t *testing.T) {
		err := validateImageUrls([]string{"http://cdn.test:9000/f1.gif"}, 5)
		assert.Equal(t, "A0701", bizCode(t, err))
		assert.Equal(t, "文件格式不支持", err.Error())
	})
}
