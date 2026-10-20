package ai

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/bo"
	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/stretchr/testify/require"
)

// TestCreateProviderKeyDeduplicatesPerProvider 唯一键为 (provider_id, key_hash)：同一把密钥
// 登记到多个供应商必须都成功（本地 local/local-voice/local-modal 共用 INTERNAL_SERVICE_KEY），
// 只有同一供应商内重复才拒绝；按 provider 维度查重是这条行为的前提。
func TestCreateProviderKeyDeduplicatesPerProvider(t *testing.T) {
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	providerRepo := airepo.NewProviderRepository(db)
	keyRepo := airepo.NewProviderKeyRepository(db)
	svc := NewProviderService(providerRepo, keyRepo, nil)
	ctx := context.Background()
	stamp := time.Now().UnixNano()

	makeProvider := func(suffix string) *model.SysAiProvider {
		p := &model.SysAiProvider{
			ProviderCode:       fmt.Sprintf("kdup_%s_%d", suffix, stamp),
			DisplayName:        "密钥去重探针 " + suffix,
			ApiBaseUrl:         "http://127.0.0.1:8993/v1",
			ProtocolType:       "openai_audio",
			AuthType:           "bearer",
			Status:             1,
			HealthCheckEnabled: 0,
		}
		require.NoError(t, providerRepo.Create(ctx, p))
		return p
	}

	local, voice, modal := makeProvider("local"), makeProvider("voice"), makeProvider("modal")

	const sharedKey = "INTERNAL_SERVICE_KEY_DEDUP_PROBE"
	for _, p := range []*model.SysAiProvider{local, voice, modal} {
		created, err := svc.CreateProviderKey(ctx, p.ID, &bo.ProviderKeyCreateForm{
			Name: "共用密钥", Key: sharedKey,
		}, 1)
		require.NoError(t, err, "provider=%s 首次登记同一把密钥应成功", p.ProviderCode)
		require.NotNil(t, created)
	}

	// 同一供应商重复登记同一把密钥才拒绝，且文案与 python 一致
	_, err := svc.CreateProviderKey(ctx, local.ID, &bo.ProviderKeyCreateForm{
		Name: "重复密钥", Key: sharedKey,
	}, 1)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "同供应商重复密钥应拒绝，实际 err=%v", err)
	require.Equal(t, common.DATA_EXISTS, bizErr.Code())
	require.Contains(t, bizErr.Message(), "该供应商下该 API Key 已存在")

	// 同 hash 在不同 provider 下互不可见（查重不带 provider 会误判为已存在）
	found, err := keyRepo.GetByProviderAndHash(ctx, modal.ID, HashSecret(sharedKey))
	require.NoError(t, err)
	require.NotNil(t, found)
	require.Equal(t, modal.ID, found.ProviderID)

	other, err := keyRepo.GetByProviderAndHash(ctx, local.ID+999999, HashSecret(sharedKey))
	require.NoError(t, err)
	require.Nil(t, other)
}
