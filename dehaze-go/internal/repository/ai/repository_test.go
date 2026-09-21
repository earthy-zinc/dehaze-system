package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

func uniqueName(prefix string) string {
	return prefix + strconv.FormatInt(time.Now().UnixNano(), 10)
}

func TestModelRepositoryLifecycle(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	modelRepo := NewModelRepository(db)
	providerRepo := NewProviderRepository(db)

	provider := &model.SysAiProvider{
		ProviderCode: uniqueName("p"), DisplayName: "测试供应商",
		ApiBaseUrl: "https://example.com/v1", ProtocolType: "openai_compat",
		AuthType: "bearer", Status: 1, HealthCheckEnabled: 1,
	}
	require.NoError(t, providerRepo.Create(ctx, provider))

	modelID := uniqueName("m")
	primary := &model.SysAiModel{
		ProviderID: provider.ID, ModelID: modelID, ModelType: "chat",
		DisplayName: "主模型", MaxContextTokens: 4096, MaxOutputTokens: 4096, SupportsStreaming: 1, Status: 1,
	}
	require.NoError(t, modelRepo.Create(ctx, primary))
	require.NotZero(t, primary.ID)

	found, err := modelRepo.GetByModelAndProvider(ctx, modelID, provider.ID)
	require.NoError(t, err)
	require.NotNil(t, found)
	require.Equal(t, primary.ID, found.ID)

	// 同 model_id 不同供应商可共存（联合唯一键为 model_id + provider_id）
	otherProvider := &model.SysAiProvider{
		ProviderCode: uniqueName("p"), DisplayName: "另一供应商",
		ApiBaseUrl: "https://example.org/v1", ProtocolType: "openai_compat",
		AuthType: "bearer", Status: 1, HealthCheckEnabled: 1,
	}
	require.NoError(t, providerRepo.Create(ctx, otherProvider))
	secondary := &model.SysAiModel{
		ProviderID: otherProvider.ID, ModelID: modelID, ModelType: "chat",
		DisplayName: "同模型另一供应商", MaxContextTokens: 4096, MaxOutputTokens: 4096, SupportsStreaming: 1, Status: 1,
	}
	require.NoError(t, modelRepo.Create(ctx, secondary))

	// 分页关键字 LIKE 通配符转义：'%' 只匹配字面量
	percentModel := &model.SysAiModel{
		ProviderID: provider.ID, ModelID: uniqueName("m"), ModelType: "chat",
		DisplayName: "含%百分号", MaxContextTokens: 4096, MaxOutputTokens: 4096, SupportsStreaming: 1, Status: 1,
	}
	require.NoError(t, modelRepo.Create(ctx, percentModel))
	pageModels, total, err := modelRepo.PaginateModels(ctx, 1, 10, "%", "")
	require.NoError(t, err)
	require.EqualValues(t, 1, total, "keyword '%' 应只命中含字面量 % 的行")
	require.Len(t, pageModels, 1)
	require.Equal(t, percentModel.ID, pageModels[0].ID)

	// 降级链引用统计（仅统计启用行）
	fallbackTarget := primary.ID
	require.NoError(t, modelRepo.Update(ctx, secondary.ID, map[string]interface{}{"fallback_model_id": fallbackTarget}))
	refs, err := modelRepo.CountFallbackTargets(ctx, fallbackTarget)
	require.NoError(t, err)
	require.EqualValues(t, 1, refs)

	enabled, err := modelRepo.ListEnabledByPKs(ctx, []int64{primary.ID, secondary.ID})
	require.NoError(t, err)
	require.Len(t, enabled, 2)

	// 软删：deleted = id，业务键查询不再命中
	require.NoError(t, modelRepo.SoftDeleteByIDs(ctx, []int64{secondary.ID}, 1))
	after, err := modelRepo.GetByPK(ctx, secondary.ID)
	require.NoError(t, err)
	require.Nil(t, after)
	refs, err = modelRepo.CountFallbackTargets(ctx, fallbackTarget)
	require.NoError(t, err)
	require.EqualValues(t, 0, refs, "软删行不计入降级链引用")

	// 删除模型后（deleted=id）同 (model_id, provider_id) 可重建
	require.NoError(t, modelRepo.SoftDeleteByIDs(ctx, []int64{primary.ID}, 1))
	recreated := &model.SysAiModel{
		ProviderID: provider.ID, ModelID: primary.ModelID, ModelType: "chat",
		DisplayName: "重建模型", MaxContextTokens: 4096, MaxOutputTokens: 4096, SupportsStreaming: 1, Status: 1,
	}
	require.NoError(t, modelRepo.Create(ctx, recreated), "唯一键含 deleted，软删后可重建同组合")
}

func TestProviderCodeIsWhitelistAcrossSoftDelete(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	repo := NewProviderRepository(db)

	code := uniqueName("whitelist")
	provider := &model.SysAiProvider{
		ProviderCode: code, DisplayName: "白名单校验", ApiBaseUrl: "https://example.com/v1",
		ProtocolType: "openai_compat", AuthType: "bearer", Status: 1, HealthCheckEnabled: 1,
	}
	require.NoError(t, repo.Create(ctx, provider))
	require.NoError(t, repo.SoftDelete(ctx, provider.ID, 1))

	active, err := repo.GetByCode(ctx, code, false)
	require.NoError(t, err)
	require.Nil(t, active)

	historical, err := repo.GetByCode(ctx, code, true)
	require.NoError(t, err)
	require.NotNil(t, historical, "软删行须可查，用于 provider_code 白名单判定")
	require.NotZero(t, historical.Deleted)

	// 供应商下存在模型时删除被拦截（CountModels 含禁用模型）
	modelRepo := NewModelRepository(db)
	require.NoError(t, modelRepo.Create(ctx, &model.SysAiModel{
		ProviderID: provider.ID, ModelID: uniqueName("m"), ModelType: "chat",
		DisplayName: "关联模型", MaxContextTokens: 4096, MaxOutputTokens: 4096, SupportsStreaming: 1, Status: 0,
	}))
	count, err := repo.CountModels(ctx, provider.ID)
	require.NoError(t, err)
	require.EqualValues(t, 1, count)
}

func TestProviderKeyRepository(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	providerRepo := NewProviderRepository(db)
	keyRepo := NewProviderKeyRepository(db)

	provider := &model.SysAiProvider{
		ProviderCode: uniqueName("keyp"), DisplayName: "Key 供应商", ApiBaseUrl: "https://example.com/v1",
		ProtocolType: "openai_compat", AuthType: "bearer", Status: 1, HealthCheckEnabled: 1,
	}
	require.NoError(t, providerRepo.Create(ctx, provider))

	// key_hash 固定 64 字符（char(64) 唯一键），用时间戳左补零保证唯一
	hash := fmt.Sprintf("%064d", time.Now().UnixNano())
	prefix := "sk-abcde..."
	key := &model.SysAiProviderKey{
		ProviderID: provider.ID, Name: "主 Key", KeyHash: hash, KeyPrefix: &prefix,
		KeyCipher: "cipher", Status: 1, Priority: 0, Weight: 1,
	}
	require.NoError(t, keyRepo.Create(ctx, key))

	found, err := keyRepo.GetByHash(ctx, hash)
	require.NoError(t, err)
	require.NotNil(t, found)
	require.Equal(t, key.ID, found.ID)

	enabled, err := keyRepo.CountEnabledByProvider(ctx, provider.ID)
	require.NoError(t, err)
	require.EqualValues(t, 1, enabled)

	// Key 为状态控制（物理删除）
	require.NoError(t, keyRepo.DeleteByID(ctx, key.ID))
	removed, err := keyRepo.GetByID(ctx, key.ID)
	require.NoError(t, err)
	require.Nil(t, removed)
}

func TestSkillRepositoryAgentCountAndMarket(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	repo := NewSkillRepository(db)

	name := uniqueName("skill-")
	instruction := "测试指令"
	skill := &model.SysAiSkill{
		Name: name, Description: "测试 Skill", Scene: "测试",
		Instruction: &instruction, Status: 1, Source: "admin", MarketShared: 1,
	}
	require.NoError(t, repo.Create(ctx, skill))

	// 无关 Skill 不应计入关联数
	otherName := uniqueName("skill-")
	other := &model.SysAiSkill{
		Name: otherName, Description: "另一 Skill", Instruction: &instruction, Status: 0, Source: "admin",
	}
	require.NoError(t, repo.Create(ctx, other))

	for i := 0; i < 2; i++ {
		require.NoError(t, db.Table("sys_ai_agent_skill").Create(map[string]interface{}{
			"agent_id": i + 1, "skill_name": name,
		}).Error)
	}
	counts, err := repo.CountByNames(ctx, []string{name, otherName})
	require.NoError(t, err)
	require.EqualValues(t, 2, counts[name])
	require.EqualValues(t, 0, counts[otherName])

	market, err := repo.ListMarket(ctx)
	require.NoError(t, err)
	containsSkill := false
	for i := range market {
		if market[i].ID == skill.ID {
			containsSkill = true
		}
		if market[i].ID == other.ID {
			t.Fatal("未共享或未启用的 Skill 不应出现在市场目录")
		}
	}
	require.True(t, containsSkill)

	// 软删后按名称查重（includeDeleted）仍可见
	require.NoError(t, repo.SoftDelete(ctx, skill.ID, 1))
	active, err := repo.GetByName(ctx, name, false)
	require.NoError(t, err)
	require.Nil(t, active)
	historical, err := repo.GetByName(ctx, name, true)
	require.NoError(t, err)
	require.NotNil(t, historical)
}

func TestMcpRepositoryServersAndNamespaces(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	repo := NewMcpRepository(db)

	name := uniqueName("server-")
	description := "测试 Server"
	endpoint := "https://example.com/mcp"
	authType := "api_key"
	server := &model.SysAiMcpServer{
		Name: name, Description: &description, ProtocolType: "streamable-http",
		Endpoint: &endpoint, AuthType: &authType, Status: 1,
	}
	require.NoError(t, repo.CreateServer(ctx, server))

	found, err := repo.GetServerByName(ctx, name, false)
	require.NoError(t, err)
	require.NotNil(t, found)
	require.Equal(t, server.ID, found.ID)

	// 凭据以 JSON 密文落库，仅写入不回显
	credentials, err := json.Marshal(map[string]any{"api_key": "cipher-text"})
	require.NoError(t, err)
	require.NoError(t, repo.UpdateServer(ctx, server.ID, map[string]interface{}{"credentials": credentials}))
	withCred, err := repo.GetServer(ctx, server.ID)
	require.NoError(t, err)
	require.JSONEq(t, `{"api_key":"cipher-text"}`, string(withCred.Credentials))

	toolNames, err := json.Marshal([]string{"echo"})
	require.NoError(t, err)
	require.NoError(t, repo.ReplaceNamespaces(ctx, server.ID, []model.SysAiMcpNamespace{
		{ServerID: server.ID, Namespace: "group_a", ToolNames: toolNames},
	}))
	namespaces, err := repo.ListNamespaces(ctx, server.ID)
	require.NoError(t, err)
	require.Len(t, namespaces, 1)
	require.Equal(t, "group_a", namespaces[0].Namespace)

	// 覆盖式更新：整组替换
	require.NoError(t, repo.ReplaceNamespaces(ctx, server.ID, nil))
	namespaces, err = repo.ListNamespaces(ctx, server.ID)
	require.NoError(t, err)
	require.Empty(t, namespaces)

	// 调用审计分页
	userID := int64(7)
	require.NoError(t, repo.CreateCall(ctx, &model.SysAiMcpCall{
		UserID: &userID, ServerID: server.ID, ServerName: &name,
		ToolName: "echo", Status: 1, Result: "success",
	}))
	calls, total, err := repo.PaginateCalls(ctx, 1, 10, &server.ID, "echo")
	require.NoError(t, err)
	require.EqualValues(t, 1, total)
	require.Len(t, calls, 1)
	require.Equal(t, "success", calls[0].Result)
}
