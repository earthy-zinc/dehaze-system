package api_key

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	apikeyrepo "github.com/earthyzinc/dehaze-go/internal/repository/api_key"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// seedModel 造一条 sys_ai_model 行（该表无外键，provider_id 可直填；deleted 用于造软删行）
func seedModel(t *testing.T, db *gorm.DB, modelID string, status int, deleted int64) {
	t.Helper()
	require.NoError(t, db.Exec(
		"INSERT INTO sys_ai_model (provider_id, model_id, model_type, display_name, status, deleted) "+
			"VALUES (?, ?, ?, ?, ?, ?)",
		1, modelID, "chat", "白名单造数", status, deleted).Error)
}

// TestCreateApiKeyEchoesGovernanceParams 创建带治理参数的密钥：dailyQuota/monthlyQuota/rpmLimit
// 必须**落库并原样回传**（python `ApiKeyCreate`/`ApiKeyResult` 同形态；SDK
// `api-key.test.ts`「创建带治理参数(配额)的密钥应返回并透传」断言三字段分别为 1000/20000/60）。
func TestCreateApiKeyEchoesGovernanceParams(t *testing.T) {
	db := testutil.NewTestDB(t)
	repo := apikeyrepo.NewApiKeyRepository(db)
	svc := NewApiKeyService(repo, nil, airepo.NewModelRepository(db))
	ctx := context.Background()

	daily, monthly, rpm := int64(1000), int64(20000), 60
	result, err := svc.CreateApiKey(ctx, 960001, &dto.ApiKeyCreateRequest{
		Name:         "治理参数密钥",
		DailyQuota:   &daily,
		MonthlyQuota: &monthly,
		RpmLimit:     &rpm,
	})
	require.NoError(t, err)
	require.NotNil(t, result.DailyQuota)
	require.NotNil(t, result.MonthlyQuota)
	require.NotNil(t, result.RpmLimit)
	require.EqualValues(t, 1000, *result.DailyQuota)
	require.EqualValues(t, 20000, *result.MonthlyQuota)
	require.EqualValues(t, 60, *result.RpmLimit)

	// 落库核对（不只是回显）
	stored, err := repo.FindByUserID(ctx, 960001)
	require.NoError(t, err)
	require.Len(t, stored, 1)
	require.NotNil(t, stored[0].DailyQuota)
	require.EqualValues(t, 1000, *stored[0].DailyQuota)
	require.EqualValues(t, 20000, *stored[0].MonthlyQuota)
	require.EqualValues(t, 60, *stored[0].RpmLimit)

	// 不传治理参数 = 不限制（NULL），不得被默认值填充
	second, err := svc.CreateApiKey(ctx, 960002, &dto.ApiKeyCreateRequest{Name: "不限额密钥"})
	require.NoError(t, err)
	require.Nil(t, second.DailyQuota)
	require.Nil(t, second.MonthlyQuota)
	require.Nil(t, second.RpmLimit)

	// 列表同样回传三字段（python ApiKeyResult 同形态）
	list, err := svc.ListApiKeys(ctx, 960001)
	require.NoError(t, err)
	require.Len(t, list, 1)
	require.NotNil(t, list[0].DailyQuota)
	require.EqualValues(t, 1000, *list[0].DailyQuota)
}

// TestCreateApiKeyModelWhitelist 模型白名单治理字段（python `ApiKeyCreate.model_whitelist` / `ApiKeyResult` 同口径）：
// 不传与空数组都是"继承用户可见模型"、必须落 SQL NULL（而非 JSON null 字面量或空数组），
// JSON 数组必须落库并在创建/列表两处原样回显。
func TestCreateApiKeyModelWhitelist(t *testing.T) {
	db := testutil.NewTestDB(t)
	repo := apikeyrepo.NewApiKeyRepository(db)
	svc := NewApiKeyService(repo, nil, airepo.NewModelRepository(db))
	ctx := context.Background()

	columnIsNull := func(userID int64) bool {
		var isNull int
		require.NoError(t, db.Raw(
			"SELECT model_whitelist IS NULL FROM sys_api_key WHERE user_id = ?", userID).Scan(&isNull).Error)
		return isNull == 1
	}

	plain, err := svc.CreateApiKey(ctx, 962001, &dto.ApiKeyCreateRequest{Name: "不限额密钥"})
	require.NoError(t, err)
	require.Nil(t, plain.ModelWhitelist)
	require.True(t, columnIsNull(962001), "不传白名单必须落 SQL NULL")

	empty, err := svc.CreateApiKey(ctx, 962002, &dto.ApiKeyCreateRequest{
		Name: "空数组密钥", ModelWhitelist: []string{},
	})
	require.NoError(t, err)
	require.Nil(t, empty.ModelWhitelist)
	require.True(t, columnIsNull(962002), "空数组与不传同义（python `model_whitelist or None`）")

	// 脏语料（全角/emoji/列宽上限 60 字符）：白名单已按 python 口径校验模型存在性，
	// 故元素必须真实存在（model_id 列为 varchar(64)，超长标识在 python 同样不可能命中）
	dirty := []string{"qwen3-0.6b", "全角模型", "emoji-🙂", strings.Repeat("m", 60)}
	for _, modelID := range dirty {
		seedModel(t, db, modelID, 1, 0)
	}
	created, err := svc.CreateApiKey(ctx, 962003, &dto.ApiKeyCreateRequest{
		Name: "白名单密钥", ModelWhitelist: dirty,
	})
	require.NoError(t, err)
	require.Equal(t, dirty, created.ModelWhitelist)

	var storedJSON string
	require.NoError(t, db.Raw(
		"SELECT CAST(model_whitelist AS CHAR) FROM sys_api_key WHERE user_id = ?", 962003).Scan(&storedJSON).Error)
	expected, err := json.Marshal(dirty)
	require.NoError(t, err)
	require.JSONEq(t, string(expected), storedJSON)

	list, err := svc.ListApiKeys(ctx, 962003)
	require.NoError(t, err)
	require.Len(t, list, 1)
	require.Equal(t, dirty, list[0].ModelWhitelist)
}

// TestCreateApiKeyRejectsUnavailableWhitelistModel 对齐 python `ApiKeyService._validate_whitelist`：
// 白名单内每个 model_id 必须**存在且启用**（不存在 / 已禁用 / 已软删均不可用），否则 A0400 且不落库。
// 拼错或已停用的模型若被写进白名单，该 Key 的兼容调用会全部 403（`compatible_governance._check_whitelist`）且无从自查。
func TestCreateApiKeyRejectsUnavailableWhitelistModel(t *testing.T) {
	db := testutil.NewTestDB(t)
	repo := apikeyrepo.NewApiKeyRepository(db)
	svc := NewApiKeyService(repo, nil, airepo.NewModelRepository(db))
	ctx := context.Background()

	seedModel(t, db, "wl_ok", 1, 0)
	seedModel(t, db, "wl_disabled", 0, 0)
	seedModel(t, db, "wl_deleted", 1, 1)

	cases := []struct{ name, model string }{
		{"不存在的模型", "wl_missing"},
		{"已禁用模型", "wl_disabled"},
		{"已软删模型", "wl_deleted"},
		{"尾部空格（NO PAD 排序规则下不等价）", "wl_ok "},
		{"超出列宽的超长标识", strings.Repeat("m", 200)},
	}
	for i, tc := range cases {
		userID := int64(963001 + i)
		_, err := svc.CreateApiKey(ctx, userID, &dto.ApiKeyCreateRequest{
			Name: "非法白名单密钥", ModelWhitelist: []string{tc.model},
		})

		bizErr, ok := common.AsBizError(err)
		require.True(t, ok, "%s 应被拒绝，实际 err=%v", tc.name, err)
		require.Equal(t, common.PARAM_ERROR, bizErr.Code(), "%s 错误码应为 A0400", tc.name)
		require.Contains(t, bizErr.Message(), tc.model+" 不存在或未启用", "%s 文案须对齐 python", tc.name)

		var count int64
		require.NoError(t, db.Table("sys_api_key").Where("user_id = ?", userID).Count(&count).Error)
		require.Zero(t, count, "%s 拒绝后不得落库", tc.name)
	}

	// 混合白名单：合法 + 非法必须整体拒绝（逐个校验，不得因首个合法就短路放行）
	_, err := svc.CreateApiKey(ctx, 963099, &dto.ApiKeyCreateRequest{
		Name: "混合白名单密钥", ModelWhitelist: []string{"wl_ok", "wl_missing"},
	})
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "混合白名单含非法项应被拒绝，实际 err=%v", err)
	require.Equal(t, common.PARAM_ERROR, bizErr.Code())
	var count int64
	require.NoError(t, db.Table("sys_api_key").Where("user_id = ?", 963099).Count(&count).Error)
	require.Zero(t, count)
}
