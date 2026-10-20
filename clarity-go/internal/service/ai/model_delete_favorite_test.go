package ai

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/bo"
	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	favrepo "github.com/earthyzinc/clarity-go/internal/repository/favorite"
	memberrepo "github.com/earthyzinc/clarity-go/internal/repository/member"
	favoriteservice "github.com/earthyzinc/clarity-go/internal/service/favorite"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// newTestFavoriteService 只覆盖 model 目标：memberRepo 供容量读取、modelRepo 供存在性校验，
// 数据集/处理记录目标不在本用例范围
func newTestFavoriteService(db *gorm.DB) favoriteservice.IFavoriteService {
	return favoriteservice.NewFavoriteService(
		favrepo.NewFavoriteRepository(db),
		memberrepo.NewMemberRepository(db),
		airepo.NewModelRepository(db),
		nil,
		nil,
		nil,
	)
}

// TestDeleteModelMarksFavoritesInvalid 模型删除必须与数据集删除同口径联动收藏失效：
// 收藏行 is_invalid=1；已删模型不可再次收藏（存在性校验按 deleted=0 查主键 → A0401）。
func TestDeleteModelMarksFavoritesInvalid(t *testing.T) {
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	ctx := context.Background()

	favSvc := newTestFavoriteService(db)
	svc := NewModelService(
		db, airepo.NewModelRepository(db), airepo.NewModelPriceRepository(db), nil, nil, favSvc, nil,
	)

	modelID := fmt.Sprintf("delete_favorite_%d", time.Now().UnixNano())
	created, err := svc.CreateModel(ctx, &bo.AiModelCreateForm{
		ProviderID: 1, ModelID: modelID, ModelType: "image_edit", DisplayName: "待删除模型",
	}, 1)
	require.NoError(t, err)

	userID := int64(990001)
	favoriteID, err := favSvc.Add(ctx, userID, &bo.FavoriteForm{TargetType: "model", TargetID: created.ID})
	require.NoError(t, err)

	require.NoError(t, svc.DeleteModel(ctx, modelID, 1))

	var favorite model.SysFavorite
	require.NoError(t, db.Where("id = ?", favoriteID).First(&favorite).Error)
	require.EqualValues(t, 1, favorite.IsInvalid, "模型删除后收藏必须标记失效")

	_, err = favSvc.Add(ctx, userID, &bo.FavoriteForm{TargetType: "model", TargetID: created.ID})
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "已删除模型再次收藏应报业务错误，实际 err=%v", err)
	require.Equal(t, common.RESOURCE_NOT_FOUND, bizErr.Code(), "已删除模型再次收藏应为 A0401")
}
