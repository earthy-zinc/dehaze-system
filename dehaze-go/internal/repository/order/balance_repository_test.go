package order

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// seedBalanceAccount 建一个干净账户（NewTestDB 单事务回滚，无需手动清理）
func seedBalanceAccount(t *testing.T, db *gorm.DB, userID, balance int64) {
	t.Helper()
	require.NoError(t, db.Where("user_id = ?", userID).Delete(&model.SysBalance{}).Error)
	require.NoError(t, db.Create(&model.SysBalance{UserID: userID, Balance: balance}).Error)
}

func TestAdjustBalanceRelativeUpdate(t *testing.T) {
	db := testutil.NewTestDB(t)
	repo := NewBalanceAccountRepository(db)
	ctx := context.Background()
	userID := int64(990001)
	seedBalanceAccount(t, db, userID, 1000)

	ok, err := repo.AdjustBalance(ctx, userID, -300)
	require.NoError(t, err)
	require.True(t, ok)

	ok, err = repo.AdjustBalance(ctx, userID, 50)
	require.NoError(t, err)
	require.True(t, ok)

	account, err := repo.GetOrCreate(ctx, userID)
	require.NoError(t, err)
	require.Equal(t, int64(750), account.Balance)
}

func TestAdjustBalanceRejectsOverdraft(t *testing.T) {
	db := testutil.NewTestDB(t)
	repo := NewBalanceAccountRepository(db)
	ctx := context.Background()
	userID := int64(990002)
	seedBalanceAccount(t, db, userID, 100)

	ok, err := repo.AdjustBalance(ctx, userID, -200)
	require.NoError(t, err)
	require.False(t, ok, "balance + delta < 0 时守卫应拒绝扣减")

	account, err := repo.GetOrCreate(ctx, userID)
	require.NoError(t, err)
	require.Equal(t, int64(100), account.Balance, "被拒绝的扣减不得改变余额")
}

func TestAdjustBalanceMissingAccountReturnsFalse(t *testing.T) {
	db := testutil.NewTestDB(t)
	repo := NewBalanceAccountRepository(db)

	ok, err := repo.AdjustBalance(context.Background(), 99999999, 100)
	require.NoError(t, err)
	require.False(t, ok, "账户不存在时 rowcount=0 → false")
}
