package testutil

import (
	"strconv"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/stretchr/testify/require"
)

// TestNewTestDB_WriteRollback 验证 NewTestDB 的写入-回滚语义：事务内写入可见，
// 回滚后（模拟 t.Cleanup 的 Rollback）数据不真正落库，新事务查不到。
// 注意：该自测同样会触发进程级库重建，故只做最轻量的行为验证。
func TestNewTestDB_WriteRollback(t *testing.T) {
	username := "testutil_rollback_" + strconv.FormatInt(time.Now().UnixNano(), 10)

	db := NewTestDB(t)
	u := &model.SysUser{Username: username, Nickname: "回滚验证"}
	require.NoError(t, db.Create(u).Error)

	var count int64
	require.NoError(t, db.Model(&model.SysUser{}).Where("username = ?", username).Count(&count).Error)
	require.Equal(t, int64(1), count, "写入应对当前事务可见")

	// 手动回滚，模拟 t.Cleanup 的行为（cleanup 会再次 Rollback，重复回滚无害）
	require.NoError(t, db.Rollback().Error)

	db2 := NewTestDB(t)
	var count2 int64
	require.NoError(t, db2.Model(&model.SysUser{}).Where("username = ?", username).Count(&count2).Error)
	require.Zero(t, count2, "回滚后数据不应落库")
}
