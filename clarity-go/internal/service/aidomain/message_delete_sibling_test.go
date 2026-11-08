package aidomain

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/earthyzinc/clarity-go/internal/model/aicontract"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/earthyzinc/clarity-go/pkg/cache/redis"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/earthyzinc/clarity-go/pkg/config"
	"github.com/earthyzinc/clarity-go/pkg/config/options"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

// 删除 / 详情兄弟序号 / 切换分支中断校验回归。
//
// 删除策略：链尾删除重指指针（可继续对话且历史仍在）、非叶子拒绝（不断链）、
// 非激活链旧分支可删且指针不动。详情与列表同一口径回填 siblingIndex/siblingCount。
// 切换分支在指针上仍有可恢复中断时拒绝：消息的 pending_interrupt（DB）是事实源，
// Redis ai:interrupt:{conv}:{msg} 句柄是补充信号。

var (
	messageRedisOnce sync.Once
	messageRedisErr  error
)

// initMessageRedis 将全局 Redis 客户端指向进程内 miniredis（单测不触达真实 Redis），
// 使 SwitchBranch 的中断校验可读写同一份键。
func initMessageRedis(t *testing.T) *goredis.Client {
	t.Helper()
	messageRedisOnce.Do(func() {
		testutil.LoadTestConfig(t)
		mr, err := miniredis.Run()
		if err != nil {
			messageRedisErr = err
			return
		}
		config.Config.Cache = options.Cache{
			Type:  "redis",
			Redis: options.Redis{Enabled: true, Addr: mr.Addr()},
		}
		if _, err := redis.InitRedis(); err != nil {
			messageRedisErr = err
		}
	})
	require.NoError(t, messageRedisErr, "初始化测试 Redis 失败")
	client := redis.GetClient()
	require.NotNil(t, client, "Redis 客户端不可用")
	return client
}

// TestMessageDeleteTailRetargetsPointerAndKeepsHistory 删除激活链尾：指针回退到 parent，
// 继续发送后历史仍在（新消息挂在 parent 下，而不是新开一条链）。
func TestMessageDeleteTailRetargetsPointerAndKeepsHistory(t *testing.T) {
	f := newMessageChainFixture(t, 8201)
	ctx := context.Background()
	u1 := f.add(t, nil, aicontract.RoleUser)
	a1 := f.add(t, &u1.ID, aicontract.RoleAssistant)
	f.setBranch(t, a1.ID)

	require.NoError(t, f.service.Delete(ctx, a1.ID, f.userID))

	deleted, err := f.msgs.GetByID(ctx, a1.ID)
	require.NoError(t, err)
	require.Nil(t, deleted, "链尾消息必须被软删")

	conv, err := f.convs.GetByIDAndUser(ctx, f.convID, f.userID)
	require.NoError(t, err)
	require.NotNil(t, conv.CurrentBranchMessageID)
	require.Equal(t, u1.ID, *conv.CurrentBranchMessageID, "指针回退到 parent")

	// 继续发送：新 user 消息以当前指针为 parent（send 的落库口径），随后指针推进到新消息
	u2 := f.add(t, conv.CurrentBranchMessageID, aicontract.RoleUser)
	f.setBranch(t, u2.ID)

	res := f.list(t, nil, 50, false)
	require.Equal(t, []int64{u2.ID, u1.ID}, idsOf(res.List), "删除后继续对话，历史消息仍在")
	require.EqualValues(t, 2, res.Total)
}

// TestMessageDeleteRootAssistantClearsPointer 根级助手消息无 parent：删除后指针置空。
func TestMessageDeleteRootAssistantClearsPointer(t *testing.T) {
	f := newMessageChainFixture(t, 8202)
	ctx := context.Background()
	root := f.add(t, nil, aicontract.RoleAssistant)
	f.setBranch(t, root.ID)

	require.NoError(t, f.service.Delete(ctx, root.ID, f.userID))

	conv, err := f.convs.GetByIDAndUser(ctx, f.convID, f.userID)
	require.NoError(t, err)
	require.Nil(t, conv.CurrentBranchMessageID, "parent 为 NULL 时指针必须置空")

	res := f.list(t, nil, 50, false)
	require.Empty(t, res.List, "删掉唯一消息后激活链为空")
}

// TestMessageDeleteMiddleRejected 非叶子拒绝删除：返回明确业务错误，消息保留、链完整。
func TestMessageDeleteMiddleRejected(t *testing.T) {
	f := newMessageChainFixture(t, 8203)
	ctx := context.Background()
	u1 := f.add(t, nil, aicontract.RoleUser)
	a1 := f.add(t, &u1.ID, aicontract.RoleAssistant)
	u2 := f.add(t, &a1.ID, aicontract.RoleUser)
	a2 := f.add(t, &u2.ID, aicontract.RoleAssistant)
	f.setBranch(t, a2.ID)

	err := f.service.Delete(ctx, a1.ID, f.userID)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际 err=%v", err)
	require.Equal(t, common.DATA_STATE_NOT_ALLOW.Code, bizErr.Code().Code)
	require.Contains(t, bizErr.Message(), "请先删除该回复之后的对话")

	kept, err := f.msgs.GetByID(ctx, a1.ID)
	require.NoError(t, err)
	require.NotNil(t, kept, "拒绝删除时消息必须保留")

	res := f.list(t, nil, 50, false)
	require.Equal(t, []int64{a2.ID, u2.ID, a1.ID, u1.ID}, idsOf(res.List), "链不得被截断")
	require.EqualValues(t, 4, res.Total)
}

// TestMessageDeleteOffChainBranchKeepsPointer 非激活链上的旧分支可删，指针不动。
func TestMessageDeleteOffChainBranchKeepsPointer(t *testing.T) {
	f := newMessageChainFixture(t, 8204)
	ctx := context.Background()
	u1 := f.add(t, nil, aicontract.RoleUser)
	active := f.add(t, &u1.ID, aicontract.RoleAssistant)
	offChain := f.add(t, &u1.ID, aicontract.RoleAssistant)
	f.setBranch(t, active.ID)

	require.NoError(t, f.service.Delete(ctx, offChain.ID, f.userID))

	conv, err := f.convs.GetByIDAndUser(ctx, f.convID, f.userID)
	require.NoError(t, err)
	require.NotNil(t, conv.CurrentBranchMessageID)
	require.Equal(t, active.ID, *conv.CurrentBranchMessageID, "旧分支删除不得改指针")

	res := f.list(t, nil, 50, false)
	require.Equal(t, []int64{active.ID, u1.ID}, idsOf(res.List))
}

// TestMessageGetIncludesSiblingPosition 详情回填兄弟序号：与列表同口径（同父未删 assistant、
// id 升序、1 起），root 级与非 assistant 保持 null，软删兄弟不占位。
func TestMessageGetIncludesSiblingPosition(t *testing.T) {
	f := newMessageChainFixture(t, 8205)
	ctx := context.Background()
	userMsg := f.add(t, nil, aicontract.RoleUser)
	f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	deleted := f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	second := f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	f.setBranch(t, second.ID)
	require.NoError(t, f.msgs.SoftDeleteByIDs(ctx, []int64{deleted.ID}, f.userID))

	detail, err := f.service.Get(ctx, second.ID, f.userID, false)
	require.NoError(t, err)
	require.NotNil(t, detail.SiblingIndex)
	require.EqualValues(t, 2, *detail.SiblingIndex, "软删兄弟不占位，序号从 1 连续")
	require.NotNil(t, detail.SiblingCount)
	require.EqualValues(t, 3, *detail.SiblingCount, "软删兄弟不计入总数")

	t.Run("非 assistant 为 null", func(t *testing.T) {
		userDetail, err := f.service.Get(ctx, userMsg.ID, f.userID, false)
		require.NoError(t, err)
		require.Nil(t, userDetail.SiblingIndex)
		require.Nil(t, userDetail.SiblingCount)
	})

	t.Run("root 级 assistant 为 null", func(t *testing.T) {
		root := f.add(t, nil, aicontract.RoleAssistant)
		rootDetail, err := f.service.Get(ctx, root.ID, f.userID, false)
		require.NoError(t, err)
		require.Nil(t, rootDetail.SiblingIndex)
		require.Nil(t, rootDetail.SiblingCount)
	})
}

// TestSwitchBranchRejectsPendingInterrupt 当前指针存在可恢复中断（Redis 句柄）时拒绝切换；
// 中断清除后允许切换，指针按切换目标更新。
func TestSwitchBranchRejectsPendingInterrupt(t *testing.T) {
	f := newMessageChainFixture(t, 8206)
	client := initMessageRedis(t)
	ctx := context.Background()
	u1 := f.add(t, nil, aicontract.RoleUser)
	active := f.add(t, &u1.ID, aicontract.RoleAssistant)
	other := f.add(t, &u1.ID, aicontract.RoleAssistant)
	f.setBranch(t, active.ID)

	key := fmt.Sprintf(interruptKeyFmt, f.convID, active.ID)
	require.NoError(t, client.Set(ctx, key, `{"type":"confirm","data":{}}`, 0).Err())
	t.Cleanup(func() { _ = client.Del(context.Background(), key).Err() })

	_, err := f.service.SwitchBranch(ctx, f.convID, other.ID, f.userID)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际 err=%v", err)
	require.Equal(t, common.BUSINESS_ERROR.Code, bizErr.Code().Code)
	require.Contains(t, bizErr.Message(), "未完成的中断确认")

	conv, err := f.convs.GetByIDAndUser(ctx, f.convID, f.userID)
	require.NoError(t, err)
	require.NotNil(t, conv.CurrentBranchMessageID)
	require.Equal(t, active.ID, *conv.CurrentBranchMessageID, "拒绝切换时指针不得变动")

	// 中断清除（用户确认或停止）后恢复可切换
	require.NoError(t, client.Del(ctx, key).Err())
	_, err = f.service.SwitchBranch(ctx, f.convID, other.ID, f.userID)
	require.NoError(t, err)
	conv, err = f.convs.GetByIDAndUser(ctx, f.convID, f.userID)
	require.NoError(t, err)
	require.NotNil(t, conv.CurrentBranchMessageID)
	require.Equal(t, other.ID, *conv.CurrentBranchMessageID)
}

// TestSwitchBranchRejectsDBPendingInterrupt pending_interrupt 以 DB 为事实源：
// Redis 无句柄但当前指针消息 actionable（status=awaiting_input 且未过期）时拒绝切换；
// 过期或状态已不是 awaiting_input 时不再 actionable，放行。
func TestSwitchBranchRejectsDBPendingInterrupt(t *testing.T) {
	f := newMessageChainFixture(t, 8207)
	client := initMessageRedis(t)
	ctx := context.Background()
	u1 := f.add(t, nil, aicontract.RoleUser)
	active := f.add(t, &u1.ID, aicontract.RoleAssistant)
	other := f.add(t, &u1.ID, aicontract.RoleAssistant)
	f.setBranch(t, active.ID)
	// Redis 侧显式无句柄：拒绝必须由 DB 事实源独立给出
	require.NoError(t, client.Del(ctx, fmt.Sprintf(interruptKeyFmt, f.convID, active.ID)).Err())

	f.setInterrupt(t, active.ID, aicontract.MessageStatusAwaitingInput,
		interruptPayload(t, "it-db-live", time.Now().Add(time.Hour)))

	_, err := f.service.SwitchBranch(ctx, f.convID, other.ID, f.userID)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际 err=%v", err)
	require.Equal(t, common.BUSINESS_ERROR.Code, bizErr.Code().Code, "拒绝切换仍返回 A0500")
	require.Contains(t, bizErr.Message(), "未完成的中断确认")

	conv, err := f.convs.GetByIDAndUser(ctx, f.convID, f.userID)
	require.NoError(t, err)
	require.NotNil(t, conv.CurrentBranchMessageID)
	require.Equal(t, active.ID, *conv.CurrentBranchMessageID, "拒绝切换时指针不得变动")

	t.Run("过期中断放行", func(t *testing.T) {
		f.setInterrupt(t, active.ID, aicontract.MessageStatusAwaitingInput,
			interruptPayload(t, "it-db-expired", time.Now().Add(-time.Minute)))
		_, err := f.service.SwitchBranch(ctx, f.convID, other.ID, f.userID)
		require.NoError(t, err, "过期中断不再 actionable，不阻断切换")
	})

	t.Run("非 awaiting_input 残留放行", func(t *testing.T) {
		f.setBranch(t, active.ID)
		f.setInterrupt(t, active.ID, aicontract.MessageStatusCompleted,
			interruptPayload(t, "it-db-stale", time.Now().Add(time.Hour)))
		_, err := f.service.SwitchBranch(ctx, f.convID, other.ID, f.userID)
		require.NoError(t, err, "消息已不是 awaiting_input，残留 payload 不阻断切换")
	})
}

// TestMessageDeleteAfterSendAdvancedPointerRejected 顺序模拟并发 send：新消息已作为旧尾的
// 子消息落库、指针已推进到新尾，此时再删旧尾必须在事务内被判定为非叶子并拒绝——
// 既不能软删旧尾（会删断新消息的父链），也不能把指针回退覆盖新尾。
func TestMessageDeleteAfterSendAdvancedPointerRejected(t *testing.T) {
	f := newMessageChainFixture(t, 8208)
	ctx := context.Background()
	u1 := f.add(t, nil, aicontract.RoleUser)
	a1 := f.add(t, &u1.ID, aicontract.RoleAssistant)
	f.setBranch(t, a1.ID)

	// send 完成：新 user 消息挂到旧尾、指针推进到它
	u2 := f.add(t, &a1.ID, aicontract.RoleUser)
	f.setBranch(t, u2.ID)

	err := f.service.Delete(ctx, a1.ID, f.userID)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际 err=%v", err)
	require.Equal(t, common.DATA_STATE_NOT_ALLOW.Code, bizErr.Code().Code)
	require.Contains(t, bizErr.Message(), "请先删除该回复之后的对话")

	conv, err := f.convs.GetByIDAndUser(ctx, f.convID, f.userID)
	require.NoError(t, err)
	require.NotNil(t, conv.CurrentBranchMessageID)
	require.Equal(t, u2.ID, *conv.CurrentBranchMessageID, "指针保持在新尾，不得被旧尾回退覆盖")

	kept, err := f.msgs.GetByID(ctx, a1.ID)
	require.NoError(t, err)
	require.NotNil(t, kept, "非叶子旧尾不得被软删")
}

// TestMessageBranchesExcludeUserChildren 删尾后继续发送会让 user 消息与 assistant 分支同父。
// 分支列表只返回 assistant：user 子消息不参与切换，也不占用兄弟序号；但删除校验仍看全部
// 子消息——user 后续提问同样让消息变成非叶子。
func TestMessageBranchesExcludeUserChildren(t *testing.T) {
	f := newMessageChainFixture(t, 8209)
	ctx := context.Background()
	u1 := f.add(t, nil, aicontract.RoleUser)
	a1 := f.add(t, &u1.ID, aicontract.RoleAssistant)
	f.setBranch(t, a1.ID)
	require.NoError(t, f.service.Delete(ctx, a1.ID, f.userID), "先删链尾，指针回退到 u1")

	// 删尾后继续发送：新 user 提问挂到 u1，重新生成的 assistant 成为同父分支
	f.add(t, &u1.ID, aicontract.RoleUser)
	branch := f.add(t, &u1.ID, aicontract.RoleAssistant)
	f.setBranch(t, branch.ID)

	branches, err := f.service.GetBranches(ctx, f.convID, u1.ID, f.userID)
	require.NoError(t, err)
	require.Equal(t, []int64{branch.ID}, idsOf(branches), "分支列表不得混入 user 子消息")

	// 兄弟序号与分支列表同口径：user 子消息不计入
	res := f.list(t, nil, 50, false)
	require.Equal(t, []int64{branch.ID, u1.ID}, idsOf(res.List))
	require.NotNil(t, res.List[0].SiblingIndex)
	require.EqualValues(t, 1, *res.List[0].SiblingIndex)
	require.NotNil(t, res.List[0].SiblingCount)
	require.EqualValues(t, 1, *res.List[0].SiblingCount)

	// assistant 下的 user 后续提问不是可切换分支，但仍算子消息
	f.add(t, &branch.ID, aicontract.RoleUser)
	branchChildren, err := f.service.GetBranches(ctx, f.convID, branch.ID, f.userID)
	require.NoError(t, err)
	require.Empty(t, branchChildren, "assistant 下的 user 子消息不是分支")

	err = f.service.Delete(ctx, branch.ID, f.userID)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际 err=%v", err)
	require.Equal(t, common.DATA_STATE_NOT_ALLOW.Code, bizErr.Code().Code)
	require.Contains(t, bizErr.Message(), "请先删除该回复之后的对话", "user 子消息同样阻止删除")
}
