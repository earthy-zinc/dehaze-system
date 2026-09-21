package aidomain

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// esRecorder 记录 ES 请求，供「删除记忆必须清向量文档」断言使用
type esRecorder struct {
	server   *httptest.Server
	status   int
	method   string
	path     string
	query    string
	auth     string
	hitCount int
}

func newESRecorder(status int) *esRecorder {
	recorder := &esRecorder{status: status}
	recorder.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		recorder.method = req.Method
		recorder.path = req.URL.Path
		recorder.query = req.URL.RawQuery
		recorder.auth = req.Header.Get("Authorization")
		recorder.hitCount++
		w.WriteHeader(recorder.status)
	}))
	return recorder
}

// useAsES 将全局 ES 配置指向本记录器（与 python/java 共用同一 ES_* 配置源）
func (r *esRecorder) useAsES(t *testing.T) {
	t.Helper()
	cfg := testutil.LoadTestConfig(t)
	cfg.ES.URL = r.server.URL
	cfg.ES.Username = "elastic"
	cfg.ES.Password = "Dehaze2026"
}

func seedMemory(t *testing.T, svc *MemoryService, userID int64) *model.SysAiMemory {
	t.Helper()
	memory := &model.SysAiMemory{
		UserID:     userID,
		MemoryType: "semantic",
		Content:    "用户偏好简洁回复",
		Metadata:   "{}",
		Importance: 50,
		Source:     "manual",
		Status:     1,
		CreateTime: time.Now(),
	}
	require.NoError(t, svc.memories.Create(context.Background(), memory))
	return memory
}

func requireSoftDeleted(t *testing.T, db *gorm.DB, id int64) {
	t.Helper()
	var deleted int64
	require.NoError(t, db.Table("sys_ai_memory").Select("deleted").
		Where("id = ?", id).Scan(&deleted).Error)
	require.NotZero(t, deleted, "DB 软删标记必须写入（30 天恢复窗口）")
}

// TestDeleteRemovesEsVectorDoc 删除记忆必须同步删除 ES 向量文档：索引 ai_memory、_id 为记忆 id
// （对齐 python ai_memory_index.delete_memory_doc / es_client.delete_doc）。残留文档会让已删
// 记忆仍被向量检索召回，属隐私缺陷。
func TestDeleteRemovesEsVectorDoc(t *testing.T) {
	es := newESRecorder(http.StatusOK)
	defer es.server.Close()
	es.useAsES(t)

	db := testutil.NewTestDB(t)
	svc := NewMemoryService(repo.NewMemoryRepository(db), nil)
	ctx := context.Background()
	memory := seedMemory(t, svc, 1001)

	require.NoError(t, svc.Delete(ctx, memory.ID, 1001))

	require.Equal(t, 1, es.hitCount, "删除记忆必须发起一次 ES 文档删除")
	require.Equal(t, http.MethodDelete, es.method)
	require.Equal(t, fmt.Sprintf("/ai_memory/_doc/%d", memory.ID), es.path)
	require.Equal(t, "refresh=true", es.query)
	require.Equal(t, "Basic "+base64.StdEncoding.EncodeToString([]byte("elastic:Dehaze2026")), es.auth)

	requireSoftDeleted(t, db, memory.ID)
}

// TestDeleteToleratesMissingEsDoc 文档本就不存在（404）视为已清理，不影响删除主流程
func TestDeleteToleratesMissingEsDoc(t *testing.T) {
	es := newESRecorder(http.StatusNotFound)
	defer es.server.Close()
	es.useAsES(t)

	db := testutil.NewTestDB(t)
	svc := NewMemoryService(repo.NewMemoryRepository(db), nil)
	memory := seedMemory(t, svc, 1002)

	require.NoError(t, svc.Delete(context.Background(), memory.ID, 1002))
	require.Equal(t, 1, es.hitCount)
}

// TestDeleteSucceedsWhenEsUnreachable ES 不可达只告警：DB 已软删，不因检索侧清理失败回滚删除
// （对齐 python es_client.delete_doc 的降级：捕获异常记 warning 并返回 False）。
func TestDeleteSucceedsWhenEsUnreachable(t *testing.T) {
	es := newESRecorder(http.StatusOK)
	es.useAsES(t)
	// 关停 ES：连接被拒
	es.server.Close()

	db := testutil.NewTestDB(t)
	svc := NewMemoryService(repo.NewMemoryRepository(db), nil)
	memory := seedMemory(t, svc, 1003)

	require.NoError(t, svc.Delete(context.Background(), memory.ID, 1003))

	requireSoftDeleted(t, db, memory.ID)
}
