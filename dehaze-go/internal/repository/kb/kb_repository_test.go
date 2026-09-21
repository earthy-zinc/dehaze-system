package kb

import (
	"context"
	"encoding/json"
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

func TestKnowledgeBaseRepositoryVisibilityAndSearch(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	repo := NewRepository(db)

	owner := int64(1001)
	stranger := int64(2002)
	privateKB := &model.SysKnowledgeBase{
		Name: uniqueName("private-"), Visibility: "private", EmbeddingProvider: "openai",
		EmbeddingModel: "text-embedding-3-small", ChunkingStrategy: "semantic",
		ChunkSize: 800, ChunkOverlap: 80, SearchStrategy: "hybrid", HybridWeight: 0.7,
		TopK: 5, ScoreThreshold: 0.5, Status: 1,
	}
	privateKB.CreateBy = owner
	require.NoError(t, db.Create(privateKB).Error)

	// 私有库仅 owner 可见
	ownerItems, _, err := repo.PaginateVisible(ctx, owner, "", 1, 10)
	require.NoError(t, err)
	require.True(t, containsKB(ownerItems, privateKB.ID))

	strangerItems, _, err := repo.PaginateVisible(ctx, stranger, "", 1, 10)
	require.NoError(t, err)
	require.False(t, containsKB(strangerItems, privateKB.ID), "私有库不应出现在他人可见列表")

	// 管理端全量视角可见
	allItems, _, err := repo.PaginateAll(ctx, "", 1, 50)
	require.NoError(t, err)
	require.True(t, containsKB(allItems, privateKB.ID))

	// 关键字 LIKE 转义：% 只匹配字面量
	percentKB := &model.SysKnowledgeBase{
		Name: "含%百分号-" + uniqueName(""), Visibility: "public", EmbeddingProvider: "openai",
		EmbeddingModel: "text-embedding-3-small", ChunkingStrategy: "semantic",
		ChunkSize: 800, ChunkOverlap: 80, SearchStrategy: "hybrid", HybridWeight: 0.7, TopK: 5, ScoreThreshold: 0.5, Status: 1,
	}
	percentKB.CreateBy = owner
	require.NoError(t, db.Create(percentKB).Error)

	matched, total, err := repo.PaginateAll(ctx, "%", 1, 50)
	require.NoError(t, err)
	require.EqualValues(t, 1, total)
	require.Len(t, matched, 1)
	require.Equal(t, percentKB.ID, matched[0].ID)

	// 同 owner 名称查重
	found, err := repo.GetByNameAndOwner(ctx, privateKB.Name, owner)
	require.NoError(t, err)
	require.NotNil(t, found)
	missing, err := repo.GetByNameAndOwner(ctx, privateKB.Name, stranger)
	require.NoError(t, err)
	require.Nil(t, missing)

	count, err := repo.CountPrivateByOwner(ctx, owner)
	require.NoError(t, err)
	require.EqualValues(t, 1, count)

	// 更新可编辑字段
	require.NoError(t, repo.UpdateKnowledgeBase(ctx, privateKB.ID, map[string]interface{}{"top_k": 9, "update_by": owner}))
	updated, err := repo.GetKnowledgeBase(ctx, privateKB.ID)
	require.NoError(t, err)
	require.Equal(t, 9, updated.TopK)
}

func TestDocumentTestSetAndLowQualityQueries(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	repo := NewRepository(db)

	kb := &model.SysKnowledgeBase{
		Name: uniqueName("kb-"), Visibility: "private", EmbeddingProvider: "local",
		EmbeddingModel: "qwen3-embedding-0.6b", ChunkingStrategy: "fixed",
		ChunkSize: 800, ChunkOverlap: 80, SearchStrategy: "hybrid", HybridWeight: 0.7,
		TopK: 5, ScoreThreshold: 0.5, Status: 1,
	}
	kb.CreateBy = 7
	require.NoError(t, db.Create(kb).Error)

	content := "文档正文"
	doc := &model.SysKnowledgeDocument{
		KnowledgeBaseID: kb.ID, Title: "测试文档", Source: "manual", Version: 1,
		ParsingStrategy: "auto", Content: &content, ProcessingStatus: "completed",
	}
	require.NoError(t, repo.CreateDocument(ctx, doc))
	require.NotZero(t, doc.ID)

	docs, total, err := repo.PaginateDocuments(ctx, kb.ID, "", 1, 10)
	require.NoError(t, err)
	require.EqualValues(t, 1, total)
	require.Len(t, docs, 1)

	// 处理状态过滤
	_, pendingTotal, err := repo.PaginateDocuments(ctx, kb.ID, "pending", 1, 10)
	require.NoError(t, err)
	require.EqualValues(t, 0, pendingTotal)

	fetched, err := repo.GetDocument(ctx, doc.ID)
	require.NoError(t, err)
	require.NotNil(t, fetched)
	require.Equal(t, "测试文档", fetched.Title)

	// 召回测试集
	chunkIDs := []int64{11, 12}
	payload, err := json.Marshal(chunkIDs)
	require.NoError(t, err)
	testSet := &model.SysKnowledgeTestSet{KnowledgeBaseID: kb.ID, Question: "如何配置？", ExpectedChunkIds: payload}
	require.NoError(t, repo.CreateTestSet(ctx, testSet))

	testSets, tsTotal, err := repo.PaginateTestSets(ctx, kb.ID, 1, 10)
	require.NoError(t, err)
	require.EqualValues(t, 1, tsTotal)
	require.Len(t, testSets, 1)

	// 低质量片段：仅统计 rating=-1 且属于该库的分块，按点踩次数降序
	require.NoError(t, db.Exec(
		"INSERT INTO sys_knowledge_chunk (id, document_id, knowledge_base_id, chunk_index, content, token_count) VALUES (?, ?, ?, ?, ?, ?)",
		chunkIDs[0], doc.ID, kb.ID, 0, "低质量片段", 10).Error)
	// 唯一键为 (chunk_id, user_id)，点踩两次须来自两个不同用户
	for _, uid := range []int{7, 9} {
		require.NoError(t, db.Exec(
			"INSERT INTO sys_knowledge_chunk_feedback (chunk_id, user_id, rating) VALUES (?, ?, ?)",
			chunkIDs[0], uid, -1).Error)
	}
	// 点赞（rating=1）不应计入
	require.NoError(t, db.Exec(
		"INSERT INTO sys_knowledge_chunk_feedback (chunk_id, user_id, rating) VALUES (?, ?, ?)",
		chunkIDs[0], 8, 1).Error)

	rows, lowTotal, err := repo.ListLowQuality(ctx, kb.ID, 1, 10)
	require.NoError(t, err)
	require.EqualValues(t, 1, lowTotal, "按分块去重计数")
	require.Len(t, rows, 1)
	require.EqualValues(t, 2, rows[0].ThumbsDownCount)
	require.Equal(t, "低质量片段", rows[0].Content)
	require.Equal(t, doc.ID, rows[0].DocumentID)
}

func containsKB(items []model.SysKnowledgeBase, id int64) bool {
	for i := range items {
		if items[i].ID == id {
			return true
		}
	}
	return false
}
