package kb

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	kborepo "github.com/earthyzinc/dehaze-go/internal/repository/kb"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// TestListChunks 文档分块列表：chunk_index 正序分页 + 字段映射（含 metadata JSON 解析）
func TestListChunks(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewService(kborepo.NewRepository(db))
	ctx := context.Background()

	owner := int64(1001)
	kb := seedKnowledgeBase(t, db, owner, "private")
	doc := seedDocument(t, db, kb.ID)

	// 乱序写入，验证读取按 chunk_index 正序（python order_by chunk_index asc）
	seedChunk(t, db, kb.ID, doc.ID, 2, "第三块", 30, nil)
	seedChunk(t, db, kb.ID, doc.ID, 0, "第一块", 10, json.RawMessage(`{"page":1,"section":"4.2"}`))
	seedChunk(t, db, kb.ID, doc.ID, 1, "第二块", 20, json.RawMessage(`{"page":2}`))

	page, err := svc.ListChunks(ctx, doc.ID, owner, 1, 2)
	require.NoError(t, err)
	require.EqualValues(t, 3, page.Total)
	require.Len(t, page.List, 2)
	require.Equal(t, 0, page.List[0].ChunkIndex)
	require.Equal(t, "第一块", page.List[0].Content)
	require.EqualValues(t, 10, page.List[0].TokenCount)
	require.Equal(t, doc.ID, page.List[0].DocumentID)
	require.NotNil(t, page.List[0].CreateTime)
	require.JSONEq(t, `{"page":1,"section":"4.2"}`, string(page.List[0].Metadata))
	require.Equal(t, 1, page.List[1].ChunkIndex)

	// 第二页
	page, err = svc.ListChunks(ctx, doc.ID, owner, 2, 2)
	require.NoError(t, err)
	require.EqualValues(t, 3, page.Total)
	require.Len(t, page.List, 1)
	require.Equal(t, 2, page.List[0].ChunkIndex)
	require.Nil(t, page.List[0].Metadata, "metadata 为 NULL 时不下发")

	// 分页缺省与越界一律由 handler 的 parsePaginationWithSize 裁决（拒绝集/放行集见 api 包扫掠用例），
	// service 只按透传值分页：此处用合法上限 pageSize=100 验证可一次取回全部 3 块
	page, err = svc.ListChunks(ctx, doc.ID, owner, 1, 100)
	require.NoError(t, err)
	require.Len(t, page.List, 3)
}

// TestListChunksVisibility 私有库文档分块仅 owner 可读（越权 A0301）
func TestListChunksVisibility(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewService(kborepo.NewRepository(db))
	ctx := context.Background()

	owner := int64(1001)
	stranger := int64(2002)
	privateKB := seedKnowledgeBase(t, db, owner, "private")
	doc := seedDocument(t, db, privateKB.ID)
	seedChunk(t, db, privateKB.ID, doc.ID, 0, "私有分块", 10, nil)

	_, err := svc.ListChunks(ctx, doc.ID, stranger, 1, 10)
	requireBizCode(t, err, common.ACCESS_UNAUTHORIZED)

	// 公开库他人可读
	publicKB := seedKnowledgeBase(t, db, owner, "public")
	publicDoc := seedDocument(t, db, publicKB.ID)
	seedChunk(t, db, publicKB.ID, publicDoc.ID, 0, "公开分块", 10, nil)
	page, err := svc.ListChunks(ctx, publicDoc.ID, stranger, 1, 10)
	require.NoError(t, err)
	require.EqualValues(t, 1, page.Total)
	require.Equal(t, "公开分块", page.List[0].Content)
}

// TestListChunksDocumentMissing 文档不存在报 A0401
func TestListChunksDocumentMissing(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewService(kborepo.NewRepository(db))

	_, err := svc.ListChunks(context.Background(), 999999999, 1001, 1, 10)
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)
}

// TestListChunksKnowledgeBaseMissing 文档存在但所属知识库已被清理时报 A0401（与 python _check_doc_readable 一致）
func TestListChunksKnowledgeBaseMissing(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewService(kborepo.NewRepository(db))
	ctx := context.Background()

	doc := seedDocument(t, db, 999999999)
	_, err := svc.ListChunks(ctx, doc.ID, 1001, 1, 10)
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)
}

// TestUpdateKnowledgeBaseRejectsOutOfRange 通用编辑表单的取值校验：python KnowledgeBaseUpdateForm
// 由 pydantic 拦截（top_k 1~100 / score_threshold [0,1) / hybrid_weight 0~1 /
// search_strategy Literal / name 1~255 / rerank_model ≤64 / embedding_model ≤64 /
// chunking_strategy 五值），go 侧须在 service 层等价拦截。
func TestUpdateKnowledgeBaseRejectsOutOfRange(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewService(kborepo.NewRepository(db))
	ctx := context.Background()
	owner := int64(1001)
	kb := seedKnowledgeBase(t, db, owner, "private")

	topKTooLarge, topKZero := 101, 0
	thresholdOne, thresholdNegative := 1.0, -0.1
	weightTooLarge := 1.5
	strategyInvalid := "fuzzy"
	emptyName := "   "
	longRerank := strings.Repeat("r", 65)
	longName := strings.Repeat("n", 256)
	longEmbedding := strings.Repeat("e", 65)
	// chunking 非法字面量：python 侧由 Literal 在 pydantic 阶段拦截（A0400），
	// 不是"合法值但创建后不可改"那条 A0500 路径。
	chunkingInvalid, chunkingLegal := "bogus", "fixed"

	cases := []struct {
		name string
		form *bo.KnowledgeBaseUpdateForm
	}{
		{"topK 超上限", &bo.KnowledgeBaseUpdateForm{TopK: &topKTooLarge}},
		{"topK 为 0", &bo.KnowledgeBaseUpdateForm{TopK: &topKZero}},
		{"scoreThreshold 等于 1", &bo.KnowledgeBaseUpdateForm{ScoreThreshold: &thresholdOne}},
		{"scoreThreshold 为负", &bo.KnowledgeBaseUpdateForm{ScoreThreshold: &thresholdNegative}},
		{"hybridWeight 超上限", &bo.KnowledgeBaseUpdateForm{HybridWeight: &weightTooLarge}},
		{"searchStrategy 非法", &bo.KnowledgeBaseUpdateForm{SearchStrategy: &strategyInvalid}},
		{"name 仅空白", &bo.KnowledgeBaseUpdateForm{Name: &emptyName}},
		{"name 超长", &bo.KnowledgeBaseUpdateForm{Name: &longName}},
		{"rerankModel 超长", &bo.KnowledgeBaseUpdateForm{RerankModel: &longRerank}},
		{"embeddingModel 超长", &bo.KnowledgeBaseUpdateForm{EmbeddingModel: &longEmbedding}},
		{"chunkingStrategy 非法字面量", &bo.KnowledgeBaseUpdateForm{ChunkingStrategy: &chunkingInvalid}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := svc.UpdateKnowledgeBase(ctx, kb.ID, owner, false, c.form)
			requireBizCode(t, err, common.PARAM_ERROR)
		})
	}

	// 校验先于查库：知识库不存在时，非法取值仍报 A0400（对齐 python pydantic 先于 404/权限）
	err := svc.UpdateKnowledgeBase(ctx, kb.ID+999999, owner, false,
		&bo.KnowledgeBaseUpdateForm{ChunkingStrategy: &chunkingInvalid})
	requireBizCode(t, err, common.PARAM_ERROR)

	// 合法值但携带不可修改项 → 业务错误 A0500（python 同理，且晚于权限校验）
	err = svc.UpdateKnowledgeBase(ctx, kb.ID, owner, false,
		&bo.KnowledgeBaseUpdateForm{ChunkingStrategy: &chunkingLegal})
	requireBizCode(t, err, common.BUSINESS_ERROR)

	// 越界值不得落库（校验先于查询/写入）
	var stored struct {
		TopK           int     `gorm:"column:top_k"`
		ScoreThreshold float64 `gorm:"column:score_threshold"`
	}
	require.NoError(t, db.Table("sys_knowledge_base").Select("top_k, score_threshold").
		Where("id = ?", kb.ID).Scan(&stored).Error)
	require.Equal(t, kb.TopK, stored.TopK)
	require.Equal(t, kb.ScoreThreshold, stored.ScoreThreshold)

	// 合法值照常生效
	topK := 20
	threshold := 0.8
	weight := 0.4
	require.NoError(t, svc.UpdateKnowledgeBase(ctx, kb.ID, owner, false, &bo.KnowledgeBaseUpdateForm{
		TopK: &topK, ScoreThreshold: &threshold, HybridWeight: &weight,
	}))
	require.NoError(t, db.Table("sys_knowledge_base").Select("top_k, score_threshold").
		Where("id = ?", kb.ID).Scan(&stored).Error)
	require.Equal(t, 20, stored.TopK)
	require.InDelta(t, 0.8, stored.ScoreThreshold, 0.001)
}

func uniqueName(prefix string) string {
	return prefix + strconv.FormatInt(time.Now().UnixNano(), 10)
}

func seedKnowledgeBase(t *testing.T, db *gorm.DB, owner int64, visibility string) *model.SysKnowledgeBase {
	t.Helper()
	kb := &model.SysKnowledgeBase{
		Name: uniqueName("kb-"), Visibility: visibility, EmbeddingProvider: "local",
		EmbeddingModel: "qwen3-embedding-0.6b", ChunkingStrategy: "fixed",
		ChunkSize: 800, ChunkOverlap: 80, SearchStrategy: "hybrid", HybridWeight: 0.7,
		TopK: 5, ScoreThreshold: 0.5, Status: 1,
	}
	kb.CreateBy = owner
	require.NoError(t, db.Create(kb).Error)
	return kb
}

func seedDocument(t *testing.T, db *gorm.DB, kbID int64) *model.SysKnowledgeDocument {
	t.Helper()
	content := "文档正文"
	doc := &model.SysKnowledgeDocument{
		KnowledgeBaseID: kbID, Title: uniqueName("doc-"), Source: "manual", Version: 1,
		ParsingStrategy: "auto", Content: &content, ProcessingStatus: "completed",
	}
	require.NoError(t, db.Create(doc).Error)
	return doc
}

func seedChunk(
	t *testing.T, db *gorm.DB, kbID, documentID int64, chunkIndex int,
	content string, tokenCount int, metadata json.RawMessage,
) *model.SysKnowledgeChunk {
	t.Helper()
	chunk := &model.SysKnowledgeChunk{
		DocumentID: documentID, KnowledgeBaseID: kbID, ChunkIndex: chunkIndex,
		Content: content, TokenCount: tokenCount, Metadata: metadata,
	}
	require.NoError(t, db.Create(chunk).Error)
	return chunk
}

func requireBizCode(t *testing.T, err error, code *common.ResultCode) {
	t.Helper()
	require.Error(t, err)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "期望业务错误，实际: %v", err)
	require.Equal(t, code.Code, bizErr.Code().Code)
}
