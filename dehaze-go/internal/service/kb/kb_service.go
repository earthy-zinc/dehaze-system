package kb

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	kborepo "github.com/earthyzinc/dehaze-go/internal/repository/kb"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	goredis "github.com/redis/go-redis/v9"
)

// 缓存约定（与 dehaze-python knowledge_base_service 一致）：
// kb:list:{user_id} / kb:list:admin 10min，kb:detail:{kb_id} 30min；
// 仅默认分页（无关键词/第一页/默认 size）读写缓存，避免不同 size 互相污染。
const (
	kbListTTL        = 600 * time.Second
	kbDetailTTL      = 1800 * time.Second
	defaultPageSize  = 10
	kbListAdminKey   = "kb:list:admin"
	kbAuditAdminView = "admin"
)

// 可编辑字段（分块策略与 embedding 模型创建后不可修改）
var knowledgeBaseEditableFields = []string{
	"name", "description", "search_strategy", "top_k",
	"score_threshold", "enable_rerank", "rerank_model", "hybrid_weight",
}

type Service struct {
	repo *kborepo.Repository
}

func NewService(repo *kborepo.Repository) *Service {
	return &Service{repo: repo}
}

func cacheClient() *goredis.Client {
	return redis.GetClient()
}

// ListKnowledgeBases 知识库列表（adminView=true 为管理端全量视角，权限由路由层校验）
func (s *Service) ListKnowledgeBases(ctx context.Context, userID int64, q *bo.KnowledgeBasePageQuery, adminView bool) (*vo.PageResult[vo.KnowledgeBaseVO], error) {
	page, size := q.PageNum, q.PageSize
	cacheKey := "kb:list:" + strconv.FormatInt(userID, 10)
	if adminView {
		cacheKey = kbListAdminKey
	}
	cacheable := q.Keyword == "" && page == 1 && size == defaultPageSize
	if cacheable {
		if cached := readCachedPage(ctx, cacheKey); cached != nil {
			return cached, nil
		}
	}

	var (
		items []model.SysKnowledgeBase
		total int64
		err   error
	)
	if adminView {
		items, total, err = s.repo.PaginateAll(ctx, q.Keyword, page, size)
	} else {
		items, total, err = s.repo.PaginateVisible(ctx, userID, q.Keyword, page, size)
	}
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询知识库列表失败", err)
	}
	rows := make([]vo.KnowledgeBaseVO, 0, len(items))
	for i := range items {
		rows = append(rows, toKnowledgeBaseVO(&items[i]))
	}
	result := &vo.PageResult[vo.KnowledgeBaseVO]{List: rows, Total: total}
	if cacheable {
		writeCachedPage(ctx, cacheKey, result, kbListTTL)
	}
	return result, nil
}

// GetKnowledgeBase 知识库详情（私有库仅 owner 可见；可见性校验先于缓存读取，避免越权读缓存）
func (s *Service) GetKnowledgeBase(ctx context.Context, kbID, userID int64) (*vo.KnowledgeBaseVO, error) {
	kb, err := s.repo.GetKnowledgeBase(ctx, kbID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询知识库失败", err)
	}
	if kb == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "知识库不存在")
	}
	if kb.Visibility == "private" && kb.CreateBy != userID {
		return nil, common.NewBizError(common.ACCESS_UNAUTHORIZED, "无权查看他人私有知识库")
	}
	detailKey := "kb:detail:" + strconv.FormatInt(kbID, 10)
	if client := cacheClient(); client != nil {
		if raw, readErr := client.Get(ctx, detailKey).Bytes(); readErr == nil && len(raw) > 0 {
			var cached vo.KnowledgeBaseVO
			if json.Unmarshal(raw, &cached) == nil {
				return &cached, nil
			}
		}
	}
	result := toKnowledgeBaseVO(kb)
	if client := cacheClient(); client != nil {
		if payload, marshalErr := json.Marshal(result); marshalErr == nil {
			client.Set(ctx, detailKey, payload, kbDetailTTL)
		}
	}
	return &result, nil
}

// validateUpdateForm 编辑表单取值校验：对齐 python KnowledgeBaseUpdateForm 的字段约束
// （name 1~255 / search_strategy Literal / hybrid_weight 0~1 / top_k 1~100 /
// score_threshold [0,1) / rerank_model ≤64 / embedding_model ≤64 / chunking_strategy 五值），
// 非法报 A0400。字段顺序与 python 表单声明顺序一致，保证"多字段同时非法时首个错误"也一致。
//
// 这些约束 python 由 pydantic 在请求校验阶段拦截，go 无对应层，必须在 service 层补齐：
// ① 六个可编辑字段否则越界值直接落库；
// ② embedding_model/chunking_strategy 虽无论合法与否都会被拒，但 python 的 pydantic 校验
// 先于业务拒绝，非法长度/非法字面量在 python 是 A0400（校验层）而 A0500 只留给"合法值但
// 创建后不可改"，故此处必须同样先判合法性，否则 go 会把"超长/拼错"也报成 A0500，误导调用方。
func validateUpdateForm(form *bo.KnowledgeBaseUpdateForm) error {
	if form.Name != nil && (strings.TrimSpace(*form.Name) == "" || len([]rune(*form.Name)) > 255) {
		return common.NewBizError(common.PARAM_ERROR, "name 长度需为 1~255")
	}
	if form.SearchStrategy != nil {
		switch *form.SearchStrategy {
		case "vector", "keyword", "hybrid":
		default:
			return common.NewBizError(common.PARAM_ERROR, "searchStrategy 仅支持 vector/keyword/hybrid")
		}
	}
	if form.HybridWeight != nil && (*form.HybridWeight < 0 || *form.HybridWeight > 1) {
		return common.NewBizError(common.PARAM_ERROR, "hybridWeight 取值需在 0~1")
	}
	if form.TopK != nil && (*form.TopK < 1 || *form.TopK > 100) {
		return common.NewBizError(common.PARAM_ERROR, "topK 取值需在 1~100")
	}
	if form.ScoreThreshold != nil && (*form.ScoreThreshold < 0 || *form.ScoreThreshold >= 1) {
		return common.NewBizError(common.PARAM_ERROR, "scoreThreshold 取值需在 [0,1)")
	}
	if form.RerankModel != nil && len([]rune(*form.RerankModel)) > 64 {
		return common.NewBizError(common.PARAM_ERROR, "rerankModel 长度不能超过 64")
	}
	if form.EmbeddingModel != nil && len([]rune(*form.EmbeddingModel)) > 64 {
		return common.NewBizError(common.PARAM_ERROR, "embeddingModel 长度不能超过 64")
	}
	if form.ChunkingStrategy != nil {
		switch *form.ChunkingStrategy {
		case "fixed", "semantic", "recursive", "qa", "table":
		default:
			return common.NewBizError(common.PARAM_ERROR, "chunkingStrategy 仅支持 fixed/semantic/recursive/qa/table")
		}
	}
	return nil
}

// UpdateKnowledgeBase 编辑知识库（名称查重；embedding 模型/分块策略创建后不可改）
func (s *Service) UpdateKnowledgeBase(ctx context.Context, kbID, userID int64, isAdmin bool, form *bo.KnowledgeBaseUpdateForm) error {
	if err := validateUpdateForm(form); err != nil {
		return err
	}
	kb, err := s.repo.GetKnowledgeBase(ctx, kbID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询知识库失败", err)
	}
	if kb == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "知识库不存在")
	}
	if kb.Visibility == "public" {
		if !isAdmin {
			return common.NewBizError(common.ACCESS_UNAUTHORIZED, "普通用户不能管理公共知识库")
		}
	} else if kb.CreateBy != userID {
		return common.NewBizError(common.ACCESS_UNAUTHORIZED, "无权操作他人私有知识库")
	}
	if form.EmbeddingModel != nil || form.ChunkingStrategy != nil {
		return common.NewBizError(common.BUSINESS_ERROR, "创建后不可修改 embedding 模型或分块策略")
	}
	if form.Name != nil && *form.Name != kb.Name {
		existing, findErr := s.repo.GetByNameAndOwner(ctx, *form.Name, kb.CreateBy)
		if findErr != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "校验知识库名称失败", findErr)
		}
		if existing != nil {
			return common.NewBizError(common.BUSINESS_ERROR, "知识库名称已存在")
		}
	}

	updates := map[string]interface{}{"update_by": userID}
	if form.Name != nil {
		updates["name"] = *form.Name
	}
	if form.Description != nil {
		updates["description"] = *form.Description
	}
	if form.SearchStrategy != nil {
		updates["search_strategy"] = *form.SearchStrategy
	}
	if form.TopK != nil {
		updates["top_k"] = *form.TopK
	}
	if form.ScoreThreshold != nil {
		updates["score_threshold"] = *form.ScoreThreshold
	}
	if form.HybridWeight != nil {
		updates["hybrid_weight"] = *form.HybridWeight
	}
	if form.EnableRerank != nil {
		updates["enable_rerank"] = boolToInt8(*form.EnableRerank)
	}
	if form.RerankModel != nil {
		updates["rerank_model"] = *form.RerankModel
	}
	if len(updates) > 1 {
		if err := s.repo.UpdateKnowledgeBase(ctx, kbID, updates); err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "更新知识库失败", err)
		}
	}
	s.invalidateCache(ctx, userID, kbID)
	return nil
}

// ListDocuments 文档列表（剔除 content 大字段）
func (s *Service) ListDocuments(ctx context.Context, kbID, userID int64, q *bo.KnowledgeDocumentQuery) (*vo.PageResult[vo.KnowledgeDocumentVO], error) {
	if err := s.checkKbReadable(ctx, kbID, userID); err != nil {
		return nil, err
	}
	page, size := q.PageNum, q.PageSize
	items, total, err := s.repo.PaginateDocuments(ctx, kbID, q.ProcessingStatus, page, size)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询文档列表失败", err)
	}
	rows := make([]vo.KnowledgeDocumentVO, 0, len(items))
	for i := range items {
		rows = append(rows, toDocumentVO(&items[i], false))
	}
	return &vo.PageResult[vo.KnowledgeDocumentVO]{List: rows, Total: total}, nil
}

// GetDocument 文档详情（含解析后 content）
func (s *Service) GetDocument(ctx context.Context, documentID, userID int64) (*vo.KnowledgeDocumentVO, error) {
	doc, err := s.repo.GetDocument(ctx, documentID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询文档失败", err)
	}
	if doc == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "文档不存在")
	}
	if err := s.checkKbReadable(ctx, doc.KnowledgeBaseID, userID); err != nil {
		return nil, err
	}
	result := toDocumentVO(doc, true)
	return &result, nil
}

// ListChunks 文档分块列表（chunk_index 正序；私有库仅 owner 可读）
func (s *Service) ListChunks(ctx context.Context, documentID, userID int64, page, size int) (*vo.PageResult[vo.KnowledgeChunkVO], error) {
	doc, err := s.repo.GetDocument(ctx, documentID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询文档失败", err)
	}
	if doc == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "文档不存在")
	}
	if err := s.checkKbReadable(ctx, doc.KnowledgeBaseID, userID); err != nil {
		return nil, err
	}
	items, total, err := s.repo.PaginateChunks(ctx, documentID, page, size)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询文档分块失败", err)
	}
	rows := make([]vo.KnowledgeChunkVO, 0, len(items))
	for i := range items {
		rows = append(rows, vo.KnowledgeChunkVO{
			ID: items[i].ID, DocumentID: items[i].DocumentID, ChunkIndex: items[i].ChunkIndex,
			Content: items[i].Content, TokenCount: items[i].TokenCount,
			Metadata: items[i].Metadata, CreateTime: &items[i].CreatedAt,
		})
	}
	return &vo.PageResult[vo.KnowledgeChunkVO]{List: rows, Total: total}, nil
}

// ==================== 召回测试集与低质量片段 ====================

func (s *Service) CreateTestSet(ctx context.Context, kbID int64, form *bo.TestSetCreateForm, operatorID int64) (*vo.KnowledgeTestSetVO, error) {
	if err := s.requireKnowledgeBase(ctx, kbID); err != nil {
		return nil, err
	}
	payload, err := json.Marshal(form.ExpectedChunkIds)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "期望分块序列化失败", err)
	}
	testSet := &model.SysKnowledgeTestSet{
		KnowledgeBaseID: kbID, Question: form.Question, ExpectedChunkIds: payload,
	}
	testSet.CreateBy = operatorID
	testSet.UpdateBy = operatorID
	if err := s.repo.CreateTestSet(ctx, testSet); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建召回测试集失败", err)
	}
	result := toTestSetVO(testSet)
	return &result, nil
}

func (s *Service) ListTestSets(ctx context.Context, kbID int64, page, size int) (*vo.PageResult[vo.KnowledgeTestSetVO], error) {
	if err := s.requireKnowledgeBase(ctx, kbID); err != nil {
		return nil, err
	}
	items, total, err := s.repo.PaginateTestSets(ctx, kbID, page, size)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询召回测试集失败", err)
	}
	rows := make([]vo.KnowledgeTestSetVO, 0, len(items))
	for i := range items {
		rows = append(rows, toTestSetVO(&items[i]))
	}
	return &vo.PageResult[vo.KnowledgeTestSetVO]{List: rows, Total: total}, nil
}

func (s *Service) ListLowQuality(ctx context.Context, kbID int64, page, size int) (*vo.PageResult[vo.LowQualityChunkVO], error) {
	if err := s.requireKnowledgeBase(ctx, kbID); err != nil {
		return nil, err
	}
	rows, total, err := s.repo.ListLowQuality(ctx, kbID, page, size)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询低质量片段失败", err)
	}
	if rows == nil {
		// 空结果必须是 []，不能是 null（python PageResult.list 恒为数组，SDK 断言 Array.isArray）
		rows = []vo.LowQualityChunkVO{}
	}
	return &vo.PageResult[vo.LowQualityChunkVO]{List: rows, Total: total}, nil
}

// ==================== 内部方法 ====================

func (s *Service) requireKnowledgeBase(ctx context.Context, kbID int64) error {
	kb, err := s.repo.GetKnowledgeBase(ctx, kbID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询知识库失败", err)
	}
	if kb == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "知识库不存在")
	}
	return nil
}

func (s *Service) checkKbReadable(ctx context.Context, kbID, userID int64) error {
	kb, err := s.repo.GetKnowledgeBase(ctx, kbID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询知识库失败", err)
	}
	if kb == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "知识库不存在")
	}
	if kb.Visibility == "private" && kb.CreateBy != userID {
		return common.NewBizError(common.ACCESS_UNAUTHORIZED, "无权查看他人私有知识库")
	}
	return nil
}

func (s *Service) invalidateCache(ctx context.Context, userID, kbID int64) {
	client := cacheClient()
	if client == nil {
		return
	}
	client.Del(ctx,
		"kb:list:"+strconv.FormatInt(userID, 10),
		"kb:detail:"+strconv.FormatInt(kbID, 10),
		kbListAdminKey,
	)
}

func readCachedPage(ctx context.Context, key string) *vo.PageResult[vo.KnowledgeBaseVO] {
	client := cacheClient()
	if client == nil {
		return nil
	}
	raw, err := client.Get(ctx, key).Bytes()
	if err != nil || len(raw) == 0 {
		return nil
	}
	var cached vo.PageResult[vo.KnowledgeBaseVO]
	if json.Unmarshal(raw, &cached) != nil {
		return nil
	}
	return &cached
}

func writeCachedPage(ctx context.Context, key string, result *vo.PageResult[vo.KnowledgeBaseVO], ttl time.Duration) {
	client := cacheClient()
	if client == nil {
		return
	}
	if payload, err := json.Marshal(result); err == nil {
		client.Set(ctx, key, payload, ttl)
	}
}

func toKnowledgeBaseVO(kb *model.SysKnowledgeBase) vo.KnowledgeBaseVO {
	createBy := kb.CreateBy
	return vo.KnowledgeBaseVO{
		ID:                kb.ID,
		Name:              kb.Name,
		Description:       kb.Description,
		Visibility:        kb.Visibility,
		EmbeddingProvider: kb.EmbeddingProvider,
		EmbeddingModel:    kb.EmbeddingModel,
		ChunkingStrategy:  kb.ChunkingStrategy,
		ChunkSize:         kb.ChunkSize,
		ChunkOverlap:      kb.ChunkOverlap,
		SearchStrategy:    kb.SearchStrategy,
		HybridWeight:      kb.HybridWeight,
		TopK:              kb.TopK,
		ScoreThreshold:    kb.ScoreThreshold,
		EnableRerank:      kb.EnableRerank,
		RerankModel:       kb.RerankModel,
		DocumentCount:     kb.DocumentCount,
		ChunkCount:        kb.ChunkCount,
		TotalTokens:       kb.TotalTokens,
		Status:            kb.Status,
		CreateBy:          &createBy,
		CreateTime:        kb.CreatedAt,
		UpdateTime:        kb.UpdatedAt,
	}
}

func toDocumentVO(doc *model.SysKnowledgeDocument, includeContent bool) vo.KnowledgeDocumentVO {
	createdAt := doc.CreatedAt
	updatedAt := doc.UpdatedAt
	result := vo.KnowledgeDocumentVO{
		ID:               doc.ID,
		KnowledgeBaseID:  doc.KnowledgeBaseID,
		FileID:           doc.FileID,
		Title:            doc.Title,
		Source:           doc.Source,
		Version:          doc.Version,
		ParsingStrategy:  doc.ParsingStrategy,
		RawContent:       doc.RawContent,
		ChunkCount:       doc.ChunkCount,
		TotalTokens:      doc.TotalTokens,
		ProcessingStatus: doc.ProcessingStatus,
		Error:            doc.Error,
		CreateTime:       &createdAt,
		UpdateTime:       &updatedAt,
	}
	if includeContent {
		result.Content = doc.Content
	}
	return result
}

func toTestSetVO(testSet *model.SysKnowledgeTestSet) vo.KnowledgeTestSetVO {
	var chunkIDs []int64
	if len(testSet.ExpectedChunkIds) > 0 {
		_ = json.Unmarshal(testSet.ExpectedChunkIds, &chunkIDs)
	}
	if chunkIDs == nil {
		chunkIDs = []int64{}
	}
	return vo.KnowledgeTestSetVO{
		ID:               testSet.ID,
		KnowledgeBaseID:  testSet.KnowledgeBaseID,
		Question:         testSet.Question,
		ExpectedChunkIds: chunkIDs,
		CreateTime:       testSet.CreatedAt,
	}
}

func boolToInt8(v bool) int8 {
	if v {
		return 1
	}
	return 0
}
