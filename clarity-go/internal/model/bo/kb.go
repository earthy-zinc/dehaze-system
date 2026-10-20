package bo

// KnowledgeBasePageQuery 知识库分页查询（view=admin 为管理端只读监控视角）
type KnowledgeBasePageQuery struct {
	AiPageQuery
	Keyword string `form:"keyword"`
	View    string `form:"view"`
}

// KnowledgeBaseUpdateForm 编辑知识库表单（仅可编辑项；embedding_model/chunking_strategy 创建后不可修改）
type KnowledgeBaseUpdateForm struct {
	Name             *string  `json:"name"`
	Description      *string  `json:"description"`
	SearchStrategy   *string  `json:"searchStrategy"`
	HybridWeight     *float64 `json:"hybridWeight"`
	TopK             *int     `json:"topK"`
	ScoreThreshold   *float64 `json:"scoreThreshold"`
	EnableRerank     *bool    `json:"enableRerank"`
	RerankModel      *string  `json:"rerankModel"`
	EmbeddingModel   *string  `json:"embeddingModel"`
	ChunkingStrategy *string  `json:"chunkingStrategy"`
}

// KnowledgeDocumentQuery 文档分页查询
type KnowledgeDocumentQuery struct {
	AiPageQuery
	ProcessingStatus string `form:"processingStatus"`
}

// TestSetCreateForm 创建召回测试集表单
type TestSetCreateForm struct {
	Question         string  `json:"question" binding:"required,max=1000"`
	ExpectedChunkIds []int64 `json:"expectedChunkIds" binding:"required,min=1"`
}
