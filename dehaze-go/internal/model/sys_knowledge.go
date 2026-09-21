package model

import (
	"encoding/json"
)

// SysKnowledgeBase AI 知识库主表
type SysKnowledgeBase struct {
	BaseModel
	Name              string  `gorm:"column:name;type:varchar(255);not null;comment:知识库名称" json:"name"`
	Description       *string `gorm:"column:description;type:text;comment:知识库描述" json:"description"`
	Visibility        string  `gorm:"column:visibility;type:varchar(16);not null;default:private;comment:可见性(public/private)" json:"visibility"`
	EmbeddingProvider string  `gorm:"column:embedding_provider;type:varchar(32);not null;default:openai;comment:Embedding提供商" json:"embeddingProvider"`
	EmbeddingModel    string  `gorm:"column:embedding_model;type:varchar(64);not null;comment:Embedding模型标识" json:"embeddingModel"`
	ChunkingStrategy  string  `gorm:"column:chunking_strategy;type:varchar(16);not null;default:semantic;comment:分块策略" json:"chunkingStrategy"`
	ChunkSize         int     `gorm:"column:chunk_size;type:int;not null;default:800;comment:分块大小(token)" json:"chunkSize"`
	ChunkOverlap      int     `gorm:"column:chunk_overlap;type:int;not null;default:80;comment:分块重叠数(token)" json:"chunkOverlap"`
	SearchStrategy    string  `gorm:"column:search_strategy;type:varchar(16);not null;default:hybrid;comment:检索策略" json:"searchStrategy"`
	HybridWeight      float64 `gorm:"column:hybrid_weight;type:decimal(3,2);not null;default:0.70;comment:混合检索向量权重" json:"hybridWeight"`
	TopK              int     `gorm:"column:top_k;type:int;not null;default:5;comment:默认检索Top-K数" json:"topK"`
	ScoreThreshold    float64 `gorm:"column:score_threshold;type:decimal(4,3);not null;default:0.500;comment:相似度阈值" json:"scoreThreshold"`
	EnableRerank      int8    `gorm:"column:enable_rerank;type:tinyint;not null;default:0;comment:是否启用重排序" json:"enableRerank"`
	RerankModel       *string `gorm:"column:rerank_model;type:varchar(64);comment:重排序模型标识" json:"rerankModel"`
	DocumentCount     int     `gorm:"column:document_count;type:int;not null;default:0;comment:文档数(冗余统计)" json:"documentCount"`
	ChunkCount        int     `gorm:"column:chunk_count;type:int;not null;default:0;comment:分块总数(冗余统计)" json:"chunkCount"`
	TotalTokens       int64   `gorm:"column:total_tokens;type:bigint;not null;default:0;comment:编码Token总数" json:"totalTokens"`
	Status            int8    `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:启用;2:处理中;0:禁用)" json:"status"`
	Deleted           int64   `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysKnowledgeBase) TableName() string {
	return "sys_knowledge_base"
}

// SysKnowledgeDocument AI 知识库文档表
type SysKnowledgeDocument struct {
	BaseModel
	KnowledgeBaseID  int64   `gorm:"column:knowledge_base_id;type:bigint;not null;index:idx_kb_status" json:"knowledgeBaseId"`
	FileID           *int64  `gorm:"column:file_id;type:bigint;comment:文件ID" json:"fileId"`
	Title            string  `gorm:"column:title;type:varchar(512);not null;comment:文档标题" json:"title"`
	Source           string  `gorm:"column:source;type:varchar(16);not null;default:upload;comment:文档来源" json:"source"`
	Version          int     `gorm:"column:version;type:int;not null;default:1;comment:文档版本号" json:"version"`
	ParsingStrategy  string  `gorm:"column:parsing_strategy;type:varchar(16);not null;default:auto;comment:解析策略" json:"parsingStrategy"`
	Content          *string `gorm:"column:content;type:longtext;comment:解析后的纯文本内容" json:"content"`
	RawContent       *string `gorm:"column:raw_content;type:longtext;comment:原始富文本" json:"rawContent"`
	ChunkCount       int     `gorm:"column:chunk_count;type:int;not null;default:0;comment:分块数(冗余统计)" json:"chunkCount"`
	TotalTokens      int64   `gorm:"column:total_tokens;type:bigint;not null;default:0;comment:编码Token总数" json:"totalTokens"`
	ProcessingStatus string  `gorm:"column:processing_status;type:varchar(16);not null;default:pending;comment:处理状态" json:"processingStatus"`
	Error            *string `gorm:"column:error;type:text;comment:失败原因" json:"error"`
	Deleted          int64   `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysKnowledgeDocument) TableName() string {
	return "sys_knowledge_document"
}

// SysKnowledgeTestSet AI 知识库召回测试集
type SysKnowledgeTestSet struct {
	BaseModel
	KnowledgeBaseID  int64           `gorm:"column:knowledge_base_id;type:bigint;not null;index:idx_kb_id" json:"knowledgeBaseId"`
	Question         string          `gorm:"column:question;type:varchar(1000);not null;comment:测试问题" json:"question"`
	ExpectedChunkIds json.RawMessage `gorm:"column:expected_chunk_ids;type:json;not null;comment:期望命中分块ID数组" json:"expectedChunkIds"`
	Deleted          int64           `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysKnowledgeTestSet) TableName() string {
	return "sys_knowledge_test_set"
}

// SysKnowledgeChunk AI 知识库分块表（无逻辑删除列，随文档删除/重处理由写入方清理）
type SysKnowledgeChunk struct {
	BaseModel
	DocumentID      int64           `gorm:"column:document_id;type:bigint;not null;comment:文档ID(关联sys_knowledge_document.id)" json:"documentId"`
	KnowledgeBaseID int64           `gorm:"column:knowledge_base_id;type:bigint;not null;comment:知识库ID(冗余，便于跨文档检索)" json:"knowledgeBaseId"`
	ChunkIndex      int             `gorm:"column:chunk_index;type:int;not null;comment:分块序号(从0开始)" json:"chunkIndex"`
	SectionIndex    int             `gorm:"column:section_index;type:int;not null;default:0;comment:所属小节序号(0=无标题文档整篇)" json:"sectionIndex"`
	SectionPath     *string         `gorm:"column:section_path;type:varchar(255);comment:小节标题路径" json:"sectionPath"`
	Content         string          `gorm:"column:content;type:text;not null;comment:分块后的文本片段" json:"content"`
	TokenCount      int             `gorm:"column:token_count;type:int;not null;default:0;comment:分块Token数" json:"tokenCount"`
	Metadata        json.RawMessage `gorm:"column:metadata;type:json;comment:分块元数据(来源文档/页码/段落/表格行等)" json:"metadata"`
}

func (SysKnowledgeChunk) TableName() string {
	return "sys_knowledge_chunk"
}
