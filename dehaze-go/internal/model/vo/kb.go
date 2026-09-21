package vo

import (
	"encoding/json"
	"time"
)

// KnowledgeBaseVO 知识库视图对象（对齐 python KnowledgeBaseVO）
type KnowledgeBaseVO struct {
	ID                int64     `json:"id"`
	Name              string    `json:"name"`
	Description       *string   `json:"description"`
	Visibility        string    `json:"visibility"`
	EmbeddingProvider string    `json:"embeddingProvider"`
	EmbeddingModel    string    `json:"embeddingModel"`
	ChunkingStrategy  string    `json:"chunkingStrategy"`
	ChunkSize         int       `json:"chunkSize"`
	ChunkOverlap      int       `json:"chunkOverlap"`
	SearchStrategy    string    `json:"searchStrategy"`
	HybridWeight      float64   `json:"hybridWeight"`
	TopK              int       `json:"topK"`
	ScoreThreshold    float64   `json:"scoreThreshold"`
	EnableRerank      int8      `json:"enableRerank"`
	RerankModel       *string   `json:"rerankModel"`
	DocumentCount     int       `json:"documentCount"`
	ChunkCount        int       `json:"chunkCount"`
	TotalTokens       int64     `json:"totalTokens"`
	Status            int8      `json:"status"`
	CreateBy          *int64    `json:"createBy"`
	CreateTime        time.Time `json:"createTime"`
	UpdateTime        time.Time `json:"updateTime"`
}

// KnowledgeDocumentVO 文档视图对象（列表接口剔除 content 大字段）
type KnowledgeDocumentVO struct {
	ID               int64      `json:"id"`
	KnowledgeBaseID  int64      `json:"knowledgeBaseId"`
	FileID           *int64     `json:"fileId"`
	Title            string     `json:"title"`
	Source           string     `json:"source"`
	Version          int        `json:"version"`
	ParsingStrategy  string     `json:"parsingStrategy"`
	Content          *string    `json:"content,omitempty"`
	RawContent       *string    `json:"rawContent"`
	ChunkCount       int        `json:"chunkCount"`
	TotalTokens      int64      `json:"totalTokens"`
	ProcessingStatus string     `json:"processingStatus"`
	Error            *string    `json:"error"`
	CreateTime       *time.Time `json:"createTime"`
	UpdateTime       *time.Time `json:"updateTime"`
}

// KnowledgeTestSetVO 召回测试集视图对象
type KnowledgeTestSetVO struct {
	ID               int64     `json:"id"`
	KnowledgeBaseID  int64     `json:"knowledgeBaseId"`
	Question         string    `json:"question"`
	ExpectedChunkIds []int64   `json:"expectedChunkIds"`
	CreateTime       time.Time `json:"createTime"`
}

// KnowledgeChunkVO 文档分块视图对象（metadata 为 JSON 列解析后的对象，NULL 时不下发）
type KnowledgeChunkVO struct {
	ID         int64           `json:"id"`
	DocumentID int64           `json:"documentId"`
	ChunkIndex int             `json:"chunkIndex"`
	Content    string          `json:"content"`
	TokenCount int             `json:"tokenCount"`
	Metadata   json.RawMessage `json:"metadata,omitempty"`
	CreateTime *time.Time      `json:"createTime"`
}

// LowQualityChunkVO 低质量片段视图对象（被点踩片段，按片段点踩次数聚合）
type LowQualityChunkVO struct {
	ChunkID         int64  `json:"chunkId"`
	Content         string `json:"content"`
	DocumentID      int64  `json:"documentId"`
	ThumbsDownCount int64  `json:"thumbsDownCount"`
}
