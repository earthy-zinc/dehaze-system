package api

import (
	"testing"

	"github.com/gin-gonic/gin"
)

// 全仓分页唯一实现是 parsePaginationWithSize（python `pageNum: Query(default=1, ge=1)` /
// `pageSize: Query(default=N, ge=1, le=100)` 口径）：未传取缺省，显式传非数字、<1 或 >100 一律 A0400。
// `AiPageQuery` 的分页字段已 `form:"-"`（不参与 gin 绑定），service 侧也不再做归一化兜底，
// 故下面 13 个端点的分页行为全部由 handler 这一处裁决。用例把「越界必拒 + 边界放行」钉死，
// 防止回退成"service 内静默钳制/回退默认值"（那会让 pageSize=500 在 go 返回 500 行、python 报 A0400）。

// TestAiBillingObservabilityPaginationBounds AI 计费 5 个列表 + 可观测性 3 个端点。
func TestAiBillingObservabilityPaginationBounds(t *testing.T) {
	gin.SetMode(gin.TestMode)
	denied := []string{
		"pageSize=-1", "pageSize=0", "pageSize=101", "pageNum=0", "pageNum=-1",
		"pageSize=abc", "pageNum=abc",
	}
	accepted := []string{"", "pageNum=2&pageSize=50", "pageSize=100"}

	cases := []struct {
		name    string
		handler gin.HandlerFunc
	}{
		{"billing.ListRecords", NewAiBillingApi(nil, nil).ListRecords},
		{"billing.ListCreditLogs", NewAiBillingApi(nil, nil).ListCreditLogs},
		{"billing.ListRefunds", NewAiBillingApi(nil, nil).ListRefunds},
		{"billing.ListAnomalies", NewAiBillingApi(nil, nil).ListAnomalies},
		{"billing.ListCosts", NewAiBillingApi(nil, nil).ListCosts},
		{"observability.ListTraces", NewAiObservabilityApi(nil).ListTraces},
		{"observability.ExportTraces", NewAiObservabilityApi(nil).ExportTraces},
		{"observability.GetCosts", NewAiObservabilityApi(nil).GetCosts},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertPageBounds(t, "/probe", "/probe", tc.handler, denied, accepted)
		})
	}
}

// TestAiKbPaginationBounds KB 原生列表端点。其中 ListKnowledgeBases/ListDocuments 曾走 gin 结构体绑定，
// `AiPageQuery` 改 `form:"-"` 后与其余端点同口径：非数字同样是 A0400（不再是绑定失败兜底的 B0001），
// 故此处与上一组使用同一套拒绝集。
func TestAiKbPaginationBounds(t *testing.T) {
	gin.SetMode(gin.TestMode)
	denied := []string{
		"pageSize=-1", "pageSize=0", "pageSize=101", "pageNum=0", "pageNum=-1",
		"pageSize=abc", "pageNum=abc",
	}
	accepted := []string{"", "pageNum=2&pageSize=50", "pageSize=100"}

	cases := []struct {
		name    string
		pattern string
		path    string
		handler gin.HandlerFunc
	}{
		{"kb.ListKnowledgeBases", "/probe", "/probe", NewAiKbApi(nil).ListKnowledgeBases},
		{"kb.ListDocuments", "/probe/:id", "/probe/1", NewAiKbApi(nil).ListDocuments},
		{"kb.ListDocumentChunks", "/probe/:id", "/probe/1", NewAiKbApi(nil).ListDocumentChunks},
		{"kb.ListTestSets", "/probe/:id", "/probe/1", NewAiKbApi(nil).ListTestSets},
		{"kb.ListLowQuality", "/probe/:id", "/probe/1", NewAiKbApi(nil).ListLowQuality},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertPageBounds(t, tc.pattern, tc.path, tc.handler, denied, accepted)
		})
	}
}
