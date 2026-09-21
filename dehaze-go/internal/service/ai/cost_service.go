package ai

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// membershipAIRatio 会员卡实收计入 AI 参考毛利的比例（与 python settings.AI_BILLING_MEMBERSHIP_AI_RATIO 默认值一致）
const membershipAIRatio = 0.3

// CostService 成本单价维护（价格版本化）与成本-利润核算（毛利双口径）。
type CostService struct {
	repo *airepo.CostRepository
}

func NewCostService(repo *airepo.CostRepository) *CostService {
	return &CostService{repo: repo}
}

// ListCosts 成本单价分页（含档位明细）
func (s *CostService) ListCosts(ctx context.Context, q *bo.ModelCostQuery) (*vo.PageResult[vo.ModelCostVO], error) {
	page, size := q.PageNum, q.PageSize
	costs, total, err := s.repo.PaginateCosts(ctx, page, size, q.Keyword, q.ModelID, q.ProviderID)
	if err != nil {
		return nil, err
	}
	items := make([]vo.ModelCostVO, 0, len(costs))
	for i := range costs {
		details, detailErr := s.repo.ListCostDetails(ctx, costs[i].ID)
		if detailErr != nil {
			return nil, detailErr
		}
		items = append(items, *costToVO(&costs[i], details))
	}
	return &vo.PageResult[vo.ModelCostVO]{List: items, Total: total}, nil
}

// validateCostCurrency 币种长度校验：对齐 python ModelCostCreate/UpdateRequest 的 currency max_length=8。
func validateCostCurrency(currency string) error {
	if len([]rune(currency)) > 8 {
		return common.NewBizError(common.PARAM_ERROR, "currency 长度不能超过 8")
	}
	return nil
}

// validateCostDetails 成本档位明细校验：对齐 python ModelCostDetailForm 的 unit_price ≥0 / min_tokens ≥0。
// python 由 pydantic 在请求校验阶段拦截，go 无对应层，须在 service 层补齐，
// 否则负单价会直接落库（成本核算与毛利统计随之失真）。
func validateCostDetails(details []bo.ModelCostDetailForm) error {
	for _, detail := range details {
		if detail.UnitPrice < 0 {
			return common.NewBizError(common.PARAM_ERROR, "unitPrice 不能为负数")
		}
		if detail.MinTokens < 0 {
			return common.NewBizError(common.PARAM_ERROR, "minTokens 不能为负数")
		}
	}
	return nil
}

// CreateCost 新增成本单价：同模型同供应商生成新价格版本，历史版本保留可追溯
func (s *CostService) CreateCost(ctx context.Context, operatorID int64, form *bo.ModelCostCreateForm) (*vo.ModelCostVO, error) {
	if form.ModelID == "" || len([]rune(form.ModelID)) > 64 {
		return nil, common.NewBizError(common.PARAM_ERROR, "modelId 长度需为 1~64")
	}
	if err := validateCostCurrency(form.Currency); err != nil {
		return nil, err
	}
	if err := validateCostDetails(form.Details); err != nil {
		return nil, err
	}
	version, err := s.repo.NextPriceVersion(ctx, form.ModelID, form.ProviderID)
	if err != nil {
		return nil, err
	}
	effectiveFrom := time.Now()
	if form.EffectiveFrom != nil {
		effectiveFrom = *form.EffectiveFrom
	}
	currency := form.Currency
	if currency == "" {
		currency = "CNY"
	}
	status := 1
	if form.Status != nil {
		status = *form.Status
	}
	cost := &model.SysAiModelCost{
		ModelID: form.ModelID, ProviderID: form.ProviderID, PriceVersion: version,
		Currency: currency, EffectiveFrom: effectiveFrom, EffectiveTo: form.EffectiveTo,
		Status: status, CreateBy: &operatorID, UpdateBy: &operatorID,
	}
	if err := s.repo.CreateCost(ctx, cost); err != nil {
		return nil, err
	}
	details := make([]model.SysAiModelCostDetail, 0, len(form.Details))
	for _, d := range form.Details {
		details = append(details, model.SysAiModelCostDetail{
			PriceID: cost.ID, TokenType: d.TokenType, TimeSlot: d.TimeSlot,
			MinTokens: d.MinTokens, MaxTokens: d.MaxTokens, UnitPrice: d.UnitPrice,
		})
	}
	if err := s.repo.CreateCostDetails(ctx, details); err != nil {
		return nil, err
	}
	return costToVO(cost, details), nil
}

// UpdateCost 更新成本价格版本主表字段（币种/生效时间/状态）
func (s *CostService) UpdateCost(ctx context.Context, costID, operatorID int64, form *bo.ModelCostUpdateForm) (*vo.ModelCostVO, error) {
	if form.Currency != nil {
		if err := validateCostCurrency(*form.Currency); err != nil {
			return nil, err
		}
	}
	cost, err := s.repo.GetCost(ctx, costID)
	if err != nil {
		return nil, err
	}
	if cost == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "成本单价不存在")
	}
	fields := map[string]any{"update_by": operatorID}
	if form.Currency != nil {
		fields["currency"] = *form.Currency
	}
	if form.EffectiveFrom != nil {
		fields["effective_from"] = *form.EffectiveFrom
	}
	if form.EffectiveTo != nil {
		fields["effective_to"] = *form.EffectiveTo
	}
	if form.Status != nil {
		fields["status"] = *form.Status
	}
	if err := s.repo.UpdateCost(ctx, costID, fields); err != nil {
		return nil, err
	}
	details, err := s.repo.ListCostDetails(ctx, costID)
	if err != nil {
		return nil, err
	}
	if form.Currency != nil {
		cost.Currency = *form.Currency
	}
	if form.EffectiveFrom != nil {
		cost.EffectiveFrom = *form.EffectiveFrom
	}
	if form.EffectiveTo != nil {
		cost.EffectiveTo = form.EffectiveTo
	}
	if form.Status != nil {
		cost.Status = *form.Status
	}
	return costToVO(cost, details), nil
}

// DeleteCost 删除成本单价（主表与档位明细逻辑删除）
func (s *CostService) DeleteCost(ctx context.Context, costID, operatorID int64) error {
	cost, err := s.repo.GetCost(ctx, costID)
	if err != nil {
		return err
	}
	if cost == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "成本单价不存在")
	}
	if err := s.repo.SoftDeleteCost(ctx, costID, operatorID); err != nil {
		return err
	}
	return s.repo.SoftDeleteCostDetails(ctx, costID, operatorID)
}

// CostStats 成本-利润统计：整体双口径毛利（overall/ai）或模型/供应商成本分解
func (s *CostService) CostStats(ctx context.Context, q *bo.CostStatQuery) ([]vo.CostStatVO, error) {
	groupBy := q.GroupBy
	if groupBy == "" {
		groupBy = "overall"
	}
	start, err := parseBillingTime(q.StartTime)
	if err != nil {
		return nil, err
	}
	end, err := parseBillingTime(q.EndTime)
	if err != nil {
		return nil, err
	}

	switch groupBy {
	case "overall":
		totalCost, err := s.repo.SumBillingCost(ctx, start, end)
		if err != nil {
			return nil, err
		}
		income, err := s.repo.SumPaidOrderAmountByType(ctx, start, end)
		if err != nil {
			return nil, err
		}
		creditIncome := float64(income["credit"]) / 100 // 分 → 元
		var vipIncome float64
		for packageType, amount := range income {
			if packageType != "credit" {
				vipIncome += float64(amount) / 100
			}
		}
		return []vo.CostStatVO{
			buildCostStat("overall", creditIncome+vipIncome, totalCost),
			buildCostStat("ai", creditIncome+vipIncome*membershipAIRatio, totalCost),
		}, nil
	case "model", "provider":
		rows, err := s.repo.SumBillingCostByGroup(ctx, groupBy, start, end, q.ModelID, q.ProviderID)
		if err != nil {
			return nil, err
		}
		results := make([]vo.CostStatVO, 0, len(rows))
		for _, row := range rows {
			dimension := row.Dimension
			results = append(results, vo.CostStatVO{Dimension: &dimension, Cost: row.Cost})
		}
		return results, nil
	default:
		return nil, common.NewBizError(common.PARAM_ERROR, "groupBy 仅支持 overall/model/provider")
	}
}

// ImportReconcile 供应商账单导入（最小实现：按非空行计数，对账骨架）
func (s *CostService) ImportReconcile(form *bo.ReconcileImportForm) int {
	imported := 0
	for _, line := range strings.Split(form.Content, "\n") {
		if strings.TrimSpace(line) != "" {
			imported++
		}
	}
	return imported
}

// buildCostStat 组装毛利统计项：收入/成本/毛利四舍五入到分，毛利率按未取整毛利计算后保留 4 位
func buildCostStat(metric string, revenue, cost float64) vo.CostStatVO {
	profitRate := 0.0
	if revenue != 0 {
		profitRate = round4((revenue - cost) / revenue)
	}
	roundedRevenue := round2(revenue)
	roundedCost := round2(cost)
	roundedProfit := round2(revenue - cost)
	return vo.CostStatVO{
		Metric: &metric, Revenue: &roundedRevenue, Cost: roundedCost,
		Profit: &roundedProfit, ProfitRate: &profitRate,
	}
}

func costToVO(cost *model.SysAiModelCost, details []model.SysAiModelCostDetail) *vo.ModelCostVO {
	detailVOs := make([]vo.ModelCostDetailVO, 0, len(details))
	for _, d := range details {
		detailVOs = append(detailVOs, vo.ModelCostDetailVO{
			ID: d.ID, PriceID: d.PriceID, TokenType: d.TokenType, TimeSlot: d.TimeSlot,
			MinTokens: d.MinTokens, MaxTokens: d.MaxTokens,
			// unit_price 为 decimal(12,4)，python Decimal 序列化为 4 位小数字符串
			UnitPrice: fmt.Sprintf("%.4f", d.UnitPrice),
		})
	}
	return &vo.ModelCostVO{
		ID: cost.ID, ModelID: cost.ModelID, ProviderID: cost.ProviderID,
		PriceVersion: cost.PriceVersion, Currency: cost.Currency,
		EffectiveFrom: cost.EffectiveFrom, EffectiveTo: cost.EffectiveTo, Status: cost.Status,
		Details: detailVOs, CreateTime: &cost.CreateTime, UpdateTime: cost.UpdateTime,
	}
}

func round2(v float64) float64 {
	return math.Round(v*100) / 100
}
