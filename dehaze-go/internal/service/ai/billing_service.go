package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	goredis "github.com/redis/go-redis/v9"
	"gorm.io/gorm"
)

// 缓存键与 dehaze-python app/service/billing/* 完全一致（三端共享 Redis）：
//   - ai:balance:{user_id}     余额（TTL 1 小时）
//   - ai:arrears:{user_id}     欠费标记（无 TTL）
//   - ai:quota:daily|monthly   日/月已用配额
//   - ai:bill:{user_id}:{month} 月结账单（TTL 90 天）
const (
	balanceKeyFmt    = "ai:balance:%d"
	arrearsKeyFmt    = "ai:arrears:%d"
	quotaDailyKeyFmt = "ai:quota:daily:%d:%s"
	quotaMonthKeyFmt = "ai:quota:monthly:%d:%s"
	billKeyFmt       = "ai:bill:%d:%s"

	balanceTTL      = 3600 * time.Second
	billCacheTTL    = 90 * 24 * 3600 * time.Second
	balanceCASRetry = 3
)

// chatBillTypes chat 类计费类型：token 口径统计仅含主对话与子 Agent 实报实销，
// asr/tts 的 input_tokens 存秒数/字符数，混入会使 token 统计失真
var chatBillTypes = []string{"chat", "chat_subagent"}

// rechargeSources 账单口径的充值类来源（消费与充值退款除外）
var rechargeSources = map[string]bool{
	"recharge": true, "vip_gift": true, "trial": true, "admin_adjust": true, "vip_gift_expire": true,
}

// shanghaiTZ 配额日/月重置与异常事件时间统一按 Asia/Shanghai（与 python ZoneInfo 一致）
var shanghaiTZ = time.FixedZone("Asia/Shanghai", 8*3600)

// BillingService AI 计费：余额/配额（Redis 为准实时权威 + MySQL 乐观锁落库）、查询、退款与管理员操作。
type BillingService struct {
	db    *gorm.DB
	repo  *airepo.BillingRepository
	redis *goredis.Client
}

func NewBillingService(db *gorm.DB, repo *airepo.BillingRepository, redisClient *goredis.Client) *BillingService {
	return &BillingService{db: db, repo: repo, redis: redisClient}
}

// ==================== 余额与配额 ====================

// Balance 余额账户视图（权益缺失/停用时限额展示为 0，配额校验侧 fail-closed）
func (s *BillingService) Balance(ctx context.Context, userID int64) (*vo.BillingBalanceVO, error) {
	balance, err := s.balanceString(ctx, userID)
	if err != nil {
		return nil, err
	}
	arrears, err := s.isArrears(ctx, userID)
	if err != nil {
		return nil, err
	}
	dailyUsed, monthlyUsed, err := s.quotaUsed(ctx, userID)
	if err != nil {
		return nil, err
	}
	var dailyLimit, monthlyLimit int64
	if limits, limitErr := s.repo.QuotaLimits(ctx, userID); limitErr != nil {
		return nil, limitErr
	} else if limits != nil {
		dailyLimit, monthlyLimit = limits[0], limits[1]
	}
	return &vo.BillingBalanceVO{
		UserID:         userID,
		CreditsBalance: balance,
		ArrearsStatus:  arrears,
		DailyUsed:      dailyUsed,
		DailyLimit:     dailyLimit,
		MonthlyUsed:    monthlyUsed,
		MonthlyLimit:   monthlyLimit,
	}, nil
}

// balanceString 余额字符串：Redis 优先，未命中查 MySQL 并整数化回填（对齐 python get_balance 的坏值处理）
func (s *BillingService) balanceString(ctx context.Context, userID int64) (string, error) {
	if s.redis == nil {
		return "", common.NewBizError(common.MIDDLEWARE_SERVICE_ERROR, "缓存服务不可用")
	}
	key := fmt.Sprintf(balanceKeyFmt, userID)
	raw, err := s.redis.Get(ctx, key).Result()
	if err == nil {
		if _, convErr := strconv.ParseFloat(raw, 64); convErr == nil {
			return raw, nil
		}
		// 坏值（非数字）：删除后走 MySQL 整数化回填
		_ = s.redis.Del(ctx, key).Err()
	} else if err != goredis.Nil {
		return "", common.WrapBizError(common.MIDDLEWARE_SERVICE_ERROR, "读取余额缓存失败", err)
	}

	balance := 0.0
	if current, balErr := s.repo.GetUserBalance(ctx, userID); balErr != nil {
		return "", balErr
	} else if current != nil {
		balance = current.Balance
	}
	_ = s.redis.SetEx(ctx, key, strconv.FormatInt(int64(balance), 10), balanceTTL).Err()
	return fmt.Sprintf("%.2f", balance), nil
}

// isArrears 欠费标记（Redis ai:arrears:{user_id}）
func (s *BillingService) isArrears(ctx context.Context, userID int64) (bool, error) {
	if s.redis == nil {
		return false, common.NewBizError(common.MIDDLEWARE_SERVICE_ERROR, "缓存服务不可用")
	}
	n, err := s.redis.Exists(ctx, fmt.Sprintf(arrearsKeyFmt, userID)).Result()
	if err != nil {
		return false, common.WrapBizError(common.MIDDLEWARE_SERVICE_ERROR, "读取欠费标记失败", err)
	}
	return n > 0, nil
}

// quotaUsed 日/月已用配额（Redis 不存在时为 0）
func (s *BillingService) quotaUsed(ctx context.Context, userID int64) (int64, int64, error) {
	if s.redis == nil {
		return 0, 0, common.NewBizError(common.MIDDLEWARE_SERVICE_ERROR, "缓存服务不可用")
	}
	now := time.Now().In(shanghaiTZ)
	dailyKey := fmt.Sprintf(quotaDailyKeyFmt, userID, now.Format("2006-01-02"))
	monthKey := fmt.Sprintf(quotaMonthKeyFmt, userID, now.Format("2006-01"))
	values, err := s.redis.MGet(ctx, dailyKey, monthKey).Result()
	if err != nil {
		return 0, 0, common.WrapBizError(common.MIDDLEWARE_SERVICE_ERROR, "读取配额失败", err)
	}
	return redisIntValue(values[0]), redisIntValue(values[1]), nil
}

func redisIntValue(value any) int64 {
	if value == nil {
		return 0
	}
	n, err := strconv.ParseInt(fmt.Sprintf("%v", value), 10, 64)
	if err != nil {
		return 0
	}
	return n
}

// increaseBalance 增加余额：Redis INCRBY → MySQL CAS 落库 → 清欠费标记 → 写流水
//
// 返回变动后余额（Redis 为准实时权威值）。余额与流水在同一事务内落库，避免账实不一致。
func (s *BillingService) increaseBalance(
	ctx context.Context, userID, amount int64, source string,
	relatedID *int64, reason *string, operatorID *int64,
) (string, error) {
	if s.redis == nil {
		return "", common.NewBizError(common.MIDDLEWARE_SERVICE_ERROR, "缓存服务不可用")
	}
	key := fmt.Sprintf(balanceKeyFmt, userID)
	if err := s.redis.IncrBy(ctx, key, amount).Err(); err != nil {
		return "", common.WrapBizError(common.MIDDLEWARE_SERVICE_ERROR, "余额预增失败", err)
	}

	var balanceAfter int64
	err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txRepo := airepo.NewBillingRepository(tx)
		if casErr := s.addBalanceCAS(ctx, txRepo, userID, amount); casErr != nil {
			return casErr
		}
		if current, readErr := txRepo.GetUserBalance(ctx, userID); readErr != nil {
			return readErr
		} else if current != nil {
			balanceAfter = int64(current.Balance)
		}
		return txRepo.CreateCreditLog(ctx, &model.SysAiCreditLog{
			UserID: userID, Source: source, Amount: amount,
			BalanceAfter: balanceAfter, RelatedID: relatedID,
			Reason: reason, OperatorID: operatorID,
		})
	})
	if err != nil {
		// 落库失败必须回滚 Redis 预增，否则余额永久背离
		_ = s.redis.IncrBy(ctx, key, -amount).Err()
		return "", err
	}

	// 余额为整数积分语义：缓存统一整数化（Decimal 的 "100.00" 形式会使后续 DECRBY 报错）
	if err := s.redis.Set(ctx, key, strconv.FormatInt(balanceAfter, 10), 0).Err(); err != nil {
		return "", common.WrapBizError(common.MIDDLEWARE_SERVICE_ERROR, "余额缓存回写失败", err)
	}
	_ = s.redis.Del(ctx, fmt.Sprintf(arrearsKeyFmt, userID)).Err()
	return strconv.FormatInt(balanceAfter, 10), nil
}

// addBalanceCAS 乐观锁增加余额，CAS 失败重试 3 次
func (s *BillingService) addBalanceCAS(ctx context.Context, repo *airepo.BillingRepository, userID, amount int64) error {
	for i := 0; i < balanceCASRetry; i++ {
		current, err := repo.GetUserBalance(ctx, userID)
		if err != nil {
			return err
		}
		if current == nil {
			// Redis 已加而 MySQL 无账户：静默跳过会造成余额永久背离，必须显式失败
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "用户不存在")
		}
		ok, err := repo.AddBalanceCAS(ctx, userID, amount, current.Version)
		if err != nil {
			return err
		}
		if ok {
			return nil
		}
	}
	return common.NewBizError(common.SYSTEM_EXECUTION_ERROR, "余额落库失败（CAS 重试耗尽）")
}

// ==================== 用户端查询 ====================

// Summary 用户端消耗汇总（仅 chat 类记录，仅本人数据）
func (s *BillingService) Summary(ctx context.Context, userID int64, dimension string) (*vo.BillingSummaryVO, error) {
	now := time.Now()
	// create_time 以秒级精度入库，上界取下一整秒确保包含刚写入的记录
	nowCeiling := now.Add(time.Second).Truncate(time.Second)
	var periodStart time.Time
	switch dimension {
	case "month":
		periodStart = time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, time.Local)
	case "day":
		periodStart = time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.Local)
	default:
		return nil, common.NewBizError(common.PARAM_ERROR, "dimension 仅支持 day/month")
	}

	rows, err := s.repo.SumBillingByPeriod(ctx, userID, periodStart, nowCeiling, dimension, chatBillTypes)
	if err != nil {
		return nil, err
	}
	trend := make([]vo.BillingTrendPointVO, 0, len(rows))
	var totalCredits, totalInput, totalOutput, cachedInput, creditsSaved int64
	for _, row := range rows {
		trend = append(trend, vo.BillingTrendPointVO{
			Date: row.Date, Credits: row.Credits,
			InputTokens: row.InputTokens, OutputTokens: row.OutputTokens,
		})
		totalCredits += row.Credits
		totalInput += row.InputTokens
		totalOutput += row.OutputTokens
		cachedInput += row.CachedInputTokens
		creditsSaved += row.CreditsSaved
	}

	distRows, err := s.repo.SumBillingByModel(ctx, userID, periodStart, nowCeiling, chatBillTypes)
	if err != nil {
		return nil, err
	}
	sort.SliceStable(distRows, func(i, j int) bool { return distRows[i].Credits > distRows[j].Credits })
	if len(distRows) > 5 {
		distRows = distRows[:5]
	}
	distribution := make([]vo.BillingModelDistVO, 0, len(distRows))
	for _, row := range distRows {
		distribution = append(distribution, vo.BillingModelDistVO{
			Model: row.Model, Credits: row.Credits,
			Tokens: row.InputTokens + row.OutputTokens,
		})
	}

	return &vo.BillingSummaryVO{
		TotalCredits: totalCredits, InputTokens: totalInput, OutputTokens: totalOutput,
		Trend:             trend,
		ModelDistribution: distribution,
		Savings:           vo.BillingSavingsVO{CachedInputTokens: cachedInput, CreditsSaved: creditsSaved},
	}, nil
}

// Records 计费明细分页（含每笔记录的最新退款申请状态）
func (s *BillingService) Records(ctx context.Context, userID int64, q *bo.BillingRecordQuery) (*vo.PageResult[vo.BillingRecordVO], error) {
	page, size := q.PageNum, q.PageSize
	start, err := parseBillingTime(q.DateStart)
	if err != nil {
		return nil, err
	}
	end, err := parseBillingTime(q.DateEnd)
	if err != nil {
		return nil, err
	}
	records, total, err := s.repo.PaginateBillingByUser(
		ctx, userID, page, size, q.ConversationID, q.BillType, q.ModelID, start, end)
	if err != nil {
		return nil, err
	}
	ids := make([]int64, 0, len(records))
	for i := range records {
		ids = append(ids, records[i].ID)
	}
	statusMap, err := s.repo.LatestRefundStatusByBillingIDs(ctx, ids)
	if err != nil {
		return nil, err
	}
	items := make([]vo.BillingRecordVO, 0, len(records))
	for i := range records {
		r := &records[i]
		items = append(items, vo.BillingRecordVO{
			ID: r.ID, UserID: r.UserID,
			ConversationID: r.ConversationID, MessageID: r.MessageID,
			Model: r.Model, ActualModel: r.ActualModel, BillType: r.BillType,
			InputTokens: r.InputTokens, CachedInputTokens: r.CachedInputTokens,
			OutputTokens: r.OutputTokens, Credits: r.Credits, CreditsSaved: r.CreditsSaved,
			ToolCredits: r.ToolCredits, QuotaConsumed: r.QuotaConsumed, PreDeduct: r.PreDeduct,
			RefundStatus: statusMap[r.ID], CreateTime: r.CreateTime,
		})
	}
	return &vo.PageResult[vo.BillingRecordVO]{List: items, Total: total}, nil
}

// CreditLogs 余额流水分页（Decimal 整数语义字段按 python 序列化为字符串）
func (s *BillingService) CreditLogs(ctx context.Context, userID int64, q *bo.CreditLogQuery) (*vo.PageResult[vo.CreditLogVO], error) {
	page, size := q.PageNum, q.PageSize
	start, err := parseBillingTime(q.DateStart)
	if err != nil {
		return nil, err
	}
	end, err := parseBillingTime(q.DateEnd)
	if err != nil {
		return nil, err
	}
	logs, total, err := s.repo.PaginateCreditLogs(ctx, userID, page, size, q.Source, start, end)
	if err != nil {
		return nil, err
	}
	items := make([]vo.CreditLogVO, 0, len(logs))
	for i := range logs {
		log := &logs[i]
		items = append(items, vo.CreditLogVO{
			ID: log.ID, UserID: log.UserID, Source: log.Source,
			Amount:       strconv.FormatInt(log.Amount, 10),
			BalanceAfter: strconv.FormatInt(log.BalanceAfter, 10),
			RelatedID:    log.RelatedID, Reason: log.Reason, OperatorID: log.OperatorID,
			CreateTime: log.CreateTime,
		})
	}
	return &vo.PageResult[vo.CreditLogVO]{List: items, Total: total}, nil
}

// Bill 月结账单（Redis 优先，未命中重新生成；非当前月份且无任何记录视为账单不存在）
func (s *BillingService) Bill(ctx context.Context, userID int64, month string) (*vo.BillVO, error) {
	monthStart, monthEnd, err := monthBounds(month)
	if err != nil {
		return nil, err
	}
	if s.redis == nil {
		return nil, common.NewBizError(common.MIDDLEWARE_SERVICE_ERROR, "缓存服务不可用")
	}
	cacheKey := fmt.Sprintf(billKeyFmt, userID, month)
	if raw, getErr := s.redis.Get(ctx, cacheKey).Bytes(); getErr == nil && len(raw) > 0 {
		if cached, unmarshalErr := billFromCache(raw); unmarshalErr == nil {
			if isEmptyBill(cached, month) {
				return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "账单不存在")
			}
			return cached, nil
		}
	}

	bill, err := s.generateBill(ctx, userID, month, monthStart, monthEnd)
	if err != nil {
		return nil, err
	}
	// 空账期不入缓存，否则后续查询命中全 0 缓存会错误返回成功而非 A0401
	if isEmptyBill(bill, month) {
		_ = s.redis.Del(ctx, cacheKey).Err()
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "账单不存在")
	}
	if payload, marshalErr := billToCache(bill); marshalErr == nil {
		_ = s.redis.Set(ctx, cacheKey, payload, billCacheTTL).Err()
	}
	return bill, nil
}

// billCacheDTO 账单缓存传输对象：键名一律 snake_case，与 python（bill_service.py model_dump(mode="json")）
// 和 java（AiJsonUtils SNAKE_MAPPER）写入的缓存互认；与对外 camelCase 的 vo.BillVO 严格隔离，
// 改对外契约不得影响缓存格式（反之亦然）。
type billCacheDTO struct {
	UserID        int64             `json:"user_id"`
	Month         string            `json:"month"`
	TotalConsume  int64             `json:"total_consume"`
	TotalRecharge int64             `json:"total_recharge"`
	TotalRefund   int64             `json:"total_refund"`
	BalanceStart  cachedDecimalText `json:"balance_start"`
	BalanceEnd    cachedDecimalText `json:"balance_end"`
	ItemSummary   map[string]int64  `json:"item_summary"`
}

// cachedDecimalText 缓存金额字段：python 的 Decimal 序列化为 JSON 字符串、java 的 BigDecimal 为 JSON 数字，
// 反序列化两者都接受，序列化统一输出字符串（Jackson 可将字符串转 BigDecimal，python 同形）。
type cachedDecimalText string

func (d *cachedDecimalText) UnmarshalJSON(raw []byte) error {
	text := strings.TrimSpace(string(raw))
	if len(text) >= 2 && text[0] == '"' {
		var unquoted string
		if err := json.Unmarshal(raw, &unquoted); err != nil {
			return err
		}
		*d = cachedDecimalText(unquoted)
		return nil
	}
	if text == "null" {
		text = "0"
	}
	*d = cachedDecimalText(text)
	return nil
}

func billToCache(bill *vo.BillVO) ([]byte, error) {
	return json.Marshal(billCacheDTO{
		UserID: bill.UserID, Month: bill.Month,
		TotalConsume: bill.TotalConsume, TotalRecharge: bill.TotalRecharge,
		TotalRefund:  bill.TotalRefund,
		BalanceStart: cachedDecimalText(bill.BalanceStart), BalanceEnd: cachedDecimalText(bill.BalanceEnd),
		ItemSummary: bill.ItemSummary,
	})
}

func billFromCache(raw []byte) (*vo.BillVO, error) {
	var dto billCacheDTO
	if err := json.Unmarshal(raw, &dto); err != nil {
		return nil, err
	}
	return &vo.BillVO{
		UserID: dto.UserID, Month: dto.Month,
		TotalConsume: dto.TotalConsume, TotalRecharge: dto.TotalRecharge,
		TotalRefund:  dto.TotalRefund,
		BalanceStart: string(dto.BalanceStart), BalanceEnd: string(dto.BalanceEnd),
		ItemSummary: dto.ItemSummary,
	}, nil
}

func (s *BillingService) generateBill(
	ctx context.Context, userID int64, month string, monthStart, monthEnd time.Time,
) (*vo.BillVO, error) {
	byType, err := s.repo.SumBillingByBillType(ctx, userID, monthStart, monthEnd)
	if err != nil {
		return nil, err
	}
	itemSummary := make(map[string]int64, len(byType))
	var totalConsume int64
	for _, row := range byType {
		itemSummary[row.BillType] = row.Credits
		totalConsume += row.Credits
	}

	bySource, err := s.repo.SumCreditLogBySource(ctx, userID, &monthStart, &monthEnd)
	if err != nil {
		return nil, err
	}
	var totalRecharge int64
	for source, amount := range bySource {
		if rechargeSources[source] {
			totalRecharge += amount
		}
	}

	balanceStart, err := s.repo.BalanceAtOrBefore(ctx, userID, monthStart.Add(-time.Second))
	if err != nil {
		return nil, err
	}
	balanceEnd, err := s.repo.BalanceAtOrBefore(ctx, userID, monthEnd)
	if err != nil {
		return nil, err
	}

	return &vo.BillVO{
		UserID: userID, Month: month,
		TotalConsume: totalConsume, TotalRecharge: totalRecharge, TotalRefund: bySource["refund"],
		BalanceStart: strconv.FormatInt(balanceStart, 10),
		BalanceEnd:   strconv.FormatInt(balanceEnd, 10),
		ItemSummary:  itemSummary,
	}, nil
}

// ==================== 退款申请 ====================

// ApplyRefund 用户申请误扣退款：记录归属校验 → 防重复申请 → 建单（待审核）
func (s *BillingService) ApplyRefund(ctx context.Context, userID int64, form *bo.BillingRefundApplyForm) (*vo.RefundVO, error) {
	if form.Amount <= 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "退款积分数必须大于 0")
	}
	record, err := s.repo.GetBilling(ctx, form.BillingID)
	if err != nil {
		return nil, err
	}
	if record == nil || record.UserID != userID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "计费记录不存在")
	}
	if form.Amount > record.Credits {
		return nil, common.NewBizError(common.PARAM_ERROR, "退款积分数超过该笔记录实际消耗")
	}
	existing, err := s.repo.GetPendingRefundByBilling(ctx, form.BillingID)
	if err != nil {
		return nil, err
	}
	if existing != nil {
		return nil, common.NewBizError(common.AI_REFUND_ALREADY_EXISTS, common.AI_REFUND_ALREADY_EXISTS.Msg)
	}
	refund := &model.SysAiRefund{
		UserID: userID, BillingID: form.BillingID, Amount: form.Amount,
		Reason: form.Reason, Status: 1, CreateBy: &userID,
	}
	if err := s.repo.CreateRefund(ctx, refund); err != nil {
		return nil, err
	}
	return refuseToVO(refund), nil
}

// ListRefunds 退款申请分页列表（管理端审核中心）
func (s *BillingService) ListRefunds(ctx context.Context, q *bo.RefundQuery) (*vo.PageResult[vo.RefundVO], error) {
	page, size := q.PageNum, q.PageSize
	if q.Status != nil && (*q.Status < 1 || *q.Status > 3) {
		return nil, common.NewBizError(common.PARAM_ERROR, "退款状态仅支持 1/2/3")
	}
	start, err := parseBillingTime(q.DateStart)
	if err != nil {
		return nil, err
	}
	end, err := parseBillingTime(q.DateEnd)
	if err != nil {
		return nil, err
	}
	refunds, total, err := s.repo.PaginateRefunds(ctx, page, size, q.Status, q.UserID, start, end)
	if err != nil {
		return nil, err
	}
	items := make([]vo.RefundVO, 0, len(refunds))
	for i := range refunds {
		items = append(items, *refuseToVO(&refunds[i]))
	}
	return &vo.PageResult[vo.RefundVO]{List: items, Total: total}, nil
}

// AuditRefund 管理员审核：通过则余额回补（不清日/月已用计数），驳回仅标记
func (s *BillingService) AuditRefund(ctx context.Context, refundID int64, approved bool, remark *string, operatorID int64) (*vo.RefundVO, error) {
	refund, err := s.repo.GetRefund(ctx, refundID)
	if err != nil {
		return nil, err
	}
	if refund == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "退款申请不存在")
	}
	if refund.Status != 1 {
		return nil, common.NewBizError(common.REFUND_AUDIT_FAILED, "该退款申请已审核")
	}

	status := 3
	if approved {
		// 原计费记录必须存在，且该记录未补偿过（防重复回补余额）
		record, recErr := s.repo.GetBilling(ctx, refund.BillingID)
		if recErr != nil {
			return nil, recErr
		}
		if record == nil {
			return nil, common.NewBizError(common.REFUND_AUDIT_FAILED, "原计费记录不存在")
		}
		approvedBefore, existErr := s.repo.HasApprovedRefund(ctx, refund.BillingID, refund.ID)
		if existErr != nil {
			return nil, existErr
		}
		if approvedBefore {
			return nil, common.NewBizError(common.REFUND_AUDIT_FAILED, "原计费记录已退款")
		}
		reason := "退款: " + refund.Reason
		if _, incErr := s.increaseBalance(
			ctx, refund.UserID, int64(refund.Amount), "refund",
			&refund.BillingID, &reason, &operatorID,
		); incErr != nil {
			return nil, incErr
		}
		status = 2
	}

	fields := map[string]any{"status": status, "auditor_id": operatorID, "audit_remark": remark, "update_time": time.Now()}
	if err := s.repo.UpdateRefund(ctx, refundID, fields); err != nil {
		return nil, err
	}
	refund.Status = status
	refund.AuditorID = &operatorID
	refund.AuditRemark = remark
	updated := time.Now()
	refund.UpdateTime = &updated
	return refuseToVO(refund), nil
}

// ==================== 管理员操作 ====================

// Adjust 管理员手动调整积分（正数增加/负数扣减），返回调整后余额
func (s *BillingService) Adjust(ctx context.Context, operatorID int64, form *bo.CreditAdjustForm) (*vo.BillingBalanceVO, error) {
	if form.Amount == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "调整积分数不能为 0")
	}
	exists, err := s.repo.ExistsUser(ctx, form.UserID)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "用户不存在")
	}
	reason := form.Reason
	if _, err := s.increaseBalance(ctx, form.UserID, int64(form.Amount), "admin_adjust", nil, &reason, &operatorID); err != nil {
		return nil, err
	}
	return s.Balance(ctx, form.UserID)
}

// Stats 管理员分维度统计（user/model/billType/day）
func (s *BillingService) Stats(ctx context.Context, q *bo.BillingStatQuery) ([]vo.BillingStatVO, error) {
	groupBy := q.GroupBy
	if groupBy == "" {
		groupBy = "model"
	}
	// 白名单前置校验：非法维度不得穿透到仓储层（与 python billing_stat_service.stats / java 一致，报 A0400）
	switch groupBy {
	case "user", "model", "billType", "day":
	default:
		return nil, common.NewBizError(common.PARAM_ERROR, "groupBy 仅支持 user/model/billType/day")
	}
	start, err := parseBillingTime(q.DateStart)
	if err != nil {
		return nil, err
	}
	end, err := parseBillingTime(q.DateEnd)
	if err != nil {
		return nil, err
	}
	rows, err := s.repo.StatsBillingByDimension(ctx, groupBy, q.UserID, q.ModelID, q.BillType, start, end)
	if err != nil {
		return nil, err
	}
	results := make([]vo.BillingStatVO, 0, len(rows))
	for _, row := range rows {
		cacheHitRate := 0.0
		if row.TotalInputTokens > 0 {
			cacheHitRate = round4(float64(row.ChatCachedTokens) / float64(row.TotalInputTokens))
		}
		results = append(results, vo.BillingStatVO{
			Dimension:         row.Dimension,
			TotalCredits:      row.TotalCredits,
			TotalInputTokens:  row.TotalInputTokens,
			TotalOutputTokens: row.TotalOutputToks,
			CacheHitRate:      cacheHitRate,
			CreditsSaved:      row.CreditsSaved,
			DegradationCount:  row.DegradationCount,
		})
	}
	return results, nil
}

// Anomalies 异常计费记录分页（管理端）
func (s *BillingService) Anomalies(ctx context.Context, q *bo.AnomalyQuery) (*vo.PageResult[vo.AnomalyVO], error) {
	page, size := q.PageNum, q.PageSize
	start, err := parseBillingTime(q.DateStart)
	if err != nil {
		return nil, err
	}
	end, err := parseBillingTime(q.DateEnd)
	if err != nil {
		return nil, err
	}
	items, total, err := s.repo.PaginateAnomalies(ctx, page, size, q.UserID, q.AnomalyType, q.Status, start, end)
	if err != nil {
		return nil, err
	}
	result := make([]vo.AnomalyVO, 0, len(items))
	for i := range items {
		item := &items[i]
		result = append(result, vo.AnomalyVO{
			ID: item.ID, UserID: item.UserID, BillingID: item.BillingID,
			AnomalyType: item.AnomalyType, Detail: item.Detail, Status: item.Status,
			TriggerAt: item.TriggerAt, CreateTime: item.CreateTime,
		})
	}
	return &vo.PageResult[vo.AnomalyVO]{List: result, Total: total}, nil
}

// ==================== 辅助 ====================

func refuseToVO(refund *model.SysAiRefund) *vo.RefundVO {
	return &vo.RefundVO{
		ID: refund.ID, UserID: refund.UserID, BillingID: refund.BillingID,
		Amount: refund.Amount, Reason: refund.Reason, Status: refund.Status,
		AuditorID: refund.AuditorID, AuditRemark: refund.AuditRemark,
		CreateTime: refund.CreateTime, UpdateTime: refund.UpdateTime,
	}
}

// parseBillingTime 解析前端日期字符串（"%Y-%m-%d %H:%M:%S" / "%Y-%m-%d"），非法报参数错误
func parseBillingTime(value string) (*time.Time, error) {
	if value == "" {
		return nil, nil
	}
	for _, layout := range []string{"2006-01-02 15:04:05", "2006-01-02"} {
		if parsed, err := time.ParseInLocation(layout, value, time.Local); err == nil {
			return &parsed, nil
		}
	}
	return nil, common.NewBizError(common.PARAM_ERROR, "时间格式不正确: "+value)
}

// monthBounds 解析月份 → (月初, 月末 23:59:59)。
// python 侧为 strptime("%Y-%m")，对非零填充的 "2026-1" 同样接受（得到同一个月），
// 故此处按两种布局依次尝试：否则 "2026-1" 在 go 报 A0400、在 python 走成"空账期 A0401"，形成分叉。
func monthBounds(month string) (time.Time, time.Time, error) {
	for _, layout := range []string{"2006-01", "2006-1"} {
		if start, err := time.ParseInLocation(layout, month, time.Local); err == nil {
			return start, start.AddDate(0, 1, 0).Add(-time.Second), nil
		}
	}
	return time.Time{}, time.Time{}, common.NewBizError(common.PARAM_ERROR, "月份格式不正确，应为 YYYY-MM")
}

// isEmptyBill 账单是否为空账期（当前月份允许全 0，月初尚无数据属正常）
func isEmptyBill(bill *vo.BillVO, month string) bool {
	if month == time.Now().Format("2006-01") {
		return false
	}
	return bill.TotalConsume == 0 && bill.TotalRecharge == 0 && bill.TotalRefund == 0
}
