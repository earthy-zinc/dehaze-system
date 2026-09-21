package ai

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	goredis "github.com/redis/go-redis/v9"
	"gorm.io/gorm"
)

// Redis Key 约定与 dehaze-python app/infrastructure/provider/provider_health_service.py 完全一致，
// 三端共享同一批键：熔断解除、健康开关、模型/供应商缓存均需跨端可见，不能走 Go 多级缓存的 L1。
const (
	modelListCacheKey    = "ai:model:list"
	providerListCacheKey = "ai:provider:list"
	cacheTTLHour         = 3600 * time.Second

	// 缓存失效广播频道（三端共用；MCP 变更后据此通知 python 失效推理图缓存）
	cacheInvalidationChannel = "cache:invalidation"

	// graphInvalidateMsgType 推理图缓存失效消息类型（python 订阅同频道后失效进程内图缓存）
	graphInvalidateMsgType = "ai_graph_invalidate"

	userLevelCachePrefix = "user:level:"
	userLevelCacheTTL    = 1800 * time.Second

	healthEnabledKeyFmt = "ai:provider:%d:health_enabled"
	circuitKeyFmt       = "ai:provider:%d:circuit_open"
	recoveryKeyFmt      = "ai:provider:%d:circuit_recovery"
	probeKeyFmt         = "ai:provider:%d:half_open_probe"
	streakKeyFmt        = "ai:provider:%d:fail_streak"
	windowKeyFmt        = "ai:provider:%d:window"
	latencyKeyFmt       = "ai:provider:%d:latency"
	snapshotKeyFmt      = "ai:provider:%d:health"
	thresholdsCacheKey  = "ai:provider:health:thresholds"

	healthDictType = "ai_provider_health"

	windowBuckets     = 24
	bucketSeconds     = 3600
	latencyWindowSize = 500
	snapshotTTL       = 60 * time.Second
	thresholdsTTL     = 300 * time.Second
)

// HealthThresholds 供应商熔断阈值（与 config/sql/data/sys_dict.sql 的 ai_provider_health 同源默认值）
type HealthThresholds struct {
	ErrorRateWarn      float64 `json:"error_rate_warn"`
	ErrorRateOpen      float64 `json:"error_rate_open"`
	MinWindowCalls     int     `json:"min_window_calls"`
	ConsecutiveFails   int     `json:"consecutive_failures"`
	CircuitCooldownSec int     `json:"circuit_cooldown"`
}

var seedThresholds = HealthThresholds{
	ErrorRateWarn:      0.10,
	ErrorRateOpen:      0.30,
	MinWindowCalls:     20,
	ConsecutiveFails:   5,
	CircuitCooldownSec: 60,
}

// HealthService 供应商健康与熔断读取（健康数据由 Python 调用链路实时聚合，Go 只读/清理）
type HealthService struct {
	db *gorm.DB
}

func NewHealthService(db *gorm.DB) *HealthService {
	return &HealthService{db: db}
}

func redisClient() *goredis.Client {
	return redis.GetClient()
}

func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}

// Snapshot 读取供应商健康快照；缓存 miss 时按滑动窗口重算并回填（对齐 python get_health_snapshot）
func (h *HealthService) Snapshot(ctx context.Context, providerID int64) map[string]any {
	client := redisClient()
	if client == nil {
		return map[string]any{"status": "healthy", "circuit_open": false, "success_rate": 1.0, "error_rate": 0.0, "limit_rate": 0.0, "p95_latency_ms": 0}
	}
	key := fmt.Sprintf(snapshotKeyFmt, providerID)
	if raw, err := client.Get(ctx, key).Bytes(); err == nil && len(raw) > 0 {
		var cached map[string]any
		if json.Unmarshal(raw, &cached) == nil {
			return cached
		}
	}

	thresholds := h.Thresholds(ctx)
	enabled := h.HealthCheckEnabled(ctx, providerID)
	circuitOpen := client.Exists(ctx, fmt.Sprintf(circuitKeyFmt, providerID)).Val() > 0
	total, failed, limit := h.windowCounts(ctx, providerID)
	p95 := h.p95Latency(ctx, providerID)

	status := "healthy"
	if enabled && !circuitOpen && total >= thresholds.MinWindowCalls {
		errorRate := float64(failed) / float64(total)
		if errorRate >= thresholds.ErrorRateOpen {
			status = "open"
		} else if errorRate >= thresholds.ErrorRateWarn {
			status = "suspicious"
		}
	} else if enabled && circuitOpen {
		status = "open"
	}

	snapshot := map[string]any{
		"status":          status,
		"circuit_open":    circuitOpen,
		"total_calls_24h": total,
		"success_rate":    rateOr(total-failed, total, 1.0),
		"error_rate":      rateOr(failed, total, 0.0),
		"limit_rate":      rateOr(limit, total, 0.0),
		"p95_latency_ms":  p95,
	}
	if b, err := json.Marshal(snapshot); err == nil {
		client.Set(ctx, key, b, snapshotTTL)
	}
	return snapshot
}

// Status 只取健康状态字符串（列表展示用）；供应商不存在快照时按健康处理
func (h *HealthService) Status(ctx context.Context, providerID int64) string {
	snapshot := h.Snapshot(ctx, providerID)
	if status, ok := snapshot["status"].(string); ok && status != "" {
		return status
	}
	return "healthy"
}

func rateOr(numerator, total int, empty float64) float64 {
	if total == 0 {
		return empty
	}
	return round4(float64(numerator) / float64(total))
}

func (h *HealthService) windowCounts(ctx context.Context, providerID int64) (int, int, int) {
	client := redisClient()
	if client == nil {
		return 0, 0, 0
	}
	bucket := time.Now().Unix() / bucketSeconds
	fields := make([]string, 0, windowBuckets*3)
	for offset := 0; offset < windowBuckets; offset++ {
		for _, suffix := range []string{"t", "f", "l"} {
			fields = append(fields, fmt.Sprintf("%d:%s", bucket-int64(offset), suffix))
		}
	}
	values, err := client.HMGet(ctx, fmt.Sprintf(windowKeyFmt, providerID), fields...).Result()
	if err != nil {
		return 0, 0, 0
	}
	total, failed, limit := 0, 0, 0
	for i := 0; i < windowBuckets; i++ {
		total += intValue(values[i*3])
		failed += intValue(values[i*3+1])
		limit += intValue(values[i*3+2])
	}
	return total, failed, limit
}

func intValue(v any) int {
	if v == nil {
		return 0
	}
	n, err := strconv.Atoi(fmt.Sprintf("%v", v))
	if err != nil {
		return 0
	}
	return n
}

func (h *HealthService) p95Latency(ctx context.Context, providerID int64) int {
	client := redisClient()
	if client == nil {
		return 0
	}
	raw, err := client.LRange(ctx, fmt.Sprintf(latencyKeyFmt, providerID), 0, -1).Result()
	if err != nil || len(raw) == 0 {
		return 0
	}
	values := make([]int, 0, len(raw))
	for _, s := range raw {
		if n, convErr := strconv.Atoi(s); convErr == nil {
			values = append(values, n)
		}
	}
	if len(values) == 0 {
		return 0
	}
	sort.Ints(values)
	idx := int(float64(len(values))*0.95) - 1
	if idx < 0 {
		idx = 0
	}
	return values[idx]
}

// Thresholds 熔断阈值：Redis 缓存 → sys_dict（默认为种子值，与 python 同源）
func (h *HealthService) Thresholds(ctx context.Context) HealthThresholds {
	client := redisClient()
	if client != nil {
		if raw, err := client.Get(ctx, thresholdsCacheKey).Bytes(); err == nil && len(raw) > 0 {
			var cached HealthThresholds
			if json.Unmarshal(raw, &cached) == nil && cached.MinWindowCalls > 0 {
				return cached
			}
		}
	}
	thresholds := seedThresholds
	if h.db != nil {
		var rows []struct {
			Name  string `gorm:"column:name"`
			Value string `gorm:"column:value"`
		}
		err := h.db.WithContext(ctx).Table("sys_dict").
			Select("name, value").
			Where("type_code = ? AND status = 1 AND deleted = 0", healthDictType).
			Scan(&rows).Error
		if err == nil {
			for _, row := range rows {
				applyThreshold(&thresholds, row.Name, row.Value)
			}
		}
	}
	if client != nil {
		if b, err := json.Marshal(thresholds); err == nil {
			client.Set(ctx, thresholdsCacheKey, b, thresholdsTTL)
		}
	}
	return thresholds
}

func applyThreshold(t *HealthThresholds, name, value string) {
	switch name {
	case "error_rate_warn":
		if f, err := strconv.ParseFloat(value, 64); err == nil {
			t.ErrorRateWarn = f
		}
	case "error_rate_open":
		if f, err := strconv.ParseFloat(value, 64); err == nil {
			t.ErrorRateOpen = f
		}
	case "min_window_calls":
		if n, err := strconv.Atoi(value); err == nil {
			t.MinWindowCalls = n
		}
	case "consecutive_failures":
		if n, err := strconv.Atoi(value); err == nil {
			t.ConsecutiveFails = n
		}
	case "circuit_cooldown":
		if n, err := strconv.Atoi(value); err == nil {
			t.CircuitCooldownSec = n
		}
	}
}

// HealthCheckEnabled 健康检查开关（供应商 CRUD 写入缓存；缺省视为开启）
func (h *HealthService) HealthCheckEnabled(ctx context.Context, providerID int64) bool {
	client := redisClient()
	if client == nil {
		return true
	}
	val, err := client.Get(ctx, fmt.Sprintf(healthEnabledKeyFmt, providerID)).Result()
	if err != nil {
		return true
	}
	return val != "0"
}

// SetHealthCheckEnabled 供应商 CRUD 时写入健康检查开关缓存
func (h *HealthService) SetHealthCheckEnabled(ctx context.Context, providerID int64, enabled bool) {
	client := redisClient()
	if client == nil {
		return
	}
	flag := "0"
	if enabled {
		flag = "1"
	}
	client.Set(ctx, fmt.Sprintf(healthEnabledKeyFmt, providerID), flag, 0)
}

// CloseCircuit 管理员手动解除熔断：清熔断标记/恢复周期/探测租约/连续失败计数与快照缓存
func (h *HealthService) CloseCircuit(ctx context.Context, providerID int64) {
	client := redisClient()
	if client == nil {
		return
	}
	client.Del(ctx,
		fmt.Sprintf(circuitKeyFmt, providerID),
		fmt.Sprintf(recoveryKeyFmt, providerID),
		fmt.Sprintf(probeKeyFmt, providerID),
		fmt.Sprintf(streakKeyFmt, providerID),
		fmt.Sprintf(snapshotKeyFmt, providerID),
	)
}

// ClearProviderHealth 删除供应商健康相关 Key（删除供应商时清理）
func (h *HealthService) ClearProviderHealth(ctx context.Context, providerID int64) {
	client := redisClient()
	if client == nil {
		return
	}
	client.Del(ctx,
		fmt.Sprintf(circuitKeyFmt, providerID),
		fmt.Sprintf(recoveryKeyFmt, providerID),
		fmt.Sprintf(probeKeyFmt, providerID),
		fmt.Sprintf(streakKeyFmt, providerID),
		fmt.Sprintf(windowKeyFmt, providerID),
		fmt.Sprintf(latencyKeyFmt, providerID),
		fmt.Sprintf(snapshotKeyFmt, providerID),
		fmt.Sprintf(healthEnabledKeyFmt, providerID),
	)
}

// ==================== 密钥加解密（跨端互认：AES-256-CBC + SHA256 派生密钥 + PKCS7 + base64(iv+ct)） ====================

func secretCipherKey() []byte {
	sum := sha256.Sum256([]byte(os.Getenv("AI_PROVIDER_KEY_ENCRYPTION_KEY")))
	return sum[:]
}

func EncryptSecret(plain string) (string, error) {
	block, err := aes.NewCipher(secretCipherKey())
	if err != nil {
		return "", err
	}
	iv := make([]byte, aes.BlockSize)
	if _, err := rand.Read(iv); err != nil {
		return "", err
	}
	padded := pkcs7Pad([]byte(plain), aes.BlockSize)
	out := make([]byte, aes.BlockSize+len(padded))
	copy(out, iv)
	cipher.NewCBCEncrypter(block, iv).CryptBlocks(out[aes.BlockSize:], padded)
	return base64.StdEncoding.EncodeToString(out), nil
}

func DecryptSecret(encoded string) (string, error) {
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return "", err
	}
	if len(raw) < aes.BlockSize || (len(raw)-aes.BlockSize)%aes.BlockSize != 0 {
		return "", fmt.Errorf("密文长度非法")
	}
	block, err := aes.NewCipher(secretCipherKey())
	if err != nil {
		return "", err
	}
	iv, cipherText := raw[:aes.BlockSize], raw[aes.BlockSize:]
	plain := make([]byte, len(cipherText))
	cipher.NewCBCDecrypter(block, iv).CryptBlocks(plain, cipherText)
	return string(pkcs7Unpad(plain, aes.BlockSize)), nil
}

func pkcs7Pad(data []byte, blockSize int) []byte {
	padding := blockSize - len(data)%blockSize
	for i := 0; i < padding; i++ {
		data = append(data, byte(padding))
	}
	return data
}

func pkcs7Unpad(data []byte, blockSize int) []byte {
	if len(data) == 0 {
		return data
	}
	padding := int(data[len(data)-1])
	if padding == 0 || padding > blockSize || padding > len(data) {
		return data
	}
	return data[:len(data)-padding]
}

// HashSecret SHA256 hex（查重用，与 python hash_key 一致）
func HashSecret(plain string) string {
	sum := sha256.Sum256([]byte(plain))
	return hex.EncodeToString(sum[:])
}

// MaskSecret 截取前 8 位 + "..."（展示用，与 python mask_key 一致）
func MaskSecret(plain string) string {
	if len(plain) > 8 {
		return plain[:8] + "..."
	}
	return plain + "..."
}

// ==================== 用户会员等级（模型 VIP 过滤，缓存键与 python 一致） ====================

var memberLevelMap = map[string]int{"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}

func (h *HealthService) UserLevel(ctx context.Context, userID int64) int {
	client := redisClient()
	key := userLevelCachePrefix + strconv.FormatInt(userID, 10)
	if client != nil {
		if raw, err := client.Get(ctx, key).Result(); err == nil {
			if n, convErr := strconv.Atoi(raw); convErr == nil {
				return n
			}
		}
	}
	level := 0
	if h.db != nil {
		var levelCode *string
		err := h.db.WithContext(ctx).Table("sys_member").
			Select("level_code").
			Where("user_id = ? AND deleted = 0", userID).
			Scan(&levelCode).Error
		if err == nil && levelCode != nil {
			if mapped, ok := memberLevelMap[*levelCode]; ok {
				level = mapped
			}
		}
	}
	if client != nil {
		client.Set(ctx, key, strconv.Itoa(level), userLevelCacheTTL)
	}
	return level
}

func clearModelCache(ctx context.Context) {
	if client := redisClient(); client != nil {
		client.Del(ctx, modelListCacheKey)
	}
}

func clearProviderCache(ctx context.Context) {
	if client := redisClient(); client != nil {
		client.Del(ctx, providerListCacheKey)
	}
}

// invalidateReasoningGraphs 广播推理图缓存失效（MCP 工具集变更后调用）。
//
// 推理图由 python 构建并缓存在进程内，Go 原生端改 MCP（共享库）不会触发其重建，
// 已建图仍持旧的外部工具集；广播后各 python 实例失效自己的图缓存。
func invalidateReasoningGraphs(ctx context.Context) {
	client := redisClient()
	if client == nil {
		return
	}
	payload, err := json.Marshal(map[string]any{"type": graphInvalidateMsgType, "senderId": "dehaze-go"})
	if err != nil {
		return
	}
	_ = client.Publish(ctx, cacheInvalidationChannel, payload).Err()
}
