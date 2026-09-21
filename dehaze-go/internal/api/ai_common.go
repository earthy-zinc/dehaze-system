package api

import (
	"strconv"
	"time"

	aidomain "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

// parseID 解析路径参数中的整型 ID（非法时写入参数错误并返回 false）。
func parseID(c *gin.Context, name string) (int64, bool) {
	value, err := strconv.ParseInt(c.Param(name), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return 0, false
	}
	return value, true
}

// queryTimeRange 解析可选时间范围参数（startKey/endKey 为 query 参数名）：
// 未传返回 nil,true；任一格式非法写回参数错误并返回 false（对齐 python datetime 参数的 A0400 语义）。
func queryTimeRange(c *gin.Context, startKey, endKey string) (*time.Time, *time.Time, bool) {
	start, ok := aidomain.QueryTimePtr(c.Query(startKey))
	if !ok {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, startKey+" 时间格式不正确"))
		return nil, nil, false
	}
	end, ok := aidomain.QueryTimePtr(c.Query(endKey))
	if !ok {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, endKey+" 时间格式不正确"))
		return nil, nil, false
	}
	return start, end, true
}

// parseRangedOptionalInt 解析可选整型查询参数并校验闭区间 [lo, hi]
// （未传返回 nil,true；格式非法或越界返回 nil,false）——区间取自 python 对应 Query 的 ge/le。
func parseRangedOptionalInt(raw string, lo, hi int) (*int, bool) {
	value, ok := parseOptionalInt(raw)
	if !ok || (value != nil && (*value < lo || *value > hi)) {
		return nil, false
	}
	return value, true
}

// parseOptionalInt 解析可选整型查询参数（未传返回 nil,true；格式非法返回 nil,false）。
func parseOptionalInt(raw string) (*int, bool) {
	if raw == "" {
		return nil, true
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return nil, false
	}
	return &value, true
}

// parseMessageCursor 解析会话消息列表的游标分页参数：
//   - before：可选，>=1，仅返回 id < before 的消息；缺省取最新一页；
//   - limit：可选，默认 50，范围 1..100。
//
// 非数字或越界一律写回 A0400。刻意不复用 parsePagination*：该端点已弃用 pageNum/pageSize
// （python 侧对应游标契约），旧分页参数作为无关参数被忽略。
func parseMessageCursor(c *gin.Context) (before *int64, limit int, ok bool) {
	limit = 50
	if raw := c.Query("before"); raw != "" {
		value, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || value < 1 {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "分页参数不合法：before>=1，1<=limit<=100"))
			return nil, 0, false
		}
		before = &value
	}
	if raw := c.Query("limit"); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value < 1 || value > 100 {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "分页参数不合法：before>=1，1<=limit<=100"))
			return nil, 0, false
		}
		limit = value
	}
	return before, limit, true
}
