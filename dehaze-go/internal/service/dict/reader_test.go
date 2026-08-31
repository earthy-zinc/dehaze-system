package dict_test

import (
	"context"
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/service/dict"
	"github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// options 构造字典项选项：Label 为字典项 Name（键），Value 为字典值（整型值字符串）。
// 约定见 config/sql/data/sys_dict.sql。
func options(pairs ...struct{ key, val string }) []vo.Option {
	out := make([]vo.Option, 0, len(pairs))
	for _, p := range pairs {
		out = append(out, vo.Option{Label: p.key, Value: p.val})
	}
	return out
}

func TestGetIntValue_Hit(t *testing.T) {
	svc := mocks.NewMockIDictService(t)
	svc.EXPECT().GetByTypeCode(mock.Anything, "member_growth_rules").
		Return(options(struct{ key, val string }{"sign_in_value", "3"}), nil)

	got := dict.GetIntValue(context.Background(), svc, "member_growth_rules", "sign_in_value", 99)
	assert.Equal(t, int64(3), got)
}

func TestGetIntValue_MissingKey_FallsBack(t *testing.T) {
	svc := mocks.NewMockIDictService(t)
	svc.EXPECT().GetByTypeCode(mock.Anything, "favorite_capacity").
		Return(options(struct{ key, val string }{"vip1", "500"}), nil)

	got := dict.GetIntValue(context.Background(), svc, "favorite_capacity", "svip", 200)
	assert.Equal(t, int64(200), got)
}

func TestGetIntValue_InvalidValue_FallsBack(t *testing.T) {
	svc := mocks.NewMockIDictService(t)
	svc.EXPECT().GetByTypeCode(mock.Anything, "member_growth_rules").
		Return(options(struct{ key, val string }{"sign_in_value", "abc"}), nil)

	got := dict.GetIntValue(context.Background(), svc, "member_growth_rules", "sign_in_value", 5)
	assert.Equal(t, int64(5), got)
}

func TestGetIntValue_Error_FallsBack(t *testing.T) {
	svc := mocks.NewMockIDictService(t)
	svc.EXPECT().GetByTypeCode(mock.Anything, "favorite_capacity").
		Return(nil, errors.New("db down"))

	got := dict.GetIntValue(context.Background(), svc, "favorite_capacity", "default", 200)
	assert.Equal(t, int64(200), got)
}

func TestGetIntValue_NilService_FallsBack(t *testing.T) {
	got := dict.GetIntValue(context.Background(), nil, "favorite_capacity", "default", 200)
	assert.Equal(t, int64(200), got)
}
