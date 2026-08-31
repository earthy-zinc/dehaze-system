package favorite

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	repomocks "github.com/earthyzinc/dehaze-go/internal/repository/mocks"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	servicemocks "github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// newCapacityService 组装 getCapacity 所需依赖：memberRepo 提供用户等级，dictSvc 提供容量字典。
func newCapacityService(member memberrepo.IMemberRepository, dictSvc dictservice.IDictService) *FavoriteService {
	return &FavoriteService{
		memberRepo: member,
		dictSvc:    dictSvc,
	}
}

// capOptions 构造 favorite_capacity 字典选项：Label 为键，Value 为容量值。
func capOptions(key, val string) []vo.Option {
	return []vo.Option{{Label: key, Value: val}}
}

func TestGetCapacity_LevelMapping(t *testing.T) {
	cases := []struct {
		levelCode string
		dictKey   string
		dictVal   string
		want      int
	}{
		{"level_0", "default", "200", 200},
		{"level_1", "vip1", "500", 500},
		{"level_2", "vip2", "1000", 1000},
		{"level_3", "svip", "3000", 3000},
	}

	for _, c := range cases {
		t.Run(c.levelCode, func(t *testing.T) {
			memberMock := repomocks.NewMockIMemberRepository(t)
			memberMock.EXPECT().FindByUserID(mock.Anything, int64(1)).
				Return(&model.SysMember{LevelCode: c.levelCode}, nil)

			dictMock := servicemocks.NewMockIDictService(t)
			dictMock.EXPECT().GetByTypeCode(mock.Anything, favoriteCapacityDictType).
				Return(capOptions(c.dictKey, c.dictVal), nil)

			svc := newCapacityService(memberMock, dictMock)
			assert.Equal(t, c.want, svc.getCapacity(context.Background(), 1))
		})
	}
}

func TestGetCapacity_MemberNotFound_FallsBackToDefault(t *testing.T) {
	memberMock := repomocks.NewMockIMemberRepository(t)
	memberMock.EXPECT().FindByUserID(mock.Anything, int64(1)).
		Return(nil, nil)

	dictMock := servicemocks.NewMockIDictService(t)
	dictMock.EXPECT().GetByTypeCode(mock.Anything, favoriteCapacityDictType).
		Return(capOptions("default", "200"), nil)

	svc := newCapacityService(memberMock, dictMock)
	assert.Equal(t, 200, svc.getCapacity(context.Background(), 1))
}

func TestGetCapacity_UnknownLevel_FallsBackToDefaultKey(t *testing.T) {
	memberMock := repomocks.NewMockIMemberRepository(t)
	memberMock.EXPECT().FindByUserID(mock.Anything, int64(1)).
		Return(&model.SysMember{LevelCode: "level_9"}, nil)

	// 未知等级应走 default 键
	dictMock := servicemocks.NewMockIDictService(t)
	dictMock.EXPECT().GetByTypeCode(mock.Anything, favoriteCapacityDictType).
		Return(capOptions("default", "200"), nil)

	svc := newCapacityService(memberMock, dictMock)
	assert.Equal(t, 200, svc.getCapacity(context.Background(), 1))
}

func TestGetCapacity_DictMissingKey_FallsBackTo200(t *testing.T) {
	memberMock := repomocks.NewMockIMemberRepository(t)
	memberMock.EXPECT().FindByUserID(mock.Anything, int64(1)).
		Return(&model.SysMember{LevelCode: "level_3"}, nil)

	// svip 键缺失 → 回退默认 200
	dictMock := servicemocks.NewMockIDictService(t)
	dictMock.EXPECT().GetByTypeCode(mock.Anything, favoriteCapacityDictType).
		Return(capOptions("default", "200"), nil)

	svc := newCapacityService(memberMock, dictMock)
	assert.Equal(t, 200, svc.getCapacity(context.Background(), 1))
}

func TestGetCapacity_DictNilService_FallsBackTo200(t *testing.T) {
	memberMock := repomocks.NewMockIMemberRepository(t)
	memberMock.EXPECT().FindByUserID(mock.Anything, int64(1)).
		Return(&model.SysMember{LevelCode: "level_1"}, nil)

	svc := newCapacityService(memberMock, nil)
	assert.Equal(t, 200, svc.getCapacity(context.Background(), 1))
}
