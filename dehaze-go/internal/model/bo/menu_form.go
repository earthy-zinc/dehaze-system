package bo

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/enum"
)

// MenuType 菜单类型，支持字符串枚举名和整数两种 JSON 表示
// 对齐 Java MenuTypeEnum 的 Jackson 序列化（字符串枚举名）
type MenuType int8

// UnmarshalJSON 支持反序列化字符串枚举名("CATALOG")或整数(2)
func (m *MenuType) UnmarshalJSON(data []byte) error {
	// 尝试作为字符串解析
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		if val, ok := enum.MenuTypeNameToValue[s]; ok {
			*m = MenuType(val)
			return nil
		}
		return fmt.Errorf("无效的菜单类型: %s", s)
	}

	// 尝试作为整数解析
	var i int8
	if err := json.Unmarshal(data, &i); err == nil {
		if _, ok := enum.MenuTypeEnumNames[int(i)]; ok {
			*m = MenuType(i)
			return nil
		}
		return fmt.Errorf("无效的菜单类型值: %d", i)
	}

	// 尝试作为数字字符串解析（如 "2"）
	if num, err := strconv.Atoi(string(data)); err == nil {
		if _, ok := enum.MenuTypeEnumNames[num]; ok {
			*m = MenuType(num)
			return nil
		}
	}

	return fmt.Errorf("无效的菜单类型: %s", string(data))
}

// MarshalJSON 序列化为字符串枚举名
func (m MenuType) MarshalJSON() ([]byte, error) {
	name := enum.GetMenuTypeEnumName(int(m))
	if name == "" {
		return json.Marshal(int(m))
	}
	return json.Marshal(name)
}

// Scan 实现 sql.Scanner 接口，支持 GORM 从数据库 tinyint 扫描
func (m *MenuType) Scan(value interface{}) error {
	if value == nil {
		*m = 0
		return nil
	}
	switch v := value.(type) {
	case int64:
		*m = MenuType(v)
	case int8:
		*m = MenuType(v)
	case int:
		*m = MenuType(v)
	case []byte:
		num, err := strconv.Atoi(string(v))
		if err != nil {
			return err
		}
		*m = MenuType(num)
	default:
		return fmt.Errorf("无法扫描 %T 到 MenuType", value)
	}
	return nil
}

var _ sql.Scanner = (*MenuType)(nil)

// MenuForm 菜单表单对象
type MenuForm struct {
	// 菜单ID
	ID *int64 `json:"id"`
	// 父菜单ID
	ParentID int64 `json:"parentId" binding:"min=0"`
	// 菜单名称
	Name string `json:"name" binding:"required,max=64,no_xss"`
	// 菜单类型(MENU/CATALOG/EXTLINK/BUTTON)
	Type MenuType `json:"type" binding:"required"`
	// 路由路径
	Path string `json:"path" binding:"omitempty,max=128"`
	// 组件路径(vue页面完整路径，省略.vue后缀)
	Component string `json:"component" binding:"omitempty,max=128"`
	// 权限标识
	Perm string `json:"perm" binding:"omitempty,max=128"`
	// 显示状态(1:显示;0:隐藏)
	Visible int `json:"visible" binding:"oneof=0 1"`
	// 排序(数字越小排名越靠前)
	Sort int `json:"sort" binding:"min=0"`
	// 菜单图标
	Icon string `json:"icon" binding:"omitempty,max=64"`
	// 跳转路径
	Redirect string `json:"redirect" binding:"omitempty,max=128"`
	// 【菜单】是否开启页面缓存(1:开启;0:关闭)
	KeepAlive int `json:"keepAlive" binding:"oneof=0 1"`
	// 【目录】只有一个子路由是否始终显示(1:是 0:否)
	AlwaysShow int `json:"alwaysShow" binding:"oneof=0 1"`
}
