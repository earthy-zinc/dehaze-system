package model

type SysPackage struct {
	BaseModel
	Name             string `gorm:"column:name;type:varchar(32);not null;uniqueIndex:uk_name;comment:套餐名称" json:"name"`
	LevelCode        string `gorm:"column:level_code;type:varchar(16);not null;index:idx_level_code;comment:会员等级" json:"levelCode"`
	Period           string `gorm:"column:period;type:varchar(16);not null;index:idx_period;comment:计费周期" json:"period"`
	PeriodDays       int    `gorm:"column:period_days;type:int;not null;comment:有效期天数" json:"periodDays"`
	OriginalPrice    int64  `gorm:"column:original_price;type:bigint;not null;comment:原价（分）" json:"originalPrice"`
	SalePrice        int64  `gorm:"column:sale_price;type:bigint;not null;comment:促销价（分）" json:"salePrice"`
	Description      string `gorm:"column:description;type:varchar(256);comment:套餐描述" json:"description"`
	BenefitOverrides string `gorm:"column:benefit_overrides;type:json;comment:权益覆盖项" json:"benefitOverrides"`
	SalesCount       int64  `gorm:"column:sales_count;type:bigint;not null;default:0;comment:销量" json:"salesCount"`
	Sort             int    `gorm:"column:sort;type:int;not null;default:0;comment:排序值" json:"sort"`
	Status           int8   `gorm:"column:status;type:tinyint;not null;default:0;index:idx_status;comment:上下架状态" json:"status"`
	Deleted          int8   `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysPackage) TableName() string {
	return "sys_package"
}
