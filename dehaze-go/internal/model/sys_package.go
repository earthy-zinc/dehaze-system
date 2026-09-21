package model

import "database/sql"

type SysPackage struct {
	BaseModel
	Name string `gorm:"column:name;type:varchar(32);not null;comment:套餐名称" json:"name"`
	// 商品类型创建后锁定（vip:会员卡;credit:积分卡），积分卡 level/period/period_days 为 NULL
	PackageType      string         `gorm:"column:package_type;type:varchar(16);not null;default:vip;comment:商品类型" json:"packageType"`
	LevelCode        sql.NullString `gorm:"column:level_code;type:varchar(16);comment:关联会员等级(积分卡为NULL)" json:"levelCode"`
	Period           sql.NullString `gorm:"column:period;type:varchar(16);comment:计费周期(积分卡为NULL)" json:"period"`
	PeriodDays       sql.NullInt64  `gorm:"column:period_days;type:int;comment:有效期天数(积分卡为NULL)" json:"periodDays"`
	CreditAmount     sql.NullInt64  `gorm:"column:credit_amount;type:bigint;comment:可得积分数量(会员卡为NULL)" json:"creditAmount"`
	OriginalPrice    int64          `gorm:"column:original_price;type:bigint;not null;comment:原价（分）" json:"originalPrice"`
	SalePrice        int64          `gorm:"column:sale_price;type:bigint;not null;comment:促销价（分）" json:"salePrice"`
	Description      string         `gorm:"column:description;type:varchar(256);comment:套餐描述" json:"description"`
	BenefitOverrides sql.NullString `gorm:"column:benefit_overrides;type:json;comment:权益覆盖项" json:"benefitOverrides"`
	SalesCount       int64          `gorm:"column:sales_count;type:bigint;not null;default:0;comment:销量" json:"salesCount"`
	Sort             int            `gorm:"column:sort;type:int;not null;default:0;comment:排序值" json:"sort"`
	Status           int8           `gorm:"column:status;type:tinyint;not null;default:0;index:idx_status;comment:上下架状态" json:"status"`
	Deleted          int64          `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)" json:"deleted"`
}

func (SysPackage) TableName() string {
	return "sys_package"
}
