package model

// SysRecommendationRule 推荐规则配置表
type SysRecommendationRule struct {
	BaseModel
	RuleName     string `gorm:"column:rule_name;type:varchar(64);not null;comment:规则名称" json:"ruleName"`
	SceneType    string `gorm:"column:scene_type;type:varchar(32);not null;index:idx_scene_type;comment:场景类型" json:"sceneType"`
	AlgorithmIds string `gorm:"column:algorithm_ids;type:json;not null;comment:候选算法ID列表(JSON数组)" json:"algorithmIds"`
	Weight       int    `gorm:"column:weight;type:int;not null;default:0;comment:规则权重" json:"weight"`
	Enabled      int8   `gorm:"column:enabled;type:tinyint;not null;default:1;index:idx_enabled;comment:是否启用" json:"enabled"`
	Deleted      int8   `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysRecommendationRule) TableName() string {
	return "sys_recommendation_rule"
}
