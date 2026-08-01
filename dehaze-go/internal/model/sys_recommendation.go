package model

// SysRecommendation 推荐记录表
type SysRecommendation struct {
	BaseModel
	UserID            int64  `gorm:"column:user_id;type:bigint;not null;index:idx_user_id;comment:用户ID" json:"userId"`
	ImageMd5          string `gorm:"column:image_md5;type:char(32);not null;index:idx_image_md5;comment:图像MD5" json:"imageMd5"`
	TargetType        string `gorm:"column:target_type;type:varchar(32);not null;default:algorithm;comment:推荐对象类型" json:"targetType"`
	TopAlgorithms     string `gorm:"column:top_algorithms;type:json;not null;comment:推荐算法列表(JSON)" json:"topAlgorithms"`
	AnalysisResult    *string `gorm:"column:analysis_result;type:json;comment:图像特征分析结果(JSON)" json:"analysisResult"`
	Feedback          int8   `gorm:"column:feedback;type:tinyint;not null;default:0;comment:推荐反馈(0:未反馈;1:有用;2:无用)" json:"feedback"`
	AdoptedAlgorithmID *int64 `gorm:"column:adopted_algorithm_id;type:bigint;comment:用户采纳的算法ID" json:"adoptedAlgorithmId"`
}

func (SysRecommendation) TableName() string {
	return "sys_recommendation"
}
