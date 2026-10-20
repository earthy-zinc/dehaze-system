package model

// SysRecommendation 推荐记录表
type SysRecommendation struct {
	BaseModel
	UserID         int64   `gorm:"column:user_id;type:bigint;not null;index:idx_user_id;comment:用户ID" json:"userId"`
	ImageMd5       string  `gorm:"column:image_md5;type:char(32);not null;index:idx_image_md5;comment:图像MD5" json:"imageMd5"`
	TargetType     string  `gorm:"column:target_type;type:varchar(32);not null;default:model;comment:推荐对象类型(model:图像模型;preset:参数方案;dataset:数据集;enhance:增强策略)" json:"targetType"`
	TopModels      string  `gorm:"column:top_models;type:json;not null;comment:推荐图像模型列表(JSON数组，Top 3：模型 slug 及匹配度)" json:"topModels"`
	AnalysisResult *string `gorm:"column:analysis_result;type:json;comment:图像特征分析结果(JSON)" json:"analysisResult"`
	Feedback       int8    `gorm:"column:feedback;type:tinyint;not null;default:0;comment:推荐反馈(0:未反馈;1:有用;2:无用)" json:"feedback"`
	AdoptedModelID *string `gorm:"column:adopted_model_id;type:varchar(64);comment:用户采纳的图像模型标识（null:未采纳）" json:"adoptedModelId"`
}

func (SysRecommendation) TableName() string {
	return "sys_recommendation"
}
