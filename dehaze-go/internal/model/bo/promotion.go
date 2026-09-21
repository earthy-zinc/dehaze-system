package bo

type PromotionForm struct {
	ID            int64                  `json:"id"`
	Name          string                 `json:"name" binding:"required"`
	Type          string                 `json:"type" binding:"required,oneof=discount new_user holiday full_reduction"`
	Description   string                 `json:"description"`
	StartTime     string                 `json:"startTime" binding:"required"`
	EndTime       string                 `json:"endTime" binding:"required"`
	ActivityRules map[string]interface{} `json:"activityRules"`
	NewUserOnly   *int                   `json:"newUserOnly" binding:"omitempty,min=0,max=1"`
	Status        *int                   `json:"status" binding:"omitempty,min=0,max=1"`
}

type PromotionPackageForm struct {
	PackageIDs []int64 `json:"packageIds" binding:"required,min=1"`
}
