package bo

// AlgorithmStatus 算法生命周期状态
const (
	AlgorithmStatusDraft     = 1 // 草稿
	AlgorithmStatusTesting   = 2 // 测试中
	AlgorithmStatusAuditing  = 3 // 待审核
	AlgorithmStatusPublished = 4 // 已发布
	AlgorithmStatusDisabled  = 5 // 已停用
	AlgorithmStatusArchived  = 6 // 已归档
)

// AlgorithmFormBO 算法表单业务对象
type AlgorithmFormBO struct {
	ID          int64  `json:"id"`
	ParentID    int64  `json:"parentId"`
	Type        string `json:"type" binding:"required,max=32"`
	Name        string `json:"name" binding:"required,max=128"`
	Path        string `json:"path" binding:"omitempty,max=255"`
	ImportPath  string `json:"importPath" binding:"omitempty,max=255"`
	Description string `json:"description" binding:"omitempty,max=255"`
	Status      int8   `json:"status" binding:"oneof=1 2 3 4 5 6"`
}

// allowedTransitions 定义允许的状态流转映射
// key: 当前状态, value: 允许跳转的目标状态集合
var allowedTransitions = map[int8]map[int8]bool{
	AlgorithmStatusDraft:     {AlgorithmStatusTesting: true, AlgorithmStatusAuditing: true},
	AlgorithmStatusTesting:   {AlgorithmStatusAuditing: true, AlgorithmStatusDisabled: true},
	AlgorithmStatusAuditing:  {AlgorithmStatusPublished: true, AlgorithmStatusDraft: true}, // 驳回回到草稿
	AlgorithmStatusPublished: {AlgorithmStatusDisabled: true, AlgorithmStatusArchived: true},
	AlgorithmStatusDisabled:  {AlgorithmStatusTesting: true},
	AlgorithmStatusArchived:  {}, // 单向，不可逆
}

// CanTransitionTo 校验状态流转是否合法
func CanTransitionTo(currentStatus, targetStatus int8) bool {
	if target, ok := allowedTransitions[currentStatus]; ok {
		return target[targetStatus]
	}
	return false
}
