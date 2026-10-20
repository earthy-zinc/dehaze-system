package model

type LogStatus int8

const (
	LogStatusProcessing LogStatus = 1
	LogStatusCompleted  LogStatus = 2
	LogStatusFailed     LogStatus = 3
	// LogStatusCancelled 已取消（用户主动取消；仅"处理中"可流转到此态，python LogStatus.CANCELLED 同值）
	LogStatusCancelled LogStatus = 4
)

func (s LogStatus) String() string {
	switch s {
	case LogStatusProcessing:
		return "processing"
	case LogStatusCompleted:
		return "completed"
	case LogStatusFailed:
		return "failed"
	case LogStatusCancelled:
		return "cancelled"
	default:
		return "unknown"
	}
}
