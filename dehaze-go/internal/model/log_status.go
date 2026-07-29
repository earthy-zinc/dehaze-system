package model

type LogStatus int8

const (
	LogStatusProcessing LogStatus = 1
	LogStatusCompleted  LogStatus = 2
	LogStatusFailed     LogStatus = 3
)

func (s LogStatus) String() string {
	switch s {
	case LogStatusProcessing:
		return "processing"
	case LogStatusCompleted:
		return "completed"
	case LogStatusFailed:
		return "failed"
	default:
		return "unknown"
	}
}
