package model

type LogStatus int8

const (
	LogStatusProcessing LogStatus = 1
	LogStatusCompleted  LogStatus = 2
	LogStatusFailed     LogStatus = 3
)
