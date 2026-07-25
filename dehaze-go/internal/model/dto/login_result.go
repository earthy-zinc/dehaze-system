package dto

type LoginResult struct {
	SessionID string     `json:"sessionId"`
	User      *LoginUser `json:"user"`
}

type LoginUser struct {
	ID       int64  `json:"id"`
	Username string `json:"username"`
	Nickname string `json:"nickname"`
}
