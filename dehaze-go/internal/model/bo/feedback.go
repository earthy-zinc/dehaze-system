package bo

type RatingCreateForm struct {
	PredLogID   int64    `json:"predLogId"`
	Rating      int      `json:"rating"`
	Comment     string   `json:"comment"`
	Tags        []string `json:"tags"`
	ImageUrls   []string `json:"imageUrls"`
	IsAnonymous int      `json:"isAnonymous"`
}

type FeedbackCreateForm struct {
	FeedbackType  string   `json:"feedbackType"`
	Title         string   `json:"title"`
	Content       string   `json:"content"`
	Contact       string   `json:"contact"`
	Images        []string `json:"images"`
	RelatedModule string   `json:"relatedModule"`
}

type FeedbackSupplementForm struct {
	Content     string   `json:"content"`
	Attachments []string `json:"attachments"`
}

type FeedbackReplyForm struct {
	Content     string   `json:"content"`
	ReplyType   string   `json:"replyType"`
	Attachments []string `json:"attachments"`
}

type FeedbackAssignForm struct {
	AssigneeID int64 `json:"assigneeId"`
}

type FeedbackCloseForm struct {
	CloseReason string `json:"closeReason"`
}
