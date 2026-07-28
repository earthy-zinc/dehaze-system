package vo

type MyRatingVO struct {
	ID            int64    `json:"id"`
	PredLogID     int64    `json:"predLogId"`
	AlgorithmName string   `json:"algorithmName"`
	Rating        int      `json:"rating"`
	Comment       string   `json:"comment,omitempty"`
	Tags          []string `json:"tags,omitempty"`
	ImageUrls     []string `json:"imageUrls,omitempty"`
	IsAnonymous   int      `json:"isAnonymous"`
	AdminReply    string   `json:"adminReply,omitempty"`
	ReplyTime     string   `json:"replyTime,omitempty"`
	CreateTime    string   `json:"createTime"`
}

type RatingPageVO struct {
	MyRatingVO
	UserID     int64  `json:"userId"`
	Username   string `json:"username,omitempty"`
	UserAvatar string `json:"userAvatar,omitempty"`
	IsHidden   int    `json:"isHidden"`
}

type RatingDetailVO struct {
	RatingPageVO
	AlgorithmID int64 `json:"algorithmId"`
}

type FeedbackPageVO struct {
	ID            int64    `json:"id"`
	UserID        int64    `json:"userId"`
	Username      string   `json:"username"`
	FeedbackType  string   `json:"feedbackType"`
	Title         string   `json:"title"`
	Content       string   `json:"content"`
	Status        string   `json:"status"`
	Priority      int      `json:"priority"`
	AssigneeID    *int64   `json:"assigneeId,omitempty"`
	AssigneeName  string   `json:"assigneeName,omitempty"`
	RelatedModule string   `json:"relatedModule,omitempty"`
	Tags          []string `json:"tags,omitempty"`
	CreateTime    string   `json:"createTime"`
	UpdateTime    string   `json:"updateTime,omitempty"`
}

type FeedbackDetailVO struct {
	FeedbackPageVO
	Contact      string             `json:"contact,omitempty"`
	Images       []string           `json:"images,omitempty"`
	AssignedTime string             `json:"assignedTime,omitempty"`
	CloseReason  string             `json:"closeReason,omitempty"`
	Replies      []FeedbackReplyVO  `json:"replies"`
}

type FeedbackReplyVO struct {
	ID          int64    `json:"id"`
	FeedbackID  int64    `json:"feedbackId"`
	ReplierID   int64    `json:"replierId"`
	ReplierName string   `json:"replierName"`
	ReplierType int      `json:"replierType"`
	Content     string   `json:"content"`
	ReplyType   string   `json:"replyType,omitempty"`
	Attachments []string `json:"attachments,omitempty"`
	CreateTime  string   `json:"createTime"`
}

type RatingStatsVO struct {
	TotalRatings        int64                  `json:"totalRatings"`
	AverageRating       float64                `json:"averageRating"`
	RatingDistribution  map[int]int64          `json:"ratingDistribution"`
	PositiveTagRanking  []TagCount             `json:"positiveTagRanking"`
	NegativeTagRanking  []TagCount             `json:"negativeTagRanking"`
	AlgorithmStats      []AlgorithmRatingStat `json:"algorithmStats"`
}

type FeedbackStatsVO struct {
	TotalFeedback      int64                `json:"totalFeedback"`
	TypeDistribution   map[string]int64     `json:"typeDistribution"`
	ModuleDistribution []ModuleCount        `json:"moduleDistribution"`
	StatusDistribution map[string]int64     `json:"statusDistribution"`
	AverageResponseTime float64             `json:"averageResponseTime"`
	AverageCloseTime   float64             `json:"averageCloseTime"`
	TopKeywords        []KeywordCount        `json:"topKeywords"`
}

type TagCount struct {
	Tag   string `json:"tag"`
	Count int64  `json:"count"`
}

type AlgorithmRatingStat struct {
	AlgorithmID   int64   `json:"algorithmId"`
	AlgorithmName string  `json:"algorithmName"`
	AverageRating float64 `json:"averageRating"`
	TotalRatings  int64   `json:"totalRatings"`
	LowRatingRate float64 `json:"lowRatingRate"`
}

type ModuleCount struct {
	Module string `json:"module"`
	Count  int64  `json:"count"`
}

type KeywordCount struct {
	Keyword string `json:"keyword"`
	Count   int64  `json:"count"`
}
