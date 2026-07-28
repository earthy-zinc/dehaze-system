package query

type RatingPageQuery struct {
	PageNum     int      `json:"pageNum"`
	PageSize    int      `json:"pageSize"`
	Keywords    string   `json:"keywords"`
	AlgorithmID *int64   `json:"algorithmId"`
	RatingMin   *int     `json:"ratingMin"`
	RatingMax   *int     `json:"ratingMax"`
	HasComment  *bool    `json:"hasComment"`
	Tags        []string `json:"tags"`
	StartTime   string   `json:"startTime"`
	EndTime     string   `json:"endTime"`
}

type FeedbackPageQuery struct {
	PageNum        int    `json:"pageNum"`
	PageSize       int    `json:"pageSize"`
	Keywords       string `json:"keywords"`
	FeedbackType   string `json:"feedbackType"`
	Status         string `json:"status"`
	RelatedModule  string `json:"relatedModule"`
	Priority       *int   `json:"priority"`
	AssigneeID     *int64 `json:"assigneeId"`
	StartTime      string `json:"startTime"`
	EndTime        string `json:"endTime"`
}
