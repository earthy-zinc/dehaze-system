package vo

// MessageTemplateVO 模板列表项
type MessageTemplateVO struct {
	ID            int64  `json:"id"`
	Code          string `json:"code"`
	Name          string `json:"name"`
	Type          string `json:"type"`
	TitleTemplate string `json:"titleTemplate"`
	Priority      int    `json:"priority"`
	Status        int    `json:"status"`
	CreateTime    string `json:"createTime"`
}

// MessageTemplateDetailVO 模板详情
type MessageTemplateDetailVO struct {
	ID              int64            `json:"id"`
	Code            string           `json:"code"`
	Name            string           `json:"name"`
	Type            string           `json:"type"`
	TitleTemplate   string           `json:"titleTemplate"`
	ContentTemplate string           `json:"contentTemplate"`
	Priority        int              `json:"priority"`
	Channels        interface{}      `json:"channels"`
	Variables       []TemplateVarVO  `json:"variables"`
	Status          int              `json:"status"`
	CreateTime      string           `json:"createTime"`
	UpdateTime      string           `json:"updateTime"`
}

// TemplateVarVO 模板变量
type TemplateVarVO struct {
	Name string `json:"name"`
	Desc string `json:"desc"`
}
