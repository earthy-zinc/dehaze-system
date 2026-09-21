package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	aidomain "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	dehazevalidator "github.com/earthyzinc/dehaze-go/pkg/validator"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestWriteFormNumericBounds 请求体表单的字段约束与 python Field(...) 同口径：
// 越界/超长/白名单外必须在绑定阶段被拒（A0400），不得直达服务层写库（脏数据）。
//
// 放行侧用"是否 panic"判定：探针 handler 在 bind 返回 true（即越过校验）后主动 panic。
// 注意 python 对**无界**的同类字段（如会话 `pinned`/`status`）刻意不加约束，此处同样不加。
func TestWriteFormNumericBounds(t *testing.T) {
	gin.SetMode(gin.TestMode)
	// 与 app 启动一致：装配 json tag 字段名与中文翻译器（缺此步翻译器为空指针）
	dehazevalidator.Init()

	probe := func(bind func(*gin.Context) bool) *gin.Engine {
		engine := gin.New()
		engine.Use(middleware.ContextErrorHandler())
		engine.POST("/probe", func(c *gin.Context) {
			if !bind(c) {
				return
			}
			panic("越过校验进入服务层")
		})
		return engine
	}

	assertBounds := func(t *testing.T, engine *gin.Engine, denied, accepted []string) {
		t.Helper()
		post := func(body string) *httptest.ResponseRecorder {
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/probe", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			engine.ServeHTTP(rec, req)
			return rec
		}
		for _, body := range denied {
			t.Run("deny "+body, func(t *testing.T) {
				assert.Contains(t, post(body).Body.String(), common.PARAM_ERROR.Code)
			})
		}
		for _, body := range accepted {
			t.Run("accept "+body, func(t *testing.T) {
				require.Panics(t, func() { post(body) }, "参数应通过校验进入服务层")
			})
		}
	}

	bindAgentCreate := func(c *gin.Context) bool {
		var form aidomain.AgentCreateForm
		if err := c.ShouldBindJSON(&form); err != nil {
			_ = c.Error(err)
			return false
		}
		return true
	}
	bindAgentUpdate := func(c *gin.Context) bool {
		var form aidomain.AgentUpdateForm
		if err := c.ShouldBindJSON(&form); err != nil {
			_ = c.Error(err)
			return false
		}
		return true
	}
	bindEndpointCreate := func(c *gin.Context) bool {
		var form aidomain.EndpointCreateForm
		if err := c.ShouldBindJSON(&form); err != nil {
			_ = c.Error(err)
			return false
		}
		return true
	}
	bindEndpointUpdate := func(c *gin.Context) bool {
		var form aidomain.EndpointUpdateForm
		if err := c.ShouldBindJSON(&form); err != nil {
			_ = c.Error(err)
			return false
		}
		return true
	}
	bindScheduleUpdate := func(c *gin.Context) bool {
		var form aidomain.ScheduleUpdateForm
		if err := c.ShouldBindJSON(&form); err != nil {
			_ = c.Error(err)
			return false
		}
		return true
	}
	bindConversationUpdate := func(c *gin.Context) bool {
		var form aidomain.ConversationUpdateForm
		if err := c.ShouldBindJSON(&form); err != nil {
			_ = c.Error(err)
			return false
		}
		return true
	}
	bindFeedback := func(c *gin.Context) bool {
		var form aidomain.FeedbackCreateForm
		if err := c.ShouldBindJSON(&form); err != nil {
			_ = c.Error(err)
			return false
		}
		return true
	}

	t.Run("创建 Agent：status/sortOrder 边界 + 必填与范式白名单", func(t *testing.T) {
		base := `"agentCode":"a1","name":"n1","modelId":"m1"`
		assertBounds(t, probe(bindAgentCreate),
			[]string{
				`{` + base + `,"status":9}`,
				`{` + base + `,"status":-1}`,
				`{` + base + `,"sortOrder":-1}`,
				`{` + base + `,"reasoningMode":"unknown"}`,
				`{"name":"n1","modelId":"m1"}`,
				`{"agentCode":"a1","modelId":"m1"}`,
				`{"agentCode":"a1","name":"n1"}`,
				`{"agentCode":"a1","name":"` + strings.Repeat("x", 129) + `","modelId":"m1"}`,
				`{"agentCode":"a1","name":"n1","modelId":"` + strings.Repeat("m", 65) + `"}`,
			},
			[]string{
				`{` + base + `}`,
				`{` + base + `,"status":1}`,
				`{` + base + `,"status":0}`,
				`{` + base + `,"sortOrder":0}`,
				`{` + base + `,"reasoningMode":"plan_execute"}`,
				`{` + base + `,"description":"` + strings.Repeat("d", 512) + `"}`,
			},
		)
	})

	t.Run("更新 Agent：sortOrder 与字符串约束", func(t *testing.T) {
		assertBounds(t, probe(bindAgentUpdate),
			[]string{
				`{"sortOrder":-1}`,
				`{"name":""}`,
				`{"reasoningMode":"unknown"}`,
				`{"modelId":"` + strings.Repeat("m", 65) + `"}`,
			},
			[]string{
				`{"sortOrder":3}`,
				`{"name":"n1"}`,
				`{"reasoningMode":"react"}`,
				`{}`,
			},
		)
	})

	t.Run("创建 A2A 端点 status [0,1]", func(t *testing.T) {
		assertBounds(t, probe(bindEndpointCreate),
			[]string{`{"status":9}`},
			[]string{`{"status":0}`, `{"status":1}`, `{}`},
		)
	})

	t.Run("更新 A2A 端点 status [0,1]", func(t *testing.T) {
		assertBounds(t, probe(bindEndpointUpdate),
			[]string{`{"status":9}`, `{"status":-1}`},
			[]string{`{"status":1}`, `{}`},
		)
	})

	t.Run("更新定时任务 enabled [0,1]", func(t *testing.T) {
		assertBounds(t, probe(bindScheduleUpdate),
			[]string{`{"enabled":9}`, `{"enabled":-1}`},
			[]string{`{"enabled":1}`, `{"enabled":0}`, `{}`},
		)
	})

	// python 会话表单：title≤255 / model≤64 / agentCode≤64；pinned/status 为无界裸 int（刻意不加）
	t.Run("更新会话：长度约束 + pinned/status 刻意无界", func(t *testing.T) {
		assertBounds(t, probe(bindConversationUpdate),
			[]string{
				`{"title":"` + strings.Repeat("t", 256) + `"}`,
				`{"model":"` + strings.Repeat("m", 65) + `"}`,
				`{"agentCode":"` + strings.Repeat("a", 65) + `"}`,
			},
			[]string{
				`{"title":"` + strings.Repeat("t", 255) + `"}`,
				`{"pinned":2}`,  // python 无 ge/le，不得收紧
				`{"status":99}`, // 同上
				`{}`,
			},
		)
	})

	// 消息反馈 rating 在 python 是 Literal[1,-1]（白名单，非区间）；comment max_length=2000
	t.Run("消息反馈 rating 白名单 / comment 长度", func(t *testing.T) {
		assertBounds(t, probe(bindFeedback),
			[]string{
				`{}`,
				`{"rating":0}`,
				`{"rating":5}`,
				`{"rating":1,"comment":"` + strings.Repeat("c", 2001) + `"}`,
			},
			[]string{
				`{"rating":1}`,
				`{"rating":-1}`,
				`{"rating":1,"comment":"ok"}`,
			},
		)
	})
}
