package api

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// TestLastPathID 同一路径含多个 :id 时（如 /ai/models/:id/prices/:id），
// gin 的 Param("id") 只返回首个匹配，末位参数必须走 c.Params。
func TestLastPathID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.GET("/a/:id/b/:id", func(c *gin.Context) {
		first, err := parseAiPathID(c, "id")
		if err != nil {
			c.String(http.StatusInternalServerError, err.Error())
			return
		}
		last, err := lastAiPathID(c)
		if err != nil {
			c.String(http.StatusInternalServerError, err.Error())
			return
		}
		c.String(http.StatusOK, "%d|%d", first, last)
	})

	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/a/7/b/42", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "7|42", rec.Body.String())
}

func TestParsePathIDRejectsNonNumeric(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.GET("/a/:id", func(c *gin.Context) {
		if _, err := parseAiPathID(c, "id"); err != nil {
			c.String(http.StatusBadRequest, "bad")
			return
		}
		c.String(http.StatusOK, "ok")
	})

	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/a/not-a-number", nil))
	require.Equal(t, http.StatusBadRequest, rec.Code)
}
