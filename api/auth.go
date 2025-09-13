package api

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type AuthApi struct{}

func (a *AuthApi) Captcha(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"code": 0, "message": "ok"})
}

func (a *AuthApi) Login(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"code": 0, "message": "ok"})
}
