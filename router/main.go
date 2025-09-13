package router

import "github.com/earthyzinc/dehaze-go/api"

type RouterGroup struct {
	AuthRouter
}

var RouterGroupApp = new(RouterGroup)

var (
	authApi = api.ApiGroupApp.AuthApi
)
