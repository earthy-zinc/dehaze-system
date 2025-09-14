package api

import "github.com/earthyzinc/dehaze-go/service"

type ApiGroup struct {
	AuthApi
}

var ApiGroupApp = new(ApiGroup)

var (
	userService = service.ServiceGroupApp.UserService
)
