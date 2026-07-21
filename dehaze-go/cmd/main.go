package main

import (
	"github.com/earthyzinc/dehaze-go/internal/app"
)

// @title Dehaze System API
// @version 1.0
// @description 去雾系统后端 API 接口文档
// @termsOfService http://swagger.io/terms/

// @contact.name API Support
// @contact.email support@dehaze.com

// @license.name MIT
// @license.url https://opensource.org/licenses/MIT

// @host localhost:8990
// @BasePath /api/v1
// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
// @description Type "Bearer" followed by a space and JWT token.

//go:generate go env -w GO111MODULE=on
//go:generate go env -w GOPROXY=https://goproxy.cn,direct
//go:generate go mod tidy
//go:generate go mod download
//go:generate mockery --config ../config/.mockery.yaml
func main() {
	if err := app.Run(); err != nil {
		panic(err)
	}
}
