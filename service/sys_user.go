package service

import (
	"github.com/earthyzinc/dehaze-go/model"
)

type UserService struct{}

func (userService *UserService) Login(u *model.SysUser) (userInter *model.SysUser, err error) {

	return &model.SysUser{
		Username: "admin",
	}, nil
}
