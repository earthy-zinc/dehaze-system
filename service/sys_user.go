package service

import (
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
)

type UserService struct{}

func (userService *UserService) Login(u *model.SysUser) (userAuthInfo *model.UserAuthInfo, err error) {
	// 先通过用户名查找用户
	err = global.DB.Where("username = ? AND deleted = 0", u.Username).First(u).Error
	if err != nil {
		return nil, err
	}

	// 验证密码
	err = bcrypt.CompareHashAndPassword([]byte(u.Password), []byte(u.Password))
	if err != nil {
		return nil, err
	}

	// 获取用户认证信息
	return userService.GetUserAuthInfo(u.Username)
}

// GetUserAuthInfo 根据用户名获取认证信息
func (userService *UserService) GetUserAuthInfo(username string) (userAuthInfo *model.UserAuthInfo, err error) {
	userAuthInfo = &model.UserAuthInfo{}

	user := model.SysUser{}
	// 查询用户认证信息
	err = global.DB.Table("sys_user").
		Select("id as user_id, username, nickname, password, status, dept_id").
		Where("username = ? AND deleted = 0", username).
		Scan(&user).Error

	userAuthInfo.UserId = user.ID
	userAuthInfo.Username = user.Username
	userAuthInfo.Nickname = user.Nickname
	userAuthInfo.Password = user.Password
	userAuthInfo.Status = int(user.Status)
	userAuthInfo.DeptId = int64(user.DeptID)

	if err != nil && err != gorm.ErrRecordNotFound {
		return nil, err
	}

	// 如果查询到用户信息
	if userAuthInfo.UserId != 0 {
		// 查询用户角色列表
		var roles []string
		err = global.DB.Table("sys_user t1").
			Select("t3.code").
			Joins("LEFT JOIN sys_user_role t2 ON t2.user_id = t1.id").
			Joins("LEFT JOIN sys_role t3 ON t3.id = t2.role_id").
			Where("t1.username = ? AND t1.deleted = 0 AND t3.code IS NOT NULL", username).
			Pluck("t3.code", &roles).Error

		if err != nil {
			return nil, err
		}

		userAuthInfo.Roles = roles

		// 查询角色权限列表
		if len(roles) > 0 {
			var perms []string
			err = global.DB.Table("sys_menu t1").
				Select("DISTINCT t1.perm").
				Joins("INNER JOIN sys_role_menu t2 ON t1.id = t2.menu_id").
				Joins("INNER JOIN sys_role t3 ON t3.id = t2.role_id").
				Where("t1.type = ? AND t1.perm IS NOT NULL", 4). // 4表示按钮类型
				Where("t3.code IN ?", roles).
				Pluck("t1.perm", &perms).Error

			if err != nil {
				return nil, err
			}

			userAuthInfo.Perms = perms
		}

		// 获取最大范围的数据权限
		if len(roles) > 0 {
			var dataScope *int8
			err = global.DB.Table("sys_role").
				Select("MIN(data_scope)").
				Where("code IN ?", roles).
				Scan(&dataScope).Error

			if err != nil {
				return nil, err
			}

			if dataScope != nil {
				userAuthInfo.DataScope = int(*dataScope)
			}
		}
	}

	return userAuthInfo, nil
}
