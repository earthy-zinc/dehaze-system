---
order: 4
---

# API接口设计

## 认证接口
### POST 登录

POST /api/v1/auth/login

#### 请求参数

| 名称     | 位置  | 类型   | 必选 | 说明   |
| -------- | ----- | ------ | ---- | ------ |
| username | query | string | 是   | 用户名 |
| password | query | string | 是   | 密码   |

> 返回示例

```json
{
  "code": "string",
  "data": {
    "accessToken": "string",
    "tokenType": "Bearer",
    "refreshToken": "string",
    "expires": 0
  },
  "msg": "string"
}
```

### GET 获取验证码

GET /api/v1/auth/captcha

> 返回示例
```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### DELETE 注销

DELETE /api/v1/auth/logout

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

## 用户接口


### PUT 修改用户

PUT /api/v1/users/{userId}

> Body 请求参数
```json
{
  "id": 0,
  "username": "string",
  "nickname": "string",
  "mobile": "string",
  "gender": 0,
  "avatar": "string",
  "email": "string",
  "status": 0,
  "deptId": 0,
  "roleIds": [
    0
  ]
}
```

### 请求参数

| 名称   | 位置 | 类型                        | 必选 | 说明   |
| ------ | ---- | --------------------------- | ---- | ------ |
| userId | path | integer                     | 是   | 用户ID |
| body   | body | [UserForm](#schemauserform) | 否   | none   |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### POST 新增用户

POST /api/v1/users

> Body 请求参数

```json
{
  "id": 0,
  "username": "string",
  "nickname": "string",
  "mobile": "string",
  "gender": 0,
  "avatar": "string",
  "email": "string",
  "status": 0,
  "deptId": 0,
  "roleIds": [
    0
  ]
}
```

#### 请求参数

| 名称 | 位置 | 类型                        | 必选 | 说明 |
| ---- | ---- | --------------------------- | ---- | ---- |
| body | body | [UserForm](#schemauserform) | 否   | none |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```


### POST 导入用户

POST /api/v1/users/_import

#### 请求参数

| 名称   | 位置  | 类型    | 必选 | 说明   |
| ------ | ----- | ------- | ---- | ------ |
| deptId | query | integer | 是   | 部门ID |
| file   | query | string  | 是   | none   |

> 返回示例
```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```


### PATCH 修改用户状态

PATCH /api/v1/users/{userId}/status

#### 请求参数

| 名称   | 位置  | 类型    | 必选 | 说明                    |
| ------ | ----- | ------- | ---- | ----------------------- |
| userId | path  | integer | 是   | 用户ID                  |
| status | query | integer | 是   | 用户状态(1:启用;0:禁用) |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### PATCH 修改用户密码

PATCH /api/v1/users/{userId}/password

#### 请求参数

| 名称     | 位置  | 类型    | 必选 | 说明   |
| -------- | ----- | ------- | ---- | ------ |
| userId   | path  | integer | 是   | 用户ID |
| password | query | string  | 是   | none   |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### GET 用户表单数据

GET /api/v1/users/{userId}/form

#### 请求参数

| 名称   | 位置 | 类型    | 必选 | 说明   |
| ------ | ---- | ------- | ---- | ------ |
| userId | path | integer | 是   | 用户ID |

> 返回示例

```json
{
  "code": "string",
  "data": {
    "id": 0,
    "username": "string",
    "nickname": "string",
    "mobile": "string",
    "gender": 0,
    "avatar": "string",
    "email": "string",
    "status": 0,
    "deptId": 0,
    "roleIds": [
      0
    ]
  },
  "msg": "string"
}
```

### GET 用户导入模板下载

GET /api/v1/users/template


### GET 用户分页列表

GET /api/v1/users/page

#### 请求参数

| 名称     | 位置  | 类型   | 必选 | 说明                       |
| -------- | ----- | ------ | ---- | -------------------------- |
| keywords | query | string | 否   | 关键字(用户名/昵称/手机号) |
| status   | query | string | 否   | 用户状态                   |
| deptId   | query | string | 否   | 部门ID                     |
| pageNum  | query | string | 否   | 页码                       |
| pageSize | query | string | 否   | 每页记录数                 |

> 返回示例
```json
{
  "code": "string",
  "data": {
    "list": [
      {
        "id": 0,
        "username": "string",
        "nickname": "string",
        "mobile": "string",
        "genderLabel": "string",
        "avatar": "string",
        "email": "string",
        "status": 0,
        "deptName": "string",
        "roleNames": "string",
        "createTime": "2019-08-24T14:15:22Z"
      }
    ],
    "total": 0
  },
  "msg": "string"
}
```

### GET 获取当前登录用户信息

GET /api/v1/users/me

> 返回示例

```json
{
  "code": "string",
  "data": {
    "userId": 0,
    "nickname": "string",
    "avatar": "string",
    "roles": [
      "string"
    ],
    "perms": [
      "string"
    ]
  },
  "msg": "string"
}
```

### GET 导出用户

GET /api/v1/users/_export

#### 请求参数

| 名称     | 位置  | 类型    | 必选 | 说明                       |
| -------- | ----- | ------- | ---- | -------------------------- |
| pageNum  | query | integer | 否   | 页码                       |
| pageSize | query | integer | 否   | 每页记录数                 |
| keywords | query | string  | 否   | 关键字(用户名/昵称/手机号) |
| status   | query | integer | 否   | 用户状态                   |
| deptId   | query | integer | 否   | 部门ID                     |


### DELETE 删除用户

DELETE /api/v1/users/{ids}

#### 请求参数

| 名称 | 位置 | 类型   | 必选 | 说明                          |
| ---- | ---- | ------ | ---- | ----------------------------- |
| ids  | path | string | 是   | 用户ID，多个以英文逗号(,)分割 |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

## 角色接口

### PUT 修改角色状态

PUT /api/v1/roles/{roleId}/status

#### 请求参数

| 名称   | 位置  | 类型    | 必选 | 说明                |
| ------ | ----- | ------- | ---- | ------------------- |
| roleId | path  | integer | 是   | 角色ID              |
| status | query | integer | 是   | 状态(1:启用;0:禁用) |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### PUT 分配菜单权限给角色

PUT /api/v1/roles/{roleId}/menus

> Body 请求参数

```json
[
  0
]
```

### 请求参数

| 名称   | 位置 | 类型           | 必选 | 说明 |
| ------ | ---- | -------------- | ---- | ---- |
| roleId | path | integer        | 是   | none |
| body   | body | array[integer] | 否   | none |

> 返回示例
```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### PUT 修改角色

PUT /api/v1/roles/{id}

> Body 请求参数

```json
{
  "id": 0,
  "name": "string",
  "code": "string",
  "sort": 0,
  "status": 0,
  "dataScope": 0
}
```

#### 请求参数

| 名称 | 位置 | 类型                        | 必选 | 说明 |
| ---- | ---- | --------------------------- | ---- | ---- |
| id   | path | string                      | 是   | none |
| body | body | [RoleForm](#schemaroleform) | 否   | none |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```


### POST 新增角色

POST /api/v1/roles

> Body 请求参数

```json
{
  "id": 0,
  "name": "string",
  "code": "string",
  "sort": 0,
  "status": 0,
  "dataScope": 0
}
```

#### 请求参数

| 名称 | 位置 | 类型                        | 必选 | 说明 |
| ---- | ---- | --------------------------- | ---- | ---- |
| body | body | [RoleForm](#schemaroleform) | 否   | none |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### GET 获取角色的菜单ID集合

GET /api/v1/roles/{roleId}/menuIds

#### 请求参数

| 名称   | 位置 | 类型    | 必选 | 说明   |
| ------ | ---- | ------- | ---- | ------ |
| roleId | path | integer | 是   | 角色ID |

> 返回示例

```json
{
  "code": "string",
  "data": [
    0
  ],
  "msg": "string"
}
```

### GET 角色表单数据

GET /api/v1/roles/{roleId}/form

#### 请求参数

| 名称   | 位置 | 类型    | 必选 | 说明   |
| ------ | ---- | ------- | ---- | ------ |
| roleId | path | integer | 是   | 角色ID |

> 返回示例

```json
{
  "code": "string",
  "data": {
    "id": 0,
    "name": "string",
    "code": "string",
    "sort": 0,
    "status": 0,
    "dataScope": 0
  },
  "msg": "string"
}
```

### GET 角色分页列表

GET /api/v1/roles/page

#### 请求参数

| 名称     | 位置  | 类型   | 必选 | 说明                      |
| -------- | ----- | ------ | ---- | ------------------------- |
| keywords | query | string | 否   | 关键字(角色名称/角色编码) |
| pageNum  | query | string | 否   | 页码                      |
| pageSize | query | string | 否   | 每页记录数                |

> 返回示例

```json
{
  "code": "string",
  "data": {
    "list": [
      {
        "id": 0,
        "name": "string",
        "code": "string",
        "status": 0,
        "sort": 0,
        "createTime": "2019-08-24T14:15:22Z",
        "updateTime": "2019-08-24T14:15:22Z"
      }
    ],
    "total": 0
  },
  "msg": "string"
}
```


### GET 角色下拉列表

GET /api/v1/roles/options

> 返回示例


```json
{
  "code": "string",
  "data": [
    {
      "value": {},
      "label": "string",
      "children": [
        {
          "value": {},
          "label": "string"
        }
      ]
    }
  ],
  "msg": "string"
}
```

### DELETE 删除角色

DELETE /api/v1/roles/{ids}

#### 请求参数

| 名称 | 位置 | 类型   | 必选 | 说明                            |
| ---- | ---- | ------ | ---- | ------------------------------- |
| ids  | path | string | 是   | 删除角色，多个以英文逗号(,)分割 |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

## 菜单接口

### PUT 修改菜单

PUT /api/v1/menus/{id}

> Body 请求参数

```json
{
  "id": 0,
  "parentId": 0,
  "name": "string",
  "type": "NULL",
  "path": "string",
  "component": "string",
  "perm": "string",
  "visible": 0,
  "sort": 0,
  "icon": "string",
  "redirect": "string"
}
```

#### 请求参数

| 名称 | 位置 | 类型                        | 必选 | 说明 |
| ---- | ---- | --------------------------- | ---- | ---- |
| id   | path | string                      | 是   | none |
| body | body | [MenuForm](#schemamenuform) | 否   | none |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### DELETE 删除菜单

DELETE /api/v1/menus/{id}

#### 请求参数

| 名称 | 位置 | 类型    | 必选 | 说明                      |
| ---- | ---- | ------- | ---- | ------------------------- |
| id   | path | integer | 是   | 菜单ID，多个以英文(,)分割 |

> 返回示例
```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### GET 菜单列表

GET /api/v1/menus

#### 请求参数

| 名称     | 位置  | 类型   | 必选 | 说明                   |
| -------- | ----- | ------ | ---- | ---------------------- |
| keywords | query | string | 否   | 关键字(菜单名称)       |
| status   | query | string | 否   | 状态(1->显示；0->隐藏) |

> 返回示例

```json
{
  "code": "string",
  "data": [
    {
      "id": 0,
      "parentId": 0,
      "name": "string",
      "type": "NULL",
      "path": "string",
      "component": "string",
      "sort": 0,
      "visible": 0,
      "icon": "string",
      "redirect": "string",
      "perm": "string",
      "children": [
        {
          "id": 0,
          "parentId": 0,
          "name": "string",
          "type": "NULL",
          "path": "string",
          "component": "string",
          "sort": 0,
          "visible": 0,
          "icon": "string",
          "redirect": "string",
          "perm": "string",
          "children": [
            {}
          ]
        }
      ]
    }
  ],
  "msg": "string"
}
```

### POST 新增菜单

POST /api/v1/menus

> Body 请求参数

```json
{
  "id": 0,
  "parentId": 0,
  "name": "string",
  "type": "NULL",
  "path": "string",
  "component": "string",
  "perm": "string",
  "visible": 0,
  "sort": 0,
  "icon": "string",
  "redirect": "string"
}
```

#### 请求参数

| 名称 | 位置 | 类型                        | 必选 | 说明 |
| ---- | ---- | --------------------------- | ---- | ---- |
| body | body | [MenuForm](#schemamenuform) | 否   | none |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### PATCH 修改菜单显示状态

PATCH /api/v1/menus/{menuId}

#### 请求参数

| 名称    | 位置  | 类型    | 必选 | 说明                    |
| ------- | ----- | ------- | ---- | ----------------------- |
| menuId  | path  | integer | 是   | 菜单ID                  |
| visible | query | integer | 是   | 显示状态(1:显示;0:隐藏) |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### GET 菜单表单数据

GET /api/v1/menus/{id}/form

#### 请求参数

| 名称 | 位置 | 类型    | 必选 | 说明   |
| ---- | ---- | ------- | ---- | ------ |
| id   | path | integer | 是   | 菜单ID |

> 返回示例

```json
{
  "code": "string",
  "data": {
    "id": 0,
    "parentId": 0,
    "name": "string",
    "type": "NULL",
    "path": "string",
    "component": "string",
    "perm": "string",
    "visible": 0,
    "sort": 0,
    "icon": "string",
    "redirect": "string"
  },
  "msg": "string"
}
```

### GET 路由列表

GET /api/v1/menus/routes

> 返回示例

```json
{
  "code": "string",
  "data": [
    {
      "path": "user",
      "component": "system/user/index",
      "redirect": "https://www.youlai.tech",
      "name": "string",
      "meta": {
        "title": "string",
        "icon": "string",
        "hidden": true,
        "roles": "['ADMIN','ROOT']",
        "keepAlive": true
      },
      "children": [
        {
          "path": "user",
          "component": "system/user/index",
          "redirect": "https://www.youlai.tech",
          "name": "string",
          "meta": {
            "title": null,
            "icon": null,
            "hidden": null,
            "roles": null,
            "keepAlive": null
          },
          "children": [
            {}
          ]
        }
      ]
    }
  ],
  "msg": "string"
}
```

### GET 菜单下拉列表

GET /api/v1/menus/options

> 返回示例


```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

## 字典接口

### PUT 修改字典

PUT /api/v1/dict/{id}

> Body 请求参数

```json
{
  "id": 0,
  "typeCode": "string",
  "name": "string",
  "value": "string",
  "status": 0,
  "sort": 0,
  "remark": "string"
}
```

#### 请求参数

| 名称 | 位置 | 类型                        | 必选 | 说明 |
| ---- | ---- | --------------------------- | ---- | ---- |
| id   | path | integer                     | 是   | none |
| body | body | [DictForm](#schemadictform) | 否   | none |

> 返回示例


```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### PUT 修改字典类型

PUT /api/v1/dict/types/{id}

> Body 请求参数

```json
{
  "id": 0,
  "name": "string",
  "code": "string",
  "status": 0,
  "remark": "string"
}
```

#### 请求参数

| 名称 | 位置 | 类型                                | 必选 | 说明 |
| ---- | ---- | ----------------------------------- | ---- | ---- |
| id   | path | integer                             | 是   | none |
| body | body | [DictTypeForm](#schemadicttypeform) | 否   | none |

> 返回示例
```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### POST 新增字典

POST /api/v1/dict

> Body 请求参数

```json
{
  "id": 0,
  "typeCode": "string",
  "name": "string",
  "value": "string",
  "status": 0,
  "sort": 0,
  "remark": "string"
}
```

#### 请求参数

| 名称 | 位置 | 类型                        | 必选 | 说明 |
| ---- | ---- | --------------------------- | ---- | ---- |
| body | body | [DictForm](#schemadictform) | 否   | none |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### POST 新增字典类型

POST /api/v1/dict/types

> Body 请求参数

```json
{
  "id": 0,
  "name": "string",
  "code": "string",
  "status": 0,
  "remark": "string"
}
```

#### 请求参数

| 名称 | 位置 | 类型                                | 必选 | 说明 |
| ---- | ---- | ----------------------------------- | ---- | ---- |
| body | body | [DictTypeForm](#schemadicttypeform) | 否   | none |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### GET 字典下拉列表

GET /api/v1/dict/{typeCode}/options

#### 请求参数

| 名称     | 位置 | 类型   | 必选 | 说明         |
| -------- | ---- | ------ | ---- | ------------ |
| typeCode | path | string | 是   | 字典类型编码 |

> 返回示例

```json
{
  "code": "string",
  "data": [
    {
      "value": {},
      "label": "string",
      "children": [
        {
          "value": {},
          "label": "string"
        }
      ]
    }
  ],
  "msg": "string"
}
```

### GET 字典数据表单数据

GET /api/v1/dict/{id}/form

#### 请求参数

| 名称 | 位置 | 类型    | 必选 | 说明   |
| ---- | ---- | ------- | ---- | ------ |
| id   | path | integer | 是   | 字典ID |

> 返回示例

```json
{
  "code": "string",
  "data": {
    "id": 0,
    "typeCode": "string",
    "name": "string",
    "value": "string",
    "status": 0,
    "sort": 0,
    "remark": "string"
  },
  "msg": "string"
}
```

### GET 字典类型表单数据

GET /api/v1/dict/types/{id}/form

#### 请求参数

| 名称 | 位置 | 类型    | 必选 | 说明   |
| ---- | ---- | ------- | ---- | ------ |
| id   | path | integer | 是   | 字典ID |

> 返回示例

```json
{
  "code": "string",
  "data": {
    "id": 0,
    "name": "string",
    "code": "string",
    "status": 0,
    "remark": "string"
  },
  "msg": "string"
}
```


### GET 字典类型分页列表

GET /api/v1/dict/types/page

#### 请求参数

| 名称     | 位置  | 类型   | 必选 | 说明                      |
| -------- | ----- | ------ | ---- | ------------------------- |
| keywords | query | string | 否   | 关键字(类型名称/类型编码) |
| pageNum  | query | string | 否   | 页码                      |
| pageSize | query | string | 否   | 每页记录数                |

> 返回示例
```json
{
  "code": "string",
  "data": {
    "list": [
      {
        "id": 0,
        "name": "string",
        "code": "string",
        "status": 0
      }
    ],
    "total": 0
  },
  "msg": "string"
}
```

### GET 字典分页列表

GET /api/v1/dict/page

#### 请求参数

| 名称     | 位置  | 类型   | 必选 | 说明               |
| -------- | ----- | ------ | ---- | ------------------ |
| keywords | query | string | 否   | 关键字(字典项名称) |
| typeCode | query | string | 否   | 字典类型编码       |
| pageNum  | query | string | 否   | 页码               |
| pageSize | query | string | 否   | 每页记录数         |

> 返回示例

> 200 Response

```json
{
  "code": "string",
  "data": {
    "list": [
      {
        "id": 0,
        "name": "string",
        "value": "string",
        "status": 0
      }
    ],
    "total": 0
  },
  "msg": "string"
}
```

### DELETE 删除字典

DELETE /api/v1/dict/{ids}

#### 请求参数

| 名称 | 位置 | 类型   | 必选 | 说明                          |
| ---- | ---- | ------ | ---- | ----------------------------- |
| ids  | path | string | 是   | 字典ID，多个以英文逗号(,)拼接 |

> 返回示例

> 200 Response

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```


### DELETE 删除字典类型

DELETE /api/v1/dict/types/{ids}

#### 请求参数

| 名称 | 位置 | 类型   | 必选 | 说明                              |
| ---- | ---- | ------ | ---- | --------------------------------- |
| ids  | path | string | 是   | 字典类型ID，多个以英文逗号(,)分割 |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```


## 部门接口

### PUT 修改部门

PUT /api/v1/dept/{deptId}

> Body 请求参数

```json
{
  "id": 0,
  "name": "string",
  "parentId": 0,
  "status": 0,
  "sort": 0
}
```

#### 请求参数

| 名称   | 位置 | 类型                        | 必选 | 说明 |
| ------ | ---- | --------------------------- | ---- | ---- |
| deptId | path | integer                     | 是   | none |
| body   | body | [DeptForm](#schemadeptform) | 否   | none |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

### GET 获取部门列表

GET /api/v1/dept

#### 请求参数

| 名称     | 位置  | 类型   | 必选 | 说明                   |
| -------- | ----- | ------ | ---- | ---------------------- |
| keywords | query | string | 否   | 关键字(部门名称)       |
| status   | query | string | 否   | 状态(1->正常；0->禁用) |

> 返回示例

```json
{
  "code": "string",
  "data": [
    {
      "id": 0,
      "parentId": 0,
      "name": "string",
      "sort": 0,
      "status": 0,
      "children": [
        {
          "id": 0,
          "parentId": 0,
          "name": "string",
          "sort": 0,
          "status": 0,
          "children": [
            {}
          ],
          "createTime": "2019-08-24T14:15:22Z",
          "updateTime": "2019-08-24T14:15:22Z"
        }
      ],
      "createTime": "2019-08-24T14:15:22Z",
      "updateTime": "2019-08-24T14:15:22Z"
    }
  ],
  "msg": "string"
}
```

### POST 新增部门

POST /api/v1/dept

> Body 请求参数

```json
{
  "id": 0,
  "name": "string",
  "parentId": 0,
  "status": 0,
  "sort": 0
}
```

#### 请求参数

| 名称 | 位置 | 类型                        | 必选 | 说明 |
| ---- | ---- | --------------------------- | ---- | ---- |
| body | body | [DeptForm](#schemadeptform) | 否   | none |

> 返回示例

> 200 Response

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```


### GET 获取部门表单数据

GET /api/v1/dept/{deptId}/form

#### 请求参数

| 名称   | 位置 | 类型    | 必选 | 说明   |
| ------ | ---- | ------- | ---- | ------ |
| deptId | path | integer | 是   | 部门ID |

> 返回示例

```json
{
  "code": "string",
  "data": {
    "id": 0,
    "name": "string",
    "parentId": 0,
    "status": 0,
    "sort": 0
  },
  "msg": "string"
}
```

### GET 获取部门下拉选项

GET /api/v1/dept/options

> 返回示例
```json
{
  "code": "string",
  "data": [
    {
      "value": {},
      "label": "string",
      "children": [
        {
          "value": {},
          "label": "string"
        }
      ]
    }
  ],
  "msg": "string"
}
```

### DELETE 删除部门

DELETE /api/v1/dept/{ids}

### 请求参数

| 名称 | 位置 | 类型   | 必选 | 说明                          |
| ---- | ---- | ------ | ---- | ----------------------------- |
| ids  | path | string | 是   | 部门ID，多个以英文逗号(,)分割 |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

## 文件接口
### POST 文件上传

POST /api/v1/files

> Body 请求参数

```json
{
  "file": "string"
}
```

#### 请求参数

| 名称   | 位置 | 类型           | 必选 | 说明         |
| ------ | ---- | -------------- | ---- | ------------ |
| body   | body | object         | 否   | none         |
| » file | body | string(binary) | 是   | 表单文件对象 |

> 返回示例

```json
{
  "code": "string",
  "data": {
    "name": "string",
    "url": "string"
  },
  "msg": "string"
}
```

### DELETE 文件删除

DELETE /api/v1/files

#### 请求参数

| 名称     | 位置  | 类型   | 必选 | 说明     |
| -------- | ----- | ------ | ---- | -------- |
| filePath | query | string | 是   | 文件路径 |

> 返回示例

```json
{
  "code": "string",
  "data": {},
  "msg": "string"
}
```

## 去雾接口

### POST 上传图片

POST /upload/

#### 请求参数

| 名称 | 位置 | 类型           | 必选 | 说明 |
| ---- | ---- | -------------- | ---- | ---- |
| body | body | string(binary) | 否   | none |

> 返回示例

```json
{
  "code": "string",
  "msg": "string",
  "data": {
    "image_name": "string"
  }
}
```

### GET 下载图片

GET /download/{image_name}/

#### 请求参数

| 名称       | 位置 | 类型   | 必选 | 说明 |
| ---------- | ---- | ------ | ---- | ---- |
| image_name | path | string | 是   | none |


### POST 图片去雾

POST /dehazeImage/

> Body 请求参数

```json
{
  "haze_image": "string",
  "model_name": "C2PNet/OTS.pkl"
}
```

#### 请求参数

| 名称         | 位置 | 类型   | 必选 | 说明             |
| ------------ | ---- | ------ | ---- | ---------------- |
| body         | body | object | 否   | none             |
| » haze_image | body | string | 是   | 有雾图片的文件名 |
| » model_name | body | string | 是   | none             |

##### 枚举值

| 属性         | 值             |
| ------------ | -------------- |
| » model_name | C2PNet/OTS.pkl |

> 返回示例

```json
{
  "code": "string",
  "msg": "string",
  "data": {
    "image_name": "string"
  }
}
```

### POST 评价指标

POST /calculateIndex/

获取图像去雾的评价指标，PSNR和SSIM

> Body 请求参数

```json
{
  "haze_image": "string",
  "clear_image": "string"
}
```

#### 请求参数

| 名称          | 位置 | 类型   | 必选 | 说明 |
| ------------- | ---- | ------ | ---- | ---- |
| body          | body | object | 否   | none |
| » haze_image  | body | string | 是   | none |
| » clear_image | body | string | 是   | none |

> 返回示例

```json
{
  "code": "string",
  "msg": "string",
  "data": {
    "psnr": 0,
    "ssim": 0
  }
}
```

### GET 获取去雾模型列表

GET /model/

> 返回示例
```json
{
  "code": "string",
  "msg": "string",
  "data": [
    {
      "value": "string",
      "label": "string",
      "children": [
        {
          "value": "string",
          "label": "string",
          "children": [
            {}
          ]
        }
      ]
    }
  ]
}
```

