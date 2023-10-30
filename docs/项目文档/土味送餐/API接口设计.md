---
order: 5
---

# API接口设计

API接口是本项目前后端进行交流的工具，本项目中针对不同的用例采用统一的规范设计接口，并对其做出实现，以下是接口的公共参数表。

表公共参数表


| **参数名称** | **参数类型** | **是否必须** | **描述** | **示例值** |
| ------------ | ------------ | ------------ | -------- | ---------- |
| code         | int          | 是           | 响应代码 | 1          |
| msg          | string       | 是           | 响应信息 | 成功       |
| data         | object       | 否           | 消息内容 | {}         |

以下表格是本项目接口表。


| ***\*序号\**** | ***\*所属用例\**** | ***\*子流程\**** | ***\*方法名称\****    | ***\*URI\****                        | ***\*请求方法\**** | ***\*详情\****         |
| -------------- | ------------------ | ---------------- | --------------------- | ------------------------------------ | ------------------ | ---------------------- |
| 1              | 创建订单           |                  | editOrderDetail       | /order                               | put                | 修改订单状态           |
| 2              |                    |                  | addOrderApi           | /order/submit                        | post               | 提交订单               |
| 3              | 查看订单           |                  | orderListApi          | /order/list                          | get                | 查询订单               |
| 4              |                    |                  | orderPagingApi        | /order/userPage                      | get                | 查询订单（分页）       |
| 5              |                    |                  | getOrderDetailPage    | /order/page                          | get                | 查看订单详情（分页）   |
| 6              |                    |                  | queryOrderDetailById  | /orderDetail/${id}                   | get                | 查看订单详情           |
| 7              | 登录               |                  | login                 | /user/login                          | post               | 用户登录               |
| 8              | 退出               |                  |                       | logout                               | /user/loginout     | post                   |
| 9              | 管理收货地址       | 获取地址         | getAddressBook        | /addressBook/list                    | get                | 获取所有收货地址       |
| 10             |                    |                  | getAddressBookById    | /addressBook/${id}                   | get                | 根据id获取收货地址     |
| 11             | 增加地址           |                  | addAddressBook        | /addressBook                         | post               | 增加收货地址           |
| 12             | 修改地址           |                  | alterAddressBook      | /addressBook                         | put                | 修改收货地址           |
| 13             | 删除地址           |                  | deleteAddressBook     | /addressBook                         | delete             | 删除收货地址           |
| 14             | 设置默认地址       |                  | setDefaultAddressBook | /addressBook/default                 | put                | 设置默认收货地址       |
| 15             |                    |                  | getDefaultAddressBook | /addressBook/default                 | get                | 获取默认收货地址       |
| 16             | 管理菜品分类       | 获取菜品分类     | pageCategory          | /category/page                       | get                | 分页查询菜品分类       |
| 17             |                    |                  | getCategory           | /category/list                       | get                | 获取菜品分类           |
| 18             |                    |                  | getCategoryById       | /category/${id}                      | get                | 根据id获取菜品分类     |
| 19             | 删除菜品分类       |                  | deleteCategory        | /category                            | delete             | 删除菜品分类           |
| 20             | 修改菜品分类       |                  | alterCategory         | /category                            | put                | 修改菜品分类           |
| 21             | 增加菜品分类       |                  | addCategory           | /category                            | post               | 增加菜品分类           |
| 22             | 管理套餐           | 获取套餐         | pageSetmeal           | /setmeal/page                        | get                | 分页查询套餐           |
| 23             |                    |                  | getSetmealById        | /setmeal/${id}                       | get                | 根据id查询套餐         |
| 24             |                    |                  | getSetmeal            | /setmeal/list                        | get                | 获取菜品分类对应的套餐 |
| 25             | 删除套餐           |                  | deleteSetmeal         | /setmeal                             | delete             | 删除套餐               |
| 26             | 修改套餐           |                  | alterSetmeal          | /setmeal                             | put                | 修改套餐               |
| 27             | 增加套餐           |                  | addSetmeal            | /setmeal                             | post               | 增加套餐               |
| 28             | 起售/停售          |                  | setSetmealStatus      | /setmeal/status/${params.**status**} | post               | 批量起售/停售套餐      |
| 29             | 管理菜品           | 获取菜品         | getDishBySetmealID    | /setmeal/dish/${id}                  | get                | 获取套餐的全部菜品     |
| 30             |                    |                  | pageDish              | /dish/page                           | get                | 查询菜品               |
| 31             |                    |                  | getDishById           | /dish/${id}                          | get                | 按id查找菜品           |
| 32             |                    |                  | getDish               | /dish/list                           | get                | 查询全部菜品           |
| 33             | 删除菜品           |                  | deleteDish            | /dish                                | delete             | 删除菜品               |
| 34             | 修改菜品           |                  | alterDish             | /dish                                | put                | 修改菜品               |
| 35             | 新增菜品           |                  | addDish               | /dish                                | post               | 新增菜品               |
| 36             | 起售/停售          |                  | setDishStatus         | /dish/status/${params.**status**}    | post               | 批量起售/停售菜品      |
