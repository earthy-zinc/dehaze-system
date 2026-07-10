# 字典管理模块跨后端一致性差异分析报告

> 生成时间：2026-07-11
> 基准：`dehaze-doc/docs/03-模块设计/基础模块/字典管理/API接口.md` + `后端实现.md` + `需求规格.md`
> 三端代码：Java(8989) / Go(8999) / Python(8014)

## 一、文档基准关键规范

| 维度 | 规范 |
|------|------|
| 权限码-字典类型 | `sys:dict:type:add` / `sys:dict:type:edit` / `sys:dict:type:delete` |
| 权限码-字典数据 | `sys:dict:data:add` / `sys:dict:data:edit` / `sys:dict:data:delete` |
| 查询接口权限 | page/form/options 仅需登录态，无权限码 |
| 缓存 | key=`dict:options:{typeCode}`，TTL=1h，增删改时主动失效 |
| 排序 | 字典数据：`sort ASC, create_time DESC` |
| typeCode编辑 | 字典数据的 typeCode 编辑时只读不可修改（需求规格 3.7.2） |
| 编码变更级联 | 字典类型 code 变更时，同步更新 sys_dict.type_code（需求规格 3.3.3） |
| 删除约束 | 字典类型有关联字典数据时禁止删除，错误码 A0504 |
| 唯一性-类型 | sys_dict_type.code 全局唯一 |
| 唯一性-数据 | sys_dict.(type_code, value) 同类型下唯一 |
| DictTypePageVO | id, name, code, status, remark, createTime |
| DictPageVO | id, name, value, typeCode, defaulted, sort, status, remark, createTime |
| DictTypeForm | id, name, code, status, remark |
| DictForm | id, name, value, typeCode, defaulted, sort, status, remark |
| OptionVO | label, value |
| typeCode在page中 | GET /dict/page 的 typeCode 必填 |
| 错误码 | A0400参数错误 / A0401资源不存在 / A0501数据已存在 / A0504存在关联数据 |

## 二、逐 API 差异对照

### API 1: GET /dict/types/page — 字典类型分页列表

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限 | 仅登录 | 仅登录 ✅ | 仅登录 ✅ | 仅登录 ✅ | 一致 |
| keywords | 可选,匹配name/code | 可选,匹配name OR code ✅ | 可选,匹配name OR code ✅ | 可选,匹配name/code ✅ | 一致 |
| 返回VO | id,name,code,status,remark,createTime | id,name,code,status,remark **缺createTime** ❌ | 全部 ✅ | 全部 ✅ | **Java缺createTime** |
| 排序 | 未明确 | 无 | 无 | create_time DESC | 低优先级 |

**修复项**：Java 的 DictTypePageVO 补充 createTime 字段

---

### API 2: GET /dict/types/{id}/form — 字典类型表单回显

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限 | 仅登录 | 仅登录 ✅ | 仅登录 ✅ | 仅登录 ✅ | 一致 |
| 返回字段 | id,name,code,status,remark | ✅ | ✅ | ✅ | 一致 |
| 不存在时 | A0401 | B0001(IllegalArgumentException) ❌ | A0401 ✅ | 返回None(无错误) ❌ | **Java/Python不一致** |

**修复项**：
- Java：不存在时应返回 A0401 而非 B0001（需用 BusinessException）
- Python：不存在时应返回 A0401 而非返回 None

---

### API 3: POST /dict/types — 新增字典类型

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限码 | sys:dict:type:add | sys:dict_type:add ❌ | **无权限码** ❌ | sys:dict:type:add ✅ | **Java权限码错误, Go缺权限** |
| 校验 | name必填max64, code必填max32 | **无@Valid,无校验注解** ❌ | required,max ✅ | required,min/max ✅ | **Java无校验** |
| 唯一性 | code全局唯一 | **无应用层检查** ❌ | 有 ✅ | 有 ✅ | **Java无唯一性检查** |
| 错误码 | A0501 | DB异常→B0001 ❌ | A0501 ✅ | B0001 ❌ | **Java/Python错误码错误** |
| 防重复 | - | @PreventDuplicateSubmit | 无 | 无 | 低优先级 |

**修复项**：
- Java：① 权限码改为 `sys:dict:type:add` ② DictTypeForm 加校验注解 ③ Controller 加 @Valid ④ Service 加唯一性检查 ⑤ 唯一性冲突返回 A0501
- Go：补权限码中间件 `sys:dict:type:add`
- Python：唯一性冲突返回 A0501 而非 B0001

---

### API 4: PUT /dict/types/{id} — 修改字典类型

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限码 | sys:dict:type:edit | sys:dict_type:edit ❌ | **无权限码** ❌ | sys:dict:type:edit ✅ | **Java权限码错误, Go缺权限** |
| 校验 | name/code必填 | **无@Valid** ❌ | required ✅ | required ✅ | **Java无校验** |
| 唯一性 | code唯一(排除自身) | **无** ❌ | 有 ✅ | 有 ✅ | **Java无唯一性检查** |
| 级联更新 | code变更→同步sys_dict.type_code | 有但**无@Transactional** ❌ | 有,事务 ✅ | **无** ❌ | **Java无事务, Python无级联** |
| 缓存失效 | - | 无缓存 | typeCode变更未清缓存 ❌ | typeCode变更未清缓存 ❌ | **Go/Python缓存缺陷** |
| 错误码 | A0501/A0401 | B0001 ❌ | A0501/A0401 ✅ | B0001 ❌ | **Java/Python错误码错误** |

**修复项**：
- Java：① 权限码改 `sys:dict:type:edit` ② 加 @Valid ③ 加唯一性检查 ④ 级联更新加 @Transactional ⑤ 错误码改 A0501/A0401
- Go：补权限码 `sys:dict:type:edit`；typeCode 变更时清缓存
- Python：① 补级联更新 sys_dict.type_code ② typeCode 变更时清缓存 ③ 错误码改 A0501/A0401

---

### API 5: DELETE /dict/types/{ids} — 删除字典类型

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限码 | sys:dict:type:delete | sys:dict_type:delete ❌ | **无权限码** ❌ | sys:dict:type:delete ✅ | **Java权限码错误, Go缺权限** |
| 关联检查 | 有,禁止删除 | 有 ✅ | 有 ✅ | 有 ✅ | 一致 |
| 错误码 | A0504 | B0001 ❌ | A0504 ✅ | B0001 ❌ | **Java/Python错误码错误** |
| 死代码 | - | 有(remove不可达) | 无 | 无 | **Java清理死代码** |

**修复项**：
- Java：① 权限码改 `sys:dict:type:delete` ② 错误码改 A0504 ③ 清理死代码
- Go：补权限码 `sys:dict:type:delete`
- Python：错误码改 A0504

---

### API 6: GET /dict/page — 字典数据分页列表

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限 | 仅登录 | 仅登录 ✅ | 仅登录 ✅ | 仅登录 ✅ | 一致 |
| typeCode | **必填** | 可选 ❌ | 可选 ❌ | 可选 ❌ | **三端均与文档不符** |
| keywords | 可选 | 匹配name ✅ | 匹配name ✅ | 匹配name/value | **Python匹配范围不同** |
| 返回VO | id,name,value,typeCode,defaulted,sort,status,remark,createTime | id,name,value,status **缺5个字段** ❌ | 全部 ✅ | 全部 ✅ | **Java返回字段不全** |
| 排序 | sort ASC, create_time DESC | **无排序** ❌ | **无排序** ❌ | sort ASC, create_time DESC ✅ | **Java/Go无排序** |
| typeCode缺失时 | A0410 | 无校验 | 无校验 | 无校验 | **三端均缺校验** |

**修复项**：
- Java：① DictPageVO 补全 typeCode/defaulted/sort/remark/createTime 字段 ② 加排序 ③ typeCode 必填校验
- Go：① 加排序 ② typeCode 必填校验
- Python：① typeCode 必填校验 ② keywords 匹配范围对齐(仅name)
- 全端：typeCode 缺失时返回 A0410

---

### API 7: GET /dict/{id}/form — 字典数据表单回显

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限 | 仅登录 | 仅登录 ✅ | 仅登录 ✅ | 仅登录 ✅ | 一致 |
| 返回字段 | id,name,value,typeCode,defaulted,sort,status,remark | id,typeCode,name,value,status,sort,remark **缺defaulted** ❌ | 全部 ✅ | 全部 ✅ | **Java缺defaulted** |
| 不存在时 | A0401 | B0001 ❌ | A0401 ✅ | 返回None ❌ | **Java/Python不一致** |

**修复项**：
- Java：① DictForm/返回补 defaulted 字段 ② 不存在时返回 A0401
- Python：不存在时返回 A0401

---

### API 8: POST /dict — 新增字典数据

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限码 | sys:dict:data:add | sys:dict:add ❌ | **无权限码** ❌ | sys:dict:data:add ✅ | **Java权限码错误, Go缺权限** |
| 校验 | name/value/typeCode必填 | 有@Valid ✅ | required ✅ | required ✅ | 一致(Java校验注解有) |
| 类型存在性 | typeCode必须存在 | **无检查** ❌ | 有 ✅ | 有 ✅ | **Java无类型存在性检查** |
| 唯一性 | (typeCode,value)唯一 | **无** ❌ | 有 ✅ | 有 ✅ | **Java无唯一性检查** |
| 缓存失效 | 清dict:options:{typeCode} | **无缓存** ❌ | 有 ✅ | 有 ✅ | **Java无缓存** |
| defaulted | 默认0 | Form无此字段 ❌ | 有 | Form无此字段 ❌ | **Java/Python缺defaulted** |
| 错误码 | A0501 | DB异常B0001 ❌ | A0501 ✅ | B0001 ❌ | **Java/Python错误码错误** |
| 路径 | POST /dict | ✅ | ✅ | POST /dict/ **尾部斜杠** ❌ | **Python路径不一致** |

**修复项**：
- Java：① 权限码改 `sys:dict:data:add` ② 加类型存在性检查 ③ 加唯一性检查 ④ 加缓存 ⑤ DictForm 加 defaulted 字段 ⑥ 错误码改 A0501
- Go：补权限码 `sys:dict:data:add`
- Python：① 权限码已有 ✅ ② DictForm 加 defaulted 字段 ③ 错误码改 A0501 ④ 路径去尾部斜杠

---

### API 9: PUT /dict/{id} — 修改字典数据

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限码 | sys:dict:data:edit | sys:dict:edit ❌ | **无权限码** ❌ | sys:dict:data:edit ✅ | **Java权限码错误, Go缺权限** |
| 校验 | name/value/typeCode必填 | **无@Valid** ❌ | required ✅ | required ✅ | **Java无校验** |
| typeCode只读 | 不可修改 | **可修改** ❌ | **可修改** ❌ | **可修改** ❌ | **三端均不符** |
| 唯一性 | (typeCode,value)排除自身 | **无** ❌ | 有 ✅ | 有 ✅ | **Java无唯一性检查** |
| 缓存失效 | 清缓存 | **无缓存** ❌ | 有(新旧typeCode) ✅ | 有(新旧typeCode) ✅ | **Java无缓存** |
| defaulted | 可编辑 | Form无此字段 ❌ | 有 | Form无此字段 ❌ | **Java/Python缺defaulted** |
| 错误码 | A0501/A0401 | B0001 ❌ | A0501/A0401 ✅ | B0001 ❌ | **Java/Python错误码错误** |

**修复项**：
- Java：① 权限码改 `sys:dict:data:edit` ② 加 @Valid ③ typeCode 只读(忽略前端传入) ④ 加唯一性检查 ⑤ 加缓存 ⑥ DictForm 加 defaulted ⑦ 错误码改 A0501/A0401
- Go：补权限码 `sys:dict:data:edit`；typeCode 只读
- Python：① typeCode 只读 ② DictForm 加 defaulted ③ 错误码改 A0501/A0401

---

### API 10: DELETE /dict/{ids} — 删除字典数据

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限码 | sys:dict:data:delete | sys:dict:delete ❌ | **无权限码** ❌ | sys:dict:data:delete ✅ | **Java权限码错误, Go缺权限** |
| 缓存失效 | 清缓存(反查typeCode) | **无缓存** ❌ | 有 ✅ | 有 ✅ | **Java无缓存** |
| 错误码 | - | B0001 ❌ | - | A0400 | 低优先级 |

**修复项**：
- Java：① 权限码改 `sys:dict:data:delete` ② 加缓存清理
- Go：补权限码 `sys:dict:data:delete`

---

### API 11: GET /dict/{typeCode}/options — 字典下拉选项

| 维度 | 文档规范 | Java | Go | Python | 差异判定 |
|------|---------|------|----|--------|---------|
| 权限 | 仅登录 | 仅登录 ✅ | 仅登录 ✅ | **无需认证** ❌ | **Python认证不一致** |
| 缓存 | dict:options:{typeCode} TTL 1h | **无缓存** ❌ | 有 ✅ | 有 ✅ | **Java无缓存** |
| 排序 | sort ASC, create_time DESC | **无排序** ❌ | 有 ✅ | 有 ✅ | **Java无排序** |
| 过滤 | - | 无status过滤 | status==1 ✅ | status==1 ✅ | **Java无status过滤** |
| 返回 | [{label, value}] | [{value, label}] ✅ | [{value, label}] ✅ | [{value, label}] ✅ | 一致 |
| 路径 | /dict/{typeCode}/options | ✅ | /dict/options/{typeCode} ❌ | ✅ | **Go路径不一致** |

**修复项**：
- Java：① 加缓存 ② 加排序 ③ 加 status==1 过滤
- Go：① 路径改为 `/dict/{typeCode}/options` ② (已有缓存/排序/过滤)
- Python：① 补认证(需登录态)

---

## 三、修复优先级汇总

### P0 — 阻断性差异（路径/参数/权限码不一致）

| # | API | 端 | 问题 | 修复 |
|---|-----|-----|------|------|
| 1 | API 11 | Go | 路径 `/dict/options/{typeCode}` 应为 `/dict/{typeCode}/options` | 改路由 |
| 2 | API 8 | Python | 路径 `POST /dict/` 尾部斜杠 | 去斜杠 |
| 3 | API 3 | Java | 权限码 `sys:dict_type:add` → `sys:dict:type:add` | 改注解 |
| 4 | API 4 | Java | 权限码 `sys:dict_type:edit` → `sys:dict:type:edit` | 改注解 |
| 5 | API 5 | Java | 权限码 `sys:dict_type:delete` → `sys:dict:type:delete` | 改注解 |
| 6 | API 8 | Java | 权限码 `sys:dict:add` → `sys:dict:data:add` | 改注解 |
| 7 | API 9 | Java | 权限码 `sys:dict:edit` → `sys:dict:data:edit` | 改注解 |
| 8 | API 10 | Java | 权限码 `sys:dict:delete` → `sys:dict:data:delete` | 改注解 |
| 9 | API 3-5,8-10 | Go | 全部6个写接口缺权限码中间件 | 补中间件 |
| 10 | API 11 | Python | 下拉接口无需认证，应需登录 | 补认证依赖 |

### P1 — 逻辑差异（校验/缓存/级联/错误码）

| # | API | 端 | 问题 | 修复 |
|---|-----|-----|------|------|
| 11 | API 3 | Java | DictTypeForm 无校验注解 + Controller 无 @Valid | 加注解+@Valid |
| 12 | API 3 | Java | 无 code 唯一性检查 | Service 加检查 |
| 13 | API 4 | Java | 无 @Valid + 无唯一性检查 + 级联无 @Transactional | 全部补齐 |
| 14 | API 4 | Python | 无级联更新 sys_dict.type_code | Service 加级联 |
| 15 | API 4 | Go/Python | typeCode 变更未清缓存 | 加缓存清理 |
| 16 | API 5 | Java | 死代码(remove不可达) | 删除死代码 |
| 17 | API 6 | 三端 | typeCode 应必填但三端均可选 | 改必填+A0410 |
| 18 | API 6 | Java/Go | 无排序 sort ASC, create_time DESC | 加排序 |
| 19 | API 8 | Java | 无类型存在性检查 + 无唯一性检查 + 无缓存 | 全部补齐 |
| 20 | API 9 | 三端 | typeCode 应只读但三端均可修改 | 改只读 |
| 21 | API 9 | Java | 无 @Valid + 无唯一性 + 无缓存 | 全部补齐 |
| 22 | API 10 | Java | 无缓存清理 | 加缓存 |
| 23 | API 11 | Java | 无缓存 + 无排序 + 无 status 过滤 | 全部补齐 |
| 24 | API 2,7 | Java | 不存在时返回 B0001 应为 A0401 | 改 BusinessException |
| 25 | API 2,7 | Python | 不存在时返回 None 应为 A0401 | 改抛异常 |
| 26 | API 3,4,5,8,9 | Java/Python | 错误码 B0001 应为 A0501/A0504/A0401 | 改错误码 |

### P2 — 字段差异（返回结构不完整）

| # | API | 端 | 问题 | 修复 |
|---|-----|-----|------|------|
| 27 | API 1 | Java | DictTypePageVO 缺 createTime | 补字段 |
| 28 | API 6 | Java | DictPageVO 缺 typeCode/defaulted/sort/remark/createTime | 补字段 |
| 29 | API 7 | Java | DictForm 缺 defaulted | 补字段 |
| 30 | API 8,9 | Java/Python | DictForm 缺 defaulted | 补字段 |
| 31 | API 6 | Python | keywords 匹配 name/value 应仅 name | 改匹配范围 |
| 32 | API 5 | Java | status 语义在 sys_dict_type 表与 Form/VO 相反 | 统一语义 |
