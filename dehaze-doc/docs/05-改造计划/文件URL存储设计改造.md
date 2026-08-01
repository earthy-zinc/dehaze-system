# 文件 URL 存储设计改造

> 状态：待实施
> 创建：2026-07-31
> 关联文档：[文件管理/后端实现](../03-模块设计/基础模块/文件管理/后端实现.md)、[02-系统架构/03-数据库设计](../02-系统架构/03-数据库设计.md)

## 一、背景与问题

### 1.1 现状

`sys_file` 表同时持有 `url`、`object_name`、`path` 三个定位字段：

| 字段 | 含义 | 是否环境相关 |
|------|------|------------|
| `url` | 完整访问地址（TEXT） | **是**，含 `http://host:port/api/v1/files/download/` 前缀 |
| `object_name` | 对象键（如 `upload/20250119/abc.jpg`） | 否 |
| `path` | 逻辑路径，与 `object_name` 语义重叠 | 否 |

上传时把**完整 URL 持久化进 `url`**：

- Java：`FileBOFactory.createFileBO` → `filePathBuilder.buildUrl(objectName)` → `fileBO.setUrl(url)`
- Python：`minio_storage.get_url` / `prediction_service._upload_result` 拼 URL 后入库
- Go：`FileService.UploadFile` → `storageService.GetURL` → 写入 `file.URL`

而前缀来自配置（dev=`http://127.0.0.1:8989/...`、prod=`http://10.16.12.65:8989/...`）。

### 1.2 问题

1. **环境绑定**：`url` 字段存了环境相关前缀，换 host/端口/域名/scheme，库内所有历史 `url` 全部失效，必须批量 `UPDATE`。dev 与 prod 数据不通用。
2. **来源分析混入通用模块**：`sys_file` 混入了两种存储语义的文件——托管文件（MinIO/本地，流式下载）与 nginx 直服的数据集文件（外部静态，302 跳转）。下载接口靠 `url.startsWith(baseUrl)` 这种**前缀启发式判断**区分二者，环境一变即误判。
3. **URL 处理逻辑散落**：`FilePathBuilder.buildUrl`、`FileUploadUtils.createFileBO`、`MinioFileService.setUrl`、`FileController.download` 前缀判断、Python `generate_file_url`/`_upload_result`/`_get_allowed_image_url_prefixes`、`router/file.py` 前缀判断、Go `sys_file.go` 前缀判断……没有单一出口。
4. **任务结果同样受灾**：`sys_task.result` 存的也是完整 URL，存在同样的环境绑定问题。

### 1.3 设计目标

以「**对象键 + 存储后端**」为唯一真相，URL 永远运行时拼接、永不落库，所有存储后端走同一套抽象。文件管理模块只懂"对象键 + 存储后端"，不感知数据集/上传/预测等业务来源。

---

## 二、目标设计

### 2.1 核心原则

1. **数据库只存稳定标识**：`object_name`（对象键）+ `storage`（存储后端标识），与环境无关。
2. **URL 纯运行时拼接**：`url = storage.baseUrl + "/" + object_name`，只在 API 响应时生成，不落库。环境迁移只改配置，数据库零改动。
3. **存储后端统一抽象**：MinIO、本地、nginx 静态服务都是 `StorageService` 的实现，文件管理模块不因后端不同而分支。
4. **下载无分支**：下载接口只认 `object_name + storage`，从对应后端读取，删除前缀判断。
5. **URL 必须是完整地址**：返回给前端的 url 一律是带 scheme+host 的完整 URL（`http(s)://host/...`），禁止后端相对路径（如 `/api/v1/files/...`）。各存储后端 `baseUrl` 配置必须是完整 URL。服务端**接收**文件 URL 时也直接按完整 URL 处理，不再兼容相对路径——特别是 Python 端，删除 `FILE_BASE_URL` 留空回退相对路径 `/api/v1/files/download/...` 的逻辑，不再新增任何拼接相对路径的代码。

### 2.2 数据模型

`sys_file` 表调整：

```sql
-- 新增：存储后端标识（minio / local / nginx-static），默认按全局配置
ALTER TABLE sys_file ADD COLUMN storage VARCHAR(32) NOT NULL DEFAULT 'minio' COMMENT '存储后端标识';

-- 删除：环境相关的完整 URL 字段
ALTER TABLE sys_file DROP COLUMN url;

-- 合并：path 与 object_name 语义重叠，直接删除
ALTER TABLE sys_file DROP COLUMN path;
```

调整后字段：

| 字段 | 含义 | 说明 |
|------|------|------|
| `object_name` | 对象键 | 文件在存储后端中的定位，与环境无关 |
| `storage` | 存储后端标识 | `minio` / `local` / `nginx-static` |
| ~~`url`~~ | 删除 | 运行时拼接，不落库 |
| ~~`path`~~ | 删除 | 与 `object_name` 重叠 |

`sys_task.result` 同步改造：存 `object_name`（导出文件对象键），下载时动态拼接，不再存完整 URL。

### 2.3 存储后端抽象

新增 `nginx-static` 存储后端，使 nginx 直服的数据集文件纳入统一抽象，消除 `saveFileRecord` 特例：

| storage 后端 | baseUrl 配置 | object_name 示例 | 适用 |
|---|---|---|---|
| `minio` | MinIO 访问地址或下载接口地址 | `upload/20250119/abc.jpg` | 用户上传、算法预测结果 |
| `local` | 本地下载接口相对路径 | `upload/20250119/abc.jpg` | 开发/单机部署 |
| `nginx-static` | nginx 静态服务**根地址** | `datasets/AECR-Net/clear/01.jpg` | nginx 直服的静态文件 |

> nginx 静态服务统一托管多种资源——数据集（`/datasets`）、模型权重（`/models`），将来还可能有 css/js 等。`baseUrl` 配置为服务根地址（如 `http://host:9000`），**不带任何具体资源路径**。`object_name` 自带资源类型前缀（`datasets/...`、`models/...`），新增资源类型无需改配置，只需约定前缀。
>
> 注：模型权重当前走独立配置（`algorithm.model.baseUrl=http://host:9000/models` + `sys_algorithm.path`），其 URL 生成方式与 nginx-static 后端一致（根地址 + 相对路径）。是否将其纳入 nginx-static 统一后端为后续可选优化，不在本次改造范围。

`saveFileRecord` 语义保留（"对象已存在于后端，只登记元数据"），但不再传"预构建 url"，改为传 `object_name + storage=nginx-static`。特例消失。

### 2.4 URL 生成：单一出口

`StorageService.get_url(object_name)` 是**唯一**的 URL 生成点。三端统一：

```
url = storage.baseUrl.rstrip("/") + "/" + object_name
```

只在 API 响应序列化时调用。散落的拼接点（`FilePathBuilder.buildUrl`、`generate_file_url`、`_upload_result` 等）全部收敛到这里。

### 2.5 下载：统一，无分支

下载接口只认 `object_name`（+ `storage`），从对应后端读取并流式返回。`url.startsWith(baseUrl)` 前缀判断在 Java/Python/Go 三端下载接口**全部删除**。

下载策略对所有后端统一：后端按 `storage` 选 `StorageService`，调用其 `download(object_name)` 取流转发。是否让前端直连存储（302）由各 `storage.baseUrl` 配置决定，不在代码里分支。

---

## 三、改造项

### 3.1 数据模型（三端共享 schema）

- `config/sql/schema/sys_file.sql`：删除 `url`、`path`，新增 `storage`
- `config/sql/schema/sys_task.sql`：`result` 字段语义由"完整 URL"改为"object_name"（字段类型不变，注释更新）
- 三端实体/模型（Java `SysFile`、Python `SysFile`、Go `model.SysFile`）：移除 `url`/`path`，新增 `storage`
- 三端 `sys_task` 实体：`result` 语义注释更新

### 3.2 Java 端

| 改造点 | 文件 | 动作 |
|--------|------|------|
| 实体 | `model/entity/SysFile.java` | 删 `url`/`path`，加 `storage` |
| FileBO | `model/bo/FileBO.java` | 删 `url` 字段（内部传递不再需要 url） |
| URL 构建 | `common/util/FilePathBuilder.java` | `buildUrl` 改为按 `storage` 取后端 baseUrl 拼接；删除 `getBaseUrl`（前缀判断用） |
| FileBO 工厂 | `common/util/FileBOFactory.java`、`FileUploadUtils.java` | 不再 `setUrl`；只设 `objectName`/`storage` |
| 存储服务 | `service/impl/file/MinioFileService.java`、`LocalFileService.java` | 上传后不再 `setUrl`；实现按 `storage` 注册 |
| 文件登记 | `service/impl/SysFileServiceImpl.java` | `saveFile`/`saveFileRecord` 统一为"存 object_name + storage"；`saveFileRecord` 不再接预构建 url |
| 下载接口 | `controller/FileController.java` | 删除 `!file.getUrl().startsWith(baseUrl)` → 302 分支；按 `object_name + storage` 选后端读取 |
| 响应序列化 | 各 VO（`FileVO`/`ImageUrlVO`/`ItemFileVO`）填充处 | `url` 字段改为调用 `storageService.get_url(objectName)` 动态填充 |
| 任务结果 | `service/impl/TaskServiceImpl.java`、导出策略 | `result` 存 `object_name`；`getDownloadUrl` 动态拼接返回 |
| 缩略图 | `SysItemFileServiceImpl` | 缩略图文件同样按新模型处理 |
| 单测 | `FilePathBuilderTest`、`FileBOFactoryTest`、`SysItemFileServiceIT` 等 | 移除 url 断言，改为 object_name + storage 断言；测试数据不再 setUrl |

### 3.3 Python 端

| 改造点 | 文件 | 动作 |
|--------|------|------|
| 配置 | `app/config.py` | 删除 `FILE_BASE_URL` 留空回退相对路径 `/api/v1/files/download/...` 的逻辑；`baseUrl` 必须配置为完整 URL，运行时拼接用、不入库；接收侧不再处理相对路径 |
| 存储 | `app/service/storage/base.py`、`minio_storage.py`、`local_storage.py` | `get_url` 统一为 baseUrl+object_name；新增按 `storage` 选后端的工厂 |
| 新增后端 | `app/service/storage/nginx_storage.py`（新增） | nginx 静态服务后端实现 |
| 文件服务 | `app/service/file_service.py` | `generate_file_url` 收敛为调用 storage.get_url；入库不再写 url |
| 预测结果 | `app/service/prediction_service.py` | `_upload_result` 返回 object_name，url 由响应层拼 |
| 反馈校验 | `app/service/feedback_service.py` | `_get_allowed_image_url_prefixes` 改为基于 storage baseUrl 校验，或按 object_name 校验 |
| 下载路由 | `app/router/file.py` | 删除 `file_info.url.startswith(FILE_BASE_URL)` → 302 分支；按 storage 选后端读取 |
| 数据集服务 | `app/service/dataset_service.py` | `_build_file_vo` 的 `url`/`thumbnailUrl` 动态拼 |
| 模型 | `app/models/schema/file.py`、实体 | 移除 url 持久化字段，新增 storage |

### 3.4 Go 端

| 改造点 | 文件 | 动作 |
|--------|------|------|
| 存储接口 | `pkg/storage/interface.go` | `GetURL` 语义明确为运行时拼接 |
| 存储工厂 | `pkg/storage/factory.go` | 新增 `nginx-static` 分支；支持按 `storage` 取实例 |
| 新增后端 | `pkg/storage/nginx.go`（新增） | nginx 静态服务后端实现（Download = HTTP GET） |
| MinIO/Local | `pkg/storage/minio.go`、`local.go` | `GetURL` 改为 baseUrl+object_name（minio 不再返回 endpoint/bucket 拼接） |
| 文件服务 | `internal/service/file/sys_file.go` | 上传不再写 URL；`UploadFile` 入参移除 `baseURL` |
| 下载接口 | `internal/api/sys_file.go` | 删除 `!strings.HasPrefix(*file.URL, cfg.File.BaseURL)` → 302 分支；按 storage 选后端读取 |
| 任务结果 | `internal/service/task/task_service.go` | `DownloadExportFile` 动态拼接返回 |
| 配置 | `config/config.yaml` | `file.baseUrl` 语义为运行时拼接；新增 nginx-static 后端配置 |
| 模型 | `internal/model/SysFile` | 删 URL/Path，加 Storage |

### 3.5 nginx-static 存储后端（三端新增）

- 配置项：`file.nginx-static.baseUrl`（nginx 静态服务**根地址**，如 `http://host:9000`，不带 `/datasets`、`/models` 等资源子路径）
- `Download(object_name)`：HTTP GET `{baseUrl}/{object_name}` 取流（object_name 含资源前缀，如 `datasets/AECR-Net/clear/01.jpg`、`models/AECR-Net/NH_train.pk`）
- `GetURL(object_name)`：返回 `{baseUrl}/{object_name}`（完整 URL）
- `Exists(object_name)`：HTTP HEAD 校验
- 数据集导入流程改为：文件已在 nginx 目录 → 调 `saveFileRecord(object_name="datasets/...", storage=nginx-static)` 登记元数据

### 3.6 散落 URL 处理收敛点（清单）

改造后以下位置的 URL 拼接/判断全部收敛到 `StorageService.get_url`，不再各自拼接：

- Java：`FilePathBuilder.buildUrl`、`FileBOFactory`、`FileUploadUtils.createFileBO`、`FileController.download` 前缀判断
- Python：`file_service.generate_file_url`、`prediction_service._upload_result`、`feedback_service._get_allowed_image_url_prefixes`、`router/file.py` 前缀判断、`local_storage.get_url`/`minio_storage.get_url`
- Go：`sys_file.go` download 前缀判断、`FileService.UploadFile` 的 baseURL 拼接兜底

### 3.7 初始化脚本（scripts/）

`scripts/init_dataset.py` 与 `scripts/init_wpx_file.py` 直接操作 `sys_file` 表写入 `url`/`path`/`object_name`，需同步改造：

| 脚本 | 当前问题 | 改造动作 |
|------|---------|---------|
| `init_dataset.py` | `insert_file_record` 写 `url=f"{dataset_base_url}/{rel}"`、`path=rel`；`--dataset-base-url` 传 nginx 含 `/datasets` 的地址 | 不再写 `url`/`path`；写入 `storage='nginx-static'`；`object_name` 改为 `datasets/{rel}`（含资源前缀，对齐 nginx-static 根地址拼接）；`--dataset-base-url` 参数语义改为 nginx 根地址（如 `http://host:9000`），或重命名为 `--nginx-base-url` |
| `init_wpx_file.py` | `insert_file_record` 写 `url=f"{base_url}/{new_rel}"`、`path=new_rel`；同样 `--dataset-base-url` 传含 `/datasets` 地址 | 同上：不写 `url`/`path`，写 `storage='nginx-static'`，`object_name` 改为 `datasets/{new_rel}`（WPX 图在 `datasets/WPX/...` 下）；`--dataset-base-url` 语义改为 nginx 根地址 |

要点：
- nginx-static 后端 `baseUrl` 是根地址（`http://host:9000`），`object_name` 必须含 `datasets/` 资源前缀，才能拼出 `http://host:9000/datasets/...`。脚本内 `rel`（相对数据集根目录的路径）需前置 `datasets/`。
- `sys_wpx_file` 表的 `origin_path`/`new_path` 为路径快照、非访问 URL，本次不强制改动；若需对齐可一并清理冗余 path。

---

## 四、数据迁移

> 现有数据库中暂无文件记录，**无需数据迁移**。schema 变更（新增 `storage`、删除 `url`/`path`）直接执行即可，不涉及存量数据回填与 `sys_task.result` 前缀剥离。

仍遵循"禁止兼容历史烂逻辑"：三端代码与 schema 一次性切换，不保留旧 `url` 字段读取的兜底分支。

---

## 五、改造后的使用方式

### 5.1 上传：不再碰 URL

```java
// Java：FileBOFactory 只产 object_name + storage，不 setUrl
FileBO bo = fileBOFactory.createFileBO(file, "upload/20250119");
// bo.objectName = "upload/20250119/abc.jpg", bo.storage = "minio"
sysFileService.saveFile(bo);
// 入库：object_name + storage，无 url 字段
```

```python
# Python：预测结果上传只返回 object_name
object_name = "predictions/20250731/xxx.png"
storage.upload("dehaze", object_name, data, "image/png")
# 入库/返回只存 object_name + storage="minio"
```

### 5.2 响应：URL 运行时拼

```java
// VO 填充：url 动态生成，永远反映当前环境
vo.setUrl(storageService.get_url(sysFile.getObjectName(), sysFile.getStorage()));
// 例：minio 后端     → http://当前host:port/api/v1/files/download/upload/20250119/abc.jpg
//     nginx-static 后端 → http://nginx-host/datasets/AECR-Net/clear/01.jpg  （根地址 + 资源前缀）
```

前端拿到的 `url` 始终是当前环境正确地址，环境迁移后无需改库。

### 5.3 下载：统一无分支

```
GET /api/v1/files/download/{objectName}
```

后端：查 `sys_file` 拿到 `storage` → 按 `storage` 选 `StorageService` → `download(object_name)` 流式返回。无前缀判断、无 302 分支。无论文件在 MinIO、本地、还是 nginx，下载链路一致。

### 5.4 数据集文件登记：纳入统一抽象

```java
// 旧：saveFileRecord(fileBO)  // url 已预构建为 nginx 完整地址
// 新：统一登记，只传 object_name + storage
FileBO bo = FileBO.builder()
    .objectName("datasets/AECR-Net/clear/01.jpg")
    .storage("nginx-static")
    .md5(md5).name("01.jpg").build();
sysFileService.saveFileRecord(bo);  // 不再特例，与托管文件同构
```

### 5.5 环境迁移：只改配置

```yaml
# dev（所有 baseUrl 必须是完整 URL，禁止相对路径）
file:
  storage:
    minio:        { baseUrl: http://127.0.0.1:8989/api/v1/files/download }
    local:        { baseUrl: http://127.0.0.1:8989/api/v1/files/download }
    nginx-static: { baseUrl: http://127.0.0.1:9000 }   # 根地址，不带 /datasets

# prod（数据库零改动，只改配置）
file:
  storage:
    minio:        { baseUrl: http://10.16.12.65:8989/api/v1/files/download }
    local:        { baseUrl: http://10.16.12.65:8989/api/v1/files/download }
    nginx-static: { baseUrl: http://cdn.example.com }  # 根地址
```

### 5.6 前后对比

| 场景 | 改造前 | 改造后 |
|------|--------|--------|
| 环境迁移 | 批量 `UPDATE sys_file.url` 替换前缀 | 改配置，数据库不动 |
| 下载接口 | `if url.startsWith(baseUrl) 流式 else 302` | 按 `storage` 选后端读取，无分支 |
| URL 拼接 | 散落 8+ 处各自拼 | `StorageService.get_url` 单一出口 |
| 数据集文件 | `saveFileRecord` + 预构建 url 特例 | 纳入 nginx-static 后端，与托管文件同构 |
| `sys_file` 字段 | url + object_name + path（3 个重叠） | object_name + storage（2 个职责清晰） |

---

## 六、影响面与文档同步

### 6.1 代码影响面

- **三端后端**：dehaze-java、dehaze-python、dehaze-go（表结构、实体、存储服务、文件服务、下载接口、任务结果）
- **单元/集成测试**：三端涉及文件/任务的测试（url 断言改 object_name+storage 断言；测试数据不再 setUrl）
- **SDK 接口测试**：dehaze-sdk-js/test（若断言 url 完整值，改为校验 object_name 或动态 url 规则）
- **初始化脚本**：scripts/init_dataset.py、scripts/init_wpx_file.py（写库逻辑改造）
- **前端**：dehaze-front-react、dehaze-front-vue、dehaze_flutter、dehaze-react-native、dehaze-android、dehaze-uniapp、dehaze-taro——前端必须**直接使用后端返回的完整 URL**，禁止自行拼接相对路径或 baseUrl；若现有代码有用 `/api/v1/files/...` 相对路径访问文件的逻辑，需改为使用接口返回的完整 URL

### 6.2 文档同步

- `dehaze-doc/docs/03-模块设计/基础模块/文件管理/后端实现.md`：§3 FileBO 去掉 url；§4 上传/下载流程去 url 持久化与前缀判断；§6.3 存储路径规则补 nginx-static
- `dehaze-doc/docs/03-模块设计/基础模块/文件管理/API接口.md`：url 字段说明改为"运行时动态生成"
- `dehaze-doc/docs/02-系统架构/03-数据库设计.md`：sys_file 字段调整、sys_task.result 语义
- `dehaze-doc/docs/03-模块设计/基础模块/任务管理/后端实现.md`：导出结果存储改为 object_name
- `dehaze-doc/docs/03-模块设计/核心模块/数据集管理/后端实现.md`：数据集文件登记改 nginx-static 后端
- 各端架构文档（Java/Python/Go）：存储后端抽象与 URL 生成说明

---

## 七、实施步骤

1. **schema 与实体**：改 `sys_file`/`sys_task` schema，三端实体同步（先加 `storage`，暂不删 `url`/`path`）
2. **存储后端抽象**：三端 `StorageService` 统一 `get_url`；新增 nginx-static 后端实现与配置
3. **URL 单一出口**：三端上传/登记不再写 url，响应层统一动态拼
4. **下载接口去分支**：三端删除前缀判断，按 storage 选后端读取
5. **任务结果改造**：`sys_task.result` 改存 object_name，下载动态拼
6. **初始化脚本改造**：改 `scripts/init_dataset.py`、`scripts/init_wpx_file.py`（不写 url/path，写 storage=nginx-static，object_name 含 `datasets/` 前缀）
7. **删除 url/path 字段**：三端代码与脚本切换完成后，DROP COLUMN
8. **测试**：三端单测/集成测试改写；SDK 接口测试校验
9. **文档同步**：按 §6.2 更新所有相关文档

每步在三端同步推进，不跨端遗留中间态。
