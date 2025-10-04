## 📦 `pei-workflow` 模块概述

`pei-workflow` 是基于 **WarmFlow** 实现的工作流引擎模块，提供流程定义、流程实例管理、任务处理等核心功能。它是 RuoYi-Cloud-Plus 项目中用于支持业务流程自动化的重要组件。

---

## 🧩 主要功能及其实现

### 1. **流程定义管理**
- **作用**：定义和管理流程模型（如审批流程、请假流程等）。
- **实现方式**：
    - 在 [FlwDefinitionController.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwDefinitionController.java) 中提供 REST 接口。
    - 使用 [IFlwDefinitionService.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\IFlwDefinitionService.java#L20-L49) 接口封装流程定义的业务逻辑。
    - 具体实现在 [FlwDefinitionServiceImpl.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwDefinitionServiceImpl.java#L30-L578)。
- **关键方法**：
    - [list(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwDefinitionController.java#L48-L52)：查询流程定义列表。
    - [add(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwDefinitionController.java#L92-L96)：新增流程定义。
    - [publish(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwDefinitionController.java#L117-L121)：发布流程定义。
    - [exportDef(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwDefinitionController.java#L159-L163)：导出流程定义为 JSON 文件。
- **数据结构**：
    - [FlowDefinition](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\FlowDefinition.java#L18-L120)：流程定义实体类。
    - [FlowDefinitionVo](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\vo\FlowDefinitionVo.java#L18-L120)：流程定义视图对象。
    - [Definition](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\warm\flow\core\entity\Definition.java#L16-L132)：流程定义的核心实体（来自 WarmFlow 引擎）。

---

### 2. **流程实例管理**
- **作用**：管理流程的运行时实例，包括启动、终止、挂起等操作。
- **实现方式**：
    - 提供 [FlwInstanceController.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwInstanceController.java#L18-L250) 控制器。
    - 接口 [IFlwInstanceService.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\IFlwInstanceService.java#L20-L88) 定义了流程实例的 CRUD 操作。
    - 实现类 [FlwInstanceServiceImpl.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwInstanceServiceImpl.java#L30-L646) 处理具体逻辑。
- **关键方法**：
    - [startProcess(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwInstanceServiceImpl.java#L100-L114)：启动一个新流程实例。
    - [terminate(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwInstanceServiceImpl.java#L300-L314)：终止流程实例。
    - [suspend(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwInstanceServiceImpl.java#L330-L344)：挂起流程实例。
- **数据结构**：
    - [FlowInstance](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\FlowInstance.java#L18-L104)：流程实例实体。
    - [FlowInstanceVo](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\vo\FlowInstanceVo.java#L18-L104)：流程实例视图对象。

---

### 3. **任务管理**
- **作用**：处理用户在流程中的任务（如审批、退回、完成等）。
- **实现方式**：
    - 控制器 [FlwTaskController.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwTaskController.java#L20-L320) 提供任务相关接口。
    - 接口 [IFlwTaskService.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\IFlwTaskService.java#L20-L108) 定义任务操作。
    - 实现类 [FlwTaskServiceImpl.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwTaskServiceImpl.java#L30-L1078) 实现任务逻辑。
- **关键方法**：
    - [completeTask(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwTaskServiceImpl.java#L150-L164)：完成当前任务。
    - [backTask(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwTaskServiceImpl.java#L300-L314)：退回任务。
    - [claim(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwTaskServiceImpl.java#L400-L414)：签收任务。
- **数据结构**：
    - [FlowTask](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\FlowTask.java#L18-L104)：任务实体。
    - [FlowTaskVo](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\vo\FlowTaskVo.java#L18-L104)：任务视图对象。
    - [CompleteTaskBo](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\bo\CompleteTaskBo.java#L18-L64)：任务完成请求参数。

---

### 4. **流程分类管理**
- **作用**：对流程进行分类管理，便于组织和权限控制。
- **实现方式**：
    - 控制器 [FlwCategoryController.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwCategoryController.java#L18-L168) 提供 REST 接口。
    - 接口 [IFlwCategoryService.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\IFlwCategoryService.java#L20-L58) 定义服务接口。
    - 实现类 [FlwCategoryServiceImpl.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwCategoryServiceImpl.java#L30-L348) 实现分类管理。
- **关键方法**：
    - [list(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwCategoryController.java#L48-L52)：获取所有流程分类。
    - [add(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwCategoryController.java#L92-L96)：新增流程分类。
    - [remove(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\FlwCategoryController.java#L132-L136)：删除流程分类。
- **数据结构**：
    - [FlowCategory](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\FlowCategory.java#L18-L104)：流程分类实体。
    - [FlowCategoryVo](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\vo\FlowCategoryVo.java#L18-L104)：流程分类视图对象。

---

### 5. **流程引擎集成**
- **作用**：集成 WarmFlow 流程引擎，处理流程执行、节点跳转等底层逻辑。
- **实现方式**：
    - 使用 [WarmFlowConfig.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\config\WarmFlowConfig.java#L16-L30) 配置流程引擎。
    - 核心接口 [DefService.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\warm\flow\core\service\DefService.java#L16-L128) 封装流程定义服务。
    - 流程执行由 [WorkflowService.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\WorkflowService.java#L18-L88) 管理。
- **关键类**：
    - [DefService](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\warm\flow\core\service\DefService.java#L16-L128)：流程定义服务接口。
    - [NodeService](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\warm\flow\core\service\NodeService.java#L16-L128)：节点服务接口。
    - [TaskService](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\warm\flow\core\service\TaskService.java#L16-L128)：任务服务接口。

---

### 6. **示例流程：TestLeave（请假流程）**
- **作用**：提供一个典型的业务流程示例，演示如何使用流程引擎实现请假审批。
- **实现方式**：
    - 控制器 [TestLeaveController.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\controller\TestLeaveController.java#L18-L148) 提供请假流程接口。
    - 接口 [ITestLeaveService.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\ITestLeaveService.java#L18-L58) 定义请假服务。
    - 实现类 [TestLeaveServiceImpl.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\TestLeaveServiceImpl.java#L30-L328) 实现请假逻辑。
- **关键方法**：
    - [apply(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\TestLeaveServiceImpl.java#L50-L64)：提交请假申请。
    - [audit(...)](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\TestLeaveServiceImpl.java#L80-L94)：审核请假申请。
- **数据结构**：
    - [TestLeave](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\TestLeave.java#L18-L98)：请假申请实体。
    - [TestLeaveVo](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\vo\TestLeaveVo.java#L18-L98)：请假申请视图对象。
    - [TestLeaveBo](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\bo\TestLeaveBo.java#L18-L138)：请假申请业务对象。

---

### 7. **流程监听与事件处理**
- **作用**：监听流程生命周期事件（如流程启动、任务创建、任务完成等），并触发相应的业务逻辑。
- **实现方式**：
    - 使用 [listener](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\listener) 包中的监听器。
    - 通过 [WorkflowListener.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\listener\WorkflowListener.java#L18-L108) 监听流程事件。
- **典型事件**：
    - 流程启动事件。
    - 任务创建事件。
    - 任务完成事件。
    - 流程结束事件。

---

### 8. **流程配置与扩展**
- **作用**：提供流程引擎的自定义配置，如流程变量、节点扩展属性等。
- **实现方式**：
    - 使用 [common](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\common) 包中的工具类。
    - 扩展 [FlwNodeExtServiceImpl.java](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\service\impl\FlwNodeExtServiceImpl.java#L30-L198) 实现节点扩展逻辑。
- **关键类**：
    - [FlwNodeExt](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\FlwNodeExt.java#L18-L104)：流程节点扩展信息。
    - [FlwNodeExtVo](file://E:\ProgramProject\RuoYi-Cloud-Plus\pei-modules\pei-workflow\src\main\java\org\dromara\workflow\domain\vo\FlwNodeExtVo.java#L18-L104)：流程节点扩展视图对象。

---

## 📁 包结构详解

```
com.pei.workflow
├── config/               // 配置类
│   └── WarmFlowConfig.java // 流程引擎配置
├── controller/           // 控制器层
│   ├── FlwDefinitionController.java // 流程定义控制器
│   ├── FlwInstanceController.java // 流程实例控制器
│   ├── FlwTaskController.java // 任务控制器
│   └── TestLeaveController.java // 请假流程控制器
├── domain/               // 数据传输对象（DTO）
│   ├── FlowCategory.java // 流程分类实体
│   ├── FlowDefinition.java // 流程定义实体
│   ├── FlowInstance.java // 流程实例实体
│   ├── FlowTask.java // 流程任务实体
│   ├── bo/ // 业务对象
│   └── vo/ // 视图对象
├── mapper/               // 数据库操作接口
│   ├── FlwCategoryMapper.java // 分类 Mapper
│   ├── FlwDefinitionMapper.java // 定义 Mapper
│   ├── FlwInstanceMapper.java // 实例 Mapper
│   └── FlwTaskMapper.java // 任务 Mapper
├── service/              // 业务逻辑层
│   ├── IFlwDefinitionService.java // 流程定义服务接口
│   ├── IFlwInstanceService.java // 流程实例服务接口
│   ├── IFlwTaskService.java // 任务服务接口
│   └── impl/ // 服务实现类
├── handler/              // 自定义流程处理器
│   ├── NodeHandler.java // 节点处理器
│   └── TaskHandler.java // 任务处理器
└── listener/             // 流程监听器
    └── WorkflowListener.java // 流程生命周期监听器
```


---

## 🧠 技术栈与架构设计

### 技术栈
| 技术 | 用途 |
|------|------|
| Spring Boot | 快速构建微服务应用 |
| MyBatis Plus | 数据库操作 |
| Dubbo | 服务间通信 |
| Lombok | 减少样板代码 |
| WarmFlow | 流程引擎核心 |
| Hutool | 工具类库 |

### 架构图（文字描述）

```
[REST API] → [Controller] → [Service] → [Mapper]
                              ↓
                        [流程引擎: WarmFlow]
                              ↓
                         [数据库: MySQL/PostgreSQL]
```


---

## ✅ 总结

`pei-workflow` 模块是一个完整的流程引擎模块，具备以下核心能力：

- **流程定义管理**：支持流程建模、发布、导入、导出。
- **流程实例管理**：支持流程启动、终止、挂起、恢复。
- **任务处理**：支持任务签收、完成、退回、指派。
- **流程监听**：支持流程生命周期事件监听。
- **示例流程**：提供请假流程作为示例，方便开发者学习和参考。

