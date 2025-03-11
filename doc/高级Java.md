一、核心技术能力

1. Java语言深度掌握
   核心特性：精通集合框架、多线程（并发包java.util.concurrent）、IO/NIO、反射、泛型、注解等。
   JVM原理：理解内存模型（堆、栈、方法区）、垃圾回收算法（CMS、G1、ZGC）、类加载机制、性能调优（JVM参数配置、内存泄漏排查）。
   新版本特性：熟练使用Java 8+特性（Lambda、Stream API、Optional），了解模块化（Project Jigsaw）、记录类（Record）、模式匹配等。
2. 主流框架与工具
   Spring生态：
   Spring Boot：自动配置、Starter开发、Actuator监控。
   Spring MVC：RESTful API设计、拦截器、全局异常处理。
   Spring Cloud：服务注册与发现（Eureka/Nacos）、配置中心（Config/Apollo）、熔断（Hystrix/Sentinel）、网关（Zuul/Gateway）、分布式事务（Seata）。
   Spring Data：JPA与MyBatis整合、复杂查询优化。
   ORM框架：熟练使用JPA/Hibernate、MyBatis（动态SQL、二级缓存）。
   其他工具：Lombok、MapStruct、Guava、Apache Commons。
3. 数据库与存储技术
   关系型数据库：
   MySQL/PostgreSQL：索引优化（B+树、覆盖索引）、事务隔离级别、锁机制（行锁、间隙锁）、分库分表（ShardingSphere）、读写分离。
   执行计划分析：EXPLAIN工具使用、慢查询优化。
   NoSQL：
   Redis：数据结构（String/Hash/ZSet）、持久化（RDB/AOF）、集群（主从、Cluster）、缓存穿透/雪崩解决方案。
   MongoDB：文档模型设计、聚合管道、副本集与分片。
   Elasticsearch：倒排索引、DSL查询、分词器优化。
4. 分布式系统设计
   分布式理论：CAP定理、BASE理论、一致性算法（Raft/Paxos）。
   分布式事务：TCC模式、Saga模式、消息最终一致性（本地消息表）。
   消息队列：
   Kafka：高吞吐设计、分区与副本机制、Exactly-Once语义。
   RabbitMQ：交换机类型、死信队列、消息确认机制。
   服务治理：服务熔断降级、限流策略（令牌桶/漏桶）、API网关（鉴权、路由、限流）。
5. 性能优化与高并发
   系统调优：JVM参数优化（堆大小、GC策略）、线程池配置（核心参数、拒绝策略）、减少上下文切换。
   高并发场景：缓存策略（多级缓存）、异步处理（CompletableFuture/Reactive编程）、数据库连接池（HikariCP）、CDN加速。
   压测工具：JMeter、Gatling、wrk的使用与分析。
   二、架构与设计能力
1. 系统架构设计
   架构模式：微服务、事件驱动（Event Sourcing）、CQRS、Serverless。
   设计原则：SOLID原则、DRY、KISS，熟悉DDD（领域驱动设计）的战术与战略模式。
   设计模式：工厂、代理、策略、观察者、责任链等模式的实际应用。
2. 可扩展性与容灾
   水平扩展：无状态服务设计、数据库分片、分布式Session管理。
   容灾策略：多活架构、异地容灾、数据备份与恢复（XtraBackup/LVM快照）。
3. API设计
   RESTful规范：资源命名、HTTP状态码、HATEOAS。
   GraphQL：按需查询、Schema设计、Resolver实现。
   文档工具：Swagger/OpenAPI、Postman。
   三、DevOps与工程实践
1. 持续集成与交付（CI/CD）
   工具链：Jenkins Pipeline、GitLab CI、GitHub Actions。
   自动化流程：单元测试（JUnit/Mockito）、集成测试、SonarQube代码质量检查。
   容器化：Docker镜像构建、Kubernetes部署（Deployment/Service/Ingress）、Helm Chart管理。
2. 监控与运维
   日志系统：ELK（Elasticsearch、Logstash、Kibana）或Loki+Grafana。
   链路追踪：SkyWalking、Zipkin、Jaeger。
   监控告警：Prometheus（指标采集）、Alertmanager（规则配置）、Grafana可视化。
3. 基础设施即代码（IaC）
   工具：Terraform（云资源编排）、Ansible（配置管理）、CloudFormation（AWS专用）。
   四、安全与合规
1. 应用安全
   认证与授权：OAuth2.0、JWT、RBAC/ABAC权限模型。
   漏洞防护：防止SQL注入、XSS、CSRF、SSRF（使用工具如OWASP ZAP扫描）。
   数据安全：加密算法（AES/RSA）、敏感信息脱敏、合规要求（GDPR、等保）。
2. 网络安全
   HTTPS配置：证书管理（Let's Encrypt）、TLS版本选择。
   防火墙规则：安全组配置（云平台）、DDoS防护（Cloudflare/AWS Shield）。
   五、软技能与综合素质
1. 团队协作
   代码规范：遵守团队编码规范，熟练使用Git（分支策略、Rebase/Cherry-pick）。
   Code Review：提供建设性反馈，关注代码可维护性与性能。
2. 项目管理
   敏捷开发：Scrum流程（Sprint规划、每日站会）、任务拆分（用户故事地图）。
   文档能力：编写技术方案、架构设计文档、API手册。
3. 问题解决
   根因分析：熟练使用Arthas、jstack、jmap定位问题，结合日志与监控快速响应。
   技术预研：评估新技术（如Quarkus、GraalVM）的适用性，输出对比报告。
   六、扩展技能（加分项）
   云原生技术：Service Mesh（Istio）、Serverless（AWS Lambda）、云数据库（Aurora/CosmosDB）。
   大数据处理：Hadoop/Spark流式计算、Flink实时处理。
   AI集成：TensorFlow Serving模型部署、Python脚本编写。
