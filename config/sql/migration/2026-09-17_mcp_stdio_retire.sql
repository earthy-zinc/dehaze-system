-- 2026-09-17 外部 MCP Server：stdio 协议退役 + 健康巡检探测时间字段
-- 开发库增量执行（测试库由 conftest 全量重建自动生效）
-- stdio 为本地进程协议，无网络端点：工具拉取/运行时装载/健康探测三处均不支持，
-- 存量注册记录统一置禁用，管理员须改用 streamable-http/sse 后重新启用。
UPDATE `sys_ai_mcp_server`
SET `status` = 0
WHERE `protocol_type` = 'stdio'
  AND `deleted` = 0;

ALTER TABLE `sys_ai_mcp_server`
    ADD COLUMN `last_check_time` datetime NULL DEFAULT NULL COMMENT '最近一次健康探测时间' AFTER `health`;
