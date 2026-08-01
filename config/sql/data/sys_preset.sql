SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- 系统预设种子数据
INSERT INTO `sys_preset` (`name`, `type`, `algorithm_id`, `params`, `user_id`, `is_default`) VALUES
('默认去雾', 'system', 13, '{"gamma": 1.0, "strength": "medium"}', NULL, 1),
('轻度去雾', 'system', 13, '{"gamma": 0.8, "strength": "light"}', NULL, 0),
('深度去雾', 'system', 13, '{"gamma": 1.5, "strength": "strong"}', NULL, 0)
ON DUPLICATE KEY UPDATE `name` = VALUES(`name`);

SET FOREIGN_KEY_CHECKS = 1;