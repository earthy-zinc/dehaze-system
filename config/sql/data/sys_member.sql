SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_member (user_id, level_code, level_source, growth_value, total_consumption, expire_time,
                        become_member_time, monthly_dehaze_quota, monthly_dehaze_used,
                        monthly_evaluate_quota, monthly_evaluate_used, quota_reset_month,
                        status, deleted, create_time, update_time, create_by, update_by)
values (4, 'level_0', 'growth', 100, 0, null, null, 20, 0, 20, 0, 202607, 1, 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2),
       (5, 'level_0', 'growth', 100, 0, null, null, 20, 0, 20, 0, 202607, 1, 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2),
       (6, 'level_1', 'growth', 1500, 9900, null, '2024-06-08 19:05:51', 100, 0, 100, 0, 202607, 1, 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2),
       (7, 'level_2', 'growth', 8000, 49900, null, '2024-06-08 19:05:51', 500, 0, 500, 0, 202607, 1, 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2),
       (8, 'level_3', 'growth', 25000, 199900, null, '2024-06-08 19:05:51', 3000, 0, 3000, 0, 202607, 1, 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2);

SET FOREIGN_KEY_CHECKS = 1;
