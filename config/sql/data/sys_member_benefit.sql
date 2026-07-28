SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_member_benefit (level_code, level_name, growth_min, growth_max,
                                monthly_dehaze_quota, monthly_evaluate_quota,
                                history_retention, batch_limit, priority,
                                advanced_params, hd_export, report_export, batch_download,
                                sort, status, deleted, create_time, update_time, create_by, update_by)
values ('level_0', '普通用户', 0, 999, 20, 20, 100, 10, 1, 0, 0, 0, 0, 1, 1, 0, '2026-07-01 00:00:00', '2026-07-01 00:00:00', null, null),
       ('level_1', 'VIP1', 1000, 4999, 100, 100, 500, 50, 2, 1, 1, 1, 1, 2, 1, 0, '2026-07-01 00:00:00', '2026-07-01 00:00:00', null, null),
       ('level_2', 'VIP2', 5000, 19999, 500, 500, 2000, 200, 3, 1, 1, 1, 1, 3, 1, 0, '2026-07-01 00:00:00', '2026-07-01 00:00:00', null, null),
       ('level_3', 'SVIP', 20000, 0, 3000, 3000, 10000, 1000, 4, 1, 1, 1, 1, 4, 1, 0, '2026-07-01 00:00:00', '2026-07-01 00:00:00', null, null);

SET FOREIGN_KEY_CHECKS = 1;
