SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_member_benefit (level_code, level_name, growth_min, growth_max,
                                monthly_dehaze_quota, monthly_derain_quota, monthly_desnow_quota,
                                monthly_lowlight_quota, monthly_super_resolution_quota,
                                monthly_denoise_quota, monthly_inpaint_quota, monthly_evaluate_quota,
                                history_retention, batch_limit, priority,
                                advanced_params, hd_export, report_export, batch_download,
                                ai_credits_daily, ai_credits_monthly, vip_gift_credits,
                                multimodal_limit,
                                sort, status, deleted, create_time, update_time, create_by, update_by)
values ('level_0', '普通用户', 0, 999,
        20, 20, 20, 20, 10, 20, 10, 20,
        100, 10, 1, 0, 0, 0, 0,
        10000, 100000, 0, 5,
        1, 1, 0, '2026-07-01 00:00:00', '2026-07-01 00:00:00', null, null),
       ('level_1', 'VIP1', 1000, 4999,
        100, 100, 100, 100, 50, 100, 50, 100,
        500, 50, 2, 1, 1, 1, 1,
        50000, 500000, 1000, 10,
        2, 1, 0, '2026-07-01 00:00:00', '2026-07-01 00:00:00', null, null),
       ('level_2', 'VIP2', 5000, 19999,
        500, 500, 500, 500, 200, 500, 200, 500,
        2000, 200, 3, 1, 1, 1, 1,
        200000, 2000000, 5000, 20,
        3, 1, 0, '2026-07-01 00:00:00', '2026-07-01 00:00:00', null, null),
       ('level_3', 'SVIP', 20000, 0,
        3000, 3000, 3000, 3000, 1000, 3000, 1000, 3000,
        10000, 1000, 4, 1, 1, 1, 1,
        1000000, 10000000, 20000, 50,
        4, 1, 0, '2026-07-01 00:00:00', '2026-07-01 00:00:00', null, null);

SET FOREIGN_KEY_CHECKS = 1;
