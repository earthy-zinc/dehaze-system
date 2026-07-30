SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_user (id, username, nickname, gender, password, dept_id, avatar, mobile, status, email, deleted,
                      create_time, update_time, create_by, update_by)
values (1, 'root', 'root', 0, '$2a$10$H0948esWlZjhDC0v0AxsjOMZ/oI0am1Qg3HikbNoWxVEIR1.0r1xS', null,
        '', '18838027307', 1,
        '1066365803@qq.com', 0, null, null, null, null),
       (2, 'admin', 'admin', 1, '$2a$10$H0948esWlZjhDC0v0AxsjOMZ/oI0am1Qg3HikbNoWxVEIR1.0r1xS', 1,
        '', '18537958917', 1,
        'w1066365803@163.com', 0, '2019-10-10 13:41:22', '2024-11-13 14:40:05', null, null),
       (3, 'test', '测试用户', 1, '$2a$10$H0948esWlZjhDC0v0AxsjOMZ/oI0am1Qg3HikbNoWxVEIR1.0r1xS', 3,
        '', '19122145917', 1,
        'w1066365803@icloud.com', 0, '2021-06-05 01:31:29', '2021-06-05 01:31:29', null, null),
       (4, 'dept_admin', '部门管理员', 1, '$2a$10$H0948esWlZjhDC0v0AxsjOMZ/oI0am1Qg3HikbNoWxVEIR1.0r1xS', 2,
        '', '13800000004', 1,
        'dept_admin@dehaze.com', 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2),
       (5, 'user', '普通用户', 1, '$2a$10$H0948esWlZjhDC0v0AxsjOMZ/oI0am1Qg3HikbNoWxVEIR1.0r1xS', 2,
        '', '13800000005', 1,
        'user@dehaze.com', 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2),
       (6, 'vip1', 'VIP1用户', 1, '$2a$10$H0948esWlZjhDC0v0AxsjOMZ/oI0am1Qg3HikbNoWxVEIR1.0r1xS', 2,
        '', '13800000006', 1,
        'vip1@dehaze.com', 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2),
       (7, 'vip2', 'VIP2用户', 2, '$2a$10$H0948esWlZjhDC0v0AxsjOMZ/oI0am1Qg3HikbNoWxVEIR1.0r1xS', 2,
        '', '13800000007', 1,
        'vip2@dehaze.com', 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2),
       (8, 'svip', 'SVIP用户', 1, '$2a$10$H0948esWlZjhDC0v0AxsjOMZ/oI0am1Qg3HikbNoWxVEIR1.0r1xS', 2,
        '', '13800000008', 1,
        'svip@dehaze.com', 0, '2024-06-08 19:05:51', '2024-06-08 19:05:51', 2, 2);

SET FOREIGN_KEY_CHECKS = 1;
