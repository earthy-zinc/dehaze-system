SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_role (id, name, code, sort, status, data_scope, deleted, create_time, update_time)
values (1, '超级管理员', 'ROOT', 1, 1, 0, 0, '2021-05-21 14:56:51', '2018-12-23 16:00:00'),
       (2, '系统管理员', 'ADMIN', 2, 1, 0, 0, '2021-03-25 12:39:54', null),
       (3, '访问游客', 'GUEST', 3, 1, 2, 0, '2021-05-26 15:49:05', '2019-05-05 16:00:00'),
       (4, '部门管理员', 'DEPT_ADMIN', 4, 1, 1, 0, '2024-06-08 19:05:51', null),
       (5, '普通用户', 'USER', 5, 1, 3, 0, '2024-06-08 19:05:51', null);

SET FOREIGN_KEY_CHECKS = 1;
