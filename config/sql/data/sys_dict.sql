SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_dict (id, type_code, name, value, sort, status, defaulted, remark, create_time, update_time)
values (1, 'gender', '男', '1', 1, 1, 0, null, '2019-05-05 13:07:52', '2022-06-12 23:20:39'),
       (2, 'gender', '女', '2', 2, 1, 0, null, '2019-04-19 11:33:00', '2019-07-02 14:23:05'),
       (3, 'gender', '未知', '0', 1, 1, 0, null, '2020-10-17 08:09:31', '2020-10-17 08:09:31');

SET FOREIGN_KEY_CHECKS = 1;