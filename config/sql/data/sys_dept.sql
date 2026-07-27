SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_dept (id, name, parent_id, tree_path, sort, status, deleted, create_time, update_time, create_by,
                      update_by)
values (1, '重庆邮电大学', 0, '0', 1, 1, 0, null, '2024-11-13 14:39:21', 1, 2),
       (2, '软件工程学院', 1, '0,1', 1, 1, 0, null, '2024-11-13 14:39:32', 2, 2),
       (3, '计算机学院', 1, '0,1', 1, 1, 0, null, '2024-11-13 14:39:42', 2, 2);

SET FOREIGN_KEY_CHECKS = 1;