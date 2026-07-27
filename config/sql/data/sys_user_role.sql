SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_user_role (user_id, role_id)
values (1, 1),
       (2, 2),
       (3, 3);

SET FOREIGN_KEY_CHECKS = 1;