SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

INSERT INTO sys_api_key (`id`,`user_id`,`name`,`key_prefix`,`key_hash`,`status`,`expires_at`,`last_used_at`,`create_time`,`update_time`,`create_by`,`update_by`) 
VALUES ('1','2','apifox','dhak_vhUN','eb31363e6da489f245d8fa0be286175ce215a6e5c5c1423b21387854a172ac4f',1,NULL,'2026-07-24 08:42:36','2026-07-24 08:25:49','2026-07-24 08:25:49','2','2');

insert into sys_api_key (user_id, name, key_prefix, key_hash, status, expires_at, create_time, create_by)
values (2, 'M2M服务间调用', 'dhak_m2m', 'ee3b5be0ae739e8a7e09883575d3742e4d19042fce82f8a48aae7f76116c8463', 1, null, now(), 2);

SET FOREIGN_KEY_CHECKS = 1;
