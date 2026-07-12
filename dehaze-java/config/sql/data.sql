use dehaze;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Records of sys_dept
-- ----------------------------
insert into sys_dept (id, name, parent_id, tree_path, sort, status, deleted, create_time, update_time, create_by,
                      update_by)
values (1, '重庆邮电大学', 0, '0', 1, 1, 0, null, '2024-11-13 14:39:21', 1, 2),
       (2, '软件工程学院', 1, '0,1', 1, 1, 0, null, '2024-11-13 14:39:32', 2, 2),
       (3, '计算机学院', 1, '0,1', 1, 1, 0, null, '2024-11-13 14:39:42', 2, 2);

-- ----------------------------
-- Records of sys_dict
-- ----------------------------
insert into sys_dict (id, type_code, name, value, sort, status, defaulted, remark, create_time, update_time)
values (1, 'gender', '男', '1', 1, 1, 0, null, '2019-05-05 13:07:52', '2022-06-12 23:20:39'),
       (2, 'gender', '女', '2', 2, 1, 0, null, '2019-04-19 11:33:00', '2019-07-02 14:23:05'),
       (3, 'gender', '未知', '0', 1, 1, 0, null, '2020-10-17 08:09:31', '2020-10-17 08:09:31');

-- ----------------------------
-- Records of sys_dict_type
-- ----------------------------
INSERT INTO `sys_dict_type`
VALUES (1, '性别', 'gender', 1, NULL, '2019-12-06 19:03:32', '2022-06-12 16:21:28');

-- ----------------------------
-- Records of sys_menu
-- ----------------------------
insert into sys_menu (id, parent_id, tree_path, name, type, path, component, perm, visible, sort, icon, redirect,
                      create_time, update_time, always_show, keep_alive)
values (1, 0, '0', '系统管理', 2, '/system', 'Layout', null, 1, 1, 'system', '/system/user', '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, null),
       (2, 1, '0,1', '用户管理', 1, 'user', 'system/user/index', null, 1, 1, 'user', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (3, 1, '0,1', '角色管理', 1, 'role', 'system/role/index', null, 1, 2, 'role', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (4, 1, '0,1', '菜单管理', 1, 'menu', 'system/menu/index', null, 1, 3, 'menu', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (5, 1, '0,1', '部门管理', 1, 'dept', 'system/dept/index', null, 1, 4, 'tree', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (6, 1, '0,1', '字典管理', 1, 'dict', 'system/dict/index', null, 1, 5, 'dict', null, '2021-08-28 09:12:21',
        '2021-08-28 09:12:21', null, 1),
       (31, 2, '0,1,2', '用户新增', 4, '', null, 'sys:user:add', 1, 1, '', '', '2022-10-23 11:04:08',
        '2022-10-23 11:04:11', null, null),
       (32, 2, '0,1,2', '用户编辑', 4, '', null, 'sys:user:edit', 1, 2, '', '', '2022-10-23 11:04:08',
        '2022-10-23 11:04:11', null, null),
       (33, 2, '0,1,2', '用户删除', 4, '', null, 'sys:user:delete', 1, 3, '', '', '2022-10-23 11:04:08',
        '2022-10-23 11:04:11', null, null),
       (70, 3, '0,1,3', '角色新增', 4, '', null, 'sys:role:add', 1, 1, '', null, '2023-05-20 23:39:09',
        '2023-05-20 23:39:09', null, null),
       (71, 3, '0,1,3', '角色编辑', 4, '', null, 'sys:role:edit', 1, 2, '', null, '2023-05-20 23:40:31',
        '2023-05-20 23:40:31', null, null),
       (72, 3, '0,1,3', '角色删除', 4, '', null, 'sys:role:delete', 1, 3, '', null, '2023-05-20 23:41:08',
        '2023-05-20 23:41:08', null, null),
       (73, 4, '0,1,4', '菜单新增', 4, '', null, 'sys:menu:add', 1, 1, '', null, '2023-05-20 23:41:35',
        '2023-05-20 23:41:35', null, null),
       (74, 4, '0,1,4', '菜单编辑', 4, '', null, 'sys:menu:edit', 1, 3, '', null, '2023-05-20 23:41:58',
        '2023-05-20 23:41:58', null, null),
       (75, 4, '0,1,4', '菜单删除', 4, '', null, 'sys:menu:delete', 1, 3, '', null, '2023-05-20 23:44:18',
        '2023-05-20 23:44:18', null, null),
       (76, 5, '0,1,5', '部门新增', 4, '', null, 'sys:dept:add', 1, 1, '', null, '2023-05-20 23:45:00',
        '2023-05-20 23:45:00', null, null),
       (77, 5, '0,1,5', '部门编辑', 4, '', null, 'sys:dept:edit', 1, 2, '', null, '2023-05-20 23:46:16',
        '2023-05-20 23:46:16', null, null),
       (78, 5, '0,1,5', '部门删除', 4, '', null, 'sys:dept:delete', 1, 3, '', null, '2023-05-20 23:46:36',
        '2023-05-20 23:46:36', null, null),
       (79, 6, '0,1,6', '字典类型新增', 4, '', null, 'sys:dict_type:add', 1, 1, '', null, '2023-05-21 00:16:06',
        '2023-05-21 00:16:06', null, null),
       (81, 6, '0,1,6', '字典类型编辑', 4, '', null, 'sys:dict_type:edit', 1, 2, '', null, '2023-05-21 00:27:37',
        '2023-05-21 00:27:37', null, null),
       (84, 6, '0,1,6', '字典类型删除', 4, '', null, 'sys:dict_type:delete', 1, 3, '', null, '2023-05-21 00:29:39',
        '2023-05-21 00:29:39', null, null),
       (85, 6, '0,1,6', '字典数据新增', 4, '', null, 'sys:dict:add', 1, 4, '', null, '2023-05-21 00:46:56',
        '2023-05-21 00:47:06', null, null),
       (86, 6, '0,1,6', '字典数据编辑', 4, '', null, 'sys:dict:edit', 1, 5, '', null, '2023-05-21 00:47:36',
        '2023-05-21 00:47:36', null, null),
       (87, 6, '0,1,6', '字典数据删除', 4, '', null, 'sys:dict:delete', 1, 6, '', null, '2023-05-21 00:48:10',
        '2023-05-21 00:48:20', null, null),
       (88, 2, '0,1,2', '重置密码', 4, '', null, 'sys:user:password:reset', 1, 4, '', null, '2023-05-21 00:49:18',
        '2024-04-28 00:38:22', null, null),
       (105, 2, '0,1,2', '用户查询', 4, '', null, 'sys:user:query', 1, 0, '', null, '2024-04-28 00:37:34',
        '2024-04-28 00:37:34', 0, 0),
       (106, 2, '0,1,2', '用户导入', 4, '', null, 'sys:user:import', 1, 5, '', null, '2024-04-28 00:39:15',
        '2024-04-28 00:39:15', null, null),
       (107, 2, '0,1,2', '用户导出', 4, '', null, 'sys:user:export', 1, 6, '', null, '2024-04-28 00:39:43',
        '2024-04-28 00:39:43', null, null),
       (200, 0, '0', '算法管理', 2, '/algorithm', 'Layout', null, 1, 2, 'algorithm', '/algorithm/list', '2024-06-08 19:05:51',
        '2024-06-08 19:05:51', null, null),
       (201, 200, '0,200', '算法列表', 1, 'list', 'algorithm/list/index', null, 1, 1, 'list', null, '2024-06-08 19:05:51',
        '2024-06-08 19:05:51', null, 1),
       (202, 201, '0,200,201', '算法新增', 4, '', null, 'sys:algorithm:add', 1, 1, '', null, '2024-06-08 19:05:51',
        '2024-06-08 19:05:51', null, null),
       (203, 201, '0,200,201', '算法编辑', 4, '', null, 'sys:algorithm:edit', 1, 2, '', null, '2024-06-08 19:05:51',
        '2024-06-08 19:05:51', null, null),
       (204, 201, '0,200,201', '算法删除', 4, '', null, 'sys:algorithm:delete', 1, 3, '', null, '2024-06-08 19:05:51',
        '2024-06-08 19:05:51', null, null),
       (205, 201, '0,200,201', '算法审核', 4, '', null, 'sys:algorithm:audit', 1, 4, '', null, '2024-06-08 19:05:51',
        '2024-06-08 19:05:51', null, null),
       (206, 201, '0,200,201', '版本管理', 4, '', null, 'sys:algorithm:version', 1, 5, '', null, '2024-06-08 19:05:51',
        '2024-06-08 19:05:51', null, null),
       (207, 201, '0,200,201', '算法导出', 4, '', null, 'sys:algorithm:export', 1, 6, '', null, '2024-06-08 19:05:51',
        '2024-06-08 19:05:51', null, null),
       (208, 201, '0,200,201', '算法导入', 4, '', null, 'sys:algorithm:import', 1, 7, '', null, '2024-06-08 19:05:51',
        '2024-06-08 19:05:51', null, null);

-- ----------------------------
-- Records of sys_role
-- ----------------------------
insert into sys_role (id, name, code, sort, status, data_scope, deleted, create_time, update_time)
values (1, '超级管理员', 'ROOT', 1, 1, 0, 0, '2021-05-21 14:56:51', '2018-12-23 16:00:00'),
       (2, '系统管理员', 'ADMIN', 2, 1, 1, 0, '2021-03-25 12:39:54', null),
       (3, '访问游客', 'GUEST', 3, 1, 2, 0, '2021-05-26 15:49:05', '2019-05-05 16:00:00');

-- ----------------------------
-- Records of sys_role_menu
-- ----------------------------
insert into sys_role_menu (role_id, menu_id)
values (3, 1),
       (3, 2),
       (3, 31),
       (3, 32),
       (3, 33),
       (3, 88),
       (3, 3),
       (3, 70),
       (3, 71),
       (3, 72),
       (2, 1),
       (2, 2),
       (2, 105),
       (2, 31),
       (2, 32),
       (2, 33),
       (2, 88),
       (2, 106),
       (2, 107),
       (2, 3),
       (2, 70),
       (2, 71),
       (2, 72),
       (2, 4),
       (2, 73),
       (2, 75),
       (2, 74),
       (2, 5),
       (2, 76),
       (2, 77),
       (2, 78),
       (2, 6),
       (2, 79),
       (2, 81),
       (2, 84),
       (2, 85),
       (2, 86),
       (2, 87),
       (2, 40),
       (2, 41),
       (2, 26),
       (2, 102),
       (2, 30),
       (2, 20),
       (2, 21),
       (2, 22),
       (2, 23),
       (2, 24),
       (2, 36),
       (2, 37),
       (2, 38),
       (2, 39),
       (2, 95),
       (2, 89),
       (2, 97),
       (2, 90),
       (2, 91),
       (2, 200),
       (2, 201),
       (2, 202),
       (2, 203),
       (2, 204),
       (2, 205),
       (2, 206),
       (2, 207),
       (2, 208);

-- ----------------------------
-- Records of sys_user
-- ----------------------------
insert into sys_user (id, username, nickname, gender, password, dept_id, avatar, mobile, status, email, deleted,
                      create_time, update_time)
values (1, 'root', '有来技术', 0, '$2a$10$xVWsNOhHrCxh5UbpCE7/HuJ.PAOKcYAqRxD2CO2nVnJS.IAXkr5aq', null,
        '', '17621590365', 1,
        'youlaitech@163.com', 0, null, null),
       (2, 'admin', 'admin', 1, '$2a$10$xVWsNOhHrCxh5UbpCE7/HuJ.PAOKcYAqRxD2CO2nVnJS.IAXkr5aq', 1,
        'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD/4QAuRXhpZgAATU0AKgAAAAgAAkAAAAMAAAABAAAAAEABAAEAAAABAAAAAAAAAAD/2wBDAAoHBwkHBgoJCAkLCwoMDxkQDw4ODx4WFxIZJCAmJSMgIyIoLTkwKCo2KyIjMkQyNjs9QEBAJjBGS0U+Sjk/QD3/2wBDAQsLCw8NDx0QEB09KSMpPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT3/wAARCABkAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCxD4Xs4yCxkcjtnGa0obKCEAInPuSasVDeXC2dtJKwztHA9T2FDk92xRjd2Rm6vMhlW2THynzJGznb6U7SodkbTkHMvI9gKz4onuZxG53SSnfKw7D0/pW6AAoAwABgCuKD55Ob22R6KiqcVFb9Re1J7UtFakB+FFFB6UAFY13F9mvyAP3dxlhnsw5I/Hg/nWkb60jcq1zAr/3TIoP86h1GL7ZZFrdlaRGEkZByCw7Z9xkfjWdWHPFoum2mUAWhuYrlBueIk7R/EpGGH1xyPcV0EM0dzCssLBo2+YEf5/SuejkWWMSL91hkZqexkaG9EQcrFcE5A7OBwR6ZAP5CufC1nF+zYYmjzLmW6N75v8iiovs69yxPqWNFemeeSce1ZGsS75oYB91QZG/pWo8Z6qxRvzB/CsHUSwurjdt3kBRj8q58Q2oaddDbDJOav0J9ODiMyIoMsxzuPRV7Z/wqZ7GeTltQuVJ7RhEA/Q/zqzDH5UMaAfdUCnmRU+8yr9TRGKUUkdUm27meljqEM0bJqbSRBvnSaJWJHswxzWiCOlV7m9gt4y7zRLt5wXC5qaNxIgdSCrAEEdxVEt3HVSvtJh1GRWuXnaNVx5IlKo3uQOtXfrVa8uWtnVvJmkTH/LNd3PvyMUAnbUdFp9pCgSG1gRB2Ea/4UptkHzQgROOjIMD8R0NZMvim3t544poZ1d2KqpTO4/gTVqHxBp8zhDKUkborqw/mOaHFrdCU03oyvcQm1vCuAI58uuOgbuv49fzqOYsIi6Z8xCHX6qc/0rYurZby2MbMV6MrDqrDoRWPG7ESLMoEkTFXA6ZHp7EYP4159em4TU0dMHzRszo45d8SOhG1lDD8RRVbQ5PN0Oyfd1iX+VFekpHA4Fvr1rn75y/iSOIdCoJ/DmtsvJFGXcx4QFieRWDag3GqxXMn+sbe2PQdh+VRWa0T6seGi7t9ka1zDJMgVLiSH1KAZP4kHFY994aa/wAhr6dVLA5Lsxx6dQP0reoqk2tjVxT3M+z0Szs8MsSvIOjOAdp9hjitAelHeo5ZWjO0RPIe+zHH5mhtvcaSSsiTOaKT5tnA59DTITKUPnoqyBiPkOQR2NIBzxRyfejVj6kA0JGiD5VVfoMU6jNAgrG1GNhfzBeDLEjfjkqf0xWzWPq7ql4Ds3P5B2tnG3DE5/SsqkOeLSNIOzLOmTeRYJEifIhZV+gY0Vc0WIPpMD4Pzgv+ZJ/rRQk7GbkrlXWLjOLNDyxDSY7L6fjVS14v4j6qwqKNCinP3mOSfWnxnZdQN/tY/OuR1vaVl2N4U+SFuptUUnNHWvQMRagur2GzAMz4LfdUDJP0FNv71bKHIw0rcKp7+59q5d7hppGkAaV2PzOTgH6e304oOqhh3U1eiOifWbaMAhmbd1AH3frVuG5juU3QuGHt1H1rkd8x/wCWaj6t/wDWp0VzPbOJApBHdDn8x3oN54RW0ep2GKKpadqcV8gAZVlH3lz+oq7Qee4tOzDH5Vl3QEmshXUtH9mK/id39Aa08VRvNqSXMpUF0ttyn0+8KQLcv6WR/ZVp5bfJ5KY/75FFYdnrp0qzis2hDmIbd3t2/IYH4UUvbIz9k+wvWmucGNv7rA06myAlCq9f8K8ikm5pLe53TaUW3sbgNGc0y3kEsEbg5ytP69K9k5TJv9FkupmlE5bd1V+mPT6VANFusdYuOnJreooNo4icFZMwk0S5P3miX8SasxaDEB++ld29FO0VqUUBLEVGtyvbWFtZktDEquerYyT+NWMUUc0GLberDnvWPfFpdQmQPiMIiMoH3urYz+IrXJAGScDuTWGkhleScjCzuWX6dB+gFY11LkbiOEkpJPqOKI5yygk98UUUV5N5dzssgp0aMX3AcKOauppmfvSn8BUyWcds4xltwwxJr18NgZwmpStoeXiMZCUHGPUpaVL5Rkt2PAYlfpWkOBVSWxCOXUnnp7GpIZmJ8uRTu/vDvXTUg0x0ailFa6k5+9S59ail80ENGFYD+E96hF6o4mikU/TIrM2sWvSlqAX0H/PQD6g0G9g/56rQGpPRVf7Yh4RXf6CnySsEAVMyN91T0HuaA1KmsSN9imiiYhinzMP4QeMfUk/zqtKNkYVV4UjgdgKs3sP+hiJSZJWlRmPdsMCfw4qf7L5dnMz4MjIc+3tW8KfNBrucWIqWnG3QzaKma2mUkFeaK8d4Sr/Kz0vrNPujaqOb/VmiivpD54eOUH0qtOAhyvB9qKKzqbGlLcWJ2PBPFPNFFcMtz1o7IBGhxlF/KkMaL0RfyoopFCSEog28VEpMg+Yk/jRRWlM565ajjRPuqBUdz/x7N7gfzFFFdx5nUmNFFFMD/9k=',
        '18537958917', 1,
        'w1066365803@163.com', 0, '2019-10-10 13:41:22', '2024-11-13 14:40:05'),
       (3, 'test', '测试小用户', 1, '$2a$10$xVWsNOhHrCxh5UbpCE7/HuJ.PAOKcYAqRxD2CO2nVnJS.IAXkr5aq', 3,
        '', '17621210366', 1,
        'youlaitech@163.com', 0, '2021-06-05 01:31:29', '2021-06-05 01:31:29');

-- ----------------------------
-- Records of sys_user_role
-- ----------------------------
insert into sys_user_role (user_id, role_id)
values (1, 1),
       (2, 2),
       (3, 3);

insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (1, 0, '图像去雾', 'DENSE-HAZE', null,
        'DENSE-HAZE 引入了一种新的去雾数据集，以浓密均匀的雾霾场景为特征。该数据集包含 55对真实的浓雾图像和各种室外场景的相应无雾图像。这些朦胧图像是通过专业雾霾机器生成的真实雾霾记录的。生成的浓雾图像几乎难以辨别图像中原来存在的物体，与常规数据集相比去雾难度非常大。',
        'Dense-Haze', '234.74 MB', 1, 0, '2024-11-11 19:29:49', '2024-11-11 19:29:49', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (2, 0, '图像去雾', 'O-HAZE', null,
        'O-haze 数据集是由CVLab实验室在2016年发布的，主要用于评估和测试图像去雾算法的性能。该数据集包含了合成的有雾图像和相应的清晰图像对，这些图像都是基于真实的户外场景生成的。包含45对户外场景的有雾和清晰图像',
        'O-HAZE', '547.85 MB', 1, 0, '2024-11-11 19:36:32', '2024-11-11 19:36:32', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (3, 0, '图像去雾', 'I-HAZE', null,
        'I-haze 数据集也是由CVLab实验室在2016年发布的，与O-haze数据集类似，它主要用于评估和测试图像去雾算法的性能。不过，I-haze数据集的特点在于其图像更接近实际的室内场景。包含35对有雾和相应的无雾室内图像。',
        'I-HAZE', '312.99 MB', 1, 0, '2024-11-11 19:37:12', '2024-11-11 19:37:12', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (4, 0, '图像去雾', 'NH-HAZE', null,
        'NH-HAZE数据集旨在解决图像去雾领域中的一个重要问题：缺乏真实世界的非均匀雾度图像作为参考数据。许多现实场景中的雾并不均匀分布，因此 NH-HAZE 提供了一组真实的非均匀雾图像和相应的无雾图像对。NH-HAZE 数据集中的非均匀雾度是通过专业的雾发生器模拟真实雾天条件而引入的。是一个更具挑战性和现实性的去雾数据集。',
        'NTIRE', '1.06 GB', 1, 0, '2024-11-11 19:39:24', '2024-11-11 19:39:24', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (5, 4, '图像去雾', 'NH-HAZE-20', null,
        '在2020年，NH-HAZE数据集被用于CVPR NTIRE（New Trends in Image Restoration and Enhancement）研讨会下的图像去雾在线挑战赛中1。这是首个包含55对外部拍摄的真实有雾和对应的无雾图像的数据集，这些图像是使用专业雾生成器在高保真的条件下拍摄的，以模拟真实的非均匀雾霾环境。NH-HAZE 2020的数据集为研究人员提供了评估去雾算法性能的机会，并且由于其现实性，对于开发更加鲁棒的去雾解决方案具有重要意义',
        'NH-HAZE-2020', '316.96 MB', 1, 0, '2024-11-11 19:39:51', '2024-11-11 19:39:51', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (6, 4, '图像去雾', 'NH-HAZE-21', null,
        '到了2021年，NTIRE挑战赛继续进行，这次的非均匀去雾挑战基于扩展后的NH-HAZE数据集，增加了额外35对真实户外拍摄的无雾和非均匀有雾图像。这个扩大的数据集被称为NH-Haze2，它进一步增强了数据集的多样性和复杂度，为参与者提供了更广泛的测试平台来验证他们的算法。此外，在这次挑战中还加入了其他小规模的真实世界数据集如DENSE-HAZE等，用以对比不同方法的效果。',
        'NH-HAZE-2021', '151.36 MB', 1, 0, '2024-11-11 19:40:08', '2024-11-11 19:40:08', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (7, 4, '图像去雾', 'NH-HAZE-23', null,
        '至2023年，NTIRE举办了一次高清非均质去雾挑战赛，这次比赛采用了名为HD-NH-HAZE的新数据集。HD-NH-HAZE包含了50对高清分辨率的户外图像，其中一半是带有非均匀雾霾的图像，另一半则是同一场景的无雾霾图像。这个数据集的引入标志着单张图像去雾领域的一个重要进展，因为它不仅提高了图像的质量标准，而且也推动了去雾技术向着处理更高分辨率图像的方向发展。参赛者们提出的方法在此数据集上进行了客观评估，以便更好地衡量它们在处理实际场景中的表现',
        'NH-HAZE-2023', '618.19 MB', 1, 0, '2024-11-11 19:40:19', '2024-11-11 19:40:19', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (8, 0, '图像去雾', 'RESIDE', null,
        'RESIDE（Realistic Synthetic and Indoor-Outdoor DEhazing）数据集是由北京大学和微软亚洲研究院在2017年联合发布的，旨在为图像去雾研究提供一个大规模、多样化的基准数据集。RESIDE 数据集不仅包含合成的有雾图像和对应的清晰图像，还包含了一些真实世界中的有雾图像，使其成为图像去雾领域最全面的数据集之一。',
        'RESIDE', '19.01 GB', 1, 0, '2024-11-11 19:41:55', '2024-11-11 19:41:55', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (9, 8, '图像去雾', 'ITS', null,
        '室内训练集(ITS) 是RESIDE数据集中的一部分，主要用于算法的训练阶段。ITS包含13,990张由清晰图像生成的合成模糊图像，这些清晰图像是从现有的室内深度数据集NYU2和米德尔伯里立体数据库中选取的1,399张图像。对于每一张清晰图像，通过在不同参数设置下（例如大气光A和散射系数β）生成10张模糊图像。这些参数的设定使得生成的模糊图像能够模拟多种不同的雾霾情况。具体来说，大气光A的值在[0.7, 1.0]之间均匀随机选择，而β则在[0.6, 1.8]之间均匀随机选择。最终，这13,990张图像中的13,000张被用于训练，剩下的990张作为验证集',
        'RESIDE/ITS', '4.74 GB', 1, 0, '2024-11-11 19:42:34', '2024-11-11 19:42:34', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (10, 8, '图像去雾', 'OTS', null,
        '室外训练集(OTS) 是RESIDE-beta部分的数据集，旨在提高对室外环境下的去雾性能。OTS使用了2061张来自北京实时天气的真实室外图像，通过估计每张图像的深度信息后，根据一系列指定的大气散射系数β值（如0.04, 0.06, 0.08, 0.1, 0.15, 0.95, 1等）来合成模糊图像。最终，总共合成了72,135张户外模糊图像。这套新的图像被称为户外训练集（OTS），由成对的干净的户外图像和生成的模糊图像组成，以供算法训练使用',
        'RESIDE/OTS', '12.86 GB', 1, 0, '2024-11-11 19:42:54', '2024-11-11 19:42:54', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (14, 0, '图像去雾', 'RESIDE-6k', null,
        '虽然没有直接提到名为RESIDE-6k的数据集，但我们可以假设这可能是一个包含大约6000张图像的RESIDE数据集的一个子集。如果这是对RESIDE数据集的特定版本，则它可能专注于一个特定的场景（室内或室外）或者用于特定目的（比如训练或测试）。然而，由于没有具体的信息，我们无法确定其确切组成。通常，这样的数据集会包含成对的清晰和模糊图像，以便于模型学习如何从模糊图像恢复清晰图像。',
        'RESIDE-6k', '1.52 GB', 1, 0, '2024-11-12 22:22:42', '2024-11-12 22:22:42', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (15, 14, '图像去雾', 'RESIDE-6k-train', null,
        'RESIDE-6k 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RESIDE-6k/train', '1,021.78 MB',
        1, 0, '2024-11-12 22:23:05', '2024-11-12 22:23:05', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (16, 14, '图像去雾', 'RESIDE-6k-test', null,
        'RESIDE-6k  测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RESIDE-6k/test', '532.3 MB',
        1, 0, '2024-11-12 22:23:16', '2024-11-12 22:23:16', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (17, 0, '图像去雾', 'RESIDE-IN', null,
        '这个名称可能指的是RESIDE数据集中专注于室内场景的部分。结合RESIDE数据集的描述，我们可以合理推测RESIDE-IN可能主要包含了ITS（Indoor Training Set），即室内训练集。该集合包括了13,990个合成的模糊图像，这些图像是基于NYU2和米德尔伯里立体数据库中的1,399个清晰室内图像生成的1。此外，SOTS（Synthetic Objective Testing Set）中的部分室内图像也可能被包含在内，用于评估算法性能。',
        'RESIDE-IN', '8.74 GB', 1, 0, '2024-11-12 22:23:48', '2024-11-12 22:23:48', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (18, 17, '图像去雾', 'RESIDE-IN-train', null,
        'RESIDE-IN 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RESIDE-IN/train', '8.36 GB',
        1, 0, '2024-11-12 22:24:10', '2024-11-12 22:24:10', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (19, 17, '图像去雾', 'RESIDE-IN-test', null,
        'RESIDE-IN 测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RESIDE-IN/test', '392.11 MB',
        1, 0, '2024-11-12 22:25:09', '2024-11-12 22:25:09', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (20, 0, '图像去雾', 'RESIDE-OUT', null,
        '同样，RESIDE-OUT可能是指RESIDE数据集中专注于室外场景的部分。这意味着它可能主要由OTS（Outdoor Training Set）构成，该集合包括72,135张合成的户外模糊图像，这些图像是基于北京实时天气的真实室外图像生成的2。SOTS中的一部分室外图像也可能会被纳入其中，用于测试目的。',
        'RESIDE-OUT', '83.03 GB', 1, 0, '2024-11-12 22:26:09', '2024-11-12 22:26:09', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (21, 20, '图像去雾', 'RESIDE-OUT-train', null,
        'RESIDE-OUT 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RESIDE-OUT/train', '82.89 GB',
        1, 0, '2024-11-12 22:26:47', '2024-11-12 22:26:47', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (22, 20, '图像去雾', 'RESIDE-OUT-test', null,
        'RESIDE-OUT 测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RESIDE-OUT/test',
        '140.19 MB', 1, 0, '2024-11-12 22:27:08', '2024-11-12 22:27:08', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (23, 0, '图像去雾', 'RSHAZE', null,
        'REHAZE数据集是专为图像去雾研究设计的，旨在提供更真实的雾霾条件下的图像。它由苏黎世联邦理工大学等机构发布，包含有雾和无雾图像对，这些图像是在受控环境中使用专业设备拍摄的，以模拟不同的雾霾条件。不过，具体的细节（如图像数量、场景类型等）需要查阅原始论文或官方发布页面来获取准确信息。',
        'RSHAZE', '40.41 GB', 1, 0, '2024-11-12 22:28:28', '2024-11-12 22:28:28', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (24, 23, '图像去雾', 'RSHAZE-train', null,
        'REHAZE 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RSHAZE/train', '38.39 GB', 1,
        0, '2024-11-12 22:28:47', '2024-11-12 22:28:47', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (25, 23, '图像去雾', 'RSHAZE-test', null,
        'REHAZE 测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RSHAZE/test', '2.02 GB', 1,
        0, '2024-11-12 22:28:54', '2024-11-12 22:28:54', 2, 2);


insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (1, 0, '图像去雾', 'AECR-NET', null, 'AECR-Net/NH_train.pk', '35.1 MB', null, null, 'algorithm.AECRNet.run',
        'AECRNet 是一种深度学习模型，专门用于图像去雾任务。该模型由清华大学和微软亚洲研究院的研究人员在2019年提出，旨在解决传统去雾方法中存在的边缘模糊和细节丢失问题。AECRNet 通过引入对抗生成网络（GAN）和边缘保持机制，实现了高质量的去雾效果。',
        1, '2024-11-11 20:00:28', '2024-11-11 20:00:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (2, 0, '图像去雾', 'AODNet', null, 'AODNet/dehazer.pth', '8.41 KB', '1.72 KB', '109.12 MB',
        'algorithm.AODNet.run',
        'AODNet (All-in-One Dehazing Network) 是一种用于图像去雾的深度学习模型，由Yuan et al. 在2018年提出。传统的图像去雾方法通常依赖于大气散射模型以及一些先验知识，如暗通道先验等，这些方法虽然在某些情况下能够取得较好的效果，但是往往计算复杂度较高，且对于不同的环境条件适应性较差。AODNet旨在解决这些问题，提供一个更加高效和鲁棒的解决方案。',
        1, '2024-11-11 23:52:37', '2024-12-03 21:21:36', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (3, 0, '图像去雾', 'C2PNet', null, 'C2PNet/ITS.pkl', '35.98 MB', '6.84 MB', '429.3 GB', 'algorithm.C2PNet.run',
        'C2PNet（Cycle-to-Point Network）是一种用于图像去雾的深度学习模型。该模型设计的目的在于解决传统去雾算法中存在的问题，如色彩失真、细节损失等，并且能够有效地处理复杂多变的自然场景中的雾霾问题。',
        1, '2024-11-12 22:51:50', '2024-12-03 21:21:37', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (4, 3, '图像去雾', '室内去雾（ITS）', null, 'C2PNet/ITS.pkl', '35.98 MB', '6.84 MB', '429.3 GB',
        'algorithm.C2PNet.run',
        'C2PNet（Cycle-to-Point Network）是一种用于图像去雾的深度学习模型。该模型设计的目的在于解决传统去雾算法中存在的问题，如色彩失真、细节损失等，并且能够有效地处理复杂多变的自然场景中的雾霾问题。',
        1, '2024-11-12 22:52:15', '2024-12-03 21:21:37', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (5, 3, '图像去雾', '室外去雾（OTS）', null, 'C2PNet/OTS.pkl', '39.51 MB', '6.84 MB', '429.3 GB',
        'algorithm.C2PNet.run',
        'C2PNet（Cycle-to-Point Network）是一种用于图像去雾的深度学习模型。该模型设计的目的在于解决传统去雾算法中存在的问题，如色彩失真、细节损失等，并且能够有效地处理复杂多变的自然场景中的雾霾问题。',
        1, '2024-11-12 22:52:25', '2024-12-03 21:21:38', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (6, 0, '图像去雾', 'CMFNet', null, 'CMFNet', null, null, null, 'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        1, '2024-11-13 11:30:00', '2024-11-13 11:30:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (7, 6, '图像去雾', '去雾模型', null, 'CMFNet/dehaze_I_OHaze_CMFNet.pth', '197.53 MB', '16.44 MB', '595.07 GB',
        'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        1, '2024-11-13 11:30:16', '2024-12-03 21:21:41', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (8, 6, '图像去模糊', '去模糊模型', null, 'CMFNet/deblur_GoPro_CMFNet.pth', '197.53 MB', '16.44 MB', '595.07 GB',
        'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        1, '2024-11-13 11:30:26', '2024-12-03 21:21:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (9, 6, '图像去雨', '去雨模型', null, 'CMFNet/deraindrop_DeRainDrop_CMFNet.pth', '197.53 MB', '16.44 MB',
        '595.07 GB', 'algorithm.CMFNet.run',
        'CMFNet（Compound Multi-branch Feature Fusion Network）是一种基于深度学习的图像恢复模型，旨在解决图像去雾、去模糊等多个图像恢复任务。该模型的设计灵感来源于人类视觉系统，特别是视网膜神经节细胞（RGCs），它由三种不同类型的细胞组成：P-cells、K-cells和M-cells，每种细胞对外部刺激有着不同的敏感度。CMFNet模仿这种生物机制，构建了一个多分支的网络架构，以适应不同类型图像退化的处理需求',
        1, '2024-11-13 11:30:33', '2024-12-03 21:21:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (10, 0, '图像去雾', 'D4', null, 'D4/weights_reconstruct.pth', '88.16 MB', null, null, 'algorithm.D4.run', '', 1,
        '2024-11-13 12:39:45', '2024-11-13 12:39:45', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (11, 0, '图像去雾', 'DaclipUir', null, 'daclip-uir/daclip_ViT-B-32.pt', '1.62 GB', null, null,
        'algorithm.DaclipUir.run',
        'DaclipUir 是一种先进的图像去雾模型，它结合了深度学习与物理模型，通过优化图像的对比度和色彩，有效去除雾霾，提高图像的清晰度。该模型特别注重保留图像的细节和自然度，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:39:54', '2024-11-13 12:39:54', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (12, 0, '图像去雾', 'DCPDN', null, 'DCPDN/netG_epoch_8.pth', '255.55 MB', null, null, 'algorithm.DCPDN.run',
        'DCPDN 是一种基于深度学习的图像去雾方法，通过大气散射模型和密集连接的编码器-解码器结构，估计透射率图并进行去雾。该模型利用多级金字塔池化模块，提高了透射率估计的准确性，从而改善了去雾效果',
        1, '2024-11-13 12:40:03', '2024-11-13 12:40:03', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (13, 0, '图像去雾', 'DCP', null, '/', null, null, null, 'algorithm.DCP.run',
        'DCP 是由何凯明等人在2009年提出的经典去雾算法，基于暗原色先验理论。该算法假设无雾图像的局部区域中至少有一个颜色通道的亮度值非常低。通过估计大气光和透射率，DCP 能够有效地去除图像中的雾霾，恢复图像的清晰度',
        1, '2024-11-13 12:40:13', '2024-11-13 12:40:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (14, 0, '图像去雾', 'DEANet', null, 'DEA-Net', null, null, null, 'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        1, '2024-11-13 12:40:20', '2024-11-13 12:40:20', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (15, 14, '图像去雾', 'HAZE4k模型', null, 'DEA-Net/HAZE4K/PSNR3426_SSIM9885.pth', '14 MB', '3.48 MB', '31.7 GB',
        'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        1, '2024-11-13 12:40:55', '2024-12-03 21:22:04', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (16, 14, '图像去雾', 'ITS模型', null, 'DEA-Net/ITS/PSNR4131_SSIM9945.pth', '14 MB', '3.48 MB', '31.7 GB',
        'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        1, '2024-11-13 12:41:02', '2024-12-03 21:22:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (17, 14, '图像去雾', 'OTS模型', null, 'DEA-Net/OTS/PSNR3659_SSIM9897.pth', '14 MB', '3.48 MB', '31.7 GB',
        'algorithm.DEANet.run',
        ' DEANet 是一种用于单幅图像去雾的深度学习网络，结合了细节增强卷积（DEConv）和内容引导注意力（CGA）机制。DEConv 通过并行的普通卷积和差异卷积增强特征表示，CGA 则通过生成粗略的空间注意力图并进行细化，提高模型对图像细节的保留能力',
        1, '2024-11-13 12:41:09', '2024-12-03 21:22:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (18, 0, '图像去雾', 'Dehamer', null, 'Dehamer', null, null, null, 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:41:26', '2024-11-13 12:41:26', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (19, 18, '图像去雾', 'dense-haze模型', null, 'Dehamer/dense/PSNR1662_SSIM05602.pt', '511.68 MB', '28.08 MB',
        '55.58 GB', 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:41:56', '2024-12-03 21:22:10', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (20, 18, '图像去雾', 'indoor', null, 'Dehamer/indoor/PSNR3663_ssim09881.pt', '511.68 MB', '28.08 MB', '55.91 GB',
        'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:42:02', '2024-12-03 21:22:14', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (21, 18, '图像去雾', 'NH-HAZE-模型', null, 'Dehamer/NH/PSNR2066_SSIM06844.pt', '511.68 MB', '28.08 MB',
        '56.25 GB', 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:42:10', '2024-12-03 21:22:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (22, 18, '图像去雾', 'outdoor', null, 'Dehamer/outdoor/PSNR3518_SSIM09860.pt', '511.68 MB', '28.08 MB',
        '56.59 GB', 'algorithm.Dehamer.run',
        'Dehamer 是一种高效的图像去雾模型，通过多尺度特征融合技术和深度卷积网络，增强图像的结构信息。该模型能够在保持图像细节的同时，实现高质量的去雾效果，适用于多种场景下的图像去雾任务',
        1, '2024-11-13 12:42:17', '2024-12-03 21:22:23', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (23, 0, '图像去雾', 'DehazeFormer', null, 'DehazeFormer', null, null, null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:42:35', '2024-11-13 12:42:35', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (24, 23, '图像去雾', 'indoor', null, 'DehazeFormer/indoor', null, null, null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:42:57', '2024-11-13 12:42:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (25, 24, '图像去雾', 'indoor-b', null, 'DehazeFormer/indoor/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:45:52', '2024-12-03 21:22:24', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (26, 24, '图像去雾', 'indoor-d', null, 'DehazeFormer/indoor/dehazeformer-d.pth', '21.22 MB', '4.75 MB',
        '43.57 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:45:59', '2024-12-03 21:22:26', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (27, 24, '图像去雾', 'indoor-l', null, 'DehazeFormer/indoor/dehazeformer-l.pth', '98.22 MB', '24.27 MB',
        '256.26 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:05', '2024-12-03 21:22:27', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (28, 24, '图像去雾', 'indoor-m', null, 'DehazeFormer/indoor/dehazeformer-m.pth', '18.51 MB', '4.42 MB',
        '43.79 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:12', '2024-12-03 21:22:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (29, 24, '图像去雾', 'indoor-s', null, 'DehazeFormer/indoor/dehazeformer-s.pth', '5.46 MB', '1.22 MB',
        '11.23 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:19', '2024-12-03 21:22:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (30, 24, '图像去雾', 'indoor-t', null, 'DehazeFormer/indoor/dehazeformer-t.pth', '2.9 MB', '670.36 KB',
        '5.82 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:25', '2024-12-03 21:22:28', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (31, 24, '图像去雾', 'indoor-w', null, 'DehazeFormer/indoor/dehazeformer-w.pth', '38.06 MB', '9.23 MB',
        '83.69 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:33', '2024-12-03 21:22:29', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (32, 23, '图像去雾', 'outdoor', null, 'DehazeFormer/outdoor', null, null, null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:46:44', '2024-11-13 12:46:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (33, 32, '图像去雾', 'outdoor-b', null, 'DehazeFormer/outdoor/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:51:32', '2024-12-03 21:22:30', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (34, 32, '图像去雾', 'outdoor-m', null, 'DehazeFormer/outdoor/dehazeformer-m.pth', '18.51 MB', '4.42 MB',
        '43.79 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:51:38', '2024-12-03 21:22:31', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (35, 32, '图像去雾', 'outdoor-s', null, 'DehazeFormer/outdoor/dehazeformer-s.pth', '5.46 MB', '1.22 MB',
        '11.23 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:51:44', '2024-12-03 21:22:31', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (36, 32, '图像去雾', 'outdoor-t', null, 'DehazeFormer/outdoor/dehazeformer-t.pth', '2.9 MB', '670.36 KB',
        '5.82 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:51:50', '2024-12-03 21:22:31', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (37, 23, '图像去雾', 'reside6k', null, 'DehazeFormer/reside6k', null, null, null, 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:00', '2024-11-13 12:52:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (38, 37, '图像去雾', 'reside6k-b', null, 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:13', '2024-12-03 21:22:32', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (39, 37, '图像去雾', 'reside6k-b', null, 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:19', '2024-12-03 21:22:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (40, 37, '图像去雾', 'reside6k-b', null, 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:26', '2024-12-03 21:22:34', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (41, 37, '图像去雾', 'reside6k-b', null, 'DehazeFormer/reside6k/dehazeformer-b.pth', '10.71 MB', '2.4 MB',
        '22.01 GB', 'algorithm.DehazeFormer.run',
        'DehazeFormer 是一种基于 Transformer 架构的图像去雾模型，通过长距离依赖建模，提高了去雾模型的泛化能力和细节保留。该模型在多个去雾数据集上表现出色，尤其是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度21',
        1, '2024-11-13 12:52:34', '2024-12-03 21:22:35', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (42, 23, '图像去雾', 'rshaze', null, 'DehazeFormer/rshaze', null, null, null, 'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 12:52:50', '2024-11-13 12:52:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (43, 42, '图像去雾', 'rshaze-b', null, 'DehazeFormer/rshaze/dehazeformer-b.pth', '10.71 MB', null, null,
        'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 12:59:58', '2024-11-13 12:59:58', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (44, 42, '图像去雾', 'rshaze-m', null, 'DehazeFormer/rshaze/dehazeformer-m.pth', '18.51 MB', null, null,
        'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 13:00:05', '2024-11-13 13:00:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (45, 42, '图像去雾', 'rshaze-s', null, 'DehazeFormer/rshaze/dehazeformer-s.pth', '5.46 MB', null, null,
        'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 13:00:18', '2024-11-13 13:00:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (46, 42, '图像去雾', 'rshaze-t', null, 'DehazeFormer/rshaze/dehazeformer-t.pth', '2.9 MB', null, null,
        'algorithm.DehazeNet.run',
        'DehazeNet 是早期基于卷积神经网络的图像去雾方法，通过多尺度映射层和非线性回归层，直接从输入图像预测透射率图。该模型结构简单，计算复杂度低，但在去雾效果上仍有提升空间',
        1, '2024-11-13 13:00:25', '2024-11-13 13:00:25', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (47, 0, '图像去雾', 'FCD', null, 'FCD/framework_da_230221_121802_gen.pth', '592.75 MB', null, null,
        'algorithm.FCD.run',
        'FCD 是一种基于全卷积网络的图像去雾方法，通过密集连接的卷积层进行端到端的去雾处理。该模型简化了模型结构，提高了计算效率，适用于实时去雾应用',
        1, '2024-11-13 13:00:33', '2024-11-13 13:00:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (48, 0, '图像去雾', 'FFANet', null, 'FFA-Net', null, null, null, 'algorithm.FFANet.run',
        'FFANet 是一种端到端的图像去雾模型，通过特征融合和注意力机制，提高了模型对复杂场景的适应能力。该模型在多个数据集上表现出色，特别是在处理薄雾和厚雾区域时，能够有效保留图像细节',
        1, '2024-11-13 13:00:40', '2024-11-13 13:00:40', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (49, 48, '图像去雾', 'its', null, 'FFA-Net/its_train_ffa_3_19.pk', '21.26 MB', null, null,
        'algorithm.FFANet.run',
        'FFANet 是一种端到端的图像去雾模型，通过特征融合和注意力机制，提高了模型对复杂场景的适应能力。该模型在多个数据集上表现出色，特别是在处理薄雾和厚雾区域时，能够有效保留图像细节',
        1, '2024-11-13 13:12:54', '2024-11-13 13:12:54', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (50, 48, '图像去雾', 'ots', null, 'FFA-Net/ots_train_ffa_3_19.pk', '25.39 MB', null, null,
        'algorithm.FFANet.run',
        'FFANet 是一种端到端的图像去雾模型，通过特征融合和注意力机制，提高了模型对复杂场景的适应能力。该模型在多个数据集上表现出色，特别是在处理薄雾和厚雾区域时，能够有效保留图像细节',
        1, '2024-11-13 13:13:01', '2024-11-13 13:13:01', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (51, 0, '图像去雾', 'FogRemoval', null, 'FogRemoval/NH-HAZE_params_0100000.pt', '512.01 MB', null, null,
        'algorithm.FogRemoval.run',
        'FogRemoval 是一种多阶段的图像去雾方法，通过逐步优化图像质量，实现自然的去雾效果。该模型结合了物理模型和深度学习，能够在不同光照条件下有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:13:11', '2024-11-13 13:13:11', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (52, 0, '图像去雾', 'GCANet', null, 'GCANet/wacv_gcanet_dehaze.pth', '2.69 MB', null, null,
        'algorithm.GCANet.run',
        'GCANet 是一种利用全局上下文模块的图像去雾模型，通过增强模型对全局信息的理解，改善去雾结果。该模型在处理复杂场景时，能够有效保留图像的结构和细节，提高去雾效果',
        1, '2024-11-13 13:13:18', '2024-11-13 13:13:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (53, 0, '图像去雾', 'GridDehazeNet', null, 'GridDehazeNet', null, null, null, 'algorithm.GridDehazeNet.run',
        'GridDehazeNet 是一种基于网格结构的图像去雾模型，通过引导透射率估计，提高了去雾的精确度。该模型在处理不同尺度的雾霾时，能够有效保持图像的自然度和清晰度',
        1, '2024-11-13 13:13:27', '2024-11-13 13:13:27', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (54, 53, '图像去雾', 'indoor', null, 'GridDehazeNet/indoor_haze_best_3_6', '3.71 MB', null, null,
        'algorithm.GridDehazeNet.run',
        'GridDehazeNet 是一种基于网格结构的图像去雾模型，通过引导透射率估计，提高了去雾的精确度。该模型在处理不同尺度的雾霾时，能够有效保持图像的自然度和清晰度',
        1, '2024-11-13 13:13:38', '2024-11-13 13:13:38', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (55, 53, '图像去雾', 'outdoor', null, 'GridDehazeNet/outdoor_haze_best_3_6', '3.71 MB', null, null,
        'algorithm.GridDehazeNet.run',
        'GridDehazeNet 是一种基于网格结构的图像去雾模型，通过引导透射率估计，提高了去雾的精确度。该模型在处理不同尺度的雾霾时，能够有效保持图像的自然度和清晰度',
        1, '2024-11-13 13:13:44', '2024-11-13 13:13:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (56, 0, '图像去雾', 'ImgRestorationSde', null, 'image-restoration-sde', null, null, null,
        'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:13:52', '2024-11-13 13:13:52', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (57, 56, '图像去模糊', 'deblurring', null, 'image-restoration-sde/deblurring/ir-sde-deblurring.pth', '523.23 MB',
        null, null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:22:32', '2024-11-13 13:22:32', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (58, 56, '图像去噪', 'denoising', null, 'image-restoration-sde/denoising/ir-sde-denoising.pth', '523.19 MB',
        null, null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:22:39', '2024-11-13 13:22:39', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (59, 56, '图像去雨', 'deraining', null, 'image-restoration-sde/deraining', null, null, null,
        'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:22:46', '2024-11-13 13:22:46', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (60, 59, '图像去雨', 'deraining-H100', null, 'image-restoration-sde/deraining/ir-sde-derainH100.pth',
        '523.23 MB', null, null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:22:55', '2024-11-13 13:22:55', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (61, 59, '图像去雨', 'deraining-L100', null, 'image-restoration-sde/deraining/ir-sde-derainL100.pth',
        '523.23 MB', null, null, 'algorithm.ImgRestorationSde.run',
        'ImageRestorationSDE (Image Restoration with Stochastic Differential Equations)是一种将图像去雾视为随机微分方程求解过程的模型，通过优化图像的恢复过程，实现高质量的去雾效果。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:23:01', '2024-11-13 13:23:01', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (62, 0, '图像去雾', 'ITBDehaze', null, 'ITBdehaze/best.pkl', '423.84 MB', null, null, 'algorithm.ITBDehaze.run',
        'ITBDehaze (Image Texture and Boundary Dehazing)是一种利用图像的纹理和边界信息的图像去雾模型，通过多尺度处理增强去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度24',
        1, '2024-11-13 13:23:09', '2024-11-13 13:23:09', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (63, 0, '图像去雾', 'LightDehazeNet', null, 'LightDehazeNet/trained_LDNet.pth', '122.61 KB', '29.48 KB',
        '1.84 GB', 'algorithm.LightDehazeNet.run',
        'LightDehazeNet 是一种轻量级的图像去雾模型，适用于移动设备上的实时去雾应用。该模型通过优化网络结构，减少了计算复杂度，同时保持了较高的去雾效果',
        1, '2024-11-13 13:23:15', '2024-12-03 21:22:41', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (64, 0, '图像去雾', 'LKDNet', null, 'LKDNet', null, null, null, 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:22', '2024-11-13 13:23:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (65, 64, '图像去雾', 'ITS', null, 'LKDNet/ITS', null, null, null, 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:29', '2024-11-13 13:23:29', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (66, 65, '图像去雾', 'ITS-b', null, 'LKDNet/ITS/LKD-b/LKD-b.pth', '4.94 MB', '1.16 MB', '11.32 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:47', '2024-12-03 21:22:41', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (67, 65, '图像去雾', 'ITS-l', null, 'LKDNet/ITS/LKD-l/LKD-l.pth', '9.67 MB', '2.27 MB', '22.2 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:53', '2024-12-03 21:22:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (68, 65, '图像去雾', 'ITS-s', null, 'LKDNet/ITS/LKD-s/LKD-s.pth', '2.57 MB', '619.33 KB', '5.89 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:23:58', '2024-12-03 21:22:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (69, 65, '图像去雾', 'ITS-t', null, 'LKDNet/ITS/LKD-t/LKD-t.pth', '1.39 MB', '335.13 KB', '3.17 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:08', '2024-12-03 21:22:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (70, 64, '图像去雾', 'OTS', null, 'LKDNet/OTS', null, null, null, 'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:15', '2024-11-13 13:24:15', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (71, 70, '图像去雾', 'OTS-b', null, 'LKDNet/OTS/LKD-b/LKD-b.pth', '4.94 MB', '1.16 MB', '11.32 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:21', '2024-12-03 21:22:43', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (72, 70, '图像去雾', 'OTS-l', null, 'LKDNet/OTS/LKD-l/LKD-l.pth', '9.67 MB', '2.27 MB', '22.2 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:27', '2024-12-03 21:22:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (73, 70, '图像去雾', 'OTS-s', null, 'LKDNet/OTS/LKD-s/LKD-s.pth', '2.57 MB', '619.33 KB', '5.89 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:35', '2024-12-03 21:22:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (74, 70, '图像去雾', 'OTS-t', null, 'LKDNet/OTS/LKD-t/LKD-t.pth', '1.39 MB', '335.13 KB', '3.17 GB',
        'algorithm.LKDNet.run',
        'LKDNet (Local and Global Knowledge Distillation Network): LKDNet 是一种通过局部和全局特征的结合，提高模型鲁棒性和泛化能力的图像去雾模型。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度。',
        1, '2024-11-13 13:24:41', '2024-12-03 21:22:44', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (75, 0, '图像去雾', 'MADN', null, 'MADN/model.pth', '2.13 MB', null, null, 'algorithm.MADN.run',
        'MADN (Multi-Adversarial Domain Network): MADN 是一种基于多对抗域网络的图像去雾模型，通过域适应技术，提高了模型对不同场景的适应能力。该模型在处理真实世界中的雾霾图像时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:24:50', '2024-11-13 13:24:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (76, 0, '图像去雾', 'MB-TaylorFormer', null, 'MB-TaylorFormer', null, null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:24:57', '2024-11-13 13:24:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (77, 76, '图像去雾', 'dense-haze', null, 'MB-TaylorFormer', null, null, null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:22', '2024-11-13 13:28:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (78, 77, '图像去雾', 'dense-haze-b', null, 'MB-TaylorFormer/densehaze-MB-TaylorFormer-B.pth', '10.49 MB', null,
        null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:29', '2024-11-13 13:28:29', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (79, 77, '图像去雾', 'dense-haze-l', null, 'MB-TaylorFormer/densehaze-MB-TaylorFormer-L.pth', '29.04 MB', null,
        null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:36', '2024-11-13 13:28:36', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (80, 76, '图像去雾', 'its', null, 'MB-TaylorFormer/ITS-MB-TaylorFormer-L.pth', '29.04 MB', null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:43', '2024-11-13 13:28:43', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (81, 76, '图像去雾', 'ohaze', null, 'MB-TaylorFormer/ohaze-MB-TaylorFormer-B.pth', '10.49 MB', null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:28:49', '2024-11-13 13:28:49', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (82, 76, '图像去雾', 'ots', null, 'MB-TaylorFormer', null, null, null, 'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:29:00', '2024-11-13 13:29:00', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (83, 82, '图像去雾', 'ots-b', null, 'MB-TaylorFormer/OTS-MB-TaylorFormer-B.pth', '10.51 MB', null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:29:06', '2024-11-13 13:29:06', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (84, 82, '图像去雾', 'ots-l', null, 'MB-TaylorFormer/OTS-MB-TaylorFormer-L.pth', '29.04 MB', null, null,
        'algorithm.MB-TaylorFormer.run',
        'MB-TaylorFormer 是一种基于泰勒展开和 Transformer 的图像去雾模型，通过精确建模大气散射过程，实现高质量的去雾效果。该模型在多个数据集上表现出色，特别是在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度',
        1, '2024-11-13 13:29:12', '2024-11-13 13:29:12', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (85, 0, '图像去雾', 'MixDehazeNet', null, 'MixDehazeNet', null, null, null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 13:29:57', '2024-11-13 13:29:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (86, 85, '图像去雾', 'haze4k', null, 'MixDehazeNet/haze4k', null, null, null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:26:20', '2024-11-13 14:26:20', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (87, 86, '图像去雾', 'haze4k-l', null, 'MixDehazeNet/haze4k/MixDehazeNet-l.pth', '143.93 MB', '11.84 MB',
        '104.58 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:26:47', '2024-12-03 21:22:48', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (88, 85, '图像去雾', 'indoor', null, 'MixDehazeNet/indoor', null, null, null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:26:54', '2024-11-13 14:26:54', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (89, 88, '图像去雾', 'indoor-b', null, 'MixDehazeNet/indoor/MixDehazeNet-b.pth', '72.44 MB', '5.96 MB',
        '52.6 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:00', '2024-12-03 21:22:50', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (90, 88, '图像去雾', 'indoor-l', null, 'MixDehazeNet/indoor/MixDehazeNet-l.pth', '143.93 MB', '11.84 MB',
        '104.58 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:06', '2024-12-03 21:22:55', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (91, 85, '图像去雾', 'outdoor', null, 'MixDehazeNet/outdoor', null, null, null, 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:13', '2024-11-13 14:27:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (92, 91, '图像去雾', 'outdoor-b', null, 'MixDehazeNet/outdoor/MixDehazeNet-b.pth', '72.44 MB', '5.96 MB',
        '52.6 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:26', '2024-12-03 21:22:57', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (93, 91, '图像去雾', 'outdoor-b', null, 'MixDehazeNet/outdoor/MixDehazeNet-l.pth', '143.93 MB', '11.84 MB',
        '104.58 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:35', '2024-12-03 21:23:01', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (94, 91, '图像去雾', 'outdoor-b', null, 'MixDehazeNet/outdoor/MixDehazeNet-s.pth', '36.69 MB', '3.02 MB',
        '26.61 GB', 'algorithm.MixDehazeNet.run',
        'MixDehazeNet 是一种融合多个去雾模型优点的图像去雾模型，通过集成学习提高去雾效果。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:41', '2024-12-03 21:23:02', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (95, 0, '图像去雾', 'MSFNet', null, 'MSFNet', null, null, null, 'algorithm.MSFNet.run',
        'MSFNet 是一种多尺度特征融合网络，通过跨尺度信息交换，增强图像细节。该模型在处理不同尺度的雾霾时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:49', '2024-11-13 14:27:49', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (96, 95, '图像去雾', 'indoor', null, 'MSFNet/indoor.pth', '4.01 MB', '1003.22 KB', '17.01 GB',
        'algorithm.MSFNet.run',
        'MSFNet 是一种多尺度特征融合网络，通过跨尺度信息交换，增强图像细节。该模型在处理不同尺度的雾霾时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:27:58', '2024-12-03 21:23:03', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (97, 95, '图像去雾', 'outdoor', null, 'MSFNet/outdoor.pth', '3.98 MB', '1003.22 KB', '17.01 GB',
        'algorithm.MSFNet.run',
        'MSFNet 是一种多尺度特征融合网络，通过跨尺度信息交换，增强图像细节。该模型在处理不同尺度的雾霾时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:04', '2024-12-03 21:23:03', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (98, 0, '图像去雾', 'PSD', null, 'PSD', null, null, null, 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:10', '2024-11-13 14:28:10', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (99, 98, '图像去雾', 'PSD-MSBDN', null, 'PSD/PSB-MSBDN', '126.4 MB', null, null, 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:19', '2024-11-13 14:28:19', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (100, 98, '图像去雾', 'PSD-FFANET', null, 'PSD/PSD-FFANET', '23.84 MB', null, null, 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:33', '2024-11-13 14:28:33', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (101, 98, '图像去雾', 'PSD-GCANET', null, 'PSD/PSD-GCANET', '9.23 MB', null, null, 'algorithm.PSD.run',
        'PSD (Physics-Driven Deep Learning): PSD 是一种物理驱动的深度学习方法，结合物理模型和深度学习，提高去雾精度。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:42', '2024-11-13 14:28:42', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (102, 0, '图像去雾', 'RIDCP', null, 'RIDCP/pretrained_RIDCP.pth', '116.41 MB', '27.39 MB', '175.69 GB',
        'algorithm.RIDCP.run',
        '目前图像去雾领域缺乏强大的先验知识，作者提出在 VQGAN1使用大规模高质量数据集，预训练出一个离散码本，封装高质量先验（HQPs）；并且引入了一种提取特征能力较强的编码器 E，以及设计了一个具有归一化特征对齐模块（NFA）的解码器 G ，共同构建出基于高质量码本先验的真实图像去雾网络（RIDCP）',
        1, '2024-11-13 14:28:49', '2024-12-03 21:23:09', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (103, 0, '图像去雾', 'SCANet', null, 'SCANet', null, null, null, 'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:28:56', '2024-11-13 14:28:56', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (104, 103, '图像去雾', 'SCANet-40', null, 'SCANet/Gmodel_40.tar', '27.7 MB', null, null, 'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:05', '2024-11-13 14:29:05', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (105, 103, '图像去雾', 'SCANet-105', null, 'SCANet/Gmodel_105.tar', '27.7 MB', null, null,
        'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:11', '2024-11-13 14:29:11', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (106, 103, '图像去雾', 'SCANet-120', null, 'SCANet/Gmodel_120.tar', '27.68 MB', null, null,
        'algorithm.SCANet.run',
        'SCANet (Spatial Context Attention Network): SCANet 是一种空间注意网络，通过空间注意力机制优化特征提取，提高去雾质量。该模型在处理复杂场景时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:16', '2024-11-13 14:29:16', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (107, 0, '图像去雾', 'SGIDPFF', null, 'SGID-PFF', null, null, null, 'algorithm.SGIDPFF.run',
        'SGIDPFF (Single Image Dehazing with Heterogeneous Task Imitation): SGIDPFF 是一种通过异构任务模仿技术，提高去雾效果的图像去雾模型。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:22', '2024-11-13 14:29:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (108, 107, '图像去雾', 'indoor', null, 'SGID-PFF/SOTS_indoor.pt', '52.94 MB', '13.22 MB', '145.66 GB',
        'algorithm.SGIDPFF.run',
        'SGIDPFF (Single Image Dehazing with Heterogeneous Task Imitation): SGIDPFF 是一种通过异构任务模仿技术，提高去雾效果的图像去雾模型。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:28', '2024-12-03 21:23:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (109, 107, '图像去雾', 'SGIDPFF', null, 'SGID-PFF/SOTS_outdoor.pt', '52.94 MB', '13.22 MB', '145.66 GB',
        'algorithm.SGIDPFF.run',
        'SGIDPFF (Single Image Dehazing with Heterogeneous Task Imitation): SGIDPFF 是一种通过异构任务模仿技术，提高去雾效果的图像去雾模型。该模型在处理不同类型的雾霾时，能够有效保留图像的细节和自然度，适用于多种场景下的图像去雾任务。',
        1, '2024-11-13 14:29:33', '2024-12-03 21:23:13', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (110, 0, '图像去雾', 'TSDNet', null, 'TSDNet/GNet.tar', '13.94 MB', null, null, 'algorithm.TSDNet.run',
        'TSDNet (Temporal-Spatial-Depth Network): TSDNet 是一种时间-空间-深度联合建模的图像去雾模型，通过优化视频序列的去雾效果，提高视频去雾的连贯性和自然度。该模型在处理视频序列时，能够有效去除雾霾，恢复图像的清晰度，适用于多种场景下的视频去雾任务。',
        1, '2024-11-13 14:29:43', '2024-11-13 14:29:43', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (111, 0, '图像去雾', 'WPXNet', null, 'WPXNet', null, null, null, 'algorithm.WPXNet.run',
        '引入无雾图像训练得到离散码本，封装具有原有图像色彩和结构的高质量先验知识。随后构建一种双分支神经网络结构，即先验匹配分支和通道注意力分支，利用邻域注意力和通道注意力提取有雾图像全局特征并学习浓雾区域与底层场景之间复杂交互特征，通过特征融合模块对两个分支提取的特征进行融合。将高质量先验约束码本与有雾图像特征通过一种可控距离重计算操作进行匹配，从而替换图像中受到雾影响的区域。本发明对原有雾图像进行重建实现了端到端的图像去雾流程，在保留原图像细节和纹理结构的情况下，提高了有雾图像的清晰度和可识别度。',
        1, '2024-11-28 14:03:07', '2024-11-28 14:03:07', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (112, 111, '图像去雾', 'DENSE-HAZE', null, 'WPXNet/dense-haze.pth', '151.75 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 1, '2024-11-28 14:04:05', '2024-12-03 21:23:17', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (113, 111, '图像去雾', 'I-HAZE', null, 'WPXNet/i-haze.pth', '151.75 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 1, '2024-11-28 14:04:24', '2024-12-03 21:23:18', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (114, 111, '图像去雾', 'O-HAZE', null, 'WPXNet/o-haze.pth', '151.74 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 1, '2024-11-28 14:04:45', '2024-12-03 21:23:21', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (115, 111, '图像去雾', 'NH-HAZE-20', null, 'WPXNet/nh-haze-20.pth', '151.74 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 1, '2024-11-28 14:05:06', '2024-12-03 21:23:22', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (116, 111, '图像去雾', 'NH-HAZE-21', null, 'WPXNet/nh-haze-21.pth', '151.74 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 1, '2024-11-28 14:05:16', '2024-12-03 21:23:24', 2, 2);
insert into sys_algorithm (id, parent_id, type, name, img, path, size, params, flops, import_path, description,
                           status, create_time, update_time, create_by, update_by)
values (117, 111, '图像去雾', 'NH-HAZE-23', null, 'WPXNet/nh-haze-23.pth', '151.75 MB', '36.99 MB', '211.24 GB',
        'algorithm.WPXNet.run', '用于浓雾数据集的权重模型', 1, '2024-11-28 14:05:23', '2024-12-03 21:23:26', 2, 2);

SET FOREIGN_KEY_CHECKS = 1;
