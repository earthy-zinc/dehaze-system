create table dehaze_test.sys_user
(
    id          int auto_increment
        primary key,
    username    varchar(64)          null comment '用户名',
    nickname    varchar(64)          null comment '昵称',
    gender      tinyint(1) default 1 null comment '性别((1:男;2:女))',
    password    varchar(100)         null comment '密码',
    dept_id     int                  null comment '部门ID',
    avatar      text                 null comment '用户头像',
    mobile      varchar(20)          null comment '联系方式',
    status      tinyint(1) default 1 null comment '用户状态((1:正常;0:禁用))',
    email       varchar(128)         null comment '用户邮箱',
    deleted     tinyint(1) default 0 null comment '逻辑删除标识(0:未删除;1:已删除)',
    create_time datetime             null comment '创建时间',
    update_time datetime             null comment '更新时间',
    constraint login_name
        unique (username)
);

