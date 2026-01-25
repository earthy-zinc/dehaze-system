create table dehaze_test.sys_role
(
    id          bigint auto_increment
        primary key,
    name        varchar(64) default '' not null comment '角色名称',
    code        varchar(32)            null comment '角色编码',
    sort        int                    null comment '显示顺序',
    status      tinyint(1)  default 1  null comment '角色状态(1-正常；0-停用)',
    data_scope  tinyint                null comment '数据权限(0-所有数据；1-部门及子部门数据；2-本部门数据；3-本人数据)',
    deleted     tinyint(1)  default 0  not null comment '逻辑删除标识(0-未删除；1-已删除)',
    create_time datetime               null comment '更新时间',
    update_time datetime               null comment '创建时间',
    constraint name
        unique (name)
);

