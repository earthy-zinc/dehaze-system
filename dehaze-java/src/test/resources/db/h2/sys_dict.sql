create table dehaze_test.sys_dict
(
    id          bigint auto_increment comment '主键'
        primary key,
    type_code   varchar(64)             null comment '字典类型编码',
    name        varchar(50)  default '' null comment '字典项名称',
    value       varchar(50)  default '' null comment '字典项值',
    sort        int          default 0  null comment '排序',
    status      tinyint      default 0  null comment '状态(1:正常;0:禁用)',
    defaulted   tinyint      default 0  null comment '是否默认(1:是;0:否)',
    remark      varchar(255) default '' null comment '备注',
    create_time datetime                null comment '创建时间',
    update_time datetime                null comment '更新时间'
);

