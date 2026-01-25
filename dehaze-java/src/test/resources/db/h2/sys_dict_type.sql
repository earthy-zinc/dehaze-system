create table dehaze_test.sys_dict_type
(
    id          bigint auto_increment comment '主键 '
        primary key,
    name        varchar(50) default '' null comment '类型名称',
    code        varchar(50) default '' null comment '类型编码',
    status      tinyint(1)  default 0  null comment '状态(0:正常;1:禁用)',
    remark      varchar(255)           null comment '备注',
    create_time datetime               null comment '创建时间',
    update_time datetime               null comment '更新时间',
    constraint type_code
        unique (code)
);

