create table dehaze_test.sys_algorithm
(
    id           bigint auto_increment comment '模型id'
        primary key,
    parent_id    bigint       default 0  null comment '模型的父id',
    type         varchar(100) default '' null comment '模型类型',
    version      varchar(50)             null comment '算法版本号',
    name         varchar(64)             not null comment '模型名称',
    img          text                    null comment '模型图片',
    path         varchar(255) default '' null comment '模型存储路径',
    size         varchar(100)            null comment '模型大小',
    params       varchar(255)            null comment '模型参数',
    flops        varchar(255)            null comment '模型浮点运算次数',
    import_path  varchar(255)            null comment '模型代码导入路径',
    description  varchar(2048)           null comment '针对该模型的详细描述',
    status       tinyint(1)   default 1  null comment '状态(1:启用；0:禁用)',
    audit_by     bigint                  null comment '审核人ID',
    audit_time   datetime                null comment '审核时间',
    audit_remark varchar(500)            null comment '审核备注',
    create_time  datetime                null comment '创建时间',
    update_time  datetime                null comment '更新时间',
    create_by    bigint                  null comment '创建人ID',
    update_by    bigint                  null comment '修改人ID'
);
