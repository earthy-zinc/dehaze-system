create table dehaze_test.sys_dept
(
    id          bigint auto_increment comment '主键'
        primary key,
    name        varchar(64)  default '' not null comment '部门名称',
    parent_id   bigint       default 0  not null comment '父节点id',
    tree_path   varchar(255) default '' null comment '父节点id路径',
    sort        int          default 0  null comment '显示顺序',
    status      tinyint      default 1  not null comment '状态(1:正常;0:禁用)',
    deleted     tinyint      default 0  null comment '逻辑删除标识(1:已删除;0:未删除)',
    create_time datetime                null comment '创建时间',
    update_time datetime                null comment '更新时间',
    create_by   bigint                  null comment '创建人ID',
    update_by   bigint                  null comment '修改人ID'
);

