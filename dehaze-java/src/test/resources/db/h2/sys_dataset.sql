create table dehaze_test.sys_dataset
(
    id          bigint auto_increment comment '数据集ID'
        primary key,
    parent_id   bigint        default 0  not null comment '父数据集ID',
    type        varchar(64)   default '' not null comment '数据集类型',
    name        varchar(64)   default '' not null comment '数据集名称',
    img         text                     null comment '数据集样例图片',
    description varchar(2048) default '' null comment '数据集描述',
    path        varchar(512)  default '' not null comment '存储位置',
    size        varchar(100)  default '' null comment '占用空间大小',
    status      tinyint       default 1  not null comment '状态(1:启用；0:禁用)',
    usage_count bigint        default 0  not null comment '使用次数',
    deleted     tinyint       default 0  null comment '逻辑删除标识(1:已删除;0:未删除)',
    create_time datetime                 null comment '创建时间',
    update_time datetime                 null comment '更新时间',
    create_by   bigint                   null comment '创建人ID',
    update_by   bigint                   null comment '修改人ID'
);

create index idx_create_time
    on dehaze_test.sys_dataset (create_time);

create index idx_deleted
    on dehaze_test.sys_dataset (deleted);

create index idx_name
    on dehaze_test.sys_dataset (name);

create index idx_parent_id
    on dehaze_test.sys_dataset (parent_id);

create index idx_parent_name
    on dehaze_test.sys_dataset (parent_id, name);

create index idx_status
    on dehaze_test.sys_dataset (status);

