create table dehaze_test.sys_dataset_item
(
    id          bigint auto_increment comment 'id'
        primary key,
    dataset_id  bigint                             not null comment '所属数据集id',
    name        varchar(64)                        null comment '数据项名称',
    create_time datetime default CURRENT_TIMESTAMP null comment '创建时间',
    update_time datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP comment '更新时间'
);

create index idx_dataset_id
    on dehaze_test.sys_dataset_item (dataset_id);

