create table dehaze_test.sys_item_file
(
    id                bigint auto_increment comment 'id'
        primary key,
    item_id           bigint                             not null comment '所属数据项id',
    file_id           bigint                             not null comment '文件id',
    thumbnail_file_id bigint                             null comment '缩略图文件id',
    type              varchar(64)                        not null comment '图片类型（清晰图、雾霾图、分割图等）',
    description       varchar(255)                       null comment '描述',
    scene_type        varchar(64)                        null comment '场景类型',
    haze_level        varchar(32)                        null comment '雾霾程度',
    width             int                                null comment '图片宽度',
    height            int                                null comment '图片高度',
    usage_count       bigint   default 0                 null comment '使用次数',
    create_time       datetime default CURRENT_TIMESTAMP null comment '创建时间',
    update_time       datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP comment '更新时间'
);

create index idx_item_id_file_id
    on dehaze_test.sys_item_file (item_id, file_id);

