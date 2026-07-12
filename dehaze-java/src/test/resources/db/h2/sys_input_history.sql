create table dehaze_test.sys_input_history
(
    id                     bigint auto_increment comment '主键'
        primary key,
    user_id                bigint       not null comment '用户ID',
    original_image_url     varchar(500) null comment '原始图片URL',
    original_thumbnail_url varchar(500) null comment '原始缩略图URL',
    result_image_url       varchar(500) null comment '处理结果图片URL',
    result_thumbnail_url   varchar(500) null comment '结果缩略图URL',
    algorithm_id           bigint       null comment '算法ID',
    algorithm_name         varchar(100) null comment '算法名称（冗余）',
    algorithm_params       text         null comment '算法参数（JSON）',
    processing_time        int          null comment '处理耗时（毫秒）',
    status                 tinyint      default 3 null comment '处理状态（1=成功，2=失败，3=处理中）',
    input_source           varchar(20)  null comment '图片来源（upload/camera/sample）',
    is_favorite            tinyint(1)   default 0 null comment '是否收藏',
    sync_status            tinyint      default 0 null comment '同步状态（0=未同步，1=已同步）',
    create_time            datetime     default CURRENT_TIMESTAMP null comment '创建时间',
    update_time            datetime     default CURRENT_TIMESTAMP null comment '更新时间',
    create_by              bigint       null comment '创建人ID',
    update_by              bigint       null comment '修改人ID'
);
