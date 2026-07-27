SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (1, 0, '图像去雾', 'DENSE-HAZE', null,
        'DENSE-HAZE 引入了一种新的去雾数据集，以浓密均匀的雾霾场景为特征。该数据集包含 55对真实的浓雾图像和各种室外场景的相应无雾图像。这些朦胧图像是通过专业雾霾机器生成的真实雾霾记录的。生成的浓雾图像几乎难以辨别图像中原来存在的物体，与常规数据集相比去雾难度非常大。',
        'Dense-Haze', '234.74 MB', 1, 0, '2024-11-11 19:29:49', '2024-11-11 19:29:49', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (2, 0, '图像去雾', 'O-HAZE', null,
        'O-haze 数据集是由CVLab实验室在2016年发布的，主要用于评估和测试图像去雾算法的性能。该数据集包含了合成的有雾图像和相应的清晰图像对，这些图像都是基于真实的户外场景生成的。包含45对户外场景的有雾和清晰图像',
        'O-HAZE', '547.85 MB', 1, 0, '2024-11-11 19:36:32', '2024-11-11 19:36:32', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (3, 0, '图像去雾', 'I-HAZE', null,
        'I-haze 数据集也是由CVLab实验室在2016年发布的，与O-haze数据集类似，它主要用于评估和测试图像去雾算法的性能。不过，I-haze数据集的特点在于其图像更接近实际的室内场景。包含35对有雾和相应的无雾室内图像。',
        'I-HAZE', '312.99 MB', 1, 0, '2024-11-11 19:37:12', '2024-11-11 19:37:12', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (4, 0, '图像去雾', 'NH-HAZE', null,
        'NH-HAZE数据集旨在解决图像去雾领域中的一个重要问题：缺乏真实世界的非均匀雾度图像作为参考数据。许多现实场景中的雾并不均匀分布，因此 NH-HAZE 提供了一组真实的非均匀雾图像和相应的无雾图像对。NH-HAZE 数据集中的非均匀雾度是通过专业的雾发生器模拟真实雾天条件而引入的。是一个更具挑战性和现实性的去雾数据集。',
        'NTIRE', '1.06 GB', 1, 0, '2024-11-11 19:39:24', '2024-11-11 19:39:24', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (5, 4, '图像去雾', 'NH-HAZE-20', null,
        '在2020年，NH-HAZE数据集被用于CVPR NTIRE（New Trends in Image Restoration and Enhancement）研讨会下的图像去雾在线挑战赛中1。这是首个包含55对外部拍摄的真实有雾和对应的无雾图像的数据集，这些图像是使用专业雾生成器在高保真的条件下拍摄的，以模拟真实的非均匀雾霾环境。NH-HAZE 2020的数据集为研究人员提供了评估去雾算法性能的机会，并且由于其现实性，对于开发更加鲁棒的去雾解决方案具有重要意义',
        'NH-HAZE-2020', '316.96 MB', 1, 0, '2024-11-11 19:39:51', '2024-11-11 19:39:51', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (6, 4, '图像去雾', 'NH-HAZE-21', null,
        '到了2021年，NTIRE挑战赛继续进行，这次的非均匀去雾挑战基于扩展后的NH-HAZE数据集，增加了额外35对真实户外拍摄的无雾和非均匀有雾图像。这个扩大的数据集被称为NH-Haze2，它进一步增强了数据集的多样性和复杂度，为参与者提供了更广泛的测试平台来验证他们的算法。此外，在这次挑战中还加入了其他小规模的真实世界数据集如DENSE-HAZE等，用以对比不同方法的效果。',
        'NH-HAZE-2021', '151.36 MB', 1, 0, '2024-11-11 19:40:08', '2024-11-11 19:40:08', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (7, 4, '图像去雾', 'NH-HAZE-23', null,
        '至2023年，NTIRE举办了一次高清非均质去雾挑战赛，这次比赛采用了名为HD-NH-HAZE的新数据集。HD-NH-HAZE包含了50对高清分辨率的户外图像，其中一半是带有非均匀雾霾的图像，另一半则是同一场景的无雾霾图像。这个数据集的引入标志着单张图像去雾领域的一个重要进展，因为它不仅提高了图像的质量标准，而且也推动了去雾技术向着处理更高分辨率图像的方向发展。参赛者们提出的方法在此数据集上进行了客观评估，以便更好地衡量它们在处理实际场景中的表现',
        'NH-HAZE-2023', '618.19 MB', 1, 0, '2024-11-11 19:40:19', '2024-11-11 19:40:19', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (8, 0, '图像去雾', 'RESIDE', null,
        'RESIDE（Realistic Synthetic and Indoor-Outdoor DEhazing）数据集是由北京大学和微软亚洲研究院在2017年联合发布的，旨在为图像去雾研究提供一个大规模、多样化的基准数据集。RESIDE 数据集不仅包含合成的有雾图像和对应的清晰图像，还包含了一些真实世界中的有雾图像，使其成为图像去雾领域最全面的数据集之一。',
        'RESIDE', '19.01 GB', 1, 0, '2024-11-11 19:41:55', '2024-11-11 19:41:55', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (9, 8, '图像去雾', 'ITS', null,
        '室内训练集(ITS) 是RESIDE数据集中的一部分，主要用于算法的训练阶段。ITS包含13,990张由清晰图像生成的合成模糊图像，这些清晰图像是从现有的室内深度数据集NYU2和米德尔伯里立体数据库中选取的1,399张图像。对于每一张清晰图像，通过在不同参数设置下（例如大气光A和散射系数β）生成10张模糊图像。这些参数的设定使得生成的模糊图像能够模拟多种不同的雾霾情况。具体来说，大气光A的值在[0.7, 1.0]之间均匀随机选择，而β则在[0.6, 1.8]之间均匀随机选择。最终，这13,990张图像中的13,000张被用于训练，剩下的990张作为验证集',
        'RESIDE/ITS', '4.74 GB', 1, 0, '2024-11-11 19:42:34', '2024-11-11 19:42:34', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (10, 8, '图像去雾', 'OTS', null,
        '室外训练集(OTS) 是RESIDE-beta部分的数据集，旨在提高对室外环境下的去雾性能。OTS使用了2061张来自北京实时天气的真实室外图像，通过估计每张图像的深度信息后，根据一系列指定的大气散射系数β值（如0.04, 0.06, 0.08, 0.1, 0.15, 0.95, 1等）来合成模糊图像。最终，总共合成了72,135张户外模糊图像。这套新的图像被称为户外训练集（OTS），由成对的干净的户外图像和生成的模糊图像组成，以供算法训练使用',
        'RESIDE/OTS', '12.86 GB', 1, 0, '2024-11-11 19:42:54', '2024-11-11 19:42:54', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (14, 0, '图像去雾', 'RESIDE-6k', null,
        '虽然没有直接提到名为RESIDE-6k的数据集，但我们可以假设这可能是一个包含大约6000张图像的RESIDE数据集的一个子集。如果这是对RESIDE数据集的特定版本，则它可能专注于一个特定的场景（室内或室外）或者用于特定目的（比如训练或测试）。然而，由于没有具体的信息，我们无法确定其确切组成。通常，这样的数据集会包含成对的清晰和模糊图像，以便于模型学习如何从模糊图像恢复清晰图像。',
        'RESIDE-6k', '1.52 GB', 1, 0, '2024-11-12 22:22:42', '2024-11-12 22:22:42', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (15, 14, '图像去雾', 'RESIDE-6k-train', null,
        'RESIDE-6k 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RESIDE-6k/train', '1,021.78 MB',
        1, 0, '2024-11-12 22:23:05', '2024-11-12 22:23:05', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (16, 14, '图像去雾', 'RESIDE-6k-test', null,
        'RESIDE-6k  测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RESIDE-6k/test', '532.3 MB',
        1, 0, '2024-11-12 22:23:16', '2024-11-12 22:23:16', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (17, 0, '图像去雾', 'RESIDE-IN', null,
        '这个名称可能指的是RESIDE数据集中专注于室内场景的部分。结合RESIDE数据集的描述，我们可以合理推测RESIDE-IN可能主要包含了ITS（Indoor Training Set），即室内训练集。该集合包括了13,990个合成的模糊图像，这些图像是基于NYU2和米德尔伯里立体数据库中的1,399个清晰室内图像生成的1。此外，SOTS（Synthetic Objective Testing Set）中的部分室内图像也可能被包含在内，用于评估算法性能。',
        'RESIDE-IN', '8.74 GB', 1, 0, '2024-11-12 22:23:48', '2024-11-12 22:23:48', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (18, 17, '图像去雾', 'RESIDE-IN-train', null,
        'RESIDE-IN 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RESIDE-IN/train', '8.36 GB',
        1, 0, '2024-11-12 22:24:10', '2024-11-12 22:24:10', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (19, 17, '图像去雾', 'RESIDE-IN-test', null,
        'RESIDE-IN 测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RESIDE-IN/test', '392.11 MB',
        1, 0, '2024-11-12 22:25:09', '2024-11-12 22:25:09', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (20, 0, '图像去雾', 'RESIDE-OUT', null,
        '同样，RESIDE-OUT可能是指RESIDE数据集中专注于室外场景的部分。这意味着它可能主要由OTS（Outdoor Training Set）构成，该集合包括72,135张合成的户外模糊图像，这些图像是基于北京实时天气的真实室外图像生成的2。SOTS中的一部分室外图像也可能会被纳入其中，用于测试目的。',
        'RESIDE-OUT', '83.03 GB', 1, 0, '2024-11-12 22:26:09', '2024-11-12 22:26:09', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (21, 20, '图像去雾', 'RESIDE-OUT-train', null,
        'RESIDE-OUT 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RESIDE-OUT/train', '82.89 GB',
        1, 0, '2024-11-12 22:26:47', '2024-11-12 22:26:47', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (22, 20, '图像去雾', 'RESIDE-OUT-test', null,
        'RESIDE-OUT 测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RESIDE-OUT/test',
        '140.19 MB', 1, 0, '2024-11-12 22:27:08', '2024-11-12 22:27:08', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (23, 0, '图像去雾', 'RSHAZE', null,
        'REHAZE数据集是专为图像去雾研究设计的，旨在提供更真实的雾霾条件下的图像。它由苏黎世联邦理工大学等机构发布，包含有雾和无雾图像对，这些图像是在受控环境中使用专业设备拍摄的，以模拟不同的雾霾条件。不过，具体的细节（如图像数量、场景类型等）需要查阅原始论文或官方发布页面来获取准确信息。',
        'RSHAZE', '40.41 GB', 1, 0, '2024-11-12 22:28:28', '2024-11-12 22:28:28', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (24, 23, '图像去雾', 'RSHAZE-train', null,
        'REHAZE 训练集 用于模型的学习阶段，让模型通过大量样本学习如何执行任务。', 'RSHAZE/train', '38.39 GB', 1,
        0, '2024-11-12 22:28:47', '2024-11-12 22:28:47', 2, 2);
insert into sys_dataset (id, parent_id, type, name, img, description, path, size, status, deleted,
                         create_time, update_time, create_by, update_by)
values (25, 23, '图像去雾', 'RSHAZE-test', null,
        'REHAZE 测试集 用于评估经过训练后的模型性能，看其在未见过的数据上的表现如何。', 'RSHAZE/test', '2.02 GB', 1,
        0, '2024-11-12 22:28:54', '2024-11-12 22:28:54', 2, 2);

SET FOREIGN_KEY_CHECKS = 1;