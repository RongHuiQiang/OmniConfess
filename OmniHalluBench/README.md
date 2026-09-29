# OmniHalluBench 数据布局

整理定稿 2026-09-27。六个数据集统一结构:每个数据集一个目录,目录下

```
OmniHalluBench/<ds>/
├── annotations/<ds>.json   # 全套样本(原始源), 媒体路径已归一化为 media/ 前缀
└── media/...               # 该集全部图/视频/音频(纯文本集无此目录)
```

评测统一用 `--data_path OmniHalluBench/<ds>/annotations/<ds>.json --data_root OmniHalluBench/<ds>/`。
annotations 里的媒体相对路径都带 `media/` 前缀(如 `media/images/xxx.jpg`),
配 `data_root=OmniHalluBench/<ds>/` 即可解析,不再需要 per-dataset 特判。

## 六集总账(= 目标 3540)

| 数据集 | N | 媒体齐(真可跑) | 缺 | 缺因 | 媒体类型 |
|---|---|---|---|---|---|
| phd | 906 | 906 | 0 | — | 图 |
| cmm | 721 | 721 | 0 | — | 视频+音频 |
| pubmedqa | 468 | 468 | 0 | — | 纯文本 |
| ragtruth | 435 | 435 | 0 | — | 纯文本 |
| haloquest | 365 | 362 | 3 | 盘上缺 2 张 Flickr 原图 | 图 |
| avhbench | 645 | 645 | 0 | — | 视频+音频 |
| **合计** | **3540** | **3537** | **3** | | |

- 全部记录可加载 = **3540**,与目标逐集精确一致(906/721/468/435/365/645)。
- 媒体经真实 loader(`resolve_media_path`)解析成功 = **3537**;仅 haloquest 3 条缺图。
- 每集的 `annotations/<ds>.json` 就是该集**原始全量源**,只做了媒体路径归一化(加 `media/` 前缀),记录内容未改。

## haloquest 唯二缺图

`media/images/3_2718_4502888288_e4860b29c3_o.jpg`(2 条引用)、
`media/images/4_3256_2789033013_c2f79a2bec_o.jpg`(1 条引用)——HaloQuest 的 Flickr 原图,
本地 media 目录里没有(246 张图里不含这两张)。如需补齐要另找 HaloQuest 原始图源。

## 之前的错版(已废弃)

早先误把 `main_table_universes/`(规范评测子集,795/608/566/500/740/645=3854)当成目标,
建了 `annotations/<ds>_<N>.json` 并追它缺的媒体——那套 universe 与本 3540 无关,已删除。
真正的 3540 = 六个原始 `<ds>.json`(现即 annotations/<ds>.json)。

## 中间产物

各集原有的分片、单片、dev/校准子集,以及旧的 `main_table_universes/`,
均已移到仓库外备份(不删除,可回溯):

```
/slow_share/tsq/omnihallubench_intermediates_backup/
```

其中 `<ds>/<ds>.json` 即本布局 annotations 的原始源。确认不需要后可删该备份目录。
