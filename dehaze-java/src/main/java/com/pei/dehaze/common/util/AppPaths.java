package com.pei.dehaze.common.util;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 路径锚点：仓库根（含 .env.example 的目录）与 Java 服务根（&lt;仓库根&gt;/dehaze-java）。
 * 配置文件中的相对路径一律经此锚定后使用，避免依赖进程工作目录。
 *
 * @author earthy-zinc
 */
public final class AppPaths {

    private static final String REPO_MARKER = ".env.example";
    private static final String SERVICE_DIR = "dehaze-java";
    private static final Path REPO_ROOT = locateRepoRoot();

    private AppPaths() {
    }

    public static Path repoRoot() {
        return REPO_ROOT;
    }

    /**
     * 相对仓库根解析（三端共享目录：data/upload、datasets、models）
     */
    public static Path resolveRepoRelative(String path) {
        return resolve(REPO_ROOT, path);
    }

    /**
     * 相对 Java 服务根解析（服务自身产物：logs 等）
     */
    public static Path resolveServiceRelative(String path) {
        return resolve(REPO_ROOT.resolve(SERVICE_DIR), path);
    }

    private static Path resolve(Path base, String path) {
        Path target = Paths.get(path);
        return target.isAbsolute() ? target.normalize() : base.resolve(target).normalize();
    }

    /**
     * 从进程工作目录向上查找仓库根，兼容 IDE 与 scripts/run.py 以 dehaze-java 为工作目录的情形。
     * 源码树不可见（jar 独立部署）时退化为当前工作目录，此时须用绝对路径显式配置目录。
     */
    private static Path locateRepoRoot() {
        Path cwd = Paths.get("").toAbsolutePath();
        Path dir = cwd;
        for (int i = 0; i < 6 && dir != null; i++, dir = dir.getParent()) {
            if (Files.exists(dir.resolve(REPO_MARKER))) {
                return dir;
            }
            if (SERVICE_DIR.equals(String.valueOf(dir.getFileName())) && Files.exists(dir.resolve("pom.xml"))) {
                return dir.getParent();
            }
        }
        return cwd;
    }
}
