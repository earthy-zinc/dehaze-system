"""
User-Agent 解析工具
"""

import re


def parse_user_agent(user_agent: str) -> tuple[str, str]:
    """
    解析 User-Agent 字符串，提取浏览器和操作系统信息

    Args:
        user_agent: User-Agent 字符串

    Returns:
        (browser, os) 元组
    """
    if not user_agent:
        return "Unknown", "Unknown"

    browser = "Unknown"
    browser_patterns = [
        (r"Edge/(\d+)", "Edge"),
        (r"Edg/(\d+)", "Edge"),
        (r"OPR/(\d+)", "Opera"),
        (r"Chrome/(\d+)", "Chrome"),
        (r"Firefox/(\d+)", "Firefox"),
        (r"Safari/(\d+)", "Safari"),
        (r"MSIE (\d+)", "IE"),
        (r"Trident/.*rv:(\d+)", "IE"),
    ]
    for pattern, name in browser_patterns:
        match = re.search(pattern, user_agent)
        if match:
            browser = f"{name} {match.group(1)}"
            break

    os_name = "Unknown"
    os_patterns = [
        (r"Windows NT (\d+\.\d+)", "Windows"),
        (r"Mac OS X (\d+[._]\d+)", "macOS"),
        (r"Linux", "Linux"),
        (r"Android (\d+)", "Android"),
        (r"iPhone OS (\d+)", "iOS"),
        (r"iPad.*OS (\d+)", "iOS"),
    ]
    for pattern, name in os_patterns:
        match = re.search(pattern, user_agent)
        if match:
            version = match.group(1).replace("_", ".") if match.groups() else ""
            os_name = f"{name} {version}" if version else name
            break

    return browser, os_name
