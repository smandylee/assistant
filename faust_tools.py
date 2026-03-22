import datetime
import json
import urllib.parse
import urllib.request

from livekit.agents import function_tool


@function_tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    웹에서 최신 정보를 간단히 조회합니다.

    Args:
        query: 검색 질의어.
        max_results: 반환할 최대 결과 개수(1~5 권장).
    """
    safe_max = max(1, min(int(max_results), 5))
    q = urllib.parse.quote_plus(query.strip())
    if not q:
        return "검색어가 비어 있습니다."

    url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"
    req = urllib.request.Request(url, headers={"User-Agent": "Faust-Agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as res:
            data = json.loads(res.read().decode("utf-8", errors="ignore"))
    except Exception as exc:
        return f"검색 실패: {exc}"

    lines: list[str] = []
    abstract = str(data.get("AbstractText", "")).strip()
    abstract_url = str(data.get("AbstractURL", "")).strip()
    if abstract:
        lines.append(f"- 요약: {abstract}")
        if abstract_url:
            lines.append(f"- 출처: {abstract_url}")

    related = data.get("RelatedTopics", [])
    for item in related:
        if len(lines) >= safe_max + 2:
            break
        if not isinstance(item, dict):
            continue
        text = str(item.get("Text", "")).strip()
        first_url = str(item.get("FirstURL", "")).strip()
        if not text:
            continue
        if first_url:
            lines.append(f"- {text} ({first_url})")
        else:
            lines.append(f"- {text}")

    if not lines:
        return "검색 결과를 찾지 못했습니다. 검색어를 더 구체적으로 입력해 보십시오."
    return "\n".join(lines[: safe_max + 2])


@function_tool
def current_datetime(tz_offset_hours: int = 9) -> str:
    """
    현재 날짜/시간을 반환합니다.

    Args:
        tz_offset_hours: UTC 기준 시간대 오프셋(한국은 9).
    """
    tz = datetime.timezone(datetime.timedelta(hours=int(tz_offset_hours)))
    now = datetime.datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


TOOLS = [web_search, current_datetime]
