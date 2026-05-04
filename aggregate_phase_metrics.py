#!/usr/bin/env python3
import json
import os
import zipfile
from bisect import bisect_left
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape


PHASE_COLUMNS = [f"P{i}" for i in range(1, 10)]
DEFAULT_VIDEOS = ("f1", "f2", "f3", "f4", "f5", "f6", "f8", "f9")
PARTS = ("chest", "hip")


def _combined_frame_lookup(result: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    lookup = {
        int(frame.get("face_on_frame", 0)): frame
        for frame in result.get("frames", []) or []
    }
    return lookup, sorted(lookup.keys())


def _nearest_frame(
    lookup: Dict[int, Dict[str, Any]],
    frame_ids: Sequence[int],
    target_frame: int,
) -> Optional[Dict[str, Any]]:
    if not frame_ids:
        return None
    pos = bisect_left(frame_ids, target_frame)
    candidates: List[int] = []
    if pos < len(frame_ids):
        candidates.append(frame_ids[pos])
    if pos > 0:
        candidates.append(frame_ids[pos - 1])
    best_id = min(candidates, key=lambda item: abs(item - target_frame))
    return lookup.get(best_id)


def _phase_frames(result: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    lookup, frame_ids = _combined_frame_lookup(result)
    frames: List[Optional[Dict[str, Any]]] = [None] * 9
    for pair in result.get("phase_frames", []) or []:
        idx = int(pair.get("phase_index", 0))
        if idx < 1 or idx > 9:
            continue
        face_frame = int((pair.get("face_on") or {}).get("frame", 0))
        frames[idx - 1] = _nearest_frame(lookup, frame_ids, face_frame)
    return frames


def _metric_value(frame: Optional[Dict[str, Any]], part: str, group: str, key: str) -> Optional[float]:
    if not frame:
        return None
    value = (((frame.get(group) or {}).get(part) or {}).get(key))
    if value is None:
        return None
    return round(float(value), 1)


def _sheet_rows(results: Dict[str, Dict[str, Any]], part: str, videos: Sequence[str]) -> List[List[Any]]:
    part_label = part.title()
    header = ["Session", "Part", "Type", *PHASE_COLUMNS]
    rows: List[List[Any]] = [header]
    for video in videos:
        result = results.get(video)
        phase_frames = _phase_frames(result) if result else [None] * 9
        x_values = [_metric_value(frame, part, "translations", "x_in") for frame in phase_frames]
        ry_values = [_metric_value(frame, part, "rotations", "y_deg") for frame in phase_frames]
        rows.append([video, part_label, "RY", *ry_values])
        rows.append([video, part_label, "x", *x_values])
    return rows


def _col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _cell_xml(value: Any, row_idx: int, col_idx: int) -> str:
    ref = f"{_col_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(rows: Sequence[Sequence[Any]]) -> str:
    row_xml = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(_cell_xml(value, row_idx, col_idx) for col_idx, value in enumerate(row, start=1))
        row_xml.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(row_xml)
        + '</sheetData></worksheet>'
    )


def _write_xlsx(path: str, sheets: Sequence[Tuple[str, Sequence[Sequence[Any]]]]) -> None:
    workbook_sheets = []
    workbook_rels = []
    content_overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]

    for idx, (sheet_name, _rows) in enumerate(sheets, start=1):
        safe_name = escape(sheet_name[:31])
        workbook_sheets.append(f'<sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        + "".join(workbook_sheets)
        + '</sheets></workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(workbook_rels)
        + '</Relationships>'
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(content_overrides)
        + '</Types>'
    )

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        for idx, (_sheet_name, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(rows))


def generate_phase_metrics_summary(
    output_root: str = "output",
    out_filename: str = "phase_metrics_summary.xlsx",
    videos: Optional[Sequence[str]] = None,
) -> str:
    selected_videos = tuple(videos or DEFAULT_VIDEOS)
    results: Dict[str, Dict[str, Any]] = {}
    for video in selected_videos:
        path = os.path.join(output_root, video, "swing_result.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            results[video] = json.load(f)

    sheets = [
        ("chest", _sheet_rows(results, "chest", selected_videos)),
        ("hip", _sheet_rows(results, "hip", selected_videos)),
    ]
    out_path = os.path.join(output_root, out_filename)
    _write_xlsx(out_path, sheets)
    return out_path


def main() -> None:
    out_path = generate_phase_metrics_summary()
    print(out_path)


if __name__ == "__main__":
    main()
