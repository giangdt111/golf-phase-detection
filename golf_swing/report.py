import os
import zipfile
from bisect import bisect_left
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape


PHASE_COLUMNS = [f"P{i}" for i in range(1, 10)]
TRANSLATION_PARTS = ("hip", "chest", "grip")
ROTATION_PARTS = ("hip", "chest")
TRANSLATION_METRICS = (
    ("X shift", "x_in"),
    ("Y shift", "y_in"),
    ("Z shift", "z_in"),
)
ROTATION_METRICS = (
    ("RX", "x_deg"),
    ("RY", "y_deg"),
    ("RZ", "z_deg"),
)


def _video_name(combined_result: Dict[str, Any]) -> str:
    path = str((combined_result.get("views", {}).get("face_on") or {}).get("video_path") or "")
    if path:
        return os.path.splitext(os.path.basename(path))[0]
    return "video"


def _combined_frame_lookup(combined_result: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    lookup = {
        int(frame.get("face_on_frame", 0)): frame
        for frame in combined_result.get("frames", []) or []
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


def _phase_frames(combined_result: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    lookup, frame_ids = _combined_frame_lookup(combined_result)
    frames: List[Optional[Dict[str, Any]]] = [None] * 9
    for pair in combined_result.get("phase_frames", []) or []:
        idx = int(pair.get("phase_index", 0))
        if idx < 1 or idx > 9:
            continue
        face_frame = int((pair.get("face_on") or {}).get("frame", 0))
        frames[idx - 1] = _nearest_frame(lookup, frame_ids, face_frame)
    return frames


def _metric_value(
    frame: Optional[Dict[str, Any]],
    group: str,
    part: str,
    key: str,
) -> Optional[float]:
    if not frame:
        return None
    value = (((frame.get(group) or {}).get(part) or {}).get(key))
    if value is None:
        return None
    return round(float(value), 1)


def _translations_rows(combined_result: Dict[str, Any], name: str) -> List[List[Any]]:
    frames = _phase_frames(combined_result)
    rows: List[List[Any]] = [["Video name", "Part", "Metric", "Unit", *PHASE_COLUMNS]]
    for part in TRANSLATION_PARTS:
        for label, key in TRANSLATION_METRICS:
            rows.append(
                [
                    name,
                    part,
                    label,
                    "in",
                    *[_metric_value(frame, "translations", part, key) for frame in frames],
                ]
            )
    return rows


def _rotations_rows(combined_result: Dict[str, Any], name: str) -> List[List[Any]]:
    frames = _phase_frames(combined_result)
    rows: List[List[Any]] = [["Video name", "Part", "Metric", "Unit", *PHASE_COLUMNS]]
    for part in ROTATION_PARTS:
        for label, key in ROTATION_METRICS:
            rows.append(
                [
                    name,
                    part,
                    label,
                    "deg",
                    *[_metric_value(frame, "rotations", part, key) for frame in frames],
                ]
            )
    return rows


def _phase_meta_rows(combined_result: Dict[str, Any], name: str) -> List[List[Any]]:
    rows: List[List[Any]] = [
        ["Video name", "Phase", "Phase name", "Face-on frame", "Face-on time", "DTL frame", "DTL time"]
    ]
    for pair in combined_result.get("phase_frames", []) or []:
        idx = int(pair.get("phase_index", 0))
        face = pair.get("face_on") or {}
        dtl = pair.get("down_the_line") or {}
        rows.append(
            [
                name,
                f"P{idx}",
                pair.get("phase", ""),
                int(face.get("frame", 0)),
                float(face.get("t", 0.0)),
                int(dtl.get("frame", 0)) if dtl.get("frame") is not None else None,
                float(dtl.get("t", 0.0)) if dtl.get("t") is not None else None,
            ]
        )
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


def export_phase_metrics_workbook(
    combined_result: Dict[str, Any],
    output_path: str,
    video_name: Optional[str] = None,
) -> str:
    """Export phase metrics shown on overlay to a simple .xlsx workbook."""
    name = video_name or _video_name(combined_result)
    sheets = [
        ("translations", _translations_rows(combined_result, name)),
        ("rotations", _rotations_rows(combined_result, name)),
        ("phase_meta", _phase_meta_rows(combined_result, name)),
    ]
    _write_xlsx(output_path, sheets)
    return output_path
