#!/usr/bin/env python3
"""Replace file-linked images in a DOCX with embedded package members."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET
import zipfile


REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
MAX_REPORT_IMAGE_WIDTH_EMU = 5_650_000  # about 157 mm; safe within A4 text area


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()

    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    rel_name = "word/_rels/document.xml.rels"
    rel_root = ET.fromstring(members[rel_name])
    additions: dict[str, bytes] = {}
    extensions: set[str] = set()
    embedded_relation_ids: set[str] = set()
    image_index = 0
    for relation in rel_root.findall(f"{{{REL_NS}}}Relationship"):
        if relation.get("Type") != IMAGE_REL or relation.get("TargetMode") != "External":
            continue
        target = relation.get("Target")
        if not target:
            raise ValueError("External image relationship lacks a target")
        parsed = urlparse(target)
        image_path = Path(unquote(parsed.path)).resolve()
        suffix = image_path.suffix.lower()
        if suffix not in MIME:
            raise ValueError(f"Unsupported linked image type: {image_path}")
        image_index += 1
        member = f"word/media/image{image_index}{suffix}"
        additions[member] = image_path.read_bytes()
        relation.set("Target", f"media/image{image_index}{suffix}")
        relation.attrib.pop("TargetMode", None)
        relation_id = relation.get("Id")
        if relation_id:
            embedded_relation_ids.add(relation_id)
        extensions.add(suffix.lstrip("."))
    if not additions:
        raise ValueError("No external image relationships were found")
    ET.register_namespace("", REL_NS)
    members[rel_name] = ET.tostring(
        rel_root, encoding="utf-8", xml_declaration=True
    )

    # LibreOffice writes linked DrawingML images as <a:blip r:link="rId...">.
    # Removing TargetMode from the relationship is not sufficient: consumers
    # still treat r:link as external. Convert it to r:embed after packaging the
    # bitmap so Word, WPS and LibreOffice all render the image offline.
    document_name = "word/document.xml"
    document_root = ET.fromstring(members[document_name])
    link_key = f"{{{OFFICE_REL_NS}}}link"
    embed_key = f"{{{OFFICE_REL_NS}}}embed"
    converted_links = 0
    for element in document_root.iter():
        relation_id = element.get(link_key)
        if relation_id in embedded_relation_ids:
            element.attrib.pop(link_key)
            element.set(embed_key, relation_id)
            converted_links += 1
    if converted_links != len(embedded_relation_ids):
        raise ValueError(
            "Embedded image relationship count does not match DrawingML links: "
            f"{len(embedded_relation_ids)} relationships, {converted_links} links"
        )
    # Writer/Web can assign the source bitmap a width far beyond the A4 text
    # area even when HTML width/CSS is present. Clamp DrawingML extents while
    # preserving aspect ratio. Both wp:extent and a:xfrm/a:ext carry the same
    # dimensions and must be updated.
    resized_extents = 0
    for element in document_root.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name not in {"extent", "ext"}:
            continue
        cx_text = element.get("cx")
        cy_text = element.get("cy")
        if not cx_text or not cy_text:
            continue
        cx = int(cx_text)
        cy = int(cy_text)
        if cx <= MAX_REPORT_IMAGE_WIDTH_EMU:
            continue
        element.set("cx", str(MAX_REPORT_IMAGE_WIDTH_EMU))
        element.set("cy", str(max(1, round(cy * MAX_REPORT_IMAGE_WIDTH_EMU / cx))))
        resized_extents += 1
    # Writer may emit VML rather than DrawingML for HTML imports. In that case
    # the HTML conversion already carries the requested width and there are no
    # EMU extents to clamp; embedding the relationships is still sufficient.
    members[document_name] = ET.tostring(
        document_root, encoding="utf-8", xml_declaration=True
    )

    content_name = "[Content_Types].xml"
    content_root = ET.fromstring(members[content_name])
    existing = {
        node.get("Extension", "").lower()
        for node in content_root.findall(f"{{{CONTENT_NS}}}Default")
    }
    for extension in sorted(extensions - existing):
        ET.SubElement(
            content_root,
            f"{{{CONTENT_NS}}}Default",
            Extension=extension,
            ContentType=MIME[f".{extension}"],
        )
    ET.register_namespace("", CONTENT_NS)
    members[content_name] = ET.tostring(
        content_root, encoding="utf-8", xml_declaration=True
    )
    members.update(additions)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    print(f"Embedded {len(additions)} image(s): {output}")


if __name__ == "__main__":
    main()
