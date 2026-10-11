"""Draw the film's words as transparent pictures to lay over the video (this ffmpeg has no text filters).
usage: titles.py SPEC.json
SPEC: {"size": [W, H], "font": path to a .ttf, "items": [{"text", "kind": "caption" | "end", "place", "out": PNG}]}
A caption sits in the lower third, bold white with a dark outline; the end card is larger and centred
(or high, or low: "place" is "middle", "top" or "bottom", to keep clear of a face),
its second line (if any) smaller; both carry a soft shadow. Long lines wrap to 80% of the width. Needs Pillow."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

spec = json.loads(Path(sys.argv[1]).read_text())
W, H = spec["size"]


def wrap(draw, text, font, width):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = f"{line} {word}".strip()
            if draw.textlength(test, font=font) <= width or not line:
                line = test
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_block(draw, lines, fonts, centre_y, stroke):
    sizes = [draw.textbbox((0, 0), l, font=f, stroke_width=stroke) for l, f in zip(lines, fonts)]
    heights = [b[3] - b[1] for b in sizes]
    gap = int(fonts[0].size * 0.25)
    y = centre_y - (sum(heights) + gap * (len(lines) - 1)) / 2
    for line, f, b, h in zip(lines, fonts, sizes, heights):
        x = (W - (b[2] - b[0])) / 2 - b[0]
        draw.text((x, y - b[1]), line, font=f, fill=(255, 255, 255, 255), stroke_width=stroke,
                  stroke_fill=(10, 10, 14, 230))
        y += h + gap


for item in spec["items"]:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if item["kind"] == "end":
        big = ImageFont.truetype(spec["font"], round(H * 0.075))
        small = ImageFont.truetype(spec["font"], round(H * 0.05))
        first, *rest = item["text"].split("\n")
        head = wrap(draw, first, big, W * 0.8)
        tail = wrap(draw, " ".join(rest), small, W * 0.8) if rest else []
        y = H * {"top": 0.2, "bottom": 0.72}.get(item.get("place"), 0.5)
        draw_block(draw, head + tail, [big] * len(head) + [small] * len(tail), y, round(H * 0.006))
    else:
        font = ImageFont.truetype(spec["font"], round(H * 0.052))
        lines = wrap(draw, item["text"], font, W * 0.8)
        draw_block(draw, lines, [font] * len(lines), H * 0.85, round(H * 0.005))
    # a soft dark shadow under the words keeps them readable on bright sand and sky
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(img.getchannel("A").point(lambda a: int(a * 0.55)).filter(ImageFilter.GaussianBlur(H * 0.008)))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(shadow, (0, round(H * 0.004)))
    out.alpha_composite(img)
    out.save(item["out"])
