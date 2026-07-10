import json
import struct
import time
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGE_DIR = BASE_DIR / "test_images"
SOURCES_FILE = TEST_IMAGE_DIR / "sources.json"

TEST_IMAGES = [
    {
        "name": "normal_office",
        "expected": "normal",
        "url": "https://media.altphotos.com/cache/images/2017/09/07/03/752/office-open-space.jpg",
        "source_page": "https://altphotos.com/photo/the-interior-of-a-large-open-space-office-2529/",
        "source_title": "The Interior of a Large Open Space Office",
        "license": "CC0",
        "ext": ".jpg",
    },
    {
        "name": "fire_or_smoke",
        "expected": "abnormal_fire_smoke",
        "url": "https://live-production.wcms.abc-cdn.net.au/adb531f8931e02a16e6dbe9f75edee97?cropH=472&cropW=840&height=485&impolicy=wcms_crop_resize&width=862&xPos=0&yPos=0",
        "source_page": "https://www.abc.net.au/news/2013-01-09/an-fiji-fire-authorities-reiterate-safety-message/4457718",
        "source_title": "ABC News house fire image",
        "license": "Source page terms apply",
        "ext": ".jpg",
    },
    {
        "name": "flooded_room",
        "expected": "abnormal_water",
        "url": "https://static.boredpanda.com/blog/wp-content/uploads/2025/05/681c47d02df26_h0c8bked9fzc1-jpeg__700.jpg",
        "source_page": "https://www.boredpanda.com/best-surreal-liminal-spaces/",
        "source_title": "Flooded room image",
        "license": "Source page terms apply",
        "ext": ".jpg",
    },
    {
        "name": "person_on_floor",
        "expected": "abnormal_person_fall",
        "url": "https://source.roboflow.com/EbPANzricqVcPCiQMmLY7oljIY92/00vZM15IA50LLEpXPmw2/thumb.jpg",
        "source_page": "https://universe.roboflow.com/humna-pose-data/falling-pose-estimation",
        "source_title": "Roboflow falling pose estimation example",
        "license": "CC BY 4.0",
        "ext": ".jpg",
    },
    {
        "name": "window_closed",
        "expected": "normal_window_closed",
        "url": "",
        "generator": "closed_window",
        "source_page": "generated locally",
        "source_title": "Generated closed-window control image",
        "license": "Generated local test image",
        "ext": ".bmp",
    },
    {
        "name": "window_open",
        "expected": "abnormal_window_open",
        "url": "https://ecochoicewindows.ca/wp-content/uploads/2023/09/a-room-with-an-open-window-2021-10-23-20-49-42-utc-scaled-1-1200x800.jpg",
        "source_page": "https://ecochoicewindows.ca/fixed-or-operable-windows-which-should-you-choose/",
        "source_title": "Room with open operable window",
        "license": "Source page terms apply",
        "ext": ".jpg",
    },
    {
        "name": "blocked_fire_exit",
        "expected": "abnormal_blocked_exit",
        "url": "https://vestafire.co.uk/wp-content/uploads/2025/04/Blocked-Fire-Escape.png",
        "source_page": "https://vestafire.co.uk/common-fire-hazards-in-small-businesses-what-you-need-to-know/",
        "source_title": "Blocked fire exit",
        "license": "Source page terms apply",
        "ext": ".png",
    },
    {
        "name": "overloaded_socket",
        "expected": "abnormal_electrical_risk",
        "url": "https://vestafire.co.uk/wp-content/uploads/2025/04/Overloaded-Socket.png",
        "source_page": "https://vestafire.co.uk/common-fire-hazards-in-small-businesses-what-you-need-to-know/",
        "source_title": "Overloaded electrical socket",
        "license": "Source page terms apply",
        "ext": ".png",
    },
    {
        "name": "unmaintained_fire_equipment",
        "expected": "abnormal_safety_equipment",
        "url": "https://vestafire.co.uk/wp-content/uploads/2025/04/Unmaintained-Fire-Equipment.png",
        "source_page": "https://vestafire.co.uk/common-fire-hazards-in-small-businesses-what-you-need-to-know/",
        "source_title": "Unmaintained fire safety equipment",
        "license": "Source page terms apply",
        "ext": ".png",
    },
    {
        "name": "broken_window_glass",
        "expected": "abnormal_broken_window",
        "url": "https://blogs.loc.gov/thesignal/files/2012/11/window05-225x300.jpg",
        "source_page": "https://blogs.loc.gov/thesignal/2012/11/when-data-loss-is-personal/",
        "source_title": "Broken window",
        "license": "Library of Congress blog terms apply",
        "ext": ".jpg",
    },
    {
        "name": "camera_night_vision_issue",
        "expected": "abnormal_camera_view",
        "url": "",
        "generator": "camera_view_issue",
        "source_page": "generated locally",
        "source_title": "Generated camera blocked/overexposed control image",
        "license": "Generated local test image",
        "ext": ".bmp",
    },
]


def download_file(url, output_path):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AutoSceneMonitorTest/1.0 (local classroom experiment)"
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())


def write_bmp(output_path, width, height, background, rectangles):
    pixels = [[list(background) for _ in range(width)] for _ in range(height)]
    for x1, y1, x2, y2, color in rectangles:
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        for y in range(y1, y2):
            row = pixels[y]
            for x in range(x1, x2):
                row[x] = list(color)

    row_size = (width * 3 + 3) & ~3
    image_size = row_size * height
    file_size = 54 + image_size

    header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
    dib = struct.pack(
        "<IIIHHIIIIII",
        40,
        width,
        height,
        1,
        24,
        0,
        image_size,
        2835,
        2835,
        0,
        0,
    )

    with open(output_path, "wb") as file:
        file.write(header)
        file.write(dib)
        padding = b"\x00" * (row_size - width * 3)
        for y in range(height - 1, -1, -1):
            for red, green, blue in pixels[y]:
                file.write(bytes((blue, green, red)))
            file.write(padding)


def generate_synthetic_image(kind, output_path):
    if kind == "closed_window":
        rectangles = [
            (0, 0, 640, 480, (235, 238, 235)),
            (110, 70, 530, 390, (245, 245, 240)),
            (135, 95, 505, 365, (70, 70, 70)),
            (155, 115, 315, 345, (150, 190, 220)),
            (325, 115, 485, 345, (150, 190, 220)),
            (316, 95, 324, 365, (70, 70, 70)),
            (148, 108, 492, 122, (230, 240, 250)),
            (470, 225, 486, 245, (30, 30, 30)),
            (140, 370, 500, 392, (210, 210, 205)),
        ]
        write_bmp(output_path, 640, 480, (235, 238, 235), rectangles)
        return

    if kind == "camera_view_issue":
        rectangles = [
            (0, 0, 640, 480, (12, 12, 12)),
            (300, 0, 640, 480, (190, 190, 190)),
            (350, 35, 610, 420, (115, 115, 115)),
            (430, 150, 620, 330, (245, 245, 245)),
            (0, 360, 640, 480, (35, 35, 35)),
            (35, 385, 105, 455, (70, 70, 70)),
            (125, 385, 195, 455, (70, 70, 70)),
            (215, 385, 285, 455, (70, 70, 70)),
            (305, 385, 375, 455, (70, 70, 70)),
            (400, 24, 610, 52, (230, 230, 230)),
        ]
        write_bmp(output_path, 640, 480, (12, 12, 12), rectangles)
        return

    raise ValueError(f"未知生成器：{kind}")


def main():
    TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    sources = []

    for item in TEST_IMAGES:
        output_path = TEST_IMAGE_DIR / f"{item['name']}{item['ext']}"
        print(f"下载测试图片：{item['name']} -> {output_path}")
        download_status = "downloaded"
        download_error = ""
        if item.get("generator"):
            generate_synthetic_image(item["generator"], output_path)
            download_status = "generated"
        else:
            try:
                download_file(item["url"], output_path)
            except Exception as exc:
                download_error = str(exc)
                if output_path.exists():
                    download_status = "kept_existing_after_error"
                    print(f"下载失败，保留本地已有图片：{item['name']}，错误：{download_error}")
                else:
                    print(f"下载失败，跳过图片：{item['name']}，错误：{download_error}")
                    continue

        source_record = {
            "name": item["name"],
            "expected": item["expected"],
            "file": str(output_path.relative_to(BASE_DIR)),
            "source_title": item["source_title"],
            "source_page": item["source_page"],
            "download_url": item["url"],
            "license": item["license"],
            "download_status": download_status,
        }
        if download_error:
            source_record["download_error"] = download_error
        sources.append(source_record)
        time.sleep(1)

    SOURCES_FILE.write_text(
        json.dumps(sources, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成，图片保存在：{TEST_IMAGE_DIR}")
    print(f"来源记录保存在：{SOURCES_FILE}")


if __name__ == "__main__":
    main()
