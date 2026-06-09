from bing_image_downloader import downloader

keyword = "drone"
limit = 300
output_dir = "drone_images"
adult_filter_off = True
force_replace = False
timeout = 60

try:
    downloader.download(
        keyword,
        limit=limit,
        output_dir=output_dir,
        adult_filter_off=adult_filter_off,
        force_replace=force_replace,
        timeout=timeout
    )
    print(f"Download complete! {limit} drone images saved to {output_dir}")
except Exception as e:
    print(f"Error during download: {e}")
