import argparse
import tinker
import urllib.request
from dotenv import load_dotenv
load_dotenv()


def download_tinker_weights(unique_id: str, out_file: str = "archive.tar") -> None:
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    tinker_path = f"tinker://{unique_id}/sampler_weights/final"
    future = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    checkpoint_archive_url_response = future.result()

    # checkpoint_archive_url_response.url is a signed URL that can be downloaded
    # until checkpoint_archive_url_response.expires
    urllib.request.urlretrieve(checkpoint_archive_url_response.url, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a checkpoint archive from a Tinker path using a unique id"
    )
    parser.add_argument("--unique-id", "-u", required=True, help="Unique id used in the tinker path")
    parser.add_argument("--output", "-o", default="archive.tar", help="Output filename to save the archive")
    args = parser.parse_args()

    download_tinker_weights(args.unique_id, args.output)