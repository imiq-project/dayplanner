#! /bin/bash

set -e

DATA_COMMIT_HASH=eda6742fabe56f284dcc22f02713944099bb1e95

download_if_not_exists() {
    local url="$1"
    local file="$2"

    if [[ -f "$file" ]]; then
        echo "File already exists: $file"
        return 0
    fi

    echo "Downloading $file"
    curl -L "$url" -o "$file"
}

download_if_not_exists \
  "https://github.com/imiq-project/data/raw/$DATA_COMMIT_HASH/magdeburg.osm.pbf" \
  "/cache/magdeburg.osm.pbf"

download_if_not_exists \
  "https://github.com/imiq-project/data/raw/$DATA_COMMIT_HASH/magdeburg_gtfs.zip" \
  "/cache/magdeburg_gtfs.zip"

java -jar graphhopper.jar server config.yml || true

# if the 'server' command returned, there was an error
echo "Graph cache might be corrupted, removing"
rm -rf /cache/graph

# try again
java -jar graphhopper.jar server config.yml
