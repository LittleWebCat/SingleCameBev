#!/usr/bin/env bash
# Install only the project DDS sysctl profile and apply it immediately.
set -euo pipefail

if (( EUID != 0 )); then
  echo 'Run once with administrator access: sudo bash tools/setup_dds_transport.sh' >&2
  exit 1
fi

bev_dds_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bev_dds_profile="$bev_dds_project_dir/config/transport/dds_sysctl.conf"
bev_dds_destination='/etc/sysctl.d/90-camera-lidar-bev.conf'
bev_dds_marker='# camera_lidar_bev managed DDS transport settings'

if [[ -L "$bev_dds_destination" ]]; then
  echo "Refusing to replace a symlink: $bev_dds_destination" >&2
  exit 1
fi
if [[ -e "$bev_dds_destination" ]]; then
  IFS= read -r bev_dds_existing_marker < "$bev_dds_destination"
  if [[ "$bev_dds_existing_marker" != "$bev_dds_marker" ]]; then
    echo "Existing file is not managed by this project: $bev_dds_destination" >&2
    exit 1
  fi
fi

declare -A bev_dds_values
while IFS='=' read -r bev_dds_name bev_dds_requested; do
  bev_dds_name="${bev_dds_name//[[:space:]]/}"
  bev_dds_requested="${bev_dds_requested//[[:space:]]/}"
  [[ -z "$bev_dds_name" || "$bev_dds_name" == \#* ]] && continue
  case "$bev_dds_name" in
    net.core.rmem_max|net.core.rmem_default|net.ipv4.ipfrag_time|net.ipv4.ipfrag_high_thresh) ;;
    *) echo "Unexpected setting in $bev_dds_profile: $bev_dds_name" >&2; exit 1 ;;
  esac
  [[ "$bev_dds_requested" =~ ^[0-9]+$ ]] || { echo 'Invalid requested sysctl value' >&2; exit 1; }
  bev_dds_current="$(sysctl -n "$bev_dds_name")"
  [[ "$bev_dds_current" =~ ^[0-9]+$ ]] || { echo "Invalid current value for $bev_dds_name" >&2; exit 1; }
  if [[ "$bev_dds_name" == net.ipv4.ipfrag_time ]]; then
    (( bev_dds_current < bev_dds_requested )) && bev_dds_requested="$bev_dds_current"
  else
    (( bev_dds_current > bev_dds_requested )) && bev_dds_requested="$bev_dds_current"
  fi
  bev_dds_values["$bev_dds_name"]="$bev_dds_requested"
done < "$bev_dds_profile"

# Keep the maximum at least as large as the selected default.
if (( bev_dds_values[net.core.rmem_max] < bev_dds_values[net.core.rmem_default] )); then
  bev_dds_values[net.core.rmem_max]="${bev_dds_values[net.core.rmem_default]}"
fi
bev_dds_temporary="$(mktemp /tmp/camera-lidar-bev-sysctl.XXXXXXXX)"
trap 'rm -f -- "$bev_dds_temporary"' EXIT
{
  echo "$bev_dds_marker"
  echo '# Installed by tools/setup_dds_transport.sh; larger active limits are retained.'
  for bev_dds_name in net.core.rmem_max net.core.rmem_default net.ipv4.ipfrag_time net.ipv4.ipfrag_high_thresh; do
    echo "$bev_dds_name = ${bev_dds_values[$bev_dds_name]}"
  done
} > "$bev_dds_temporary"
install -o root -g root -m 0644 "$bev_dds_temporary" "$bev_dds_destination"
sysctl -p "$bev_dds_destination"
echo "DDS settings applied now and saved for reboot: $bev_dds_destination"
