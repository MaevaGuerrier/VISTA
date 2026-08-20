#!/bin/bash

git config --global --add safe.directory /workspace

# Use argument
target_dir="/workspace/.packages_vista"

pip install /workspace/src/train --target="$target_dir" --upgrade

nohist() {
    (cd /workspace/src/deployment/src && echo $@ && python3 navigate_vista_without_history.py $@)
}

vista() {
    (cd /workspace/src/deployment/src && python3 navigate_vista_with_history.py $@)
}

# arg $1 is name of bag and arg $2 is name of directory to save topomap in
topo() {
    (cd /workspace/src/deployment/src && ./create_topomap.sh "$@")
}


bag() {
    local robot="limo"
    local env="unknown"
    local trial="1"
    local topics=(
        "/depth/image"
        "/depth/camera_info"
    )

    # Parse key:=value style args
    for arg in "$@"; do
        case "$arg" in
            robot:=*) robot="${arg#robot:=}" ;;
            env:=*)   env="${arg#env:=}" ;;
            trial:=*) trial="${arg#trial:=}" ;;
            *)
                echo "Unknown argument: $arg"
                return 1
                ;;
        esac
    done

    local bag_output="/workspace/src/deployment/topomaps/bags/${robot}_${env}_trial_${trial}"

    echo "Recording to: $bag_output"
    echo "Topics: ${topics[*]}"

    ros2 bag record -o "$bag_output" "${topics[@]}"
}