#!/usr/bin/env bash

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --container)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --image)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --xauth)
      XAUTH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--model vista] [--container name] [--image tag]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# --- Rest of your script ---
SESSION="ros_workspace"

# Define commands using the parsed arguments
CMD1="docker run -it --rm --network=host \
  -v /dev:/dev \
  --privileged \
  --name ${CONTAINER_NAME}_2 \
  --device-cgroup-rule='a *:* rmw' \
  --volume=/tmp/.X11-unix:/tmp/.X11-unix -v ${XAUTH}:${XAUTH} \
  -e XAUTHORITY=${XAUTH} \
  --runtime nvidia --gpus all \
  -v ${PWD}:/workspace \
  -w=/workspace \
  -e LIBGL_ALWAYS_SOFTWARE='1' \
  -e DISPLAY=${DISPLAY} \
  ${IMAGE_TAG} \
  bash --rcfile /workspace/init_project.sh"

CMD2="docker run --pull always -it \
  --net=host \
  --ipc=host \
  --env='DISPLAY=$DISPLAY' \
  --env='QT_X11_NO_MITSHM=1' \
  --volume='/tmp/.X11-unix:/tmp/.X11-unix:rw' \
  --device=/dev/dri:/dev/dri \
  maevaguerrier/vista_gazebo_limo:dev \
  bash -ic 'sim; exec bash'"

# Launch tmux
tmux new-session -d -s "$SESSION" bash -c "$CMD1"
tmux split-window -h -t "$SESSION" bash -c "$CMD2"
tmux attach-session -t "$SESSION"