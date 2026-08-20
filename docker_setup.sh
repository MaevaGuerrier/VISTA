#!/bin/bash
#version: 0.1
#Author: Anthony Suen, Soma Karthik, Maeva Guerrier
_GREEN='\e[32m'
_NORMAL='\e[0m'
_BOLD='\e[33m'
_RED='\e[31m'
clear

CHOOSE=1
MODEL=1

function CHOOSE_MODEL()
{
    echo -e "${_BOLD}--------------------------${_NORMAL}"
    echo -e "\e[1;10H Choose setup${_NORMAL}"
    echo -e "${_GREEN} 1.VISTA deploy local pc with simulation${_NORMAL}"
    echo -e "${_GREEN} 2.VISTA deploy ROS2 (on robot)${_NORMAL}"
    echo -e "${_BOLD}--------------------------${_NORMAL}"
    echo -n "Your chose(1-2):"
}


function PRINT_MENU()
{
    echo -e "${_BOLD}--------------------------${_NORMAL}"
    echo -e "\e[1;10H Menu${_NORMAL}"
    echo -e "${_GREEN} 1.Build image${_NORMAL}"
    echo -e "${_GREEN} 2.Start Container${_NORMAL}"
    echo -e "${_GREEN} 3.Delete Container${_NORMAL}"
    echo -e "${_GREEN} 4.Backup environment${_NORMAL}"
    echo -e "${_GREEN} 5.Restore environment${_NORMAL}"
    echo -e "${_GREEN} 6.Attach interactive terminal${_NORMAL}"
    echo -e "${_BOLD}--------------------------${_NORMAL}"
    echo -n "Your chose(1-6):"
}

function prepare()
{
    (mv .devcontainer .. &> /dev/null) | echo -n ""
    (mv setup.sh .. &> /dev/null) | echo -n ""
}

function BUILD_IMAGE() {
    Docker_file=.devcontainer/${model_type}
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        BUILDARCH=amd64
    elif [ "$ARCH" = "aarch64" ]; then
        BUILDARCH=arm64
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    docker build --build-arg CACHE_BUST=$(date +%s) --build-arg TARGETARCH=$BUILDARCH ${Docker_file} -t ${image_tag}
}

function start_image()
{

    xhost +local:root
    XAUTH=~/.Xauthority

    if [ "$model_type" == "vista" ]; then
        ./tmux.sh --model "$model_type" --container "$container_name" --image "$image_tag" --xauth "$XAUTH"
    else
    
        docker run -it --rm --network=host \
                    -v /dev:/dev \
                    --privileged \
                    --name ${container_name}_2 \
                    --device-cgroup-rule="a *:* rmw" \
                    --volume=/tmp/.X11-unix:/tmp/.X11-unix -v ${XAUTH}:${XAUTH} \
                    -e XAUTHORITY=${XAUTH} \
                    --runtime nvidia --gpus all \
                    -v ${PWD}:/workspace \
                    -w=/workspace \
                    -e LIBGL_ALWAYS_SOFTWARE="1"\
                    -e DISPLAY=${DISPLAY} \
                    ${image_tag} \
                    bash --rcfile /workspace/init_project.sh

    fi

}

function attach_terminal()
{
    # give docker root user X11 permissions
    # docker exec -it ${container_name} /bin/bash
    docker exec -it ${container_name}_2 /bin/bash 
}

function backup_container()
{
    docker commit ${container_name} ${container_name}:backup | (echo -e "${_RED} A backup is already exist. Use 'docker rmi ${container_name}:backup' and try again.${_NORMAL}" && exit 1)
    echo -e "${_GREEN} Do you want to save the image locally (save as a .tar file)? (Y/N):${_NORMAL}"
    read input
    case $input in
        [yY][eE][sS]|[yY])
            docker save -o ${container_name}_backup.tar ${container_name}:backup
            ;;

        [nN][oO]|[nN])
            ;;

        *)
            echo "Invalid input..."
            exit 1
            ;;
    esac
    echo -e "${_GREEN} Container backup success!${_NORMAL}"

}

function restore_image()
{
    echo -e "${_RED}This operation will overwrite your current backup image and default image. Continue?(y/n):${_NORMAL}"
    read input
    case $input in
        [yY][eE][sS]|[yY])
            docker rmi -f ${image_tag}:dev
            docker rmi -f ${container_name}:backup
            docker load < ${container_name}_backup.tar
            ;;

        [nN][oO]|[nN])
            ;;

        *)
            echo "Invalid input..."
            exit 1
            ;;
    esac
    echo -e "${_GREEN} Container restore success!${_NORMAL}"

}

function check_container()
{
    if [ "$(docker ps -q -f name=${container_name})" ]; then
        delete_container
        echo -e "${_GREEN} Deleting  existing ${container_name}"
    fi
}

function delete_container()
{
    docker rm -f ${container_name}
}



CHOOSE_MODEL

read MODEL

case "${MODEL}" in
    1)
    model_type=vista
    image_tag=vista:dev
    container_name=vista
    ;;
    2)
    model_type=vista_ros2
    image_tag=vista_ros2:dev
    container_name=vista_ros2
    ;;

esac

clear


PRINT_MENU

# prepare

read CHOOSE

case "${CHOOSE}" in
    1)
    BUILD_IMAGE 
    ;;
    2)
    start_image
    ;;
    3)
    delete_container
    ;;
    4)
    backup_container
    ;;
    5)
    restore_image
    ;;
    6)
    attach_terminal
    ;;

esac