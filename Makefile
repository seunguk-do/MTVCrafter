PROJECT_NAME := mtvcrafter
IMAGE_NAME := seunguk/${PROJECT_NAME}
SHM_SIZE := 128gb
PROJECT_ROOT := /app
DATA_DIR := /app/data

build:
	docker build \
		--tag ${IMAGE_NAME}:latest \
		--build-arg USER=$$(whoami) \
		--build-arg UID=$$(id -u) \
		--build-arg GID=$$(id -g) \
		-f Dockerfile .

run:
	docker run \
		-it \
		--rm \
		--gpus all \
		--shm-size ${SHM_SIZE} \
		--workdir="/app" \
		--volume="./${PROJECT_NAME}:/app/${PROJECT_NAME}" \
		--volume="./scripts:/app/scripts" \
		--volume="./data:${DATA_DIR}" \
		-e PROJECT_ROOT=${PROJECT_ROOT} \
		-e DATA_DIR=${DATA_DIR} \
		-e XDG_CACHE_HOME=${DATA_DIR}/cache \
		${IMAGE_NAME}:latest \
		uv run $(filter-out $@,$(MAKECMDGOALS))

claude-run:
	docker run \
		--rm \
		--gpus all \
		--shm-size ${SHM_SIZE} \
		--workdir="/app" \
		--volume="./${PROJECT_NAME}:/app/${PROJECT_NAME}" \
		--volume="./scripts:/app/scripts" \
		--volume="./data:${DATA_DIR}" \
		-e PROJECT_ROOT=${PROJECT_ROOT} \
		-e DATA_DIR=${DATA_DIR} \
		-e XDG_CACHE_HOME=${DATA_DIR}/cache \
		${IMAGE_NAME}:latest \
		uv run $(filter-out $@,$(MAKECMDGOALS))

process-motion-videos:
	$(MAKE) run -- python scripts/process_nlf.py --video-dir ${DATA_DIR}/motion_videos

animate-bogum:
	$(MAKE) run -- python scripts/infer_7b.py \
		--ref-image-path ${DATA_DIR}/ref_images/bogum.jpg \
		--motion-data-dir ${DATA_DIR}/motion_videos_smpl \
		--output-path ${DATA_DIR}/results/


.PHONY: run build claude process-motion-videos

%:
	@:
