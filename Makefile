REMOTE ?= user@your-dev-machine
REMOTE_DIR ?= ~/gomonova
CFG ?= configs/train_main.yaml
NGPU ?= 4

.PHONY: sync-up sync-down train train-ddp play test backup

sync-up:
	rsync -avz --delete \
		--exclude '.git' --exclude 'data/' --exclude 'checkpoints/' \
		--exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
		./ $(REMOTE):$(REMOTE_DIR)/

sync-down:
	rsync -avz $(REMOTE):$(REMOTE_DIR)/checkpoints/ ./checkpoints/

train:
	ssh $(REMOTE) "cd $(REMOTE_DIR) && python scripts/train.py --config $(CFG)"

train-ddp:
	ssh $(REMOTE) "cd $(REMOTE_DIR) && torchrun --nproc_per_node=$(NGPU) scripts/train.py --config $(CFG)"

play:
	python -m gomonova.cli.play

test:
	python -m pytest tests/ -v

backup:
	bash scripts/backup.sh
