.PHONY: help install inference clean docker-build docker-run

help:
	@echo "Lệnh khả dụng trong dự án Telecom Azimuth ML:"
	@echo "  make install       Cài đặt tất cả phụ thuộc Python từ requirements.txt"
	@echo "  make inference     Thực thi pipeline tính góc phương vị"
	@echo "  make clean         Dọn dẹp tệp rác, cache và __pycache__"
	@echo "  make docker-build  Đóng gói Docker Image"
	@echo "  make docker-run    Chạy container bằng Docker Compose"

install:
	pip install -r requirements.txt

inference:
	python3 entrypoint/inference.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t telecom-azimuth:latest .

docker-run:
	docker-compose up -d
