run:
	marimo edit presentation.py --no-token

server:
	python -m http.server 8000 --directory _site

copy:
	cp -r game/ _site/game


train-ssm:
	python game/diffusion.py train --ssm --out game/model.pt --steps 5000 --device cuda

exp-ssm:
	python game/diffusion.py export --ssm --model game/model.pt --out game/model.onnx
