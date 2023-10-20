# CoreRec

<p align="center">
<img src="RS_img.webp" width="800" height = "400">
</p>
<!-- <iframe src="https://giphy.com/embed/1hnRFNYcL8OpKABcVs" width="480" height="202" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/forest-woods-dusk-1hnRFNYcL8OpKABcVs">via GIPHY</a></p> -->

## installation

```bash
git clone https://github.com/Jackie-Ju/Develop.git
cd Develop
conda create -n RecCore python=3.8.13
pip install -r requirements.txt
```

## Run a simple recommender system
### Run with full training data

```bash
python recommender_run.py --method=Full --model=ease\
                          --dataset=ml100k --device=cuda:0 --batch_size=512 --epochs=200 --seed=2020
```
### Run with coreset
```bash
python recommender_run.py --method=KCore --coreset_path=KCore_repeat1_0.5\
                          --model=ease --dataset=ml100k --device=cuda:0 --batch_size=512 --epochs=200
```
See completed configurations in [recsys_parser.py](reccore/utils/recsys_parser.py).

## Run a simple coreset selection task
Below is an example of selecting a coreset of 10% of users from ml100k using Herding with BPR as the proxy model.

```bash
python selection_run.py --coreset_size=0.1 --method=herding\
                        --model=bpr --dataset=ml100k --device=cuda:0 --batch_size=512 --epochs=200 --seed=2020
```
See completed configurations in [selection_parser.py](reccore/utils/selection_parser.py)
