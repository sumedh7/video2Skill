# video2Skill
### Dependencies

- PyTorch 1.7+
- Transformers (Huggingface)
- python-chess
- tslearn
- numba

### Steps to Run 👋

First download YouCook2 data from http://youcook2.eecs.umich.edu/ and place them in folder named `feat_csv`.

Then run `make_feature.py` for both training and testing/validation files to create the feature files as `.npy` dump.

Then run `siam-cook-16.py` for event representation learning. Then run `siam-adapt_undiv.py` to learn the MDP homomorphisms. The models will be stored in `./models`. 

### License
CC-BY-NC 4.0
