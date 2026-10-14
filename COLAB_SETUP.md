# Run the project on Google Colab (free T4 GPU)

Notebook: [`colab_pothole_detection.ipynb`](colab_pothole_detection.ipynb)
Goal: run the pretrained pothole/crack model on your own road video, using Colab's T4 GPU. You do not need a GPU on your own laptop.

> Status: the notebook's code was syntax-checked and the core pipeline (`app.process`) was tested locally, but the notebook
> itself has **not been run end to end in Colab yet**. If a cell fails, see [Troubleshooting](#troubleshooting).

## What you need
- A Google account (for Colab).
- Access to the GitHub repo `mourishantony/pothole-cnn`. **The repo is private**, so pick one:
  - **Option A (recommended):** a read-only GitHub token (Step 1 below), or
  - **Option B:** make the repo public in GitHub (Settings > General > Danger Zone > Change visibility). Then no token is needed.
- A road video (`.mp4`, `.avi`, `.mov`, `.mkv` or `.webm`). Start with a 10-30 second clip.

## Step-by-step

### Step 1: Create a read-only GitHub token (skip if the repo is public)
1. GitHub > your profile picture > **Settings** > **Developer settings** > **Personal access tokens** > **Fine-grained tokens** > **Generate new token**.
2. Name it `colab-readonly`, set a short expiry (for example 7 days).
3. **Repository access:** *Only select repositories* > choose `pothole-cnn`.
4. **Permissions > Repository permissions > Contents:** set to **Read-only**.
5. Click **Generate token** and copy it. You will paste it into Colab in Step 5.
6. Keep it private. Never paste it into the notebook cell itself or commit it. Delete the token on GitHub when you are done.

### Step 2: Open the notebook in Colab
Pick one:
- **Upload:** go to <https://colab.research.google.com> > **File > Upload notebook** > choose `colab_pothole_detection.ipynb` from your computer.
- **From GitHub:** Colab > **File > Open notebook > GitHub** tab > sign in to GitHub when asked (this lets Colab see private repos) > select `mourishantony/pothole-cnn` > open `colab_pothole_detection.ipynb`.

### Step 3: Turn on the T4 GPU
1. **Runtime > Change runtime type**.
2. Hardware accelerator: **T4 GPU** > **Save**.
3. Click **Connect** (top right) if it is not already connected.

### Step 4: Check the GPU (notebook section 1)
Run the cell (click it, press **Shift+Enter**). Expected output includes `GPU 0: Tesla T4` and `GPUs: [PhysicalDevice(... 'GPU:0' ...)]`.
If it stops with "No GPU found", repeat Step 3 and run the cell again.

### Step 5: Download the code and weights (section 2)
1. Run the cell. It asks for **GitHub token**: paste the token from Step 1 and press Enter (input is hidden). If the repo is public, just press Enter.
2. It clones the repo and prints `models/final.keras = 83 MB`.
   The token is removed from the saved git settings right after cloning.

### Step 6: Install extra packages (section 3)
Run the cell. It installs only Flask and ffmpeg. Colab already provides TensorFlow and OpenCV.

### Step 7: Load the model (section 4)
Run the cell. Expected: `Model loaded. Classes: ['crack', 'normal', 'pothole']`.
If you get an `input_axes` / "could not be deserialized" error: run `!pip install -U keras` in a new cell, then **Runtime > Restart session**, and run again from Step 4.

### Step 8: Upload your video (section 5)
Run the cell, click **Choose files**, and select your video. Wait for the upload to reach 100%.

### Step 9: Analyse it (section 6)
Run the cell. When it finishes you get:
- a verdict line: `==> POTHOLE DETECTED` or `==> No pothole detected`,
- counts: frames analysed, frames with a pothole, frames with a crack,
- the annotated video (red boxes = pothole, orange = crack),
- the strongest pothole frame.

Notes:
- The verdict is "pothole detected" only if at least 3 sampled frames contain a pothole.
- For long videos, change `every_sec=0.5` to `1.0` or `2.0` in that cell. It is faster but can miss short potholes.
- Results are decision support only: a person should review them.

### Step 10 (optional): Download the annotated video (section 7)
Run the cell. Your browser downloads `annotated.mp4`.

### Step 11 (optional): Use the website inside Colab (section 8)
Run the cell. The web page appears under the cell. Upload a video there and click **Analyse video**.
(This is the same site as `python app.py` on a laptop, running on the Colab GPU.)

### Step 12: Stop Colab when you finish
**Runtime > Disconnect and delete runtime**, so you do not use up your free GPU quota. Files in `/content` are deleted when the session ends,
so download anything you want to keep.

## Optional: retrain on the T4
The last cells of the notebook retrain the model from scratch. Only do this if you need to.
- Needs a Kaggle token (kaggle.com > Settings > API > **Create New Token** > it downloads `kaggle.json`). You upload it in notebook step B.
- Downloads about 11 GB and takes hours. Colab can disconnect, so run the Google Drive cell (A) first and keep the copied checkpoints.
- Run the cells in order: A (Drive), B (Kaggle token), C (data), D (train stage 1, build `processed_v3`, train stage 2, copy to Drive).
- The final cell strips the optimizer state and saves `models/final.keras` for use with Step 7.
- Check both datasets' licenses before commercial use: [Road Damage: Potholes, Cracks and Manholes](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes), [RDD2022](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022).
- Colab has only 2 CPU cores, so data preparation (`prepare_rdd.py`) will be slower than on a desktop.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No GPU found` | Runtime > Change runtime type > T4 GPU, reconnect, re-run from Step 4. |
| `git clone` fails with "Repository not found" or 403 | The repo is private: paste a valid token with **Contents: read** on `pothole-cnn`, or make the repo public. Tokens expire. |
| `models/final.keras` is much smaller than 83 MB | The clone is incomplete or broken. Delete the `pothole-cnn` folder (`!rm -rf pothole-cnn`) and re-run Step 5. |
| `could not be deserialized ... input_axes` | Colab's Keras is older than the one that saved the weights: `!pip install -U keras`, restart the session. |
| Out of memory while analysing | Use a smaller video, or raise `every_sec`. |
| Session disconnects during training | Re-mount Drive, copy back the checkpoint, and resume with `RESUME=<file>.keras` (see `README.md`). |
| Output video does not play | Download it (Step 10) and open it locally. |

## Speed
A T4 should be much faster than CPU-only. On the laptop used for development, a 30-second clip took about 1 minute on an RTX 2050 and about 8.4 minutes on CPU.
I have not timed a T4 in Colab, so I cannot give a T4 number.
