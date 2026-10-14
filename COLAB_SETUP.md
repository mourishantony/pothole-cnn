# Run the website in Google Colab (free T4 GPU, weights only, no training)

Notebook: [`colab_pothole_detection.ipynb`](colab_pothole_detection.ipynb): starts the Flask website from this project
in Colab using the pretrained weights `final.keras`. You do not need a GPU on your own laptop.

> Status: the notebook's code was syntax-checked and the website and analysis code were tested locally, but the notebook has
> **not been run end to end in Colab yet**. If a cell fails, see [Troubleshooting](#troubleshooting).

## What you need
1. A Google account (for Colab).
2. The weights file **`final.keras`** (about 83 MB). You get it from either:
   - your project folder: `pothole-cnn/models/final.keras`, or
   - GitHub: open the repo > `models` > `final.keras` > **Download raw file**.
3. Access to the code. The repo `mourishantony/pothole-cnn` is **private**, so pick one:
   - **A (recommended):** a read-only GitHub token (Step 1), or
   - **B:** make the repo public (GitHub > Settings > General > Danger Zone > Change visibility), then no token is needed, or
   - **C:** no GitHub access in Colab: zip the project folder **without** `.venv`, `data`, `web_runs`, `external`, upload the zip to Colab's Files panel, and run `!unzip -q pothole-cnn.zip -d /content` in a new cell. The clone step is then skipped automatically.
4. A road video to test (`.mp4`, `.avi`, `.mov`, `.mkv` or `.webm`). Start with 10-30 seconds.

## Step-by-step

### Step 1: Create a read-only GitHub token (skip for option B or C)
1. GitHub > profile picture > **Settings** > **Developer settings** > **Personal access tokens** > **Fine-grained tokens** > **Generate new token**.
2. Name: `colab-readonly`. Expiry: short (for example 7 days).
3. **Repository access:** *Only select repositories* > `pothole-cnn`.
4. **Repository permissions > Contents:** **Read-only**.
5. **Generate token** and copy it. You paste it into Colab in Step 5.
6. Keep it private; never paste it into a notebook cell or commit it. Delete the token on GitHub when finished.

### Step 2: Open the notebook in Colab
- **Upload:** <https://colab.research.google.com> > **File > Upload notebook** > choose `colab_pothole_detection.ipynb`, or
- **From GitHub:** **File > Open notebook > GitHub** tab > sign in to GitHub when asked > select `mourishantony/pothole-cnn` > open the notebook.

### Step 3: Turn on the T4 GPU
**Runtime > Change runtime type** > Hardware accelerator **T4 GPU** > **Save** > **Connect**.

### Step 4: Check the GPU (notebook section 1)
Run the cell (**Shift+Enter**). Expected: `GPU 0: Tesla T4` and a GPU in the `GPUs: [...]` list.
If it stops with "No GPU", repeat Step 3.

### Step 5: Get the project code (section 2)
Run the cell. It asks for **GitHub token**: paste the token from Step 1 and press Enter (hidden input). For a public repo, press Enter.
It clones the project into `/content/pothole-cnn` and removes the token from the saved git settings.
If `/content/pothole-cnn` already exists (option C), it skips cloning.

### Step 6: Add the weights file `final.keras` (section 3)
The model must end up at exactly:

```
/content/pothole-cnn/models/final.keras
```

Section 3 first checks whether the file is already there (the clone includes it). If it is, it prints `Weights found: 83 MB` and you are done.
If not, add it with **either** way:

**Way 1: drag and drop in the Files panel**
1. Click the **folder icon** in Colab's left sidebar (Files).
2. Open `pothole-cnn` > `models` (you should see `classes.json` there).
3. Drag `final.keras` from your computer into the `models` folder (or right-click the folder > **Upload**).
4. Wait until the upload finishes. The orange circle by the file must disappear; the file is about 83 MB.
5. Run the section 3 cell. It should print `Ready: /content/pothole-cnn/models/final.keras`.

**Way 2: upload from the cell**
1. Run the section 3 cell. When it says "Weights not found", click **Choose files**.
2. Select `final.keras`. The cell moves it into `models/` for you and prints `Ready: ...`.

Notes:
- The file name must be exactly `final.keras`.
- Files in `/content` are deleted when the session ends, so repeat this step in each new session (or keep the file on Google Drive and copy it).

### Step 7: Install the extra packages (section 4)
Run the cell. It installs only Flask and ffmpeg. Colab already provides TensorFlow and OpenCV.

### Step 8: Load the model (section 5)
Run the cell. Expected: `Model loaded. Classes: ['crack', 'normal', 'pothole']`.
If you get an `input_axes` / "could not be deserialized" error: new cell with `!pip install -U keras`, then **Runtime > Restart session**, and run again from Step 4.

### Step 9: Start the website (section 6)
Run the cell. The web page appears **below the cell**.
1. Click **Choose File** and pick your road video.
2. Click **Analyse video** and wait. The page shows "Analysing...".
3. You get a verdict (**Pothole detected** / **No pothole detected**), frame counts, the annotated video (red = pothole, orange = crack) and the strongest pothole frame.

Notes:
- "Pothole detected" requires at least 3 sampled frames with a pothole.
- Results are decision support only; a person should review them.
- Keep the cell running while you use the site. To restart the site, use **Runtime > Restart session** and run again from Step 4 (the port stays busy otherwise).

### Step 10: Stop Colab when finished
**Runtime > Disconnect and delete runtime**, so you do not use up your free GPU quota. Download anything you want to keep first.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No GPU` | Runtime > Change runtime type > T4 GPU, reconnect, re-run from Step 4. |
| `git clone` fails ("Repository not found" / 403) | The repo is private. Paste a valid token with **Contents: read** on `pothole-cnn`, make the repo public, or use option C. Tokens expire. |
| `Weights not found` / `final.keras is missing or too small` | Add the file as in Step 6. It must be about 83 MB and at `/content/pothole-cnn/models/final.keras`. A partly uploaded file is too small; wait for the upload to finish. |
| `could not be deserialized ... input_axes` | `!pip install -U keras`, then restart the session (Step 8). |
| Page stays on "Analysing..." for a long time or shows an error | Use a shorter video (10-30 s). If it still fails, use the **Fallback** cell at the bottom of the notebook, which analyses a video without the web page. |
| Page does not appear or port 5000 is busy | Runtime > Restart session, then run from Step 4. |
| Output video does not play | Use the Fallback cell and download the video, or use a different browser. |

## Speed
A T4 should be much faster than CPU-only. On the development laptop, a 30-second clip took about 1 minute on an RTX 2050 and about 8.4 minutes on CPU.
I have not timed a T4 in Colab, so I cannot give a T4 number.
