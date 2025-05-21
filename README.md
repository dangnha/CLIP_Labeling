# Introduction
This repository is project apply CLIP model for multi-modal downstream task, with 3 main features: zero-shot image classification, image captioning, image retrieval. 

# Installation

Clone repo

```
git clone https://github.com/dangnha/CLIP_Labeling.git
```

Dependencies

```
pip install -r requirements.txt
```

Run the app

```
python app.py
```

# Key features
* Zero-shot Image Classification (custemable classes)
* Image captioning  
    * Change the caption 1 step verified
    * Store in database (with label)
* Image Retrieval
    * Image-to-Image
    * Text-to-Image
* Database Manage
    * Store image with caption, label.
    * Import folder unlabled image.
    * Export CSV file with image path, label, caption.
    * Search match string by label, caption.

# Demo UI
## Home page
![Image](https://github.com/dangnha/CLIP_Labeling/blob/master/demo/b638cdca-c77a-49ed-adda-064c55bcd3e8.jpg)
## Zero-shot Classification
![Image](https://github.com/dangnha/CLIP_Labeling/blob/master/demo/b6f2f751-e60a-4805-a94b-4d38b93bf161.jpg)
## Image Captioning
![Image](https://github.com/dangnha/CLIP_Labeling/blob/master/demo/93a575e1-4fbe-49a8-8a38-f741ac6d3466.jpg)
## Image Retrieval
### Text-to-Image
![Image](https://github.com/dangnha/CLIP_Labeling/blob/master/demo/fa3e943a-1873-4406-80f7-592aba97265b.jpg)
### Image-to-Image
![Image](https://github.com/dangnha/CLIP_Labeling/blob/master/demo/7335af0e-49a5-4b19-bec9-1c10fd6274a3.jpg)
# Database Management
![Image](https://github.com/dangnha/CLIP_Labeling/blob/master/demo/50fbdcf6-e974-4120-a3ff-d259c9458674.jpg)
![Image](https://github.com/dangnha/CLIP_Labeling/blob/master/demo/0b940d49-e851-46f8-bce1-1ae19ea74f09.jpg)


