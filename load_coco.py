#%%
from transformers import AutoImageProcessor, ViTMAEModel
from PIL import Image
import requests
import torchvision

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
# %%

image1 = image_processor(images=image, return_tensors="pt")
# %%
model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
# %%
outputs = model(**image1)
# %%
last_hidden_states = outputs.last_hidden_state
# %%
