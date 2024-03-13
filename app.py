import streamlit as st
import cv2
import onnx
import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

st.set_page_config(layout="wide")
st.title("Fantasy  CycleGAN StreamLit app")
st.divider()

column1,column2,column3 = st.columns(3, gap="large")

with column1:
	st.header("Upload photo")
	st.divider()
	
	file = st.file_uploader("Select an file", type=["jpg", "png"])

	if ('image' not in st.session_state):
		st.session_state['image'] = None

	if (file is not None):
		st.image(file)
		st.write(file.name)
	else:
		st.write("Error reading file!")

with column2:
	st.header("Crop a face")
	st.divider()
	
	scale = st.slider("Scale of cropping", min_value = 0.5, max_value = 2.0, value = 1.2)
	mode = st.radio("Select which type of face close-up would you like to use: ", ("Variant 1","Variant 2"))
	mode_submit = st.button("Apply transform")
		
	if (mode_submit):
		if (('image' not in st.session_state) and (file is None)):
			st.session_state.cropped_image = None
			st.write("No photo found!")
		else:
			face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_alt2.xml')
			img_cv2 = np.array(Image.open(file))[:,:,::-1].copy()

			if (img_cv2 is None):
				st.session_state.cropped_image = None
				st.write("Problem reading a file!")
			else:
				
				size_h = img_cv2.shape[0]
				size_w = img_cv2.shape[1]
				
				gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
				faces = face_cascade.detectMultiScale(gray, 1.05, 5)

				if (len(faces) == 0):
					st.write("No faces detected!")
				else:
					if (len(faces) == 1):
						x,y,w,h = faces[0]
					elif (len(faces) != 0):
						areas = faces[:,3] * faces[:,2]
						idx = np.argmax(areas)
						x,y,w,h = faces[idx]
					width = round(w * scale)
					height = round(h * scale)
					
					if (mode == "Variant 1"):
						st.write("first", scale)
						result = img_cv2[y: y + height, x : x + width]
					else:
						st.write("second", scale)
						result = img_cv2[max(0, y - round(height / 2)) : y + height + round(height / 2), max(0, x - round(width / 2)) : x + width + round(width / 2)]
					
					result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
					result = Image.fromarray(result)
					
					st.session_state.image = result
					st.session_state.cropped_image = result
				
	if ('cropped_image' not in st.session_state):
		st.session_state['cropped_image'] = None
	elif (st.session_state.cropped_image is not None):
		st.image(st.session_state.cropped_image)

with column3:
	st.header("Generation")
	st.divider()
				
	generate = st.button("Generate")

	if (generate):
		if (('image' not in st.session_state) and (file is None)):
			st.session_state.cropped_image = None
			st.write("No photo found!")
		else:
			
			onnx_model = onnx.load("files/cyclegan.onnx")
			onnx.checker.check_model(onnx_model)
			
			if (st.session_state.image is None):
				input = Image.open(file).resize((128,128))
			else:
				input = st.session_state.image.resize((128,128))

			to_tensor = transforms.ToTensor()
			img = to_tensor(input)
			img.unsqueeze_(0)

			ort_sess = ort.InferenceSession("files/cyclegan.onnx")
			output = ort_sess.run(None, {'input': np.asarray(img)})
			
			output = (np.asarray(output).squeeze())
			for i in range(128):
				for j in range(128):
					for k in range(3):
						output[k][i][j] += 1
						output[k][i][j] *= 127.5
			np.clip(output,0,255,out=output)
			result = np.empty([128,128,3], dtype=np.uint8)
			for i in range(128):
				for j in range(128):
					for k in range(3):
						result[i][j][k] = (output[k][i][j]).astype(np.uint8)
			result = Image.fromarray(result)
			st.session_state.generated_image = result

	if ('generated_image' not in st.session_state):
		st.session_state['generated_image'] = None
	elif (st.session_state.generated_image is not None):
		st.image(st.session_state.generated_image)
